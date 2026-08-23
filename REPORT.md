# Offline FB Sequence Planner для длинного горизонта

## Короткий итог

Реализован training-free high-level механизм, который сначала строит
последовательность landmark-интенций, а затем исполняет ее через замороженный
single-intention high actor. Он не обучает world model, не генерирует будущие
траектории и не использует online interaction.

На `ogbench-antmaze-medium-navigate-v0`, 5 задачах, 20 эпизодах на задачу и
seed 0, 1, 2:

| Метод | Seed 0 | Seed 1 | Seed 2 | Mean |
| --- | ---: | ---: | ---: | ---: |
| Single-intention baseline | .75 | .85 | .75 | .783 |
| Offline FB Sequence Planner | .84 | .83 | .84 | **.837** |

Абсолютная разница `+0.053`, или 251/300 против 235/300 успешных эпизодов.
Прибавка локализована в длинной задаче 4: `.533 → .800`. На остальных четырех
задачах итоговые числа полностью совпали с baseline.

Ключевой результат исследования не просто в Dijkstra. Полезной оказалась
комбинация трех решений:

1. FB используется как локальная мера достижимости на разреженном графе;
2. исходный high actor исполняет каждый локальный waypoint;
3. sequence planning включается только когда граф обнаруживает существенный
   геометрический обход, иначе сохраняется сильный baseline.

## 1. Исследовательская гипотеза

Single-intention high actor решает локальную задачу

```text
z_t = pi_high(s_t, g),     a_t = pi_low(s_t, z_t).
```

Для лабиринта это может быть близоруко: полезность ближайшей интенции не
гарантирует, что она ведет в правильный коридор. Проверяемая гипотеза:

> Замороженные FB-представления содержат достаточно информации, чтобы
> сравнивать локальные переходы между offline landmark states. Композиция таких
> переходов даст полезный маршрут на длинном горизонте, даже без модели сырых
> будущих состояний.

Самый дешевый фальсифицирующий тест: сохранить тот же checkpoint, dataset,
задачи и episode seeds; заменить только test-time high-level decision и
сравнить success.

## 2. Предложенный механизм

### 2.1 Landmark-интенции

Из 20 000 наблюдений offline dataset выбираются 256 реальных состояний
`w_1, ..., w_N` методом farthest-point sampling по XY. Для каждого состояния
вычисляется замороженная интенция

```text
z_j = B(w_j).
```

Это не новые сгенерированные subgoals: каждая вершина является реально
наблюдавшимся offline состоянием.

### 2.2 Почему граф не полный

Ребро не определяется наличием одного dataset action, которое точно переводит
`w_i` в `w_j`: в непрерывной среде такое совпадение почти бессмысленно. Также
не оцениваются все `N^2` пары. Для каждой вершины рассматриваются только 12
ближайших XY landmarks. Поэтому возможны максимум `256 * 12 = 3072`
направленных кандидатных ребра вместо 65 280 пар без self-loops.

Локальность выполняет две функции:

- убирает ложные дальние shortcut edges, которые часто переоценивает
  аппроксиматор successor measure;
- превращает задачу в композицию локальных навыков, для которых frozen policy
  лучше откалибрована.

### 2.3 FB-reachability edge score

Для кандидатного перехода `w_i → w_j` политика должна следовать интенции
`z_j = B(w_j)`. Для каждой головы forward ensemble вычисляется отношение

```text
q_e(i,j) = clip(
    M_e(w_i, z_j, w_j) / M_e(w_j, z_j, w_j),
    eps, 1
).
```

Числитель оценивает discounted occupancy цели при старте из `w_i`, знаменатель
нормирует self-occupancy цели. В коде оба значения получаются из замороженной
FB-факторизации через `F(s,z)^T B(w_j)`.

Консервативный score ансамбля:

```text
r(i,j) = exp(mean_e(log q_e) - beta * std_e(log q_e)),   beta = 0.5.
```

Важно: `r(i,j)` — proxy discounted reachability, а не доказанно
калиброванная вероятность физического перехода. Именно поэтому граф ограничен
локальными кандидатами и имеет online replanning.

### 2.4 Планирование последовательности

Текущее состояние и goal соединяются только с 12 ближайшими landmarks.
Для каждого ребра используется стоимость

```text
c(i,j) = -log r(i,j) + lambda_switch,   lambda_switch = 0.02.
```

Bounded Dijkstra ищет до 32 landmarks:

```text
argmin_route sum c(i,j)
  = argmax_route product r(i,j) * exp(-lambda_switch * number_of_switches).
```

То есть интуиция пользователя про максимальную общую достижимость маршрута
верна, но с двумя поправками: перемножаются не истинные вероятности, а
консервативные FB scores; поиск идет только по sparse local graph.

### 2.5 Когда использовать последовательность

Ранний вариант, который всегда исполнял графовый путь, сильно ухудшал короткие
задачи. Поэтому planner измеряет

```text
route_excess = graph_route_XY_length - straight_line_XY_distance.
```

Если `route_excess < 22`, весь эпизод исполняет исходный single-intention
baseline. Gate не использует task id или reward: решение принимается из
геометрии построенного маршрута.

### 2.6 Исполнение и replanning

Полный графовый маршрут сжимается, оставляя каждый третий landmark. Для каждого
waypoint latent `B(w_j)` передается исходному high actor как локальная задача;
тот выбирает интенцию для low-level actor. Это nested hierarchy:

```text
sequence planner:      s, g -> [B(w_1), B(w_2), ..., B(w_K)]
released high actor:   s, B(w_k) -> z_low
released low actor:    s, z_low -> a
```

Переход к следующему waypoint происходит на расстоянии 1.75. При 120 шагах на
subgoal или 40 шагах без прогресса маршрут перестраивается из текущего
состояния. Если путь не найден, управление безопасно возвращается baseline.

## 3. Что именно обучалось

Ничего. Во всех экспериментах заморожены:

- `modules_forward_repr`;
- `modules_backward_repr`;
- `modules_actor`;
- `modules_high_actor`.

Новый механизм состоит из offline graph, test-time search и нескольких
скалярных hyperparameters. Поэтому нового neural checkpoint нет. Для
воспроизведения нужны исходный `params.pkl` и `results/final_config.json`.

## 4. Экспериментальный протокол

- Environment: `ogbench-antmaze-medium-navigate-v0`.
- Metric: mean binary success по всем пяти фиксированным OGBench tasks.
- Final budget: 20 episodes/task, 5 tasks, 3 seeds = 300 episodes/method.
- Одинаковый checkpoint, dataset, temperature 0 и лимит эпизода.
- Парный reset для baseline и method:

  ```text
  episode_seed = eval_seed * 1_000_000 + task_id * 10_000 + episode_index.
  ```

- Seed задается NumPy, Gymnasium action space и `env.reset`.
- Planner graph seed фиксирован в 0 и не меняется с evaluation seed.
- Screening runs отделены от финальных runs в журнале и CSV.

До исправления evaluator использовал неполное seeding; полученные тогда числа
явно отброшены. Два повторных deterministic checks дали идентичные CSV.

## 5. Финальные результаты

### 5.1 По seed

| Метод | Seed 0 | Seed 1 | Seed 2 | Mean ± sample SD |
| --- | ---: | ---: | ---: | ---: |
| Baseline | .75 | .85 | .75 | .783 ± .058 |
| Sequence planner | .84 | .83 | .84 | **.837 ± .006** |
| Delta | +.09 | -.02 | +.09 | **+.053** |

### 5.2 По задачам

| Метод | T1 | T2 | T3 | T4 | T5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline | .900 | .867 | .833 | .533 | .783 |
| Sequence planner | .900 | .867 | .833 | **.800** | .783 |

На T4 planner был активирован в 85%, 85%, 90% эпизодов для seed 0, 1, 2.
На остальных задачах activation rate равен нулю. Результат подтверждает более
узкую гипотезу: последовательность помогает на обнаруженном длинном обходе;
на коротком горизонте лучше не вмешиваться.

## 6. Насколько результату можно доверять

Повторный targeted run T4 с сохранением individual outcomes точно воспроизвел
финальные значения:

- baseline 32/60 = .533;
- sequence planner 48/60 = .800;
- 24 пары `failure → success`;
- 8 пар `success → failure`;
- exact two-sided McNemar `p = 0.0070`.

Wilson 95% intervals для T4: baseline `[.409, .654]`, planner `[.682, .882]`.
Это свидетельствует, что эффект в парных эпизодах T4 трудно объяснить одной
случайной флуктуацией reset states.

Но есть важная оговорка. Парные overall deltas по трем seed равны
`[+.09, -.02, +.09]`; Student-t 95% interval для их среднего широк:
`[-.104, .211]`. Три seed недостаточны для сильного утверждения о переносе на
другие maze layouts. Честная формулировка: механизм победил baseline в данном
зафиксированном medium benchmark и имеет значимый парный эффект на T4, но еще
не доказал универсальность.

## 7. Что не сработало и почему

Полная последовательность через direct low actor дала `.52` против `.68`
baseline в screening. Латент `B(w)` хорошо задает направленность successor
measure, но оказался плохо откалиброван как непосредственная locomotion
команда.

Nested high actor на всех задачах дал `.48`, хотя T4 выросла `.20 → .60`.
Планирование имеет цену: waypoint error, лишние переключения и расход
горизонта. Нужен conditional planner, а не безусловная замена baseline.

Gate по числу узлов маршрута был нестабилен относительно FPS seed. Число
landmarks зависит от дискретизации, а геометрический excess лучше соответствует
реальному обходу.

Порог excess 20 оказался слишком мягким: один false positive на T5 ухудшил
результат. Порог 22 сохранил baseline на всех коротких задачах.

Слишком patient executor (`max=220`, `stall=80`) был хуже fast replanning:
ошибочный waypoint съедал значительную часть конечного горизонта.

## 8. Ограничения и возможный leakage

1. Hyperparameters выбирались по тем же task definitions и seed 0–2, а не по
   отдельному validation maze. Финальный 20-episode budget больше screening,
   но это не полноценный held-out test.
2. XY используется для kNN, progress и route-excess. Это допустимо для
   AntMaze state observations, но не является универсальным representation-only
   решением.
3. FB ratio не откалиброван как probability; `-log` имеет полезную
   оптимизационную интерпретацию, но не строгую вероятностную гарантию.
4. Planner улучшает один тип длинного маршрута. Проверка large, giant и
   teleport не проводилась из-за CPU/time budget.
5. У planner нет явной модели correlated edge failures; произведение scores
   предполагает аддитивную стоимость пути.
6. Gate может пропустить сложную задачу с небольшим XY excess или включиться на
   длинной, но простой траектории в другой среде.

## 9. Приоритетные гипотезы для следующей итерации

### P0 — необходимы для проверки обобщения

1. **Held-out calibration.** Настроить gate и executor на отдельном наборе
   start-goal pairs или validation split, один раз заморозить параметры, затем
   проверить на новых seed и maze sizes.
2. **Large/giant/teleport evaluation.** Главная гипотеза должна усиливаться с
   горизонтом. Если gain исчезает, текущий успех, вероятно, специфичен T4.
3. **Больше seed.** Минимум 10 evaluation seeds и paired bootstrap по эпизодам;
   основной критерий — lower confidence bound delta выше нуля.

### P1 — наиболее вероятные улучшения механизма

4. **Learned edge calibration только из offline данных.** Положительные пары
   брать из реально следующих окон траектории, отрицательные — из несовместимых
   сегментов; изотонически или логистически калибровать FB ratio. Это может
   превратить score в более честную hitting-probability proxy без world model.
5. **Advantage gate вместо XY threshold.** Сравнивать score лучшего multi-hop
   пути со score прямой single intention и учитывать ensemble uncertainty.
   Включать planner только при положительном консервативном преимуществе.
6. **Top-K risk-aware routes.** Искать несколько различных путей и выбирать по
   lower-confidence score или CVaR. При stall переключаться на альтернативный
   маршрут, а не строить почти тот же путь.
7. **Reachability-aware landmarks.** Вместо чистого XY FPS выбирать bottleneck
   и frontier states по mutual FB reachability, degree/centrality и покрытию.
8. **Adaptive route compression.** Пропускать waypoint, если локальный FB score
   следующего узла достаточно высок; сохранять плотные subgoals около
   поворотов и bottlenecks.

### P2 — более исследовательские идеи

9. **Representation-space graph без XY.** Строить kNN по симметризованной
   FB-distance или по `B(s)`, а progress оценивать изменением route value.
10. **Offline termination model.** Обучить легкий classifier достижения option
    на offline windows; сравнить с фиксированной XY tolerance.
11. **Latent sequence MPC.** Оптимизировать короткую последовательность
    интенций CEM/beam search, используя только FB edge values, без генерации
    observations.
12. **Uncertainty-aware replanning.** Учитывать disagreement forward ensemble
    не только в edge cost, но и в моменте переключения и выборе fallback.
13. **Graph connectivity audit.** Удалять односторонние spurious edges через
    mutual reachability или проверку cycle consistency.

## 10. Критерии дальнейшего успеха

Следующий вариант считается подтвержденным, если после однократной настройки
на validation split он:

1. превосходит paired baseline по mean success минимум на двух maze sizes;
2. имеет положительную нижнюю границу paired bootstrap 95% CI;
3. не ухудшает short-horizon subset больше чем на 1 процентный пункт;
4. сохраняет training-free режим или явно учитывает равный offline training
   budget для baseline;
5. проходит ablation `FB score → geometric-only edge score`, чтобы доказать,
   что прирост дает representation, а не только карта XY.

## 11. Воспроизведение и артефакты

Основная команда:

```bash
for seed in 0 1 2; do
  bash scripts/evaluate_cpu.sh baseline "$seed" 20 final_baseline
  bash scripts/evaluate_cpu.sh multiswitch "$seed" 20 final_sequence
done
```

Подробные инструкции находятся в `MULTISWITCH.md`. Результаты:

- `results/final_results.csv` — финальная таблица;
- `results/ablation_results.csv` — последовательность решений;
- `results/statistics.json` — интервалы и парный тест;
- `results/final_config.json` — полностью зафиксированный planner;
- `EXPERIMENT_LOG.md` — хронологический журнал;
- `experiments/` — исходные CSV и flags всех запусков, а также raw outcomes
  подтверждающего T4-прогона.

Checkpoint не менялся: `artifacts/checkpoints/medium/params.pkl`, SHA-256
`c7efb93cf2caba0d311b87a1c73313b5fe6acda93e7122f47eea5db665858bfe`.

## Ссылки

- [Switching Successor Measures: paper](https://arxiv.org/abs/2605.13207)
- [Официальный baseline repository](https://github.com/stestoKTH/switching-successor-measures)
- [Forward-Backward representations](https://arxiv.org/abs/2103.07945)
- [OGBench](https://arxiv.org/abs/2410.20092)
