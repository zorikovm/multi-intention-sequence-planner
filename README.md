# Планировщик последовательности интенций

В исходном FB pi-Switch агент на каждом шаге выбирает одну интенцию. На длинных
маршрутах этого может быть недостаточно: для достижения цели необходимо заранее
учесть последовательность коридоров и поворотов.

В этом проекте поверх замороженных FB-представлений и политик добавлен
high-level планировщик. Он строит граф промежуточных состояний из
офлайн-датасета, находит маршрут до цели и исполняет его как последовательность
латентных интенций. Механизм не обучается и не использует дополнительное
взаимодействие со средой.

Код основан на репозитории
[Switching Successor Measures](https://github.com/stestoKTH/switching-successor-measures).

## Результат

Сравнение проведено на antmaze-medium-navigate-v0: пять задач, 20 эпизодов на
задачу, сиды 0, 1, 2.

| Метод | Общий success | Задача 4 |
|---|---:|---:|
| Single-intention baseline | 0.783 | 0.533 |
| Sequence planner | **0.837** | **0.800** |

Полные результаты находятся в
[results/final_results.csv](results/final_results.csv). Исходные CSV,
параметры запусков и парные исходы эпизодов сохранены в
[experiments/](experiments/). Последовательность гипотез и экспериментов
описана в [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md), подробный отчет находится в
[REPORT.md](REPORT.md).

## Метод

Планировщик выбирает 256 промежуточных состояний из офлайн-датасета и
сопоставляет состоянию w замороженную интенцию B(w). Между близкими состояниями
строится разреженный ориентированный граф. Ребра оцениваются через
консервативную FB-оценку достижимости. Алгоритм Дейкстры находит
последовательность промежуточных интенций.

Для исполнения используются исходные high-level actor и low-level policy. На
коротких маршрутах управление остается у single-intention baseline.

Основная реализация находится в
[utils/multiswitch_planner.py](utils/multiswitch_planner.py). Более подробное
описание алгоритма приведено в [MULTISWITCH.md](MULTISWITCH.md).

## Установка

Эксперименты проводились с Python 3.11.15, JAX 0.4.38 на CPU и OGBench 1.1.4.

~~~bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements-eval-cpu.txt
~~~

Чекпоинт и датасеты должны находиться в следующих путях:

~~~text
artifacts/
  checkpoints/medium/
    flags.json
    params.pkl
  data/
    antmaze-medium-navigate-v0.npz
    antmaze-medium-navigate-v0-val.npz
~~~

Medium checkpoint можно скачать из
[папки с чекпоинтами](https://drive.google.com/drive/folders/1dKYhaDJH9lUREo-kUV3AwmTLrxvKO7Ek).
Если файл называется params_1000000.pkl, его нужно переименовать в params.pkl.

Датасеты:

~~~bash
mkdir -p artifacts/data artifacts/checkpoints/medium
wget -P artifacts/data \
  https://rail.eecs.berkeley.edu/datasets/ogbench/antmaze-medium-navigate-v0.npz
wget -P artifacts/data \
  https://rail.eecs.berkeley.edu/datasets/ogbench/antmaze-medium-navigate-v0-val.npz
~~~

Проверка входных файлов:

~~~bash
sha256sum -c results/checksums.txt
~~~

## Запуск

Короткая проверка:

~~~bash
bash scripts/evaluate_cpu.sh baseline 0 1 smoke_baseline
bash scripts/evaluate_cpu.sh multiswitch 0 1 smoke_sequence
~~~

Полное сравнение, 300 эпизодов на каждый метод:

~~~bash
for seed in 0 1 2; do
  bash scripts/evaluate_cpu.sh baseline "$seed" 20 final_baseline
  bash scripts/evaluate_cpu.sh multiswitch "$seed" 20 final_sequence
done
~~~

Парный прогон задачи 4:

~~~bash
for seed in 0 1 2; do
  bash scripts/evaluate_cpu.sh baseline "$seed" 20 confirm_task4_baseline \
    --eval_tasks=4
  bash scripts/evaluate_cpu.sh multiswitch "$seed" 20 confirm_task4_sequence \
    --eval_tasks=4
done
~~~

Подсчет итоговых метрик:

~~~bash
.venv/bin/python scripts/summarize_results.py
~~~

Проверка кода:

~~~bash
.venv/bin/python -m pytest -q
.venv/bin/python -m py_compile \
  main.py agents/fbpiswitch.py utils/evaluation.py utils/multiswitch_planner.py
~~~

## Основные файлы

- utils/multiswitch_planner.py — построение графа и исполнение маршрута;
- utils/evaluation.py — детерминированное парное сравнение;
- scripts/evaluate_cpu.sh — запуск baseline и planner;
- scripts/summarize_results.py — расчет итоговых метрик;
- results/ — финальная конфигурация и агрегированные результаты;
- experiments/ — исходные результаты всех экспериментов.

Проект распространяется под лицензией MIT.
