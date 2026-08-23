# Sequence-of-intentions planner

This repository adds a training-free outer planner to the released FB
pi-Switch agent. It reasons over a sequence of offline landmark intentions and
uses the released single-intention high actor as both a local executor and a
safe fallback.

## Mechanism

1. Select 256 actual offline observations with farthest-point sampling in XY.
2. Attach the frozen latent intention `B(w_j)` to every landmark `w_j`.
3. For each landmark, consider only its 12 spatial nearest landmarks. This
   locality constraint is why the graph is sparse rather than complete.
4. Score a directed local edge with the conservative FB ratio

   ```text
   r(i, j) = exp(mean_e log q_e(i, j) - beta * std_e log q_e(i, j))
   q_e(i, j) = clip(M_e(w_i, B(w_j), w_j) / M_e(w_j, B(w_j), w_j), eps, 1)
   ```

   `r(i, j)` is a discounted reachability proxy, not a calibrated transition
   probability.
5. Run bounded Dijkstra with edge cost `-log r(i, j) + switch_cost`. This
   approximately maximizes the product of local reachability scores while
   penalizing unnecessarily long intention sequences.
6. Use the sequence only when its XY length exceeds the straight-line distance
   by at least 22 units. Otherwise call the released baseline unchanged.
7. Execute every third planned landmark. Each local `B(w_j)` is passed to the
   released high actor, which produces the low-level intention for the frozen
   actor. Switch at 1.75 XY distance; replan after 120 steps or 40 stalled
   steps.

The planner does not generate future observations or trajectories and has no
learned parameters.

## Installation

The reference CPU dependencies are pinned in `requirements-eval-cpu.txt`.
The evaluated setup used Python 3.11.15, JAX 0.4.38 on CPU, and OGBench 1.1.4.

Place the released medium checkpoint and OGBench files as follows:

```text
artifacts/
  checkpoints/medium/
    flags.json
    params.pkl
  data/
    antmaze-medium-navigate-v0.npz
    antmaze-medium-navigate-v0-val.npz
```

The checkpoint SHA-256 used in the report is:

```text
c7efb93cf2caba0d311b87a1c73313b5fe6acda93e7122f47eea5db665858bfe  params.pkl
```

## Reproduce the final comparison on CPU

The helper fixes CPU/headless settings and guarantees `--eval_only`.

```bash
for seed in 0 1 2; do
  bash scripts/evaluate_cpu.sh baseline "$seed" 20 final_baseline
  bash scripts/evaluate_cpu.sh multiswitch "$seed" 20 final_sequence
done
```

`--multiswitch` now uses the locked final configuration by default. Its
explicit equivalent is:

```bash
bash scripts/evaluate_cpu.sh multiswitch 0 20 final_sequence \
  --multiswitch_planner_seed=0 \
  --multiswitch_landmarks=256 \
  --multiswitch_candidates=20000 \
  --multiswitch_neighbors=12 \
  --multiswitch_max_waypoints=32 \
  --multiswitch_waypoint_tolerance=1.75 \
  --multiswitch_max_subgoal_steps=120 \
  --multiswitch_stall_steps=40 \
  --nomultiswitch_replan_on_waypoint \
  --multiswitch_use_high_actor_for_waypoints \
  --multiswitch_min_route_excess=22 \
  --multiswitch_route_stride=3
```

Every run writes `eval.csv`, `episode_outcomes.jsonl`, and `flags.json` under
`experiments/GROUP/sdSEED_TIMESTAMP/`. Summarize the locked runs with:

```bash
.venv/bin/python scripts/summarize_results.py
```

## Tests

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m py_compile \
  main.py agents/fbpiswitch.py utils/evaluation.py utils/multiswitch_planner.py
```

## Files changed by the extension

- `utils/multiswitch_planner.py`: graph construction, FB edge scoring,
  bounded Dijkstra, complexity gate, and route executor.
- `agents/fbpiswitch.py`: explicit frozen low-level action API.
- `utils/evaluation.py`: deterministic paired episode seeds and planner loop.
- `main.py`: eval-only path, planner flags, diagnostics, and raw outcomes.
- `utils/env_utils.py`: state-only renderer bypass for headless CPU evaluation.
- `utils/flax_utils.py`: support for the released `params.pkl` filename.
- `scripts/evaluate_cpu.sh`: reproducible baseline/method entry point.
- `tests/test_multiswitch_planner.py`: unit tests for reachability, landmarks,
  and multi-hop execution.
