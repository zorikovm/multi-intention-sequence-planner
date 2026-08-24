#!/usr/bin/env bash
set -euo pipefail

method="${1:?usage: evaluate_large_cpu.sh METHOD SEED EPISODES GROUP [extra flags ...]}"
seed="${2:?missing seed}"
episodes="${3:?missing number of episodes per task}"
group="${4:?missing experiment group}"
shift 4

export MUJOCO_GL="${MUJOCO_GL:-glfw}"
export JAX_PLATFORM_NAME="${JAX_PLATFORM_NAME:-cpu}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

common_args=(
  --env_name=ogbench-antmaze-large-navigate-v0
  --dataset_dir=artifacts/data
  --headless_no_renderer
  --agent=agents/fbpiswitch.py
  --restore_path=artifacts/checkpoints/large
  --eval_only
  --eval_tasks=2,4,5
  --eval_episodes="$episodes"
  --video_episodes=0
  --enable_wandb=0
  --seed="$seed"
  --save_dir=experiments
  --wandb_run_group="$group"
)

if [[ "$method" == "baseline" ]]; then
  exec .venv/bin/python main.py "${common_args[@]}" "$@"
elif [[ "$method" == "multiswitch" || "$method" == "safe_hybrid" ]]; then
  exec .venv/bin/python main.py "${common_args[@]}" \
    --multiswitch \
    --multiswitch_terminal_tolerance=0.5 \
    --multiswitch_min_route_waypoints=20 \
    --multiswitch_max_route_detour=4.0 \
    --multiswitch_max_replans_before_fallback=0 \
    "$@"
else
  echo "unknown method: $method" >&2
  exit 2
fi
