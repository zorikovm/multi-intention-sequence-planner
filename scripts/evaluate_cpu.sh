#!/usr/bin/env bash
set -euo pipefail

method="${1:?usage: evaluate_cpu.sh METHOD SEED EPISODES GROUP [extra flags ...]}"
seed="${2:?missing seed}"
episodes="${3:?missing number of episodes per task}"
group="${4:?missing experiment group}"
shift 4

extra_flags=()
if [[ "$method" == "multiswitch" ]]; then
  extra_flags+=(--multiswitch)
elif [[ "$method" != "baseline" ]]; then
  echo "unknown method: $method" >&2
  exit 2
fi

export MUJOCO_GL="${MUJOCO_GL:-glfw}"
export JAX_PLATFORM_NAME="${JAX_PLATFORM_NAME:-cpu}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

exec .venv/bin/python main.py \
  --env_name=ogbench-antmaze-medium-navigate-v0 \
  --dataset_dir=artifacts/data \
  --headless_no_renderer \
  --agent=agents/fbpiswitch.py \
  --restore_path=artifacts/checkpoints/medium \
  --eval_only \
  --eval_episodes="$episodes" \
  --video_episodes=0 \
  --enable_wandb=0 \
  --seed="$seed" \
  --save_dir=experiments \
  --wandb_run_group="$group" \
  "${extra_flags[@]}" \
  "$@"
