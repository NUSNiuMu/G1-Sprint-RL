#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
cd "$REPO_ROOT"

PYTHON_BIN=${PYTHON_BIN:-/home/niumu/anaconda3/envs/unitree-rl/bin/python}
TASK=${TASK:-g1_sprint_track}
NUM_ENVS=${NUM_ENVS:-24}
MAX_ITERS=${MAX_ITERS:-800}
RUN_NAME=${RUN_NAME:-watchloop_800}
PLAY_STEPS=${PLAY_STEPS:-1500}
RECORD_DIR=${RECORD_DIR:-$REPO_ROOT/logs/$TASK/eval_recordings}

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python interpreter not found: $PYTHON_BIN" >&2
  exit 1
fi

echo "[watch-loop] repo=$REPO_ROOT"
echo "[watch-loop] python=$PYTHON_BIN"
echo "[watch-loop] task=$TASK num_envs=$NUM_ENVS max_iters=$MAX_ITERS run_name=$RUN_NAME"
echo "[watch-loop] record_dir=$RECORD_DIR play_steps=$PLAY_STEPS"

"$PYTHON_BIN" legged_gym/scripts/train.py \
  --task="$TASK" \
  --num_envs="$NUM_ENVS" \
  --max_iterations="$MAX_ITERS" \
  --headless \
  --run_name="$RUN_NAME"

"$PYTHON_BIN" legged_gym/scripts/play.py \
  --task="$TASK" \
  --num_envs=1 \
  --play_steps="$PLAY_STEPS" \
  --record_play \
  --record_dir="$RECORD_DIR"
