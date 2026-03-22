#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
cd "$REPO_ROOT"

TASK=${TASK:-g1_sprint_track}
NUM_ENVS=${NUM_ENVS:-24}
MAX_ITERS=${MAX_ITERS:-800}
RUN_NAME=${RUN_NAME:-watchloop_800}
PLAY_STEPS=${PLAY_STEPS:-1500}
RECORD_DIR=${RECORD_DIR:-$REPO_ROOT/logs/$TASK/eval_recordings}

python3 legged_gym/scripts/train.py   --task="$TASK"   --num_envs="$NUM_ENVS"   --max_iterations="$MAX_ITERS"   --headless   --run_name="$RUN_NAME"

python3 legged_gym/scripts/play.py   --task="$TASK"   --num_envs=1   --play_steps="$PLAY_STEPS"   --record_play   --record_dir="$RECORD_DIR"
