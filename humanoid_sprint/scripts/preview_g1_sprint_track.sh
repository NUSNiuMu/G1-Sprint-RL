#!/usr/bin/env bash
set -euo pipefail

cd /home/niumu/unitree_ws/src/unitree_rl_gym

# 低负载可视化预览：显示标准多人跑道（6 道）
TORCH_EXTENSIONS_DIR=/tmp/torch_extensions \
conda run -n unitree-rl python legged_gym/scripts/play.py \
  --task=g1_sprint_track \
  --num_envs=8 \
  --headless \
  --sim_device=cpu \
  --rl_device=cpu
