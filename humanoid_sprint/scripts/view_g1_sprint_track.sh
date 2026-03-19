#!/usr/bin/env bash
set -euo pipefail

cd /home/niumu/unitree_ws/src/unitree_rl_gym

# 可视化查看跑道调试线（需要图形界面）
TORCH_EXTENSIONS_DIR=/tmp/torch_extensions \
conda run -n unitree-rl python legged_gym/scripts/play.py \
  --task=g1_sprint_track \
  --num_envs=8 \
  --sim_device=cuda:0 \
  --rl_device=cuda:0
