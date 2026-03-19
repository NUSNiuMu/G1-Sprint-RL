# Step 1.1 多跑道参数化生成说明

## 本步目标
在现有 G1 训练框架中新增“标准多人跑道”参数化生成能力，支持按配置快速调整道数、道宽、道长，并可进行可视化检查。

## 已实现能力

1. 参数化跑道配置（统一入口）
- 文件：`legged_gym/envs/base/legged_robot_config.py`
- 新增：`terrain.track` 配置块
- 参数包括：
  - `enabled`
  - `visualize_in_viewer`
  - `num_lanes`
  - `lane_width`
  - `lane_length`
  - `boundary_width`
  - `separator_width`
  - `curb_height`
  - `lane_mark_height`

2. 训练环境内跑道布局生成
- 文件：`legged_gym/envs/base/legged_robot.py`
- 新增 `_build_track_layout()`，按参数生成：
  - `lane_centers`
  - `lane_boundaries`
  - 左右边界位置
- 该布局会在环境初始化阶段创建，供后续“越道判定、奖励、障碍生成”复用。

3. Viewer 调试可视化
- 文件：`legged_gym/envs/base/legged_robot.py`
- 新增 `_draw_track_debug_lines()`：
  - 白线：外边界
  - 黄线：分道线
- 在非 headless 且 `visualize_in_viewer=True` 时显示。

4. 新任务注册
- 文件：`legged_gym/envs/g1/g1_config.py`
  - `G1SprintTrackCfg`
  - `G1SprintTrackCfgPPO`
- 文件：`legged_gym/envs/__init__.py`
  - 注册任务名：`g1_sprint_track`

5. 离线俯视图检查脚本
- 文件：`humanoid_sprint/scripts/plot_track_layout.py`
- 输出：`humanoid_sprint/reports/track_layout_preview.png`

## 当前默认跑道参数（g1_sprint_track）
- 道数：6
- 道宽：1.25 m
- 道长：18.0 m

## 验证命令

1) 最小训练烟测（CPU）
```bash
TORCH_EXTENSIONS_DIR=/tmp/torch_extensions \
conda run -n unitree-rl python legged_gym/scripts/train.py \
  --task=g1_sprint_track --headless --num_envs=8 --max_iterations=1 \
  --sim_device=cpu --rl_device=cpu
```

2) 俯视图导出
```bash
conda run -n unitree-rl python humanoid_sprint/scripts/plot_track_layout.py \
  --num_lanes 6 --lane_width 1.25 --lane_length 18.0
```

## 说明
- 本步重点是“跑道参数化生成能力”和“可视化检查能力”。
- 语义标签（可训练感知）、越道判定与障碍逻辑将在后续步骤接入。

