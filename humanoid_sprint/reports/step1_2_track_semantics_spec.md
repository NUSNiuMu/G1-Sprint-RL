# Step 1.2 跑道语义标注说明

## 本步目标
为参数化跑道建立可训练可评估的语义层，明确区分：
- 非跑道区域
- 跑道内部
- 跑道边界/分道线

## 语义定义

语义 ID（运行时）：
- `0`: non_track（非跑道）
- `1`: lane（跑道内部）
- `2`: boundary（边界/分道线）

## 已实现内容

1) 语义配置参数
- 文件：`legged_gym/envs/base/legged_robot_config.py`
- 新增字段：
  - `terrain.track.semantic_enabled`
  - `terrain.track.semantic_boundary_tol`

2) 跑道规模联动参数（当前基线为“一 env 一跑道”）
- 文件：`legged_gym/envs/base/legged_robot_config.py`
- 新增字段：
  - `terrain.track.auto_match_num_envs`
  - `terrain.track.auto_scale_length_with_grid`
  - `terrain.track.env_grid_rows`
  - `terrain.track.base_grid_cols`

联动规则：
- 道数：
  - 当前基线：`num_lanes = 1`（`auto_match_num_envs=False`）
  - 可选模式：`num_lanes = num_envs`（当 `auto_match_num_envs=True`）
- 跑道长度：
  - 先取 `lane_length` 基线
  - 按 `env_grid_cols / base_grid_cols` 放大（当 `auto_scale_length_with_grid=True`）

示例（当前默认）：
- `num_envs=24`
- `env_grid_rows=2` -> `env_grid_cols=12`
- `base_grid_cols=6`
- `lane_length=6` -> 实际 `track_lane_length=12`
- `track_num_lanes=1`（每个 env 内单道）

3) 运行时语义判定
- 文件：`legged_gym/envs/base/legged_robot.py`
- 新增：`_update_track_semantics()`
- 基于机器人在本环境中的局部 `(x,y)` 计算语义 ID，并写入 `track_semantic_id_buf`。

4) 训练日志输出新增语义指标
- `semantic_lane_ratio`
- `semantic_boundary_ratio`
- `semantic_non_track_ratio`
- 同时输出：
  - `track_num_lanes`
  - `track_lane_length`

5) 语义可视化脚本
- 文件：`humanoid_sprint/scripts/plot_track_semantics.py`
- 输出：`humanoid_sprint/reports/track_semantic_preview.png`

## 说明
- 本步是“几何语义层”与“可观测语义指标”打通。
- 真正的相机语义（RGB-D端到端识别）将在后续感知步骤接入。
