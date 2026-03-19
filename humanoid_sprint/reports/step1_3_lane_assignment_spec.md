# Step 1.3 多智能体初始站位与赛道分配说明

## 本步目标
- 每个机器人分配到明确 lane。
- reset 与初始生成时都落在对应 lane 中线附近。
- 避免初始跨道和大扰动，提升起步稳定性。

## 核心机制

1) env 到 lane 的映射
- 在 `_build_track_layout()` 内建立：
  - `env_lane_ids`
  - `env_lane_center_y`
- 映射规则：`lane_id = env_id % num_lanes`

2) 初始生成（create_envs）
- 当 track 启用时：
  - `x` 方向：小范围抖动（`spawn_x_jitter`）
  - `y` 方向：围绕分配的 lane center，按 `spawn_y_margin` 控制安全余量

3) reset 逻辑
- `_reset_root_states()` 在 track 模式下调用 `_reset_track_positions()`：
  - 继续保证机器人回到对应 lane 中线附近
  - 不再使用全局 `[-1, 1]` 的横向随机扰动

4) 跑道规模联动（延续 Step 1.2）
- `auto_match_num_envs=True` 时：`num_lanes=num_envs`
- `auto_scale_length_with_grid=True` 时：长度按网格列数比例放大

## 新增配置参数
- `terrain.track.spawn_x_jitter`
- `terrain.track.spawn_y_margin`

## 新增日志检查字段
- `track_lane_assignment_coverage`
  - 覆盖率 = 实际被分配的不同 lane 数 / 总 lane 数
  - `1.0` 表示每个 lane 都至少有一个机器人分配（在 `num_lanes=num_envs` 时即一一对应）

## 验证目标（24 机器人案例）
- `track_num_lanes = 24`
- `track_lane_length = 12`
- `track_lane_assignment_coverage = 1.0`

