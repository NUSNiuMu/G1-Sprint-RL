# Step 0.3 指标规范（Humanoid Sprint）

## 目标
建立稳定的指标字段约定，让训练与评估使用同一套 KPI 名称。

## KPI 字段（通过 `infos["episode"]` 输出）

当前已实现：
- `metric_speed_mps`
  - 每个 episode 的平均前向速度：`mean(|base_lin_vel_x|)`。
- `metric_collision_rate`
  - 每个 episode 中发生碰撞步数占比。
- `metric_fall_rate`
  - 以“非超时终止”结束的 episode 占比。
- `metric_success_rate`
  - 以“超时结束（生存完成）”结束的 episode 占比。
- `metric_torque_utilization`
  - 每步关节 `abs(torque)/torque_limit` 的均值，再对 episode 求均值。
- `metric_torque_violation_rate`
  - 任一关节超过 `soft_torque_limit * torque_limit` 的步数占比。
- `metric_lane_violation_rate`
  - Step 0.3 占位字段（已接线，当前固定 0.0）。
- `metric_obstacle_avoid_rate`
  - Step 0.3 占位字段（已接线，当前固定 0.0）。

已有性能日志（runner）：
- `Perf/total_fps`
- `Perf/collection time`
- `Perf/learning_time`

## 阶段 0 验收标准

在语义跑道与障碍接入前，先验证“指标链路可用”：
- 训练过程能够输出所有 `metric_*` 字段。
- 评估脚本（`play.py`）在 episode 汇总时能打印 `metric_*` 字段。
- 接入指标后不引入运行时回归。

## 后续阶段映射

- Step 1.x / 2.x：
  - 重点使用 `metric_speed_mps`、`metric_fall_rate`、`metric_success_rate`、`metric_torque_*`。
- Step 3.x：
  - 接入语义跑道后启用真实 `metric_lane_violation_rate`。
- Step 4.x：
  - 接入障碍后启用真实 `metric_obstacle_avoid_rate` 与障碍碰撞指标。
- Step 5.x：
  - 高速课程下收紧 `metric_torque_violation_rate` 阈值。

