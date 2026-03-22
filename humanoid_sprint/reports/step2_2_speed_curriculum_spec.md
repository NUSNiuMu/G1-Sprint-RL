# Step 2.2 速度课程学习规范（完成版）

## 1. 目标
- 在 Step 2.1 Oracle Lane 基线之上，推进速度课程学习。
- 在跑道长度翻倍后，建立可复现的“稳定直线走”版本，为后续提速做准备。
- 本步的验收重点不是绝对高速，而是：真实前进、不过界、不摔倒、可继续课程提升。

## 2. 本步最终采用的技术路线

1) 双倍跑道长度 + 更长 episode
- 文件：`legged_gym/envs/g1/g1_config.py`
- 关键配置：
  - `lane_length = 12.0`
  - `episode_length_s = 30.0`
- 目的：
  - 保证双倍跑道上的低速稳定前进有足够时间完成。

2) 稳定 gait 迁移，而不是从零硬训
- 迁移来源：`logs/g1/Feb04_22-36-49_g1_track_semantic_v3/model_3000.pt`
- 迁移方式：`--resume_model_only`
- 说明：
  - 仅加载模型参数，不继承旧优化器状态和旧迭代数。
  - 在新任务上重新开始优化，避免旧任务优化轨迹把策略带偏。

3) Oracle Lane 观测注入 actor
- 文件：`legged_gym/envs/g1/g1_env.py`
- 新增观测：
  - 中线偏差
  - 左右边界余量
  - 航向误差
  - 到终点剩余距离
- 对应配置：
  - `num_observations = 52`
  - `num_privileged_obs = 55`
  - `oracle_lane_obs = True`
- 目的：
  - 让 actor 显式感知跑道，而不是只靠奖励事后修正。

4) 真实前进与越道趋势奖励
- 文件：`legged_gym/envs/base/legged_robot.py`
- 使用的关键奖励/惩罚：
  - `track_progress`
  - `stalling`
  - `overspeed`
  - `lane_offset`
  - `lane_divergence`
  - `heading_alignment`
  - `heading_error`
  - `yaw_rate`
  - `lateral_velocity`
- 目标：
  - 让策略沿跑道真实前进。
  - 抑制“原地摆动”“超速硬冲”“越走越偏”。

5) 更干净的 reset 初始条件
- 文件：`legged_gym/envs/g1/g1_config.py`
- 关键配置：
  - `spawn_x_jitter = 0.15`
  - `spawn_y_margin = 0.45`
- 目的：
  - 减少初始横向偏差对 lane 任务的污染。

## 3. 最终接受版本
- run：`Mar22_17-21-48_autoloop_r12_oracle_lane_transfer800`
- W&B：`iul5phys`
- checkpoint：`model_50.pt`

## 4. 验收指标

### 4.1 训练阶段指标
- `metric_finish_rate ≈ 0.9659`
- `metric_out_of_track_fail_rate = 0.0`
- `metric_fall_rate = 0.0`
- `metric_speed_mps ≈ 0.2301`

### 4.2 固定相机 play 指标
- 评估目录：`/tmp/autoloop_eval_r12_50/g1_sprint_track_20260322_172328`
- 关键结果：
  - `net_x = 3.9323 m`
  - `net_y = 0.1348 m`
  - `reset_count = 0`
  - `cumulative_positive_dx = 3.9487 m`
  - `cumulative_negative_dx = -0.0164 m`
- 说明：
  - 策略并非原地晃动，而是真实向终点方向推进。
  - 横向漂移已经较小，且未触发越道失败。

## 5. 下午尝试与结论

1) 单纯调奖励但不给 lane 观测：失败
- 原因：actor 无法直接知道自己相对跑道的位置，只能靠奖励后验摸索，稳定性不足。

2) 从已退化 sprint 策略继续训：失败
- 原因：旧策略会保留“快冲、偏航、越道”的局部最优。

3) 从零开始训 sprint：失败
- 原因：虽然越道可能暂时变少，但 gait 本身不稳定，fall rate 过高。

4) 最终有效方案：成功
- 稳定 G1 gait 迁移
- model-only transfer
- Oracle Lane 观测
- 固定相机 + 净位移验收

## 6. 复现实验

### 6.1 训练
```bash
cd /home/niumu/unitree_ws/src/unitree_rl_gym
python3 legged_gym/scripts/train.py \
  --task=g1_sprint_track \
  --num_envs=4096 \
  --max_iterations=800 \
  --headless \
  --resume \
  --resume_model_only \
  --experiment_name=g1 \
  --load_run=Feb04_22-36-49_g1_track_semantic_v3 \
  --checkpoint=3000 \
  --run_name=autoloop_r12_oracle_lane_transfer800
```

### 6.2 回放
```bash
cd /home/niumu/unitree_ws/src/unitree_rl_gym
python3 legged_gym/scripts/play.py \
  --task=g1_sprint_track \
  --num_envs=1 \
  --experiment_name=g1 \
  --load_run=Mar22_17-21-48_autoloop_r12_oracle_lane_transfer800 \
  --checkpoint=50 \
  --play_steps=1500 \
  --record_play \
  --record_camera_mode=fixed \
  --record_dir=/tmp/check_r12_best
```

## 7. 下一步接口
- 以 `model_50.pt` 作为 Step 2.3/后续提速课程的起点。
- 后续只提升前向目标速度，不再移除 Oracle Lane 观测。
- 若进入更高速度阶段，再逐步平衡：
  - `track_progress`
  - `overspeed`
  - `heading_alignment`
  - `lane_offset`
