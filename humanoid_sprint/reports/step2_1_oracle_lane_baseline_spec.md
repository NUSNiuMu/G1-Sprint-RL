# Step 2.1 无视觉基线任务（Oracle Lane）说明

## 本步目标
- 在不接入视觉（仅本体状态）的前提下，先训练可稳定前进的跑道基线策略。
- 使用仿真已知的 lane 中心线信息（oracle）提供辅助约束。

## 已实现内容

1) Oracle lane 奖励
- 文件：`legged_gym/envs/base/legged_robot.py`
- 新增函数：`_reward_lane_centering()`
- 逻辑：
  - 计算机器人在本 env 局部坐标中的横向位置 `local_y`
  - 与分配 lane 中心 `env_lane_center_y` 比较，按高斯形式给奖励
  - track 未启用时返回 0

2) Step 2.1 基线训练配置（`g1_sprint_track`）
- 文件：`legged_gym/envs/g1/g1_config.py`
- 配置策略：
  - 保持 `ActorCriticRecurrent`（与官方风格一致）
  - 命令固定直线前进：
    - `lin_vel_x=[0.6, 1.2]`
    - `lin_vel_y=[0.0, 0.0]`
    - `ang_vel_yaw=[0.0, 0.0]`
    - `heading=[0.0, 0.0]`
  - 关闭本步不需要的随机化：
    - `randomize_friction=False`
    - `randomize_base_mass=False`
    - `push_robots=False`

3) 奖励重心（符合 Step 2.1 要求）
- 前进速度：`tracking_lin_vel=2.0`
- 姿态稳定：`orientation=-1.0`、`base_height=-3.0`
- 步态平滑：`action_rate=-0.02`、`dof_vel=-2e-4`、`dof_acc=-2.5e-7`
- 摔倒惩罚：`termination=-5.0`
- 赛道先验：`lane_centering=1.0`

## 验证命令

1) 最小训练烟测
```bash
cd /home/niumu/unitree_ws/src/unitree_rl_gym
python3 legged_gym/scripts/train.py --task=g1_sprint_track --num_envs=24 --max_iterations=1 --headless
```

2) 训练观察（建议）
```bash
cd /home/niumu/unitree_ws/src/unitree_rl_gym
python3 legged_gym/scripts/train.py --task=g1_sprint_track --num_envs=24
```

## 通过标准（Step 2.1）
- 在固定跑道上可持续前进。
- 不频繁跌倒（`metric_fall_rate` 相比 Step 1 基线明显下降）。
- 轨迹在本 lane 中心附近波动（`lane_centering` 奖励为正且稳定）。
