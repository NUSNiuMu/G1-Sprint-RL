# Step 2.1 无视觉基线任务（Oracle Lane）规范（完成版）

## 1. 目标
- 在不接入视觉输入（仅本体状态）的条件下，建立可复现的跑道基线。
- 先完成稳定站立，再进入低速前进，为 Step 2.2 速度课程学习提供可复用 checkpoint。

## 2. 已实现功能

1) Oracle lane 居中约束
- 文件：`legged_gym/envs/base/legged_robot.py`
- 函数：`_reward_lane_centering()`
- 机制：
  - 计算 `local_y = base_pos_y - env_origin_y`
  - 与 `env_lane_center_y` 比较得到横向误差
  - 用高斯函数输出居中奖励

2) g1_sprint_track 的 Step2.1 基线配置
- 文件：`legged_gym/envs/g1/g1_config.py`
- 策略：`ActorCriticRecurrent`（LSTM, hidden_size=64）
- 域随机化：本步关闭（`randomize_friction=False`、`randomize_base_mass=False`、`push_robots=False`）
- 命令与奖励：
  - 站稳阶段：`lin_vel_x=[0.0, 0.0]`
  - 低速前进阶段：`lin_vel_x=[0.1, 0.4]`
  - 当前完成版奖励：
    - `tracking_lin_vel=1.8`
    - `orientation=-0.5`
    - `base_height=-1.5`
    - `action_rate=-0.005`
    - `collision=-0.2`
    - `stand_still=-0.02`
    - `lane_centering=1.0`
    - `termination=-1.0`

## 3. 结果判定（本步）
- 用户确认：Step 2.1 完成。
- 支撑依据：
  - 已出现稳定站立 run（历史 run `jqu52hcs`，`fall_rate=0`）。
  - 当前 run（`qpb9sund`）已进入低速前进阶段，`mean_episode_length` 与 `mean_reward` 持续提升。

## 4. 运行与复现实验

```bash
cd /home/niumu/unitree_ws/src/unitree_rl_gym
python3 legged_gym/scripts/train.py --task=g1_sprint_track --num_envs=24 --headless
```

```bash
cd /home/niumu/unitree_ws/src/unitree_rl_gym
python3 legged_gym/scripts/play.py --task=g1_sprint_track --num_envs=24
```

## 5. 下一步接口（Step 2.2）
- 在本步 checkpoint 基础上继续课程学习：
  1. `lin_vel_x: [0.1,0.4] -> [0.3,0.8]`
  2. 再提升到 `[0.6,1.2]`
- 每一档速度以 `fall_rate` 与 `mean_episode_length` 作为放行门槛。
