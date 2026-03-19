# Step 2.2 速度课程学习规范

## 目标
- 在 Step2.1 稳定基线基础上，逐步提高目标前向速度。
- 保持一 env 一跑道与 lane 约束不变，优先保证稳定性，再提升速度。

## 实现点

1) 命令课程学习开启（g1_sprint_track）
- 文件：`legged_gym/envs/g1/g1_config.py`
- 配置：
  - `commands.curriculum = True`
  - `commands.max_curriculum = 1.2`
  - `commands.curriculum_step = 0.1`
  - `commands.curriculum_threshold = 0.75`
  - `commands.forward_curriculum_only = True`
  - 初始速度范围：`lin_vel_x=[0.2, 0.4]`

2) 课程更新逻辑支持前向专用
- 文件：`legged_gym/envs/base/legged_robot.py`
- 函数：`update_command_curriculum()`
- 逻辑：
  - 当 tracking_lin_vel 达到阈值时扩展命令范围
  - `forward_curriculum_only=True` 时仅提升 `lin_vel_x` 上限，不扩展负向速度

3) 基础配置新增课程参数
- 文件：`legged_gym/envs/base/legged_robot_config.py`
- 新增字段：
  - `curriculum_step`
  - `curriculum_threshold`
  - `forward_curriculum_only`

4) 奖励微调
- `tracking_lin_vel` 调整为 `2.0`，增强提速驱动。

## 验证方法
1. 启动训练：
```bash
cd /home/niumu/unitree_ws/src/unitree_rl_gym
python3 legged_gym/scripts/train.py --task=g1_sprint_track --num_envs=1024 --headless
```

2. 观察 W&B：
- `episode/max_command_x` 是否随训练逐步上升
- `Episode/metric_speed_mps` 是否同步上升
- `Episode/metric_fall_rate` 不出现持续恶化

## 通过标准（Step2.2）
- `max_command_x` 由初始值逐步提升至更高区间（目标接近 1.0~1.2）。
- 速度提升同时，跌倒率保持可控（不出现长期崩塌）。
