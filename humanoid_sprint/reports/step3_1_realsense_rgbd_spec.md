# Step 3.1 RealSense RGB-D 观测接入说明

## 目标
在 `g1_sprint_track` 跑道任务基础上接入 RealSense 风格 RGB-D 传感器，并提供稳定的数据流与调试脚本。

## 新任务
- `g1_sprint_track_rgbd`

## 设计原则
- 本步骤只接入相机与元数据，不把图像直接拼进 policy observation。
- 当前稳定的无视觉基线 `g1_sprint_track` 保持不变。
- 视觉融合留到 Step 3.2 完成。

## 关键文件
- `src/unitree_rl_gym/legged_gym/envs/base/legged_robot_config.py`
- `src/unitree_rl_gym/legged_gym/envs/base/base_task.py`
- `src/unitree_rl_gym/legged_gym/envs/base/legged_robot.py`
- `src/unitree_rl_gym/legged_gym/envs/g1/g1_env.py`
- `src/unitree_rl_gym/legged_gym/envs/g1/g1_config.py`
- `src/unitree_rl_gym/legged_gym/envs/__init__.py`
- `src/unitree_rl_gym/humanoid_sprint/scripts/inspect_g1_rgbd.py`

## 默认相机参数
- `attach_to_body = head_link`
- `width = 84`
- `height = 48`
- `horizontal_fov_deg = 87.0`
- `near_plane = 0.1`
- `far_plane = 12.0`
- `local_pos = [0.12, 0.0, 0.02]`
- `local_rot_euler_deg = [0.0, 0.0, 0.0]`

## 验证命令
```bash
cd /home/niumu/unitree_ws/src/unitree_rl_gym

source /home/niumu/anaconda3/etc/profile.d/conda.sh && \
conda run -n unitree-rl python humanoid_sprint/scripts/inspect_g1_rgbd.py \
  --task g1_sprint_track_rgbd \
  --num_envs 2 \
  --num_steps 20 \
  --sim_device cuda:0 \
  --output_dir /tmp/g1_rgbd_inspect_smoke
```

## 验证通过标准
满足以下条件即可通过：
1. RGB 可读取，不为空帧
2. Depth 可读取，不为空帧
3. `camera_drop_counter_sum = 0`
4. 可导出内参、外参与矩阵元数据
5. headless 模式下运行稳定

## 当前结果
- `camera_drop_counter_sum = 0`
- `empty_rgb_steps_env0 = 0`
- `empty_depth_steps_env0 = 0`

## 输出文件
- `/tmp/g1_rgbd_inspect_smoke/rgb_first.png`
- `/tmp/g1_rgbd_inspect_smoke/depth_first.npy`
- `/tmp/g1_rgbd_inspect_smoke/depth_first_preview.png`
- `/tmp/g1_rgbd_inspect_smoke/rgbd_summary.json`

## 结论
Step 3.1 已完成。下一步进入 Step 3.2：视觉编码器与本体状态融合。
