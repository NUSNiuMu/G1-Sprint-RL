# Step 2.3 基线评估脚本说明

## 目标
为 `g1_sprint_track` 提供统一的离线评估入口，支持多随机种子、固定相机录像、轨迹指标与 episode 级指标汇总，作为后续感知与障碍阶段的统一基线。

## 脚本位置
- `src/unitree_rl_gym/humanoid_sprint/scripts/eval_g1_sprint_baseline.py`

## 依赖改动
### 1. play 指标持久化
- 文件：`src/unitree_rl_gym/legged_gym/scripts/play.py`
- 新增输出字段：
  - `num_completed_episodes`
  - `episode_metrics_mean`
  - `episode_metrics`

### 2. seed 传递修复
- 文件：`src/unitree_rl_gym/legged_gym/utils/helpers.py`
- 修复内容：当命令行显式传入 `--seed` 时，同时更新 `env_cfg.seed`

## 用法
在仓库目录下执行：

```bash
source /home/niumu/anaconda3/etc/profile.d/conda.sh && \
conda run -n unitree-rl python humanoid_sprint/scripts/eval_g1_sprint_baseline.py \
  --task g1_sprint_track \
  --experiment_name g1 \
  --load_run Mar22_18-32-10_autoloop_r20_longrun_from_r19_300_800 \
  --checkpoint 950 \
  --seeds 1,7,13 \
  --play_steps 1200 \
  --dt 0.02 \
  --output_dir /tmp/step2_3_eval_r20_950_v2 \
  --summary_name step2_3_baseline_eval_r20_950_v2
```

## 输出内容
脚本会在 `output_dir` 下生成：
- 每个 seed 的录像与轨迹指标目录
- 聚合汇总 JSON
- 聚合汇总 Markdown

## 当前推荐基线
- `run = Mar22_18-32-10_autoloop_r20_longrun_from_r19_300_800`
- `checkpoint = 950`

## 当前验收结果
- `mean_avg_forward_speed_mps = 0.9996`
- `metric_speed_mps = 1.0238 +/- 0.0047`
- `metric_finish_rate = 1.0`
- `metric_fall_rate = 0.0`
- `metric_out_of_track_fail_rate = 0.0`
- `metric_lane_violation_rate = 0.0`
- `semantic_lane_ratio = 1.0`

## 判定标准
满足以下条件即可通过：
1. 多随机种子评估结果稳定
2. 平均前向速度接近当前训练目标
3. 不出现跌倒、越道失败或跑道违规
4. 评估命令和输出文件可直接复用

## 结论
Step 2.3 已通过。该脚本与当前基线模型可直接作为 Step 3 感知、障碍和后续 sim2real 前的标准验收入口。
