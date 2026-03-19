# Humanoid Sprint Workspace Layout

This folder isolates our sprint project artifacts from the official baseline `logs/`.

## Directory layout

- `humanoid_sprint/runs/`
  - One folder per training run metadata and quick notes.
- `humanoid_sprint/logs/`
  - TensorBoard/W&B exports, evaluation logs, rollout traces.
- `humanoid_sprint/configs/`
  - Versioned experiment configs (env/reward/train/domain-rand).
- `humanoid_sprint/assets/`
  - Project-only assets (track templates, obstacle assets, semantic materials).
- `humanoid_sprint/reports/`
  - Step reports, ablation summaries, milestone records.
- `humanoid_sprint/scripts/`
  - Helper scripts for train/eval/export/deploy wrappers.

## Naming convention

Use one canonical run id everywhere:

`{date}_{task}_{stage}_{sensor}_{envs}_{seed}_{tag}`

Example:

`20260319_g1_sprint_s2_rgbd_e1024_s1_baseline`

Field meaning:
- `date`: `YYYYMMDD`
- `task`: `g1` / `g1_sprint`
- `stage`: `s1`(run), `s2`(lane-keep), `s3`(obstacle), `s4`(speed), `s5`(sim2real)
- `sensor`: `blind` / `rgbd`
- `envs`: parallel env count, e.g. `e256`, `e1024`
- `seed`: e.g. `s1`, `s2`, `s3`
- `tag`: free label, e.g. `baseline`, `dr_v2`, `ablate_reward`

## Config versioning rule

- Keep immutable snapshots in `configs/` before each training run.
- Suggested files:
  - `configs/<run_id>.env.yaml`
  - `configs/<run_id>.train.yaml`
  - `configs/<run_id>.reward.yaml`
  - `configs/<run_id>.notes.md`

## Minimal run checklist

1. Create `run_id`.
2. Snapshot config files into `configs/`.
3. Start training with explicit `--num_envs` and `--seed`.
4. Save eval summary into `reports/`.
5. Link checkpoint path and metrics into `runs/<run_id>.md`.

## Recommended launch style

Use explicit devices and env count to avoid accidental overload:

```bash
TORCH_EXTENSIONS_DIR=/tmp/torch_extensions \
conda run -n unitree-rl python legged_gym/scripts/train.py \
  --task=g1 --headless --num_envs=256 --seed=1
```

