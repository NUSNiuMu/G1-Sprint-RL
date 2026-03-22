#!/usr/bin/env python3
import argparse
import json
import subprocess
import sys
from pathlib import Path
from statistics import mean, pstdev

def _parse_args():
    parser = argparse.ArgumentParser(description="Step 2.3 baseline evaluator for G1 sprint track.")
    parser.add_argument("--task", type=str, default="g1_sprint_track")
    parser.add_argument("--experiment_name", type=str, default="g1")
    parser.add_argument("--load_run", type=str, required=True)
    parser.add_argument("--checkpoint", type=int, required=True)
    parser.add_argument("--seeds", type=str, default="1,2,3", help="Comma-separated random seeds.")
    parser.add_argument("--play_steps", type=int, default=1500)
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--dt", type=float, default=0.02, help="Simulation control dt used to convert displacement to speed.")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--summary_name", type=str, default=None)
    return parser.parse_args()


def _default_output_dir(repo_root: Path, task_name: str, run_name: str, checkpoint: int):
    return repo_root / "humanoid_sprint" / "logs" / "baseline_eval" / f"{task_name}_{run_name}_ckpt{checkpoint}"


def _run_single_eval(repo_root: Path, args, seed: int, seed_dir: Path):
    seed_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "legged_gym/scripts/play.py",
        "--task", args.task,
        "--num_envs", str(args.num_envs),
        "--experiment_name", args.experiment_name,
        "--load_run", args.load_run,
        "--checkpoint", str(args.checkpoint),
        "--seed", str(seed),
        "--play_steps", str(args.play_steps),
        "--record_play",
        "--record_camera_mode", "fixed",
        "--record_dir", str(seed_dir),
    ]
    subprocess.run(cmd, cwd=repo_root, check=True)
    candidates = sorted([p for p in seed_dir.iterdir() if p.is_dir()])
    if not candidates:
        raise RuntimeError(f"No evaluation artifacts found under {seed_dir}")
    latest = candidates[-1]
    metrics_path = latest / "trajectory_metrics.json"
    if not metrics_path.exists():
        raise RuntimeError(f"Missing trajectory_metrics.json under {latest}")
    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    metrics["artifact_dir"] = str(latest)
    metrics["seed"] = seed
    return metrics


def _safe_mean(values):
    return mean(values) if values else 0.0


def _safe_std(values):
    return pstdev(values) if len(values) > 1 else 0.0


def _build_summary(results, dt, args):
    avg_forward_speeds = [result["cumulative_positive_dx"] / (args.play_steps * dt) for result in results]
    net_forward_speeds = [result["net_x"] / (args.play_steps * dt) for result in results]
    metric_keys = sorted({key for result in results for key in result.get("episode_metrics_mean", {}).keys()})
    episode_metric_means = {}
    for key in metric_keys:
        values = [result["episode_metrics_mean"][key] for result in results if key in result.get("episode_metrics_mean", {})]
        if values:
            episode_metric_means[key] = {
                "mean": _safe_mean(values),
                "std": _safe_std(values),
            }

    return {
        "task": args.task,
        "experiment_name": args.experiment_name,
        "load_run": args.load_run,
        "checkpoint": args.checkpoint,
        "play_steps": args.play_steps,
        "dt": dt,
        "seeds": [result["seed"] for result in results],
        "aggregate": {
            "mean_net_x": _safe_mean([result["net_x"] for result in results]),
            "std_net_x": _safe_std([result["net_x"] for result in results]),
            "mean_net_y": _safe_mean([result["net_y"] for result in results]),
            "std_net_y": _safe_std([result["net_y"] for result in results]),
            "mean_abs_net_y": _safe_mean([abs(result["net_y"]) for result in results]),
            "mean_path_len_xy": _safe_mean([result["path_len_xy"] for result in results]),
            "mean_cumulative_positive_dx": _safe_mean([result["cumulative_positive_dx"] for result in results]),
            "mean_cumulative_abs_dy": _safe_mean([result["cumulative_abs_dy"] for result in results]),
            "mean_avg_forward_speed_mps": _safe_mean(avg_forward_speeds),
            "std_avg_forward_speed_mps": _safe_std(avg_forward_speeds),
            "mean_net_forward_speed_mps": _safe_mean(net_forward_speeds),
            "mean_reset_count": _safe_mean([result["reset_count"] for result in results]),
        },
        "episode_metric_means": episode_metric_means,
        "trials": results,
    }


def _write_summary(summary, output_dir: Path, summary_name: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{summary_name}.json"
    md_path = output_dir / f"{summary_name}.md"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    agg = summary["aggregate"]
    episode = summary["episode_metric_means"]
    lines = [
        f"# Step 2.3 Baseline Eval Summary",
        "",
        f"- task: `{summary['task']}`",
        f"- run: `{summary['load_run']}`",
        f"- checkpoint: `model_{summary['checkpoint']}.pt`",
        f"- seeds: `{summary['seeds']}`",
        f"- play_steps: `{summary['play_steps']}`",
        "",
        "## 聚合结果",
        f"- 平均净前进距离 `mean_net_x`: `{agg['mean_net_x']:.4f} m`",
        f"- 平均横向净偏移 `mean_net_y`: `{agg['mean_net_y']:.4f} m`",
        f"- 平均横向净偏移绝对值 `mean_abs_net_y`: `{agg['mean_abs_net_y']:.4f} m`",
        f"- 平均累计正向位移 `mean_cumulative_positive_dx`: `{agg['mean_cumulative_positive_dx']:.4f} m`",
        f"- 平均累计横向摆动 `mean_cumulative_abs_dy`: `{agg['mean_cumulative_abs_dy']:.4f} m`",
        f"- 平均正向速度 `mean_avg_forward_speed_mps`: `{agg['mean_avg_forward_speed_mps']:.4f} m/s`",
        f"- 正向速度标准差 `std_avg_forward_speed_mps`: `{agg['std_avg_forward_speed_mps']:.4f}`",
        f"- 平均 reset 次数 `mean_reset_count`: `{agg['mean_reset_count']:.4f}`",
        "",
        "## Episode 指标均值",
    ]
    for key in sorted(episode.keys()):
        lines.append(f"- `{key}`: mean=`{episode[key]['mean']:.6f}`, std=`{episode[key]['std']:.6f}`")
    lines.extend(["", "## 单次试验", ""])
    for result in summary["trials"]:
        lines.extend([
            f"### seed {result['seed']}",
            f"- `net_x`: `{result['net_x']:.4f} m`",
            f"- `net_y`: `{result['net_y']:.4f} m`",
            f"- `cumulative_positive_dx`: `{result['cumulative_positive_dx']:.4f} m`",
            f"- `cumulative_abs_dy`: `{result['cumulative_abs_dy']:.4f} m`",
            f"- `reset_count`: `{result['reset_count']}`",
            f"- `artifact_dir`: `{result['artifact_dir']}`",
            "",
        ])
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return json_path, md_path


def main():
    args = _parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    dt = args.dt
    seeds = [int(seed.strip()) for seed in args.seeds.split(",") if seed.strip()]
    output_dir = Path(args.output_dir) if args.output_dir else _default_output_dir(repo_root, args.task, args.load_run, args.checkpoint)
    summary_name = args.summary_name or "baseline_eval_summary"

    results = []
    for seed in seeds:
        seed_dir = output_dir / f"seed_{seed}"
        print(f"[step2.3] Evaluating seed {seed} -> {seed_dir}")
        results.append(_run_single_eval(repo_root, args, seed, seed_dir))

    summary = _build_summary(results, dt, args)
    json_path, md_path = _write_summary(summary, output_dir, summary_name)
    print(f"[step2.3] Saved summary JSON to: {json_path}")
    print(f"[step2.3] Saved summary Markdown to: {md_path}")


if __name__ == "__main__":
    main()
