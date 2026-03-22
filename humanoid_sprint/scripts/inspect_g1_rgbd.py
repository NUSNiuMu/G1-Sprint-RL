import argparse
import json
import os
from types import SimpleNamespace

import imageio.v2 as imageio
import numpy as np
from isaacgym import gymapi

from legged_gym.envs import *
from legged_gym.utils.task_registry import task_registry
import torch


def build_args(cli_args):
    sim_device = cli_args.sim_device
    return SimpleNamespace(
        task=cli_args.task,
        resume=False,
        resume_model_only=False,
        experiment_name=None,
        run_name=None,
        load_run=None,
        checkpoint=None,
        headless=cli_args.headless,
        horovod=False,
        rl_device=sim_device,
        num_envs=cli_args.num_envs,
        seed=cli_args.seed,
        max_iterations=None,
        use_wandb=False,
        wandb_project="legged_gym",
        wandb_entity=None,
        play_steps=None,
        record_play=False,
        record_dir=None,
        record_width=960,
        record_height=540,
        record_interval=2,
        record_camera_mode="fixed",
        physics_engine=gymapi.SIM_PHYSX,
        device="cuda" if sim_device.startswith("cuda") else "cpu",
        use_gpu=sim_device.startswith("cuda"),
        subscenes=0,
        use_gpu_pipeline=sim_device.startswith("cuda"),
        num_threads=0,
        sim_device=sim_device,
        sim_device_id=int(sim_device.split(":")[1]) if ":" in sim_device else 0,
    )


def save_depth_preview(depth_frame, output_path):
    finite_depth = depth_frame[np.isfinite(depth_frame)]
    if finite_depth.size == 0:
        preview = np.zeros_like(depth_frame, dtype=np.uint8)
    else:
        max_depth = np.percentile(finite_depth, 95)
        max_depth = max(max_depth, 1e-3)
        preview = np.clip(depth_frame / max_depth, 0.0, 1.0)
        preview = (255.0 * (1.0 - preview)).astype(np.uint8)
    imageio.imwrite(output_path, preview)


def main():
    parser = argparse.ArgumentParser(description="Inspect RealSense-style RGB-D stream for G1 sprint task.")
    parser.add_argument("--task", type=str, default="g1_sprint_track_rgbd")
    parser.add_argument("--num_envs", type=int, default=8)
    parser.add_argument("--num_steps", type=int, default=120)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--sim_device", type=str, default="cuda:0")
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--output_dir", type=str, default="/tmp/g1_rgbd_inspect")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    lg_args = build_args(args)
    env, env_cfg = task_registry.make_env(name=args.task, args=lg_args)
    obs, _ = env.reset()
    del obs

    empty_rgb_steps = 0
    empty_depth_steps = 0
    first_rgb_saved = False
    first_depth_saved = False

    for step in range(args.num_steps):
        actions = torch.zeros(env.num_envs, env.num_actions, device=env.device)
        env.step(actions)

        if env.camera_rgb_buf is None or env.camera_depth_buf is None:
            raise RuntimeError("Camera buffers were not initialized. Check sensor.camera.enabled configuration.")

        rgb = env.camera_rgb_buf[0].detach().cpu().numpy()
        depth = env.camera_depth_buf[0].detach().cpu().numpy()

        if np.allclose(rgb, 0.0):
            empty_rgb_steps += 1
        if not np.any(np.isfinite(depth)) or np.allclose(depth, 0.0):
            empty_depth_steps += 1

        if not first_rgb_saved:
            imageio.imwrite(os.path.join(args.output_dir, "rgb_first.png"), (255.0 * np.clip(rgb, 0.0, 1.0)).astype(np.uint8))
            first_rgb_saved = True
        if not first_depth_saved:
            np.save(os.path.join(args.output_dir, "depth_first.npy"), depth)
            save_depth_preview(depth, os.path.join(args.output_dir, "depth_first_preview.png"))
            first_depth_saved = True

    metadata = env.get_camera_metadata(env_id=0)
    summary = {
        "task": args.task,
        "num_envs": args.num_envs,
        "num_steps": args.num_steps,
        "seed": args.seed,
        "sim_device": args.sim_device,
        "camera_drop_counter_sum": int(env.camera_drop_counter.sum().item()) if env.camera_drop_counter is not None else 0,
        "empty_rgb_steps_env0": int(empty_rgb_steps),
        "empty_depth_steps_env0": int(empty_depth_steps),
        "camera_metadata": metadata,
    }

    with open(os.path.join(args.output_dir, "rgbd_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
