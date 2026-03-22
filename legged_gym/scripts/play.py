import os
import json
from datetime import datetime
from pathlib import Path

import isaacgym
from isaacgym import gymapi
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *
from legged_gym.utils import get_args, export_policy_as_jit, task_registry, Logger

import imageio.v2 as imageio
import numpy as np
import torch


def _prepare_record_dir(train_cfg, args):
    if args.record_dir is not None:
        base_dir = Path(args.record_dir)
    else:
        base_dir = Path(LEGGED_GYM_ROOT_DIR) / "logs" / train_cfg.runner.experiment_name / "eval_recordings"
    out_dir = base_dir / f"{args.task}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _create_record_camera(env, env_cfg, args):
    if env.graphics_device_id < 0:
        raise RuntimeError("record_play requires graphics. Run play without --headless.")
    camera_props = gymapi.CameraProperties()
    camera_props.width = args.record_width
    camera_props.height = args.record_height
    camera_props.enable_tensors = False
    camera_handle = env.gym.create_camera_sensor(env.envs[0], camera_props)
    _update_record_camera(env, env_cfg, camera_handle, args)
    return camera_handle, camera_props


def _update_record_camera(env, env_cfg, camera_handle, args):
    mode = (args.record_camera_mode or "fixed").lower()
    if mode == "follow":
        base_pos = env.root_states[0, :3].detach().cpu().numpy()
        eye = gymapi.Vec3(float(base_pos[0] - 2.5), float(base_pos[1] - 1.5), float(base_pos[2] + 1.2))
        target = gymapi.Vec3(float(base_pos[0] + 1.5), float(base_pos[1]), float(base_pos[2] + 0.7))
    else:
        env_origin = env.env_origins[0].detach().cpu().numpy()
        eye_cfg = np.asarray(env_cfg.viewer.pos, dtype=np.float32)
        target_cfg = np.asarray(env_cfg.viewer.lookat, dtype=np.float32)
        eye = gymapi.Vec3(*(env_origin + eye_cfg).tolist())
        target = gymapi.Vec3(*(env_origin + target_cfg).tolist())
    env.gym.set_camera_location(camera_handle, env.envs[0], eye, target)


def _capture_camera_frame(env, camera_handle, camera_props):
    env.gym.step_graphics(env.sim)
    env.gym.render_all_camera_sensors(env.sim)
    image = env.gym.get_camera_image(env.sim, env.envs[0], camera_handle, gymapi.IMAGE_COLOR)
    if image.shape[0] == 0:
        return None
    return image.reshape(camera_props.height, camera_props.width, 4)[:, :, :3].copy()


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    if args.record_play:
        env_cfg.env.num_envs = 1
    else:
        env_cfg.env.num_envs = min(env_cfg.env.num_envs, 100)
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.env.test = True

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()

    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)

    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", train_cfg.runner.experiment_name, "exported", "policies")
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print("Exported policy as jit script to:", path)

    logger = Logger(env.dt)
    stop_rew_log = env.max_episode_length + 1
    total_steps = args.play_steps if args.play_steps is not None else 10 * int(env.max_episode_length)

    writer = None
    record_dir = None
    camera_handle = None
    camera_props = None
    first_frame_saved = False
    last_frame = None
    positions = []
    reset_steps = []
    episode_summaries = []

    if args.record_play:
        record_dir = _prepare_record_dir(train_cfg, args)
        camera_handle, camera_props = _create_record_camera(env, env_cfg, args)
        fps = max(1, int(round((1.0 / env.dt) / max(1, args.record_interval))))
        writer = imageio.get_writer(record_dir / "play.mp4", fps=min(60, fps))
        print("Recording play evaluation to:", record_dir)
        print("Recording camera mode:", (args.record_camera_mode or "fixed").lower())

    for i in range(total_steps):
        positions.append(env.root_states[0, :3].detach().cpu().numpy().copy())
        actions = policy(obs.detach())
        obs, _, rews, dones, infos = env.step(actions.detach())
        if bool(dones[0].item()):
            reset_steps.append(i)

        if writer is not None and i % max(1, args.record_interval) == 0:
            _update_record_camera(env, env_cfg, camera_handle, args)
            frame = _capture_camera_frame(env, camera_handle, camera_props)
            if frame is not None:
                writer.append_data(frame)
                if not first_frame_saved:
                    imageio.imwrite(record_dir / "frame_start.png", frame)
                    first_frame_saved = True
                last_frame = frame

        if 0 < i < stop_rew_log:
            if infos["episode"]:
                num_episodes = torch.sum(env.reset_buf).item()
                if num_episodes > 0:
                    logger.log_rewards(infos["episode"], num_episodes)
                    episode_summary = {"step": int(i), "num_episodes": int(num_episodes)}
                    for key, value in infos["episode"].items():
                        if torch.is_tensor(value):
                            if value.numel() == 1:
                                episode_summary[key] = float(value.item())
                            else:
                                episode_summary[key] = float(value.float().mean().item())
                        elif isinstance(value, (int, float)):
                            episode_summary[key] = float(value)
                    episode_summaries.append(episode_summary)
        elif i == stop_rew_log:
            logger.print_rewards()

    positions.append(env.root_states[0, :3].detach().cpu().numpy().copy())
    pos_arr = np.asarray(positions, dtype=np.float32)
    delta = pos_arr[-1] - pos_arr[0]
    step_delta = pos_arr[1:, :2] - pos_arr[:-1, :2]
    valid_step_mask = np.ones(len(step_delta), dtype=bool)
    for step in reset_steps:
        if 0 <= step < len(valid_step_mask):
            # Ignore teleport jumps caused by successful/terminal env resets.
            valid_step_mask[step] = False
    filtered_step_delta = step_delta[valid_step_mask]
    path_len_xy = float(np.linalg.norm(filtered_step_delta, axis=1).sum()) if len(filtered_step_delta) > 0 else 0.0
    cumulative_positive_dx = float(np.clip(filtered_step_delta[:, 0], a_min=0.0, a_max=None).sum()) if len(filtered_step_delta) > 0 else 0.0
    cumulative_negative_dx = float(np.clip(filtered_step_delta[:, 0], a_min=None, a_max=0.0).sum()) if len(filtered_step_delta) > 0 else 0.0
    cumulative_abs_dy = float(np.abs(filtered_step_delta[:, 1]).sum()) if len(filtered_step_delta) > 0 else 0.0
    segment_forward_dx = []
    segment_start = 0
    for step in reset_steps:
        segment_end = min(step + 1, len(pos_arr) - 1)
        segment_forward_dx.append(float(pos_arr[segment_end, 0] - pos_arr[segment_start, 0]))
        segment_start = min(step + 1, len(pos_arr) - 1)
    if segment_start < len(pos_arr) - 1:
        segment_forward_dx.append(float(pos_arr[-1, 0] - pos_arr[segment_start, 0]))
    net_xy = float(np.linalg.norm(delta[:2]))
    trajectory_metrics = {
        "camera_mode": (args.record_camera_mode or "fixed").lower(),
        "total_steps": int(total_steps),
        "start_pos": pos_arr[0].tolist(),
        "end_pos": pos_arr[-1].tolist(),
        "delta_xyz": delta.tolist(),
        "net_x": float(delta[0]),
        "net_y": float(delta[1]),
        "net_xy": net_xy,
        "path_len_xy": path_len_xy,
        "cumulative_positive_dx": cumulative_positive_dx,
        "cumulative_negative_dx": cumulative_negative_dx,
        "cumulative_abs_dy": cumulative_abs_dy,
        "segment_forward_dx": segment_forward_dx,
        "best_segment_forward_dx": max(segment_forward_dx) if segment_forward_dx else 0.0,
        "reset_count": int(len(reset_steps)),
        "first_reset_step": int(reset_steps[0]) if reset_steps else None,
    }
    if episode_summaries:
        metric_keys = sorted({key for episode in episode_summaries for key in episode.keys() if key not in {"step", "num_episodes"}})
        metric_means = {}
        for key in metric_keys:
            values = [episode[key] for episode in episode_summaries if key in episode]
            if values:
                metric_means[key] = float(np.mean(values))
        trajectory_metrics["num_completed_episodes"] = len(episode_summaries)
        trajectory_metrics["episode_metrics_mean"] = metric_means
        trajectory_metrics["episode_metrics"] = episode_summaries
    else:
        trajectory_metrics["num_completed_episodes"] = 0
        trajectory_metrics["episode_metrics_mean"] = {}
        trajectory_metrics["episode_metrics"] = []
    print("Play trajectory metrics:", trajectory_metrics)

    if writer is not None:
        if last_frame is not None:
            imageio.imwrite(record_dir / "frame_end.png", last_frame)
        with open(record_dir / "trajectory_metrics.json", "w", encoding="utf-8") as f:
            json.dump(trajectory_metrics, f, indent=2)
        writer.close()
        print("Saved play recording to:", record_dir)


if __name__ == '__main__':
    EXPORT_POLICY = True
    args = get_args()
    play(args)
