import sys
from legged_gym import LEGGED_GYM_ROOT_DIR
import os
import sys
from legged_gym import LEGGED_GYM_ROOT_DIR

import isaacgym
from isaacgym import gymapi
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger

import numpy as np
import torch


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 100)
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False

    env_cfg.env.test = True

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    
    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print('Exported policy as jit script to: ', path)

    # Video recording
    import imageio
    video_path = os.path.join(LEGGED_GYM_ROOT_DIR, 'play.mp4')
    video = imageio.get_writer(video_path, fps=30)
    print(f"Recording video to {video_path}")

    for i in range(10*int(env.max_episode_length)):
        actions = policy(obs.detach())
        obs, _, rews, dones, infos = env.step(actions.detach())
        
        if i % 50 == 0:
            # Check base state (assumes root_states is available or obs contains it)
            # Obs layout for G1: 0-2 lin vel, 3-5 ang vel, 6-8 proj grav, 9-44 commands/dof
            # We can't easily get exact pos from obs, but we can check if obs changes
            pass 
            # Better to rely on visual or just completion. 
            # Actually, let's print a simple "Step {i}" to confirm it runs.
            print(f"Sim Step {i}")
        
        # Capture Camera
        if hasattr(env, 'camera_handles') and len(env.camera_handles) > 0:
            img = env.gym.get_camera_image(env.sim, env.envs[0], env.camera_handles[0], gymapi.IMAGE_COLOR)
            if img.shape[0] > 0:
                h, w = env.cfg.sensor.camera.height, env.cfg.sensor.camera.width
                img = img.reshape(h, w, 4)
                img = img[:, :, :3] # Remove alpha
                video.append_data(img)
            
    video.close()
    print("Done recording")

if __name__ == '__main__':
    EXPORT_POLICY = True
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    play(args)
