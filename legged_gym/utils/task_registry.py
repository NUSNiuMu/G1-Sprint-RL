import os
from datetime import datetime
from typing import Tuple
import torch
import numpy as np
import sys

from rsl_rl.env import VecEnv
from rsl_rl.runners import OnPolicyRunner

from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
from .helpers import get_args, update_cfg_from_args, class_to_dict, get_load_path, set_seed, parse_sim_params
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


def _flexible_load_actor_critic(module, loaded_state_dict):
    current_state = module.state_dict()
    merged_state = {}
    exact_keys = []
    expanded_keys = []
    skipped_keys = []

    for key, current_tensor in current_state.items():
        if key not in loaded_state_dict:
            skipped_keys.append(key)
            continue

        loaded_tensor = loaded_state_dict[key]
        if loaded_tensor.shape == current_tensor.shape:
            merged_state[key] = loaded_tensor
            exact_keys.append(key)
            continue

        if (
            loaded_tensor.ndim == current_tensor.ndim
            and loaded_tensor.ndim >= 2
            and tuple(loaded_tensor.shape[:-1]) == tuple(current_tensor.shape[:-1])
        ):
            overlap = min(loaded_tensor.shape[-1], current_tensor.shape[-1])
            patched_tensor = current_tensor.clone()
            patched_tensor[..., :overlap] = loaded_tensor[..., :overlap]
            merged_state[key] = patched_tensor
            expanded_keys.append(key)
            continue

        skipped_keys.append(key)

    current_state.update(merged_state)
    module.load_state_dict(current_state, strict=False)
    return exact_keys, expanded_keys, skipped_keys

class TaskRegistry():
    """ 任务注册表类，用于管理、创建环境和训练算法。
    它负责存储环境类及其对应的配置，并提供统一的接口来实例化这些对象。
    """
    def __init__(self):
        """ 初始化注册表，创建用于存储任务类、环境配置和训练配置的字典。 """
        self.task_classes = {}
        self.env_cfgs = {}
        self.train_cfgs = {}
    
    def register(self, name: str, task_class: VecEnv, env_cfg: LeggedRobotCfg, train_cfg: LeggedRobotCfgPPO):
        """ 将一个任务及其配置注册到注册表中。

        Args:
            name (str): 任务的唯一标识名称。
            task_class (VecEnv): 环境对应的类。
            env_cfg (LeggedRobotCfg): 环境的配置信息。
            train_cfg (LeggedRobotCfgPPO): 训练算法（如PPO）的配置信息。
        """
        self.task_classes[name] = task_class
        self.env_cfgs[name] = env_cfg
        self.train_cfgs[name] = train_cfg
    
    def get_task_class(self, name: str) -> VecEnv:
        """ 根据名称获取注册的任务类。 """
        return self.task_classes[name]
    
    def get_cfgs(self, name) -> Tuple[LeggedRobotCfg, LeggedRobotCfgPPO]:
        """ 获取指定任务的环境配置和训练配置。同时会将训练配置中的随机种子同步到环境配置中。 """
        train_cfg = self.train_cfgs[name]
        env_cfg = self.env_cfgs[name]
        # 同步随机种子
        env_cfg.seed = train_cfg.seed
        return env_cfg, train_cfg
    
    def make_env(self, name, args=None, env_cfg=None) -> Tuple[VecEnv, LeggedRobotCfg]:
        """ 根据注册名称或提供的配置创建一个环境实例。

        Args:
            name (string): 已注册的环境名称。
            args (Args, 可选): Isaac Gym 命令行参数。如果为 None，则调用 get_args()。
            env_cfg (Dict, 可选): 用于覆盖已注册配置的环境配置字典。

        Raises:
            ValueError: 如果找不到对应的注册环境名称。

        Returns:
            isaacgym.VecTaskPython: The created environment
            Dict: the corresponding config file
        """
        # 如果没有传入参数，则从命令行获取
        if args is None:
            args = get_args()
        # 检查环境是否已注册
        if name in self.task_classes:
            task_class = self.get_task_class(name)
        else:
            raise ValueError(f"Task with name: {name} was not registered")
        if env_cfg is None:
            # 加载注册的配置文件
            env_cfg, _ = self.get_cfgs(name)
        
        # 根据命令行参数更新配置（如果指定了参数）
        env_cfg, _ = update_cfg_from_args(env_cfg, None, args)
        set_seed(env_cfg.seed)
        
        # 解析仿真参数（先转换为字典）
        sim_params = {"sim": class_to_dict(env_cfg.sim)}
        sim_params = parse_sim_params(args, sim_params)
        
        # 实例化环境类
        env = task_class(   cfg=env_cfg,
                            sim_params=sim_params,
                            physics_engine=args.physics_engine,
                            sim_device=args.sim_device,
                            headless=args.headless)
        return env, env_cfg

    def make_alg_runner(self, env, name=None, args=None, train_cfg=None, log_root="default") -> Tuple[OnPolicyRunner, LeggedRobotCfgPPO]:
        """ 创建训练算法运行器（Runner）。

        Args:
            env (isaacgym.VecTaskPython): The environment to train (TODO: remove from within the algorithm)
            name (string, optional): Name of a registered env. If None, the config file will be used instead. Defaults to None.
            args (Args, optional): Isaac Gym comand line arguments. If None get_args() will be called. Defaults to None.
            train_cfg (Dict, optional): Training config file. If None 'name' will be used to get the config file. Defaults to None.
            log_root (str, optional): Logging directory for Tensorboard. Set to 'None' to avoid logging (at test time for example). 
                                      Logs will be saved in <log_root>/<date_time>_<run_name>. Defaults to "default"=<path_to_LEGGED_GYM>/logs/<experiment_name>.

        Raises:
            ValueError: Error if neither 'name' or 'train_cfg' are provided
            Warning: If both 'name' or 'train_cfg' are provided 'name' is ignored

        Returns:
            PPO: The created algorithm
            Dict: the corresponding config file
        """
        # if no args passed get command line arguments
        if args is None:
            args = get_args()
        # if config files are passed use them, otherwise load from the name
        if train_cfg is None:
            if name is None:
                raise ValueError("Either 'name' or 'train_cfg' must be not None")
            # load config files
            _, train_cfg = self.get_cfgs(name)
        else:
            if name is not None:
                print(f"'train_cfg' provided -> Ignoring 'name={name}'")
        # override cfg from args (if specified)
        _, train_cfg = update_cfg_from_args(None, train_cfg, args)

        # 设置日志目录
        if log_root=="default":
            log_root = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name)
            log_dir = os.path.join(log_root, datetime.now().strftime('%b%d_%H-%M-%S') + '_' + train_cfg.runner.run_name)
        elif log_root is None:
            log_dir = None
        else:
            log_dir = os.path.join(log_root, datetime.now().strftime('%b%d_%H-%M-%S') + '_' + train_cfg.runner.run_name)

        # 在创建新日志目录之前先解析 resume 路径，否则“最新 run”会错误指向当前空目录。
        resume = train_cfg.runner.resume
        resume_model_only = getattr(train_cfg.runner, "resume_model_only", False)
        resume_path = None
        if resume:
            resume_path = get_load_path(log_root, load_run=train_cfg.runner.load_run, checkpoint=train_cfg.runner.checkpoint)

        # 将配置类转换为字典以适配 Runner
        train_cfg_dict = class_to_dict(train_cfg)
        runner = OnPolicyRunner(env, train_cfg_dict, log_dir, device=args.rl_device)
        if resume and resume_path is not None:
            # 加载之前训练的模型权重
            print(f"Loading model from: {resume_path}")
            if resume_model_only:
                loaded_dict = torch.load(resume_path, map_location=args.rl_device)
                exact_keys, expanded_keys, skipped_keys = _flexible_load_actor_critic(
                    runner.alg.actor_critic, loaded_dict["model_state_dict"]
                )
                runner.current_learning_iteration = 0
                print(
                    "Model-only transfer load summary: "
                    f"exact={len(exact_keys)}, expanded={len(expanded_keys)}, skipped={len(skipped_keys)}"
                )
            else:
                runner.load(resume_path, load_optimizer=True)

        return runner, train_cfg

# 创建全局任务注册表实例
task_registry = TaskRegistry()
