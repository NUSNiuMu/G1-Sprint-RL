from legged_gym import LEGGED_GYM_ROOT_DIR, envs
import time
from warnings import WarningMessage
import numpy as np
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
from torch import Tensor
from typing import Tuple, Dict

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.base_task import BaseTask
from legged_gym.utils.math import wrap_to_pi
from legged_gym.utils.isaacgym_utils import get_euler_xyz as get_euler_xyz_in_tensor
from legged_gym.utils.helpers import class_to_dict
from .legged_robot_config import LeggedRobotCfg

class LeggedRobot(BaseTask):
    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
        """ Parses the provided config file,
            calls create_sim() (which creates, simulation and environments),
            initilizes pytorch buffers used during training

        Args:
            cfg (Dict): Environment config file
            sim_params (gymapi.SimParams): simulation parameters
            physics_engine (gymapi.SimType): gymapi.SIM_PHYSX (must be PhysX)
            device_type (string): 'cuda' or 'cpu'
            device_id (int): 0, 1, ...
            headless (bool): Run without rendering if True
        """
        self.cfg = cfg
        self.sim_params = sim_params
        self.height_samples = None
        self.debug_viz = False
        self.init_done = False
        self._parse_cfg(self.cfg)
        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)

        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
        self._init_buffers()
        self._prepare_reward_function()
        self.init_done = True

    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """

        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        # step physics and render each frame
        self.render()
        for _ in range(self.cfg.control.decimation):
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)

            if self.cfg.env.test:
                elapsed_time = self.gym.get_elapsed_time(self.sim)
                sim_time = self.gym.get_sim_time(self.sim)
                if sim_time-elapsed_time>0:
                    time.sleep(sim_time-elapsed_time)
            
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
        self.post_physics_step()

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras

    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations 
            calls self._draw_debug_vis() if needed
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        # prepare quantities
        self.base_pos[:] = self.root_states[:, 0:3]
        self.base_quat[:] = self.root_states[:, 3:7]
        self.rpy[:] = get_euler_xyz_in_tensor(self.base_quat[:])
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        self._post_physics_step_callback()

        # compute observations, rewards, resets, ...
        self.check_termination()
        self._update_track_semantics()
        self._update_episode_metrics()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)
        
        if self.cfg.domain_rand.push_robots:
            self._push_robots()

        self.compute_observations() # in some cases a simulation step might be required to refresh some obs (for example body positions)

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]
        self.last_track_local_x[:] = self.root_states[:, 0] - self.env_origins[:, 0]
        if self.cfg.terrain.track.enabled and self.cfg.terrain.track.visualize_in_viewer and not self.headless:
            self._draw_track_debug_lines()

    def check_termination(self):
        """ Check if environments need to be reset
        """
        self.contact_termination_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)
        self.attitude_termination_buf = torch.logical_or(torch.abs(self.rpy[:,1])>1.0, torch.abs(self.rpy[:,0])>0.8)
        self.track_termination_buf[:] = False
        self.finish_termination_buf[:] = False
        self.reset_buf = self.contact_termination_buf | self.attitude_termination_buf
        if self.track_layout is not None:
            local_x = self.base_pos[:, 0] - self.env_origins[:, 0]
            local_y = self.base_pos[:, 1] - self.env_origins[:, 1]
            half_length = 0.5 * self.track_layout["lane_length"]
            margin = float(self.cfg.terrain.track.out_of_track_margin)
            in_lateral = torch.logical_and(
                local_y >= (self.track_layout["left_boundary"] + margin),
                local_y <= (self.track_layout["right_boundary"] - margin),
            )
            if self.cfg.terrain.track.success_on_reach_lane_end:
                finish_margin = float(self.cfg.terrain.track.lane_end_success_margin)
                safe_finish = ~(self.contact_termination_buf | self.attitude_termination_buf)
                self.finish_termination_buf = (local_x >= (half_length - finish_margin)) & in_lateral & safe_finish
            if self.cfg.terrain.track.terminate_on_out_of_track:
                past_track_end = local_x > half_length
                before_track_start = local_x < -half_length
                self.track_termination_buf = (~in_lateral) | past_track_end | before_track_start
                self.track_termination_buf &= ~self.finish_termination_buf
                self.reset_buf |= self.track_termination_buf
            self.reset_buf |= self.finish_termination_buf
        self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
        self.reset_buf |= self.time_out_buf

    def reset_idx(self, env_ids):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        if len(env_ids) == 0:
            return
        
        # DEBUG: Print reset occurrence
        # print(f"DEBUG: Resetting envs {env_ids}")
        
        # reset robot states
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)

        self._resample_commands(env_ids)

        # reset buffers
        self.actions[env_ids] = 0.
        self.last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.last_track_local_x[env_ids] = self.root_states[env_ids, 0] - self.env_origins[env_ids, 0]
        self.feet_air_time[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        ep_steps = torch.clamp(self.metric_episode_steps[env_ids], min=1.0)
        self.extras["episode"]["metric_speed_mps"] = torch.mean(self.metric_speed_sum[env_ids] / ep_steps)
        self.extras["episode"]["metric_collision_rate"] = torch.mean(self.metric_collision_steps[env_ids] / ep_steps)
        self.extras["episode"]["metric_fall_rate"] = torch.mean((self.contact_termination_buf[env_ids] | self.attitude_termination_buf[env_ids]).float())
        self.extras["episode"]["metric_success_rate"] = torch.mean(self.finish_termination_buf[env_ids].float())
        self.extras["episode"]["metric_finish_rate"] = torch.mean(self.finish_termination_buf[env_ids].float())
        self.extras["episode"]["metric_out_of_track_fail_rate"] = torch.mean(self.track_termination_buf[env_ids].float())
        self.extras["episode"]["metric_torque_utilization"] = torch.mean(self.metric_torque_util_sum[env_ids] / ep_steps)
        self.extras["episode"]["metric_torque_violation_rate"] = torch.mean(self.metric_torque_violation_steps[env_ids] / ep_steps)
        self.extras["episode"]["metric_lane_violation_rate"] = torch.mean(self.metric_lane_violation_steps[env_ids] / ep_steps)
        self.extras["episode"]["metric_obstacle_avoid_rate"] = torch.mean(self.metric_obstacle_avoid_sum[env_ids] / ep_steps)
        self.extras["episode"]["semantic_lane_ratio"] = torch.mean(self.metric_semantic_lane_steps[env_ids] / ep_steps)
        self.extras["episode"]["semantic_boundary_ratio"] = torch.mean(self.metric_semantic_boundary_steps[env_ids] / ep_steps)
        self.extras["episode"]["semantic_non_track_ratio"] = torch.mean(self.metric_lane_violation_steps[env_ids] / ep_steps)
        if self.track_layout is not None:
            self.extras["episode"]["track_num_lanes"] = float(self.track_layout["num_lanes"])
            self.extras["episode"]["track_lane_length"] = float(self.track_layout["lane_length"])
            self.extras["episode"]["track_lane_assignment_coverage"] = float(torch.unique(self.env_lane_ids).numel()) / float(self.track_layout["num_lanes"])
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf
        self.metric_episode_steps[env_ids] = 0.
        self.metric_speed_sum[env_ids] = 0.
        self.metric_collision_steps[env_ids] = 0.
        self.metric_torque_util_sum[env_ids] = 0.
        self.metric_torque_violation_steps[env_ids] = 0.
        self.metric_lane_violation_steps[env_ids] = 0.
        self.metric_obstacle_avoid_sum[env_ids] = 0.
        self.metric_semantic_lane_steps[env_ids] = 0.
        self.metric_semantic_boundary_steps[env_ids] = 0.
    
    def compute_reward(self):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew
    
    def compute_observations(self):
        """ Computes observations
        """
        self.obs_buf = torch.cat((  self.base_lin_vel * self.obs_scales.lin_vel,
                                    self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    self.commands[:, :3] * self.commands_scale,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions
                                    ),dim=-1)
        # add perceptive inputs if not blind
        # add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec


    def create_sim(self):
        """ Creates simulation, terrain and evironments
        """
        self.up_axis_idx = 2 # 2 for z, 1 for y -> adapt gravity accordingly
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs()

    def set_camera(self, position, lookat):
        """ Set camera position and direction
        """
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    #------------- Callbacks --------------
    def _process_rigid_shape_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the rigid shape properties of each environment.
            Called During environment creation.
            Base behavior: randomizes the friction of each environment

        Args:
            props (List[gymapi.RigidShapeProperties]): Properties of each shape of the asset
            env_id (int): Environment id

        Returns:
            [List[gymapi.RigidShapeProperties]]: Modified rigid shape properties
        """
        if self.cfg.domain_rand.randomize_friction:
            if env_id==0:
                # prepare friction randomization
                friction_range = self.cfg.domain_rand.friction_range
                num_buckets = 64
                bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1))
                friction_buckets = torch_rand_float(friction_range[0], friction_range[1], (num_buckets,1), device='cpu')
                self.friction_coeffs = friction_buckets[bucket_ids]

            for s in range(len(props)):
                props[s].friction = self.friction_coeffs[env_id]
        return props

    def _process_dof_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the DOF properties of each environment.
            Called During environment creation.
            Base behavior: stores position, velocity and torques limits defined in the URDF

        Args:
            props (numpy.array): Properties of each DOF of the asset
            env_id (int): Environment id

        Returns:
            [numpy.array]: Modified DOF properties
        """
        if env_id==0:
            self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False)
            self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            for i in range(len(props)):
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_vel_limits[i] = props["velocity"][i].item()
                self.torque_limits[i] = props["effort"][i].item()
                # soft limits
                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
                self.dof_pos_limits[i, 0] = m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
                self.dof_pos_limits[i, 1] = m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
        return props

    def _process_rigid_body_props(self, props, env_id):
        # if env_id==0:
        #     sum = 0
        #     for i, p in enumerate(props):
        #         sum += p.mass
        #         print(f"Mass of body {i}: {p.mass} (before randomization)")
        #     print(f"Total mass {sum} (before randomization)")
        # randomize base mass
        if self.cfg.domain_rand.randomize_base_mass:
            rng = self.cfg.domain_rand.added_mass_range
            props[0].mass += np.random.uniform(rng[0], rng[1])
        return props
    
    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        # 
        env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt)==0).nonzero(as_tuple=False).flatten()
        self._resample_commands(env_ids)
        if self.cfg.commands.heading_command:
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 2] = torch.clip(0.5*wrap_to_pi(self.commands[:, 3] - heading), -1., 1.)

    def _resample_commands(self, env_ids):
        """ Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch_rand_float(self.command_ranges["heading"][0], self.command_ranges["heading"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        else:
            self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1), device=self.device).squeeze(1)

        # set small commands to zero
        self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)

    def _compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        #pd controller
        actions_scaled = actions * self.cfg.control.action_scale
        control_type = self.cfg.control.control_type
        if control_type=="P":
            torques = self.p_gains*(actions_scaled + self.default_dof_pos - self.dof_pos) - self.d_gains*self.dof_vel
        elif control_type=="V":
            torques = self.p_gains*(actions_scaled - self.dof_vel) - self.d_gains*(self.dof_vel - self.last_dof_vel)/self.sim_params.dt
        elif control_type=="T":
            torques = actions_scaled
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def _reset_dofs(self, env_ids):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """
        self.dof_pos[env_ids] = self.default_dof_pos * torch_rand_float(0.5, 1.5, (len(env_ids), self.num_dof), device=self.device)
        self.dof_vel[env_ids] = 0.

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
    def _reset_root_states(self, env_ids):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        # base position
        if self.custom_origins:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            if self.track_layout is None:
                self.root_states[env_ids, :2] += torch_rand_float(-1., 1., (len(env_ids), 2), device=self.device) # xy position within 1m of the center
            else:
                self._reset_track_positions(env_ids)
        else:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            if self.track_layout is not None:
                self._reset_track_positions(env_ids)
        # base velocities
        # self.root_states[env_ids, 7:13] = torch_rand_float(-0.5, 0.5, (len(env_ids), 6), device=self.device) # [7:10]: lin vel, [10:13]: ang vel
        self.root_states[env_ids, 7:13] = 0. # Zero initial velocity for stability
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _reset_track_positions(self, env_ids):
        lane_center_y = self.env_lane_center_y[env_ids]
        x_jitter = min(self.cfg.terrain.track.spawn_x_jitter, 0.25 * self.track_layout["lane_length"])
        y_jitter_limit = max(
            0.0,
            0.5 * self.track_layout["lane_width"]
            - max(self.cfg.terrain.track.boundary_width, self.cfg.terrain.track.separator_width)
            - self.cfg.terrain.track.spawn_y_margin,
        )
        x_jitter_tensor = torch_rand_float(-x_jitter, x_jitter, (len(env_ids), 1), device=self.device).squeeze(1)
        if y_jitter_limit > 0.0:
            y_jitter_tensor = torch_rand_float(-y_jitter_limit, y_jitter_limit, (len(env_ids), 1), device=self.device).squeeze(1)
        else:
            y_jitter_tensor = torch.zeros(len(env_ids), device=self.device)
        self.root_states[env_ids, 0] += x_jitter_tensor
        self.root_states[env_ids, 1] += lane_center_y + y_jitter_tensor

    def _push_robots(self):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity. 
        """
        env_ids = torch.arange(self.num_envs, device=self.device)
        push_env_ids = env_ids[self.episode_length_buf[env_ids] % int(self.cfg.domain_rand.push_interval) == 0]
        if len(push_env_ids) == 0:
            return
        max_vel = self.cfg.domain_rand.max_push_vel_xy
        self.root_states[:, 7:9] = torch_rand_float(-max_vel, max_vel, (self.num_envs, 2), device=self.device) # lin vel x/y
        
        env_ids_int32 = push_env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                    gymtorch.unwrap_tensor(self.root_states),
                                                    gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

   
    
    def update_command_curriculum(self, env_ids):
        """ Implements a curriculum of increasing commands

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # If the tracking reward is above 80% of the maximum, increase the range of commands
        if torch.mean(self.episode_sums["tracking_lin_vel"][env_ids]) / self.max_episode_length > 0.8 * self.reward_scales["tracking_lin_vel"]:
            self.command_ranges["lin_vel_x"][0] = np.clip(self.command_ranges["lin_vel_x"][0] - 0.5, -self.cfg.commands.max_curriculum, 0.)
            self.command_ranges["lin_vel_x"][1] = np.clip(self.command_ranges["lin_vel_x"][1] + 0.5, 0., self.cfg.commands.max_curriculum)


    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[:3] = noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel
        noise_vec[3:6] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[6:9] = noise_scales.gravity * noise_level
        noise_vec[9:12] = 0. # commands
        noise_vec[12:12+self.num_actions] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[12+self.num_actions:12+2*self.num_actions] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[12+2*self.num_actions:12+3*self.num_actions] = 0. # previous actions

        return noise_vec

    #----------------------------------------
    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)

        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.base_quat = self.root_states[:, 3:7]
        self.rpy = get_euler_xyz_in_tensor(self.base_quat)
        self.base_pos = self.root_states[:, 0:3]

        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3) # shape: num_envs, num_bodies, xyz axis

        # initialize some data used later on
        self.common_step_counter = 0
        self.extras = {}
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.p_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float, device=self.device, requires_grad=False) # x vel, y vel, yaw vel, heading
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel], device=self.device, requires_grad=False,) # TODO change this
        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self.metric_episode_steps = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.metric_speed_sum = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.metric_collision_steps = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.metric_torque_util_sum = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.metric_torque_violation_steps = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.metric_lane_violation_steps = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.metric_obstacle_avoid_sum = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.metric_semantic_lane_steps = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.metric_semantic_boundary_steps = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.track_semantic_id_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device, requires_grad=False)
        self.contact_termination_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)
        self.attitude_termination_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)
        self.track_termination_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)
        self.finish_termination_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)
        self.last_track_local_x = self.root_states[:, 0] - self.env_origins[:, 0]
      

        # joint positions offsets and PD gains
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
                if self.cfg.control.control_type in ["P", "V"]:
                    print(f"PD gain of joint {name} were not defined, setting them to zero")
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)

    def _prepare_reward_function(self):
        """ Prepares a list of reward functions, whcih will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale==0:
                self.reward_scales.pop(key) 
            else:
                self.reward_scales[key] *= self.dt
        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name=="termination":
                continue
            self.reward_names.append(name)
            name = '_reward_' + name
            self.reward_functions.append(getattr(self, name))

        # reward episode sums
        self.episode_sums = {name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
                             for name in self.reward_scales.keys()}

    def _update_episode_metrics(self):
        # episode-local counters used for sprint KPI logging
        self.metric_episode_steps += 1.
        self.metric_speed_sum += torch.abs(self.base_lin_vel[:, 0])
        torque_util = torch.mean(torch.abs(self.torques) / (self.torque_limits.unsqueeze(0) + 1e-6), dim=1)
        self.metric_torque_util_sum += torque_util
        torque_limit = self.cfg.rewards.soft_torque_limit * self.torque_limits.unsqueeze(0)
        torque_violation = torch.any(torch.abs(self.torques) > torque_limit, dim=1)
        self.metric_torque_violation_steps += torque_violation.float()
        if len(self.penalised_contact_indices) > 0:
            collision_step = torch.any(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 1., dim=1)
            self.metric_collision_steps += collision_step.float()
        # obstacle metric placeholder (will be activated in Step 4.x)
        self.metric_obstacle_avoid_sum += 0.

    def _update_track_semantics(self):
        if self.track_layout is None or not self.cfg.terrain.track.semantic_enabled:
            return
        local_x = self.base_pos[:, 0] - self.env_origins[:, 0]
        local_y = self.base_pos[:, 1] - self.env_origins[:, 1]
        half_length = 0.5 * self.track_layout["lane_length"]
        in_longitudinal = torch.abs(local_x) <= half_length
        left = self.track_layout["left_boundary"]
        right = self.track_layout["right_boundary"]
        in_lateral = torch.logical_and(local_y >= left, local_y <= right)
        on_track = torch.logical_and(in_longitudinal, in_lateral)

        boundary_tol = max(
            0.5 * self.cfg.terrain.track.boundary_width,
            0.5 * self.cfg.terrain.track.separator_width,
            self.cfg.terrain.track.semantic_boundary_tol,
        )
        boundary_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        for y_boundary in self.track_layout["lane_boundaries"]:
            boundary_mask |= torch.abs(local_y - y_boundary) <= boundary_tol
        boundary_mask &= on_track
        lane_mask = on_track & (~boundary_mask)

        # semantic id: 0 non-track, 1 lane interior, 2 boundary/line
        self.track_semantic_id_buf[:] = 0
        self.track_semantic_id_buf[lane_mask] = 1
        self.track_semantic_id_buf[boundary_mask] = 2

        self.metric_lane_violation_steps += (~on_track).float()
        self.metric_semantic_lane_steps += lane_mask.float()
        self.metric_semantic_boundary_steps += boundary_mask.float()

    def _create_ground_plane(self):
        """ Adds a ground plane to the simulation, sets friction and restitution based on the cfg.
        """
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.cfg.terrain.static_friction
        plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        plane_params.restitution = self.cfg.terrain.restitution
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self):
        """ Creates environments:
             1. loads the robot URDF/MJCF asset,
             2. For each environment
                2.1 creates the environment, 
                2.2 calls DOF and Rigid shape properties callbacks,
                2.3 create actor with these properties and add them to the env
             3. Store indices of different bodies of the robot
        """
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)

        # save body names from the asset
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.num_bodies = len(body_names)
        self.num_dofs = len(self.dof_names)
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])

        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self._get_env_origins()
        self._build_track_layout()
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.actor_handles = []
        self.envs = []
        
        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            env_center = self.env_origins[i].clone()
            pos = env_center.clone()
            if self.track_layout is None:
                pos[:2] += torch_rand_float(-1., 1., (2,1), device=self.device).squeeze(1)
            else:
                x_jitter = min(self.cfg.terrain.track.spawn_x_jitter, 0.25 * self.track_layout["lane_length"])
                y_jitter_limit = max(
                    0.0,
                    0.5 * self.track_layout["lane_width"]
                    - max(self.cfg.terrain.track.boundary_width, self.cfg.terrain.track.separator_width)
                    - self.cfg.terrain.track.spawn_y_margin,
                )
                x_jitter_val = float(torch_rand_float(-x_jitter, x_jitter, (1, 1), device=self.device).item())
                if y_jitter_limit > 0.0:
                    y_jitter_val = float(torch_rand_float(-y_jitter_limit, y_jitter_limit, (1, 1), device=self.device).item())
                else:
                    y_jitter_val = 0.0
                pos[0] += x_jitter_val
                pos[1] += self.env_lane_center_y[i] + y_jitter_val
            start_pose.p = gymapi.Vec3(*pos)
                
            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            actor_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, self.cfg.asset.name, i, self.cfg.asset.self_collisions, 0)
            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)

        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], feet_names[i])

        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], penalized_contact_names[i])

        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], termination_contact_names[i])

    def _build_track_layout(self):
        if not self.cfg.terrain.track.enabled:
            self.track_layout = None
            return
        num_lanes = int(self.cfg.terrain.track.num_lanes)
        if self.cfg.terrain.track.auto_match_num_envs:
            num_lanes = int(self.num_envs)
        lane_width = self.cfg.terrain.track.lane_width
        lane_length = self.cfg.terrain.track.lane_length
        if self.cfg.terrain.track.auto_scale_length_with_grid:
            env_grid_rows = int(self.cfg.terrain.track.env_grid_rows)
            if env_grid_rows > 0:
                env_grid_cols = int(np.ceil(self.num_envs / env_grid_rows))
            else:
                env_grid_cols = int(np.floor(np.sqrt(self.num_envs)))
            base_cols = max(1, int(self.cfg.terrain.track.base_grid_cols))
            lane_length *= max(1.0, float(env_grid_cols) / float(base_cols))
        total_track_width = lane_width * num_lanes
        left_boundary = -0.5 * total_track_width
        lane_centers = [left_boundary + (i + 0.5) * lane_width for i in range(num_lanes)]
        lane_boundaries = [left_boundary + i * lane_width for i in range(num_lanes + 1)]
        lane_offset = int(self.cfg.terrain.track.lane_assignment_offset)
        if self.cfg.terrain.track.auto_match_num_envs:
            lane_offset = num_lanes // 2
        lane_ids = (np.arange(self.num_envs, dtype=np.int64) + lane_offset) % num_lanes
        env_lane_center_y = np.array([lane_centers[idx] for idx in lane_ids], dtype=np.float32)
        self.track_layout = {
            "num_lanes": num_lanes,
            "lane_width": lane_width,
            "lane_length": lane_length,
            "lane_centers": lane_centers,
            "lane_boundaries": lane_boundaries,
            "left_boundary": lane_boundaries[0],
            "right_boundary": lane_boundaries[-1],
            "semantic_ids": {"non_track": 0, "lane": 1, "boundary": 2},
        }
        self.env_lane_ids = torch.from_numpy(lane_ids).to(self.device)
        self.env_lane_center_y = torch.from_numpy(env_lane_center_y).to(self.device)

    def _draw_track_debug_lines(self):
        if self.track_layout is None:
            return
        half_length = 0.5 * self.track_layout["lane_length"]
        z = 0.03
        self.gym.clear_lines(self.viewer)
        draw_env_ids = list(range(len(self.envs))) if self.cfg.terrain.track.visualize_all_env_tracks else [int(self.cfg.viewer.ref_env)]
        for env_id in draw_env_ids:
            if env_id < 0 or env_id >= len(self.envs):
                continue
            env_handle = self.envs[env_id]
            center = self.env_origins[env_id]
            vertices = []
            colors = []
            for i, y_offset in enumerate(self.track_layout["lane_boundaries"]):
                x0 = float(center[0] - half_length)
                x1 = float(center[0] + half_length)
                y = float(center[1] + y_offset)
                vertices.extend([x0, y, z, x1, y, z])
                if i == 0 or i == len(self.track_layout["lane_boundaries"]) - 1:
                    colors.extend([1.0, 1.0, 1.0])
                else:
                    colors.extend([0.95, 0.95, 0.2])
            self.gym.add_lines(
                self.viewer,
                env_handle,
                len(self.track_layout["lane_boundaries"]),
                np.array(vertices, dtype=np.float32),
                np.array(colors, dtype=np.float32),
            )

    def _get_env_origins(self):
        """ Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
            Otherwise create a grid.
        """
      
        self.custom_origins = False
        self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
        # create a grid of robots
        num_cols = np.floor(np.sqrt(self.num_envs))
        num_rows = np.ceil(self.num_envs / num_cols)
        xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
        spacing = self.cfg.env.env_spacing
        self.env_origins[:, 0] = spacing * xx.flatten()[:self.num_envs]
        self.env_origins[:, 1] = spacing * yy.flatten()[:self.num_envs]
        self.env_origins[:, 2] = 0.

    def _parse_cfg(self, cfg):
        self.dt = self.cfg.control.decimation * self.sim_params.dt
        self.obs_scales = self.cfg.normalization.obs_scales
        self.reward_scales = class_to_dict(self.cfg.rewards.scales)
        self.command_ranges = class_to_dict(self.cfg.commands.ranges)
     

        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)

        self.cfg.domain_rand.push_interval = np.ceil(self.cfg.domain_rand.push_interval_s / self.dt)


    #------------ reward functions----------------
    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])
    
    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)
    
    def _reward_orientation(self):
        # Penalize non flat base orientation
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    def _reward_base_height(self):
        # Penalize base height away from target
        base_height = self.root_states[:, 2]
        return torch.square(base_height - self.cfg.rewards.base_height_target)
    
    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.dof_vel), dim=1)
    
    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)
    
    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)
    
    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return torch.sum(1.*(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)
    
    def _reward_termination(self):
        # Terminal reward / penalty
        return self.reset_buf * ~(self.time_out_buf | self.finish_termination_buf)
    
    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.) # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    def _reward_dof_vel_limits(self):
        # Penalize dof velocities too close to the limit
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        return torch.sum((torch.abs(self.dof_vel) - self.dof_vel_limits*self.cfg.rewards.soft_dof_vel_limit).clip(min=0., max=1.), dim=1)

    def _reward_torque_limits(self):
        # penalize torques too close to the limit
        return torch.sum((torch.abs(self.torques) - self.torque_limits*self.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1)

    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error/self.cfg.rewards.tracking_sigma)
    
    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw) 
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error/self.cfg.rewards.tracking_sigma)

    def _reward_feet_air_time(self):
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts) 
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += self.dt
        rew_airTime = torch.sum((self.feet_air_time - 0.5) * first_contact, dim=1) # reward only on first contact with the ground
        rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1 #no reward for zero command
        self.feet_air_time *= ~contact_filt
        return rew_airTime
    
    def _reward_stumble(self):
        # Penalize feet hitting vertical surfaces
        return torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) >\
             5 *torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1)
        
    def _reward_stand_still(self):
        # Penalize motion at zero commands
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1) * (torch.norm(self.commands[:, :2], dim=1) < 0.1)

    def _reward_feet_contact_forces(self):
        # penalize high contact forces
        return torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) -  self.cfg.rewards.max_contact_force).clip(min=0.), dim=1)

    def _reward_lane_centering(self):
        """Oracle lane-keeping reward used in Step 2.1 baseline.

        Encourages the base to stay close to the assigned lane centerline.
        """
        if self.track_layout is None:
            return torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        local_y = self.base_pos[:, 1] - self.env_origins[:, 1]
        center_error = local_y - self.env_lane_center_y
        lane_sigma = max(0.1, 0.5 * float(self.track_layout["lane_width"]))
        return torch.exp(-(center_error ** 2) / (lane_sigma ** 2))

    def _reward_lane_offset(self):
        """Penalize lateral offset from the assigned lane centerline."""
        if self.track_layout is None:
            return torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        local_y = self.base_pos[:, 1] - self.env_origins[:, 1]
        center_error = local_y - self.env_lane_center_y
        return center_error ** 2

    def _reward_heading_alignment(self):
        """Reward facing toward track direction (+x in world/env frame)."""
        forward = quat_apply(self.base_quat, self.forward_vec)
        return torch.clamp(forward[:, 0], min=0.0)

    def _reward_heading_error(self):
        """Penalize heading error away from the track forward direction."""
        forward = quat_apply(self.base_quat, self.forward_vec)
        return torch.square(1.0 - torch.clamp(forward[:, 0], min=-1.0, max=1.0))

    def _reward_yaw_rate(self):
        """Penalize turning rate so the robot keeps moving straight."""
        return torch.square(self.base_ang_vel[:, 2])

    def _reward_lateral_velocity(self):
        """Penalize sideways drift across the lane."""
        local_vy = self.root_states[:, 8]
        return torch.square(local_vy)

    def _reward_track_progress(self):
        """Reward actual forward displacement along the track rather than body-frame sway."""
        local_x = self.base_pos[:, 0] - self.env_origins[:, 0]
        progress_speed = (local_x - self.last_track_local_x) / self.dt
        max_progress_speed = max(0.5, float(self.command_ranges["lin_vel_x"][1]) * 2.0)
        return torch.clamp(progress_speed, min=0.0, max=max_progress_speed)

    def _reward_stalling(self):
        """Penalize failing to convert the forward command into actual forward motion."""
        local_x = self.base_pos[:, 0] - self.env_origins[:, 0]
        progress_speed = (local_x - self.last_track_local_x) / self.dt
        commanded_speed = torch.clamp(self.commands[:, 0], min=0.0)
        return torch.clamp(commanded_speed - progress_speed, min=0.0)

    def _reward_finish_bonus(self):
        """Give a one-time bonus only when the robot safely reaches the lane end."""
        return self.finish_termination_buf.float()
