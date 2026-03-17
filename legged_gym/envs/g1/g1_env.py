
from legged_gym.envs.base.legged_robot import LeggedRobot

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil
import torch

class G1Robot(LeggedRobot):
    
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
        noise_vec[:3] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[3:6] = noise_scales.gravity * noise_level
        noise_vec[6:9] = 0. # commands
        noise_vec[9:9+self.num_actions] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[9+self.num_actions:9+2*self.num_actions] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[9+2*self.num_actions:9+3*self.num_actions] = 0. # previous actions
        noise_vec[9+3*self.num_actions:9+3*self.num_actions+2] = 0. # sin/cos phase
        
        return noise_vec

    def _init_foot(self):
        self.feet_num = len(self.feet_indices)
        
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state)
        self.rigid_body_states_view = self.rigid_body_states.view(self.num_envs, -1, 13)
        self.feet_state = self.rigid_body_states_view[:, self.feet_indices, :]
        self.feet_pos = self.feet_state[:, :, :3]
        self.feet_vel = self.feet_state[:, :, 7:10]
        
    def _init_buffers(self):
        super()._init_buffers()
        self._init_foot()

    def update_feet_state(self):
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        
        self.feet_state = self.rigid_body_states_view[:, self.feet_indices, :]
        self.feet_pos = self.feet_state[:, :, :3]
        self.feet_vel = self.feet_state[:, :, 7:10]
        
    def _post_physics_step_callback(self):
        self.update_feet_state()

        period = 0.8
        offset = 0.5
        self.phase = (self.episode_length_buf * self.dt) % period / period
        self.phase_left = self.phase
        self.phase_right = (self.phase + offset) % 1
        self.leg_phase = torch.cat([self.phase_left.unsqueeze(1), self.phase_right.unsqueeze(1)], dim=-1)
        
        return super()._post_physics_step_callback()
    
    
    def compute_observations(self):
        """ Computes observations
        """
        sin_phase = torch.sin(2 * np.pi * self.phase ).unsqueeze(1)
        cos_phase = torch.cos(2 * np.pi * self.phase ).unsqueeze(1)
        
        # Proprioception
        proprio_obs = torch.cat((  self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    self.commands[:, :3] * self.commands_scale,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions,
                                    sin_phase,
                                    cos_phase
                                    ),dim=-1)
                                    
        proprio_privileged = torch.cat((  self.base_lin_vel * self.obs_scales.lin_vel,
                                    self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    self.commands[:, :3] * self.commands_scale,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions,
                                    sin_phase,
                                    cos_phase
                                    ),dim=-1)

        # Add Noise to Proprio
        if self.add_noise:
            # We need to apply noise only to proprio part.
            # However, self.noise_scale_vec (from _get_noise_scale_vec) might be full size (27695) or proprio size?
            # _get_noise_scale_vec uses self.obs_buf[0] size.
            # If obs_buf is 27695, then noise_vec is 27695.
            # But the first "proprio_dim" elements are set.
            # So if we add noise_vec, the 0s in visual part will ensure no noise on vision.
            # BUT we haven't concatenated yet.
            # So we should construct full obs first.
            pass

        # Camera Observations
        visual_obs = None
        if hasattr(self, 'camera_handles') and len(self.camera_handles) > 0:
             visual_obs_list = []
             for i in range(self.num_envs):
                 # GPU tensor access failed, switching to CPU for robustness
                 # cam_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, self.envs[i], self.camera_handles[i], gymapi.IMAGE_COLOR)
                 # torch_cam = gymtorch.wrap_tensor(cam_tensor)
                 
                 image = self.gym.get_camera_image(self.sim, self.envs[i], self.camera_handles[i], gymapi.IMAGE_COLOR)
                 if image.shape[0] == 0:
                     # Rendering failed or not ready
                     if i == 0 and self.common_step_counter % 100 == 0 and not hasattr(self, 'camera_warning_printed'):
                         print("Warning: Camera image empty. Rendering might be disabled or failing. (Suppressing further warnings)")
                         self.camera_warning_printed = True
                     image = np.zeros((self.cfg.sensor.camera.height, self.cfg.sensor.camera.width, 4), dtype=np.uint8)
                 else:
                     image = image.reshape(self.cfg.sensor.camera.height, self.cfg.sensor.camera.width, 4)
                     
                 torch_cam = torch.from_numpy(image).to(self.device).float() / 255.0
                 
                 visual_obs_list.append(torch_cam[:, :, :3].reshape(-1))
             
             visual_obs = torch.stack(visual_obs_list)
        
        # Combine
        # If no visual_obs, we might crash if config expects it. But let's assume we have it.
        if visual_obs is not None:
            self.obs_buf = torch.cat((proprio_obs, visual_obs), dim=-1)
            self.privileged_obs_buf = torch.cat((proprio_privileged, visual_obs), dim=-1)
        else:
            self.obs_buf = proprio_obs
            self.privileged_obs_buf = proprio_privileged

        # Add Noise
        if self.add_noise:
            # noise_scale_vec should match obs_buf size
            if self.noise_scale_vec is None or self.noise_scale_vec.shape != self.obs_buf[0].shape:
                 self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)
            
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

        
    def _reward_tracking_lin_vel(self):
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error / 0.25)

    def _reward_contact(self):
        res = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        for i in range(self.feet_num):
            is_stance = self.leg_phase[:, i] < 0.55
            contact = self.contact_forces[:, self.feet_indices[i], 2] > 1
            res += ~(contact ^ is_stance)
        return res
    
    def _reward_feet_swing_height(self):
        contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1.
        pos_error = torch.square(self.feet_pos[:, :, 2] - 0.08) * ~contact
        return torch.sum(pos_error, dim=(1))
    
    def _reward_alive(self):
        # Reward for staying alive
        return 1.0
    
    def _reward_contact_no_vel(self):
        # Penalize contact with no velocity
        contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1.
        contact_feet_vel = self.feet_vel * contact.unsqueeze(-1)
        penalize = torch.square(contact_feet_vel[:, :, :3])
        return torch.sum(penalize, dim=(1,2))
    
    def _reward_hip_pos(self):
        return torch.sum(torch.square(self.dof_pos[:,[1,2,7,8]]), dim=1)
    