from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class G1RoughCfg( LeggedRobotCfg ):
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.82] # x,y,z [m] - Safe spawn height
        default_joint_angles = { # = target angles [rad] when action = 0.0
           'left_hip_yaw_joint' : 0. ,   
           'left_hip_roll_joint' : 0,               
           'left_hip_pitch_joint' : -0.1,         
           'left_knee_joint' : 0.3,       
           'left_ankle_pitch_joint' : -0.2,     
           'left_ankle_roll_joint' : 0,     
           'right_hip_yaw_joint' : 0., 
           'right_hip_roll_joint' : 0, 
           'right_hip_pitch_joint' : -0.1,                                       
           'right_knee_joint' : 0.3,                                             
           'right_ankle_pitch_joint': -0.2,                              
           'right_ankle_roll_joint' : 0,       
           'torso_joint' : 0.
        }
    
    class env(LeggedRobotCfg.env):
        num_envs = 64 # Training batch size
        num_observations = 47
        num_privileged_obs = 50 # 47 + 3 (lin_vel)
        num_actions = 12

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'trimesh'
        selected = True
        terrain_kwargs = {'type': 'track_terrain', 'terrain_kwargs': {'track_width': 10.0, 'num_lanes': 8, 'lane_width': 1.22}}
        curriculum = False
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.
        terrain_length = 20.
        terrain_width = 12.
        num_rows = 10
        num_cols = 10

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.5, 1.25]
        randomize_base_mass = True
        added_mass_range = [-1., 3.]
        push_robots = True # Enable pushing for training robustness
        push_interval_s = 5
        max_push_vel_xy = 1.5

    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        stiffness = {'hip_yaw': 100,
                     'hip_roll': 100,
                     'hip_pitch': 100,
                     'knee': 150,
                     'ankle': 40,
                     }  # [N*m/rad]
        damping = {  'hip_yaw': 2,
                     'hip_roll': 2,
                     'hip_pitch': 2,
                     'knee': 4,
                     'ankle': 2,
                     }  # [N*m/rad]  # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/g1_description/g1_12dof.urdf'
        name = "g1"
        foot_name = "ankle_roll"
        penalize_contacts_on = ["hip", "knee"]
        terminate_after_contacts_on = ["pelvis"]
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter
        collapse_fixed_joints = False # G1 might need fixed joints preserved for structure
        flip_visual_attachments = False # Try disabling this if meshes are rotated wrong
        fix_base_link = False
        disable_gravity = False
  
    class rewards(LeggedRobotCfg.rewards):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.78
        class scales(LeggedRobotCfg.rewards.scales):
            tracking_lin_vel = 1.5
            tracking_ang_vel = 0.5
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = -1.0
            base_height = -2.0
            dof_acc = -2.5e-7
            dof_vel = -2.5e-4
            feet_air_time = 1.0
            collision = -1.0
            action_rate = -0.01

class G1RoughCfgPPO(LeggedRobotCfgPPO):
    class policy:
        init_noise_std = 0.5
        actor_hidden_dims = [32]
        critic_hidden_dims = [32]
        activation = 'elu'
        # only for 'ActorCriticRecurrent':
        rnn_type = 'lstm'
        rnn_hidden_size = 64
        rnn_num_layers = 1
        
    class runner(LeggedRobotCfgPPO.runner):
        run_name = ''
        experiment_name = 'g1'
        policy_class_name = "ActorCriticRecurrent"
        max_iterations = 10000
    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.001

class G1SprintTrackCfg(G1RoughCfg):
    class env(G1RoughCfg.env):
        # Keep each env visually separated in the global viewer.
        env_spacing = 14.0
        episode_length_s = 30.0
        num_observations = 52
        num_privileged_obs = 55

    class viewer(G1RoughCfg.viewer):
        ref_env = 0
        pos = [-2.5, -1.5, 1.8]
        lookat = [1.5, 0.0, 0.8]

    class terrain(G1RoughCfg.terrain):
        mesh_type = "plane"
        curriculum = False
        class track(G1RoughCfg.terrain.track):
            enabled = True
            visualize_all_env_tracks = True
            num_lanes = 1
            lane_width = 1.25
            lane_length = 12.0
            auto_match_num_envs = False
            auto_scale_length_with_grid = False
            env_grid_rows = 2
            base_grid_cols = 6
            boundary_width = 0.08
            separator_width = 0.04
            curb_height = 0.025
            lane_mark_height = 0.006
            oracle_lane_obs = True
            spawn_x_jitter = 0.15
            spawn_y_margin = 0.45
            terminate_on_out_of_track = True
            out_of_track_margin = 0.0
            success_on_reach_lane_end = True
            lane_end_success_margin = 0.20

    class commands(G1RoughCfg.commands):
        heading_command = False
        resampling_time = 10.0
        class ranges(G1RoughCfg.commands.ranges):
            lin_vel_x = [0.82, 0.98]
            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [0.0, 0.0]
            heading = [0.0, 0.0]

    class domain_rand(G1RoughCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.1, 1.25]
        randomize_base_mass = True
        added_mass_range = [-1., 3.]
        push_robots = False
        push_interval_s = 5
        max_push_vel_xy = 1.5

    class rewards(G1RoughCfg.rewards):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.78
        class scales(G1RoughCfg.rewards.scales):
            tracking_lin_vel = 1.8
            tracking_ang_vel = 1.2
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = -1.0
            base_height = -10.0
            dof_acc = -2.5e-7
            dof_vel = -1e-3
            feet_air_time = 0.0
            collision = -0.2
            action_rate = -0.01
            dof_pos_limits = -5.0
            lane_centering = 0.25
            lane_offset = -4.0
            lane_divergence = -8.0
            heading_alignment = 1.3
            heading_error = -1.2
            yaw_rate = -1.0
            lateral_velocity = -1.0
            track_progress = 7.2
            stalling = -7.0
            overspeed = -22.0
            finish_bonus = 90.0
            alive = 0.03
            hip_pos = -1.0
            contact_no_vel = -0.2
            feet_swing_height = -20.0
            contact = 0.18

class G1SprintTrackCfgPPO(G1RoughCfgPPO):
    class policy(G1RoughCfgPPO.policy):
        init_noise_std = 0.3

    class algorithm(G1RoughCfgPPO.algorithm):
        entropy_coef = 0.0005
        learning_rate = 1.0e-4
        desired_kl = 0.005

    class runner(G1RoughCfgPPO.runner):
        experiment_name = "g1_sprint_track"
        policy_class_name = "ActorCriticRecurrent"

  
