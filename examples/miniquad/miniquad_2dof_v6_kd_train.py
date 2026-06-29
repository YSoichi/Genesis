"""
miniquad_2dof_v6_kd_train.py
=============================
2DOF-hip 4脚ロボット v6: ダンピング低減による速度向上

【v5 → v6 の変更点: kd 激減が速度ボトルネック】

  問題分析 (v5):
    - femur 回転角速度: 0.80 rad / 0.5s = 1.6 rad/s (1Hz trot, amp=0.40)
    - kd=0.2 のダンピングトルク: 0.2 × 1.6 = 0.32 N⋅m
    - 足が地面を押す推進力 (体重 0.2kg → 0.025 N⋅m) の 13× がダンピングに消費
    - → 推進力の大半がダンピングで失われ、速度が上がらない

  改善: kd 0.2 → 0.05 (4分の1)
    - ダンピングトルク: 0.32 → 0.08 N⋅m → 推進力の 3× に縮小
    - より多くの推進力が実際の体の前進に使われる
    - リスク: kd が低すぎると振動が生じる可能性
    - → kp=50 を維持することで位置精度を保ちつつダンピングを減らす

  その他変更:
    - episode_length 100s → 30s (安定性確認のため短縮)
    - femur_amp: 0.40 → 0.35 rad (安定性のために少し戻す)
    - v4@2800 から転移学習 (安定性が最良)

【実行方法】
  cd examples/miniquad
  PYTHONPATH=/home/mutsumi/rsl_rl:$PYTHONPATH python3 miniquad_2dof_v6_kd_train.py \\
    -e miniquad-2dof-v6-kd -B 16384 --max_iterations 3001 \\
    --load_run miniquad-2dof-v4-terrain --load_checkpoint 2800 \\
    > /tmp/miniquad_v6_kd_train.log 2>&1 &
"""
import argparse
import os
import pickle
import shutil
import glob
import math

import genesis as gs
import torch
from rsl_rl.runners import OnPolicyRunner
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat


def gs_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower


class miniquad2DOFV6KdEnv:
    """v6: kd=0.05 低ダンピングで推進効率を高める。"""

    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg,
                 show_viewer=False, device="cuda"):
        self.device = torch.device(device)
        self.num_envs     = num_envs
        self.num_obs      = obs_cfg["num_obs"]
        self.num_privileged_obs = None
        self.num_actions  = env_cfg["num_actions"]
        self.num_commands = command_cfg["num_commands"]

        self.simulate_action_latency = True
        self.dt = 0.02
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)

        self.env_cfg      = env_cfg
        self.obs_cfg      = obs_cfg
        self.reward_cfg   = reward_cfg
        self.command_cfg  = command_cfg
        self.obs_scales   = obs_cfg["obs_scales"]
        self.reward_scales= reward_cfg["reward_scales"]
        self.push_interval= int(env_cfg.get("push_interval_s", 5.0) / self.dt)

        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=2),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=int(0.5 / self.dt),
                camera_pos=(5.0, -2.0, 2.0),
                camera_lookat=(5.0, 2.0, 0.0),
                camera_fov=40,
            ),
            vis_options=gs.options.VisOptions(n_rendered_envs=1),
            rigid_options=gs.options.RigidOptions(
                dt=self.dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
            ),
            show_viewer=show_viewer,
        )

        # 平坦地形 16m × 4m
        self.terrain = self.scene.add_entity(
            morph=gs.morphs.Terrain(
                n_subterrains=(8, 2),
                subterrain_size=(2.0, 2.0),
                horizontal_scale=0.05,
                vertical_scale=0.005,
                subterrain_types=[["flat_terrain", "flat_terrain"]] * 8,
                randomize=False,
                name="miniquad_flat_v6",
            ),
        )

        self.base_init_pos  = torch.tensor(env_cfg["base_init_pos"],  device=self.device)
        self.base_init_quat = torch.tensor(env_cfg["base_init_quat"], device=self.device)
        self.inv_base_init_quat = inv_quat(self.base_init_quat)

        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file="miniquad_2dof.urdf",
                pos=self.base_init_pos.cpu().numpy(),
                quat=self.base_init_quat.cpu().numpy(),
                recompute_inertia=True,
            ),
        )
        self.scene.build(n_envs=num_envs)

        self.motor_dofs = [self.robot.get_joint(name).dof_idx_local
                           for name in self.env_cfg["dof_names"]]
        self.num_links  = len(self.robot.links)

        self.robot.set_dofs_kp([env_cfg["kp"]] * self.num_actions, self.motor_dofs)
        self.robot.set_dofs_kv([env_cfg["kd"]] * self.num_actions, self.motor_dofs)

        self.reward_functions, self.episode_sums = dict(), dict()
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros((num_envs,), device=self.device, dtype=gs.tc_float)

        self.base_lin_vel       = torch.zeros((num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.base_lin_vel_world = torch.zeros((num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.base_ang_vel       = torch.zeros((num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.projected_gravity  = torch.zeros((num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.global_gravity = torch.tensor([0.0, 0.0, -1.0], device=self.device, dtype=gs.tc_float).repeat(num_envs, 1)

        self.obs_buf    = torch.zeros((num_envs, self.num_obs), device=self.device, dtype=gs.tc_float)
        self.rew_buf    = torch.zeros((num_envs,),              device=self.device, dtype=gs.tc_float)
        self.reset_buf  = torch.ones ((num_envs,),              device=self.device, dtype=gs.tc_int)
        self.episode_length_buf = torch.zeros((num_envs,),      device=self.device, dtype=gs.tc_int)

        self.commands = torch.zeros((num_envs, self.num_commands), device=self.device, dtype=gs.tc_float)
        self.commands_scale = torch.tensor(
            [self.obs_scales["lin_vel"], self.obs_scales["lin_vel"], self.obs_scales["ang_vel"]],
            device=self.device, dtype=gs.tc_float,
        )

        self.actions      = torch.zeros((num_envs, self.num_actions), device=self.device, dtype=gs.tc_float)
        self.last_actions = torch.zeros_like(self.actions)
        self.dof_pos      = torch.zeros_like(self.actions)
        self.dof_vel      = torch.zeros_like(self.actions)
        self.last_dof_vel = torch.zeros_like(self.actions)
        self.base_pos     = torch.zeros((num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.base_quat    = torch.zeros((num_envs, 4), device=self.device, dtype=gs.tc_float)

        self.default_dof_pos = torch.tensor(
            [env_cfg["default_joint_angles"][name] for name in env_cfg["dof_names"]],
            device=self.device, dtype=gs.tc_float,
        )
        self.trot_phase = torch.zeros(num_envs, device=self.device, dtype=gs.tc_float)
        self.extras = {"observations": {}}

    def _get_trot_bias(self):
        freq  = self.env_cfg.get("trot_bias_freq",      1.0)
        c_amp = self.env_cfg.get("trot_bias_coxa_amp",  0.25)
        f_amp = self.env_cfg.get("trot_bias_femur_amp", 0.35)

        t = self.episode_length_buf.float() * self.dt
        phase_a = torch.remainder(freq * t, 1.0)
        phase_b = torch.remainder(freq * t + 0.5, 1.0)

        def gait_pattern(p):
            stance = (p < 0.5).float()
            swing  = 1.0 - stance
            s = torch.clamp(p / 0.5, 0.0, 1.0)
            femur_stance = f_amp * (1.0 - 2.0 * s)
            w = torch.clamp((p - 0.5) / 0.5, 0.0, 1.0)
            femur_swing  = f_amp * (2.0 * w - 1.0)
            coxa_swing   = c_amp * torch.sin(math.pi * w)
            femur = stance * femur_stance + swing * femur_swing
            coxa  = swing * coxa_swing
            return coxa, femur

        c_a, f_a = gait_pattern(phase_a)
        c_b, f_b = gait_pattern(phase_b)
        return torch.stack([c_a, f_a, c_b, f_b, c_b, f_b, c_a, f_a], dim=-1)

    def _apply_push(self, push_envs_idx):
        if len(push_envs_idx) == 0:
            return
        noise_dof_vel = self.dof_vel[push_envs_idx] + gs_rand_float(
            -0.5, 0.5, (len(push_envs_idx), self.num_actions), self.device
        )
        self.robot.set_dofs_velocity(
            velocity=noise_dof_vel,
            dofs_idx_local=self.motor_dofs,
            envs_idx=push_envs_idx,
        )

    def _resample_commands(self, envs_idx):
        self.commands[envs_idx, 0] = gs_rand_float(*self.command_cfg["lin_vel_x_range"], (len(envs_idx),), self.device)
        self.commands[envs_idx, 1] = gs_rand_float(*self.command_cfg["lin_vel_y_range"], (len(envs_idx),), self.device)
        self.commands[envs_idx, 2] = gs_rand_float(*self.command_cfg["ang_vel_range"],   (len(envs_idx),), self.device)

    def step(self, actions):
        self.actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])
        exec_actions = self.last_actions if self.simulate_action_latency else self.actions

        trot_bias = self._get_trot_bias()
        target_dof_pos = exec_actions * self.env_cfg["action_scale"] + self.default_dof_pos + trot_bias
        self.robot.control_dofs_position(target_dof_pos, self.motor_dofs)
        self.scene.step()

        self.episode_length_buf += 1
        if self.push_interval > 0:
            push_envs = (self.episode_length_buf % self.push_interval == 0).nonzero(as_tuple=False).flatten()
            self._apply_push(push_envs)

        self.base_pos[:]  = self.robot.get_pos()
        self.base_quat[:] = self.robot.get_quat()
        self.base_lin_vel_world[:] = self.robot.get_vel()
        self.base_euler = quat_to_xyz(
            transform_quat_by_quat(torch.ones_like(self.base_quat) * self.inv_base_init_quat, self.base_quat)
        )
        inv_base_quat = inv_quat(self.base_quat)
        self.base_lin_vel[:]   = transform_by_quat(self.robot.get_vel(), inv_base_quat)
        self.base_ang_vel[:]   = transform_by_quat(self.robot.get_ang(), inv_base_quat)
        self.projected_gravity = transform_by_quat(self.global_gravity, inv_base_quat)
        self.dof_pos[:] = self.robot.get_dofs_position(self.motor_dofs)
        self.dof_vel[:] = self.robot.get_dofs_velocity(self.motor_dofs)

        resample_idx = (
            (self.episode_length_buf % int(self.env_cfg["resampling_time_s"] / self.dt) == 0)
            .nonzero(as_tuple=False).flatten()
        )
        self._resample_commands(resample_idx)

        self.reset_buf  = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= torch.abs(self.base_euler[:, 1]) > self.env_cfg["termination_if_pitch_greater_than"]
        self.reset_buf |= torch.abs(self.base_euler[:, 0]) > self.env_cfg["termination_if_roll_greater_than"]
        self.reset_buf |= self.base_pos[:, 2] < -0.10
        self.reset_buf |= self.base_pos[:, 0] > 15.0

        time_out_idx = (self.episode_length_buf > self.max_episode_length).nonzero(as_tuple=False).flatten()
        self.extras["time_outs"] = torch.zeros_like(self.reset_buf, device=self.device, dtype=gs.tc_float)
        self.extras["time_outs"][time_out_idx] = 1.0
        self.reset_idx(self.reset_buf.nonzero(as_tuple=False).flatten())

        self.rew_buf[:] = 0.0
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew

        yaw = self.base_euler[:, 2:3]
        heading_obs = torch.cat([torch.cos(yaw), torch.sin(yaw)], dim=-1)
        self.obs_buf = torch.cat([
            self.base_ang_vel * self.obs_scales["ang_vel"],
            self.projected_gravity,
            self.commands * self.commands_scale,
            self.base_lin_vel * self.obs_scales["lin_vel"],
            (self.dof_pos - self.default_dof_pos) * self.obs_scales["dof_pos"],
            self.dof_vel * self.obs_scales["dof_vel"],
            self.actions,
            heading_obs,
        ], axis=-1)

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:]  = self.dof_vel[:]
        self.extras["observations"] = {}
        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    def get_observations(self):
        return self.obs_buf, self.extras

    def get_privileged_observations(self):
        return None

    def reset_idx(self, envs_idx):
        if len(envs_idx) == 0:
            return
        self.dof_pos[envs_idx] = self.default_dof_pos
        self.dof_vel[envs_idx] = 0.0
        self.robot.set_dofs_position(
            position=self.dof_pos[envs_idx],
            dofs_idx_local=self.motor_dofs,
            zero_velocity=True,
            envs_idx=envs_idx,
        )
        if self.env_cfg.get("randomize_friction", True):
            self.robot.set_friction_ratio(
                friction_ratio=gs_rand_float(0.5, 1.5, (len(envs_idx), self.num_links), self.device),
                links_idx_local=None,
                envs_idx=envs_idx,
            )
        self.base_pos[envs_idx]  = self.base_init_pos
        self.base_quat[envs_idx] = self.base_init_quat.reshape(1, -1)
        self.robot.set_pos(self.base_pos[envs_idx],  zero_velocity=False, envs_idx=envs_idx)
        self.robot.set_quat(self.base_quat[envs_idx], zero_velocity=False, envs_idx=envs_idx)
        self.base_lin_vel[envs_idx]  = 0
        self.base_ang_vel[envs_idx]  = 0
        self.robot.zero_all_dofs_velocity(envs_idx)
        self.last_actions[envs_idx]       = 0.0
        self.last_dof_vel[envs_idx]       = 0.0
        self.episode_length_buf[envs_idx] = 0
        self.reset_buf[envs_idx]          = True
        self.trot_phase[envs_idx] = torch.rand(len(envs_idx), device=self.device)

        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][envs_idx]).item() / self.env_cfg["episode_length_s"]
            )
            self.episode_sums[key][envs_idx] = 0.0
        self._resample_commands(envs_idx)

    def reset(self):
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        return self.obs_buf, None

    def _reward_tracking_lin_vel(self):
        lin_vel_error = torch.sum(
            torch.square(self.commands[:, :2] - self.base_lin_vel_world[:, :2]), dim=1
        )
        return torch.exp(-lin_vel_error / self.reward_cfg["tracking_sigma"])

    def _reward_forward_vel(self):
        vx = self.base_lin_vel_world[:, 0]
        return torch.clamp(vx, 0.0, 0.5)

    def _reward_alive(self):
        return torch.ones(self.num_envs, device=self.device, dtype=gs.tc_float)

    def _reward_lin_vel_z(self):
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_base_height(self):
        return torch.square(self.base_pos[:, 2] - self.reward_cfg["base_height_target"])

    def _reward_action_rate(self):
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_heading(self):
        return torch.square(self.base_euler[:, 2])

    def _reward_orientation(self):
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)


def get_train_cfg():
    return {
        "num_steps_per_env": 32,
        "save_interval": 100,
        "empirical_normalization": False,
        "algorithm": {
            "class_name": "PPO",
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": 0.0,
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 0.001,
            "max_grad_norm": 1.0,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
        },
        "policy": {
            "class_name": "ActorCritic",
            "activation": "elu",
            "actor_hidden_dims": [256, 128, 64],
            "critic_hidden_dims": [256, 128, 64],
            "init_noise_std": 1.0,
        },
        "seed": 1,
        "runner_class_name": "OnPolicyRunner",
    }


def get_cfgs():
    """v6: kd=0.05 低ダンピング設定。"""
    env_cfg = {
        "num_actions": 8,
        "default_joint_angles": {
            "body_to_coxa_fr_j":      0.0,
            "coxa_fr_to_femur_fr_j":  0.0,
            "body_to_coxa_fl_j":      0.0,
            "coxa_fl_to_femur_fl_j":  0.0,
            "body_to_coxa_rr_j":      0.0,
            "coxa_rr_to_femur_rr_j":  0.0,
            "body_to_coxa_rl_j":      0.0,
            "coxa_rl_to_femur_rl_j":  0.0,
        },
        "dof_names": [
            "body_to_coxa_fr_j",     "coxa_fr_to_femur_fr_j",
            "body_to_coxa_fl_j",     "coxa_fl_to_femur_fl_j",
            "body_to_coxa_rr_j",     "coxa_rr_to_femur_rr_j",
            "body_to_coxa_rl_j",     "coxa_rl_to_femur_rl_j",
        ],
        # v6 核心: ダンピングを 0.2 → 0.05 に激減
        "kp": 50.0,
        "kd":  0.05,   # ← 低ダンピングで推進効率を高める

        "termination_if_roll_greater_than":  1.2,
        "termination_if_pitch_greater_than": 1.2,
        "base_init_pos":  [0.5, 2.0, 0.082],
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        "episode_length_s":   30.0,   # 安定確認のため短めに
        "resampling_time_s":   5.0,
        "action_scale":        0.5,
        "simulate_action_latency": True,
        "clip_actions":       10.0,
        "use_trot_bias":       True,
        "trot_bias_freq":      1.0,
        "trot_bias_coxa_amp":  0.25,
        "trot_bias_femur_amp": 0.35,  # v2 と v5 の中間
        "randomize_friction": True,
        "push_interval_s": 5.0,
        "push_vel_max":    0.3,
    }

    obs_cfg = {
        "num_obs": 38,
        "obs_scales": {
            "lin_vel": 2.0,
            "ang_vel": 0.25,
            "dof_pos": 1.0,
            "dof_vel": 0.05,
        },
    }

    reward_cfg = {
        "tracking_sigma":    0.05,
        "base_height_target": 0.082,
        "reward_scales": {
            "forward_vel":       8.0,
            "tracking_lin_vel":  2.0,
            "alive":             1.0,
            "lin_vel_z":        -0.3,
            "base_height":      -0.3,
            "action_rate":      -0.001,
            "heading":          -0.1,
            "orientation":      -0.2,
        },
    }

    command_cfg = {
        "num_commands": 3,
        "lin_vel_x_range": [0.10, 0.20],
        "lin_vel_y_range": [0, 0],
        "ang_vel_range":   [0, 0],
    }

    return env_cfg, obs_cfg, reward_cfg, command_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name",     type=str, default="miniquad-2dof-v6-kd")
    parser.add_argument("-B", "--num_envs",      type=int, default=16384)
    parser.add_argument("--max_iterations",      type=int, default=3001)
    parser.add_argument("--load_run",            type=str, default=None)
    parser.add_argument("--load_checkpoint",     type=int, default=-1)
    args = parser.parse_args()

    gs.init(logging_level="warning")

    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    train_cfg = get_train_cfg()

    if args.load_run is None:
        if os.path.exists(log_dir):
            shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    env = miniquad2DOFV6KdEnv(
        num_envs=args.num_envs,
        env_cfg=env_cfg, obs_cfg=obs_cfg,
        reward_cfg=reward_cfg, command_cfg=command_cfg,
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device="cuda:0")
    pickle.dump(
        [env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg],
        open(f"{log_dir}/cfgs.pkl", "wb"),
    )

    if args.load_run is not None:
        load_path = f"logs/{args.load_run}"
        if args.load_checkpoint == -1:
            ckpts = sorted(glob.glob(f"{load_path}/model_*.pt"),
                           key=lambda p: int(p.split("model_")[-1].replace(".pt", "")))
            resume_path = ckpts[-1] if ckpts else None
        else:
            resume_path = f"{load_path}/model_{args.load_checkpoint}.pt"

        if resume_path and os.path.exists(resume_path):
            print(f"[INFO] 継続学習: {resume_path} からロード")
            runner.load(resume_path)

    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)


if __name__ == "__main__":
    main()
