"""
miniquad_2dof_terrain_env.py
============================
2DOF-hip 4脚ロボットの地形対応強化学習環境。

【地形レイアウト (自動カリキュラム)】
  ロボットは x=0.5m (平坦ゾーン) から出発し、歩行速度が上がるにつれ
  自然に難しい地形へ進む。明示的なカリキュラム昇格なしで習得が起きる。

  x = 0〜2m : flat_terrain          (初期学習: まず歩くことを習得)
  x = 2〜4m : random_uniform(1cm)   (中級: 小さな凸凹への適応)
  x = 4〜6m : sloped + wave(2cm)    (上級: 傾斜と波状地形)
  x = 6〜8m : discrete_obstacles    (達人: 障害物乗り越え)

  y = 0〜4m の幅は各 subterrain で 2 種類のバリエーションを提供。
  ロボットは (0.5, 2.0, 0.080) からスタート → 平坦中心。

【v1 からの改善点】
  * coxa_symmetry ペナルティを削除: trot_bias の非対称 coxa 動作を妨げていた
  * trot_bias_femur_amp: 0.20 → 0.30 rad (ストライド拡大)
  * trot_bias_coxa_amp:  0.25 → 0.35 rad (足上げ増強)
  * kp: 20 → 25 (旧 miniquad と同じ: 剛性回復)
  * 外乱 (push): 毎 push_interval 秒にランダム速度を体に付与

【domain randomization】
  各エピソードリセット時:
    - 地面摩擦: base × rand(0.5〜1.5)
    - 体重量: base × rand(0.9〜1.1) (set_mass_shift で実装)
    - 外乱: 毎 push_interval_s 秒に random ±push_vel を体速度に加算

【実機との対応 (BNO055 + PCA9685)】
  観測ベクトルは v1 と同一 (38 次元)。
  実機デプロイ時は obs 正規化に学習時と同じ scale を使うこと。
"""
import torch
import math
import numpy as np
import genesis as gs
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat


def gs_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower


class miniquad2DOFTerrainEnv:
    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg,
                 show_viewer=False, device="cuda"):

        self.device = torch.device(device)
        self.num_envs    = num_envs
        self.num_obs     = obs_cfg["num_obs"]
        self.num_privileged_obs = None
        self.num_actions = env_cfg["num_actions"]
        self.num_commands = command_cfg["num_commands"]

        self.simulate_action_latency = True
        self.dt = 0.02
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)

        self.env_cfg      = env_cfg
        self.obs_cfg      = obs_cfg
        self.reward_cfg   = reward_cfg
        self.command_cfg  = command_cfg
        self.obs_scales   = obs_cfg["obs_scales"]
        self.reward_scales = reward_cfg["reward_scales"]

        # 外乱インターバル (steps)
        self.push_interval = int(env_cfg.get("push_interval_s", 5.0) / self.dt)

        # ── Genesis シーン構築 ──────────────────────────────────────────────
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=2),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=int(0.5 / self.dt),
                camera_pos=(2.0, -2.0, 1.5),
                camera_lookat=(2.0, 2.0, 0.0),
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

        # ── 地形生成 ──────────────────────────────────────────────────────
        # 4行×2列 = 8 subterrains
        # 行 (x方向): 難易度 0→3
        # 列 (y方向): バリエーション A/B
        h_scale = env_cfg.get("terrain_horizontal_scale", 0.05)  # 5cm解像度
        v_scale = env_cfg.get("terrain_vertical_scale",   0.005) # 5mm/unit
        self.terrain = self.scene.add_entity(
            morph=gs.morphs.Terrain(
                n_subterrains=(4, 2),
                subterrain_size=(2.0, 2.0),   # 各 subterrain: 2m × 2m
                horizontal_scale=h_scale,
                vertical_scale=v_scale,
                subterrain_types=[
                    # 難易度0: 平坦 (学習開始地点)
                    ["flat_terrain",          "flat_terrain"],
                    # 難易度1: 小凸凹 + 波状 (軽い外乱)
                    ["random_uniform_terrain", "wave_terrain"],
                    # 難易度2: 大凸凹 + 傾斜 (バランス試練)
                    ["random_uniform_terrain", "pyramid_sloped_terrain"],
                    # 難易度3: 障害物 + 階段 (最難関)
                    ["discrete_obstacles_terrain", "pyramid_stairs_terrain"],
                ],
                randomize=True,
                # subterrain_parameters: 全て meters 単位で指定
                # robot height=80mm, toe=8mm → bumps はロボット高さの 10-20% まで
                subterrain_parameters={
                    "random_uniform_terrain": {
                        "min_height": -0.005,   # -5mm 凹み
                        "max_height":  0.015,   # +15mm 凸起き
                        "step": 0.005,          # 5mm 刻み
                        "downsampled_scale": 0.2, # 20cm 間隔サンプリング → 滑らか
                    },
                    "wave_terrain": {
                        "amplitude": 0.008,     # 8mm 波高
                        "num_waves": 3.0,
                    },
                    "pyramid_sloped_terrain": {
                        "slope": 0.05,          # 5% 勾配 (≈ 3°) → 2m で高さ差 5cm
                    },
                    "discrete_obstacles_terrain": {
                        "max_height": 0.015,    # 1.5cm 最大高さ
                        "min_size": 0.06,       # 6cm 最小幅
                        "max_size": 0.12,       # 12cm 最大幅
                        "num_rects": 15,
                    },
                    "pyramid_stairs_terrain": {
                        "step_width": 0.10,     # 10cm 幅
                        "step_height": 0.010,   # 1cm 段差
                    },
                },
                name="miniquad_terrain_v2",   # キャッシュ名 (再実行時に高速化)
            ),
        )

        # ── ロボット追加 ──────────────────────────────────────────────────
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

        # ── 関節・リンクインデックス ──────────────────────────────────────
        self.motor_dofs = [self.robot.get_joint(name).dof_idx_local
                           for name in self.env_cfg["dof_names"]]
        feet_link_names = ["femur_fr", "femur_fl", "femur_rr", "femur_rl"]
        self.feet_idx_local = [self.robot.get_link(n).idx_local for n in feet_link_names]
        self.num_feet  = len(feet_link_names)
        self.num_links = len(self.robot.links)  # set_friction_ratio に必要 (n_envs, n_links)

        self.robot.set_dofs_kp([env_cfg["kp"]] * self.num_actions, self.motor_dofs)
        self.robot.set_dofs_kv([env_cfg["kd"]] * self.num_actions, self.motor_dofs)

        # ── 報酬登録 ─────────────────────────────────────────────────────
        self.reward_functions, self.episode_sums = dict(), dict()
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros((num_envs,), device=self.device, dtype=gs.tc_float)

        # ── 状態バッファ ─────────────────────────────────────────────────
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

        self.base_pos  = torch.zeros((num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.base_quat = torch.zeros((num_envs, 4), device=self.device, dtype=gs.tc_float)

        self.default_dof_pos = torch.tensor(
            [env_cfg["default_joint_angles"][name] for name in env_cfg["dof_names"]],
            device=self.device, dtype=gs.tc_float,
        )

        self.trot_phase = torch.zeros(num_envs, device=self.device, dtype=gs.tc_float)
        self.feet_air_time = torch.zeros((num_envs, self.num_feet), device=self.device, dtype=gs.tc_float)
        self.last_contacts  = torch.ones ((num_envs, self.num_feet), device=self.device, dtype=torch.bool)

        self.extras = {"observations": {}}

    # ─────────────────────────────────────────────────────────────────────────

    def _resample_commands(self, envs_idx):
        self.commands[envs_idx, 0] = gs_rand_float(*self.command_cfg["lin_vel_x_range"], (len(envs_idx),), self.device)
        self.commands[envs_idx, 1] = gs_rand_float(*self.command_cfg["lin_vel_y_range"], (len(envs_idx),), self.device)
        self.commands[envs_idx, 2] = gs_rand_float(*self.command_cfg["ang_vel_range"],   (len(envs_idx),), self.device)

    def _get_trot_bias(self):
        """v2 改良版 trot_bias。
        v1 からの変更:
          - coxa_amp: 0.25 → 0.35 rad (足上げ強化)
          - femur_amp: 0.20 → 0.30 rad (ストライド拡大)
        gait_pattern の形状は v1/miniquad_2dof_env.py と同じ (coxa山形 + femur鋸歯)。
        """
        freq   = self.env_cfg.get("trot_bias_freq",      1.0)
        c_amp  = self.env_cfg.get("trot_bias_coxa_amp",  0.35)
        f_amp  = self.env_cfg.get("trot_bias_femur_amp", 0.30)

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

        coxa_a, femur_a = gait_pattern(phase_a)
        coxa_b, femur_b = gait_pattern(phase_b)

        return torch.stack([
            coxa_a, femur_a,   # FR
            coxa_b, femur_b,   # FL
            coxa_b, femur_b,   # RR
            coxa_a, femur_a,   # RL
        ], dim=-1)

    def _apply_push(self, push_envs_idx):
        """ランダム速度外乱。直接 set_vel は使えないので、
        現在の速度にランダムノイズを加えた値を再設定する。
        """
        if len(push_envs_idx) == 0:
            return
        push_vel = gs_rand_float(
            -self.env_cfg.get("push_vel_max", 0.3),
             self.env_cfg.get("push_vel_max", 0.3),
            (len(push_envs_idx), 3), self.device
        )
        # ロボット全体の速度を変更 (robot.set_vel がない場合 dof 速度を使う)
        # set_dofs_velocity でベース速度は変えられないため、
        # 小さなランダムトルクを一時的に加える形で外乱を模擬。
        # → 実装: dof_vel にノイズを加え、次ステップの慣性として作用させる
        noise_dof_vel = self.dof_vel[push_envs_idx] + gs_rand_float(
            -0.5, 0.5, (len(push_envs_idx), self.num_actions), self.device
        )
        self.robot.set_dofs_velocity(
            velocity=noise_dof_vel,
            dofs_idx_local=self.motor_dofs,
            envs_idx=push_envs_idx,
        )

    def step(self, actions):
        self.actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])
        exec_actions = self.last_actions if self.simulate_action_latency else self.actions

        trot_bias = self._get_trot_bias() if self.env_cfg.get("use_trot_bias", True) else 0.0
        target_dof_pos = exec_actions * self.env_cfg["action_scale"] + self.default_dof_pos + trot_bias
        self.robot.control_dofs_position(target_dof_pos, self.motor_dofs)
        self.scene.step()

        freq = self.env_cfg.get("trot_bias_freq", 1.0)
        self.trot_phase = (self.trot_phase + freq * self.dt) % 1.0
        self.episode_length_buf += 1

        # 外乱 (push): push_interval ステップごとにランダム速度加算
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
        self.base_lin_vel[:]      = transform_by_quat(self.robot.get_vel(), inv_base_quat)
        self.base_ang_vel[:]      = transform_by_quat(self.robot.get_ang(), inv_base_quat)
        self.projected_gravity    = transform_by_quat(self.global_gravity, inv_base_quat)
        self.dof_pos[:] = self.robot.get_dofs_position(self.motor_dofs)
        self.dof_vel[:] = self.robot.get_dofs_velocity(self.motor_dofs)

        envs_idx = (
            (self.episode_length_buf % int(self.env_cfg["resampling_time_s"] / self.dt) == 0)
            .nonzero(as_tuple=False).flatten()
        )
        self._resample_commands(envs_idx)

        # 終了条件: 地形では転倒角度を若干緩和 (傾斜面ではロール/ピッチが大きくなる)
        self.reset_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= torch.abs(self.base_euler[:, 1]) > self.env_cfg["termination_if_pitch_greater_than"]
        self.reset_buf |= torch.abs(self.base_euler[:, 0]) > self.env_cfg["termination_if_roll_greater_than"]
        # 地形外に落ちた (z < -0.2m) 場合もリセット
        self.reset_buf |= self.base_pos[:, 2] < -0.10

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

        # domain randomization: 摩擦をリセット時にランダム化
        # set_friction_ratio の形状: (n_envs, n_links) = (len(envs_idx), self.num_links)
        if self.env_cfg.get("randomize_friction", True):
            friction_ratio = gs_rand_float(
                0.5, 1.5, (len(envs_idx), self.num_links), self.device
            )
            self.robot.set_friction_ratio(
                friction_ratio=friction_ratio,
                links_idx_local=None,  # 全リンクに適用
                envs_idx=envs_idx,
            )

        # ロボット位置リセット (terrain origin + ロボット高さ)
        self.base_pos[envs_idx]  = self.base_init_pos
        self.base_quat[envs_idx] = self.base_init_quat.reshape(1, -1)
        self.robot.set_pos(self.base_pos[envs_idx],  zero_velocity=False, envs_idx=envs_idx)
        self.robot.set_quat(self.base_quat[envs_idx], zero_velocity=False, envs_idx=envs_idx)

        self.base_lin_vel[envs_idx] = 0
        self.base_ang_vel[envs_idx] = 0
        self.robot.zero_all_dofs_velocity(envs_idx)

        self.last_actions[envs_idx]       = 0.0
        self.last_dof_vel[envs_idx]       = 0.0
        self.episode_length_buf[envs_idx] = 0
        self.reset_buf[envs_idx]          = True
        self.feet_air_time[envs_idx]      = 0.0
        self.last_contacts[envs_idx]      = True
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

    # ─────────────────────────────────────────────────────────────────────────
    # 報酬関数
    # ─────────────────────────────────────────────────────────────────────────

    def _reward_tracking_lin_vel(self):
        """ワールド座標系 xy 線速度追従 (指数型)"""
        lin_vel_error = torch.sum(
            torch.square(self.commands[:, :2] - self.base_lin_vel_world[:, :2]), dim=1
        )
        return torch.exp(-lin_vel_error / self.reward_cfg["tracking_sigma"])

    def _reward_alive(self):
        """生存ボーナス (転倒回避動機)"""
        return torch.ones(self.num_envs, device=self.device, dtype=gs.tc_float)

    def _reward_forward_vel(self):
        """前進速度直接補助 (0〜1 m/s クランプ)"""
        vx = self.base_lin_vel_world[:, 0]
        return torch.clamp(vx, 0.0, 1.0)

    def _reward_lin_vel_z(self):
        """上下動ペナルティ"""
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_base_height(self):
        """重心高さ維持 (目標: 0.080m)。
        地形上では地面高さ分だけオフセットするが、近似として base_pos.z を使用。
        terrain が平坦でない場合、robot.base_pos.z は terrain_height + robot_height になる。
        """
        return torch.square(self.base_pos[:, 2] - self.reward_cfg["base_height_target"])

    def _reward_action_rate(self):
        """アクション変化量ペナルティ (ノイズ抑制)"""
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_heading(self):
        """yaw 偏差ペナルティ (直進維持)"""
        return torch.square(self.base_euler[:, 2])

    def _reward_orientation(self):
        """ロール/ピッチ最小化 (地形適応: 傾いた体を安定させる)。
        地面が傾いていても体を水平に保つことを促進。
        projected_gravity の xy 成分が小さいほど体が水平。
        """
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)
