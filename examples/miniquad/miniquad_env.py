"""
miniquad_env.py
===============
Genesis 物理エンジンを使用した小型 4脚ロボット (miniquad) の強化学習環境。

【ロボット仕様】
  重量: 約 200g (RPi4 + IMU + PLA 車体 + サーボ×8)
  サーボ: SG90 相当 (1.2〜1.6 kg·cm = 0.12〜0.16 N·m)
  自由度: 4脚 × 2関節 (hip + knee) = 8 DOF

【miniquad.urdf のリンク/ジョイント構成】
  ボディ (body): box 0.10×0.06×0.025 m, mass=0.100 kg

  各脚の親子関係: body → thigh → shank → toe(固定)
  リンク寸法:
    thigh: cylinder length=0.045m, radius=0.008m, mass=0.012kg
    shank: cylinder length=0.050m, radius=0.007m, mass=0.003kg
    toe  : sphere radius=0.010m,  mass=0.001kg (Genesis が shank に統合)

  脚の配置と関節名 (計 8 アクチュエータ = 4脚 × 2関節):
    前右 (fr): body_to_hip_fr_j(y軸) / hip_fr_to_knee_fr_j(y軸)
    前左 (fl): body_to_hip_fl_j(y軸) / hip_fl_to_knee_fl_j(y軸)
    後右 (rr): body_to_hip_rr_j(y軸) / hip_rr_to_knee_rr_j(y軸)
    後左 (rl): body_to_hip_rl_j(y軸) / hip_rl_to_knee_rl_j(y軸)

  関節軸: 全関節 y軸回転 (前後スイング + 膝屈伸)
  座標系: x=前方, y=左方, z=上方

【観測ベクトル (use_cpg_obs=False 時: 38次元, True 時: 40次元)】
  [0:3]   base_ang_vel × 0.25  : ボディ角速度
  [3:6]   projected_gravity    : 重力投影 (傾き検知)
  [6:9]   commands × scale     : 速度指令 (vx, vy, yaw)
  [9:12]  base_lin_vel × 2.0   : ボディ線速度 (ボディ座標系; v9 追加)
  [12:20] dof_pos - default     : 関節角度偏差 (8次元)
  [20:28] dof_vel × 0.05       : 関節角速度 (8次元)
  [28:36] actions              : 前ステップアクション (8次元)
  [36:38] [cos(yaw), sin(yaw)] : ヘディング情報 (2次元)
  [38:40] [sin(φ), cos(φ)]    : CPG位相 (use_cpg_obs=True のみ)
"""
import torch
import math
import genesis as gs
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat


def gs_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower


class miniquadEnv:
    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg,
                 show_viewer=False, device="cuda"):

        self.device = torch.device(device)
        self.num_envs    = num_envs
        self.num_obs     = obs_cfg["num_obs"]       # 35
        self.num_privileged_obs = None
        self.num_actions = env_cfg["num_actions"]   # 8
        self.num_commands = command_cfg["num_commands"]  # 3

        self.simulate_action_latency = True
        self.dt = 0.02   # 50Hz
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)

        self.env_cfg      = env_cfg
        self.obs_cfg      = obs_cfg
        self.reward_cfg   = reward_cfg
        self.command_cfg  = command_cfg
        self.obs_scales   = obs_cfg["obs_scales"]
        self.reward_scales = reward_cfg["reward_scales"]

        # ── Genesis シーンの構築 ──────────────────────────────────────────────
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=2),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=int(0.5 / self.dt),
                camera_pos=(0.6, 0.0, 0.5),
                camera_lookat=(0.0, 0.0, 0.1),
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

        self.scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))

        self.base_init_pos  = torch.tensor(env_cfg["base_init_pos"],  device=self.device)
        self.base_init_quat = torch.tensor(env_cfg["base_init_quat"], device=self.device)
        self.inv_base_init_quat = inv_quat(self.base_init_quat)

        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file="miniquad.urdf",
                pos=self.base_init_pos.cpu().numpy(),
                quat=self.base_init_quat.cpu().numpy(),
                recompute_inertia=True,
            ),
        )

        self.scene.build(n_envs=num_envs)

        # 関節インデックス (8 DOF)
        self.motor_dofs = [self.robot.get_joint(name).dof_idx_local
                           for name in self.env_cfg["dof_names"]]

        # 足先リンク (Genesis が fixed joint を統合するため shank を使用)
        feet_link_names = ["shank_fr", "shank_fl", "shank_rr", "shank_rl"]
        self.feet_idx_local = [self.robot.get_link(n).idx_local for n in feet_link_names]
        self.num_feet = len(feet_link_names)  # 4

        self.robot.set_dofs_kp([env_cfg["kp"]] * self.num_actions, self.motor_dofs)
        self.robot.set_dofs_kv([env_cfg["kd"]] * self.num_actions, self.motor_dofs)

        # 報酬関数の登録
        self.reward_functions, self.episode_sums = dict(), dict()
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)

        # 状態バッファ
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

        self.feet_air_time = torch.zeros((num_envs, self.num_feet), device=self.device, dtype=gs.tc_float)
        self.last_contacts  = torch.ones ((num_envs, self.num_feet), device=self.device, dtype=torch.bool)

        # CPG フェーズ変数 (3Hz トロットリズム; 観測として渡すことで歩容発見を補助)
        self.cpg_freq   = env_cfg.get("cpg_freq", 3.0)   # [Hz]
        self.cpg_phase  = torch.zeros(num_envs, device=self.device, dtype=gs.tc_float)
        self.cpg_dphi   = 2.0 * math.pi * self.cpg_freq * self.dt

        self.extras = {"observations": {}}

    # ─────────────────────────────────────────────────────────────────────────

    def _resample_commands(self, envs_idx):
        self.commands[envs_idx, 0] = gs_rand_float(*self.command_cfg["lin_vel_x_range"], (len(envs_idx),), self.device)
        self.commands[envs_idx, 1] = gs_rand_float(*self.command_cfg["lin_vel_y_range"], (len(envs_idx),), self.device)
        self.commands[envs_idx, 2] = gs_rand_float(*self.command_cfg["ang_vel_range"],   (len(envs_idx),), self.device)

    def _get_trot_bias(self):
        """CPGアクションバイアス: ゼロポリシー出力でもトロット動作が起きるよう
        非対称 duty-cycle トロットパターンをアクション空間に直接重畳する。
        ポリシーは基本トロットを「いつ・どれだけ修正するか」だけ学べばよい。
        FR/RL同位相、FL/RR逆位相のダイアゴナルトロット。

        単純な正弦波 (旧実装) は時間平均ゼロの対称振動になり、
        policy 抜きでは前進力を生まない (peak_x=0 を実測で確認済み)。
        stance相 (脚接地, hip: +amp→-amp 線形に後方へpush) と
        swing相  (脚浮上, knee持ち上げ + hip: -amp→+amp で前方へrecovery)
        を分離した非対称パターンにすることで、policy なしでも前進力を生む。
        """
        freq    = self.env_cfg.get("trot_bias_freq",      2.5)
        h_amp   = self.env_cfg.get("trot_bias_hip_amp",   0.25)
        k_amp   = self.env_cfg.get("trot_bias_knee_amp",  0.20)
        push_amp = self.env_cfg.get("trot_bias_push_amp", 0.0)  # stance中のknee push-off振幅

        t = self.episode_length_buf.float() * self.dt
        phase_a = torch.remainder(freq * t, 1.0)        # FR, RL: 0~1
        phase_b = torch.remainder(freq * t + 0.5, 1.0)  # FL, RR: 半周期ずれ

        def gait_pattern(p):
            stance = (p < 0.5).float()
            swing  = 1.0 - stance

            s = torch.clamp(p / 0.5, 0.0, 1.0)            # stance内位相 0→1
            hip_stance = -h_amp * (1.0 - 2.0 * s)         # -amp → +amp (後方へpush)
            # stance中盤 (s≈0.5) で脚を伸展させ地面を蹴る (knee角度を0方向へ、正=伸展)
            knee_stance = push_amp * torch.sin(math.pi * s)

            w = torch.clamp((p - 0.5) / 0.5, 0.0, 1.0)    # swing内位相 0→1
            hip_swing = h_amp * (1.0 - 2.0 * w)           # +amp → -amp (前方へrecovery)
            knee_swing = -k_amp * torch.sin(math.pi * w)  # swing中のみ持ち上げ

            hip  = stance * hip_stance + swing * hip_swing
            knee = stance * knee_stance + swing * knee_swing
            return hip, knee

        hip_a, knee_a = gait_pattern(phase_a)  # FR, RL
        hip_b, knee_b = gait_pattern(phase_b)  # FL, RR

        # DOF順: [hip_fr, knee_fr, hip_fl, knee_fl, hip_rr, knee_rr, hip_rl, knee_rl]
        return torch.stack([
            hip_a, knee_a,   # hip_fr, knee_fr
            hip_b, knee_b,   # hip_fl, knee_fl
            hip_b, knee_b,   # hip_rr, knee_rr
            hip_a, knee_a,   # hip_rl, knee_rl
        ], dim=-1)  # (N, 8)

    def step(self, actions):
        self.actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])
        exec_actions = self.last_actions if self.simulate_action_latency else self.actions

        trot_bias = self._get_trot_bias() if self.env_cfg.get("use_trot_bias", False) else 0.0
        target_dof_pos = exec_actions * self.env_cfg["action_scale"] + self.default_dof_pos + trot_bias
        self.robot.control_dofs_position(target_dof_pos, self.motor_dofs)
        self.scene.step()

        self.episode_length_buf += 1
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

        # 速度指令の定期再サンプリング
        envs_idx = (
            (self.episode_length_buf % int(self.env_cfg["resampling_time_s"] / self.dt) == 0)
            .nonzero(as_tuple=False).flatten()
        )
        self._resample_commands(envs_idx)

        # 終了判定
        self.reset_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= torch.abs(self.base_euler[:, 1]) > self.env_cfg["termination_if_pitch_greater_than"]
        self.reset_buf |= torch.abs(self.base_euler[:, 0]) > self.env_cfg["termination_if_roll_greater_than"]

        time_out_idx = (self.episode_length_buf > self.max_episode_length).nonzero(as_tuple=False).flatten()
        self.extras["time_outs"] = torch.zeros_like(self.reset_buf, device=self.device, dtype=gs.tc_float)
        self.extras["time_outs"][time_out_idx] = 1.0

        self.reset_idx(self.reset_buf.nonzero(as_tuple=False).flatten())

        # CPG フェーズ更新
        self.cpg_phase = (self.cpg_phase + self.cpg_dphi) % (2.0 * math.pi)

        # 報酬計算
        self.rew_buf[:] = 0.0
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew

        yaw = self.base_euler[:, 2:3]
        heading_obs = torch.cat([torch.cos(yaw), torch.sin(yaw)], dim=-1)

        obs_parts = [
            self.base_ang_vel * self.obs_scales["ang_vel"],
            self.projected_gravity,
            self.commands * self.commands_scale,
            self.base_lin_vel * self.obs_scales["lin_vel"],   # 自己速度を直接観測
            (self.dof_pos - self.default_dof_pos) * self.obs_scales["dof_pos"],
            self.dof_vel * self.obs_scales["dof_vel"],
            self.actions,
            heading_obs,
        ]
        if self.env_cfg.get("use_cpg_obs", True):
            obs_parts.append(torch.stack([
                torch.sin(self.cpg_phase),
                torch.cos(self.cpg_phase),
            ], dim=-1))

        self.obs_buf = torch.cat(obs_parts, axis=-1)

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

        self.base_pos[envs_idx]  = self.base_init_pos
        self.base_quat[envs_idx] = self.base_init_quat.reshape(1, -1)
        self.robot.set_pos(self.base_pos[envs_idx],  zero_velocity=False, envs_idx=envs_idx)
        self.robot.set_quat(self.base_quat[envs_idx], zero_velocity=False, envs_idx=envs_idx)

        self.base_lin_vel[envs_idx] = 0
        self.base_ang_vel[envs_idx] = 0
        self.robot.zero_all_dofs_velocity(envs_idx)

        self.last_actions[envs_idx]    = 0.0
        self.last_dof_vel[envs_idx]    = 0.0
        self.episode_length_buf[envs_idx] = 0
        self.reset_buf[envs_idx]       = True
        self.feet_air_time[envs_idx]   = 0.0
        self.last_contacts[envs_idx]   = True
        # CPG フェーズをランダム初期化 (多様なgaitフェーズから学習開始)
        self.cpg_phase[envs_idx] = torch.rand(len(envs_idx), device=self.device) * 2.0 * math.pi

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
        """ワールド座標系 xy 線速度追従 (指数型; 静止でも非ゼロ勾配あり)"""
        lin_vel_error = torch.sum(
            torch.square(self.commands[:, :2] - self.base_lin_vel_world[:, :2]), dim=1
        )
        return torch.exp(-lin_vel_error / self.reward_cfg["tracking_sigma"])

    def _reward_tracking_ang_vel(self):
        """yaw 角速度追従"""
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / self.reward_cfg["tracking_sigma"])

    def _reward_alive(self):
        """生存ボーナス"""
        return torch.ones(self.num_envs, device=self.device, dtype=gs.tc_float)

    def _get_foot_in_air(self):
        """足の「空中」判定を shank CoM 高さ (z 座標) で行う。
        200g ロボットの静的接触力 ≈ 0.49N が 1.0N 閾値を下回るため接触力では判定不可。
        実測: hip=0.3, knee=-0.6 の立位で shank CoM z ≈ 0.0252m。
        閾値 0.035m = 立位 + 約 1cm の浮きで「空中」判定 (体重移動搾取を防ぐ)。
        """
        foot_pos = self.robot.get_links_pos(self.feet_idx_local, ref="link_com")  # (N, 4, 3)
        return foot_pos[:, :, 2] > 0.035  # (N, 4) bool

    def _reward_feet_air_time(self):
        """足の適切な空中時間 (歩容品質の誘導)"""
        in_air    = self._get_foot_in_air()  # (N, 4)
        contacts  = ~in_air
        contact_filt  = contacts | self.last_contacts
        first_contact = (self.feet_air_time > 0.0) & contact_filt

        self.feet_air_time += self.dt
        self.feet_air_time *= ~contact_filt

        threshold = self.reward_cfg.get("feet_air_time_threshold", 0.1)
        rew = torch.sum((self.feet_air_time - threshold) * first_contact, dim=-1)
        rew *= torch.norm(self.commands[:, :2], dim=-1) > 0.1
        self.last_contacts = contacts
        return rew

    def _reward_lin_vel_z(self):
        """上下動ペナルティ"""
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_base_height(self):
        """高さ維持ペナルティ"""
        return torch.square(self.base_pos[:, 2] - self.reward_cfg["base_height_target"])

    def _reward_action_rate(self):
        """アクション滑らかさペナルティ"""
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_similar_to_default(self):
        """デフォルト姿勢への近接ペナルティ (関節の安定化)"""
        return torch.sum(torch.square(self.dof_pos - self.default_dof_pos), dim=1)

    def _reward_heading(self):
        """yaw偏差ペナルティ (直進方向 +x を維持)"""
        return torch.square(self.base_euler[:, 2])

    def _reward_foot_clearance(self):
        """CPG位相に依存しない密な足上げ報酬。
        shank z > 0.035m を「空中」とし、浮いている足の割合 (0〜1) を返す。
        前進指令がある場合のみ有効 (停止中の足上げは不要)。
        【注意】この報酬単独では「その場トロット」の局所解に収束しやすい。
        """
        in_air = self._get_foot_in_air()  # (N, 4) bool
        cmd_mask = (torch.norm(self.commands[:, :2], dim=-1) > 0.1).float()
        return in_air.float().mean(dim=-1) * cmd_mask

    def _reward_forward_vel(self):
        """前進速度への直接密報酬 (指令との追従誤差ではなく絶対的な前進を促す)。
        0〜1 m/s を 0〜1 にクランプ。
        tracking_lin_vel と組み合わせることで「前進できれば何でも報酬」となり
        ゼロから前進を発見する局面で強い勾配を与える。
        """
        vx = self.base_lin_vel_world[:, 0]
        return torch.clamp(vx, 0.0, 1.0)

    def _reward_trot_gait(self):
        """CPG位相に同期したトロット歩容誘導
        足高さ (shank z > 0.040m) で空中判定 → 体重移動搾取を防ぐ。
        """
        in_air = self._get_foot_in_air()  # (N, 4) foot height based

        sin_phase = torch.sin(self.cpg_phase)  # (N,)
        phase_pos = torch.clamp( sin_phase, 0.0, 1.0)
        phase_neg = torch.clamp(-sin_phase, 0.0, 1.0)

        rew = (
            phase_pos * in_air[:, 0].float()   # FR: sin>0 に空中
          + phase_pos * in_air[:, 3].float()   # RL: sin>0 に空中
          + phase_neg * in_air[:, 1].float()   # FL: sin<0 に空中
          + phase_neg * in_air[:, 2].float()   # RR: sin<0 に空中
        ) / 4.0

        cmd_mask = (torch.norm(self.commands[:, :2], dim=-1) > 0.1).float()
        return rew * cmd_mask
