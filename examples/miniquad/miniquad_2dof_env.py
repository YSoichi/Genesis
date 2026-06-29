"""
miniquad_2dof_env.py
====================
Genesis 物理エンジンを使用した 2DOF-hip 4脚ロボットの強化学習環境。

【ロボット仕様 (miniquad_2dof.urdf)】
  重量: 約 200-240g (RPi4 + BNO055 + PCA9685 + PLA + SG90×8 + LiPo)
  サーボ: SG90 (1.2〜1.6 kg·cm = 0.12〜0.16 N·m)
  自由度: 4脚 × 2DOF = 8DOF

【関節構成 (旧モデル hip+knee → 新モデル coxa(x) + femur(y))】
  body → coxa_link → femur → toe(固定)

  各脚の DOF:
    body_to_coxa_{fr/fl/rr/rl}_j : x軸回転 (±45°) — 側方傾斜/足上げ
    coxa_{fr/fl/rr/rl}_to_femur_{fr/fl/rr/rl}_j : y軸回転 (±60°) — 前後スイング

  DOF 順序: [coxa_fr, femur_fr, coxa_fl, femur_fl, coxa_rr, femur_rr, coxa_rl, femur_rl]

【足上げ機構 (膝なし)】
  coxa (x軸) を θ=0.30 rad 内転させると:
    足先 z 上昇 = femur_length × (1 - cosθ) ≈ 50mm × 0.045 = 2.3mm
    足先 y 移動 = femur_length × sinθ       ≈ 50mm × 0.296 = 14.8mm (内側)
  スイング相のみ coxa を正方向に傾けることで接地力なし歩行を実現。

【観測ベクトル (38次元)】
  [0:3]   base_ang_vel × 0.25  : ボディ角速度
  [3:6]   projected_gravity    : 重力投影 (傾き検知)
  [6:9]   commands × scale     : 速度指令 (vx, vy, yaw)
  [9:12]  base_lin_vel × 2.0   : ボディ線速度 (ボディ座標系)
  [12:20] dof_pos - default     : 関節角度偏差 (8次元)
  [20:28] dof_vel × 0.05       : 関節角速度 (8次元)
  [28:36] actions              : 前ステップアクション (8次元)
  [36:38] [cos(yaw), sin(yaw)] : ヘディング情報 (2次元)

【実機への対応 (RPi4 + BNO055 + PCA9685)】
  BNO055 → ang_vel (GYRO) + quaternion → projected_gravity
  PCA9685 → SG90 × 8 PWM 制御 (位置フィードバックなし → 指令値を dof_pos として使用)
"""
import torch
import math
import genesis as gs
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat


def gs_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower


class miniquad2DOFEnv:
    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg,
                 show_viewer=False, device="cuda"):

        self.device = torch.device(device)
        self.num_envs    = num_envs
        self.num_obs     = obs_cfg["num_obs"]
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

        # ── Genesis シーン構築 ──────────────────────────────────────────────
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

        # 2DOF URDF を読み込む (miniquad_2dof.urdf)
        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file="miniquad_2dof.urdf",
                pos=self.base_init_pos.cpu().numpy(),
                quat=self.base_init_quat.cpu().numpy(),
                recompute_inertia=True,
            ),
        )

        self.scene.build(n_envs=num_envs)

        # 関節インデックス (8 DOF: [coxa_fr, femur_fr, coxa_fl, femur_fl, ...])
        self.motor_dofs = [self.robot.get_joint(name).dof_idx_local
                           for name in self.env_cfg["dof_names"]]

        # 足先リンク: femur を使用 (knee なし → shank の代わり)
        # Genesis が fixed joint (femur→toe) を femur に統合するため femur で接触判定
        feet_link_names = ["femur_fr", "femur_fl", "femur_rr", "femur_rl"]
        self.feet_idx_local = [self.robot.get_link(n).idx_local for n in feet_link_names]
        self.num_feet = len(feet_link_names)

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

        # trot_bias 用フェーズ変数 (0.0〜1.0 の正規化位相)
        self.trot_phase = torch.zeros(num_envs, device=self.device, dtype=gs.tc_float)

        # feet_air_time バッファ (trot品質報酬用)
        self.feet_air_time = torch.zeros((num_envs, self.num_feet), device=self.device, dtype=gs.tc_float)
        self.last_contacts  = torch.ones ((num_envs, self.num_feet), device=self.device, dtype=torch.bool)

        self.extras = {"observations": {}}

    # ─────────────────────────────────────────────────────────────────────────

    def _resample_commands(self, envs_idx):
        self.commands[envs_idx, 0] = gs_rand_float(*self.command_cfg["lin_vel_x_range"], (len(envs_idx),), self.device)
        self.commands[envs_idx, 1] = gs_rand_float(*self.command_cfg["lin_vel_y_range"], (len(envs_idx),), self.device)
        self.commands[envs_idx, 2] = gs_rand_float(*self.command_cfg["ang_vel_range"],   (len(envs_idx),), self.device)

    def _get_trot_bias(self):
        """2DOF-hip 用トロットバイアス。
        旧モデル (hip y + knee y) の gait_pattern を coxa(x軸) + femur(y軸) に置き換え。

        スタンス相 (p: 0→0.5):
          coxa = 0       (足先を地面に置く)
          femur: +amp → -amp (前方から後方へ脚を引く → 体を前進させる)
        スイング相 (p: 0.5→1.0):
          coxa: 山形 sin(π×w) (脚を内転させて足先を地面から浮かせる)
          femur: -amp → +amp (後方から前方へ脚を戻す)

        FR/RL が同位相 (phase_a)、FL/RR が逆位相 (phase_b) のダイアゴナルトロット。
        """
        freq   = self.env_cfg.get("trot_bias_freq",      1.0)
        c_amp  = self.env_cfg.get("trot_bias_coxa_amp",  0.25)  # coxa 振幅 [rad] (x軸, 足上げ)
        f_amp  = self.env_cfg.get("trot_bias_femur_amp", 0.20)  # femur 振幅 [rad] (y軸, 前後)

        t = self.episode_length_buf.float() * self.dt
        phase_a = torch.remainder(freq * t, 1.0)         # FR, RL: 0〜1
        phase_b = torch.remainder(freq * t + 0.5, 1.0)  # FL, RR: 半周期ずれ

        def gait_pattern(p):
            stance = (p < 0.5).float()
            swing  = 1.0 - stance

            # スタンス: femur が前から後ろへ (body を前進させる)
            s = torch.clamp(p / 0.5, 0.0, 1.0)
            femur_stance = f_amp * (1.0 - 2.0 * s)   # +amp → -amp

            # スイング: femur が後ろから前へ (次の踏み出し準備)
            w = torch.clamp((p - 0.5) / 0.5, 0.0, 1.0)
            femur_swing  = f_amp * (2.0 * w - 1.0)   # -amp → +amp

            # スイング中のみ coxa を傾けて足を浮かせる (sin で滑らかに)
            coxa_swing = c_amp * torch.sin(math.pi * w)  # 山形: 0 → peak → 0

            femur = stance * femur_stance + swing * femur_swing
            coxa  = swing * coxa_swing   # スタンス中は coxa=0

            return coxa, femur

        coxa_a, femur_a = gait_pattern(phase_a)  # FR, RL グループ
        coxa_b, femur_b = gait_pattern(phase_b)  # FL, RR グループ

        # DOF 順: [coxa_fr, femur_fr, coxa_fl, femur_fl, coxa_rr, femur_rr, coxa_rl, femur_rl]
        return torch.stack([
            coxa_a, femur_a,  # FR
            coxa_b, femur_b,  # FL
            coxa_b, femur_b,  # RR
            coxa_a, femur_a,  # RL
        ], dim=-1)

    def step(self, actions):
        self.actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])
        exec_actions = self.last_actions if self.simulate_action_latency else self.actions

        trot_bias = self._get_trot_bias() if self.env_cfg.get("use_trot_bias", True) else 0.0
        target_dof_pos = exec_actions * self.env_cfg["action_scale"] + self.default_dof_pos + trot_bias
        self.robot.control_dofs_position(target_dof_pos, self.motor_dofs)
        self.scene.step()

        # trot_bias フェーズ更新 (dt ごとに進める)
        freq = self.env_cfg.get("trot_bias_freq", 1.0)
        self.trot_phase = (self.trot_phase + freq * self.dt) % 1.0

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

        # 終了判定 (2DOF モデルは低重心なので roll/pitch 閾値を若干緩和)
        self.reset_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= torch.abs(self.base_euler[:, 1]) > self.env_cfg["termination_if_pitch_greater_than"]
        self.reset_buf |= torch.abs(self.base_euler[:, 0]) > self.env_cfg["termination_if_roll_greater_than"]

        time_out_idx = (self.episode_length_buf > self.max_episode_length).nonzero(as_tuple=False).flatten()
        self.extras["time_outs"] = torch.zeros_like(self.reset_buf, device=self.device, dtype=gs.tc_float)
        self.extras["time_outs"][time_out_idx] = 1.0

        self.reset_idx(self.reset_buf.nonzero(as_tuple=False).flatten())

        # 報酬計算
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
        # trot フェーズをランダム初期化 (多様なフェーズから学習開始)
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
        """生存ボーナス (探索維持)"""
        return torch.ones(self.num_envs, device=self.device, dtype=gs.tc_float)

    def _reward_forward_vel(self):
        """前進速度への直接密報酬 (0〜1 m/s をクランプ)"""
        vx = self.base_lin_vel_world[:, 0]
        return torch.clamp(vx, 0.0, 1.0)

    def _reward_lin_vel_z(self):
        """上下動ペナルティ (バウンシング抑制)"""
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_base_height(self):
        """重心高さ維持ペナルティ (目標高さ: 0.080m)"""
        return torch.square(self.base_pos[:, 2] - self.reward_cfg["base_height_target"])

    def _reward_action_rate(self):
        """アクション変化量ペナルティ (ノイズ成長抑制)"""
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_heading(self):
        """yaw 偏差ペナルティ (直進方向 +x を維持)"""
        return torch.square(self.base_euler[:, 2])

    def _reward_coxa_symmetry(self):
        """coxa (x軸) の左右対称ペナルティ。
        左脚と右脚の coxa 角度の符号が逆になるべき (右+なら左-)。
        DOF 順: [coxa_fr(0), femur_fr(1), coxa_fl(2), femur_fl(3),
                 coxa_rr(4), femur_rr(5), coxa_rl(6), femur_rl(7)]
        """
        coxa_fr = self.dof_pos[:, 0]
        coxa_fl = self.dof_pos[:, 2]
        coxa_rr = self.dof_pos[:, 4]
        coxa_rl = self.dof_pos[:, 6]
        # 左右対称: coxa_fr + coxa_fl ≈ 0 (右内転=左内転のミラー)
        asym_front = torch.square(coxa_fr + coxa_fl)
        asym_rear  = torch.square(coxa_rr + coxa_rl)
        return asym_front + asym_rear
