"""
controller/state.py
===================
ロボットの状態推定と観測ベクトル (38次元) の構築。

シミュレーションとのマッピング:
  obs = [
    base_ang_vel × 0.25          (3) ← IMU gyro (rad/s)
    projected_gravity             (3) ← IMU gravity vector (body frame)
    commands × [2, 2, 0.25]      (3) ← 速度コマンド (m/s, m/s, rad/s)
    base_lin_vel × 2.0           (3) ← 線形速度推定 (加速度積分)
    (dof_pos - default) × 1.0   (8) ← サーボ位置推定 (コマンド値を追跡)
    dof_vel × 0.05               (8) ← 関節速度推定 (dof_pos の差分)
    last_actions                  (8) ← 前ステップのポリシー出力
    [cos(yaw), sin(yaw)]          (2) ← ヘディング (IMU euler yaw)
  ]
  計: 38次元
"""
import math
from typing import Optional

import numpy as np

from .gait import TrotBias, TrotBiasConfig


class VelocityEstimator:
    """加速度積分による線形速度推定。

    BNO055 の linear_acceleration (重力除去済み) を積分して速度を推定する。
    ドリフトが蓄積するため、定期的にリセットする必要がある (歩行開始時など)。
    """

    def __init__(self, dt: float = 0.02, decay: float = 0.95):
        """
        Args:
            dt: 制御周期 (秒)
            decay: 速度の減衰係数 (スリップ/ドリフト補正)
        """
        self._dt    = dt
        self._decay = decay
        self._vel   = np.zeros(3, dtype=np.float32)

    def update(self, lin_accel: tuple) -> np.ndarray:
        """線形加速度 (m/s²) を積分して速度 (m/s) を更新する。"""
        a = np.array(lin_accel, dtype=np.float32)
        self._vel = self._vel * self._decay + a * self._dt
        return self._vel.copy()

    def reset(self):
        """速度推定をリセットする (転倒検知後など)。"""
        self._vel[:] = 0.0

    @property
    def velocity(self) -> np.ndarray:
        return self._vel.copy()


class RobotStateEstimator:
    """IMU + サーボコマンド追跡からシミュレーションと同じ 38次元観測を構築する。

    使い方:
        estimator = RobotStateEstimator(meta)
        estimator.reset()
        obs = estimator.update(imu_data, last_actions, command=[0.10, 0, 0])
    """

    # シミュレーションの obs_scales と完全一致させる
    _OBS_SCALES = {
        "ang_vel":  0.25,
        "lin_vel":  2.0,
        "dof_pos":  1.0,
        "dof_vel":  0.05,
    }
    # commands の scaling: [lin_vel_x × 2, lin_vel_y × 2, ang_vel × 0.25]
    _CMD_SCALES = np.array([2.0, 2.0, 0.25], dtype=np.float32)

    def __init__(self, meta: dict, dt: float = 0.02):
        """
        Args:
            meta: export_onnx.py が生成した policy_meta.json の内容
            dt  : 制御周期 (秒)
        """
        self._dt       = dt
        self._meta     = meta
        self._dof_names = meta["dof_names"]   # 8関節名 (順序がポリシー学習時と一致)
        self._default_pos = np.array(
            [meta["default_joint_angles"][n] for n in self._dof_names], dtype=np.float32
        )

        # trot バイアス計算器
        tb_cfg = TrotBiasConfig(
            freq_hz  = meta.get("trot_bias_freq", 2.0),
            coxa_amp = meta.get("trot_bias_coxa_amp", 0.30),
            femur_amp= meta.get("trot_bias_femur_amp", 0.25),
        )
        self._trot = TrotBias(tb_cfg)

        # 速度推定
        self._vel_est = VelocityEstimator(dt=dt)

        # サーボ位置追跡 (エンコーダなし → コマンド値を使用)
        self._dof_pos     = self._default_pos.copy()
        self._dof_pos_prev= self._default_pos.copy()
        self._last_actions= np.zeros(len(self._dof_names), dtype=np.float32)
        self._step_count  = 0

    def reset(self):
        """エピソード開始時に状態をリセットする。"""
        self._dof_pos[:]      = self._default_pos
        self._dof_pos_prev[:] = self._default_pos
        self._last_actions[:] = 0.0
        self._step_count = 0
        self._vel_est.reset()

    def update(
        self,
        imu_data,           # hardware.IMUData
        last_actions: np.ndarray,
        command: list = None,
    ) -> np.ndarray:
        """38次元観測ベクトルを構築する。

        Args:
            imu_data   : IMUReader.read() の返り値
            last_actions: 前ステップのポリシー出力 (shape=(8,))
            command    : [vx_cmd, vy_cmd, yaw_cmd] (省略時は [0.10, 0, 0])
        Returns:
            obs: shape=(38,) の観測ベクトル (float32)
        """
        if command is None:
            command = [0.10, 0.0, 0.0]

        cmd = np.array(command, dtype=np.float32)

        # ── 角速度 (×0.25) ────────────────────────────────────────────────
        gyro = np.array(imu_data.gyro_rad, dtype=np.float32)
        ang_vel_scaled = gyro * self._OBS_SCALES["ang_vel"]

        # ── 重力ベクトル (body frame) ──────────────────────────────────────
        # BNO055 の gravity は外向き (9.81 m/s²)。
        # シミュレーションの projected_gravity は downward 方向の単位ベクトル。
        # 重力ベクトルをそのまま正規化して使う (下向き = 負の z)
        grav = np.array(imu_data.gravity, dtype=np.float32)

        # ── 速度推定 (×2.0) ───────────────────────────────────────────────
        lin_vel_body = self._vel_est.update(imu_data.lin_accel)
        lin_vel_scaled = lin_vel_body * self._OBS_SCALES["lin_vel"]

        # ── 関節位置・速度 ─────────────────────────────────────────────────
        # エンコーダなし: 前ステップで送ったコマンド値でトラッキング
        dof_pos_rel = (self._dof_pos - self._default_pos) * self._OBS_SCALES["dof_pos"]
        dof_vel     = (self._dof_pos - self._dof_pos_prev) / self._dt * self._OBS_SCALES["dof_vel"]

        # ── ヘディング ─────────────────────────────────────────────────────
        yaw = imu_data.euler_rad[2]
        heading = np.array([math.cos(yaw), math.sin(yaw)], dtype=np.float32)

        # ── 連結 ───────────────────────────────────────────────────────────
        obs = np.concatenate([
            ang_vel_scaled,           # 3
            grav,                     # 3
            cmd * self._CMD_SCALES,   # 3
            lin_vel_scaled,           # 3
            dof_pos_rel,              # 8
            dof_vel,                  # 8
            last_actions,             # 8
            heading,                  # 2
        ])  # 合計 38

        self._last_actions = last_actions.copy()
        self._step_count  += 1
        return obs.astype(np.float32)

    def apply_actions(self, raw_actions: np.ndarray) -> np.ndarray:
        """ポリシー出力 → サーボ目標角度 (rad) に変換する。

        target_dof_pos = clip(action, ±10) × action_scale + default_pos + trot_bias

        Args:
            raw_actions: ポリシー出力 (shape=(8,), ONNX の actions 出力)
        Returns:
            target_dof_pos: shape=(8,) の目標関節角度 (rad)
        """
        clip = self._meta.get("clip_actions", 10.0)
        scale = self._meta.get("action_scale", 0.5)

        actions = np.clip(raw_actions, -clip, clip)
        trot_bias = self._trot.step(self._step_count, self._dt)
        target = actions * scale + self._default_pos + trot_bias

        # 次ステップの dof_pos 追跡を更新
        self._dof_pos_prev[:] = self._dof_pos
        self._dof_pos[:]      = target

        return target

    @property
    def dof_names(self):
        return self._dof_names
