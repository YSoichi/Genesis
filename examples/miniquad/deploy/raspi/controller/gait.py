"""
controller/gait.py
==================
trot バイアス計算。シミュレーションの _get_trot_bias() と完全に同じロジック。

ロボットは 4脚 trot パターン (対角線ペア同期):
  Phase A: FR + RL (前右 + 後左) が swing
  Phase B: FL + RR (前左 + 後右) が swing (A から 0.5 cycle 遅れ)

step_count × dt でフェーズを管理する (エピソードリセット相当は不要)。
"""
import math
from dataclasses import dataclass

import numpy as np


@dataclass
class TrotBiasConfig:
    """trot バイアスの設定値。policy_meta.json から読み込む想定。"""
    freq_hz:   float = 2.0    # v4: 2Hz
    coxa_amp:  float = 0.30   # rad
    femur_amp: float = 0.25   # rad


class TrotBias:
    """trot バイアスを時間ベースで計算するクラス。

    使い方:
        tb = TrotBias(TrotBiasConfig(freq_hz=2.0, coxa_amp=0.30, femur_amp=0.25))
        bias = tb.step(step_count=42, dt=0.02)
        # bias は shape=(8,) numpy array
        # 順: [coxa_fr, femur_fr, coxa_fl, femur_fl, coxa_rr, femur_rr, coxa_rl, femur_rl]
    """

    def __init__(self, config: TrotBiasConfig | None = None):
        self.cfg = config or TrotBiasConfig()

    def _gait_pattern(self, phase: float) -> tuple:
        """1脚分の gait パターンを計算する。

        Args:
            phase: 0.0–1.0 の gait フェーズ (0.5 未満 = swing, 以降 = stance)
        Returns:
            (coxa_rad, femur_rad) のタプル
        """
        c_amp = self.cfg.coxa_amp
        f_amp = self.cfg.femur_amp

        if phase < 0.5:
            # stance フェーズ: femur が前→後に線形移動, coxa は 0
            s = phase / 0.5          # 0→1
            femur = f_amp * (1.0 - 2.0 * s)
            coxa  = 0.0
        else:
            # swing フェーズ: femur が後→前に線形移動, coxa が山なりに上昇
            w = (phase - 0.5) / 0.5  # 0→1
            femur = f_amp * (2.0 * w - 1.0)
            coxa  = c_amp * math.sin(math.pi * w)

        return coxa, femur

    def compute(self, t: float) -> np.ndarray:
        """時刻 t (秒) での全関節 trot バイアスを返す。

        Args:
            t: 経過時間 (秒)
        Returns:
            shape=(8,) の numpy array
            [coxa_fr, femur_fr, coxa_fl, femur_fl, coxa_rr, femur_rr, coxa_rl, femur_rl]
        """
        freq = self.cfg.freq_hz
        phase_a = math.fmod(freq * t, 1.0)
        phase_b = math.fmod(freq * t + 0.5, 1.0)

        c_a, f_a = self._gait_pattern(phase_a)
        c_b, f_b = self._gait_pattern(phase_b)

        return np.array([
            c_a, f_a,   # FR (phase A: swing 先行)
            c_b, f_b,   # FL (phase B: 0.5 遅れ)
            c_b, f_b,   # RR (phase B)
            c_a, f_a,   # RL (phase A)
        ], dtype=np.float32)

    def step(self, step_count: int, dt: float = 0.02) -> np.ndarray:
        """step_count × dt から trot バイアスを計算する。compute() の薄いラッパー。"""
        return self.compute(step_count * dt)
