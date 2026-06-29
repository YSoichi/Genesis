"""
hardware/config.py
==================
ハードウェア設定ファイル。実機に合わせて変更する唯一の場所。

【校正手順】
  1. 各サーボを手で中立位置 (関節角度 0 rad) に置く
  2. SERVO_OFFSET_DEG を調整して servo_angle ≈ 90° になるようにする
  3. SERVO_DIRECTION を確認: 正の関節角度でサーボが正しい方向へ動くか
  4. python3 -c "from hardware.servo import ServoController; ServoController().calibrate()" で確認

【SG90 PWM仕様】
  周波数 : 50 Hz (周期 20ms)
  0°   → 500μs  (PCA9685 カウント: ~102)
  90°  → 1500μs (PCA9685 カウント: ~307)  ← 中立位置
  180° → 2500μs (PCA9685 カウント: ~512)
"""

# ─── I2C バス ──────────────────────────────────────────────────────────────
I2C_BUS = 1                # RPi4 のデフォルト I2C バス

# ─── PCA9685 ──────────────────────────────────────────────────────────────
PCA9685_ADDRESS  = 0x40    # アドレス未変更なら 0x40
PCA9685_FREQ_HZ  = 50      # SG90 推奨: 50Hz

# SG90 パルス幅 (μs)
SG90_PULSE_MIN_US  = 500   # 0°
SG90_PULSE_MAX_US  = 2500  # 180°

# ─── BNO055 ───────────────────────────────────────────────────────────────
BNO055_ADDRESS = 0x28      # SA0 = LOW なら 0x28, HIGH なら 0x29

# ─── サーボ割り当て ────────────────────────────────────────────────────────
# PCA9685 チャンネル → 関節名
# joint 順は訓練スクリプトの dof_names と完全一致させる
# dof_names = [
#   "body_to_coxa_fr_j",    "coxa_fr_to_femur_fr_j",   # FR: ch0, ch1
#   "body_to_coxa_fl_j",    "coxa_fl_to_femur_fl_j",   # FL: ch2, ch3
#   "body_to_coxa_rr_j",    "coxa_rr_to_femur_rr_j",   # RR: ch4, ch5
#   "body_to_coxa_rl_j",    "coxa_rl_to_femur_rl_j",   # RL: ch6, ch7
# ]
JOINT_CHANNELS = {
    "body_to_coxa_fr_j":     0,
    "coxa_fr_to_femur_fr_j": 1,
    "body_to_coxa_fl_j":     2,
    "coxa_fl_to_femur_fl_j": 3,
    "body_to_coxa_rr_j":     4,
    "coxa_rr_to_femur_rr_j": 5,
    "body_to_coxa_rl_j":     6,
    "coxa_rl_to_femur_rl_j": 7,
}

# サーボ取り付け方向:
#   +1 = シミュレーションと同方向
#   -1 = 逆向き (左脚は鏡像になることが多い)
# 実機で確認して必要なら変更する
SERVO_DIRECTION = {
    "body_to_coxa_fr_j":      1,
    "coxa_fr_to_femur_fr_j":  1,
    "body_to_coxa_fl_j":     -1,   # 左脚: 鏡像
    "coxa_fl_to_femur_fl_j":  1,
    "body_to_coxa_rr_j":      1,
    "coxa_rr_to_femur_rr_j":  1,
    "body_to_coxa_rl_j":     -1,   # 左脚: 鏡像
    "coxa_rl_to_femur_rl_j":  1,
}

# 角度オフセット (度): サーボの中立位置ずれを補正する
# 工場出荷状態のサーボは 90° が正確な中立とは限らない
SERVO_OFFSET_DEG = {
    "body_to_coxa_fr_j":      0.0,
    "coxa_fr_to_femur_fr_j":  0.0,
    "body_to_coxa_fl_j":      0.0,
    "coxa_fl_to_femur_fl_j":  0.0,
    "body_to_coxa_rr_j":      0.0,
    "coxa_rr_to_femur_rr_j":  0.0,
    "body_to_coxa_rl_j":      0.0,
    "coxa_rl_to_femur_rl_j":  0.0,
}

# 関節可動範囲 (rad): ハードウェアリミット (SG90 は約 ±90° だが実用は ±75°)
# coxa: ±45° (設計値), femur: ±60° (設計値)
import math
JOINT_LIMITS_RAD = {
    "body_to_coxa_fr_j":      (-math.pi / 4, math.pi / 4),    # ±45°
    "coxa_fr_to_femur_fr_j":  (-math.pi / 3, math.pi / 3),    # ±60°
    "body_to_coxa_fl_j":      (-math.pi / 4, math.pi / 4),
    "coxa_fl_to_femur_fl_j":  (-math.pi / 3, math.pi / 3),
    "body_to_coxa_rr_j":      (-math.pi / 4, math.pi / 4),
    "coxa_rr_to_femur_rr_j":  (-math.pi / 3, math.pi / 3),
    "body_to_coxa_rl_j":      (-math.pi / 4, math.pi / 4),
    "coxa_rl_to_femur_rl_j":  (-math.pi / 3, math.pi / 3),
}

# ─── 制御パラメータ ────────────────────────────────────────────────────────
CONTROL_FREQ_HZ   = 50     # 制御ループ周波数 (シミュレーションと同じ dt=0.02s)
CONTROL_DT        = 1.0 / CONTROL_FREQ_HZ

# ─── BNO055 座標系補正 ─────────────────────────────────────────────────────
# BNO055 をロボットに取り付けた向きによって異なる。
# 基本方向: x=前方, y=左方, z=上方 (シミュレーションと同じ)
# 実機で確認して必要なら変更する (1=正, -1=反転)
IMU_AXIS_MAP = {
    "x_sign":  1,   # 前方
    "y_sign":  1,   # 左方
    "z_sign":  1,   # 上方
    # BNO055 の物理軸とロボット軸のマッピング (0=x, 1=y, 2=z)
    "forward_axis": 0,  # BNO055 の x軸がロボットの前方
    "left_axis":    1,  # BNO055 の y軸がロボットの左方
    "up_axis":      2,  # BNO055 の z軸がロボットの上方
}
