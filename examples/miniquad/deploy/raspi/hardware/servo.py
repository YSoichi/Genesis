"""
hardware/servo.py
=================
PCA9685 経由の SG90 サーボ制御。

依存: adafruit-circuitpython-pca9685, adafruit-blinka
インストール: pip3 install adafruit-circuitpython-pca9685 adafruit-blinka

【単体テスト】
  python3 -m hardware.servo
"""
import math
import time

from .config import (
    I2C_BUS, PCA9685_ADDRESS, PCA9685_FREQ_HZ,
    SG90_PULSE_MIN_US, SG90_PULSE_MAX_US,
    JOINT_CHANNELS, SERVO_DIRECTION, SERVO_OFFSET_DEG, JOINT_LIMITS_RAD,
)


def angle_to_duty(angle_deg: float, freq_hz: int = PCA9685_FREQ_HZ) -> int:
    """角度 (0-180°) を PCA9685 12bit デューティカウントに変換する。"""
    # パルス幅 (μs) を計算
    pulse_us = SG90_PULSE_MIN_US + (angle_deg / 180.0) * (SG90_PULSE_MAX_US - SG90_PULSE_MIN_US)
    # 1 周期 = 1_000_000 / freq_hz μs
    period_us = 1_000_000.0 / freq_hz
    # 12bit 分解能 (0–4095) でのデューティカウント
    return int(pulse_us / period_us * 4096)


class ServoController:
    """PCA9685 を介した全関節サーボの一元管理クラス。

    使い方:
        sc = ServoController()
        sc.set_joint_rad("body_to_coxa_fr_j", 0.3)      # 単関節
        sc.set_all_joints_rad({"body_to_coxa_fr_j": 0.3, ...})  # 全関節
        sc.set_neutral()                                  # 全関節を中立位置へ
    """

    def __init__(self):
        # adafruit ライブラリの遅延インポート (RPi4 以外でもモジュールを読める)
        import board
        import busio
        from adafruit_pca9685 import PCA9685

        i2c = busio.I2C(board.SCL, board.SDA)
        self._pca = PCA9685(i2c, address=PCA9685_ADDRESS)
        self._pca.frequency = PCA9685_FREQ_HZ
        print(f"[ServoController] PCA9685 初期化完了 (addr=0x{PCA9685_ADDRESS:02x}, {PCA9685_FREQ_HZ}Hz)")

    def _set_duty(self, channel: int, duty: int):
        """PCA9685 チャンネルにデューティカウントを設定する。"""
        # adafruit PCA9685 は duty_cycle を 0–65535 で受け取る
        self._pca.channels[channel].duty_cycle = int(duty * 65535 / 4095)

    def rad_to_servo_deg(self, joint_name: str, angle_rad: float) -> float:
        """関節角度 (rad) をサーボ物理角度 (°) に変換する。

        servo_deg = 90 + rad_to_deg(angle_rad) * direction + offset
        90° = サーボ中立位置 = 関節角度 0 rad
        """
        lo, hi = JOINT_LIMITS_RAD[joint_name]
        angle_clamped = max(lo, min(hi, angle_rad))   # ハードウェアリミット適用

        direction = SERVO_DIRECTION[joint_name]
        offset    = SERVO_OFFSET_DEG[joint_name]
        servo_deg = 90.0 + math.degrees(angle_clamped) * direction + offset
        # サーボ物理範囲: 0–180° にクランプ
        return max(0.0, min(180.0, servo_deg))

    def set_joint_rad(self, joint_name: str, angle_rad: float):
        """指定した関節を角度 (rad) に動かす。"""
        channel   = JOINT_CHANNELS[joint_name]
        servo_deg = self.rad_to_servo_deg(joint_name, angle_rad)
        duty      = angle_to_duty(servo_deg)
        self._set_duty(channel, duty)

    def set_all_joints_rad(self, joint_angles: dict):
        """全関節を一括で動かす。

        Args:
            joint_angles: {joint_name: angle_rad} の辞書
        """
        for joint_name, angle_rad in joint_angles.items():
            self.set_joint_rad(joint_name, angle_rad)

    def set_neutral(self):
        """全サーボを中立位置 (90°) へ移動する。"""
        for joint_name in JOINT_CHANNELS:
            self.set_joint_rad(joint_name, 0.0)
        print("[ServoController] 全サーボを中立位置に設定")

    def disable_all(self):
        """全チャンネルを無効化してトルクをゼロにする (非常停止)。"""
        for ch in range(16):
            self._pca.channels[ch].duty_cycle = 0
        print("[ServoController] 全サーボ無効化")

    def calibrate(self):
        """校正モード: 各サーボを順に 60°→90°→120° と動かして方向・オフセットを確認する。"""
        print("[校正] 各関節を 60°→90°→120° と動かします")
        for joint_name, channel in JOINT_CHANNELS.items():
            print(f"  {joint_name} (ch{channel}) ...")
            for servo_deg in [60.0, 90.0, 120.0]:
                self._set_duty(channel, angle_to_duty(servo_deg))
                time.sleep(0.8)
            # 中立へ戻す
            self._set_duty(channel, angle_to_duty(90.0))
            time.sleep(0.4)
        print("[校正] 完了")

    def __del__(self):
        try:
            self.disable_all()
            self._pca.deinit()
        except Exception:
            pass


# ─── 単体テスト ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== ServoController 単体テスト ===")
    sc = ServoController()
    sc.set_neutral()
    time.sleep(1.0)

    print("FR コクサを 0.3 rad (≈17°) に動かす")
    sc.set_joint_rad("body_to_coxa_fr_j", 0.3)
    time.sleep(1.0)

    print("FR フェマーを -0.2 rad (≈-11°) に動かす")
    sc.set_joint_rad("coxa_fr_to_femur_fr_j", -0.2)
    time.sleep(1.0)

    print("中立に戻す")
    sc.set_neutral()
    time.sleep(0.5)
    sc.disable_all()
