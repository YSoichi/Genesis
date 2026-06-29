"""
hardware/imu.py
===============
BNO055 9DOF IMU の読み取りクラス。

BNO055 NDOF モード使用:
  - 加速度 + ジャイロ + 地磁気のフュージョン
  - 出力: Euler角, 角速度, 重力ベクトル, 線形加速度

依存: adafruit-circuitpython-bno055
インストール: pip3 install adafruit-circuitpython-bno055

【座標系の注意】
  BNO055 のデフォルト軸とロボット軸が一致するように IMU_AXIS_MAP で補正する。
  シミュレーション座標: x=前方, y=左方, z=上方

【単体テスト】
  python3 -m hardware.imu
"""
import math
import time
from dataclasses import dataclass, field
from typing import Tuple

import numpy as np

from .config import BNO055_ADDRESS, IMU_AXIS_MAP


@dataclass
class IMUData:
    """IMU から読み取った全データをまとめる構造体。"""
    # 角速度 (rad/s) in body frame
    gyro_rad: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    # 重力ベクトル (単位ベクトル) in body frame
    gravity:  Tuple[float, float, float] = (0.0, 0.0, -1.0)
    # Euler 角 (rad): roll, pitch, yaw
    euler_rad: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    # 線形加速度 (m/s²) in body frame (重力除去済み)
    lin_accel: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    # キャリブレーション状態 (0–3): sys, gyro, accel, mag
    calibration: Tuple[int, int, int, int] = (0, 0, 0, 0)
    # 有効フラグ (フュージョンが収束しているか)
    is_valid: bool = False


class IMUReader:
    """BNO055 から定期的に IMU データを読み取るクラス。

    使い方:
        imu = IMUReader()
        imu.wait_for_calibration()   # 初回校正待ち
        data = imu.read()
        print(data.gyro_rad, data.gravity)
    """

    def __init__(self):
        import board
        import busio
        import adafruit_bno055

        i2c = busio.I2C(board.SCL, board.SDA)
        self._sensor = adafruit_bno055.BNO055_I2C(i2c, address=BNO055_ADDRESS)
        # NDOF モード: 加速度+ジャイロ+地磁気フュージョン
        self._sensor.mode = adafruit_bno055.NDOF_MODE
        print(f"[IMUReader] BNO055 初期化完了 (addr=0x{BNO055_ADDRESS:02x}, NDOF モード)")

        self._axis_map = IMU_AXIS_MAP
        self._last_data = IMUData()

    def _apply_axis_map(self, vec: tuple) -> tuple:
        """BNO055 の物理軸をロボット座標系に変換する。"""
        fwd  = self._axis_map["forward_axis"]
        left = self._axis_map["left_axis"]
        up   = self._axis_map["up_axis"]
        sx   = self._axis_map["x_sign"]
        sy   = self._axis_map["y_sign"]
        sz   = self._axis_map["z_sign"]
        return (
            vec[fwd]  * sx,
            vec[left] * sy,
            vec[up]   * sz,
        )

    def read(self) -> IMUData:
        """BNO055 から最新データを読み取って IMUData を返す。"""
        try:
            # 角速度 (°/s → rad/s)
            gyro_dps = self._sensor.gyro or (0.0, 0.0, 0.0)
            gyro_rad = tuple(math.radians(v) for v in gyro_dps)
            gyro_rad = self._apply_axis_map(gyro_rad)

            # 重力ベクトル (m/s²) → 正規化して単位ベクトルにする
            grav_ms2 = self._sensor.gravity or (0.0, 0.0, -9.81)
            grav_ms2 = self._apply_axis_map(grav_ms2)
            norm = math.sqrt(sum(v * v for v in grav_ms2)) or 1.0
            gravity = tuple(v / norm for v in grav_ms2)

            # Euler 角 (°): BNO055 は (heading, roll, pitch)
            euler_deg = self._sensor.euler or (0.0, 0.0, 0.0)
            # BNO055 Euler: heading=yaw, roll, pitch の順
            yaw_rad   = math.radians(euler_deg[0] or 0.0)
            roll_rad  = math.radians(euler_deg[1] or 0.0)
            pitch_rad = math.radians(euler_deg[2] or 0.0)

            # 線形加速度 (重力除去済み, m/s²)
            lin_acc = self._sensor.linear_acceleration or (0.0, 0.0, 0.0)
            lin_acc = self._apply_axis_map(lin_acc)

            # キャリブレーション状態
            cal = self._sensor.calibration_status or (0, 0, 0, 0)

            # sys >= 1 ならフュージョン有効
            is_valid = cal[0] >= 1

            data = IMUData(
                gyro_rad   = gyro_rad,
                gravity    = gravity,
                euler_rad  = (roll_rad, pitch_rad, yaw_rad),
                lin_accel  = lin_acc,
                calibration= cal,
                is_valid   = is_valid,
            )
            self._last_data = data
            return data

        except Exception as e:
            print(f"[IMUReader] 読み取りエラー: {e}")
            return self._last_data   # 前回値を返す (フォールバック)

    def wait_for_calibration(self, min_sys: int = 1, timeout_s: float = 60.0):
        """フュージョンが有効になるまで待機する。

        Args:
            min_sys: 最低限必要な sys キャリブレーションレベル (0–3)
            timeout_s: タイムアウト秒数
        """
        print(f"[IMUReader] キャリブレーション待機中 (sys >= {min_sys})...")
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            data = self.read()
            sys, gyro, accel, mag = data.calibration
            print(f"\r  sys={sys} gyro={gyro} accel={accel} mag={mag}", end="", flush=True)
            if sys >= min_sys:
                print(f"\n[IMUReader] キャリブレーション完了 (sys={sys})")
                return True
            time.sleep(0.5)
        print(f"\n[IMUReader] キャリブレーションタイムアウト (続行します)")
        return False

    @property
    def last(self) -> IMUData:
        """最後に読み取ったデータを返す (再読み取りなし)。"""
        return self._last_data


# ─── 単体テスト ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== IMUReader 単体テスト ===")
    imu = IMUReader()
    imu.wait_for_calibration(min_sys=1)

    print("10秒間データを読み取ります...")
    for i in range(100):
        d = imu.read()
        print(
            f"  gyro=({d.gyro_rad[0]:+.3f},{d.gyro_rad[1]:+.3f},{d.gyro_rad[2]:+.3f}) rad/s"
            f"  grav=({d.gravity[0]:+.3f},{d.gravity[1]:+.3f},{d.gravity[2]:+.3f})"
            f"  yaw={math.degrees(d.euler_rad[2]):+.1f}°"
            f"  cal={d.calibration}"
        )
        time.sleep(0.1)
