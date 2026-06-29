"""
utils/logger.py
===============
軽量なロガー: コンソール表示 + CSV 記録。

CSV は後から Python/pandas で解析できる。
"""
import csv
import os
import time
from datetime import datetime
from pathlib import Path


class RobotLogger:
    """ロボット状態をコンソールと CSV に記録するクラス。

    使い方:
        logger = RobotLogger(log_dir="~/miniquad/logs")
        logger.log(step=0, imu_data=d, actions=a, target_dof=t)
        logger.close()
    """

    def __init__(self, log_dir: str = "~/miniquad/logs", enabled: bool = True):
        self._enabled = enabled
        if not enabled:
            return

        log_dir = Path(log_dir).expanduser()
        log_dir.mkdir(parents=True, exist_ok=True)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = log_dir / f"run_{ts}.csv"

        self._file   = open(csv_path, "w", newline="")
        self._writer = csv.writer(self._file)

        # ヘッダー行
        headers = (
            ["step", "time_s"]
            + [f"gyro_{a}" for a in "xyz"]
            + [f"grav_{a}"  for a in "xyz"]
            + [f"action_{i}"  for i in range(8)]
            + [f"target_{i}"  for i in range(8)]
            + ["yaw_deg", "vel_x", "vel_y", "vel_z"]
            + ["cal_sys", "cal_gyro", "cal_accel", "cal_mag"]
        )
        self._writer.writerow(headers)
        self._t0 = time.monotonic()
        print(f"[Logger] CSVログ開始: {csv_path}")

    def log(self, step: int, imu_data, actions, target_dof, vel_est=None):
        """1ステップ分のデータを記録する。"""
        if not self._enabled:
            return

        import math
        import numpy as np

        t_s = time.monotonic() - self._t0
        row = (
            [step, round(t_s, 4)]
            + list(imu_data.gyro_rad)
            + list(imu_data.gravity)
            + list(actions.tolist())
            + list(target_dof.tolist())
            + [
                round(math.degrees(imu_data.euler_rad[2]), 2),
                *(vel_est.tolist() if vel_est is not None else [0.0, 0.0, 0.0]),
            ]
            + list(imu_data.calibration)
        )
        self._writer.writerow(row)

    def print_status(self, step: int, imu_data, actions, target_dof, vel_est=None):
        """コンソールに 1行ステータスを表示する (50Hz では重すぎるので間引く)。"""
        import math
        yaw_deg  = math.degrees(imu_data.euler_rad[2])
        roll_deg = math.degrees(imu_data.euler_rad[0])
        pitch_deg= math.degrees(imu_data.euler_rad[1])
        cal = imu_data.calibration
        vx = vel_est[0] if vel_est is not None else 0.0
        print(
            f"\r[{step:5d}] "
            f"r={roll_deg:+5.1f}° p={pitch_deg:+5.1f}° y={yaw_deg:+6.1f}°"
            f"  vx={vx:+.3f}m/s"
            f"  cal={cal}",
            end="", flush=True,
        )

    def close(self):
        """ログファイルをクローズする。"""
        if self._enabled:
            self._file.close()
            print("\n[Logger] CSVログ終了")


class RobotLogger:
    """ロボット状態をコンソールと CSV に記録するクラス。"""

    def __init__(self, log_dir: str = "~/miniquad/logs", enabled: bool = True):
        self._enabled = enabled
        if not enabled:
            return

        log_dir = Path(log_dir).expanduser()
        log_dir.mkdir(parents=True, exist_ok=True)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = log_dir / f"run_{ts}.csv"

        self._file   = open(csv_path, "w", newline="")
        self._writer = csv.writer(self._file)
        self._writer.writerow(
            ["step", "time_s",
             "gyro_x", "gyro_y", "gyro_z",
             "grav_x", "grav_y", "grav_z",
             "a0","a1","a2","a3","a4","a5","a6","a7",
             "t0","t1","t2","t3","t4","t5","t6","t7",
             "yaw_deg", "vel_x", "vel_y", "vel_z",
             "cal_sys","cal_gyro","cal_accel","cal_mag"]
        )
        self._t0 = time.monotonic()
        print(f"[Logger] CSVログ開始: {csv_path}")

    def log(self, step: int, imu_data, actions, target_dof, vel_est=None):
        if not self._enabled:
            return
        import math
        t_s = time.monotonic() - self._t0
        vel = list(vel_est.tolist()) if vel_est is not None else [0.0, 0.0, 0.0]
        self._writer.writerow(
            [step, round(t_s, 4)]
            + list(imu_data.gyro_rad)
            + list(imu_data.gravity)
            + list(actions.tolist())
            + list(target_dof.tolist())
            + [round(__import__('math').degrees(imu_data.euler_rad[2]), 2)]
            + vel
            + list(imu_data.calibration)
        )

    def print_status(self, step: int, imu_data, vel_est=None):
        import math
        yaw  = math.degrees(imu_data.euler_rad[2])
        roll = math.degrees(imu_data.euler_rad[0])
        pit  = math.degrees(imu_data.euler_rad[1])
        vx   = vel_est[0] if vel_est is not None else 0.0
        print(
            f"\r[{step:5d}] roll={roll:+5.1f}° pit={pit:+5.1f}° yaw={yaw:+6.1f}°"
            f"  vx={vx:+.4f}m/s  cal={imu_data.calibration}",
            end="", flush=True,
        )

    def close(self):
        if self._enabled:
            self._file.close()
            print("\n[Logger] CSVログ終了")
