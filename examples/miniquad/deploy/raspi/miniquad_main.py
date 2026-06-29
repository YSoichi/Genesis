"""
miniquad_main.py
================
miniquad 4脚ロボット メインコントローラー (Raspberry Pi 4)

【必要なハードウェア】
  - PCA9685  (I2C: 0x40) ─ SG90 × 8 制御
  - BNO055   (I2C: 0x28) ─ 姿勢・角速度センシング
  - SG90 × 8             ─ 各脚 coxa(x軸) + femur(y軸)

【実行方法】
  cd ~/miniquad/raspi
  python3 miniquad_main.py --model ~/miniquad/policy.onnx

  オプション:
    --model  PATH     ONNX ファイルパス
    --vx     FLOAT    目標前進速度 m/s (デフォルト 0.10)
    --no-log          CSVログを無効化
    --dry-run         ハードウェアなし動作確認モード

【キー操作】(Ctrl+C で停止)
  起動後は Ctrl+C を押すまで自律走行を続ける。
  緊急停止は Ctrl+C → サーボが無効化される。

【拡張ポイント】
  ・速度コマンドの変更: CommandManager クラスで管理
    (キーボード/ジョイスティック/TCP等を追加可能)
  ・転倒検知: FallDetector クラスに検知ロジックを追加
  ・ステートマシン: RobotFSM クラスで状態遷移を管理
"""
import argparse
import math
import signal
import sys
import time
from pathlib import Path

import numpy as np


# ─── 転倒検知 ─────────────────────────────────────────────────────────────
class FallDetector:
    """IMU ロール・ピッチが閾値を超えたら転倒と判定する。

    拡張例: 連続 N ステップで超えた場合のみ転倒とする等
    """

    def __init__(self, roll_limit_deg: float = 50.0, pitch_limit_deg: float = 50.0):
        self._roll_lim  = math.radians(roll_limit_deg)
        self._pitch_lim = math.radians(pitch_limit_deg)

    def is_fallen(self, imu_data) -> bool:
        roll  = abs(imu_data.euler_rad[0])
        pitch = abs(imu_data.euler_rad[1])
        return roll > self._roll_lim or pitch > self._pitch_lim


# ─── 速度コマンド管理 ─────────────────────────────────────────────────────
class CommandManager:
    """ロボットへの速度コマンドを管理するクラス。

    拡張例:
      - キーボードスキャン (curses)
      - ジョイスティック入力 (inputs ライブラリ)
      - TCP ソケット受信
      - ROS2 サブスクライバー
    """

    def __init__(self, vx: float = 0.10):
        self._cmd = [float(vx), 0.0, 0.0]   # [vx, vy, yaw_rate]

    def get(self) -> list:
        """現在のコマンドを返す。"""
        return self._cmd.copy()

    def set(self, vx: float = 0.0, vy: float = 0.0, yaw: float = 0.0):
        """コマンドを更新する。"""
        self._cmd = [vx, vy, yaw]

    def stop(self):
        """停止コマンド。"""
        self.set(0.0, 0.0, 0.0)


# ─── メイン制御ループ ──────────────────────────────────────────────────────
class MiniquadController:
    """miniquad の制御ループ全体を管理するクラス。

    使い方:
        ctrl = MiniquadController(args)
        ctrl.run()
    """

    def __init__(self, args):
        self._args    = args
        self._running = False

        # ── ポリシー読み込み ───────────────────────────────────────────
        print("[Main] ポリシーを読み込んでいます...")
        from controller.inference import PolicyRunner
        self._policy = PolicyRunner(args.model)
        self._policy.benchmark(n_runs=50)

        # ── 状態推定器 ────────────────────────────────────────────────
        from controller.state import RobotStateEstimator
        from hardware.config  import CONTROL_DT
        self._state   = RobotStateEstimator(self._policy.meta, dt=CONTROL_DT)
        self._dt      = CONTROL_DT
        self._cmd_mgr = CommandManager(vx=args.vx)
        self._fall    = FallDetector()

        # ── ハードウェア初期化 ────────────────────────────────────────
        if not args.dry_run:
            from hardware.servo import ServoController
            from hardware.imu   import IMUReader
            self._servo = ServoController()
            self._imu   = IMUReader()
            self._imu.wait_for_calibration(min_sys=1, timeout_s=30.0)
            self._servo.set_neutral()
            time.sleep(0.5)
        else:
            print("[Main] DRY-RUN モード: ハードウェアなし")
            self._servo = None
            self._imu   = None

        # ── ロガー ────────────────────────────────────────────────────
        from utils.logger import RobotLogger
        log_enabled = not args.no_log
        self._logger = RobotLogger(log_dir="~/miniquad/logs", enabled=log_enabled)

        # Ctrl+C ハンドラ
        signal.signal(signal.SIGINT, self._on_sigint)

    def _on_sigint(self, sig, frame):
        print("\n[Main] 停止シグナル受信")
        self._running = False

    def _read_imu(self):
        """IMU データを読み取る (dry-run 時はダミーを返す)。"""
        if self._imu is not None:
            return self._imu.read()
        # ダミー IMU データ
        from hardware.imu import IMUData
        return IMUData(
            gyro_rad   = (0.0, 0.0, 0.0),
            gravity    = (0.0, 0.0, -1.0),
            euler_rad  = (0.0, 0.0, 0.0),
            lin_accel  = (0.0, 0.0, 0.0),
            calibration= (3, 3, 3, 3),
            is_valid   = True,
        )

    def _send_to_servos(self, joint_angles: dict):
        """サーボへコマンドを送信する (dry-run 時はスキップ)。"""
        if self._servo is not None:
            self._servo.set_all_joints_rad(joint_angles)

    def run(self):
        """メイン制御ループ (50Hz)。"""
        from hardware.config import CONTROL_DT
        self._state.reset()
        self._running = True
        step = 0
        last_actions = np.zeros(8, dtype=np.float32)

        print("[Main] 制御ループ開始 (Ctrl+C で停止)")
        print(f"  目標速度: vx={self._args.vx:.2f} m/s")

        loop_t = time.monotonic()

        while self._running:
            t_start = time.monotonic()

            # ── IMU 読み取り ──────────────────────────────────────────
            imu_data = self._read_imu()

            # ── 転倒検知 ──────────────────────────────────────────────
            if self._fall.is_fallen(imu_data):
                print(f"\n[Main] 転倒検知 (step={step}): 停止します")
                break

            # ── 観測ベクトル構築 ──────────────────────────────────────
            cmd = self._cmd_mgr.get()
            obs = self._state.update(imu_data, last_actions, command=cmd)

            # ── ポリシー推論 ──────────────────────────────────────────
            raw_actions = self._policy.infer(obs)

            # ── アクション → 目標関節角度 ─────────────────────────────
            target_dof_pos = self._state.apply_actions(raw_actions)

            # ── サーボ送信 ────────────────────────────────────────────
            joint_angles = {
                name: float(target_dof_pos[i])
                for i, name in enumerate(self._state.dof_names)
            }
            self._send_to_servos(joint_angles)

            # ── ログ・表示 ────────────────────────────────────────────
            vel = self._state._vel_est.velocity
            self._logger.log(step, imu_data, raw_actions, target_dof_pos, vel)
            if step % 50 == 0:   # 1秒ごとに表示
                self._logger.print_status(step, imu_data, vel)

            last_actions = raw_actions
            step += 1

            # ── 50Hz ループ維持 ───────────────────────────────────────
            elapsed = time.monotonic() - t_start
            sleep_t = CONTROL_DT - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)
            elif step % 100 == 0:
                print(f"\n[Main] 警告: ループ超過 {elapsed*1000:.1f}ms (budget {CONTROL_DT*1000:.0f}ms)")

        # ── 終了処理 ──────────────────────────────────────────────────
        self._shutdown()

    def _shutdown(self):
        """安全な終了処理。"""
        print("\n[Main] シャットダウン中...")
        if self._servo is not None:
            self._servo.set_neutral()
            time.sleep(0.3)
            self._servo.disable_all()
        self._logger.close()
        print("[Main] 終了")


# ─── エントリポイント ──────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="miniquad 4脚ロボット コントローラー (RPi4)"
    )
    parser.add_argument(
        "--model", type=str, default="~/miniquad/policy.onnx",
        help="ONNX ポリシーファイルのパス"
    )
    parser.add_argument(
        "--vx", type=float, default=0.10,
        help="目標前進速度 (m/s)"
    )
    parser.add_argument(
        "--no-log", action="store_true",
        help="CSV ログを無効化する"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="ハードウェアなしで動作確認する"
    )
    args = parser.parse_args()

    args.model = str(Path(args.model).expanduser())

    ctrl = MiniquadController(args)
    ctrl.run()


if __name__ == "__main__":
    main()
