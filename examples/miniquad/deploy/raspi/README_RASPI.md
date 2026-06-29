# miniquad RPi4 セットアップガイド

## OS 推奨

| OS | Python | I2C | 推奨度 | 備考 |
|----|--------|-----|--------|------|
| Ubuntu 20.04 LTS | 3.8 | ○ | ✗ | **EOL (2025/4)** |
| **Ubuntu 22.04 LTS 64bit** | 3.10 | ○ | **◎** | バランス良好 |
| Ubuntu 24.04 LTS 64bit | 3.12 | ○ | ○ | 最新だが一部パッケージ未対応 |
| RPi OS Bookworm Lite 64bit | 3.11 | ◎ | ○ | ハードウェア互換性最高 |

### 推奨: Ubuntu 22.04 LTS 64bit Server

Ubuntu 20.04 は **2025年4月で EOL** のため、**再構築を推奨**する。

#### インストール手順

```bash
# 1. Raspberry Pi Imager で Ubuntu 22.04 LTS Server (64bit) を書き込む
#    https://ubuntu.com/download/raspberry-pi

# 2. 初回起動後 (SSH でログイン)
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3-pip python3-venv i2c-tools git

# 3. I2C 有効化
sudo raspi-config   # または /boot/firmware/config.txt に dtparam=i2c_arm=on を追記
# Ubuntu 22.04 の場合:
echo "dtparam=i2c_arm=on" | sudo tee -a /boot/firmware/config.txt
sudo reboot

# 4. I2C デバイス確認 (再起動後)
sudo i2cdetect -y 1
# 0x28: BNO055, 0x40: PCA9685 が表示されれば OK
```

## miniquad セットアップ

```bash
# 1. Python 仮想環境作成
python3 -m venv ~/miniquad/venv
source ~/miniquad/venv/bin/activate

# 2. 依存パッケージインストール
pip install -r requirements.txt

# 3. PC からモデルをコピー (PC 側で実行)
#    まず PC で ONNX エクスポート:
cd ~/Genesis/examples/miniquad
PYTHONPATH=/home/mutsumi/rsl_rl:$PYTHONPATH python3 deploy/export_onnx.py \
  --run miniquad-2dof-v4-terrain --checkpoint 2800

#    RPi4 へコピー:
scp logs/miniquad-2dof-v4-terrain/policy.onnx      pi@<raspi_ip>:~/miniquad/
scp logs/miniquad-2dof-v4-terrain/policy_meta.json  pi@<raspi_ip>:~/miniquad/

# 4. Dry-run テスト (ハードウェアなし)
cd ~/miniquad/raspi
python3 miniquad_main.py --dry-run

# 5. ハードウェアテスト (IMU → サーボの順で確認)
python3 -m hardware.imu    # BNO055 確認 (10秒間データ表示)
python3 -m hardware.servo  # SG90 確認 (各サーボを順に動かす)

# 6. 本番起動
python3 miniquad_main.py --model ~/miniquad/policy.onnx --vx 0.08
```

## ファイル構成

```
raspi/
├── miniquad_main.py          # メイン制御ループ (ここから実行)
├── requirements.txt          # pip 依存
├── hardware/
│   ├── config.py             # ★ サーボ校正・I2C設定 (要変更)
│   ├── servo.py              # PCA9685 ドライバ
│   └── imu.py                # BNO055 ドライバ
├── controller/
│   ├── gait.py               # trot バイアス計算
│   ├── state.py              # 38次元観測ベクトル構築
│   └── inference.py          # ONNX 推論
└── utils/
    └── logger.py             # CSV/コンソールログ
```

## サーボ校正

`hardware/config.py` の `SERVO_OFFSET_DEG` と `SERVO_DIRECTION` を調整する。

```bash
# 各サーボを 60°→90°→120° と動かして方向確認
python3 -c "
from hardware.servo import ServoController
sc = ServoController()
sc.calibrate()
"
```

## 観測ベクトル (38次元)

| インデックス | 内容 | スケール |
|------------|------|--------|
| 0–2 | 角速度 (rad/s) body frame | ×0.25 |
| 3–5 | 重力ベクトル (normalized) body frame | ×1.0 |
| 6–8 | 速度コマンド [vx, vy, yaw] | ×[2, 2, 0.25] |
| 9–11 | 線形速度推定 (m/s) | ×2.0 |
| 12–19 | 関節角度 − デフォルト (rad) | ×1.0 |
| 20–27 | 関節角速度 (rad/s) | ×0.05 |
| 28–35 | 前ステップのアクション | ×1.0 |
| 36–37 | ヘディング [cos(yaw), sin(yaw)] | ×1.0 |

## 拡張ポイント

- **速度コマンド変更**: `CommandManager.set(vx, vy, yaw)` を呼ぶ
- **ジョイスティック**: `CommandManager` に `inputs` ライブラリ連携を追加
- **ROS2 連携**: `CommandManager` に `rclpy` サブスクライバーを追加
- **転倒回復**: `FallDetector` が `True` を返したときの回復モーション追加
- **遠隔監視**: `RobotLogger` に WebSocket 送信を追加
