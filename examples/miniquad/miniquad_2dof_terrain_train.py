"""
miniquad_2dof_terrain_train.py
================================
2DOF-hip 4脚ロボットの地形対応強化学習スクリプト。

【地形対応の概要】
  フラット → 凸凹 → 傾斜 → 障害物 の順に難しくなる地形を生成。
  ロボットは平坦地点からスタートし、歩行速度が向上するにつれ
  自然に難しい地形に到達する (自動カリキュラム)。

【v1 からの改善点】
  1. coxa_symmetry ペナルティを削除
     - v1 でこのペナルティが trot_bias の正しい非対称 coxa 動作を妨げていた
     - FR/FL は位相が逆なので coxa_fr + coxa_fl ≠ 0 が正常動作
     → 削除により前進速度の向上を期待

  2. trot_bias 振幅を拡大
     - coxa_amp: 0.25 → 0.35 rad (足上げ高さ ≈ 3mm → 4mm)
     - femur_amp: 0.20 → 0.30 rad (1ストライド幅 ≈ 30mm → 45mm)

  3. kp: 20 → 25 (剛性回復: 旧 miniquad と同じ設定)

  4. orientation 報酬を追加
     - 傾斜地形でも体を水平に保つことを促進
     - projected_gravity の xy 成分最小化

  5. base_init_pos: [0.0, 0.0, 0.080] → [0.5, 2.0, 0.080]
     - 地形の平坦ゾーン (x=0-2m) の中央に配置
     - y=2.0: 4m 幅の地形の中央

【実行方法】
  # フルスタート (v2 初回)
  cd examples/miniquad
  PYTHONPATH=/home/mutsumi/rsl_rl:$PYTHONPATH python3 miniquad_2dof_terrain_train.py \\
    -e miniquad-2dof-v2-terrain -B 16384 --max_iterations 5001 \\
    > /tmp/miniquad_terrain_train.log 2>&1 &

  # v1 チェックポイントから継続学習 (転移学習)
  PYTHONPATH=/home/mutsumi/rsl_rl:$PYTHONPATH python3 miniquad_2dof_terrain_train.py \\
    -e miniquad-2dof-v2-terrain -B 16384 --max_iterations 5001 \\
    --load_run miniquad-2dof-v1 --load_checkpoint 5000 \\
    > /tmp/miniquad_terrain_train.log 2>&1 &
"""
import argparse
import os
import pickle
import shutil

from miniquad_2dof_terrain_env import miniquad2DOFTerrainEnv
from rsl_rl.runners import OnPolicyRunner
import genesis as gs


def get_train_cfg():
    return {
        "num_steps_per_env": 32,
        "save_interval": 100,
        "empirical_normalization": False,
        "algorithm": {
            "class_name": "PPO",
            "clip_param": 0.2,
            "desired_kl": 0.01,
            # entropy_coef=0.0: v15/v1 で noise_std 爆発の根本原因と確認
            "entropy_coef": 0.0,
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 0.001,
            "max_grad_norm": 1.0,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
        },
        "policy": {
            "class_name": "ActorCritic",
            "activation": "elu",
            "actor_hidden_dims": [256, 128, 64],
            "critic_hidden_dims": [256, 128, 64],
            "init_noise_std": 1.0,
        },
        "seed": 1,
        "runner_class_name": "OnPolicyRunner",
    }


def get_cfgs():
    """
    2DOF-hip 地形対応設定。

    【地形スタート位置】
      地形レイアウト:
        x=0〜2m: flat (ロボット開始ゾーン)
        x=2〜4m: 小凸凹
        x=4〜6m: 傾斜・波状
        x=6〜8m: 障害物・階段
      base_init_pos: [0.5, 2.0, 0.080] = 平坦ゾーン中央

    【報酬設計】
      - coxa_symmetry 削除: v1 の前進速度低下の主因と判断
      - orientation 追加: 傾斜適応
    """
    env_cfg = {
        "num_actions": 8,

        "default_joint_angles": {
            "body_to_coxa_fr_j":      0.0,
            "coxa_fr_to_femur_fr_j":  0.0,
            "body_to_coxa_fl_j":      0.0,
            "coxa_fl_to_femur_fl_j":  0.0,
            "body_to_coxa_rr_j":      0.0,
            "coxa_rr_to_femur_rr_j":  0.0,
            "body_to_coxa_rl_j":      0.0,
            "coxa_rl_to_femur_rl_j":  0.0,
        },

        "dof_names": [
            "body_to_coxa_fr_j",     "coxa_fr_to_femur_fr_j",
            "body_to_coxa_fl_j",     "coxa_fl_to_femur_fl_j",
            "body_to_coxa_rr_j",     "coxa_rr_to_femur_rr_j",
            "body_to_coxa_rl_j",     "coxa_rl_to_femur_rl_j",
        ],

        # kp: 20 → 25 (旧 miniquad と同じ。v1 では 20 で歩幅が出なかった可能性)
        "kp": 25.0,
        "kd":  0.5,

        # 傾斜地形では roll/pitch が大きくなるため v1 より緩和
        "termination_if_roll_greater_than":  1.2,
        "termination_if_pitch_greater_than": 1.2,

        # 地形の平坦ゾーン (x=0-2m) 内に配置; y=2.0 は 4m 幅の中央
        "base_init_pos":  [0.5, 2.0, 0.080],
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],

        "episode_length_s":   20.0,
        "resampling_time_s":   4.0,
        "action_scale":        0.5,
        "simulate_action_latency": True,
        "clip_actions":       10.0,

        # trot_bias v2: 振幅拡大
        "use_trot_bias":       True,
        "trot_bias_freq":      1.0,
        "trot_bias_coxa_amp":  0.35,  # v1: 0.25 → v2: 0.35 (足上げ強化)
        "trot_bias_femur_amp": 0.30,  # v1: 0.20 → v2: 0.30 (ストライド拡大)

        # 地形パラメータ
        "terrain_horizontal_scale": 0.05,   # 5cm 解像度
        "terrain_vertical_scale":   0.005,  # 5mm/unit

        # domain randomization
        "randomize_friction": True,   # 各エピソードで摩擦係数をランダム化

        # 外乱 (push): 5秒ごとにランダム関節速度を加算 (体への間接的な外乱)
        "push_interval_s": 5.0,
        "push_vel_max":    0.3,       # [rad/s] 各関節に加算するランダム速度の最大値
    }

    obs_cfg = {
        # 38 = ang_vel(3) + gravity(3) + cmd(3) + lin_vel(3) + dof_pos(8) + dof_vel(8) + actions(8) + heading(2)
        "num_obs": 38,
        "obs_scales": {
            "lin_vel": 2.0,
            "ang_vel": 0.25,
            "dof_pos": 1.0,
            "dof_vel": 0.05,
        },
    }

    reward_cfg = {
        "tracking_sigma":    0.01,
        "base_height_target": 0.080,
        "reward_scales": {
            "tracking_lin_vel":  4.0,   # メイン速度追従
            "forward_vel":       2.0,   # 前進補助 (v2 も維持)
            "alive":             0.5,   # 生存ボーナス
            "lin_vel_z":        -0.5,   # 上下バウンシング抑制
            "base_height":      -0.5,   # 重心高さ維持 (傾斜地形では自然な高さ変動があるため v1 より緩和)
            "action_rate":      -0.02,  # ノイズ抑制
            "heading":          -0.2,   # 直進維持
            # v2 追加: 体の水平維持 (傾斜適応)
            # coxa_symmetry は v1 の問題原因として削除
            "orientation":      -0.3,
        },
    }

    command_cfg = {
        "num_commands": 3,
        "lin_vel_x_range": [0.05, 0.15],
        "lin_vel_y_range": [0, 0],
        "ang_vel_range":   [0, 0],
    }

    return env_cfg, obs_cfg, reward_cfg, command_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name",         type=str, default="miniquad-2dof-v2-terrain")
    parser.add_argument("-B", "--num_envs",          type=int, default=16384)
    parser.add_argument("--max_iterations",          type=int, default=5001)
    parser.add_argument("--load_run",                type=str, default=None,
                        help="継続学習元の実験名 (例: miniquad-2dof-v1)")
    parser.add_argument("--load_checkpoint",         type=int, default=-1,
                        help="読み込むチェックポイント番号 (-1 = 最新)")
    args = parser.parse_args()

    gs.init(logging_level="warning")

    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    train_cfg = get_train_cfg()

    # 新規実験の場合は古いログを削除
    if args.load_run is None:
        if os.path.exists(log_dir):
            shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    env = miniquad2DOFTerrainEnv(
        num_envs=args.num_envs,
        env_cfg=env_cfg, obs_cfg=obs_cfg,
        reward_cfg=reward_cfg, command_cfg=command_cfg,
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device="cuda:0")
    pickle.dump(
        [env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg],
        open(f"{log_dir}/cfgs.pkl", "wb"),
    )

    # 継続学習: v1 チェックポイントから重みをロード
    if args.load_run is not None:
        load_path = f"logs/{args.load_run}"
        if args.load_checkpoint == -1:
            # 最新チェックポイントを自動検出
            import glob
            ckpts = sorted(glob.glob(f"{load_path}/model_*.pt"))
            if ckpts:
                resume_path = ckpts[-1]
                print(f"[INFO] 継続学習: {resume_path} からロード")
                runner.load(resume_path)
            else:
                print(f"[WARN] チェックポイントが見つかりません: {load_path}")
        else:
            resume_path = f"{load_path}/model_{args.load_checkpoint}.pt"
            print(f"[INFO] 継続学習: {resume_path} からロード")
            runner.load(resume_path)

    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)


if __name__ == "__main__":
    main()
