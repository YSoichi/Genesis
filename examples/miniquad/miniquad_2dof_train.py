"""
miniquad_2dof_train.py
======================
2DOF-hip 4脚ロボット (coxa x軸 + femur y軸) の強化学習スクリプト。

【実行方法】
  cd examples/miniquad
  python miniquad_2dof_train.py -e miniquad-2dof-v1 -B 16384 --max_iterations 5001

【v1 設計方針 (miniquad v15 の知見を活用)】
  v15 での重要発見:
    * entropy_coef=0.0  : PPO エントロピーボーナスを無効化 → noise_std 爆発を防ぐ
    * tracking_sigma=0.01: 指令速度スケールに合わせた急勾配
    * lin_vel_x_range=[0.05, 0.15]: 物理限界内の指令速度

  2DOF 固有の変更点:
    * coxa(x軸) + femur(y軸): 膝なしでも足先が浮く 2DOF-hip 構成
    * trot_bias_coxa_amp=0.25 rad: スイング中に内転 → 足先が 2-3mm 浮く
    * coxa_symmetry ペナルティ: 左右非対称な coxa 動作を抑制
    * base_height_target=0.080m (旧 0.113m: 膝なしで重心が低い)
    * 終了条件: roll/pitch 閾値を 1.0 rad に緩和 (低重心設計)

【GPU 最適化】
  RTX 4080 SUPER (16GB VRAM):
    推奨 num_envs = 16384 (8192 の 2 倍; VRAM ~14-15GB 想定)
    num_steps_per_env = 32 (旧 24 → サンプル効率向上; collection:learning ≈ 4:1 維持)
    → 理論 throughput = 16384 × 32 / iter_time ≈ 400K-500K steps/s (旧比 2倍+)

  VRAM が不足する場合 (OOM エラー時):
    -B 12288  → VRAM ~10-11GB
    -B 8192   → VRAM ~7-8GB (確実に動作する)

  モニタリング:
    nvidia-smi dmon -s mu -d 5  # VRAM・GPU利用率をリアルタイム表示
"""
import argparse
import os
import pickle
import shutil

from miniquad_2dof_env import miniquad2DOFEnv
from rsl_rl.runners import OnPolicyRunner
import genesis as gs


def get_train_cfg():
    return {
        "num_steps_per_env": 32,        # 旧 24 → 32: サンプル効率向上 (1 iter のデータ量を増やす)
        "save_interval": 100,
        "empirical_normalization": False,
        "algorithm": {
            "class_name": "PPO",
            "clip_param": 0.2,
            "desired_kl": 0.01,
            # entropy_coef=0.0: v15 実験で noise_std 爆発の根本原因と確認
            # エントロピーボーナスを無効化し action_rate ペナルティのみでノイズを制御
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
            "init_noise_std": 1.0,   # v15 で 0.25 → 爆発。1.0 が安定。
        },
        "seed": 1,
        "runner_class_name": "OnPolicyRunner",
    }


def get_cfgs():
    """
    2DOF-hip 用設定。

    【関節 DOF 順序】
      [0] coxa_fr  (x軸, 外転)   [1] femur_fr (y軸, 前後)
      [2] coxa_fl  (x軸, 外転)   [3] femur_fl (y軸, 前後)
      [4] coxa_rr  (x軸, 外転)   [5] femur_rr (y軸, 前後)
      [6] coxa_rl  (x軸, 外転)   [7] femur_rl (y軸, 前後)

    【デフォルト姿勢 (coxa=0, femur=0)】
      全関節 0 rad → 脚が直下に伸びた状態
      重心高さ: body_z(80mm) = body_height/2(10) + coxa(12) + femur(50) + toe(8)
    """
    env_cfg = {
        "num_actions": 8,

        "default_joint_angles": {
            "body_to_coxa_fr_j":      0.0,   # coxa: デフォルトは垂直 (傾きなし)
            "coxa_fr_to_femur_fr_j":  0.0,   # femur: デフォルトは直下
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

        "kp": 20.0,   # SG90 剛性: 旧 25.0 → 20.0 (2DOF は coxa 軸が柔らかい傾向)
        "kd":  0.5,

        # 終了条件: 低重心設計なので旧 0.8 rad → 1.0 rad に緩和
        "termination_if_roll_greater_than":  1.0,
        "termination_if_pitch_greater_than": 1.0,

        # 重心高さ: body_center = body_h/2(10mm) + coxa(12mm) + femur(50mm) + toe(8mm) = 80mm
        "base_init_pos":  [0.0, 0.0, 0.080],
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],

        "episode_length_s":   20.0,
        "resampling_time_s":   4.0,
        "action_scale":        0.5,
        "simulate_action_latency": True,
        "clip_actions":       10.0,   # v15 知見: clip=1.0 は noise 爆発を招く

        # 2DOF trot_bias: coxa(x軸)で足上げ + femur(y軸)で前後推進
        "use_trot_bias":       True,
        "trot_bias_freq":      1.0,   # [Hz] 1.0 Hz が安定 (2.5 Hz は転倒確認済み)
        "trot_bias_coxa_amp":  0.25,  # [rad] スイング時の内転角 (足先 ~3mm 浮き)
        "trot_bias_femur_amp": 0.20,  # [rad] 前後スイング幅
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
        # sigma=0.01: v15 で有効性を確認 (cmd=0.10 vs actual=0.04 で 48% 追従報酬)
        "tracking_sigma": 0.01,
        "base_height_target": 0.080,   # 2DOF 用: 旧 0.113 → 0.080m
        "reward_scales": {
            "tracking_lin_vel":  4.0,   # メイン速度追従 (sigma=0.01 で v14 の 50 倍勾配)
            "forward_vel":       2.0,   # 前進速度補助 (ゼロから前進発見を助ける)
            "alive":             0.5,   # 探索維持 (転倒回避動機)
            "lin_vel_z":        -0.5,   # 上下バウンシング抑制
            "base_height":      -1.0,   # 重心高さ維持 (2DOF は高さ変動が少ない)
            "action_rate":      -0.02,  # ノイズ成長抑制 (entropy_coef=0.0 と協調)
            "heading":          -0.2,   # 直進方向維持
            # coxa 対称ペナルティ: 左右の coxa が反対称になるよう誘導
            # 右脚が正方向に傾くなら左脚は負方向 → 体重が一方に偏るのを防ぐ
            "coxa_symmetry":    -0.5,
        },
    }

    command_cfg = {
        "num_commands": 3,
        # v15 で有効性を確認: ロボット物理限界 (~0.05 m/s) を含む現実的な範囲
        "lin_vel_x_range": [0.05, 0.15],
        "lin_vel_y_range": [0, 0],
        "ang_vel_range":   [0, 0],
    }

    return env_cfg, obs_cfg, reward_cfg, command_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name",      type=str, default="miniquad-2dof-v1")
    parser.add_argument("-B", "--num_envs",       type=int, default=16384,
                        help="並列環境数: RTX4080S(16GB) → 16384 推奨。OOM なら 12288 または 8192")
    parser.add_argument("--max_iterations",       type=int, default=5001)
    args = parser.parse_args()

    gs.init(logging_level="warning")

    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    train_cfg = get_train_cfg()

    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    env = miniquad2DOFEnv(
        num_envs=args.num_envs,
        env_cfg=env_cfg, obs_cfg=obs_cfg,
        reward_cfg=reward_cfg, command_cfg=command_cfg,
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device="cuda:0")
    pickle.dump(
        [env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg],
        open(f"{log_dir}/cfgs.pkl", "wb"),
    )

    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)


if __name__ == "__main__":
    main()
