"""
miniquad_2dof_terrain_v3_train.py
===================================
2DOF-hip 4脚ロボット 地形対応 v3 訓練スクリプト。

【v2 → v3 の主な変更点】

  問題: v2 は 1Hz trot で物理最大速度 ≈ 0.059 m/s に対し実効率 17% (0.010 m/s)。
        プラトーが iter 500 付近で発生し、以降は noise_std = 0.02 で停滞した。

  改善 1: trot_bias_freq 1.0 → 2.0 Hz
    - 1 サイクル中のストライド回数が 2 倍 → 理論速度上限 2 倍 (≈ 0.118 m/s)
    - femur_amp は 0.30 → 0.25 rad に抑え、2Hz での安定性を確保

  改善 2: action_rate ペナルティ緩和 (-0.02 → -0.005)
    - v2 では action_rate が noise_std を 0.02 まで急速に縮小させた
    - ポリシーが trot_bias をより積極的に補正できるようにする

  改善 3: trot_bias_coxa_amp 0.35 → 0.30 rad
    - 2Hz ではスイング時間が 0.25s に短縮; 足上げは少し小さくして安定させる

  改善 4: tracking_sigma 0.01 → 0.02
    - 速度追従の勾配を若干緩やかにし、policy の探索自由度を高める

  改善 5: lin_vel_x_range [0.05, 0.15] → [0.05, 0.12]
    - 2Hz trot の理論最大 0.118 m/s に合わせ、コマンド上限を調整

  不変:
    - entropy_coef = 0.0 (必須: ノイズ爆発防止)
    - coxa_symmetry なし (v2 で削除済み、v1 の失敗原因)
    - 地形: flat → bumpy → sloped → obstacles (自動カリキュラム)
    - domain randomization: 摩擦ランダム化

【実行方法】
  cd examples/miniquad
  PYTHONPATH=/home/mutsumi/rsl_rl:$PYTHONPATH python3 miniquad_2dof_terrain_v3_train.py \\
    -e miniquad-2dof-v3-terrain -B 16384 --max_iterations 5001 \\
    > /tmp/miniquad_terrain_v3_train.log 2>&1 &

  # v2 model_500 から継続学習 (転移学習)
  PYTHONPATH=/home/mutsumi/rsl_rl:$PYTHONPATH python3 miniquad_2dof_terrain_v3_train.py \\
    -e miniquad-2dof-v3-terrain -B 16384 --max_iterations 5001 \\
    --load_run miniquad-2dof-v2-terrain --load_checkpoint 500 \\
    > /tmp/miniquad_terrain_v3_train.log 2>&1 &
"""
import argparse
import os
import pickle
import shutil
import glob

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
            "entropy_coef": 0.0,   # v2と同じ: noise_std爆発防止
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
    2DOF-hip 地形対応 v3 設定。
    2Hz trot + 緩和した action_rate で速度向上を目指す。
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

        "kp": 25.0,
        "kd":  0.5,

        "termination_if_roll_greater_than":  1.2,
        "termination_if_pitch_greater_than": 1.2,

        "base_init_pos":  [0.5, 2.0, 0.080],
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],

        "episode_length_s":   20.0,
        "resampling_time_s":   4.0,
        "action_scale":        0.5,
        "simulate_action_latency": True,
        "clip_actions":       10.0,

        # v3: trot 周波数 2倍 (1Hz → 2Hz) で速度上限を 2 倍に
        "use_trot_bias":       True,
        "trot_bias_freq":      2.0,   # v2:1.0 → v3:2.0 (2倍速ストライド)
        "trot_bias_coxa_amp":  0.30,  # v2:0.35 → v3:0.30 (2Hz安定化のため若干低減)
        "trot_bias_femur_amp": 0.25,  # v2:0.30 → v3:0.25 (同上)

        # 地形パラメータ (v2 と同一)
        "terrain_horizontal_scale": 0.05,
        "terrain_vertical_scale":   0.005,
        "randomize_friction": True,
        "push_interval_s": 5.0,
        "push_vel_max":    0.3,
    }

    obs_cfg = {
        "num_obs": 38,
        "obs_scales": {
            "lin_vel": 2.0,
            "ang_vel": 0.25,
            "dof_pos": 1.0,
            "dof_vel": 0.05,
        },
    }

    reward_cfg = {
        # tracking_sigma: 0.01 → 0.02 (追従要求を緩め、探索を促進)
        "tracking_sigma":    0.02,
        "base_height_target": 0.080,
        "reward_scales": {
            "tracking_lin_vel":  4.0,
            "forward_vel":       2.0,
            "alive":             0.5,
            "lin_vel_z":        -0.5,
            "base_height":      -0.5,
            # action_rate: -0.02 → -0.005 (探索自由度向上; noise_std 崩壊を遅らせる)
            "action_rate":      -0.005,
            "heading":          -0.2,
            "orientation":      -0.3,
        },
    }

    command_cfg = {
        "num_commands": 3,
        # 2Hz trot 理論上限 ≈ 0.118 m/s に合わせて上限調整
        "lin_vel_x_range": [0.05, 0.12],
        "lin_vel_y_range": [0, 0],
        "ang_vel_range":   [0, 0],
    }

    return env_cfg, obs_cfg, reward_cfg, command_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name",         type=str, default="miniquad-2dof-v3-terrain")
    parser.add_argument("-B", "--num_envs",          type=int, default=16384)
    parser.add_argument("--max_iterations",          type=int, default=5001)
    parser.add_argument("--load_run",                type=str, default=None,
                        help="継続学習元の実験名 (例: miniquad-2dof-v2-terrain)")
    parser.add_argument("--load_checkpoint",         type=int, default=-1,
                        help="読み込むチェックポイント番号 (-1 = 最新)")
    args = parser.parse_args()

    gs.init(logging_level="warning")

    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    train_cfg = get_train_cfg()

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

    if args.load_run is not None:
        load_path = f"logs/{args.load_run}"
        if args.load_checkpoint == -1:
            ckpts = sorted(glob.glob(f"{load_path}/model_*.pt"),
                           key=lambda p: int(p.split("model_")[-1].replace(".pt", "")))
            resume_path = ckpts[-1] if ckpts else None
        else:
            resume_path = f"{load_path}/model_{args.load_checkpoint}.pt"

        if resume_path and os.path.exists(resume_path):
            print(f"[INFO] 継続学習: {resume_path} からロード")
            runner.load(resume_path)
        else:
            print(f"[WARN] チェックポイントが見つかりません")

    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)


if __name__ == "__main__":
    main()
