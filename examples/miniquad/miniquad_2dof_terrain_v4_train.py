"""
miniquad_2dof_terrain_v4_train.py
===================================
2DOF-hip 4脚ロボット 地形対応 v4 訓練スクリプト。

【v3 → v4 の主な変更点: スタート位置を地形変化点の直前に移動】

  問題点 (v2/v3):
    - 現在の歩行速度 ≈ 0.011 m/s × 20s = 0.22m
    - 平坦ゾーンは x=0 から x=2.0m (スタート x=0.5 から 1.5m 先)
    - → ロボットは全エピソードで平坦地しか歩かない! 地形学習が無効

  解決策:
    - base_init_pos を [0.5, 1.0, 0.080] → [1.9, 1.0, 0.080] に変更
    - スタートが平坦ゾーン終端 (x=2.0m) の 0.1m 手前
    - 現在速度 0.011 m/s で約 9 秒後に凸凹地形に到達
    - エピソードの後半 55% を凸凹・傾斜地形で歩行 → 実効的な地形訓練

  【地形レイアウト (再確認)】
    x=0.0〜2.0m: flat (スタートゾーン)
    x=2.0〜4.0m: random_uniform + wave (小凸凹)       ← ここから走行
    x=4.0〜6.0m: random_uniform + pyramid_sloped (傾斜)
    x=6.0〜8.0m: discrete_obstacles + stairs (障害物)

  追加変更点:
    - trot_bias_freq: 2.0 Hz (v3 から継続)
    - action_rate: -0.005 (v3 から継続)

【実行方法】
  # v3 最良チェックポイントから継続
  cd examples/miniquad
  PYTHONPATH=/home/mutsumi/rsl_rl:$PYTHONPATH python3 miniquad_2dof_terrain_v4_train.py \\
    -e miniquad-2dof-v4-terrain -B 16384 --max_iterations 5001 \\
    --load_run miniquad-2dof-v3-terrain --load_checkpoint -1 \\
    > /tmp/miniquad_terrain_v4_train.log 2>&1 &

  # v2 model_500 から (v2 が最良だった場合)
  PYTHONPATH=/home/mutsumi/rsl_rl:$PYTHONPATH python3 miniquad_2dof_terrain_v4_train.py \\
    -e miniquad-2dof-v4-terrain -B 16384 --max_iterations 5001 \\
    --load_run miniquad-2dof-v2-terrain --load_checkpoint 500 \\
    > /tmp/miniquad_terrain_v4_train.log 2>&1 &
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
    v4: スタート位置を地形変化点直前に変更。
    ロボットが実際に凸凹・傾斜地形を経験するように。
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

        # v4: 地形上では傾斜があるのでより緩和
        "termination_if_roll_greater_than":  1.4,
        "termination_if_pitch_greater_than": 1.4,

        # v4 核心: x=1.9m → 0.1m 歩けば凸凹ゾーンに突入
        # y=1.0m は 4m 幅の terrain 左半分 (col 0: y=0-2m) の中央
        "base_init_pos":  [1.9, 1.0, 0.085],  # 地形最大凸起 (15mm) + 余裕 5mm 分を z に追加
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],

        "episode_length_s":   20.0,
        "resampling_time_s":   4.0,
        "action_scale":        0.5,
        "simulate_action_latency": True,
        "clip_actions":       10.0,

        "use_trot_bias":       True,
        "trot_bias_freq":      2.0,   # 2Hz (v3継続)
        "trot_bias_coxa_amp":  0.30,
        "trot_bias_femur_amp": 0.25,

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
        "tracking_sigma":    0.02,
        # v4: 重心高さ目標を 0.085 に合わせる (地形最大凸起分を吸収)
        "base_height_target": 0.085,
        "reward_scales": {
            "tracking_lin_vel":  4.0,
            "forward_vel":       2.0,
            "alive":             0.5,
            "lin_vel_z":        -0.5,
            "base_height":      -0.3,  # 地形での高さ変動が大きいためさらに緩和
            "action_rate":      -0.005,
            "heading":          -0.2,
            "orientation":      -0.5,  # 傾斜地形での体の水平維持を強化
        },
    }

    command_cfg = {
        "num_commands": 3,
        "lin_vel_x_range": [0.05, 0.12],
        "lin_vel_y_range": [0, 0],
        "ang_vel_range":   [0, 0],
    }

    return env_cfg, obs_cfg, reward_cfg, command_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name",         type=str, default="miniquad-2dof-v4-terrain")
    parser.add_argument("-B", "--num_envs",          type=int, default=16384)
    parser.add_argument("--max_iterations",          type=int, default=5001)
    parser.add_argument("--load_run",                type=str, default=None)
    parser.add_argument("--load_checkpoint",         type=int, default=-1)
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
