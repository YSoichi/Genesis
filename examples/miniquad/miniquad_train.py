"""
miniquad_train.py
=================
小型 4脚ロボット (miniquad, 200g クラス) の強化学習スクリプト。

【実行方法】
  cd examples/miniquad
  python miniquad_train.py -e miniquad-walking-v14 -B 8192 --max_iterations 3001

【v14 変更点 (v13 → v14)】
  - v13 (push_amp=0.15) は policy 学習後に peak_x=0.689m と v12(0.926m)より
    悪化。final_y (横ドリフト) が 0.09~0.24m と大きく、push-off と policy の
    相互作用で直進性が崩れたことが原因と判明。
  - push_amp を 0.15 → 0.1 に縮小して再ベースライン測定:
      push=0.05 → peak_x=0.257m final_y=-0.002m
      push=0.08 → peak_x=0.281m final_y=-0.010m
      push=0.10 → peak_x=0.282m final_y=-0.004m (最良、ドリフトも最小)
    ベースライン単体ではどの push_amp でも横ドリフトはほぼ無し
    (-0.002~-0.010m) のため、v13の悪化はpolicy学習側の問題と判断。
    push_amp=0.1 を採用し、heading報酬は維持して再学習する。

【v12 からの継承】
  - trot_bias_freq=1.0 Hz (サーボトルク制限 effort=0.16N·m 内での追従性重視)
  - action_rate=-0.02 (noise均衡点 σ≈1.77 で安定)
  - kp=25.0, kd=0.6
  - lin_vel 観測 (num_obs=38), alive=0.5
  - forward_vel=3.0, tracking_lin_vel=4.0
  - similar_to_default 削除, trot_gait 削除, foot_clearance 削除

【コマンドライン引数】
  -e / --exp_name    : 実験名
  -B / --num_envs    : 並列環境数 (デフォルト 8192)
  --max_iterations   : PPO 最大イテレーション数
"""
import argparse
import os
import pickle
import shutil

from miniquad_env import miniquadEnv
from rsl_rl.runners import OnPolicyRunner
import genesis as gs


def get_train_cfg():
    return {
        "num_steps_per_env": 24,
        "save_interval": 100,
        "empirical_normalization": False,
        "algorithm": {
            "class_name": "PPO",
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": 0.005,   # noise 崩壊防止; hoot(0.01) より保守的
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
    環境・観測・報酬・速度指令の設定を返す。

    【デフォルト関節角度】
      hip  = +0.3 rad (脚を若干前傾させた立位姿勢)
      knee = -0.6 rad (膝を屈曲させてクッションのある立位)
      → 重心高さ ≈ 0.11m

    【トルク確認】
      支持脚 1本にかかる荷重 ≈ 200g / 3 = 66.7g (三脚支持)
      knee トルク = 0.0667kg × 9.81 × 0.05m × sin(45°) ≈ 0.023 N·m = 0.23 kg·cm
      サーボ定格 1.2 kg·cm に対して十分な余裕あり
    """
    env_cfg = {
        "num_actions": 8,

        "default_joint_angles": {
            "body_to_hip_fr_j":      0.3,
            "hip_fr_to_knee_fr_j":  -0.6,
            "body_to_hip_fl_j":      0.3,
            "hip_fl_to_knee_fl_j":  -0.6,
            "body_to_hip_rr_j":      0.3,
            "hip_rr_to_knee_rr_j":  -0.6,
            "body_to_hip_rl_j":      0.3,
            "hip_rl_to_knee_rl_j":  -0.6,
        },

        "dof_names": [
            "body_to_hip_fr_j",  "hip_fr_to_knee_fr_j",
            "body_to_hip_fl_j",  "hip_fl_to_knee_fl_j",
            "body_to_hip_rr_j",  "hip_rr_to_knee_rr_j",
            "body_to_hip_rl_j",  "hip_rl_to_knee_rl_j",
        ],

        "kp": 25.0,
        "kd": 0.6,

        "termination_if_roll_greater_than":  0.8,
        "termination_if_pitch_greater_than": 0.8,

        "base_init_pos":  [0.0, 0.0, 0.113],
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],

        "episode_length_s":  20.0,
        "resampling_time_s":  4.0,
        "action_scale":  0.5,
        "simulate_action_latency": True,
        "clip_actions": 10.0,

        "use_cpg_obs": False,

        # v13: 非対称duty-cycle CPG + knee push-off (stance相で地面を蹴る)
        # サーボトルク制限 (0.16N·m) 内で追従可能な低周波・適度振幅に調整
        "use_trot_bias":     True,
        "trot_bias_freq":    1.0,   # [Hz] トロット周波数
        "trot_bias_hip_amp": 0.20,  # [rad] hip スイング振幅 (v12: 0.25 → 縮小)
        "trot_bias_knee_amp": 0.15, # [rad] knee 持ち上げ振幅 (v12: 0.20 → 縮小)
        "trot_bias_push_amp": 0.1,  # [rad] stance中盤のknee push-off振幅 (v13: 0.15 → 縮小)
    }

    obs_cfg = {
        # v9: lin_vel を追加; 38 = 3+3+3+3(lin_vel)+8+8+8+2(heading)
        "num_obs": 38,
        "obs_scales": {
            "lin_vel": 2.0,
            "ang_vel": 0.25,
            "dof_pos": 1.0,
            "dof_vel": 0.05,
        },
    }

    reward_cfg = {
        "tracking_sigma": 0.25,
        "base_height_target": 0.113,
        "feet_air_time_threshold": 0.1,
        # v9: foot_clearance を削除 (「その場トロット」局所解の原因)。
        # 代わりに forward_vel (絶対前進速度) を追加し、わずかな前進でも即座に報酬が得られる
        # 密なシグナルを提供する。tracking_lin_vel の指数関数的減衰より発見しやすい。
        # alive を 0.5 に強化 (hoot 参照: 探索を維持する floor 報酬)。
        # v10: trot_biasにより脚上げは自動確保 → feet_air_time不要。
        # forward_vel を強化して前進への勾配を明確化。
        "reward_scales": {
            "tracking_lin_vel":  4.0,    # 速度指令追従 (メイン報酬)
            "forward_vel":       3.0,    # 前進速度直接報酬 (v9: 2.0 → 強化)
            "alive":             0.5,    # 探索維持の floor
            "lin_vel_z":        -1.0,
            "base_height":      -5.0,
            "action_rate":      -0.02,  # v11: 倍増 (noise均衡点 2.5 → 1.77)
            "heading":          -0.3,
        },
    }

    command_cfg = {
        "num_commands": 3,
        # v13: trot_bias単体速度(~0.009 m/s)が向上したため、わずかに上げる
        "lin_vel_x_range": [0.15, 0.45],
        "lin_vel_y_range": [0, 0],
        "ang_vel_range":   [0, 0],
    }

    return env_cfg, obs_cfg, reward_cfg, command_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name",       type=str, default="miniquad-walking-v14")
    parser.add_argument("-B", "--num_envs",        type=int, default=8192)
    parser.add_argument("--max_iterations",        type=int, default=3001)
    args = parser.parse_args()

    gs.init(logging_level="warning")

    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    train_cfg = get_train_cfg()

    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    env = miniquadEnv(
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
