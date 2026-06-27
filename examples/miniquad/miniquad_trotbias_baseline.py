"""
miniquad_trotbias_baseline.py
==============================
policy のアクションをゼロに固定し、trot_bias のみでの走行速度を測定する。
これにより policy が trot_bias の効果を妨害しているか、物理的限界かを切り分ける。
"""
import pickle
import torch
import genesis as gs
from miniquad_env import miniquadEnv

EXP_NAME  = "miniquad-walking-v12"
MAX_STEPS = 1500  # 30s x 50Hz


def main():
    gs.init(logging_level="warning")

    log_dir = f"logs/{EXP_NAME}"
    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(
        open(f"{log_dir}/cfgs.pkl", "rb")
    )
    reward_cfg["reward_scales"] = {}

    import sys
    if "--freq" in sys.argv:
        env_cfg["trot_bias_freq"] = float(sys.argv[sys.argv.index("--freq") + 1])
    if "--hip" in sys.argv:
        env_cfg["trot_bias_hip_amp"] = float(sys.argv[sys.argv.index("--hip") + 1])
    if "--knee" in sys.argv:
        env_cfg["trot_bias_knee_amp"] = float(sys.argv[sys.argv.index("--knee") + 1])
    if "--push" in sys.argv:
        env_cfg["trot_bias_push_amp"] = float(sys.argv[sys.argv.index("--push") + 1])

    env = miniquadEnv(
        num_envs=1, env_cfg=env_cfg, obs_cfg=obs_cfg,
        reward_cfg=reward_cfg, command_cfg=command_cfg, show_viewer=False,
    )

    obs, _ = env.reset()
    env.commands[:, 0] = 0.3
    env.commands[:, 1] = 0.0
    env.commands[:, 2] = 0.0

    zero_actions = torch.zeros((1, env_cfg["num_actions"]), device="cuda:0")

    peak_x = 0.0
    final_y = 0.0
    with torch.no_grad():
        for step in range(MAX_STEPS):
            x = env.base_pos[0, 0].item()
            y = env.base_pos[0, 1].item()
            z = env.base_pos[0, 2].item()
            if x > peak_x:
                peak_x = x
            final_y = y
            if step % 200 == 0:
                print(f"  step {step:4d}: x={x:+.4f} y={y:+.4f} z={z:.4f}")
            obs, _, dones, _ = env.step(zero_actions)
            env.commands[:, 0] = 0.3
            env.commands[:, 1] = 0.0
            env.commands[:, 2] = 0.0
            if dones[0].item():
                print(f"  step {step}: episode終了 (転倒など) x={x:+.4f} y={y:+.4f} z={z:.4f}")
                break

    print(f"\ntrot_bias のみ (policy action=0) での走行結果:")
    print(f"  peak_x = {peak_x:.3f} m / {MAX_STEPS/50:.0f}s")
    print(f"  final_y = {final_y:+.3f} m (横方向ドリフト)")
    print(f"  平均速度 = {peak_x / (MAX_STEPS/50):.4f} m/s")
    print(f"\n設定: freq={env_cfg.get('trot_bias_freq')} hip_amp={env_cfg.get('trot_bias_hip_amp')} knee_amp={env_cfg.get('trot_bias_knee_amp')} push_amp={env_cfg.get('trot_bias_push_amp')}")


if __name__ == "__main__":
    main()
