"""
miniquad_ckpt_eval.py
=====================
全チェックポイントを固定指令で評価し、最良モデルを特定する。
"""
import glob, re, os, pickle, torch
import genesis as gs
from miniquad_env import miniquadEnv
from rsl_rl.runners import OnPolicyRunner

EVAL_VX   = 0.25   # [m/s] 評価速度 (v14: cmd=[0.15,0.45] の中央付近)
MAX_STEPS = 1500   # 30s × 50Hz
TARGET_X  = 10.0   # [m] 目標距離


def eval_ckpt(runner, env, steps=MAX_STEPS):
    obs, _ = env.reset()
    env.commands[:, 0] = EVAL_VX
    env.commands[:, 1] = 0.0
    env.commands[:, 2] = 0.0

    peak_x = 0.0
    final_x = final_y = 0.0
    reach_step = None
    with torch.no_grad():
        for step in range(steps):
            x = env.base_pos[0, 0].item()
            y = env.base_pos[0, 1].item()
            if x > peak_x:
                peak_x = x
            if reach_step is None and x >= TARGET_X:
                reach_step = step
            final_x, final_y = x, y
            actions = runner.get_inference_policy(device="cuda:0")(obs)
            obs, _, dones, _ = env.step(actions)
            env.commands[:, 0] = EVAL_VX
            env.commands[:, 1] = 0.0
            env.commands[:, 2] = 0.0
            if dones[0].item():
                break
    reach_time = reach_step / 50.0 if reach_step is not None else None
    return peak_x, final_x, final_y, reach_time


def main():
    gs.init(logging_level="warning")

    log_dir = "logs/miniquad-walking-v14"
    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(
        open(f"{log_dir}/cfgs.pkl", "rb")
    )
    reward_cfg["reward_scales"] = {}

    env = miniquadEnv(
        num_envs=1, env_cfg=env_cfg, obs_cfg=obs_cfg,
        reward_cfg=reward_cfg, command_cfg=command_cfg, show_viewer=False,
    )
    train_cfg.setdefault("algorithm", {})["class_name"] = "PPO"
    train_cfg.setdefault("policy",    {})["class_name"] = "ActorCritic"
    runner = OnPolicyRunner(env, train_cfg, log_dir, device="cuda:0")

    candidates = sorted(
        glob.glob(os.path.join(log_dir, "model_*.pt")),
        key=lambda p: int(re.search(r"model_(\d+)\.pt$", p).group(1)),
    )

    print(f"\n評価速度: {EVAL_VX} m/s  |  目標: {TARGET_X}m  |  最大時間: {MAX_STEPS/50:.0f}s")
    print(f"\n{'ckpt':>6} | {'peak_x':>7} | {'final_x':>7} | {'final_y':>7} | {'10m到達':>8} | 状態")
    print("-" * 58)

    best_ckpt, best_x = "", 0.0
    for ckpt_path in candidates:
        runner.load(ckpt_path)
        peak_x, final_x, final_y, reach_time = eval_ckpt(runner, env)
        it = int(re.search(r"model_(\d+)\.pt$", ckpt_path).group(1))
        mark = " ← best" if peak_x > best_x else ""
        if peak_x > best_x:
            best_x = peak_x
            best_ckpt = ckpt_path
        reach_str = f"{reach_time:.1f}s" if reach_time is not None else "未達"
        status = "達成!" if reach_time is not None else f"{peak_x/TARGET_X*100:.1f}%"
        print(f"{it:>6} | {peak_x:>7.3f} | {final_x:>7.3f} | {final_y:>7.3f} | {reach_str:>8} | {status}{mark}")

    print(f"\n最良: {best_ckpt}  (peak_x = {best_x:.3f}m)")
    if best_x >= TARGET_X:
        print("★ 10m タスク達成! ★")
    else:
        print(f"10m まで残り {TARGET_X - best_x:.3f}m ({best_x/TARGET_X*100:.1f}%)")


if __name__ == "__main__":
    main()
