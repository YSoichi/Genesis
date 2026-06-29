"""
miniquad_2dof_eval.py
=====================
2DOF-hip ロボットの学習済みモデル評価スクリプト。

【使用方法】
  cd examples/miniquad
  PYTHONPATH=/home/mutsumi/rsl_rl:$PYTHONPATH python3 miniquad_2dof_eval.py \\
    --exp_name miniquad-2dof-v1 --checkpoint 5000

【評価指標】
  - peak_x: 20秒間で到達した最大 x 座標 (目標: 2.0m)
  - avg_vx: 平均前進速度 (目標: 0.10 m/s)
  - episode_length: エピソード長 (MAX_STEPS = 1500 で打ち切り)
  - 転倒率: MAX_STEPS 未満で終了したエピソードの割合
"""
import argparse
import os
import pickle

import torch
import numpy as np
import genesis as gs

from miniquad_2dof_env import miniquad2DOFEnv
from rsl_rl.runners import OnPolicyRunner

EVAL_VX    = 0.10   # [m/s] 評価コマンド速度
MAX_STEPS  = 1500   # 30 秒
TARGET_X   = 2.0    # [m]   目標到達距離
NUM_ENVS   = 8      # 評価用の並列環境数 (少なめでビジュアル確認しやすくする)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name",   type=str, default="miniquad-2dof-v1")
    parser.add_argument("--checkpoint", type=int, default=-1,
                        help="チェックポイント番号 (-1 = 最新)")
    parser.add_argument("--num_envs",   type=int, default=NUM_ENVS)
    parser.add_argument("--show",       action="store_true",
                        help="ビューアを表示 (シングル env 推奨: --num_envs 1)")
    args = parser.parse_args()

    gs.init(logging_level="warning")

    log_dir = f"logs/{args.exp_name}"
    cfgs = pickle.load(open(f"{log_dir}/cfgs.pkl", "rb"))
    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = cfgs

    # 評価用に速度コマンドを固定
    command_cfg["lin_vel_x_range"] = [EVAL_VX, EVAL_VX]
    command_cfg["lin_vel_y_range"] = [0, 0]
    command_cfg["ang_vel_range"]   = [0, 0]

    env = miniquad2DOFEnv(
        num_envs=args.num_envs,
        env_cfg=env_cfg, obs_cfg=obs_cfg,
        reward_cfg=reward_cfg, command_cfg=command_cfg,
        show_viewer=args.show,
    )

    # チェックポイント読み込み
    if args.checkpoint == -1:
        import glob
        ckpts = sorted(glob.glob(f"{log_dir}/model_*.pt"),
                       key=lambda p: int(p.split("model_")[-1].replace(".pt", "")))
        ckpt_path = ckpts[-1] if ckpts else None
    else:
        ckpt_path = f"{log_dir}/model_{args.checkpoint}.pt"

    if ckpt_path is None or not os.path.exists(ckpt_path):
        print(f"[ERROR] チェックポイントが見つかりません: {ckpt_path}")
        return

    # pickleから復元した train_cfg は OnPolicyRunner が class_name を pop 済みの場合があるため追加
    train_cfg.setdefault("algorithm", {})["class_name"] = "PPO"
    train_cfg.setdefault("policy",    {})["class_name"] = "ActorCritic"
    runner = OnPolicyRunner(env, train_cfg, log_dir, device="cuda:0")
    runner.load(ckpt_path)
    policy = runner.get_inference_policy(device="cuda:0")
    print(f"[INFO] 評価開始: {ckpt_path}")

    obs, _ = env.reset()
    peak_x   = torch.zeros(args.num_envs, device="cuda:0")
    init_x   = env.base_pos[:, 0].clone()
    done_step = torch.full((args.num_envs,), MAX_STEPS, device="cuda:0")
    fell      = torch.zeros(args.num_envs, dtype=torch.bool, device="cuda:0")

    with torch.no_grad():
        for step in range(MAX_STEPS):
            actions = policy(obs)
            obs, rew, done, info = env.step(actions)

            x_travel = env.base_pos[:, 0] - init_x
            peak_x = torch.maximum(peak_x, x_travel)

            # 初回転倒を記録
            newly_fell = done & ~fell
            done_step[newly_fell] = step
            fell |= done.bool()

    print("\n===== 評価結果 =====")
    print(f"チェックポイント: {ckpt_path}")
    print(f"評価コマンド速度: vx = {EVAL_VX} m/s")
    print(f"目標到達距離: {TARGET_X} m ({MAX_STEPS * env.dt}秒)")
    print()
    print(f"peak_x (平均):     {peak_x.mean().item():.3f} m  "
          f"(目標達成率: {(peak_x.mean()/TARGET_X*100):.1f}%)")
    print(f"peak_x (最大):     {peak_x.max().item():.3f} m")
    print(f"peak_x (最小):     {peak_x.min().item():.3f} m")
    avg_vx = peak_x.mean().item() / (done_step.float().mean().item() * env.dt)
    print(f"平均前進速度:      {avg_vx:.4f} m/s")
    print(f"転倒率:            {fell.float().mean().item()*100:.1f}%  "
          f"(全 {args.num_envs} env 中 {fell.sum().item()} が転倒)")
    print(f"平均生存ステップ:  {done_step.float().mean().item():.0f} / {MAX_STEPS}")


if __name__ == "__main__":
    main()
