"""
miniquad_eval.py
================
学習済みポリシーを 3D ビューアで確認する評価スクリプト。

【実行方法】
  cd examples/miniquad
  python miniquad_eval.py --ckpt 800

【引数】
  --ckpt : 評価するチェックポイント番号 (例: 800 → model_800.pt)
           省略すると最新の model_*.pt を自動選択
"""
import argparse
import glob
import os
import pickle
import re
import torch

from miniquad_env import miniquadEnv
from rsl_rl.runners import OnPolicyRunner
import genesis as gs

EVAL_VX = 0.4   # [m/s] 評価時の固定前進速度


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="miniquad-walking")
    parser.add_argument("--ckpt", type=int, default=None,
                        help="チェックポイント番号 (省略時: 最新を自動選択)")
    args = parser.parse_args()

    gs.init()

    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(
        open(f"{log_dir}/cfgs.pkl", "rb")
    )
    reward_cfg["reward_scales"] = {}

    env = miniquadEnv(
        num_envs=1,
        env_cfg=env_cfg, obs_cfg=obs_cfg,
        reward_cfg=reward_cfg, command_cfg=command_cfg,
        show_viewer=True,
    )

    train_cfg.setdefault("algorithm", {})["class_name"] = "PPO"
    train_cfg.setdefault("policy",    {})["class_name"] = "ActorCritic"
    runner = OnPolicyRunner(env, train_cfg, log_dir, device="cuda:0")

    # チェックポイントの選択
    if args.ckpt is not None:
        ckpt_path = f"{log_dir}/model_{args.ckpt}.pt"
    else:
        candidates = sorted(
            glob.glob(os.path.join(log_dir, "model_*.pt")),
            key=lambda p: int(re.search(r"model_(\d+)\.pt$", p).group(1)),
        )
        if not candidates:
            print(f"ERROR: {log_dir} にモデルファイルが見つかりません。先に miniquad_train.py を実行してください。")
            return
        ckpt_path = candidates[-1]

    print(f"モデル読み込み: {ckpt_path}")
    runner.load(ckpt_path)
    policy = runner.get_inference_policy(device="cuda:0")

    obs, _ = env.reset()
    env.commands[:, 0] = EVAL_VX
    env.commands[:, 1] = 0.0
    env.commands[:, 2] = 0.0
    print(f"速度指令: vx = {EVAL_VX} m/s  (3D ビューアでロボットの動きを確認してください)")

    with torch.no_grad():
        while True:
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            # リセット後も指令を維持
            env.commands[:, 0] = EVAL_VX
            env.commands[:, 1] = 0.0
            env.commands[:, 2] = 0.0


if __name__ == "__main__":
    main()
