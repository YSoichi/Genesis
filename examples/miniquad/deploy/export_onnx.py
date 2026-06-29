"""
export_onnx.py
==============
学習済み miniquad-2dof ポリシーを ONNX 形式にエクスポートする。
ラズパイ4上の onnxruntime で読み込んで推論できるようになる。

【実行方法】(PC / 学習環境)
  cd examples/miniquad
  PYTHONPATH=/home/mutsumi/rsl_rl:$PYTHONPATH python3 deploy/export_onnx.py \
    --run miniquad-2dof-v4-terrain --checkpoint 2800

  出力: logs/miniquad-2dof-v4-terrain/policy.onnx

【ONNX I/O 仕様】
  Input  : "obs"     shape=(1, 38)  float32
  Output : "actions" shape=(1,  8)  float32

  観測ベクトルの順序・スケールは deploy/raspi/controller/state.py と完全一致すること。
"""
import argparse
import os
import pickle
import sys
import glob

import torch
import torch.onnx

# rsl_rl を PYTHONPATH 経由で参照
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rsl_rl.runners import OnPolicyRunner
from miniquad_2dof_terrain_env import miniquad2DOFTerrainEnv
import genesis as gs


def export_actor_to_onnx(runner: OnPolicyRunner, out_path: str, num_obs: int):
    """ActorCritic の actor 部分だけを ONNX にエクスポートする。"""
    actor_critic = runner.alg.actor_critic
    actor_critic.eval()

    # ダミー入力 (バッチサイズ=1, 次元=num_obs)
    dummy_obs = torch.zeros(1, num_obs, device=runner.device)

    # torch.onnx.export は forward() を呼ぶ必要があるが、
    # ActorCritic.act_inference が actor のみ通るラッパーなので
    # ラムダでラップする
    class ActorWrapper(torch.nn.Module):
        def __init__(self, ac):
            super().__init__()
            self.ac = ac

        def forward(self, obs):
            # empirical_normalization=False なのでそのまま actor を通す
            return self.ac.actor(obs)

    wrapper = ActorWrapper(actor_critic).to(runner.device)
    wrapper.eval()

    torch.onnx.export(
        wrapper,
        dummy_obs,
        out_path,
        input_names=["obs"],
        output_names=["actions"],
        dynamic_axes={"obs": {0: "batch"}, "actions": {0: "batch"}},
        opset_version=17,
    )
    print(f"[OK] ONNX saved: {out_path}")

    # 簡易検証: onnxruntime があれば実行
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(out_path, providers=["CPUExecutionProvider"])
        out = sess.run(None, {"obs": dummy_obs.cpu().numpy()})
        print(f"[OK] ONNX verification passed. Output shape: {out[0].shape}")
    except ImportError:
        print("[WARN] onnxruntime not installed on this machine; skip verification")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run",        type=str, default="miniquad-2dof-v4-terrain",
                        help="ログディレクトリ名 (logs/ 以下)")
    parser.add_argument("--checkpoint", type=int, default=2800,
                        help="チェックポイント番号 (-1 = 最新)")
    parser.add_argument("--out",        type=str, default=None,
                        help="出力 ONNX パス (省略時: logs/<run>/policy.onnx)")
    args = parser.parse_args()

    log_dir = f"logs/{args.run}"
    if not os.path.exists(log_dir):
        raise FileNotFoundError(f"ログディレクトリが見つかりません: {log_dir}")

    # チェックポイントパス解決
    if args.checkpoint == -1:
        ckpts = sorted(glob.glob(f"{log_dir}/model_*.pt"),
                       key=lambda p: int(p.split("model_")[-1].replace(".pt", "")))
        ckpt_path = ckpts[-1]
    else:
        ckpt_path = f"{log_dir}/model_{args.checkpoint}.pt"

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"チェックポイントが見つかりません: {ckpt_path}")

    out_path = args.out or f"{log_dir}/policy.onnx"

    # 設定読み込み
    cfgs = pickle.load(open(f"{log_dir}/cfgs.pkl", "rb"))
    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = cfgs
    train_cfg.setdefault("algorithm", {})["class_name"] = "PPO"
    train_cfg.setdefault("policy", {})["class_name"] = "ActorCritic"

    # Genesis を最小モードで初期化 (シーン構築は不要; runner には env が要る)
    gs.init(logging_level="error")

    env = miniquad2DOFTerrainEnv(
        num_envs=1,
        env_cfg=env_cfg, obs_cfg=obs_cfg, reward_cfg=reward_cfg, command_cfg=command_cfg,
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device="cuda:0")
    runner.load(ckpt_path)
    print(f"[OK] Loaded checkpoint: {ckpt_path}")

    export_actor_to_onnx(runner, out_path, obs_cfg["num_obs"])

    # ラズパイ用メタ情報を別ファイルに保存
    import json
    meta = {
        "num_obs":          obs_cfg["num_obs"],
        "num_actions":      env_cfg["num_actions"],
        "obs_scales":       obs_cfg["obs_scales"],
        "action_scale":     env_cfg["action_scale"],
        "clip_actions":     env_cfg["clip_actions"],
        "trot_bias_freq":   env_cfg.get("trot_bias_freq", 1.0),
        "trot_bias_coxa_amp":  env_cfg.get("trot_bias_coxa_amp", 0.35),
        "trot_bias_femur_amp": env_cfg.get("trot_bias_femur_amp", 0.30),
        "default_joint_angles": env_cfg["default_joint_angles"],
        "dof_names":        env_cfg["dof_names"],
    }
    meta_path = out_path.replace(".onnx", "_meta.json")
    json.dump(meta, open(meta_path, "w"), indent=2)
    print(f"[OK] Meta saved: {meta_path}")
    print()
    print(">>> ラズパイへのコピー:")
    print(f"    scp {out_path} pi@<raspi_ip>:~/miniquad/")
    print(f"    scp {meta_path} pi@<raspi_ip>:~/miniquad/")


if __name__ == "__main__":
    main()
