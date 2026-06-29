"""
controller/inference.py
=======================
ONNX Runtime を使ったポリシー推論クラス。

export_onnx.py で生成した policy.onnx と policy_meta.json を読み込み、
38次元観測から8次元アクションを出力する。

インストール: pip3 install onnxruntime   (CPU版)
"""
import json
import os
import time
from pathlib import Path

import numpy as np


class PolicyRunner:
    """ONNX ポリシーを読み込んで推論を実行するクラス。

    使い方:
        runner = PolicyRunner("~/miniquad/policy.onnx")
        actions = runner.infer(obs)   # obs: shape=(38,) の numpy array
    """

    def __init__(self, onnx_path: str):
        """
        Args:
            onnx_path: policy.onnx のパス (policy_meta.json は同じディレクトリに必要)
        """
        import onnxruntime as ort

        onnx_path = str(Path(onnx_path).expanduser().resolve())
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"ONNX ファイルが見つかりません: {onnx_path}")

        # CPU 推論 (RPi4)
        self._session = ort.InferenceSession(
            onnx_path,
            providers=["CPUExecutionProvider"],
        )
        self._input_name  = self._session.get_inputs()[0].name
        self._output_name = self._session.get_outputs()[0].name

        # メタ情報の読み込み
        meta_path = onnx_path.replace(".onnx", "_meta.json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"メタファイルが見つかりません: {meta_path}")

        with open(meta_path) as f:
            self.meta = json.load(f)

        self._num_obs     = self.meta["num_obs"]
        self._num_actions = self.meta["num_actions"]

        # 推論レイテンシ計測用
        self._latency_ms_ewma = 0.0

        print(f"[PolicyRunner] ONNX 読み込み完了")
        print(f"  入力: {self._num_obs}次元 / 出力: {self._num_actions}次元")

    def infer(self, obs: np.ndarray) -> np.ndarray:
        """観測ベクトルからアクションを推論する。

        Args:
            obs: shape=(38,) または (1,38) の float32 numpy array
        Returns:
            actions: shape=(8,) の float32 numpy array (生のポリシー出力)
        """
        if obs.ndim == 1:
            obs = obs.reshape(1, -1)
        obs = obs.astype(np.float32)

        t0 = time.monotonic()
        result = self._session.run(
            [self._output_name],
            {self._input_name: obs},
        )
        elapsed_ms = (time.monotonic() - t0) * 1000

        # 指数移動平均でレイテンシを追跡
        alpha = 0.1
        self._latency_ms_ewma = (
            elapsed_ms if self._latency_ms_ewma == 0.0
            else (1 - alpha) * self._latency_ms_ewma + alpha * elapsed_ms
        )

        return result[0].flatten()

    @property
    def avg_latency_ms(self) -> float:
        """推論レイテンシの指数移動平均 (ms)。"""
        return self._latency_ms_ewma

    def benchmark(self, n_runs: int = 200) -> float:
        """推論速度ベンチマーク (ms/step)。

        Returns:
            平均推論時間 (ms)
        """
        dummy = np.zeros((1, self._num_obs), dtype=np.float32)
        times = []
        for _ in range(n_runs):
            t0 = time.monotonic()
            self._session.run([self._output_name], {self._input_name: dummy})
            times.append((time.monotonic() - t0) * 1000)
        avg = sum(times) / len(times)
        print(f"[PolicyRunner] 推論ベンチマーク: {avg:.2f} ms/step ({1000/avg:.0f} Hz 相当)")
        return avg


# ─── 単体テスト ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("使い方: python3 -m controller.inference <policy.onnx のパス>")
        sys.exit(1)

    runner = PolicyRunner(sys.argv[1])
    runner.benchmark()

    dummy_obs = np.zeros(runner._num_obs, dtype=np.float32)
    actions = runner.infer(dummy_obs)
    print(f"ダミー推論結果: {actions}")
