"""
神经网络残差补偿的训练流水线。

步骤: 1) 用仿真生成 (特征 X, 残差标签 Y) 配对数据
      2) 训练 ResidualNet (MSE)
      3) 导出 ONNX (models/residual_net.onnx) 并校验与 PyTorch 一致

用法
----
  python training/train_nn.py --n 80000 --epochs 300 --out models/residual_net.onnx

数据生成使用与仿真一致的真值对象 (含 12% 惯量误差 + 0.15 粘性摩擦 + 0.3 Nm 库仑摩擦
+ 0~2.5 倍船体运动), 因此学到的残差恰好是"名义模型 + ESKF"所缺失的那部分。
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from ship_arm_ctl.nn_comp import (export_onnx, generate_training_data,
                                  train_model)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=80000)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--out-scale", type=float, default=40.0)
    ap.add_argument("--payload", type=float, default=0.0)
    ap.add_argument("--model-error", type=float, default=0.12)
    ap.add_argument("--friction", type=float, default=0.15)
    ap.add_argument("--coulomb", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data-cache", default=os.path.join(ROOT, "models", "train_data.npz"))
    ap.add_argument("--out", default=os.path.join(ROOT, "models", "residual_net.onnx"))
    args = ap.parse_args()

    # ---- 1) 数据 ----
    if os.path.exists(args.data_cache):
        print(f"[data] 加载缓存 {args.data_cache}")
        d = np.load(args.data_cache, allow_pickle=True)
        data = dict(X=d["X"], Y=d["Y"], meta=dict())
    else:
        t0 = time.perf_counter()
        data = generate_training_data(
            n_samples=args.n, payload=args.payload, model_error=args.model_error,
            friction=args.friction, coulomb=args.coulomb, seed=args.seed)
        print(f"[data] 生成 {args.n} 样本, 用时 {time.perf_counter()-t0:.1f}s")
        os.makedirs(os.path.dirname(args.data_cache), exist_ok=True)
        np.savez(args.data_cache, X=data["X"], Y=data["Y"])
    print(f"[data] X {data['X'].shape}  Y {data['Y'].shape}")
    print(f"[data] |Δτ|  RMS {np.linalg.norm(data['Y'])/np.sqrt(data['Y'].shape[0]):.2f} Nm,"
          f"  max {np.abs(data['Y']).max():.2f} Nm")

    # ---- 2) 训练 ----
    t0 = time.perf_counter()
    model = train_model(data, hidden=args.hidden, out_scale=args.out_scale,
                        epochs=args.epochs, seed=args.seed)
    print(f"[train] 用时 {time.perf_counter()-t0:.1f}s")

    # ---- 3) 导出 ONNX ----
    export_onnx(model, args.out, test_parity=True)
    print("[done]")


if __name__ == "__main__":
    main()
