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


# 每条臂的训练量程, 取自该臂**实测闭环 ‖q̈*‖_inf 的 p99** (output/measure_qdd.txt):
#     nero : 定点 153/1330/1312, 圆周 1044/1767/1639   (scale 0.5/1.0/2.0)
#     panda: 定点   14/19/1213, 圆周   53/73/341
# 两者差一个数量级以上 —— 轻臂惯量小, 同样的跟踪误差会被 TSID 换算成高得多的 q̈*。
# 这里按**标称工况 (scale ≤ 1.0)** 定训练域: panda 取 200 已绰绰有余; nero 需 1500。
# 剧烈工况偶发的超大 q̈* 不进训练域 —— 它们由在线的"可信域"平滑挡掉, 见 ControllerConfig。
#
# 反过来若给 panda 也硬上 1500, 标签里的 ΔM·q̈* 会冲到 365 Nm, out_scale 被迫取 394 Nm,
# 网络在正常工况 (残差仅几 Nm) 的分辨率会被彻底牺牲 —— 实测确认过了, 别这么干。
QDD_LIM = {"panda": 200.0, "nero": 1500.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=80000)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--out-scale", type=float, default=None,
                    help="输出限幅 (Nm); 默认由残差数据统计给出 (2×99.9 分位)")
    ap.add_argument("--clamp-x", type=float, default=4.0,
                    help="归一化后特征的钳位界 (标准差倍数), 烘焙进 ONNX 防外推爆掉")
    ap.add_argument("--qdd-lim", type=float, default=None,
                    help="训练时 q̈* 的采样量程 (rad/s²); 默认按机械臂取 QDD_LIM "
                         "(= 实测闭环 ‖q̈*‖_inf 的 p99, panda 200 / nero 1500)")
    ap.add_argument("--no-mag-weight", action="store_true",
                    help="关闭按 ‖q̈*‖ 的样本加权 (默认开启, 防止高加速度样本"
                         "主导 MSE 而牺牲常用中低加速度区间的精度)")
    ap.add_argument("--q-ref", type=float, default=50.0,
                    help="样本加权的参考加速度 (rad/s²), 取闭环 ‖q̈*‖ 的常用量级")
    ap.add_argument("--payload", type=float, default=0.0)
    ap.add_argument("--model-error", type=float, default=0.12)
    ap.add_argument("--friction", type=float, default=0.15)
    ap.add_argument("--coulomb", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--robot", choices=["panda", "nero"], default="panda")
    ap.add_argument("--data-cache", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    # 默认缓存/输出按机械臂分文件
    if args.data_cache is None:
        args.data_cache = os.path.join(ROOT, "models",
                                       "train_data_nero.npz" if args.robot == "nero"
                                       else "train_data.npz")
    if args.out is None:
        args.out = os.path.join(ROOT, "models",
                                "residual_net_nero.onnx" if args.robot == "nero"
                                else "residual_net.onnx")

    qdd_lim = args.qdd_lim if args.qdd_lim is not None else QDD_LIM[args.robot]
    print(f"[cfg] robot={args.robot}  qdd_lim={qdd_lim} "
          f"(默认=实测闭环 ‖q̈*‖_inf 的 p99)")

    # ---- 1) 数据 ----
    if os.path.exists(args.data_cache):
        print(f"[data] 加载缓存 {args.data_cache}")
        d = np.load(args.data_cache, allow_pickle=True)
        data = dict(X=d["X"], Y=d["Y"], meta=dict())
    else:
        t0 = time.perf_counter()
        data = generate_training_data(
            n_samples=args.n, payload=args.payload, model_error=args.model_error,
            friction=args.friction, coulomb=args.coulomb, seed=args.seed, kind=args.robot,
            qdd_lim=qdd_lim)
        print(f"[data] 生成 {args.n} 样本, 用时 {time.perf_counter()-t0:.1f}s")
        os.makedirs(os.path.dirname(args.data_cache), exist_ok=True)
        np.savez(args.data_cache, X=data["X"], Y=data["Y"])
    print(f"[data] X {data['X'].shape}  Y {data['Y'].shape}")
    print(f"[data] |Δτ|  RMS {np.linalg.norm(data['Y'])/np.sqrt(data['Y'].shape[0]):.2f} Nm,"
          f"  max {np.abs(data['Y']).max():.2f} Nm")

    # ---- 2) 训练 ----
    t0 = time.perf_counter()
    model = train_model(data, hidden=args.hidden, out_scale=args.out_scale,
                        epochs=args.epochs, seed=args.seed, clamp_x=args.clamp_x,
                        mag_weight=not args.no_mag_weight, q_ref=args.q_ref)
    print(f"[train] 用时 {time.perf_counter()-t0:.1f}s")

    # ---- 3) 导出 ONNX ----
    export_onnx(model, args.out, test_parity=True)
    print("[done]")


if __name__ == "__main__":
    main()
