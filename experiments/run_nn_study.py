"""
验证实验: 在 ship_arm 仿真里对比

    TSID  |  TSID+NN  |  TSID+LADRC  |  TSID+NN+LADRC

在 fixed_point / circle 两种任务、0.5/1.0/2.0 三种船体运动幅度下,
含 12% 惯量误差 + 摩擦的被控对象上, 看 NN 与 LADRC 补偿能否降低跟踪误差。
"""

from __future__ import annotations

import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from experiments.common import MISMATCH, make_ship, run_case, save            # noqa: E402
from ship_arm_ctl.ladrc import LadrcCompensator                                # noqa: E402
from ship_arm_ctl.nn_comp import ResidualCompensator                            # noqa: E402


def main():
    nn_path = os.path.join(ROOT, "models", "residual_net.onnx")
    if not os.path.exists(nn_path):
        print(f"[skip] 未找到 {nn_path}; 请先运行 training/train_nn.py")
        return
    nn = ResidualCompensator(onnx_path=nn_path, tau_max=None)

    tasks = ["fixed_point", "circle"]
    scales = [0.5, 1.0, 2.0]
    configs = [
        ("TSID", {}),
        ("TSID+NN", {"nn_comp": nn}),
        ("TSID+LADRC", {}),
        ("TSID+NN+LADRC", {"nn_comp": nn}),
    ]
    results = {}
    for scale in scales:
        for tk in tasks:
            for name, kw in configs:
                ladrc = LadrcCompensator(1e-3, wo=15.0) if "LADRC" in name else None
                kw2 = dict(kw)
                if ladrc is not None:
                    kw2["ladrc"] = ladrc
                r = run_case(tk, "tsid", 12.0, make_ship(scale), log_every=30,
                             tau_scale=1.0, **MISMATCH, **kw2)
                m = r["metrics"]
                results[(tk, name, scale)] = m
                print(f"{tk:<11} {name:<15} scale={scale:<4} "
                      f"pos={m['pos_mean']:8.3f}mm max={m['pos_max']:8.1f} "
                      f"rot={m['rot_mean']:6.3f} fb={r['qp_stats']['fallback']}")
                sys.stdout.flush()
    save("nn_study", results)
    print("[done] saved nn_study.pkl")


if __name__ == "__main__":
    t0 = time.perf_counter()
    main()
    print(f"elapsed {time.perf_counter()-t0:.1f}s")
