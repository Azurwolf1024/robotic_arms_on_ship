"""
验证实验: 在 ship_arm 仿真里对比

    TSID  |  TSID+NN  |  TSID+LADRC  |  TSID+NN+LADRC

在 fixed_point / circle 两种任务、0.5/1.0/2.0 三种船体运动幅度下,
含 12% 惯量误差 + 摩擦的被控对象上, 看 NN 与 LADRC 补偿能否降低跟踪误差。
"""

from __future__ import annotations

import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from experiments.common import MISMATCH, make_ship, run_case, save            # noqa: E402
from ship_arm_ctl.ladrc import LadrcCompensator                                # noqa: E402
from ship_arm_ctl.nn_comp import ResidualCompensator                            # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", choices=["panda", "nero"], default="panda")
    ap.add_argument("--save", default=None, help="结果 pkl 名 (默认 nn_study_<robot>)")
    ap.add_argument("--trust-in", type=float, default=None,
                    help="可信域内界 (默认取 ControllerConfig, 与部署一致)")
    ap.add_argument("--trust-out", type=float, default=None,
                    help="可信域外界 (默认取 ControllerConfig, 与部署一致)")
    args = ap.parse_args()

    kind = args.robot
    nn_path = os.path.join(ROOT, "models",
                           "residual_net_nero.onnx" if kind == "nero"
                           else "residual_net.onnx")
    if not os.path.exists(nn_path):
        print(f"[skip] 未找到 {nn_path}; 请先运行 training/train_nn.py --robot {kind}")
        return
    # 与部署一致: 残差补偿被限制在 0.25×tau_max 内 (见 ShipArmController)
    from experiments.common import build_robot as _build_robot
    _tau_max = _build_robot(kind).tau_max
    # 门控阈值直接取 ROBOT_GATE, 保证 "benchmark 用的就是部署用的"
    from ship_arm_ctl.config import ROBOT_GATE
    _g = ROBOT_GATE[kind]
    nn_trust_in = args.trust_in if args.trust_in is not None else _g.trust_in
    nn_trust_out = args.trust_out if args.trust_out is not None else _g.trust_out
    nn_sev_in, nn_sev_out = _g.sev_in, _g.sev_out
    nn_qdd_clip = 4000.0
    nn = ResidualCompensator(
        onnx_path=nn_path, tau_max=_tau_max, residual_frac=0.25,
        trust_in=nn_trust_in, trust_out=nn_trust_out, qdd_clip=nn_qdd_clip,
        sev_in=nn_sev_in, sev_out=nn_sev_out)
    print(f"  [nn] trust=({nn_trust_in}, {nn_trust_out})  "
          f"sev=({nn_sev_in}, {nn_sev_out})")

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
                             tau_scale=1.0, kind=kind, **MISMATCH, **kw2)
                m = r["metrics"]
                results[(tk, name, scale)] = m
                print(f"{tk:<11} {name:<15} scale={scale:<4} "
                      f"pos={m['pos_mean']:8.3f}mm max={m['pos_max']:8.1f} "
                      f"rot={m['rot_mean']:6.3f} fb={r['qp_stats']['fallback']}")
                sys.stdout.flush()
    save_name = args.save or f"nn_study_{kind}"
    save(save_name, results)
    print(f"[done] saved {save_name}.pkl")


if __name__ == "__main__":
    t0 = time.perf_counter()
    main()
    print(f"elapsed {time.perf_counter()-t0:.1f}s")
