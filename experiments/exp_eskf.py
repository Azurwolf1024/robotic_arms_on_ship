"""
论文 Table V: 基座状态估计(ESKF)的消融研究 + 动力学耦合补偿消融。

配置
------------------------------------------------------------------
A  full            : IMU + 末端位姿 + FK 反推基座位姿, 增广状态(30 维)
B  w/o IMU         : 只用位姿观测
C  w/o EE pose     : 不用末端位姿观测(只保留 FK 反推)
D  w/o FK base     : 不用 FK 反推的基座位姿(只保留末端位姿)
E  reduced state   : 缩维 ESKF(21 维, 角速度/加速度用辅助滤波补回)
F  direct base     : 直接基座位姿观测(论文对照)
G  no estimator    : 控制器直接用真值(性能上界参考)

同时给出"关闭 tau_base 前馈"(即不做基座动力学耦合补偿)的对照。
"""

from __future__ import annotations

import sys
import time

import numpy as np

from common import MISMATCH, fmt_row, make_ship, run_case, save    # noqa: E402
from ship_arm.control.tsid import TSIDOptions          # noqa: E402
from ship_arm.estimation.eskf import ESKFConfig        # noqa: E402

CFG = {
    "A full":               ESKFConfig(),
    "B w/o IMU":            ESKFConfig(use_imu=False),
    "C w/o EE pose":        ESKFConfig(use_ee_pose=False),
    "D w/o FK base":        ESKFConfig(use_fk_base_pose=False),
    "E reduced state":      ESKFConfig(augmented=False),
    "F direct base pose":   ESKFConfig(use_ee_pose=False, use_fk_base_pose=False,
                                       use_direct_base_pose=True),
}


def main():
    ship = make_ship(1.0)
    dur = 8.0
    res = {}
    print(f"{'config':<20} {'pos err':>9} {'rot err':>8} | "
          f"{'p mm':>7} {'rot deg':>8} {'v':>8} {'w':>8} {'a':>8}")
    for name, cfg in CFG.items():
        r = run_case("fixed_point", "tsid", dur, ship, eskf_cfg=cfg, **MISMATCH)
        e = r["est"]
        res[name] = dict(metrics=r["metrics"], est=e)
        print(f"{name:<20} {r['metrics']['pos_mean']:9.3f} {r['metrics']['rot_mean']:8.3f} | "
              f"{e['pos_mm']:7.3f} {e['rot_deg']:8.4f} {e['lin_vel']:8.4f} "
              f"{e['ang_vel']:8.4f} {e['lin_acc']:8.4f}")
        sys.stdout.flush()

    # ---- 动力学耦合补偿消融 ----
    print("\n----- compensation ablation (tau_base feed-forward) -----")
    r_on = run_case("fixed_point", "tsid", dur, ship,
                    tsid_opts=TSIDOptions(compensate_base=True), **MISMATCH)
    r_off = run_case("fixed_point", "tsid", dur, ship,
                     tsid_opts=TSIDOptions(compensate_base=False), **MISMATCH)
    r_nb = run_case("fixed_point", "tsid", dur, ship,
                    tsid_opts=TSIDOptions(use_base_pose=False), **MISMATCH)
    for tag, r in [("compensate ON", r_on), ("compensate OFF", r_off),
                   ("base pose ignored", r_nb)]:
        print(fmt_row(tag, r["metrics"]))
        res[tag] = dict(metrics=r["metrics"], est=r["est"])

    # ---- 无估计器(真值)上界 ----
    r_gt = run_case("fixed_point", "tsid", dur, ship, use_estimator=False, **MISMATCH)
    print(fmt_row("true base state", r_gt["metrics"]))
    res["true base state"] = dict(metrics=r_gt["metrics"], est=r_gt["est"])

    save("eskf_ablation", res)


if __name__ == "__main__":
    t0 = time.perf_counter()
    main()
    print(f"\n[total] {time.perf_counter() - t0:.1f}s")
