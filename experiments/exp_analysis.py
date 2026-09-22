"""
论文 IV-B 的两类"偏离期望阻抗动力学"的来源分析, 以及 IV-C 的计算耗时基准。

    Fig 6  δc,f 约束激活图: 归一化关节速度 × 基座运动缩放因子的二维扫描,
           等值线 ∥δc,f∥ = 100 N(Panda 的近似出力上限)
    Fig 7  基座运动**时间**尺度扫描: 跟踪误差 + 约束激活率
    Fig 8  未建模负载 0 -> 2.5 kg 下的跟踪误差
    λ      零空间权重在两个基座运动尺度下的影响(论文 IV-B1)
    Tab III 单周期计算耗时分解
"""

from __future__ import annotations

import sys
import time

import numpy as np

from common import (  # noqa: E402
    METHODS, METHOD_LABEL, MISMATCH, build_robot, make_ship, run_case, save,
)
from ship_arm.control.tsid import TSIDController, TSIDGains, TSIDOptions   # noqa: E402
from ship_arm.estimation.eskf import ESKF, ESKFConfig                      # noqa: E402
from ship_arm.qp.dense_qp import solve_qp                                  # noqa: E402
from ship_arm.robot.panda import PANDA_HOME                                # noqa: E402


# --------------------------------------------------------------------------- #
# Fig 6: δc,f 的二维参数扫描(静态求解 QP, 不做闭环仿真 —— 与论文一致)
# --------------------------------------------------------------------------- #
def fig6_contour(n_s=25, n_v=25):
    robot = build_robot(0.0)
    ctrl = TSIDController(robot, 1e-3, TSIDGains(), TSIDOptions())
    ship = make_ship(1.0)

    # 取一个基座运动"典型"时刻(接近峰值)的状态作为 scale=1 的基准
    st = ship.world_state(7.6)
    base0 = dict(R_WB=st["R_WB"], p_B=st["p_B"],
                 omega_w=st["omega_w"].copy(), v_w=st["v_w"].copy(),
                 alpha_w=st["alpha_w"].copy(), a_w=st["a_w"].copy())

    q = PANDA_HOME.copy()
    rng = np.random.default_rng(0)
    dq_dir = rng.normal(size=robot.n)
    dq_dir /= np.linalg.norm(dq_dir)

    # 论文的 scale=1 对应基座线速度 0.05 m/s; 本文的波谱标定到峰值 0.32 m/s,
    # 因此这里的 scale 轴按本文波谱的加速度量级延伸到 30, 才能覆盖"约束激活区"。
    scales = np.linspace(0.0, 30.0, n_s)
    speeds = np.linspace(0.0, 1.0, n_v)
    D = np.zeros((n_v, n_s))
    ACT = np.zeros((n_v, n_s))
    for i, s in enumerate(scales):
        est = dict(base0)
        est["omega_w"] = base0["omega_w"] * s
        est["v_w"] = base0["v_w"] * s
        est["alpha_w"] = base0["alpha_w"] * s
        est["a_w"] = base0["a_w"] * s
        for j, u in enumerate(speeds):
            dq = u * robot.dq_max * dq_dir
            R_ee, p_ee = robot.ee_pose(est["R_WB"], est["p_B"], q)
            ref = dict(R_d=R_ee, p_d=p_ee,
                       xd_dot=np.zeros(6), xd_ddot=np.zeros(6))
            out = ctrl.compute(q, dq, est, ref)
            D[j, i] = float(np.linalg.norm(out["delta_c"][3:6]))
            ACT[j, i] = int(np.sum(out["active"]))
    return dict(scales=scales, speeds=speeds, delta_f=D, n_active=ACT)


# --------------------------------------------------------------------------- #
# Fig 7: 基座运动时间尺度 -> 跟踪误差 + 约束激活率
# --------------------------------------------------------------------------- #
def fig7_temporal(dur=8.0):
    tscales = [1.0, 1.2, 1.5, 2.0, 2.4, 3.0]
    out = []
    print("\n----- Fig 7: base-motion temporal scale -----")
    print(f"{'time scale':<12} {'pos mean':>9} {'pos max':>9} {'rot mean':>9} "
          f"{'active%':>8} {'fallback%':>10}")
    for ts in tscales:
        r = run_case("circle", "tsid", dur, None, ship_time_scale=ts,
                     ship_scale=1.0, **MISMATCH, log_every=5)
        st = r["qp_stats"]
        tot = max(st["feasible"] + st["fallback"], 1)
        rec = dict(tscale=ts, metrics=r["metrics"],
                   active_pct=100.0 * st["active"] / tot,
                   fallback_pct=100.0 * st["fallback"] / tot)
        out.append(rec)
        print(f"{ts:<12.2f} {r['metrics']['pos_mean']:9.3f} "
              f"{r['metrics']['pos_max']:9.3f} {r['metrics']['rot_mean']:9.3f} "
              f"{rec['active_pct']:8.1f} {rec['fallback_pct']:10.1f}")
        sys.stdout.flush()
    return out


# --------------------------------------------------------------------------- #
# Fig 8: 未建模负载
# --------------------------------------------------------------------------- #
def fig8_payload(dur=6.0):
    masses = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
    methods = ["tsid", "pi", "xu"]
    print("\n----- Fig 8: unmodeled payload -----")
    print(f"{'payload kg':<12} " + " ".join(f"{METHOD_LABEL[m]:>16}" for m in methods))
    out = {}
    for mp in masses:
        row = {}
        for m in methods:
            r = run_case("fixed_point", m, dur, None, ship_scale=1.0,
                         payload=mp, **MISMATCH, log_every=5)
            row[m] = r["metrics"]
        out[mp] = row
        print(f"{mp:<12.2f} " + " ".join(f"{row[m]['pos_mean']:16.3f}" for m in methods))
        sys.stdout.flush()
    return out


# --------------------------------------------------------------------------- #
# λ 的影响(论文 IV-B1)
# --------------------------------------------------------------------------- #
def lambda_sweep(dur=8.0):
    lams = [0.001, 0.01, 0.1, 1.0, 10.0]
    res = {}
    print("\n----- nullspace weight lambda -----")
    print(f"{'lambda':<10} " + " ".join(f"{'scale ' + str(s):>16}" for s in [1.0, 2.4]))
    for lam in lams:
        row = {}
        for sc in [1.0, 2.4]:
            r = run_case("circle", "tsid", dur, None, ship_scale=sc,
                         gains=TSIDGains(lam=lam), **MISMATCH, log_every=5)
            row[sc] = r["metrics"]
        res[lam] = row
        print(f"{lam:<10} " + " ".join(f"{row[s]['pos_max']:16.3f}" for s in [1.0, 2.4]))
        sys.stdout.flush()
    return res


# --------------------------------------------------------------------------- #
# Table III: 计算耗时
# --------------------------------------------------------------------------- #
def table3_compute():
    print("\n----- Table III: per-cycle computation -----")
    robot = build_robot(0.0)
    ctrl = TSIDController(robot, 1e-3, TSIDGains(), TSIDOptions())
    eskf = ESKF(ESKFConfig())
    eskf.reset(np.eye(3), np.zeros(3), np.zeros(3), np.eye(3))
    arm = dict(p_E_B=np.zeros(3), R_E_B=np.eye(3),
               v_arm_B=np.zeros(3), w_arm_B=np.zeros(3))

    # 用**闭环中的典型工作点**计时(基座在动、关节在动、指令非零),
    # 而不是 q=0 的退化点 —— 退化 QP 的对偶解为 0, 内点法会跑满迭代数。
    ship = make_ship(1.0)
    st = ship.world_state(7.6)
    q = PANDA_HOME.copy()
    rng = np.random.default_rng(0)
    dq = 0.3 * robot.dq_max * (rng.normal(size=robot.n) / np.linalg.norm(rng.normal(size=robot.n)))
    est = dict(R_WB=st["R_WB"], p_B=st["p_B"], omega_w=st["omega_w"], v_w=st["v_w"],
               alpha_w=st["alpha_w"], a_w=st["a_w"])
    R_ee, p_ee = robot.ee_pose(est["R_WB"], est["p_B"], q)
    ref = dict(R_d=R_ee, p_d=p_ee + np.array([0.01, -0.01, 0.005]),
               xd_dot=np.zeros(6), xd_ddot=np.zeros(6))
    base = (est["R_WB"], est["p_B"])
    rows = np.zeros((1, 12))

    def timeit(fn, n=400, warm=50):
        for _ in range(warm):
            fn()
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        return (time.perf_counter() - t0) / n * 1e3

    t_terms = timeit(lambda: robot.state_terms(base, q, dq, rows, want_M=True, want_J=True))
    t_bias = timeit(lambda: robot.bias(base, q, dq, rows[0]))
    t_ctrl = timeit(lambda: ctrl.compute(q, dq, est, ref), n=300, warm=40)
    t_eskf = timeit(lambda: eskf.predict(1e-3, arm))

    M = np.eye(7) * 0.5
    J = np.random.default_rng(0).normal(size=(6, 7))
    Q = J.T @ J + 1e-3 * np.eye(7)
    c = np.zeros(7)
    G = np.vstack([np.eye(7), -np.eye(7)])
    h = np.ones(14) * 5.0
    t_qp = timeit(lambda: solve_qp(Q, c, G, h, max_iter=40), n=300, warm=40)

    # 闭环实测: 直接取仿真主循环里每个周期的真实墙钟耗时
    r = run_case("circle", "tsid", 6.0, None, ship_scale=1.0, **MISMATCH, log_every=5)
    cpu = float(np.mean(r["log"]["cpu_ms"]))

    out = dict(state_terms=t_terms, bias=t_bias, qp_only=t_qp,
               tsid_total=t_ctrl, eskf_predict=t_eskf,
               tsid_plus_eskf=t_ctrl + t_eskf,
               closed_loop_cycle=cpu)
    for k, v in out.items():
        print(f"  {k:<18} {v:8.4f} ms")
    return out


def main():
    res = dict(
        fig6=fig6_contour(),
        fig7=fig7_temporal(),
        fig8=fig8_payload(),
        lam=lambda_sweep(),
        table3=table3_compute(),
    )
    save("analysis", res)


if __name__ == "__main__":
    t0 = time.perf_counter()
    main()
    print(f"\n[total] {time.perf_counter() - t0:.1f}s")
