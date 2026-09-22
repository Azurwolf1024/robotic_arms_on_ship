"""
论文 IV-F / Fig 14-16: **动态轴孔装配 (peg-in-hole)**。

设置
------------------------------------------------------------------
* 孔固定在**世界系**(模拟"对岸/固定平台上的目标"), 机械臂随船体 6-DOF 运动;
  因此既要补偿基座运动带来的位姿漂移, 又要处理接触。
* 插销 Ø35 mm, 孔 Ø37 mm(径向间隙 1 mm), 孔深 45 mm, 孔口 10 mm 倒角。
* 初始对准误差 3 mm(大于间隙 -> 必须靠倒角被动导向, 否则插不进去)。
* 参考轨迹: 悬停 1 s -> 以 12 mm/s 下插 5 s -> 悬停 2 s。

对照
------------------------------------------------------------------
    ours        : TSID-QP + 六维导纳外环(腕部力反馈)
    tsid-only   : TSID-QP, 无力反馈(纯位置控制)
    no-comp     : TSID-QP + 导纳, 但关闭 tau_base 前馈
    pi          : 基线 PI(速度层, 不依赖动力学)
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass

import numpy as np

from common import MISMATCH, build_robot, save                      # noqa: E402
from ship_arm.control.admittance import Admittance6                 # noqa: E402
from ship_arm.control.tsid import TSIDOptions                       # noqa: E402
from ship_arm.platform.ship import ShipMotion                       # noqa: E402
from ship_arm.robot.panda import PANDA_HOME                         # noqa: E402
from ship_arm.sim.contact import PegHoleContact                     # noqa: E402
from ship_arm.sim.engine import SimOptions, Simulator               # noqa: E402

INSERT_SPEED = 0.012      # m/s
T_HOLD0, T_DOWN, T_HOLD1 = 1.0, 5.0, 2.0


@dataclass
class Descend:
    """下插参考轨迹(世界系)。"""
    p0: np.ndarray
    R0: np.ndarray
    duration: float = 8.0
    kind: str = "fixed_point"
    v: float = INSERT_SPEED

    def sample(self, t: float) -> dict:
        p = self.p0.copy()
        v = np.zeros(3)
        if t > T_HOLD0:
            s = min((t - T_HOLD0) * self.v, T_DOWN * self.v)
            p[2] -= s
            if t - T_HOLD0 < T_DOWN:
                v[2] = -self.v
        return dict(R_d=self.R0, p_d=p,
                    xd_dot=np.concatenate([np.zeros(3), v]),
                    xd_ddot=np.zeros(6))


def run_peg(method: str, compensate: bool = True, admittance: bool = True,
            duration: float = 8.0, offset: float = 0.003, ship_scale: float = 1.0,
            dt: float = 1e-3, seed: int = 1):
    robot_c = build_robot(0.0)
    robot_t = build_robot(0.0, model_error=MISMATCH["model_error"])
    ship = ShipMotion()
    ship.scale = ship_scale

    st0 = ship.world_state(0.0)
    R_ee0, p_ee0 = robot_c.ee_pose(st0["R_WB"], st0["p_B"], PANDA_HOME)

    # 起始时插销尖端悬于孔口上方 15 mm(先自由接近, 再入孔)
    hole = PegHoleContact(center=np.array([p_ee0[0] + offset, p_ee0[1]]),
                          top_z=p_ee0[2] - 0.015)
    task = Descend(p0=p_ee0.copy(), R0=R_ee0.copy(), duration=duration)

    opts = SimOptions(dt=dt, duration=duration, controller=method,
                      contact=hole,
                      admittance=Admittance6() if admittance else None,
                      force_noise=0.35 if admittance else 0.0,
                      friction=MISMATCH["friction"], coulomb=MISMATCH["coulomb"],
                      model_error=MISMATCH["model_error"],
                      log_every=5, seed=seed)
    if not compensate:
        pass
    sim = Simulator(robot_c, robot_t, ship, task, opts, q0=PANDA_HOME.copy(),
                    tsid_opts=TSIDOptions(compensate_base=compensate))
    log = sim.run()
    a = log.arrays()

    depth = np.clip(hole.top_z - a["p_ee"][:, 2], 0.0, hole.depth)
    lat = np.linalg.norm(a["fext"][:, 3:5], axis=1)
    axial = np.abs(a["fext"][:, 5])
    t = a["t"]
    idx = np.argmax(depth >= 0.9 * hole.depth) if np.any(depth >= 0.9 * hole.depth) else -1
    return dict(
        method=method, compensate=compensate, admittance=admittance,
        t=t, depth=depth, lat=lat, axial=axial,
        final_depth=float(depth[-1]), max_depth=float(depth.max()),
        success=bool(np.any(depth >= 0.95 * hole.depth)),
        t_insert=float(t[idx]) if idx >= 0 else float("nan"),
        peak_lat=float(lat.max()), mean_lat=float(lat.mean()),
        peak_axial=float(axial.max()),
        hole_top=float(hole.top_z), hole_center=hole.center.copy(),
        clearance=float(hole.clearance),
    )


CASES = [
    ("ours (TSID+adm)", dict(method="tsid", compensate=True, admittance=True)),
    ("TSID w/o force", dict(method="tsid", compensate=True, admittance=False)),
    ("TSID w/o comp", dict(method="tsid", compensate=False, admittance=True)),
    ("PI (baseline)", dict(method="pi", compensate=True, admittance=False)),
]

# 初始对准误差扫描: 间隙只有 1 mm, 超过 ~11 mm(倒角外缘) 就完全插不进去
OFFSETS = [0.001, 0.002, 0.003, 0.004, 0.005, 0.006]


def main():
    res = {}
    # ---- 1) 固定 3 mm 对准误差下的详细对比(时间序列用于画图) ----
    print(f"{'case':<18} {'final mm':>9} {'max mm':>8} {'insert s':>9} "
          f"{'peakFxy N':>10} {'meanFxy N':>10} {'peakFz N':>9} {'success':>8}")
    detail = {}
    for name, kw in CASES:
        r = run_peg(offset=0.003, **kw)
        detail[name] = r
        print(f"{name:<18} {r['final_depth'] * 1e3:9.2f} {r['max_depth'] * 1e3:8.2f} "
              f"{r['t_insert']:9.2f} {r['peak_lat']:10.2f} {r['mean_lat']:10.2f} "
              f"{r['peak_axial']:9.2f} {str(r['success']):>8}")
        sys.stdout.flush()
    res["detail"] = detail

    # ---- 2) 对准误差扫描 -> 成功率 / 接触力 ----
    print("\n----- initial misalignment sweep -----")
    print(f"{'offset mm':<10} " + " ".join(f"{n:>20}" for n, _ in CASES))
    sweep = {}
    for off in OFFSETS:
        row = {}
        cells = []
        for name, kw in CASES:
            r = run_peg(offset=off, **kw)
            row[name] = dict(success=r["success"], max_depth=r["max_depth"],
                             peak_lat=r["peak_lat"], mean_lat=r["mean_lat"],
                             peak_axial=r["peak_axial"], t_insert=r["t_insert"])
            cells.append(f"{'OK' if r['success'] else 'FAIL'} "
                         f"{r['max_depth'] * 1e3:5.1f}mm {r['mean_lat']:6.1f}N")
        sweep[off] = row
        print(f"{off * 1e3:<10.1f} " + " ".join(f"{c:>20}" for c in cells))
        sys.stdout.flush()
    res["sweep"] = sweep
    save("peg_in_hole", res)


if __name__ == "__main__":
    t0 = time.perf_counter()
    main()
    print(f"\n[total] {time.perf_counter() - t0:.1f}s")
