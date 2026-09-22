"""
实验公共工具: 统一的机器人/船体/任务构造与结果存取。
"""

from __future__ import annotations

import os
import pickle
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "output")
os.makedirs(OUT, exist_ok=True)

from ship_arm.control.tsid import TSIDGains, TSIDOptions          # noqa: E402
from ship_arm.estimation.eskf import ESKFConfig                   # noqa: E402
from ship_arm.platform.ship import ShipMotion                     # noqa: E402
from ship_arm.robot.model import Robot                            # noqa: E402
from ship_arm.robot.panda import PANDA_HOME, make_panda, perturb_spec  # noqa: E402
from ship_arm.sim.engine import (                                 # noqa: E402
    SimOptions, Simulator, estimation_metrics, tracking_metrics,
)
from ship_arm.sim.tasks import make_task                          # noqa: E402

METHODS = ["tsid", "pi", "woolfrey", "xu", "evolver"]
METHOD_LABEL = {
    "tsid": "Ours (TSID-QP)",
    "pi": "PI",
    "woolfrey": "[10] Woolfrey",
    "xu": "[15] Xu",
    "evolver": "[13] PI+EVOLVER",
}
# 被控对象与名义模型之间的失配(论文实机必然存在; 仿真里显式加入才公平)
MISMATCH = dict(model_error=0.12, friction=0.15, coulomb=0.3)


def build_robot(payload: float = 0.0, model_error: float = 0.0, seed: int = 7) -> Robot:
    spec = make_panda(tool_mass=0.73, payload_mass=payload)
    if model_error > 0.0:
        spec = perturb_spec(spec, rel_mass=model_error, rel_inertia=1.6 * model_error, seed=seed)
    return Robot(spec)


def make_ship(scale: float = 1.0) -> ShipMotion:
    ship = ShipMotion()
    ship.scale = scale
    return ship


def run_case(task: str, method: str, duration: float, ship: ShipMotion,
             payload: float = 0.0, model_error: float = 0.12, friction: float = 0.15,
             coulomb: float = 0.3, use_estimator: bool = True, dt: float = 1e-3,
             eskf_cfg: ESKFConfig = None, tsid_opts: TSIDOptions = None,
             gains: TSIDGains = None, q0: np.ndarray = None, contact=None,
             admittance=None, force_noise: float = 0.0, log_every: int = 10,
             ship_scale: float = None, seed: int = 1, tau_scale: float = 1.0,
             task_kw: dict = None, ship_time_scale: float = None,
             nn_comp=None, ladrc=None) -> dict:
    """跑一条实验; 返回 {'metrics':..., 'est':..., 'wall':..., 'log':(降采样数组)}。"""
    if ship_scale is not None or ship_time_scale is not None:
        ship = make_ship(ship_scale if ship_scale is not None else 1.0)
        ship.time_scale = ship_time_scale if ship_time_scale is not None else 1.0
    robot_c = build_robot(0.0)
    if tau_scale != 1.0:
        robot_c.tau_max = robot_c.tau_max * tau_scale
    robot_t = build_robot(payload, model_error=model_error)
    task_obj = make_task(task, np.zeros(3), np.eye(3), **(task_kw or {}))
    task_obj.duration = duration

    opts = SimOptions(dt=dt, duration=duration, controller=method,
                      eskf_cfg=eskf_cfg or ESKFConfig(),
                      payload_mass=payload, model_error=model_error,
                      friction=friction, coulomb=coulomb,
                      use_estimator=use_estimator, contact=contact,
                      admittance=admittance, force_noise=force_noise,
                      log_every=log_every, seed=seed,
                      nn_comp=nn_comp, ladrc=ladrc)
    sim = Simulator(robot_c, robot_t, ship, task_obj, opts,
                    q0=PANDA_HOME.copy() if q0 is None else q0,
                    gains=gains, tsid_opts=tsid_opts)
    t0 = time.perf_counter()
    log = sim.run()
    wall = time.perf_counter() - t0
    return dict(metrics=tracking_metrics(log),
                est=estimation_metrics(log),
                wall=wall,
                cpu_ms=float(np.mean(log.cpu_ms)) if len(log.cpu_ms) else 0.0,
                log=log.arrays(),
                qp_stats=dict(sim.tsid.stats))


def save(name: str, obj):
    path = os.path.join(OUT, name + ".pkl")
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    print(f"[saved] {path}")
    return path


def load(name: str):
    path = os.path.join(OUT, name + ".pkl")
    with open(path, "rb") as f:
        return pickle.load(f)


def fmt_row(label: str, m: dict) -> str:
    return (f"{label:<20} {m['pos_mean']:9.3f} {m['pos_std']:8.3f} {m['pos_max']:8.3f} | "
            f"{m['rot_mean']:9.3f} {m['rot_std']:8.3f} {m['rot_max']:8.3f}")
