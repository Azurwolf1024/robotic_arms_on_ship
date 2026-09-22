"""
论文 IV-A2 的三类参考轨迹(均在**惯性系/世界系**中给出)。

    (1) 定点稳定      fixed_point   10 s
    (2) 圆周跟踪      circle        r = 0.15 m, 30 s
    (3) 8 字跟踪      figure8       x = 0.15 sin(t), y = 0.225 sin(t) cos(t), 30 s

由于目标固定在世界系中, 而基座(船)在动, 控制器必须同时补偿
"几何层面的相对位姿变化"与"惯性层面的动力学耦合"。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Trajectory:
    kind: str
    p0: np.ndarray            # 参考中心(世界系)
    R0: np.ndarray            # 参考姿态
    radius: float = 0.075
    period: float = 6.0
    duration: float = 10.0
    t_ramp: float = 2.0       # 参考速度软启动(避免 t=0 的速度阶跃)
    e1: np.ndarray = None     # 圆周所在平面的两个基向量
    e2: np.ndarray = None

    def __post_init__(self):
        self.p0 = np.asarray(self.p0, dtype=float).reshape(3)
        self.R0 = np.asarray(self.R0, dtype=float).reshape(3, 3)
        if self.e1 is None:
            self.e1 = np.array([1.0, 0.0, 0.0])
        if self.e2 is None:
            # 默认取**水平面**(x-y): 竖直面的大行程会把 Panda 顶到关节限位上,
            # 使对比变成"谁更能撞限位"而不是控制性能。
            self.e2 = np.array([0.0, 1.0, 0.0])
        self.e1 = np.asarray(self.e1, dtype=float).reshape(3)
        self.e2 = np.asarray(self.e2, dtype=float).reshape(3)

    def _phase(self, t: float):
        """
        软启动: 让"沿路径推进的速度"从 0 平滑升到 1, 路径形状不变。

        tau(t) = t^2 / (2 T)                (t < T)      -> dtau/dt = t/T
        tau(t) = t - T/2                    (t >= T)
        """
        T = max(self.t_ramp, 1e-6)
        if t < T:
            tau = t * t / (2.0 * T)
            dtau = t / T
            ddtau = 1.0 / T
        else:
            tau = t - 0.5 * T
            dtau = 1.0
            ddtau = 0.0
        return tau, dtau, ddtau

    def sample(self, t: float) -> dict:
        """返回 R_d, p_d 以及世界系 [omega; v] / [alpha; a] 形式的一阶、二阶导数。"""
        zero6 = np.zeros(6)
        if self.kind == "fixed_point":
            return dict(R_d=self.R0, p_d=self.p0, xd_dot=zero6.copy(), xd_ddot=zero6.copy())

        tau, dtau, ddtau = self._phase(t)

        if self.kind == "circle":
            w = 2.0 * np.pi / self.period
            c, s = np.cos(w * tau), np.sin(w * tau)
            p = self.p0 + self.radius * (c * self.e1 + s * self.e2)
            v = self.radius * w * (-s * self.e1 + c * self.e2) * dtau
            a = (-self.radius * w * w * (c * self.e1 + s * self.e2) * dtau ** 2
                 + self.radius * w * (-s * self.e1 + c * self.e2) * ddtau)
            return dict(R_d=self.R0, p_d=p,
                        xd_dot=np.concatenate([np.zeros(3), v]),
                        xd_ddot=np.concatenate([np.zeros(3), a]))

        if self.kind == "figure8":
            A = self.radius
            st, ct = np.sin(tau), np.cos(tau)
            x = A * st
            y = 1.5 * A * st * ct
            vx = A * ct
            vy = 1.5 * A * (ct * ct - st * st)
            ax = -A * st
            ay = 1.5 * A * (-2.0 * st * ct) * 2.0
            p = self.p0 + x * self.e1 + y * self.e2
            v = (vx * self.e1 + vy * self.e2) * dtau
            a = (ax * self.e1 + ay * self.e2) * dtau ** 2 + (vx * self.e1 + vy * self.e2) * ddtau
            return dict(R_d=self.R0, p_d=p,
                        xd_dot=np.concatenate([np.zeros(3), v]),
                        xd_ddot=np.concatenate([np.zeros(3), a]))

        raise ValueError(f"unknown trajectory: {self.kind}")


def make_task(name: str, p0: np.ndarray, R0: np.ndarray, radius: float = 0.075,
              **kw) -> Trajectory:
    if name == "fixed_point":
        return Trajectory("fixed_point", p0, R0, duration=10.0, **kw)
    if name == "circle":
        # 参考圆心放在末端**前方**(+x): 圆周向 -x 展开会把 Panda 的第 2 关节顶到
        # 机械限位上(基座还在 6-DOF 运动), 使对比失真。
        kw.setdefault("e1", np.array([-1.0, 0.0, 0.0]))
        return Trajectory(name, p0, R0, radius=radius, duration=30.0, **kw)
    if name == "figure8":
        return Trajectory(name, p0, R0, radius=radius, duration=30.0, **kw)
    raise ValueError(name)


def make_task_scaled(name: str, p0: np.ndarray, R0: np.ndarray, scale: float = 1.0, **kw) -> Trajectory:
    """论文 IV-D2 的实机实验把轨迹半径缩小到 0.075 m。"""
    if name == "fixed_point":
        return Trajectory("fixed_point", p0, R0, duration=10.0, **kw)
    r = 0.075 * scale
    return Trajectory(name, p0, R0, radius=r, duration=30.0, **kw)
