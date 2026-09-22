"""
传感器仿真: 基座 IMU 与 动捕/视觉末端位姿反馈。

论文硬件配置(IV-D1):
    * HiPNUC IMU, 100 Hz, 刚性安装在基座上, 测量体坐标角速度与线加速度
    * 光学动捕, 120 Hz, 提供末端与基座在世界系中的位姿(本文用作"真值"与末端反馈)
本模块加入**零偏随机游走 + 白噪声**, 以便复现论文 Table V 的估计精度量级。

IMU 观测模型严格对应论文式(43):

    z_imu = [ a_meas ]   = [ a_B^B + R_BW g_W + b_a + n_a ]
            [ w_meas ]     [ omega_B^B      + b_w + n_w ]
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..core.lie import quat_to_rot, rot_to_quat
from ..robot.model import GRAVITY


@dataclass
class IMUSim:
    """100 Hz 基座 IMU。"""

    rate: float = 100.0
    sigma_a: float = 0.02          # 加速度噪声 (m/s^2, 1-sigma)
    sigma_w: float = 0.004         # 角速度噪声 (rad/s)
    sigma_ba: float = 2e-4         # 加速度零偏随机游走
    sigma_bw: float = 1e-4
    b_a: np.ndarray = field(default_factory=lambda: np.array([0.03, -0.02, 0.05]))
    b_w: np.ndarray = field(default_factory=lambda: np.array([0.002, -0.001, 0.003]))
    seed: int = 11
    _rng: np.random.Generator = None
    _last: float = -1.0

    def __post_init__(self):
        if self._rng is None:
            self._rng = np.random.default_rng(self.seed)

    def sample(self, t: float, body_state: dict) -> dict | None:
        """按固定频率采样; 未到采样时刻返回 None(模拟异步多速率)。"""
        if t < self._last + 1.0 / self.rate - 1e-12:
            return None
        self._last = t
        dt = 1.0 / self.rate
        # 零偏随机游走
        self.b_a = self.b_a + self.sigma_ba * np.sqrt(dt) * self._rng.normal(size=3)
        self.b_w = self.b_w + self.sigma_bw * np.sqrt(dt) * self._rng.normal(size=3)

        R = body_state["R_WB"]
        RBW = R.T
        a_meas = body_state["a_b"] + RBW @ GRAVITY + self.b_a + self.sigma_a * self._rng.normal(size=3)
        w_meas = body_state["omega_b"] + self.b_w + self.sigma_w * self._rng.normal(size=3)
        return {"t": t, "acc": a_meas, "gyro": w_meas}


@dataclass
class PoseSensorSim:
    """
    末端位姿反馈(论文中由动捕/eye-in-hand 提供)。120 Hz, 世界系。
    """

    rate: float = 120.0
    sigma_p: float = 1e-4          # 0.1 mm
    sigma_r: float = 1e-4          # ~0.006 deg
    seed: int = 22
    _rng: np.random.Generator = None
    _last: float = -1.0

    def __post_init__(self):
        if self._rng is None:
            self._rng = np.random.default_rng(self.seed)

    def sample(self, t: float, R_ee: np.ndarray, p_ee: np.ndarray) -> dict | None:
        if t < self._last + 1.0 / self.rate - 1e-12:
            return None
        self._last = t
        p = p_ee + self.sigma_p * self._rng.normal(size=3)
        dR = _rotvec_to_R(self.sigma_r * self._rng.normal(size=3))
        R = dR @ R_ee
        return {"t": t, "R": R, "p": p, "quat": rot_to_quat(R)}


@dataclass
class BasePoseSensorSim:
    """可选的"直接基座位姿"传感器(论文 Table V 的 direct base pose 配置)。"""

    rate: float = 120.0
    sigma_p: float = 5e-5
    sigma_r: float = 5e-5
    seed: int = 33
    _rng: np.random.Generator = None
    _last: float = -1.0

    def __post_init__(self):
        if self._rng is None:
            self._rng = np.random.default_rng(self.seed)

    def sample(self, t: float, R_B: np.ndarray, p_B: np.ndarray) -> dict | None:
        if t < self._last + 1.0 / self.rate - 1e-12:
            return None
        self._last = t
        p = p_B + self.sigma_p * self._rng.normal(size=3)
        R = _rotvec_to_R(self.sigma_r * self._rng.normal(size=3)) @ R_B
        return {"t": t, "R": R, "p": p, "quat": rot_to_quat(R)}


def _rotvec_to_R(v: np.ndarray) -> np.ndarray:
    from ..core.lie import exp_so3

    return exp_so3(v)


def sync_body_state(ship_body_state: dict) -> dict:
    """把 ShipMotion.body_state 的输出整理成 IMU 需要的字段。"""
    return ship_body_state
