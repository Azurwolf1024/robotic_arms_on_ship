"""
论文 IV-F(动态插孔)使用的 **六维导纳外环**。

    M_a xddot_a + D_a xdot_a + K_a x_a = F_m          (论文式 51)

与 4 维(平面)导纳不同, 插孔需要:
    * 侧向 x/y **柔顺**(K 小): 让倒角把插销"导"进孔里, 而不是硬顶;
    * 绕 x/y 的**姿态柔顺**: 释放卡滞时产生的力矩, 避免楔死(jamming);
    * 插入方向 z **刚硬**: 保证能推进到孔底。

输出 x_a 是**世界系**的位姿偏移 [dp(3); dtheta(3)], 直接叠加到参考位姿上。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Admittance6:
    Ma: np.ndarray = field(default_factory=lambda: np.diag([6.0, 6.0, 6.0, 0.05, 0.05, 0.05]))
    Da: np.ndarray = field(default_factory=lambda: np.diag([120.0, 120.0, 400.0, 1.2, 1.2, 1.2]))
    Ka: np.ndarray = field(default_factory=lambda: np.diag([500.0, 500.0, 8000.0, 25.0, 25.0, 25.0]))
    deadband_f: float = 1.0        # N
    deadband_t: float = 0.05       # Nm
    limit_p: float = 0.05          # 位姿偏移上限 (m)
    limit_r: float = 0.12          # 姿态偏移上限 (rad)

    def __post_init__(self):
        self.xa = np.zeros(6)
        self.dxa = np.zeros(6)

    def reset(self):
        self.xa[:] = 0.0
        self.dxa[:] = 0.0

    def step(self, Fm: np.ndarray, dt: float) -> np.ndarray:
        F = np.asarray(Fm, dtype=float).copy()
        F[0:3] = _deadband(F[0:3], self.deadband_f)
        F[3:6] = _deadband(F[3:6], self.deadband_t)
        acc = np.linalg.solve(self.Ma, F - self.Da @ self.dxa - self.Ka @ self.xa)
        self.dxa = self.dxa + acc * dt
        self.xa = self.xa + self.dxa * dt
        self.xa[0:3] = np.clip(self.xa[0:3], -self.limit_p, self.limit_p)
        self.xa[3:6] = np.clip(self.xa[3:6], -self.limit_r, self.limit_r)
        return self.xa.copy()


def _deadband(x: np.ndarray, eps: float) -> np.ndarray:
    if eps <= 0.0:
        return x
    out = np.zeros_like(x)
    for i, v in enumerate(x):
        out[i] = 0.0 if abs(v) < eps else v - np.sign(v) * eps
    return out
