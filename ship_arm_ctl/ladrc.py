"""
LADRC / ESO 自抗扰补偿 —— 云台(稳定平台)控制的主流方法。

背景
----
船载机械臂的基座是一个 6-DOF 运动平台(本质就是一台"云台"), 其高频、大幅度的
运动通过动力学耦合把扰动灌进机械臂末端。工业界做云台稳像/稳平台最常用的是
韩京清的 ADRC, 以及其线性化版本 LADRC: 用一个扩张状态观测器 (ESO) 在线估计
"总扰动"(基座耦合残差 + 模型失配 + 摩擦), 然后在前馈里把它抵消掉。

这里把 LADRC 放在 *任务空间* (末端 6-DOF 位姿) 上实现, 与论文的 TSID 框架天然契合:
    y  = [log(R_ee); p_ee]         —— 被控输出(世界系 6D 位姿)
    r  = [log(R_d);  p_d]          —— 参考
    u_des = TSID 的 xddot_c        —— 名义 PD 加速度指令
    yddot ≈ z3 + b·u               —— 把"总扰动" z3 扩张为第三个状态

每个控制周期:
    eo   = z1 − y
    z1  += dt·(z2 − β1·eo)
    z2  += dt·(z3 − β2·eo + b·u_prev)
    z3  += dt·(−β3·eo)
    u    = u_des − z3 / b           —— 抵消扰动后的加速度指令
    accel_ff = u − u_des = −z3/b    —— 返回给 TSID 作为前馈加进 xddot_c

β1=3ω, β2=3ω², β3=ω³ (ω 为 ESO 带宽)。该补偿与论文的 *解析* 基座耦合前馈(TSID
的 τ_base)互补: TSID 处理已知部分, LADRC 在线补偿其估计残差; 神经网络则是把这部分
*离线*学下来做成前馈。三者可以叠加使用。
"""

from __future__ import annotations

import numpy as np


class LadrcCompensator:
    """任务空间 6-DOF LADRC 补偿器。"""

    def __init__(self, dt: float = 1e-3, wo: float = 15.0, b0: float = 1.0,
                 accel_clip: float = 18.0):
        self.dt = dt
        self.wo = wo
        self.b0 = b0
        self.accel_clip = accel_clip
        self.beta1 = 3.0 * wo
        self.beta2 = 3.0 * wo ** 2
        self.beta3 = wo ** 3
        self._z = None      # (6, 3): z1(输出), z2(速度), z3(扰动)
        self._u_prev = np.zeros(6)
        self._initialized = False

    def reset(self):
        self._z = None
        self._u_prev = np.zeros(6)
        self._initialized = False

    def _ensure(self, y: np.ndarray):
        if self._z is None:
            self._z = np.zeros((6, 3))
            self._z[:, 0] = np.asarray(y, dtype=float)
            self._initialized = True

    def step(self, y_meas: np.ndarray, r: np.ndarray, u_des: np.ndarray) -> np.ndarray:
        """更新 ESO 并返回任务空间加速度前馈 accel_ff (6,)。

        参数
        ----
        y_meas : 当前末端 6D 位姿测量 [log(R); p]
        r      : 参考 6D 位姿 [log(R_d); p_d]  (本实现内部不直接用到, u_des 已含)
        u_des  : 名义加速度指令 (TSID xddot_c), 形状 (6,)
        """
        y = np.asarray(y_meas, dtype=float)
        self._ensure(y)
        dt = self.dt
        z = self._z
        eo = z[:, 0] - y
        z[:, 0] = z[:, 0] + dt * (z[:, 1] - self.beta1 * eo)
        z[:, 1] = z[:, 1] + dt * (z[:, 2] - self.beta2 * eo + self.b0 * self._u_prev)
        z[:, 2] = z[:, 2] + dt * (-self.beta3 * eo)
        # 限幅, 防止观测器发散把 accel_ff 推到荒谬值
        z[:, 2] = np.clip(z[:, 2], -self.accel_clip * 3, self.accel_clip * 3)
        u = np.asarray(u_des, dtype=float) - z[:, 2] / self.b0
        self._u_prev = u.copy()
        accel_ff = np.clip(u - np.asarray(u_des, dtype=float),
                           -self.accel_clip, self.accel_clip)
        return accel_ff

    # 兼容接口: 直接给末端位姿 (R, p) 与参考 (R_d, p_d)
    def step_pose(self, R_ee, p_ee, R_d, p_d, u_des) -> np.ndarray:
        from ship_arm.core.lie import pose_error
        y = pose_error((np.eye(3), np.zeros(3)), (R_ee, p_ee))  # 以世界原点为参考的 6D 位姿
        r = pose_error((np.eye(3), np.zeros(3)), (R_d, p_d))
        return self.step(y, r, u_des)
