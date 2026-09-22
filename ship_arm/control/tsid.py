"""
论文第 II 节: 基于优化的任务空间逆动力学 (TSID) 力矩控制器。

对应公式
------------------------------------------------------------------
(6)  xddot_c = xddot_d + Kp e + Kd edot,   e = log(x^{-1} xd)
(7)  J qddot = xddot_c - eta,   eta = Jdot qdot + J_B Vdot_B + Jdot_B V_B
(8)  qddot_ns = Kp_ns (q_ns - q) - Kd_ns qdot
(9)  关节位置/速度限 -> 加速度上下界
(10) min  0.5||J qddot - xddot_c + eta||^2 + 0.5*lam*||N(qddot - qddot_ns)||^2
     s.t. tau_min <= M qddot + H <= tau_max
          qddot_min <= qddot <= qddot_max
(12) tau* = M qddot* + H

工程实现要点
------------------------------------------------------------------
* 基座状态(p_B, R_WB, V_B, Vdot_B)来自 ESKF 估计值, **不是真值**;
  而 H 中显式包含的 tau_base 正是论文"前馈补偿动态耦合"的关键(式 2)。
* 提供若干开关用于**消融实验**:
    compensate_base   : 是否把基座运动项放进 H(关闭后即为"忽略动态耦合"的对照组)
    use_base_pose     : 是否用估计基座姿态做 FK/重力(关闭后相当于认为基座静止)
    impedance / nullspace / torque_limits 等。
* 记录 delta_c = Lambda J (qddot* - qddot_0), 用于论文 IV-B1 的约束激活分析。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..core.lie import pose_error
from ..qp.dense_qp import QPResult, solve_qp
from ..robot.model import Robot


@dataclass
class TSIDGains:
    """论文 Table I 的默认增益。"""

    Kp: np.ndarray = field(default_factory=lambda: 500.0 * np.eye(6))
    Kd: np.ndarray = field(default_factory=lambda: 40.0 * np.eye(6))
    Kp_ns: np.ndarray = field(default_factory=lambda: 50.0 * np.eye(7))
    Kd_ns: np.ndarray = field(default_factory=lambda: 20.0 * np.eye(7))
    lam: float = 0.001
    q_ns: np.ndarray = field(default_factory=lambda: np.array(
        [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]
    ))


@dataclass
class TSIDOptions:
    compensate_base: bool = True      # 前馈补偿 tau_base (论文核心)
    use_base_pose: bool = True        # FK/重力项使用估计基座姿态
    use_nullspace: bool = True
    torque_limits: bool = True
    accel_limits: bool = True
    mu_task: float = 1.0              # 任务项权重(论文中为 1)
    pinv_eps: float = 1e-6            # 阻尼伪逆


class TSIDController:
    def __init__(self, robot: Robot, dt: float, gains: TSIDGains = None,
                 options: TSIDOptions = None):
        self.robot = robot
        self.dt = dt
        self.gains = gains or TSIDGains()
        self.opt = options or TSIDOptions()
        self._prev_qdd = np.zeros(robot.n)
        self.stats = {"feasible": 0, "fallback": 0, "active": 0}

    # ------------------------------------------------------------------ #
    def _saturate(self, qdd0: np.ndarray, M: np.ndarray, H: np.ndarray,
                  lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
        """
        QP 不可行时的确定性兜底解。

        1) 把自由解投影到加速度盒 [lo, hi]; 盒子为空时取中点(或 0)。
        2) 用 tau = M qdd + H 折算力矩并按 tau_max 饱和;
        3) 由饱和后的力矩反解 qdd, 再投影回加速度盒。
        这样输出永远有限、且在物理上就是"力矩/加速度双重饱和"下的可行解。
        """
        lo = np.asarray(lo, dtype=float)
        hi = np.asarray(hi, dtype=float)
        mid = np.where(np.isfinite(lo) & np.isfinite(hi), 0.5 * (lo + hi), 0.0)
        lo_s = np.minimum(lo, mid)
        hi_s = np.maximum(hi, mid)
        x = np.clip(np.nan_to_num(qdd0, nan=0.0, posinf=0.0, neginf=0.0), lo_s, hi_s)
        tmax = self.robot.tau_max
        tau = np.clip(M @ x + H, -tmax, tmax)
        try:
            x = np.linalg.solve(M, tau - H)
        except np.linalg.LinAlgError:
            x = np.linalg.lstsq(M, tau - H, rcond=None)[0]
        x = np.clip(x, lo_s, hi_s)
        return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    # ------------------------------------------------------------------ #
    def _base_rows(self, est) -> np.ndarray:
        """把估计的基座状态组装成 state_terms 需要的行 [w; v; alpha; a](世界系)。"""
        rows = np.zeros((1, 12))
        if not self.opt.compensate_base:
            return rows
        rows[0, 0:3] = est["omega_w"]
        rows[0, 3:6] = est["v_w"]
        rows[0, 6:9] = est["alpha_w"]
        rows[0, 9:12] = est["a_w"]
        return rows

    # ------------------------------------------------------------------ #
    def compute(self, q: np.ndarray, dq: np.ndarray, est: dict, ref: dict,
                fext: Optional[np.ndarray] = None, accel_ff: Optional[np.ndarray] = None,
                ladrc=None, ee_pose_meas=None) -> dict:
        """
        参数
        ----
        q, dq      : 关节角/角速度(编码器反馈)
        est        : ESKF 输出的基座状态, 需含
                     R_WB, p_B, omega_w, v_w, alpha_w, a_w
        ref        : 参考轨迹, 需含 R_d, p_d, xd_dot(6), xd_ddot(6)
        fext       : 可选的外部力旋量(若控制器做力补偿, 一般不给)
        """
        robot = self.robot
        n = robot.n
        opt = self.opt

        base_pose = (est["R_WB"], est["p_B"]) if opt.use_base_pose else (np.eye(3), np.zeros(3))
        rows = self._base_rows(est)

        terms = robot.state_terms(base_pose, q, dq, rows, want_M=True, want_J=True)
        M = terms["M"]
        H = terms["tau"][0]
        J = terms["J"]
        eta = terms["eta"][0]
        R_ee, p_ee = terms["ee_pose"]
        xdot = terms["ee_vel"][0]

        # ---- 任务空间 PD (论文式 5-6) ----
        x = (R_ee, p_ee)
        xd = (ref["R_d"], ref["p_d"])
        e = pose_error(x, xd)
        edot = ref["xd_dot"] - xdot
        xddot_c = ref["xd_ddot"] + self.gains.Kp @ e + self.gains.Kd @ edot

        # ---- 可选 LADRC / ESO 在线扰动补偿 (云台稳定常用方法) ----
        # 用名义 PD 加速度指令 u_des 驱动 ESO, 估计总扰动后前馈抵消。
        if ladrc is not None and ee_pose_meas is not None:
            af = ladrc.step_pose(ee_pose_meas[0], ee_pose_meas[1],
                                 ref["R_d"], ref["p_d"], xddot_c)
            xddot_c = xddot_c + af

        # ---- 外部加速度前馈 (例如更高层导纳外环 / 预规划) ----
        if accel_ff is not None:
            xddot_c = xddot_c + np.asarray(accel_ff, dtype=float)

        # ---- 零空间任务 (式 8) ----
        if opt.use_nullspace:
            qdd_ns = self.gains.Kp_ns @ (self.gains.q_ns - q) - self.gains.Kd_ns @ dq
        else:
            qdd_ns = np.zeros(n)

        # ---- 伪逆与零空间投影 ----
        Jt = J.T
        JJt = J @ Jt + (opt.pinv_eps ** 2) * np.eye(6)
        Jpin = Jt @ np.linalg.inv(JJt)
        N = np.eye(n) - Jpin @ J

        # ---- QP (式 10 / 13-16) ----
        lam = self.gains.lam if opt.use_nullspace else 0.0
        Q = opt.mu_task * (Jt @ J) + lam * (N.T @ N)
        rhs_task = xddot_c - eta
        cvec = -(opt.mu_task * (Jt @ rhs_task) + lam * (N.T @ N) @ qdd_ns)

        qdd_lo, qdd_hi = robot.accel_bounds(q, dq, self.dt)
        if not opt.accel_limits:
            qdd_lo = np.full(n, -1e6)
            qdd_hi = np.full(n, 1e6)

        Gblocks = []
        hblocks = []
        if opt.torque_limits:
            tmax = robot.tau_max
            Gblocks += [M, -M]
            hblocks += [tmax - H, tmax + H]
        Gblocks += [np.eye(n), -np.eye(n)]
        hblocks += [qdd_hi, -qdd_lo]
        G = np.vstack(Gblocks)
        h = np.concatenate(hblocks)

        qdd0 = Jpin @ rhs_task + N @ qdd_ns          # 无约束解 (式 17)
        res = solve_qp(Q, cvec, G, h, x0=qdd0, max_iter=40)
        qdd = res.x
        if res.status == "optimal" and np.all(np.isfinite(qdd)):
            self.stats["feasible"] += 1
        else:
            # 内点法不收敛/QP 不可行(例如关节已顶到限位、力矩限与加速度限冲突)。
            # 此时必须给出一个**有限且物理合理**的解: 先按加速度限饱和, 再按力矩限饱和,
            # 最后重新投影回加速度盒 —— 直接返回内点法的迭代值会给出 1e13 量级的垃圾,
            # 足以在 1 kHz 下把整条仿真炸掉。
            qdd = self._saturate(qdd0, M, H, qdd_lo, qdd_hi)
            res = QPResult(x=qdd, status="fallback", iters=res.iters, obj=0.0,
                           active=(G @ qdd - h) > -1e-7, lam=np.zeros(h.size),
                           kkt_err=res.kkt_err)
            self.stats["fallback"] += 1
        self.stats["active"] += int(res.active.sum())

        tau = M @ qdd + H

        # ---- 约束激活引起的等效任务空间力 delta_c (论文 IV-B1) ----
        dqdd_c = qdd - qdd0
        Lambda_inv = J @ np.linalg.inv(M) @ J.T
        Lambda = np.linalg.inv(Lambda_inv + 1e-9 * np.eye(6))
        delta_c = Lambda @ J @ dqdd_c

        return {
            "tau": tau,
            "qdd": qdd,
            "qdd0": qdd0,
            "e": e,
            "edot": edot,
            "xddot_c": xddot_c,
            "eta": eta,
            "J": J,
            "M": M,
            "H": H,
            "delta_c": delta_c,
            "active": res.active,
            "status": res.status,
            "ee_pose": (R_ee, p_ee),
            "ee_vel": xdot,
        }
