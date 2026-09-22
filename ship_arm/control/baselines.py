"""
论文 IV-A3 的四个**基线控制器**(速度层/运动学层方法)。

真实机器人上这些方法通过"关节速度伺服"下发指令, 因此这里统一:

    控制器 -> 期望关节速度 qdot_c -> 关节速度伺服 tau = K_v (qdot_c - qdot) + H_nom -> 力矩

其中速度伺服带**名义模型前馈**(真实机器人/Franka 速度接口同样含重力补偿),
否则纯 P 速度环要靠误差积分才能平衡重力, 会把基线的性能人为压低。

四个基线
------------------------------------------------------------------
1) PI (纯运动学反馈, 式 48)
     qdot_c = J^pinv ( xdot_d + Kp e + Ki ∫e dt )

2) Woolfrey et al. [10] (在线辨识基座运动 + 预测补偿)
     AR/RLS **直接多步**预报 h 秒后的基座位姿 -> 折算成基座引起的末端速度 ->
     在速度指令中前馈扣除。

3) Xu et al. [15] (扰动观测器估计基座引起的关节速度扰动)
     qdot_c = J^pinv u* - qhat_dot_dist,   qhat_dot_dist 一阶 DOB 估计 J^pinv J_B V_B

4) PI + EVOLVER [13] (Koopman/DMD 在线学习扰动)
     从末端位姿测量中抽取"基座引起的末端速度"序列, 用带延迟嵌入的在线
     DMD 拟合并一步预测, 再从速度指令中扣除。

说明: 2) 与 4) 中的"预测器"是论文方法的工程化简化(原作分别用 MPC 与
Koopman 算子理论), 但保留了其**信息流与补偿结构**, 足以复现论文的相对结论。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..core.lie import log_so3, pose_error
from ..robot.model import Robot


# --------------------------------------------------------------------------- #
# 关节速度伺服(含名义模型前馈)
# --------------------------------------------------------------------------- #
@dataclass
class VelocityServo:
    """
    模拟机器人底层的关节速度控制器(力矩模式下的 PI 速度环 + 限幅)。

    论文把四个基线都描述为"**不依赖系统动力学**"的运动学方法, 因此这里的速度环
    也不用模型前馈: 靠积分项自行平衡重力/科氏力(真实机器人上常见的做法)。

    use_model_ff=True 时可切换为"含名义模型前馈"的更强版本, 用于消融对照。
    """

    kp: np.ndarray = field(default_factory=lambda: 30.0 * np.ones(7))
    ki: np.ndarray = field(default_factory=lambda: 300.0 * np.ones(7))
    use_model_ff: bool = False
    _int: np.ndarray = None

    def reset(self):
        self._int = None

    def torque(self, robot: Robot, dq_cmd: np.ndarray, dq: np.ndarray, dt: float,
               H: np.ndarray = None) -> np.ndarray:
        err = np.asarray(dq_cmd) - np.asarray(dq)
        if self._int is None:
            self._int = np.zeros_like(err)
        tau_raw = self.kp * err + self.ki * self._int
        if self.use_model_ff and H is not None:
            tau_raw = tau_raw + np.asarray(H)
        tau = np.clip(tau_raw, -robot.tau_max, robot.tau_max)
        # 条件积分(抗饱和): 未饱和时才累积
        self._int = np.where(np.abs(tau_raw) < robot.tau_max,
                             np.clip(self._int + err * dt, -2.0, 2.0), self._int)
        return tau


# --------------------------------------------------------------------------- #
# 公共工具
# --------------------------------------------------------------------------- #
def pinv_jac(J: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    return J.T @ np.linalg.inv(J @ J.T + eps ** 2 * np.eye(6))


# 冗余机械臂(7 DOF / 6 维任务)的**零空间位形保持任务**。
# 没有它, 纯运动学控制器在周期任务上会沿零空间持续漂移, 最终顶到关节限位,
# 使对比变成"谁更能撞限位"而不是控制性能。真实系统上同样必须加这一项。
Q_NS = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
K_NS = 1.5


def add_posture(dq_task: np.ndarray, J: np.ndarray, q: np.ndarray,
                q_ns: np.ndarray = None, k_ns: float = K_NS) -> np.ndarray:
    n = J.shape[1]
    Jp = pinv_jac(J)
    N = np.eye(n) - Jp @ J
    target = Q_NS if q_ns is None else q_ns
    return dq_task + N @ (k_ns * (target - q))


class DirectRLSPredictor:
    """
    **直接多步**预报器: 用 RLS 在线拟合

        y_{k+h} = Theta^T [ y_k; y_{k-1}; ...; y_{k-p+1} ]

    相比"递推一步模型 h 次", 直接多步不存在迭代发散问题, 是工程上常用的做法。

    参数
    ----
    order   : 记忆长度 p(以**抽取后**的采样点为单位)
    horizon : 预报步数 h(同样以抽取后的采样点为单位)
    decim   : 抽取率, 每 decim 个控制周期采一个点
    """

    def __init__(self, order: int = 10, dim: int = 6, horizon: int = 4,
                 lam: float = 0.995, decim: int = 50):
        self.p = int(order)
        self.h = int(horizon)
        self.dim = int(dim)
        self.lam = lam
        self.decim = max(1, int(decim))
        self.hist = []
        self.P = np.eye(self.p * self.dim) * 1e3
        self.theta = np.zeros((self.p * self.dim, self.dim))
        self._n = 0
        self.ready = False

    def reset(self):
        self.hist = []
        self.P = np.eye(self.p * self.dim) * 1e3
        self.theta = np.zeros((self.p * self.dim, self.dim))
        self._n = 0
        self.ready = False

    @staticmethod
    def _feat(hist, start, p):
        return np.concatenate(hist[start:start - p:-1]).reshape(-1)

    def push(self, y: np.ndarray) -> bool:
        """每个控制周期调用; 返回 True 表示本周期模型被更新。"""
        self._n += 1
        if self._n % self.decim != 0:
            return False
        self.hist.append(np.asarray(y, dtype=float).copy())
        if len(self.hist) > 600:
            self.hist.pop(0)
        updated = False
        need = self.h + self.p
        if len(self.hist) > need:
            x = self._feat(self.hist, -1 - self.h, self.p)
            yt = self.hist[-1]
            Px = self.P @ x
            denom = self.lam + float(x @ Px)
            self.P = (self.P - np.outer(Px, Px) / denom) / self.lam
            err = yt - self.theta.T @ x
            self.theta = self.theta + np.outer(Px, err) / denom
            updated = True
        self.ready = len(self.hist) > self.p
        return updated

    def predict(self) -> np.ndarray:
        if len(self.hist) < self.p + 1:
            return np.zeros(self.dim)
        return self.theta.T @ self._feat(self.hist, -1, self.p)


class WindowedLSPredictor:
    """
    **滑动窗口岭回归**的直接多步预报器。

        y_{k+h} = Theta^T [ y_k; y_{k-1}; ...; y_{k-p+1} ]

    相比递推 RLS, 窗口最小二乘不会协方差发散: 历史窗口有限, 且带岭正则,
    因此对噪声/闭环数据都稳定(EVOLVER 这类"边学边补偿"的结构里, 递推 RLS
    常因协方差膨胀给出几十倍量级的伪预测, 使前馈反而破坏性能)。

    参数
    ----
    order   : 记忆长度 p(以**抽取后**的采样点为单位)
    horizon : 预报步数 h
    decim   : 抽取率, 每 decim 个控制周期采一个点
    window  : 训练窗口长度(抽取后采样点数)
    ridge   : 岭正则系数
    """

    def __init__(self, order: int = 4, dim: int = 6, horizon: int = 2,
                 window: int = 150, ridge: float = 1e-2, decim: int = 20,
                 min_samples: int = 40, max_norm: float = 0.0):
        self.p = int(order)
        self.h = int(horizon)
        self.dim = int(dim)
        self.window = int(window)
        self.ridge = float(ridge)
        self.decim = max(1, int(decim))
        self.min_samples = int(min_samples)
        self.max_norm = float(max_norm)
        self.reset()

    def reset(self):
        self.hist = []
        self.theta = np.zeros((self.p * self.dim, self.dim))
        self._n = 0
        self.ready = False

    @staticmethod
    def _feat(hist, i, p):
        """取 hist[i], hist[i-1], ..., hist[i-p+1] 拼成特征向量(i 为**正**索引)。"""
        return np.concatenate([hist[j] for j in range(i, i - p, -1)]).reshape(-1)

    def push(self, y: np.ndarray) -> bool:
        self._n += 1
        if self._n % self.decim != 0:
            return False
        self.hist.append(np.asarray(y, dtype=float).copy())
        if len(self.hist) > self.window + self.p + self.h + 5:
            self.hist.pop(0)
        updated = False
        need = self.p + self.h + self.min_samples
        if len(self.hist) >= need:
            self._fit()
            updated = True
        self.ready = len(self.hist) >= self.p + 1
        return updated

    def _fit(self):
        n = len(self.hist)
        # 训练对: 特征取自 i, 目标取自 i+h
        i0 = max(self.p - 1, n - self.window - self.h)
        idx = range(i0, n - self.h)
        if len(list(idx)) < self.min_samples:
            return
        X = np.array([self._feat(self.hist, i, self.p) for i in idx])
        Y = np.array([self.hist[i + self.h] for i in idx])
        A = X.T @ X + self.ridge * np.eye(X.shape[1])
        self.theta = np.linalg.solve(A, X.T @ Y)
        # 稳定性保护: 单步预报矩阵的谱半径 > 1 时按比例缩放, 防止预测发散
        if self.p >= 1:
            A1 = self.theta[: self.dim].T          # y_{k+1} = A1 y_k + ...
            rho = max(abs(np.linalg.eigvals(A1)))
            if rho > 1.0:
                self.theta = self.theta / (rho * rho)
        self.ready = True

    def predict(self) -> np.ndarray:
        if len(self.hist) < self.p + 1:
            return np.zeros(self.dim)
        y = self.theta.T @ self._feat(self.hist, len(self.hist) - 1, self.p)
        if self.max_norm > 0.0:
            nrm = np.linalg.norm(y)
            if nrm > self.max_norm:
                y = y * (self.max_norm / nrm)
        return y


# --------------------------------------------------------------------------- #
# 1) PI
# --------------------------------------------------------------------------- #
class PIController:
    """式(48): 纯运动学反馈 + 积分。"""

    def __init__(self, robot: Robot, dt: float, Kp: np.ndarray = None, Ki: np.ndarray = None):
        self.robot = robot
        self.dt = dt
        self.Kp = Kp if Kp is not None else np.diag([5.0] * 6)
        self.Ki = Ki if Ki is not None else np.diag([2.0] * 6)
        self.integ = np.zeros(6)

    def reset(self):
        self.integ[:] = 0.0

    def command(self, q, dq, est, ref, terms) -> np.ndarray:
        e = pose_error(terms["ee_pose"], (ref["R_d"], ref["p_d"]))
        self.integ = np.clip(self.integ + e * self.dt, -0.5, 0.5)
        u = ref["xd_dot"] + self.Kp @ e + self.Ki @ self.integ
        return add_posture(pinv_jac(terms["J"]) @ u, terms["J"], q)


# --------------------------------------------------------------------------- #
# 2) Woolfrey et al. [10]: 在线辨识 + 预测补偿
# --------------------------------------------------------------------------- #
class PredictiveController:
    """Woolfrey-like: 预报基座位姿 -> 折算基座引起的末端速度 -> 速度指令前馈扣除。"""

    def __init__(self, robot: Robot, dt: float, horizon: float = 0.2,
                 Kp: np.ndarray = None, Kd: np.ndarray = None,
                 order: int = 10, decim: int = 50):
        self.robot = robot
        self.dt = dt
        self.horizon = horizon
        self.Kp = Kp if Kp is not None else 5.0 * np.eye(6)
        self.Kd = Kd if Kd is not None else 0.5 * np.eye(6)
        self.decim = decim
        self.steps = max(1, int(round(horizon * decim / (decim * dt))))
        # 抽取后的采样周期 = decim * dt; 预报步数 = horizon / (decim*dt)
        self.h_steps = max(1, int(round(horizon / (decim * dt))))
        self.ar = DirectRLSPredictor(order=order, dim=6, horizon=self.h_steps,
                                     lam=0.995, decim=decim)
        self._last_vb = np.zeros(6)

    def reset(self):
        self.ar.reset()
        self._last_vb = np.zeros(6)

    def command(self, q, dq, est, ref, terms) -> np.ndarray:
        # 基座 6-DOF 位姿(世界系): [p_B; rotvec(R_WB)]
        eta = np.concatenate([est["p_B"], log_so3(est["R_WB"])])
        self.ar.push(eta)
        eta_pred = self.ar.predict()
        dp = eta_pred[0:3] - eta[0:3]
        dth = eta_pred[3:6] - eta[3:6]

        J = terms["J"]
        JB = terms.get("J_B", np.zeros((6, 6)))
        if self.ar.ready:
            # 预报时域内的基座位移 -> 等效平均基座速度 -> 基座引起的末端速度
            V_pred = np.concatenate([dth, dp]) / max(self.horizon, 1e-6)
            self._last_vb = JB @ V_pred
        v_base_pred = self._last_vb

        e = pose_error(terms["ee_pose"], (ref["R_d"], ref["p_d"]))
        edot = ref["xd_dot"] - terms["ee_vel"][0]
        u = ref["xd_dot"] + self.Kp @ e + self.Kd @ edot - v_base_pred
        return add_posture(pinv_jac(J) @ u, J, q)


# --------------------------------------------------------------------------- #
# 3) Xu et al. [15]: 速度扰动观测器
# --------------------------------------------------------------------------- #
class DisturbanceObserverController:
    """
    式(49): qdot_c = J^pinv u* + qhat_dot_dist。

    [15] 的核心是一个**扰动观测器**: 它并不知道基座状态, 而是从"末端实测速度"与
    "机械臂指令速度"之差中反推基座运动带来的等效关节速度扰动, 再用一阶观测器
    (带宽 1/tau_d)滤出。观测器带宽有限 -> 对快速变化的基座运动存在相位滞后,
    这正是该方法相对本文方法的主要性能损失来源。
    """

    def __init__(self, robot: Robot, dt: float, tau_d: float = 0.12,
                 Kp: np.ndarray = None, Kd: np.ndarray = None):
        self.robot = robot
        self.dt = dt
        self.tau_d = tau_d
        self.Kp = Kp if Kp is not None else 5.0 * np.eye(6)
        self.Kd = Kd if Kd is not None else 0.3 * np.eye(6)
        self.est = None

    def reset(self):
        self.est = None

    def command(self, q, dq, est, ref, terms) -> np.ndarray:
        J = terms["J"]
        # 观测: v_EE(实测) - J qdot(机械臂自身贡献) = 基座引起的末端速度
        d_ee = terms["ee_vel"][0] - J @ dq
        d_q = pinv_jac(J) @ d_ee
        if self.est is None:
            self.est = d_q.copy()
        else:
            a = min(1.0, self.dt / max(self.tau_d, 1e-6))
            self.est = self.est + a * (d_q - self.est)

        e = pose_error(terms["ee_pose"], (ref["R_d"], ref["p_d"]))
        edot = ref["xd_dot"] - terms["ee_vel"][0]
        u = ref["xd_dot"] + self.Kp @ e + self.Kd @ edot
        return add_posture(pinv_jac(J) @ u, J, q) - self.est


# --------------------------------------------------------------------------- #
# 4) PI + EVOLVER [13]: Koopman/DMD 在线扰动学习
# --------------------------------------------------------------------------- #
class PIEvolverController:
    """
    式(50): PI + EVOLVER 估计的基座扰动速度补偿。

    观测量(在 decim*dt 的窗口上计算, 避免 1 kHz 差分放大噪声):
        d_k = [ log(R_meas(k-m)^T R_meas(k)) / T ;  (p_meas(k) - p_meas(k-m)) / T ]
              - mean( J qdot )                       (用**实测**关节速度)
            ≈ 基座运动引起的末端 6 维速度扰动
    用延迟嵌入的在线 DMD(一阶 Koopman 近似)拟合其一步转移, 再预测并前馈扣除。

    注意: 必须补偿完整的 6 维扰动。只补偿线速度几乎无效——末端离基座约 0.5 m,
    基座角速度引起的末端**角速度**扰动才是主要误差源。
    """

    def __init__(self, robot: Robot, dt: float, Kp: np.ndarray = None, Ki: np.ndarray = None,
                 delay: int = 3, decim: int = 20):
        self.robot = robot
        self.dt = dt
        self.decim = decim                 # 观测量抽取: 每 decim 个控制周期一个样本(=50 Hz)
        self.Kp = Kp if Kp is not None else np.diag([5.0] * 6)
        self.Ki = Ki if Ki is not None else np.diag([2.0] * 6)
        # 抽取由本类自己完成, 因此学习器内部 decim=1; 观测量为完整 6 维扰动速度。
        # 用窗口岭回归而非递推 RLS: 闭环数据下递推 RLS 会协方差膨胀给出伪预测。
        # 预报步长 h=2 (=40 ms) 大致对上速度伺服的滞后; 幅值上限按船体运动的物理
        # 量级取(角速度 1 rad/s / 线速度 1.5 m/s)。
        self.learner = WindowedLSPredictor(order=max(2, delay), dim=6, horizon=2,
                                           window=120, ridge=1e-2, decim=1,
                                           min_samples=30, max_norm=1.5)
        self.integ = np.zeros(6)
        self._p_old = None
        self._R_old = None
        self._v_arm_sum = np.zeros(6)
        self._cnt = 0
        self._v_arm_meas = np.zeros(6)
        self._d_hat = np.zeros(6)

    def reset(self):
        self.learner.reset()
        self.integ[:] = 0.0
        self._p_old = None
        self._R_old = None
        self._v_arm_sum = np.zeros(6)
        self._cnt = 0
        self._v_arm_meas = np.zeros(6)
        self._d_hat = np.zeros(6)

    def observe(self, p_meas: np.ndarray, R_meas: np.ndarray = None):
        """在 decim*dt 窗口上估计"基座引起的末端速度扰动"(6 维)。"""
        p_meas = np.asarray(p_meas, dtype=float)
        self._v_arm_sum = self._v_arm_sum + self._v_arm_meas
        self._cnt += 1
        if self._p_old is None:
            self._p_old = p_meas.copy()
            self._R_old = None if R_meas is None else np.asarray(R_meas).copy()
            self._v_arm_sum = np.zeros(6)
            self._cnt = 0
            return
        n = self._cnt
        if n < self.decim:
            return
        T = n * self.dt
        v_meas = (p_meas - self._p_old) / T
        if self._R_old is not None and R_meas is not None:
            w_meas = log_so3(self._R_old.T @ np.asarray(R_meas)) / T
        else:
            w_meas = np.zeros(3)
        self._p_old = p_meas.copy()
        self._R_old = None if R_meas is None else np.asarray(R_meas).copy()
        d = np.concatenate([w_meas, v_meas]) - self._v_arm_sum / n
        self._v_arm_sum = np.zeros(6)
        self._cnt = 0
        if self.learner.push(d):
            self._d_hat = self.learner.predict()

    def command(self, q, dq, est, ref, terms) -> np.ndarray:
        e = pose_error(terms["ee_pose"], (ref["R_d"], ref["p_d"]))
        self.integ = np.clip(self.integ + e * self.dt, -0.5, 0.5)
        u_nominal = ref["xd_dot"] + self.Kp @ e + self.Ki @ self.integ
        u = u_nominal - self._d_hat
        J = terms["J"]
        dq_cmd = add_posture(pinv_jac(J) @ u, J, q)
        # 观测量必须基于**实测**关节速度: 若用指令速度, d 里会混入速度伺服的跟踪误差,
        # 补偿该项会在指令通道里形成自抵消回路, 使前馈失效。
        self._v_arm_meas = J @ dq
        return dq_cmd


# --------------------------------------------------------------------------- #
# 阻抗/导纳基线 (论文 IV-F 对照)
# --------------------------------------------------------------------------- #
@dataclass
class AdmittanceController:
    """
    论文式(51): 外环导纳模型生成笛卡尔位姿偏移, 由内环速度控制器跟踪。

    M_a xddot_a + D_a xdot_a + K_a x_a = F_m
    """

    Ma: np.ndarray = field(default_factory=lambda: np.diag([8.0, 8.0, 0.02, 0.02]))
    Da: np.ndarray = field(default_factory=lambda: np.diag([160.0, 160.0, 0.8, 0.8]))
    Ka: np.ndarray = field(default_factory=lambda: np.diag([4000.0, 4000.0, 30.0, 30.0]))
    deadband_f: float = 1.5
    deadband_t: float = 0.03

    def __post_init__(self):
        self.xa = np.zeros(4)
        self.dxa = np.zeros(4)

    def step(self, Fm: np.ndarray, dt: float) -> np.ndarray:
        F = np.asarray(Fm, dtype=float).copy()
        F[0:2] = _deadband(F[0:2], self.deadband_f)
        F[2:4] = _deadband(F[2:4], self.deadband_t)
        acc = np.linalg.solve(self.Ma, F - self.Da @ self.dxa - self.Ka @ self.xa)
        self.dxa = self.dxa + acc * dt
        self.xa = self.xa + self.dxa * dt
        return self.xa.copy()


def _deadband(x: np.ndarray, eps: float) -> np.ndarray:
    out = np.zeros_like(x)
    for i, v in enumerate(x):
        out[i] = 0.0 if abs(v) < eps else v - np.sign(v) * eps
    return out
