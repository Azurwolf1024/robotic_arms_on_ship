"""
论文第 III 节: 基于误差状态卡尔曼滤波(ESKF)的基座状态估计。

状态定义(式 28-29)
------------------------------------------------------------------
标称状态 X:
    [ p_B^W(3), q_B^W(4), v_B^B(3), omega_B^B(3), a_B^B(3), alpha_B^B(3),
      b_a(3), b_w(3), p_E^W(3), q_E^W(4) ]

误差状态 dX in R^30 (本文件的索引约定):
    0:3   dp_B      3:6  dtheta_B    6:9   dv_B      9:12  domega_B
    12:15 da_B     15:18 dalpha_B   18:21 db_a     21:24 db_w
    24:27 dp_E     27:30 dtheta_E

预测(式 31-36): 不含 IMU 驱动的"常加速度模型", IMU 作为**观测**进入更新;
    这样便于数据平滑并直接估计基座加速度(论文 III-C1 的设计要点)。
更新(式 37-47): 多速率异步
    * IMU  100 Hz : z_imu  = [a_B + R_BW g + b_a ; omega_B + b_w]
    * 位姿 120 Hz : z_pose = [p_E,meas; q_E,meas; p_B,fk; q_B,fk]
        其中 (p_B,fk, q_B,fk) 由"末端位姿观测 + 机械臂正运动学"反推得到,
        它提供绝对位姿约束、抑制长期漂移, 但噪声大(R_fk >> R_E)。
    * 可选"直接基座位姿"观测(论文 Table V 的前三行对照配置)。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..core.lie import (
    exp_so3,
    log_so3,
    quat_exp,
    quat_mul,
    quat_normalize,
    quat_to_rot,
    rot_to_quat,
    skew,
)
from ..robot.model import GRAVITY

# 误差状态分块
IDX_PB, IDX_THB, IDX_VB, IDX_WB = 0, 3, 6, 9
IDX_AB, IDX_ALB, IDX_BA, IDX_BW = 12, 15, 18, 21
IDX_PE, IDX_THE = 24, 27
N_ERR = 30


@dataclass
class ESKFConfig:
    """论文 Table I 的估计器参数。"""

    Qd_blocks: tuple = (1e-9, 1e-8, 1e-4, 1e-5, 1.5e-3, 2e-3, 1e-4, 1e-10, 6e-6, 1e-6)
    R_imu: tuple = (3e-2, 2e-4)
    R_pose: tuple = (1e-8, 1e-8, 5e-4, 5e-4)     # (p_E, theta_E, p_Bfk, theta_Bfk)
    R_base: tuple = (1e-8, 1e-8)                  # 直接基座位姿观测(若启用)
    P0: float = 1e-3
    use_imu: bool = True
    use_ee_pose: bool = True
    use_fk_base_pose: bool = True
    use_direct_base_pose: bool = False
    augmented: bool = True                        # False -> 缩维 ESKF(论文 Table V 的对照)

    def Qd(self) -> np.ndarray:
        return np.diag(np.repeat(np.array(self.Qd_blocks, dtype=float), 3))


class ESKF:
    def __init__(self, cfg: ESKFConfig = None, n_err: int = N_ERR):
        self.cfg = cfg or ESKFConfig()
        self.aug = self.cfg.augmented
        # 缩维状态: 去掉 omega_B / a_B / alpha_B
        self.dim = N_ERR if self.aug else N_ERR - 9
        self.P = np.eye(self.dim) * self.cfg.P0

        # 标称状态
        self.p_B = np.zeros(3)
        self.q_B = np.array([1.0, 0.0, 0.0, 0.0])
        self.v_B = np.zeros(3)
        self.w_B = np.zeros(3)
        self.a_B = np.zeros(3)
        self.al_B = np.zeros(3)
        self.b_a = np.zeros(3)
        self.b_w = np.zeros(3)
        self.p_E = np.zeros(3)
        self.q_E = np.array([1.0, 0.0, 0.0, 0.0])

        # 缩维 ESKF 的辅助一阶滤波(论文: 用辅助滤波器补回被删掉的状态)
        self._w_filt = np.zeros(3)
        self._a_filt = np.zeros(3)
        self._al_filt = np.zeros(3)

    # ------------------------------------------------------------------ #
    # 索引映射(缩维时把原始块号映射到压缩后的索引)
    # ------------------------------------------------------------------ #
    def _blk(self, blk: int) -> int:
        if self.aug:
            return blk * 3          # 每个块占 3 维
        # 原始块号: 0 p_B, 1 th_B, 2 v_B, 3 w_B, 4 a_B, 5 al_B, 6 b_a, 7 b_w, 8 p_E, 9 th_E
        drop = {3, 4, 5}
        if blk in drop:
            return None
        shift = sum(1 for d in drop if d < blk)
        return (blk - shift) * 3

    def _put(self, H: np.ndarray, rows: slice, blk: int, M: np.ndarray):
        j = self._blk(blk)
        if j is None or M is None:
            return
        H[rows, j : j + 3] = M

    # ------------------------------------------------------------------ #
    def reset(self, R_WB: np.ndarray, p_B: np.ndarray, p_E: np.ndarray, R_E: np.ndarray):
        self.p_B = np.asarray(p_B, dtype=float).copy()
        self.q_B = rot_to_quat(R_WB)
        self.p_E = np.asarray(p_E, dtype=float).copy()
        self.q_E = rot_to_quat(R_E)
        self.v_B[:] = 0.0
        self.w_B[:] = 0.0
        self.a_B[:] = 0.0
        self.al_B[:] = 0.0

    # ------------------------------------------------------------------ #
    # 预测 (式 33-36)
    # ------------------------------------------------------------------ #
    def predict(self, dt: float, arm: dict):
        """
        arm: 机械臂相对基座的量(由 FK 得到)
            p_E_B     : 末端在基座系中的位置
            v_arm_B   : 末端相对基座运动的线速度(基座系)
            w_arm_B   : 末端相对基座运动的角速度(基座系)
        """
        R = quat_to_rot(self.q_B)
        w, v, a, al = self.w_B, self.v_B, self.a_B, self.al_B

        # --- 标称状态传播(式 33) ---
        self.p_B = self.p_B + R @ v * dt
        self.q_B = quat_normalize(quat_mul(self.q_B, quat_exp(w * dt)))
        self.v_B = v + (a - np.cross(w, v)) * dt
        self.w_B = w + al * dt

        p_EB = np.asarray(arm["p_E_B"], dtype=float)
        v_arm = np.asarray(arm["v_arm_B"], dtype=float)
        w_arm = np.asarray(arm["w_arm_B"], dtype=float)
        self.p_E = self.p_E + R @ (v + np.cross(w, p_EB) + v_arm) * dt
        R_EB = quat_to_rot(self.q_E).T @ quat_to_rot(self.q_B)   # R_EB = R_EW R_WB
        self.q_E = quat_normalize(quat_mul(self.q_E, quat_exp(R_EB @ (w + w_arm) * dt)))

        # --- 误差状态雅可比(式 34) ---
        F = np.zeros((self.dim, self.dim))
        I3 = np.eye(3)

        def setb(r_blk: int, c_blk: int, M):
            i = self._blk(r_blk)
            j = self._blk(c_blk)
            if i is None or j is None or M is None:
                return
            F[i : i + 3, j : j + 3] = M

        setb(0, 2, R)
        setb(0, 1, -R @ skew(v))
        setb(1, 1, -skew(w))
        setb(1, 3, I3)
        setb(2, 4, I3)
        setb(2, 2, -skew(w))
        setb(2, 3, skew(v))
        setb(3, 5, I3)
        # a_B / alpha_B / b_a / b_w: 常值
        # p_E
        tmp = v + np.cross(w, p_EB) + v_arm
        setb(8, 2, R)
        setb(8, 3, -R @ skew(p_EB))
        setb(8, 1, -R @ skew(tmp))
        # theta_E
        setb(9, 3, R_EB)
        setb(9, 9, -skew(R_EB @ (w + w_arm)))

        Phi = np.eye(self.dim) + F * dt
        Qd = self.cfg.Qd()
        if not self.aug:
            keep = [i for b in range(10) if self._blk(b) is not None for i in range(self._blk(b), self._blk(b) + 3)]
            Qd = Qd[np.ix_(keep, keep)]
        self.P = Phi @ self.P @ Phi.T + Qd * dt
        self.P = 0.5 * (self.P + self.P.T)

        if not self.aug:
            # 缩维 ESKF: 用辅助一阶滤波补回 omega / a / alpha
            tau = 0.08
            self._w_filt = self._w_filt + dt / tau * (self.w_B - self._w_filt)
            self._a_filt = self._a_filt + dt / tau * (self.a_B - self._a_filt)

    # ------------------------------------------------------------------ #
    # 通用更新
    # ------------------------------------------------------------------ #
    def _update(self, dz: np.ndarray, H: np.ndarray, R: np.ndarray):
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        dx = K @ dz
        self.P = (np.eye(self.dim) - K @ H) @ self.P
        self.P = 0.5 * (self.P + self.P.T)
        self._inject(dx)

    def _inject(self, dx: np.ndarray):
        def get(blk: int) -> np.ndarray:
            i = self._blk(blk)
            return np.zeros(3) if i is None else dx[i : i + 3]

        self.p_B += get(0)
        self.q_B = quat_normalize(quat_mul(self.q_B, quat_exp(get(1))))
        self.v_B += get(2)
        self.w_B += get(3)
        self.a_B += get(4)
        self.al_B += get(5)
        self.b_a += get(6)
        self.b_w += get(7)
        self.p_E += get(8)
        self.q_E = quat_normalize(quat_mul(self.q_E, quat_exp(get(9))))

    # ------------------------------------------------------------------ #
    # IMU 观测 (式 43-44)
    # ------------------------------------------------------------------ #
    def update_imu(self, z_acc: np.ndarray, z_gyro: np.ndarray):
        if not self.cfg.use_imu:
            return
        R_BW = quat_to_rot(self.q_B).T
        h_acc = self.a_B + R_BW @ GRAVITY + self.b_a
        h_gyro = self.w_B + self.b_w
        dz = np.concatenate([np.asarray(z_acc) - h_acc, np.asarray(z_gyro) - h_gyro])

        H = np.zeros((6, self.dim))
        self._put(H, slice(0, 3), 1, skew(R_BW @ GRAVITY))   # d[a] = [R_BW g]x dtheta_B
        self._put(H, slice(0, 3), 4, np.eye(3))              # da_B
        self._put(H, slice(0, 3), 6, np.eye(3))              # db_a
        self._put(H, slice(3, 6), 3, np.eye(3))              # domega_B
        self._put(H, slice(3, 6), 7, np.eye(3))              # db_w

        sa, sw = self.cfg.R_imu
        Rm = np.diag(np.repeat(np.array([sa, sw], dtype=float), 3))
        self._update(dz, H, Rm)

    # ------------------------------------------------------------------ #
    # 末端位姿 + FK 反推基座位姿 (式 45-47)
    # ------------------------------------------------------------------ #
    def update_pose(self, p_E_meas: np.ndarray, R_E_meas: np.ndarray, arm: dict):
        if not (self.cfg.use_ee_pose or self.cfg.use_fk_base_pose):
            return
        R_WB = quat_to_rot(self.q_B)
        R_BE_fk = np.asarray(arm["R_E_B"], dtype=float)      # 基座系下的末端朝向
        p_EB = np.asarray(arm["p_E_B"], dtype=float)

        # FK 反推的基座位姿
        R_B_fk = R_E_meas @ R_BE_fk.T
        p_B_fk = np.asarray(p_E_meas, dtype=float) - R_B_fk @ p_EB

        rows = []
        dz_parts = []
        H_parts = []
        R_parts = []
        rp = self.cfg.R_pose

        if self.cfg.use_ee_pose:
            dz_p = np.asarray(p_E_meas) - self.p_E
            dR = self._R_E().T @ R_E_meas
            dz_th = log_so3(dR)
            rows.append(np.concatenate([dz_p, dz_th]))
            H1 = np.zeros((6, self.dim))
            self._put(H1, slice(0, 3), 8, np.eye(3))
            self._put(H1, slice(3, 6), 9, np.eye(3))
            H_parts.append(H1)
            R_parts.append(np.diag(np.repeat(np.array([rp[0], rp[1]], dtype=float), 3)))

        if self.cfg.use_fk_base_pose:
            dzp = p_B_fk - self.p_B
            dRb = R_WB.T @ R_B_fk
            dz_th = log_so3(dRb)
            rows.append(np.concatenate([dzp, dz_th]))
            H2 = np.zeros((6, self.dim))
            self._put(H2, slice(0, 3), 0, np.eye(3))
            self._put(H2, slice(3, 6), 1, np.eye(3))
            H_parts.append(H2)
            R_parts.append(np.diag(np.repeat(np.array([rp[2], rp[3]], dtype=float), 3)))

        if self.cfg.use_direct_base_pose:
            # 由外部注入的直接基座观测(见 update_base_pose)
            pass

        if not rows:
            return
        dz = np.concatenate(rows)
        H = np.vstack(H_parts)
        Rm = np.zeros((dz.size, dz.size))
        off = 0
        for blk in R_parts:
            k = blk.shape[0]
            Rm[off : off + k, off : off + k] = blk
            off += k
        self._update(dz, H, Rm)

    def update_base_pose(self, p_B_meas: np.ndarray, R_B_meas: np.ndarray):
        """直接基座位姿观测(论文 Table V 中的对照配置)。"""
        if not self.cfg.use_direct_base_pose:
            return
        R_WB = quat_to_rot(self.q_B)
        dz = np.concatenate([np.asarray(p_B_meas) - self.p_B, log_so3(R_WB.T @ R_B_meas)])
        H = np.zeros((6, self.dim))
        self._put(H, slice(0, 3), 0, np.eye(3))
        self._put(H, slice(3, 6), 1, np.eye(3))
        rb = self.cfg.R_base
        Rm = np.diag(np.repeat(np.array(rb, dtype=float), 3))
        self._update(dz, H, Rm)

    # ------------------------------------------------------------------ #
    # 输出
    # ------------------------------------------------------------------ #
    def _R_E(self) -> np.ndarray:
        return quat_to_rot(self.q_E)

    @property
    def R_WB(self) -> np.ndarray:
        return quat_to_rot(self.q_B)

    def state(self) -> dict:
        R = self.R_WB
        w_b = self.w_B if self.aug else self._w_filt
        a_b = self.a_B if self.aug else self._a_filt
        al_b = self.al_B if self.aug else self._al_filt
        return dict(
            R_WB=R,
            p_B=self.p_B.copy(),
            omega_b=self.w_B.copy(),
            v_b=self.v_B.copy(),
            alpha_b=al_b.copy(),
            a_b=a_b.copy(),
            omega_w=R @ w_b,
            v_w=R @ self.v_B,
            alpha_w=R @ al_b,
            a_w=R @ a_b,
            b_a=self.b_a.copy(),
            b_w=self.b_w.copy(),
            p_E=self.p_E.copy(),
            R_E=self._R_E(),
        )

    def as_controller_state(self, dt: float = 1e-3) -> dict:
        """控制器直接使用的世界系基座状态(含姿态)。"""
        st = self.state()
        return dict(
            R_WB=st["R_WB"], p_B=st["p_B"],
            omega_w=st["omega_w"], v_w=st["v_w"],
            alpha_w=st["alpha_w"], a_w=st["a_w"],
            omega_b=st["omega_b"], v_b=st["v_b"],
            alpha_b=st["alpha_b"], a_b=st["a_b"],
        )
