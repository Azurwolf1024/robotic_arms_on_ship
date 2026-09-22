"""
刚体串链机械臂在**运动基座(moving base)**上的运动学/动力学。

对应论文式(1)-(4):

    M(q) qdd + C(q,qd) qd + g(q, R_WB) + tau_base = tau + tau_ext + tau_d
    tau_base = M_B(q) Vdot_B + C_B(q, qd, V_B) V_B

实现要点
------------------------------------------------------------------
由于本地没有 Pinocchio, 这里从零实现了一套 **批量 Newton-Euler 递归**:

* 前向递归在 **世界系** 进行, 输入基座的 omega / v / alpha / a(均已转到世界系)。
* 重力采用标准技巧: 令基座线性加速度的有效值 a0_eff = a0 - g, 并把牛顿方程中
  的显式 m*g 项去掉(两者等价, 因为平移量沿运动链恒等传递)。
* 利用"惯量项是加速度的线性函数, 偏差项与质量矩阵均与Jacobian线性"这一性质,
  所有矩阵都通过 **单位列向量批量行** 一次性算出(batch), 避免符号微分,
  同时把 numpy 调用次数降到与链长线性关系。

整个模块只依赖 numpy。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from ..core.lie import cross, rot_axis, skew

GRAVITY = np.array([0.0, 0.0, -9.81])

# Levi-Civita 符号, 用于 (K,3)x(K,3) 的批量叉乘
_EPS = np.zeros((3, 3, 3), dtype=float)
_EPS[0, 1, 2] = _EPS[1, 2, 0] = _EPS[2, 0, 1] = 1.0
_EPS[0, 2, 1] = _EPS[1, 0, 2] = _EPS[2, 1, 0] = -1.0


def _cross_bb(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(K,3) x (K,3) 的逐行叉乘 (比 np.cross 快约 3 倍)。"""
    return np.einsum("abc,ib,ic->ia", _EPS, a, b)


# --------------------------------------------------------------------------- #
# 模型定义
# --------------------------------------------------------------------------- #
@dataclass
class Link:
    """URDF 语义的单个关节/连杆。

    joint i 坐标系相对父系的固定变换为  T = Transl(offset) * Rot(rot0),
    关节变量绕 'axis'(表达在关节系内, 通常为 z)旋转 q_i 得到连杆系。
    """

    offset: np.ndarray          # (3,) 父系坐标下的平移
    rot0: np.ndarray            # (3,3) 父系->关节系 的固定旋转
    axis: np.ndarray            # (3,) 关节轴(关节系坐标)
    mass: float
    com: np.ndarray             # (3,) 质心在连杆系内的位置
    inertia: np.ndarray         # (3,3) 关于质心的惯量(连杆系坐标)

    def __post_init__(self):
        self.offset = np.asarray(self.offset, dtype=float).reshape(3)
        self.rot0 = np.asarray(self.rot0, dtype=float).reshape(3, 3)
        self.axis = np.asarray(self.axis, dtype=float).reshape(3)
        self.axis = self.axis / (np.linalg.norm(self.axis) + 1e-300)
        self.com = np.asarray(self.com, dtype=float).reshape(3)
        self.inertia = np.asarray(self.inertia, dtype=float).reshape(3, 3)


@dataclass
class ArmSpec:
    name: str = "arm"
    links: List[Link] = field(default_factory=list)
    ee_offset: np.ndarray = field(default_factory=lambda: np.zeros(3))
    ee_rot: np.ndarray = field(default_factory=lambda: np.eye(3))
    q_min: np.ndarray = None
    q_max: np.ndarray = None
    dq_max: np.ndarray = None
    tau_max: np.ndarray = None

    @property
    def n(self) -> int:
        return len(self.links)


@dataclass
class Frames:
    """给定基座位姿与关节角后的全部运动学量(与 K 无关)。"""

    R: List[np.ndarray]         # [0..n], R[0] 为基座系
    o: List[np.ndarray]         # [0..n], o[0] 为基座系原点
    z: List[np.ndarray]         # [1..n], 世界系关节轴
    R_ee: np.ndarray
    p_ee: np.ndarray


@dataclass
class BatchResult:
    """一次批量递归的输出。"""

    tau: np.ndarray                  # (K, n) 关节力矩
    base_wrench: Optional[np.ndarray]  # (K, 6) [力矩关于基座原点; 力] 基座需要施加给链的力旋量
    ee_vel: np.ndarray               # (K, 6) [omega; v] 末端世界系速度
    ee_acc: np.ndarray               # (K, 6) [alpha; a] 末端世界系加速度(原点线加速度)


# --------------------------------------------------------------------------- #
# 机器人
# --------------------------------------------------------------------------- #
class Robot:
    """7-DOF(通用 n-DOF) 串链机械臂 + 6-DOF 运动基座。"""

    def __init__(self, spec: ArmSpec):
        self.spec = spec
        self.n = spec.n
        self.links = spec.links
        self.q_min = np.asarray(spec.q_min, dtype=float).reshape(self.n)
        self.q_max = np.asarray(spec.q_max, dtype=float).reshape(self.n)
        self.dq_max = np.asarray(spec.dq_max, dtype=float).reshape(self.n)
        self.tau_max = np.asarray(spec.tau_max, dtype=float).reshape(self.n)
        self._mass_cache = {}

    # ------------------------------------------------------------------ #
    # 运动学
    # ------------------------------------------------------------------ #
    def frames(self, Rb: np.ndarray, ob: np.ndarray, q: np.ndarray) -> Frames:
        n = self.n
        R = [np.asarray(Rb, dtype=float).reshape(3, 3)]
        o = [np.asarray(ob, dtype=float).reshape(3)]
        z: List[np.ndarray] = []
        for i, L in enumerate(self.links):
            Rp = R[-1]
            Rj = Rp @ L.rot0            # 关节系(=连杆系, 未含变量旋转)的世界朝向
            zi = Rj @ L.axis            # 世界系关节轴(绕自身旋转不改变轴向)
            Ri = Rj @ rot_axis(L.axis, float(q[i]))
            oi = o[-1] + Rp @ L.offset
            R.append(Ri)
            o.append(oi)
            z.append(zi)
        Rn = R[-1]
        R_ee = Rn @ self.spec.ee_rot
        p_ee = o[-1] + Rn @ self.spec.ee_offset
        return Frames(R=R, o=o, z=z, R_ee=R_ee, p_ee=p_ee)

    def ee_pose(self, Rb, ob, q) -> Tuple[np.ndarray, np.ndarray]:
        fr = self.frames(Rb, ob, q)
        return fr.R_ee, fr.p_ee

    def fk(self, base_pose, q):
        Rb, ob = base_pose
        return self.ee_pose(Rb, ob, q)

    # ------------------------------------------------------------------ #
    # 批量 Newton-Euler 递归
    # ------------------------------------------------------------------ #
    def batch_pass(
        self,
        fr: Frames,
        dq: np.ndarray,
        w0: np.ndarray,
        v0: np.ndarray,
        al0: np.ndarray,
        a0: np.ndarray,
        qdd: Optional[np.ndarray] = None,
        grav: Optional[np.ndarray] = None,
        fext: Optional[np.ndarray] = None,
        want_base: bool = False,
    ) -> BatchResult:
        """
        参数全部是世界系; dq/qdd 形状 (K,n), 其余 (K,3) 或可广播。

        a0: 基座**真实**线加速度(世界系); 若 grav 给出, 则有效场为 a0 - grav.
        fext: (K,6) 作用在末端点的外部力旋量 [力矩(关于末端点); 力](世界系)。
        """
        n = self.n
        dq = np.atleast_2d(np.asarray(dq, dtype=float))
        K = dq.shape[0]
        if dq.shape != (K, n):
            raise ValueError("dq must be (K, n)")
        zq = np.zeros((K, n))
        if qdd is None:
            qdd = np.zeros_like(dq)
        else:
            qdd = np.atleast_2d(np.asarray(qdd, dtype=float))

        def _as(x, default=0.0):
            x = np.asarray(x, dtype=float)
            if x.size == 3:
                return np.repeat(x.reshape(1, 3), K, axis=0)
            return np.broadcast_to(x.reshape(K, 3), (K, 3))

        w0 = _as(w0)
        v0 = _as(v0)
        al0 = _as(al0)
        a0 = _as(a0)
        if grav is None:
            grav = GRAVITY
        grav = _as(grav)
        fext = None if fext is None else np.atleast_2d(np.asarray(fext, dtype=float))

        # ---- 前向: 运动学量 (索引 0..n, 0 为基座) ----
        # 所有 "固定向量" 的叉乘矩阵提前算好, 使热循环里的叉乘退化为一次 (K,3)@(3,3)
        Sz = [skew(z) for z in fr.z]                       # 关节轴
        Sr = [skew(fr.o[i] - fr.o[i - 1]) for i in range(1, n + 1)]
        rc = [fr.R[i] @ L.com for i, L in enumerate(self.links, start=1)]
        Src = [skew(c) for c in rc]
        Ic = [fr.R[i] @ L.inertia @ fr.R[i].T for i, L in enumerate(self.links, start=1)]
        It = [I.T for I in Ic]
        rn = fr.R[n] @ self.spec.ee_offset
        Srn = skew(rn)

        W = [w0]
        V = [v0]
        Al = [al0]
        A = [a0 - grav]
        Vc = [None] * (n + 1)
        Ac = [None] * (n + 1)
        for i in range(1, n + 1):
            zi = fr.z[i - 1]
            Szi, Sri = Sz[i - 1], Sr[i - 1]
            wp, vp, alp, ap = W[i - 1], V[i - 1], Al[i - 1], A[i - 1]
            dqi = dq[:, i - 1].reshape(K, 1)
            wi = wp + zi * dqi
            ali = alp + zi * qdd[:, i - 1].reshape(K, 1) + (wp @ Szi) * dqi
            ui = wp @ Sri                                   # omega x r
            vi = vp + ui
            ai = ap + alp @ Sri + _cross_bb(wp, ui)
            W.append(wi)
            V.append(vi)
            Al.append(ali)
            A.append(ai)
            ui_c = wi @ Src[i - 1]
            Vc[i] = vi + ui_c
            Ac[i] = ai + ali @ Src[i - 1] + _cross_bb(wi, ui_c)

        # 末端点 (刚连在第 n 连杆上)
        w_e, al_e = W[n], Al[n]
        ue = w_e @ Srn
        v_e = V[n] + ue
        a_e = A[n] + al_e @ Srn + _cross_bb(w_e, ue)

        # ---- 后向: 力平衡 ----
        # 约定: cross(A, b) = - b @ skew(A)   (A 为固定向量, b 为 (K,3))
        S_parent = [skew(fr.o[i] - (fr.o[i] + rc[i - 1])) for i in range(1, n + 1)]
        S_child = [
            skew(fr.o[i + 1] - (fr.o[i] + rc[i - 1])) if i < n else np.zeros((3, 3))
            for i in range(1, n + 1)
        ]
        S_ext = skew(fr.p_ee - (fr.o[n] + rc[n - 1]))

        tau = np.zeros((K, n))
        f_next = np.zeros((K, 3))
        n_next = np.zeros((K, 3))
        f_base = None
        for i in range(n, 0, -1):
            m = self.links[i - 1].mass
            wi, ali = W[i], Al[i]

            Fi = np.zeros((K, 3)) if (fext is None or i != n) else fext[:, 3:6]
            Ni = np.zeros((K, 3)) if (fext is None or i != n) else fext[:, 0:3]

            fi = m * Ac[i] + f_next - Fi
            ni = (
                ali @ It[i - 1]
                + _cross_bb(wi, wi @ It[i - 1])
                + fi @ S_parent[i - 1]
                + n_next
                - f_next @ S_child[i - 1]
                - Ni
                + Fi @ S_ext
            )
            tau[:, i - 1] = ni @ fr.z[i - 1]

            if i == 1 and want_base:
                # 基座施加给第 1 连杆的力旋量, 折算到基座原点
                # cross(A, f) = - f @ skew(A)
                f_base = np.concatenate([ni - fi @ skew(fr.o[1] - fr.o[0]), fi], axis=1)
            f_next, n_next = fi, ni

        ee_vel = np.concatenate([w_e, v_e], axis=1)
        ee_acc = np.concatenate([al_e, a_e], axis=1)
        return BatchResult(tau=tau, base_wrench=f_base, ee_vel=ee_vel, ee_acc=ee_acc)

    # ------------------------------------------------------------------ #
    # 高层: 一次调用拿到控制所需的全部量
    # ------------------------------------------------------------------ #
    def state_terms(
        self,
        base_pose: Tuple[np.ndarray, np.ndarray],
        q: np.ndarray,
        dq: np.ndarray,
        base_state_rows: np.ndarray,
        fext_rows: Optional[np.ndarray] = None,
        want_M: bool = True,
        want_J: bool = True,
    ) -> dict:
        """
        一次批量调用同时得到: 偏差项 H(=C qd + g + tau_base)、任务空间 eta、
        质量矩阵 M、世界系末端 Jacobian J 与基座 Jacobian J_B。

        参数 ``base_state_rows`` 形状 (K, 12):
            [omega_W(3), v_W(3), alpha_W(3), a_W(3)]
        K 行里第 0 行通常是"控制器使用的估计基座状态", 其余行是单位列;
        实际用时建议行 0..R-1 为真实待求的行, 后面再拼单位列。
        """
        n = self.n
        Rb, ob = base_pose
        fr = self.frames(Rb, ob, q)
        rows = np.atleast_2d(np.asarray(base_state_rows, dtype=float))
        R_count = rows.shape[0]

        # 单位 qdd 行同时给出 M 的列(tau)与 J 的列(末端加速度), 因此二者共用同一批行
        cols_U = n if (want_M or want_J) else 0
        cols_Jb = 6 if want_J else 0
        K = R_count + cols_U + cols_Jb

        dq_b = np.zeros((K, n))
        qdd_b = np.zeros((K, n))
        w0 = np.zeros((K, 3))
        v0 = np.zeros((K, 3))
        al0 = np.zeros((K, 3))
        a0 = np.zeros((K, 3))
        grav = np.zeros((K, 3))
        if fext_rows is None:
            fext_b = None
        else:
            fext_b = np.zeros((K, 6))
            fext_b[:R_count] = np.atleast_2d(fext_rows)

        # 真实行
        dq_b[:R_count] = np.atleast_2d(dq)
        w0[:R_count] = rows[:, 0:3]
        v0[:R_count] = rows[:, 3:6]
        al0[:R_count] = rows[:, 6:9]
        a0[:R_count] = rows[:, 9:12]
        grav[:R_count] = GRAVITY

        # 单位列: 质量矩阵 / Jacobian
        idx = R_count
        for k in range(cols_U):
            qdd_b[idx + k, k] = 1.0
        idx += cols_U
        # 单位列: 基座速度 -> J_B 列
        for k in range(cols_Jb):
            if k < 3:
                w0[idx + k, k] = 1.0
            else:
                v0[idx + k, k - 3] = 1.0

        res = self.batch_pass(
            fr, dq_b, w0, v0, al0, a0, qdd=qdd_b, grav=grav, fext=fext_b, want_base=False
        )

        # 重力技巧令前向递推的"加速度"实为 a_eff = a - g, 因此末端加速度里混入了 -g。
        # 论文式(7)的 eta = Jdot qdot + J_B Vdot_B + Jdot_B V_B 是**纯运动学量**, 必须把 g 加回来。
        acc_true = res.ee_acc[:R_count] + np.concatenate([np.zeros(3), GRAVITY])

        out = {
            "frames": fr,
            "tau": res.tau[:R_count],            # 每行对应的 H (C qd + g + tau_base) [+(-J^T Fext) 已含]
            "eta": acc_true,                     # 论文式(7)的 eta
            "ee_vel": res.ee_vel[:R_count],
            "ee_acc": acc_true,
        }
        i0 = R_count
        if cols_U:
            # 第 k 列: qdd_k = 1 -> tau 列为 M[:,k], 末端加速度列为 J[:,k]
            if want_M:
                out["M"] = res.tau[i0 : i0 + n].T     # (n, n)
            if want_J:
                out["J"] = res.ee_acc[i0 : i0 + n].T  # (6, n)
            i0 += n
        if cols_Jb:
            out["J_B"] = res.ee_vel[i0 : i0 + 6].T  # (6, 6)
        out["ee_pose"] = (fr.R_ee, fr.p_ee)
        return out

    # ------------------------------------------------------------------ #
    # 便捷接口
    # ------------------------------------------------------------------ #
    def mass_matrix(self, q: np.ndarray) -> np.ndarray:
        """机械臂部分的惯量阵 M(q) (n,n); 与基座姿态无关。"""
        n = self.n
        fr = self.frames(np.eye(3), np.zeros(3), q)
        dq = np.zeros((n, n))
        qdd = np.eye(n)
        res = self.batch_pass(
            fr, dq, np.zeros((n, 3)), np.zeros((n, 3)), np.zeros((n, 3)),
            np.zeros((n, 3)), qdd=qdd, grav=np.zeros(3),
        )
        return res.tau.T

    def full_mass_matrix(self, q: np.ndarray, base_pose=None) -> np.ndarray:
        """
        (6+n)x(6+n) 关节空间惯量。

        前 6 列/行对应虚拟的基座自由度, 顺序 [angular(3); linear(3)],
        **并且与所有其他接口一样采用世界系表示**, 因此本矩阵依赖于基座姿态
        (平移不影响, 旋转会影响: 关系为 H_W = blkdiag(R,R,I) H_B blkdiag(R^T,R^T,I))。
        """
        n = self.n
        Rb = np.eye(3) if base_pose is None else base_pose[0]
        ob = np.zeros(3) if base_pose is None else base_pose[1]
        fr = self.frames(Rb, ob, q)
        H = np.zeros((6 + n, 6 + n))
        for c in range(6 + n):
            dq1 = np.zeros((1, n))
            qdd1 = np.zeros((1, n))
            w01 = np.zeros((1, 3))
            v01 = np.zeros((1, 3))
            al01 = np.zeros((1, 3))
            a01 = np.zeros((1, 3))
            if c < 3:
                al01[0, c] = 1.0
            elif c < 6:
                a01[0, c - 3] = 1.0
            else:
                qdd1[0, c - 6] = 1.0
            r1 = self.batch_pass(
                fr, dq1, w01, v01, al01, a01, qdd=qdd1, grav=np.zeros(3), want_base=True
            )
            H[6:, c] = r1.tau[0]
            H[:6, c] = r1.base_wrench[0]
        return H

    def jacobians(self, base_pose, q) -> Tuple[np.ndarray, np.ndarray]:
        """返回世界系末端 Jacobian J (6,n) 与基座 Jacobian J_B (6,6)。"""
        terms = self.state_terms(base_pose, q, np.zeros(self.n), np.zeros((1, 12)), want_M=True, want_J=True)
        return terms["J"], terms["J_B"]

    def jacobian(self, base_pose, q) -> np.ndarray:
        terms = self.state_terms(base_pose, q, np.zeros(self.n), np.zeros((1, 12)), want_M=True, want_J=False)
        return terms["J"]

    def bias(self, base_pose, q, dq, base_row, fext=None) -> np.ndarray:
        """H = C qd + g(q,R_WB) + tau_base - J^T F_ext (论文式(10)中的 H)。"""
        rows = np.atleast_2d(base_row)
        fext_rows = None if fext is None else np.atleast_2d(fext)
        t = self.state_terms(base_pose, q, dq, rows, fext_rows=fext_rows, want_M=False, want_J=False)
        return t["tau"][0]

    def forward_dynamics(self, base_pose, q, dq, base_row, tau, fext=None) -> np.ndarray:
        """qdd = M^{-1}(tau - H)。"""
        M = self.mass_matrix(q)
        H = self.bias(base_pose, q, dq, base_row, fext)
        return np.linalg.solve(M, tau - H)

    def joint_torque(self, base_pose, q, dq, qdd, base_row, fext=None) -> np.ndarray:
        """逆动力学: tau = M qdd + H。"""
        M = self.mass_matrix(q)
        H = self.bias(base_pose, q, dq, base_row, fext)
        return M @ qdd + H

    def gramschmidt_damping(self, q: np.ndarray) -> float:
        """可操作度指标 sqrt(det(J J^T)), 用于零空间/奇异性监测。"""
        J = self.jacobian((np.eye(3), np.zeros(3)), q)
        return float(np.sqrt(max(np.linalg.det(J @ J.T), 0.0)))

    # ------------------------------------------------------------------ #
    # 关节限制 -> 加速度限制 (论文式(9))
    # ------------------------------------------------------------------ #
    def accel_bounds(self, q: np.ndarray, dq: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        dq_max = self.dq_max
        qdd_hi_v = (dq_max - dq) / dt
        qdd_lo_v = (-dq_max - dq) / dt
        qdd_hi_p = 2.0 * (self.q_max - q - dq * dt) / (dt * dt)
        qdd_lo_p = 2.0 * (self.q_min - q - dq * dt) / (dt * dt)
        hi = np.minimum(qdd_hi_v, qdd_hi_p)
        lo = np.maximum(qdd_lo_v, qdd_lo_p)
        return lo, hi

    def joint_state_clip(self, q: np.ndarray, dq: np.ndarray):
        qc = np.clip(q, self.q_min + 1e-6, self.q_max - 1e-6)
        dqc = np.clip(dq, -self.dq_max, self.dq_max)
        hit = (q != qc).any() or (dq != dqc).any()
        return qc, dqc, hit
