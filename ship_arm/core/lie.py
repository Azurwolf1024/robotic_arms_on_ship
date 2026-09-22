"""
SO(3) / SE(3) 与自动微分无关的基础工具。

坐标与向量约定 (贯穿整个工程, 请务必遵守)
------------------------------------------------------------------
1) 刚体位姿用 tuple/list ``(R, p)`` 表示: R 属于 SO(3), p 为原点位置。
   记 X_AB 表示 "B 系相对 A 系": 世界系记为 W。
2) 运动向量 (spatial motion) 一律采用 **角速度在前** 的排布::

       s = [omega(3); v(3)]

   其中 omega 为刚体角速度(世界坐标), v 为**该坐标系原点处的线速度**(世界坐标)。
   注意: 这里不是"参考于世界原点的螺旋量", 而是"原点线速度"版本,
   因此同一刚体运动在局部坐标与世界坐标之间只差一个块对角旋转::

       s_W = blkdiag(R, R) @ s_local

   这条性质是使用本约定最重要的理由(避免了 Ad 变换中的平移耦合项)。

3) 力/力矩 (wrench) 同样采用矩在前: f = [n(3); F(3)], 其中 n 为关于参考点的力矩。
   这样保证自然配对 f^T s = n^T omega + F^T v。

4) 由于 Python/numpy 开销敏感, 所有雅可比/质量矩阵的获取都通过
   "单位列向量批量前向递归"完成, 不使用符号/自动微分。
"""

from __future__ import annotations

import numpy as np

I3 = np.eye(3)


# --------------------------------------------------------------------------- #
# 基础算子
# --------------------------------------------------------------------------- #
def skew(w: np.ndarray) -> np.ndarray:
    """叉乘矩阵 [w]x, 使得 [w]x @ u == cross(w, u)。"""
    w = np.asarray(w, dtype=float).reshape(3)
    wx, wy, wz = w[0], w[1], w[2]
    M = np.empty((3, 3), dtype=float)
    M[0, 0] = 0.0
    M[0, 1] = -wz
    M[0, 2] = wy
    M[1, 0] = wz
    M[1, 1] = 0.0
    M[1, 2] = -wx
    M[2, 0] = -wy
    M[2, 1] = wx
    M[2, 2] = 0.0
    return M


def cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    逐行叉乘。支持 (K,3)x(K,3) 与 (K,3)x(3,) (以及 (3,)x(3,))。

    说明: numpy 的 np.cross 在 (K,3) 小矩阵上开销极大(约 30us), 而本求解器
    每个控制周期要调用上百次, 因此这里用显式的分量运算替代(约快 5 倍)。
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.ndim == 1 and b.ndim == 1:
        return np.array(
            [
                a[1] * b[2] - a[2] * b[1],
                a[2] * b[0] - a[0] * b[2],
                a[0] * b[1] - a[1] * b[0],
            ]
        )
    # 至少一个是 (K,3)
    if b.ndim == 1:
        ax, ay, az = a[:, 0], a[:, 1], a[:, 2]
        bx, by, bz = b[0], b[1], b[2]
    elif a.ndim == 1:
        ax, ay, az = a[0], a[1], a[2]
        bx, by, bz = b[:, 0], b[:, 1], b[:, 2]
    else:
        ax, ay, az = a[:, 0], a[:, 1], a[:, 2]
        bx, by, bz = b[:, 0], b[:, 1], b[:, 2]
    out = np.empty(np.broadcast(ax, bx).shape + (3,), dtype=float)
    out[..., 0] = ay * bz - az * by
    out[..., 1] = az * bx - ax * bz
    out[..., 2] = ax * by - ay * bx
    return out


def vee(M: np.ndarray) -> np.ndarray:
    """反对称矩阵 -> 向量。"""
    return np.array([M[2, 1], M[0, 2], M[1, 0]], dtype=float)


# --------------------------------------------------------------------------- #
# SO(3)
# --------------------------------------------------------------------------- #
def exp_so3(theta: np.ndarray) -> np.ndarray:
    """罗德里格斯公式: so(3) 向量 -> SO(3) 矩阵 (世界坐标下的旋转)。"""
    th = np.asarray(theta, dtype=float).reshape(3)
    ang = float(np.linalg.norm(th))
    if ang < 1e-12:
        return I3 + skew(th)
    ax = th / ang
    return rot_axis(ax, ang)


def rot_axis(axis: np.ndarray, angle: float) -> np.ndarray:
    """绕单位轴 axis 旋转 angle 弧度 (热路径, 直接写内存避免临时对象)。"""
    a = np.asarray(axis, dtype=float).reshape(3)
    n = float(np.linalg.norm(a))
    if n < 1e-300:
        return I3.copy()
    a = a / n
    x, y, z = a[0], a[1], a[2]
    c = np.cos(angle)
    s = np.sin(angle)
    t = 1.0 - c
    R = np.empty((3, 3), dtype=float)
    R[0, 0] = c + x * x * t
    R[0, 1] = x * y * t - z * s
    R[0, 2] = x * z * t + y * s
    R[1, 0] = y * x * t + z * s
    R[1, 1] = c + y * y * t
    R[1, 2] = y * z * t - x * s
    R[2, 0] = z * x * t - y * s
    R[2, 1] = z * y * t + x * s
    R[2, 2] = c + z * z * t
    return R


def log_so3(R: np.ndarray) -> np.ndarray:
    """SO(3) 矩阵 -> so(3) 旋转向量(指数坐标)。"""
    R = np.asarray(R, dtype=float).reshape(3, 3)
    c = (np.trace(R) - 1.0) * 0.5
    c = float(np.clip(c, -1.0, 1.0))
    ang = np.arccos(c)
    if ang < 1e-8:
        # 小角度: R ~= I + [w]x
        return vee(R - I3)
    if abs(np.pi - ang) < 1e-6:
        # 接近 pi, 用对称部分稳定求解
        S = (R + R.T) * 0.5
        ax = np.sqrt(np.maximum((np.diag(S) + 1.0) * 0.5, 0.0))
        idx = int(np.argmax(ax))
        v = S[:, idx] / (2.0 * max(ax[idx], 1e-12))
        w = np.pi * v / max(np.linalg.norm(v), 1e-12)
        return w
    return (ang / (2.0 * np.sin(ang))) * vee(R - R.T)


def rpy_to_rot(rpy: np.ndarray) -> np.ndarray:
    """内旋 X-Y-Z (roll, pitch, yaw) -> SO(3)。URDF rpy 语义。"""
    r, p, y = np.asarray(rpy, dtype=float).reshape(3)
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def right_jacobian_so3(theta: np.ndarray) -> np.ndarray:
    """SO(3) 右雅可比 J_r(theta), 满足 exp(theta + d) ~= exp(theta) exp(J_r(theta) d)。"""
    th = np.asarray(theta, dtype=float).reshape(3)
    ang = float(np.linalg.norm(th))
    if ang < 1e-9:
        return I3 - 0.5 * skew(th)
    ax = th / ang
    K = skew(ax)
    return (
        I3
        - ((1.0 - np.cos(ang)) / ang) * K
        + ((ang - np.sin(ang)) / ang) * (K @ K)
    )


def right_jacobian_inv_so3(theta: np.ndarray) -> np.ndarray:
    th = np.asarray(theta, dtype=float).reshape(3)
    ang = float(np.linalg.norm(th))
    if ang < 1e-9:
        return I3 + 0.5 * skew(th) + (1.0 / 12.0) * (skew(th) @ skew(th))
    ax = th / ang
    K = skew(ax)
    return (
        I3
        + 0.5 * ang * (1.0 / np.tan(ang / 2.0)) * K
        + (1.0 - (ang / (2.0 * np.sin(ang / 2.0))) * np.cos(ang / 2.0)) * (K @ K)
    )


# --------------------------------------------------------------------------- #
# SE(3)
# --------------------------------------------------------------------------- #
def se3_mul(X: tuple, Y: tuple) -> tuple:
    """(R1,p1) * (R2,p2)"""
    R1, p1 = X
    R2, p2 = Y
    return (R1 @ R2, R1 @ p2 + p1)


def se3_inv(X: tuple) -> tuple:
    R, p = X
    Rt = R.T
    return (Rt, -Rt @ p)


def se3_exp(xi: np.ndarray) -> tuple:
    """se(3) 指数映射, xi = [omega; v] (局部/右乘约定)。"""
    xi = np.asarray(xi, dtype=float).reshape(6)
    w, u = xi[:3], xi[3:]
    R = exp_so3(w)
    p = right_jacobian_so3(w) @ u
    return (R, p)


def se3_log(X: tuple) -> np.ndarray:
    """SE(3) 对数映射 -> xi = [omega; u](局部表示, 其中 u = J_r^{-1} p)。"""
    R, p = X
    w = log_so3(R)
    u = right_jacobian_inv_so3(w) @ p
    return np.concatenate([w, u])


def se3_log_map(X: tuple) -> np.ndarray:
    """log map 使用 'hat' 约定: xi^ = [[w]x, u; 0,0] 时的数值版本(与 se3_log 相同)。"""
    return se3_log(X)


def pose_error(X: tuple, Xd: tuple) -> np.ndarray:
    """
    论文式(5)的世界系等价实现。

    论文: e = log(x^{-1} xd) in se(3)。这里返回其 **世界坐标** 表示::

        e = [ R_E @ log_SO3(R_E^T R_d) ; p_d - p_E ]

    该误差向量与其世界系任务速度 dt 导数相容:
        d/dt (R_E log(R_E^T R_d)) |_{e->0} = omega_d - omega_E
        d/dt (p_d - p_E)                   = v_d - v_E
    故可直接用于 PD 控制律 xddot_c = xddot_d + Kp e + Kd edot。
    返回顺序为 [omega_err(3); v_err(3)]。
    """
    R, p = X
    Rd, pd = Xd
    err_rot = R @ log_so3(R.T @ Rd)
    err_lin = pd - p
    return np.concatenate([err_rot, err_lin])


def pose_error_local(X: tuple, Xd: tuple) -> np.ndarray:
    """严格局部系版本 e = log(X^{-1} Xd), 单位: [rad, m]。"""
    return se3_log(se3_mul(se3_inv(X), Xd))


def rot_to_world_frame(e_local: np.ndarray, R: np.ndarray) -> np.ndarray:
    """把局部系运动/误差向量转到世界系: blkdiag(R, R) @ e_local。"""
    e_local = np.asarray(e_local, dtype=float).reshape(6)
    return np.concatenate([R @ e_local[:3], R @ e_local[3:]])


# --------------------------------------------------------------------------- #
# 四元数
# --------------------------------------------------------------------------- #
def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """四元数乘法, 采用 (w, x, y, z) 排布。"""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def quat_normalize(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float).reshape(4)
    return q / (np.linalg.norm(q) + 1e-300)


def quat_exp(theta: np.ndarray) -> np.ndarray:
    """小角度旋转向量 -> 单位四元数 (论文式(30) 的 delta xi)。"""
    th = np.asarray(theta, dtype=float).reshape(3)
    ang = float(np.linalg.norm(th))
    if ang < 1e-12:
        return np.array([1.0, 0.5 * th[0], 0.5 * th[1], 0.5 * th[2]])
    ax = th / ang
    return np.concatenate([[np.cos(ang / 2.0)], ax * np.sin(ang / 2.0)])


def quat_to_rot(q: np.ndarray) -> np.ndarray:
    q = quat_normalize(q)
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def rot_to_quat(R: np.ndarray) -> np.ndarray:
    """SO(3) -> (w,x,y,z), 采用稳定的分支处理。"""
    R = np.asarray(R, dtype=float).reshape(3, 3)
    tr = np.trace(R)
    if tr > 0.0:
        s = np.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    else:
        i = int(np.argmax(np.diag(R)))
        # 最大对角元素分支
        if i == 0:
            s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif i == 1:
            s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
    return quat_normalize(np.array([w, x, y, z]))


def omega_to_quat_rate(q: np.ndarray, w_local: np.ndarray) -> np.ndarray:
    """局部角速度 -> 四元数导数 (w,x,y,z)。"""
    w, x, y, z = q
    return 0.5 * np.array(
        [
            -x * w_local[0] - y * w_local[1] - z * w_local[2],
            w * w_local[0] + y * w_local[2] - z * w_local[1],
            w * w_local[1] + z * w_local[0] - x * w_local[2],
            w * w_local[2] + x * w_local[1] - y * w_local[0],
        ]
    )


def euler_rate_matrix(rpy: np.ndarray) -> np.ndarray:
    """
    给定 rpy (R = Rz(y) Ry(p) Rx(r)), 返回把 rpy 速率映射到**体角速度**的矩阵 T:

        omega^B = T(rpy) @ [rdot, pdot, ydot]
    """
    r, p, _y = np.asarray(rpy, dtype=float).reshape(3)
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    return np.array([[1.0, 0.0, -sp], [0.0, cr, sr * cp], [0.0, -sr, cr * cp]])


def rot_rate_from_euler_rate(rpy: np.ndarray, drpy: np.ndarray) -> np.ndarray:
    """给定 rpy 与其导数, 求局部(体)角速度 omega^B。"""
    return euler_rate_matrix(rpy) @ np.asarray(drpy, dtype=float).reshape(3)
