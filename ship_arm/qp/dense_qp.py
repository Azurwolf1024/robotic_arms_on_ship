"""
稠密凸二次规划求解器(不依赖任何第三方 QP 库)。

求解标准形式::

    min_x   0.5 x^T Q x + c^T x
    s.t.    G x <= h            (m 个不等式约束)

采用 **Mehrotra 预估-校正(primal-dual interior point)** 算法, 并利用
"变量维数很小(机械臂关节数 7)" 这一特点消去所有拉格朗日变量,
每步只解一个 n x n 的线性系统::

    (Q + G^T D^{-1} G) dx = -r_d - G^T D^{-1}(r_p - Λ^{-1} r_c),   D = S Λ^{-1}

对应论文式(13)-(19): 该 QP 是有约束 TSID 的核心;
论文用的是 TSID 库(Eiquadprog), 这里给出等价的自研实现。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class QPResult:
    x: np.ndarray
    status: str                 # "optimal" | "max_iter" | "infeasible"
    iters: int
    obj: float
    active: np.ndarray          # 起作用(数值上接近边界)的约束索引
    lam: np.ndarray
    kkt_err: float


def solve_qp(
    Q: np.ndarray,
    c: np.ndarray,
    G: np.ndarray,
    h: np.ndarray,
    x0: np.ndarray = None,
    max_iter: int = 60,
    tol: float = 1e-9,
    reg: float = 1e-12,
) -> QPResult:
    """
    求解  min 0.5 x'Qx + c'x  s.t.  Gx <= h。
    """
    Q = np.asarray(Q, dtype=float)
    c = np.asarray(c, dtype=float).reshape(-1)
    G = np.asarray(G, dtype=float)
    h = np.asarray(h, dtype=float).reshape(-1)
    n = c.size
    m = h.size
    if G.size == 0:
        G = np.zeros((0, n))
        h = np.zeros(0)

    Qs = 0.5 * (Q + Q.T) + reg * np.eye(n)

    # ---- 初始点 (保证 s>0, lam>0) ----
    x = np.zeros(n) if x0 is None else np.array(x0, dtype=float)
    s = np.maximum(h - G @ x, 1.0)
    lam = np.ones(m)
    mu = max(1.0, float(np.mean(s * lam)))

    def newton_step(sigma: float, r_c_extra: np.ndarray | None = None):
        r_d = Qs @ x + c + G.T @ lam
        r_p = G @ x + s - h
        r_c = s * lam - sigma * mu * np.ones(m)
        if r_c_extra is not None:
            r_c = r_c - r_c_extra
        D = s / np.maximum(lam, 1e-300)             # S Λ^{-1}
        Dinv = lam / np.maximum(s, 1e-300)
        rhs = r_p - r_c / np.maximum(lam, 1e-300)
        A = Qs + G.T @ (Dinv[:, None] * G)
        rhs2 = -r_d - G.T @ (Dinv * rhs)
        try:
            dx = np.linalg.solve(A, rhs2)
        except np.linalg.LinAlgError:
            dx = np.linalg.lstsq(A, rhs2, rcond=None)[0]
        dlam = Dinv * (G @ dx + rhs)
        ds = -r_c / np.maximum(lam, 1e-300) - D * dlam
        return dx, dlam, ds

    def max_step(v, dv):
        idx = dv < 0
        if not np.any(idx):
            return 1.0
        return min(1.0, float(np.min(-v[idx] / dv[idx])))

    it = 0
    kkt = np.inf
    for it in range(1, max_iter + 1):
        r_d = Qs @ x + c + G.T @ lam
        r_p = G @ x + s - h
        r_c = s * lam
        mu = float(np.mean(r_c))
        kkt = max(
            float(np.max(np.abs(r_d))),
            float(np.max(np.abs(r_p))),
            mu,
        )
        if kkt < tol:
            break

        # ---- 预估步 (sigma = 0) ----
        dx_a, dlam_a, ds_a = newton_step(0.0)
        alpha_a = min(max_step(lam, dlam_a), max_step(s, ds_a))
        mu_aff = float(np.mean((s + alpha_a * ds_a) * (lam + alpha_a * dlam_a)))
        sigma = float(np.clip((mu_aff / max(mu, 1e-300)) ** 3, 1e-12, 0.99))

        # ---- 校正步 (含二阶项) ----
        dx, dlam, ds = newton_step(sigma, r_c_extra=-(ds_a * dlam_a))
        tau = 0.995
        alpha = min(
            1.0,
            tau * max_step(lam, dlam),
            tau * max_step(s, ds),
        )
        if not (np.all(np.isfinite(dx)) and np.all(np.isfinite(dlam))):
            break
        x = x + alpha * dx
        lam = lam + alpha * dlam
        s = s + alpha * ds
        if mu > 1e14:                      # 目标发散 -> 判定不可行
            break

    if not np.all(np.isfinite(x)):
        return QPResult(x=np.zeros(n), status="infeasible", iters=it, obj=np.inf,
                        active=np.zeros(m, dtype=bool), lam=np.zeros(m), kkt_err=np.inf)

    status = "optimal" if kkt < max(tol, 1e-7) else "max_iter"
    resid = G @ x - h
    active = resid > -1e-7
    obj = 0.5 * float(x @ Qs @ x) + float(c @ x)
    return QPResult(x=x, status=status, iters=it, obj=obj, active=active, lam=lam, kkt_err=kkt)


def solve_qp_box_relaxed(Q, c, G, h, x_free, max_iter=60):
    """
    带兜底的求解: 若内点法不收敛(例如约束冲突/QP 不可行),
    则退化为"沿自由解方向做最大可行缩放"的投影解, 保证控制器永远有输出。
    """
    res = solve_qp(Q, c, G, h, max_iter=max_iter)
    if res.status == "optimal":
        return res
    # 兜底: 从自由解出发, 用逐次半空间投影(SOP)强制满足 Gx <= h
    viol = G @ x_free - h
    if np.max(viol) <= 0:
        x = x_free
    else:
        # 投影到约束: 逐次半空间投影 (Dykstra 简化版)
        x = x_free.copy()
        for _ in range(30):
            v = G @ x - h
            bad = np.where(v > 0)[0]
            if bad.size == 0:
                break
            for i in bad[:8]:
                gi = G[i]
                nrm = float(gi @ gi)
                if nrm > 1e-18:
                    x = x - (v[i] + 1e-12) * gi / nrm
    res_fb = QPResult(x=x, status="fallback", iters=res.iters,
                      obj=0.5 * float(x @ Q @ x) + float(c @ x),
                      active=(G @ x - h) > -1e-7, lam=np.zeros(len(h)), kkt_err=res.kkt_err)
    return res_fb
