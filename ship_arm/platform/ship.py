"""
六自由度船体(或 Stewart 平台)运动模拟。

物理链路: 海浪谱 -> 船体响应
------------------------------------------------------------------
1) 由 **JONSWAP 谱** 生成波幅序列  {A_i, omega_i, eps_i} (等能量离散),
2) 每个自由度的 **RAO(响应幅值算子)** 把波幅映射到船体运动幅值与相位,
3) 叠加得到 6-DOF 运动   eta_d(t) = sum_i A_i |RAO_d(omega_i)| cos(omega_i t + eps_i + psi_d)

这样得到的运动具有随机海浪的频谱特性, 且**解析可导**(一阶/二阶都可直接写出),
因此可以给控制器提供干净的 V_B / Vdot_B 前馈量。

论文 IV-A2 给出的参考量级(仿真): 峰值线速度 0.32 m/s, 峰值角速度 24 deg/s。
本模块在构造时会自动标定幅值, 使 scale=1.0 时的峰值与论文一致。

同时提供:
    * 时间尺度缩放 (论文 Fig.7: 提高基座运动频率)
    * 幅值缩放     (论文 Fig.6 / IV-F: base-motion scale 0~7 / 0~1.0)
    * 世界系 / 体坐标系下的位姿、速度、加速度查询
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..core.lie import euler_rate_matrix, log_so3, rpy_to_rot

DEG = np.pi / 180.0


# --------------------------------------------------------------------------- #
# 海浪谱
# --------------------------------------------------------------------------- #
def jonswap_spectrum(omega: np.ndarray, hs: float, tp: float, gamma: float = 3.3) -> np.ndarray:
    """JONSWAP 谱 S(omega) [m^2 s / rad]。omega>0。"""
    w = np.asarray(omega, dtype=float)
    wp = 2.0 * np.pi / tp
    sigma = np.where(w <= wp, 0.07, 0.09)
    a = np.exp(-0.5 * ((w - wp) / (sigma * wp)) ** 2)
    r = (5.0 / 16.0) * (hs ** 2) * (wp ** 4) / (w ** 5)
    tail = np.exp(-1.25 * (wp / w) ** 4)
    return r * tail * gamma ** a


@dataclass
class WaveField:
    """离散化的波场: 频率、幅值、相位。"""

    omega: np.ndarray
    amp: np.ndarray            # 波幅 (m)
    phase: np.ndarray

    @staticmethod
    def make(hs: float = 0.35, tp: float = 2.0, n_comp: int = 90,
             w_min: float = 0.25, w_max: float = 6.5, seed: int = 20260724) -> "WaveField":
        rng = np.random.default_rng(seed)
        w = np.linspace(w_min, w_max, n_comp)
        dw = w[1] - w[0]
        S = jonswap_spectrum(w, hs, tp)
        amp = np.sqrt(2.0 * S * dw)
        phase = rng.uniform(0.0, 2.0 * np.pi, size=n_comp)
        return WaveField(omega=w, amp=amp, phase=phase)


# --------------------------------------------------------------------------- #
# RAO (response amplitude operator)
# --------------------------------------------------------------------------- #
@dataclass
class RAOSpec:
    """二阶共振型 RAO: |H| 与相位 psi 由 (wn, zeta, gain) 决定。"""
    wn: float
    zeta: float
    gain: float = 1.0

    def magnitude(self, w: np.ndarray) -> np.ndarray:
        r = np.asarray(w) / self.wn
        den = np.sqrt((1.0 - r ** 2) ** 2 + (2.0 * self.zeta * r) ** 2)
        return self.gain / np.maximum(den, 1e-6)

    def phase(self, w: np.ndarray) -> np.ndarray:
        r = np.asarray(w) / self.wn
        return -np.arctan2(2.0 * self.zeta * r, 1.0 - r ** 2)


DEFAULT_RAO = {
    "surge": RAOSpec(wn=0.75, zeta=0.35, gain=0.35),
    "sway": RAOSpec(wn=0.85, zeta=0.35, gain=0.40),
    "heave": RAOSpec(wn=1.70, zeta=0.28, gain=1.00),
    "roll": RAOSpec(wn=1.05, zeta=0.16, gain=1.00),
    "pitch": RAOSpec(wn=1.55, zeta=0.24, gain=0.85),
    "yaw": RAOSpec(wn=1.20, zeta=0.30, gain=0.45),
}


# --------------------------------------------------------------------------- #
# 船体运动
# --------------------------------------------------------------------------- #
@dataclass
class ShipMotion:
    """
    6-DOF 基座运动生成器。

    dof 顺序: [surge(x), sway(y), heave(z), roll, pitch, yaw] (rad)
    """

    wave: WaveField = field(default_factory=WaveField.make)
    rao: dict = field(default_factory=lambda: dict(DEFAULT_RAO))
    scale: float = 1.0          # 幅值缩放 (论文 base-motion scaling factor)
    time_scale: float = 1.0     # 时间(频率)缩放 (论文 Fig.7)
    p0: np.ndarray = field(default_factory=lambda: np.zeros(3))
    _coef: np.ndarray = None    # (6, Nc) 幅值
    _phi: np.ndarray = None     # (6, Nc) 相位
    lin_target_peak: float = 0.32        # m/s
    ang_target_peak: float = 24.0 * DEG  # rad/s

    def __post_init__(self):
        self._build()

    def _build(self):
        dofs = ["surge", "sway", "heave", "roll", "pitch", "yaw"]
        w = self.wave.omega
        coef = np.zeros((6, w.size))
        phi = np.zeros((6, w.size))
        for k, d in enumerate(dofs):
            r = self.rao[d]
            coef[k] = self.wave.amp * r.magnitude(w)
            phi[k] = self.wave.phase + r.phase(w)
        self._coef = coef
        self._phi = phi
        self._calibrate()

    def _calibrate(self):
        """标定幅值, 使 scale=1 时峰值速度接近论文给出量级。"""
        t = np.linspace(0.0, 60.0, 6001)
        v_max = 0.0
        w_max = 0.0
        for tt in t:
            _eta, _pos, _rot, (deta, _ddeta) = self._eta_and_derivs(tt, 1.0, 1.0)
            v_max = max(v_max, float(np.linalg.norm(deta[0:3])))
            w_max = max(w_max, float(np.linalg.norm(deta[3:6])))
        s_lin = self.lin_target_peak / max(v_max, 1e-9)
        s_ang = self.ang_target_peak / max(w_max, 1e-9)
        self._coef[0:3] *= s_lin
        self._coef[3:6] *= s_ang

    def _eta_and_derivs(self, t: float, scale: float, time_scale: float):
        w = self.wave.omega * time_scale
        ph = w * t + self._phi
        c = self._coef * scale
        eta = (c * np.cos(ph)).sum(axis=1)
        deta = (-(c * w) * np.sin(ph)).sum(axis=1)
        ddeta = (-(c * w ** 2) * np.cos(ph)).sum(axis=1)
        return eta, eta[0:3], eta[3:6], (deta, ddeta)

    def eta(self, t: float) -> np.ndarray:
        """6-DOF 广义位移 [m,m,m, rad,rad,rad]。"""
        eta, *_ = self._eta_and_derivs(t, self.scale, self.time_scale)
        return eta

    def eta_dot(self, t: float) -> np.ndarray:
        e, _p, _r, (de, _dde) = self._eta_and_derivs(t, self.scale, self.time_scale)
        return de

    def eta_ddot(self, t: float) -> np.ndarray:
        e, _p, _r, (_de, dde) = self._eta_and_derivs(t, self.scale, self.time_scale)
        return dde

    # ---------------- 位姿 ---------------- #
    def pose(self, t: float):
        """返回 (R_WB, p_B): 船体基座系在世界系中的位姿。"""
        e = self.eta(t)
        p = self.p0 + e[0:3]
        R = rpy_to_rot(e[3:6])
        return R, p

    def rpy(self, t: float) -> np.ndarray:
        return self.eta(t)[3:6]

    # ---------------- 速度/加速度 (世界系, 全解析) ---------------- #
    def omega_world(self, t: float, h: float = 2e-4) -> np.ndarray:
        """世界系角速度: omega_W = R(rpy) * T(rpy) * rpy_dot (解析)。"""
        e = self.eta(t)
        de = self.eta_dot(t)
        return rpy_to_rot(e[3:6]) @ (euler_rate_matrix(e[3:6]) @ de[3:6])

    def alpha_world(self, t: float) -> np.ndarray:
        """
        世界系角加速度: alpha_W = R * (dT/dt * rpy_dot + T * rpy_ddot)。

        (因为 d/dt omega_W = [omega_W]x R omega_B + R omegadot_B = R omegadot_B)
        """
        e = self.eta(t)
        de = self.eta_dot(t)
        dde = self.eta_ddot(t)
        r, p, _y = e[3:6]
        dr, dp, _dy = de[3:6]
        cr, sr = np.cos(r), np.sin(r)
        cp, sp = np.cos(p), np.sin(p)
        T = euler_rate_matrix(e[3:6])
        dTdr = np.array([[0.0, 0.0, 0.0], [0.0, -sr, cr * cp], [0.0, -cr, -sr * cp]])
        dTdp = np.array([[0.0, 0.0, -cp], [0.0, 0.0, -sr * sp], [0.0, 0.0, -cr * sp]])
        dT = dTdr * dr + dTdp * dp
        wdot_b = dT @ de[3:6] + T @ dde[3:6]
        return rpy_to_rot(e[3:6]) @ wdot_b

    def linear_state(self, t: float):
        """世界系 (v, a): 基座原点线速度/加速度。"""
        de = self.eta_dot(t)
        dde = self.eta_ddot(t)
        return de[0:3], dde[0:3]

    def world_state(self, t: float):
        """控制器/仿真直接需要的世界系基座状态: (R, p, omega, v, alpha, a)。"""
        e = self.eta(t)
        de = self.eta_dot(t)
        dde = self.eta_ddot(t)
        R = rpy_to_rot(e[3:6])
        p = self.p0 + e[0:3]
        return dict(R_WB=R, p_B=p,
                    omega_w=self.omega_world(t), v_w=de[0:3],
                    alpha_w=self.alpha_world(t), a_w=dde[0:3])

    # ---------------- 体坐标系 (ESKF/IMU 用) ---------------- #
    def body_state(self, t: float):
        """论文记号下的 V_B^B = [omega_B(3); v_B(3)] 与 Vdot_B^B = [alpha_B; a_B]。"""
        R, p = self.pose(t)
        st = self.world_state(t)
        om_b = R.T @ st["omega_w"]
        v_b = R.T @ st["v_w"]
        al_b = R.T @ st["alpha_w"]
        a_b = R.T @ st["a_w"]
        return dict(R_WB=R, p_B=p, omega_b=om_b, v_b=v_b, alpha_b=al_b, a_b=a_b,
                    omega_w=st["omega_w"], v_w=st["v_w"], alpha_w=st["alpha_w"], a_w=st["a_w"])

    # ---------------- 变体 ---------------- #
    def scaled(self, scale: float = None, time_scale: float = None) -> "ShipMotion":
        import copy

        m = copy.deepcopy(self)
        if scale is not None:
            m.scale = scale
        if time_scale is not None:
            m.time_scale = time_scale
        return m

    def peak_report(self, t_end: float = 60.0, n: int = 6001) -> dict:
        t = np.linspace(0.0, t_end, n)
        v = np.array([self.eta_dot(tt)[0:3] for tt in t])
        w = np.array([self.eta_dot(tt)[3:6] for tt in t])
        a = np.array([self.eta_ddot(tt)[0:3] for tt in t])
        return dict(
            lin_vel_peak=float(np.max(np.linalg.norm(v, axis=1))),
            ang_vel_peak_deg=float(np.max(np.linalg.norm(w, axis=1))) / DEG,
            lin_acc_peak=float(np.max(np.linalg.norm(a, axis=1))),
        )
