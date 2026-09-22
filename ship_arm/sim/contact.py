"""
接触模型: 复现论文 IV-F 的 **dynamic peg-in-hole** 实验。

几何(论文 Fig.14):
    插销直径 35 mm (r=17.5 mm)
    孔直径   37 mm (r=18.5 mm)  -> 径向间隙仅 1 mm
    孔深     45 mm
    孔口 10 mm 倒角(被动导向)

力学: 罚函数(penalty)接触 + 库仑摩擦, 作用在末端 TCP 及沿插销轴的若干采样点上,
这样既能产生侧向接触力 F_xy = sqrt(Fx^2+Fy^2), 也能体现"卡滞(jamming)"效应
(上下两个接触点同时受力 -> 产生力矩 -> 更难插入)。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class PegHoleContact:
    center: np.ndarray = field(default_factory=lambda: np.array([0.5, 0.0]))  # 孔轴 (x,y)
    top_z: float = 0.30            # 孔口平面 z
    depth: float = 0.045           # 孔深
    r_hole: float = 0.0185
    r_peg: float = 0.0175
    chamfer: float = 0.010         # 倒角深度(45°)
    peg_len: float = 0.060         # 参与接触的插销长度
    k_lat: float = 2.0e4           # 侧向接触刚度 N/m
    c_lat: float = 60.0            # 侧向阻尼
    k_ax: float = 5.0e4            # 轴向(孔底/台面)刚度
    c_ax: float = 100.0
    mu: float = 0.25
    n_samples: int = 3

    @property
    def bottom_z(self) -> float:
        return self.top_z - self.depth

    @property
    def clearance(self) -> float:
        return self.r_hole - self.r_peg

    def _sample_points(self, R_ee: np.ndarray, p_ee: np.ndarray):
        """
        插销采样点: 末端 TCP 为插销尖端, 插销沿末端系的 -z 方向"向上"延伸。
        返回 list[(p_i, n_i)]。
        """
        axis = -R_ee @ np.array([0.0, 0.0, 1.0])      # 由尖端指向插销根部
        pts = []
        for i in range(self.n_samples):
            s = self.peg_len * i / max(self.n_samples - 1, 1)
            pts.append(p_ee + axis * s)
        return pts

    def wrench(self, R_ee: np.ndarray, p_ee: np.ndarray, v_ee: np.ndarray) -> tuple:
        """
        v_ee: 世界系 [omega; v] (末端点)
        返回 (wrench[6] = [力矩关于末端点; 力], info)
        """
        F = np.zeros(3)
        N = np.zeros(3)
        info = dict(pen=0.0, lateral=0.0, depth=0.0, contact=False, jam=False)

        v_lin = np.asarray(v_ee, dtype=float)[3:6]
        for p in self._sample_points(R_ee, p_ee):
            z = p[2]
            dxy = np.array([p[0] - self.center[0], p[1] - self.center[1]])
            d = float(np.linalg.norm(dxy))
            r_out = self.r_hole + self.chamfer   # 倒角外缘半径
            f = np.zeros(3)

            if z > self.top_z + 1e-9:
                continue                          # 孔口以上: 自由空间

            if z <= self.bottom_z:                # 触底
                pen = self.bottom_z - z
                f[2] += self.k_ax * pen - self.c_ax * min(v_lin[2], 0.0)
                info["contact"] = True
                info["depth"] = self.depth
                # 孔底仍有侧向约束
                if d > self.clearance:
                    u = -dxy / max(d, 1e-9)
                    pen_lat = d - self.clearance
                    f[0:2] += (self.k_lat * pen_lat) * u - self.c_lat * (v_lin[0:2] @ (-u)) * (-u)
                    info["pen"] = max(info["pen"], pen_lat)
            else:
                # 45° 倒角: 越靠近孔口, 有效间隙越大(被动导向)
                ceff = min(self.clearance + max(0.0, self.top_z - z),
                           self.clearance + self.chamfer)
                if d > ceff:
                    u = -dxy / max(d, 1e-9)      # 指向孔轴
                    pen_lat = d - ceff
                    fn = self.k_lat * pen_lat + self.c_lat * max(0.0, -float(v_lin[0:2] @ u))
                    f[0:2] += fn * u
                    # 摩擦: 沿插入方向的阻力
                    f[2] += -self.mu * fn * np.sign(v_lin[2]) if abs(v_lin[2]) > 1e-6 else 0.0
                    info["pen"] = max(info["pen"], pen_lat)
                    info["contact"] = True
                elif d > self.r_hole:
                    pass
                if d > r_out and z > self.bottom_z:
                    # 压在孔口台面上
                    f[2] += self.k_ax * (self.top_z - z) * 0.0
            info["depth"] = max(info["depth"], self.top_z - z if z <= self.top_z else 0.0)
            F += f
            N += np.cross(p - p_ee, f)

        info["lateral"] = float(np.linalg.norm(F[0:2]))
        return np.concatenate([N, F]), info
