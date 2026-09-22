"""
Franka Emika Panda 风格的 7-DOF 机械臂模型参数。

运动学采用公开 Panda URDF 的关节固连变换/轴/限位;
质量特性采用与文献 [Gaz et al. 2019] 辨识结果同量级的近似值
(论文作者使用的正是该辨识模型, 本地无私有参数, 故取公开值)。

所有可调项集中在 ``make_panda`` 中:
    * ``tool_offset``   : 末端 TCP 相对法兰的偏移(默认含一次典型工具长度)
    * ``tool_mass``     : 末端执行器质量(论文实验给出名义值 0.73 kg)
    * ``payload_mass``  : 附加未知负载(用于复现论文 IV-B2 节鲁棒性实验)
"""

from __future__ import annotations

from dataclasses import replace
from typing import Tuple

import numpy as np

from ..core.lie import rpy_to_rot
from .model import ArmSpec, Link

Z_AXIS = np.array([0.0, 0.0, 1.0])
DEG2RAD = np.pi / 180.0

# Panda URDF 中的 nominal home 位形 (rad)
PANDA_HOME = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])


def _diag_inertia(ixx: float, iyy: float, izz: float) -> np.ndarray:
    return np.diag([ixx, iyy, izz])


def attach_payload(link: Link, offset: np.ndarray, mass: float,
                   inertia: np.ndarray = None) -> Link:
    """
    把一个刚体几何不变的负载并联合并进连杆(平行轴定理)。
    offset/inertia 均表达在该连杆自身坐标系中, inertia 关于负载自身质心。
    """
    offset = np.asarray(offset, dtype=float).reshape(3)
    if inertia is None:
        inertia = np.zeros((3, 3))
    inertia = np.asarray(inertia, dtype=float).reshape(3, 3)
    if mass <= 0.0:
        return link
    m1, m2 = link.mass, mass
    c1, c2 = link.com, offset
    m = m1 + m2
    c = (m1 * c1 + m2 * c2) / m
    d1 = c1 - c
    d2 = c2 - c
    S = lambda d: np.dot(d, d) * np.eye(3) - np.outer(d, d)
    I = link.inertia + m1 * S(d1) + inertia + m2 * S(d2)
    return replace(link, mass=m, com=c, inertia=I)


def make_panda(tool_offset: np.ndarray = np.array([0.0, 0.0, 0.1034]),
               tool_mass: float = 0.73,
               payload_mass: float = 0.0,
               payload_offset: np.ndarray = np.array([0.0, 0.0, 0.05]),
               include_flange: bool = True) -> ArmSpec:
    """
    构造 7-DOF Panda-like 模型。

    返回的 ArmSpec 中 EE 系 = 法兰 panda_link8 (若 include_flange)
    并按 tool_offset 延伸到 TCP。
    """
    # (offset xyz, rpy, mass, com xyz, (ixx, iyy, izz))
    # 注: URDF 中 joint i 的 <origin xyz rpy> 表达在父连杆系中,
    #     link i 的 <inertial origin> 表达在 joint i 旋转之后的连杆系中。
    L = [
        # joint1
        (dict(xyz=[0.0, 0.0, 0.333], rpy=[0.0, 0.0, 0.0], m=2.9275,
              com=[0.0, -0.0181, -0.0386], I=(0.0239, 0.0225, 0.0064))),
        # joint2
        (dict(xyz=[0.0, 0.0, 0.0], rpy=[-np.pi / 2, 0.0, 0.0], m=2.9355,
              com=[0.0032, -0.0743, 0.0088], I=(0.0419, 0.0251, 0.0617))),
        # joint3
        (dict(xyz=[0.0, -0.316, 0.0], rpy=[np.pi / 2, 0.0, 0.0], m=2.2449,
              com=[0.0407, -0.0048, -0.0290], I=(0.0241, 0.0197, 0.0190))),
        # joint4
        (dict(xyz=[0.0825, 0.0, 0.0], rpy=[np.pi / 2, 0.0, 0.0], m=2.6156,
              com=[-0.0459, 0.0630, -0.0085], I=(0.0345, 0.0289, 0.0413))),
        # joint5
        (dict(xyz=[-0.0825, 0.384, 0.0], rpy=[-np.pi / 2, 0.0, 0.0], m=2.3271,
              com=[-0.0016, 0.0293, -0.0973], I=(0.0516, 0.0479, 0.0164))),
        # joint6
        (dict(xyz=[0.0, 0.0, 0.0], rpy=[np.pi / 2, 0.0, 0.0], m=1.8170,
              com=[0.0597, -0.0410, -0.0102], I=(0.0054, 0.0141, 0.0161))),
        # joint7
        (dict(xyz=[0.088, 0.0, 0.0], rpy=[np.pi / 2, 0.0, 0.0], m=0.6271,
              com=[0.0045, 0.0086, -0.0162], I=(0.0002, 0.0002, 0.0001))),
    ]

    links = []
    for d in L:
        links.append(
            Link(
                offset=np.array(d["xyz"], dtype=float),
                rot0=rpy_to_rot(np.array(d["rpy"], dtype=float)),
                axis=Z_AXIS.copy(),
                mass=float(d["m"]),
                com=np.array(d["com"], dtype=float),
                inertia=_diag_inertia(*d["I"]),
            )
        )

    # ---- 末端执行器 (刚性连在 link7 上): 用法兰的工具坐标描述 ----
    if include_flange:
        flange_off = np.array([0.0, 0.0, 0.107])          # panda_link8
        flange_rot = rpy_to_rot(np.array([0.0, 0.0, -np.pi / 4]))
        # link7 系 -> link8 系的旋转与平移
        ee_offset = flange_rot @ np.asarray(tool_offset, dtype=float) + flange_off
        ee_rot = flange_rot
        _ = flange_off
    else:
        ee_offset = np.asarray(tool_offset, dtype=float)
        ee_rot = np.eye(3)

    if tool_mass > 0.0:
        I_tool = np.diag([1e-3, 1e-3, 1e-3])
        links[-1] = attach_payload(links[-1], ee_offset, tool_mass, I_tool)
    if payload_mass > 0.0:
        pay_ee = np.asarray(ee_offset, dtype=float) + np.asarray(payload_offset, dtype=float)
        links[-1] = attach_payload(links[-1], pay_ee, payload_mass, np.diag([2e-4] * 3))

    q_min = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
    q_max = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
    dq_max = np.array([2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100])
    tau_max = np.array([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0])

    return ArmSpec(
        name="panda-like-7dof",
        links=links,
        ee_offset=ee_offset,
        ee_rot=ee_rot,
        q_min=q_min,
        q_max=q_max,
        dq_max=dq_max,
        tau_max=tau_max,
    )


def perturb_spec(spec: ArmSpec, rel_mass: float = 0.05, rel_inertia: float = 0.08,
                 seed: int = 0) -> ArmSpec:
    """
    按相对误差扰动质量特性, 得到"真实对象"模型(控制器仍使用名义模型)。

    论文在 Gazebo 中做的正是这件事: 控制器用辨识参数, 被控对象是 URDF 真值,
    两者的差构成论文 II-C 节所说的模型不确定度 delta_d。
    """
    rng = np.random.default_rng(seed)
    links = []
    for L in spec.links:
        m = L.mass * (1.0 + rel_mass * rng.uniform(-1.0, 1.0))
        I = L.inertia * (1.0 + rel_inertia * rng.uniform(-1.0, 1.0, size=(3, 3)))
        I = 0.5 * (I + I.T)
        c = L.com * (1.0 + 0.05 * rng.uniform(-1.0, 1.0, size=3))
        links.append(
            Link(offset=L.offset.copy(), rot0=L.rot0.copy(), axis=L.axis.copy(),
                 mass=float(m), com=c, inertia=I)
        )
    return replace(spec, links=links)


def nominal_posture() -> np.ndarray:
    """零空间任务的目标位形 (论文 Table I 中 q_ns)。"""
    return PANDA_HOME.copy()


def ee_frame_from_spec(spec: ArmSpec) -> Tuple[np.ndarray, np.ndarray]:
    return spec.ee_offset, spec.ee_rot
