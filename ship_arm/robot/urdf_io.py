"""
从 URDF 加载机械臂模型 -> ArmSpec (ship_arm.robot.model)。

只依赖 numpy + 标准库 ``xml``。提取 ``<link>`` 的 inertial 与 ``<joint>`` 的
origin/axis/limit, 构建与 ``panda.py`` 完全同构的 ``ArmSpec``::

    * 固定基座 (world -> base_link 的 fixed joint) 当作"随动基座", 不计入活动连杆;
    * 其后一串 revolute/continuous 关节构成活动链, 每个关节对应一个 Link;
    * Link.offset / rot0 / axis 直接取自关节 origin 与 axis (URDF xyz 在父系,
      axis 在关节系); mass / com / inertia 取自子连杆的 inertial。

这样厂商 URDF (含真实质量/惯量/限位) 可以零手写地变成本库可用的动力学模型,
无需 Pinocchio/ROS。
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET

import numpy as np

from ..core.lie import rpy_to_rot
from .model import ArmSpec, Link


def _v3(text, default=(0.0, 0.0, 0.0)):
    if text is None:
        return np.array(default, dtype=float)
    return np.array([float(x) for x in text.split()], dtype=float)


def _inertia_from_urdf(el):
    if el is None:
        return np.zeros((3, 3))
    g = lambda a: (float(el.find(a).text)
                   if el.find(a) is not None and el.find(a).text is not None else 0.0)
    ixx, iyy, izz = g("ixx"), g("iyy"), g("izz")
    ixy, ixz, iyz = g("ixy"), g("ixz"), g("iyz")
    return np.array([[ixx, ixy, ixz],
                     [ixy, iyy, iyz],
                     [ixz, iyz, izz]], dtype=float)


def _parse_urdf(path):
    root = ET.parse(path).getroot()

    links = {}
    for le in root.findall("link"):
        name = le.get("name")
        iner = le.find("inertial")
        if iner is not None:
            om = iner.find("origin")
            xyz = _v3(om.get("xyz") if om is not None else None)
            rpy = _v3(om.get("rpy") if om is not None else None)
            mass_el = iner.find("mass")
            mass = float(mass_el.get("value")) if mass_el is not None else 0.0
            I = _inertia_from_urdf(iner.find("inertia"))
            # 把"质心处、相对连杆系旋转 rpy"的惯量旋回连杆系
            R = rpy_to_rot(rpy)
            I_link = R @ I @ R.T
            links[name] = dict(mass=mass, com=xyz, inertia=I_link)
        else:
            links[name] = dict(mass=0.0, com=np.zeros(3), inertia=np.zeros((3, 3)))

    joints = []
    for je in root.findall("joint"):
        jtype = je.get("type")
        parent = je.find("parent").get("link")
        child = je.find("child").get("link")
        om = je.find("origin")
        xyz = _v3(om.get("xyz") if om is not None else None)
        rpy = _v3(om.get("rpy") if om is not None else None)
        ax = je.find("axis")
        axis = _v3(ax.get("xyz") if ax is not None else "0 0 1", (0.0, 0.0, 1.0))
        lim = je.find("limit")
        if lim is not None:
            lower = float(lim.get("lower", "-6.2832"))
            upper = float(lim.get("upper", "6.2832"))
            effort = float(lim.get("effort", "100"))
            velocity = float(lim.get("velocity", "5"))
        else:
            lower, upper, effort, velocity = -6.2832, 6.2832, 100.0, 5.0
        joints.append(dict(name=je.get("name"), type=jtype, parent=parent, child=child,
                           xyz=xyz, rpy=rpy, axis=axis,
                           lower=lower, upper=upper, effort=effort, velocity=velocity))
    return links, joints


def spec_from_urdf(path, ee_link=None, tool_mass: float = 0.0, payload_mass: float = 0.0,
                   payload_offset=None, tau_max=None, dq_max=None) -> ArmSpec:
    """把 URDF 转成 ArmSpec。

    参数
    ----
    path           : URDF 文件路径
    ee_link        : 末端连杆名 (默认取活动链最后一节)
    tool_mass      : 末端执行器质量 (刚性并联合进最后一节)
    payload_mass   : 附加未知负载
    payload_offset : 负载相对 EE 的偏移 (连杆系)
    tau_max        : 覆盖关节力矩上限 (否则用 URDF effort)
    dq_max         : 覆盖关节速度上限 (否则用 URDF velocity)
    """
    links, joints = _parse_urdf(path)

    # 固定基座: 找 "world -> base_link" 这类 fixed joint 的子连杆
    base = None
    for j in joints:
        if j["type"] == "fixed" and (j["parent"] in ("world", "", None)
                                     or j["parent"] not in links):
            base = j["child"]
            break
    if base is None:  # 退化: 取没有父关节的连杆作为根
        children = {j["child"] for j in joints}
        for nm in links:
            if nm not in children:
                base = nm
                break

    out_links, q_min, q_max, dq_m, tau_m = [], [], [], [], []
    cur = base
    while True:
        cands = [j for j in joints if j["parent"] == cur
                 and j["type"] in ("revolute", "continuous")]
        if not cands:
            break
        j = cands[0]
        li = links[j["child"]]
        out_links.append(Link(
            offset=j["xyz"], rot0=rpy_to_rot(j["rpy"]), axis=j["axis"],
            mass=float(li["mass"]), com=li["com"], inertia=li["inertia"]))
        q_min.append(j["lower"]); q_max.append(j["upper"])
        dq_m.append(j["velocity"]); tau_m.append(j["effort"])
        cur = j["child"]
        if ee_link is not None and j["child"] == ee_link:
            break

    if not out_links:
        raise ValueError(f"未在 {path} 中找到 revolute 关节链")

    spec = ArmSpec(
        name=os.path.splitext(os.path.basename(path))[0],
        links=out_links,
        ee_offset=np.zeros(3), ee_rot=np.eye(3),
        q_min=np.array(q_min), q_max=np.array(q_max),
        dq_max=np.array(dq_m), tau_max=np.array(tau_m),
    )

    # 末端工具 / 负载 (平行轴定理并联合并进末节)
    if tool_mass > 0.0 or payload_mass > 0.0:
        from .panda import attach_payload
        if tool_mass > 0.0:
            spec.links[-1] = attach_payload(spec.links[-1], np.zeros(3),
                                            tool_mass, np.diag([1e-3] * 3))
        if payload_mass > 0.0:
            po = np.zeros(3) if payload_offset is None else np.asarray(payload_offset, float)
            spec.links[-1] = attach_payload(spec.links[-1], po, payload_mass,
                                            np.diag([2e-4] * 3))

    if dq_max is not None:
        spec.dq_max = np.asarray(dq_max, dtype=float).reshape(spec.n)
    if tau_max is not None:
        spec.tau_max = np.asarray(tau_max, dtype=float).reshape(spec.n)
    return spec
