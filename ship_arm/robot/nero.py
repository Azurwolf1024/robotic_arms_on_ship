"""
AgileX NERO 7-DOF 机械臂模型。

运动学/动力学/限位全部来自厂商官方 URDF
(``ship_arm/robot/urdf/nero_description.urdf``, 仓库内自带, 取自
github.com/agilexrobotics/agx_arm_urdf)。该 URDF 由 SolidWorks 导出, 含真实的
连杆质量、质心与惯量, 因此本模型是 *高保真* 的, 而非估算。

机械臂参数 (datasheet, 2024)
---------------------------
* 自由度 7, 有效负载 3 kg, 本体 4.8 kg, 工作半径 580 mm
* 关节最大角速度: J1-J3 = 180 °/s, J4-J7 = 225 °/s
* 关节运动范围: 见 URDF <limit> (已被解析进 q_min/q_max)
* 力矩上限: URDF 中 effort=100 为 SolidWorks 导出占位值, 并非真实伺服峰值扭矩;
  这里给一组"持 3 kg@0.58 m + 自重"的物理估计值, **部署到真机前请替换为
  AgileX NERO 伺服的官方峰值扭矩** (见 make_nero 的 tau_max 参数)。
"""

from __future__ import annotations

import os

import numpy as np

from .urdf_io import spec_from_urdf

# 文件自带, 不依赖网络
_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_URDF = os.path.join(_HERE, "urdf", "nero_description.urdf")

# datasheet 关节最大角速度 (rad/s)
NERO_DQ_MAX = np.deg2rad(np.array([180.0, 180.0, 180.0, 225.0, 225.0, 225.0, 225.0]))
# 物理估计的关节力矩上限 (Nm) — 占位 effort=100 偏保守/失真, 改用更贴近轻量臂的分档
NERO_TAU_MAX = np.array([80.0, 80.0, 60.0, 40.0, 40.0, 30.0, 30.0])
# 零点位形: 各关节取在限位中部并轻微弯曲, 避开腕部奇异
NERO_HOME = np.array([0.0, -0.5, 0.0, 0.8, 0.0, 0.3, 0.0])


def make_nero(urdf_path: str = None, tool_mass: float = 0.5, payload_mass: float = 0.0,
              payload_offset=None, tau_max=None, dq_max=None):
    """构造 AgileX NERO 的 ArmSpec。

    默认工具质量 0.5 kg (典型夹爪/视觉模组); 力矩/速度上限优先用 datasheet 估计值,
    可被 ``tau_max`` / ``dq_max`` 覆盖 (真机请填入官方伺服参数)。
    """
    spec = spec_from_urdf(urdf_path or DEFAULT_URDF, ee_link="link7",
                          tool_mass=tool_mass, payload_mass=payload_mass,
                          payload_offset=payload_offset)
    spec.dq_max = (np.asarray(dq_max, float) if dq_max is not None
                   else NERO_DQ_MAX.copy()).reshape(spec.n)
    spec.tau_max = (np.asarray(tau_max, float) if tau_max is not None
                    else NERO_TAU_MAX.copy()).reshape(spec.n)
    return spec
