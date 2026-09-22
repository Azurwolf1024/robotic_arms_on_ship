"""默认机器人 / 增益配置。

支持两种 7-DOF 机械臂:

* ``panda`` : Franka-Emika-Panda 风格 (论文复现默认, 名义参数)
* ``nero``  : AgileX NERO (厂商 URDF, 高保真, 见 ship_arm.robot.nero)

用 ``build_robot(kind=...)`` 统一构造; 其余增益/TSID 选项与机器人无关。
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from ship_arm.robot.model import Robot
from ship_arm.robot.panda import make_panda
from ship_arm.control.tsid import TSIDGains, TSIDOptions

ROBOT_KINDS = ("panda", "nero")


class RobotGate(NamedTuple):
    """每条臂的 NN 残差补偿门控阈值。

    两个门各管一件事，缺一不可，且都不能用全局常量 —— 两条臂的闭环量级差一个数量级以上。

    trust_in / trust_out (rad/s²) : 作用在 ``‖q̈*‖_inf`` 上的门控。标称依据是实测闭环
        ``‖q̈*‖_inf`` 分布（output/measure_qdd.txt），但**最终取值以闭环实测为准** ——
        见下面各臂的注释：NERO 不得不把门压到远低于其 p99 才不发散。

    sev_in / sev_out (m/s²) : 作用在 ESKF 基座线加速度包络 ``‖a_w‖`` 上的**主门**。
        实测船体 scale 0.5/1.0/1.5/2.0 → ‖a_w‖ 峰值 0.51/1.03/1.54/2.06，
        故 1.05 之内全额补偿（标称海况恒为 1），1.45 之外完全关掉（退化为纯 TSID 基线）。
    """
    trust_in: float = 250.0
    trust_out: float = 1500.0
    sev_in: float = 1.05
    sev_out: float = 1.45


ROBOT_GATE: dict[str, RobotGate] = {
    # Panda: 实测 ‖q̈*‖_inf 定点 14/19/1213、圆周 53/73/341 (scale 0.5/1.0/2.0);
    # 且它在 scale=2.0 加 NN **不发散** (10.95→9.27mm)。故门设在标称 p99 之上,
    # 标称工况恒开、不削收益, 只拦 scale=2.0 的离群尾部。
    "panda": RobotGate(trust_in=80.0, trust_out=400.0),
    # NERO: 实测 p99 达 1330~1767, 高一个数量级, 且 scale=2.0 **会自激发散**
    # (不门控 76.5mm / 门控 250-1500 时 103.6mm, 而基线仅 10.9mm)。
    # 只有压到 (30,150) 才平滑退化为基线 (实测 10.91mm ≈ 基线 10.92mm)。
    "nero": RobotGate(trust_in=30.0, trust_out=150.0),
}


def build_robot(kind: str = "panda", tool_mass: float = None,
                payload_mass: float = 0.0, tau_max=None, dq_max=None) -> Robot:
    """构造一个 7-DOF 机械臂模型 (名义参数)。

    kind="nero" 时 tool_mass 默认 0.5 kg (典型夹爪); "panda" 默认 0.73 kg。
    tau_max / dq_max 仅对 nero 生效 (覆盖 datasheet 估计值)。
    """
    if kind == "panda":
        tm = 0.73 if tool_mass is None else tool_mass
        return Robot(make_panda(tool_mass=tm, payload_mass=payload_mass))
    if kind == "nero":
        from ship_arm.robot.nero import make_nero
        tm = 0.5 if tool_mass is None else tool_mass
        return Robot(make_nero(tool_mass=tm, payload_mass=payload_mass,
                               tau_max=tau_max, dq_max=dq_max))
    raise ValueError(f"unknown robot kind: {kind!r} (choose from {ROBOT_KINDS})")


def default_home(kind: str = "panda") -> np.ndarray:
    """该机械臂的零空间目标位形 (也是仿真起始位形)。"""
    if kind == "nero":
        from ship_arm.robot.nero import NERO_HOME
        return NERO_HOME.copy()
    if kind == "panda":
        from ship_arm.robot.panda import PANDA_HOME
        return PANDA_HOME.copy()
    raise ValueError(f"unknown robot kind: {kind!r}")


def build_gains(kind: str = "panda", robot=None) -> TSIDGains:
    """构造与机械臂匹配的 TSID 增益。

    **关键点 —— 零空间目标位形 ``q_ns`` 必须是"这条臂"的位形**:
    TSID 的零空间任务 ``q̈_ns = Kp_ns (q_ns − q) − Kd_ns q̇`` 会持续把关节拉向
    ``q_ns``。``TSIDGains`` 的默认值是 Panda 的位形
    ``[0, −45°, 0, −135°, 0, +90°, 45°]``, 而 NERO 的关节 4 只有
    ``[−57.9°, 122.6°]``、关节 6 只有 ``[−41.8°, 54.4°]`` —— 第 4、6 关节的目标
    **直接落在限位之外**, 于是零空间任务会不停把臂往限位上顶, 表现为"任务空间
    跟踪误差不大、但关节持续漂移, 最终贴死限位后 QP 失配、系统发散"。

    这里把 ``q_ns`` 设为该臂的 home, 并额外夹到限位内留 0.05 rad 余量。
    """
    g = TSIDGains()
    r = robot if robot is not None else build_robot(kind)
    q_target = default_home(kind)
    lo, hi = r.q_min + 0.05, r.q_max - 0.05
    q_target = np.clip(q_target, np.minimum(lo, hi), np.maximum(lo, hi))
    g.q_ns = np.asarray(q_target, dtype=float).reshape(r.n)
    return g


# 单例名义模型 (无状态, 可安全共享) — 默认 Panda, 保持既有实验基线不变
DEFAULT_ROBOT = build_robot("panda")
DEFAULT_GAINS = build_gains("panda", DEFAULT_ROBOT)
DEFAULT_TSID_OPTS = TSIDOptions()
