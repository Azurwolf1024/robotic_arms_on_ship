"""默认机器人 / 增益配置。"""

from __future__ import annotations

from ship_arm.robot.panda import make_panda
from ship_arm.robot.model import Robot
from ship_arm.control.tsid import TSIDGains, TSIDOptions


def build_default_robot(tool_mass: float = 0.73) -> Robot:
    """构造一个 Franka-Emika-Panda 风格的 7-DOF 机械臂模型 (名义参数)。"""
    return Robot(make_panda(tool_mass=tool_mass))


# 单例名义模型(无状态, 可安全共享)
DEFAULT_ROBOT = build_default_robot()
DEFAULT_GAINS = TSIDGains()
DEFAULT_TSID_OPTS = TSIDOptions()
