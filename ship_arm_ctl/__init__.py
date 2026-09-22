"""ship_arm_ctl — 可部署的船载七自由度机械臂控制方案。

把论文的 TSID-QP 力矩控制 + ESKF 浮动基估计, 与两类"补偿器"组合成
一个可直接部署的控制器:

* 神经网络残差补偿 (NNCompensator)   —— 离线学习、在线前馈, 补偿未建模的
  摩擦 / 惯量参数误差 / 负载不确定度 / 基座耦合残差。
* LADRC / ESO 自抗扰补偿 (LadrcCompensator) —— 云台(稳定平台)控制的主流方法,
  在线估计"总扰动"并前馈抵消, 作为神经网络的在线互补。

对外只暴露一个干净的 ``ShipArmController.step(q, dq, ref, dt, imu, pose)``
接口, 以及硬件抽象层 (hardware.py) 与 1 kHz 实时循环 (realtime_loop.py)。
"""

from .model_nn import ResidualNet
from .nn_comp import ResidualCompensator, build_features, FEATURE_SCALES
from .ladrc import LadrcCompensator
from .controller import ShipArmController, ControllerConfig
from .config import DEFAULT_ROBOT, DEFAULT_GAINS, DEFAULT_TSID_OPTS

__all__ = [
    "ResidualNet", "ResidualCompensator", "build_features", "FEATURE_SCALES",
    "LadrcCompensator", "ShipArmController", "ControllerConfig",
    "DEFAULT_ROBOT", "DEFAULT_GAINS", "DEFAULT_TSID_OPTS",
]
