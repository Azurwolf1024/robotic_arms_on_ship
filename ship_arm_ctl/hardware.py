"""
硬件抽象层 —— 让可部署控制器与"状态/执行来源"解耦。

部署到真机时, 只要实现下面几个接口 (read_* / command_*), 控制器与 ESKF 的代码
完全不用改。本文件提供:

  * JointInterface / ImuInterface / BaseStateInterface / ForceTorqueInterface /
    EndEffectorTracker  —— 接口定义
  * SimBridge         —— 用 ship_arm 仿真把"假硬件"接进来, 用于离线验证整个
                        部署链路 (ESKF + 控制器 + 真实对象动力学) 是否正确闭环
  * FrankaInterface   —— 真机占位实现 (libfranka / ROS2), 标注了要填的接口

坐标系约定 (与世界系一致, 与 ship_arm 全部模块一致):
  运动旋量 [ω; v], ω 为角速度, v 为原点线速度。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np


# --------------------------------------------------------------------------- #
# 接口定义
# --------------------------------------------------------------------------- #
class JointInterface(ABC):
    @abstractmethod
    def read_state(self) -> tuple:
        """返回 (q (7,), dq (7,)) —— 来自关节编码器。"""

    @abstractmethod
    def command_torque(self, tau: np.ndarray) -> None:
        """下发关节力矩指令 (Nm)。"""


class ImuInterface(ABC):
    @abstractmethod
    def read(self) -> dict:
        """返回 {'acc':(3,), 'gyro':(3,)} —— 基座 IMU (本体坐标系)。"""


class BaseStateInterface(ABC):
    """可选: 若平台自带位姿/运动学编码器, 可直接读到基座状态; 否则由 ESKF 估计。"""

    @abstractmethod
    def read(self) -> dict:
        """返回 {'R_WB', 'p_B', 'omega_w', 'v_w', 'alpha_w', 'a_w'}。"""


class ForceTorqueInterface(ABC):
    @abstractmethod
    def read(self) -> np.ndarray:
        """返回腕部 6D 力旋量 [τ; f] (N, Nm)。"""


class EndEffectorTracker(ABC):
    """可选: 末端位姿外测量 (用于 LADRC / 导纳); 没有则用 ESKF+FK 估计。"""

    @abstractmethod
    def read(self) -> tuple:
        """返回 (R_ee (3,3), p_ee (3,))。"""


# --------------------------------------------------------------------------- #
# SimBridge —— 用仿真当"假硬件", 验证部署链路
# --------------------------------------------------------------------------- #
class SimBridge(JointInterface, ImuInterface, BaseStateInterface):
    """把 ship_arm 仿真对象封装成硬件接口。

    它在内部维护真实对象动力学 (含摩擦/参数失配/负载), 从而可以端到端验证
    "ESKF 估计 + 可部署控制器 + 真实对象" 是否闭环稳定、精度如何。
    """

    def __init__(self, robot_ctrl, robot_true, ship, dt=1e-3,
                 model_error=0.12, friction=0.15, coulomb=0.3, payload=0.0, seed=1):
        from ship_arm.platform.sensors import IMUSim, PoseSensorSim, BasePoseSensorSim
        self.robot_c = robot_ctrl
        self.robot_t = robot_true
        self.ship = ship
        self.dt = dt
        self.q = np.zeros(robot_ctrl.n)
        self.dq = np.zeros(robot_ctrl.n)
        self.rng = np.random.default_rng(seed)
        self.imu = IMUSim(seed=seed + 100)
        self.pose_sensor = PoseSensorSim(seed=seed + 200)
        self.base_sensor = BasePoseSensorSim(seed=seed + 300)
        self.pose_meas = None
        self.base_meas = None
        self._model_error = model_error
        self._friction = friction
        self._coulomb = coulomb

    # ---- JointInterface ----
    def read_state(self):
        return self.q.copy(), self.dq.copy()

    def command_torque(self, tau):
        t = getattr(self, "_t", 0.0)
        st = self.ship.world_state(t)
        rows = np.array([[*st["omega_w"], *st["v_w"], *st["alpha_w"], *st["a_w"]]])
        M_t = self.robot_t.mass_matrix(self.q)
        H_t = self.robot_t.bias((st["R_WB"], st["p_B"]), self.q, self.dq, rows[0])
        tau_fric = self._friction * self.dq + self._coulomb * np.tanh(self.dq / 2e-2)
        qdd = np.linalg.solve(M_t, np.asarray(tau) - H_t - tau_fric)
        self.dq = np.clip(self.dq + qdd * self.dt, -self.robot_t.dq_max, self.robot_t.dq_max)
        self.q = self.q + self.dq * self.dt
        self.q = np.clip(self.q, self.robot_t.q_min, self.robot_t.q_max)
        self._t = t + self.dt

    # ---- ImuInterface ----
    def read(self):
        t = getattr(self, "_t", 0.0)
        st_b = self.ship.body_state(t)
        return self.imu.sample(t, st_b)

    # ---- BaseStateInterface (真机可省略, 由 ESKF 估计) ----
    def read_base_true(self, t):
        return self.ship.world_state(t)

    def step_sensors(self, t):
        st_b = self.ship.body_state(t)
        st_w = self.ship.world_state(t)
        R_ee, p_ee = self.robot_t.ee_pose(st_w["R_WB"], st_w["p_B"], self.q)
        self.pose_meas = self.pose_sensor.sample(t, R_ee, p_ee)
        self.base_meas = self.base_sensor.sample(t, st_w["R_WB"], st_w["p_B"])

    def get_pose_meas(self):
        return self.pose_meas

    def get_base_meas(self):
        return self.base_meas


# --------------------------------------------------------------------------- #
# 真机占位实现 (libfranka / ROS2)
# --------------------------------------------------------------------------- #
class FrankaInterface(JointInterface, ImuInterface, ForceTorqueInterface):
    """Franka-Emika Panda 真机接口占位。

    实际部署时按下面的 TODO 接到 libfranka (C++) 或 ROS2 (franka_msgs /
    control_msgs)。坐标系: Franka 的 O_F 与本文世界系需做一次手眼/基坐标系标定。
    """

    N = 7
    DQ_MAX = 2.1750
    TAU_MAX = np.array([87, 87, 87, 87, 12, 12, 12], dtype=float)

    def __init__(self, ip: str = "172.16.0.2"):
        self.ip = ip
        # TODO: self._robot = libfranka.Robot(ip); self._model = libfranka.Model()
        # TODO: 标定基座 IMU / 平台编码器 -> 世界系 (R_WB, p_B)
        raise NotImplementedError(
            "FrankaInterface 是占位实现, 请在此接入 libfranka 或 ROS2 驱动")

    def read_state(self):
        # TODO: return (self._robot.current_q, self._robot.current_dq)
        raise NotImplementedError

    def command_torque(self, tau: np.ndarray):
        # TODO: self._robot.collision_behavior_set... ; 用力矩接口下发 tau
        raise NotImplementedError

    def read(self):
        # TODO: 读基座 IMU (acc, gyro), 转换到本体坐标系
        raise NotImplementedError

    def read_ft(self):
        # TODO: 读腕部 F/T 传感器
        raise NotImplementedError
