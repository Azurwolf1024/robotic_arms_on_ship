"""
可部署的船载机械臂顶层控制器。

把三块组合起来, 对外只暴露一个干净的 ``step()``:

    ShipArmController.step(q, dq, ref, est, ee_pose_meas) -> (tau, info)

  * TSID-QP   —— 论文核心: 解析基座耦合前馈 + 任务空间逆动力学 + 力矩/加速度约束
  * NN 残差补偿 —— 离线训练、在线前馈, 叠加到力矩上 (见 nn_comp.py)
  * LADRC/ESO —— 在线估计"总扰动"并前馈抵消 (云台稳定常用方法, 见 ladrc.py)

注意: ESKF 基座状态估计 *不* 在这个类里做, 而是由实时循环 (realtime_loop.py)
或仿真引擎 (ship_arm.sim.engine) 提供 ``est`` 字典 —— 这样控制器本身与状态来源
解耦, 既能在仿真里用 ESKF, 也能在实机上用真实 IMU+编码器。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

from .config import (DEFAULT_GAINS, DEFAULT_ROBOT, DEFAULT_TSID_OPTS,
                     ROBOT_GATE, RobotGate)
from .ladrc import LadrcCompensator
from .nn_comp import ResidualCompensator


@dataclass
class ControllerConfig:
    use_nn: bool = True
    use_ladrc: bool = False
    nn_onnx: str = "models/residual_net.onnx"
    nn_residual_frac: float = 0.25  # 残差补偿最多占力矩限的比例 (安全兜底)
    # 门控阈值: 留 None 表示按机械臂种类取 config.ROBOT_GATE —— 那里才是指标唯一的
    # 真相来源 (阈值全部由实测 ‖q̈*‖ / ‖a_w‖ 分布定出)。要手工覆盖就直接给数值。
    robot_kind: str = "panda"
    nn_trust_in: float | None = None
    nn_trust_out: float | None = None
    nn_sev_in: float | None = None
    nn_sev_out: float | None = None
    nn_sev_tau: float = 5.0     # 包络时间常数 (s); 实测 1s 太短, 5s 才贴住海况峰值
    nn_qdd_clip: float = 4000.0  # 纯数值哨兵, 正常不该触发; 不是稳定性旋钮
    ladrc_wo: float = 15.0          # ESO 带宽 (rad/s)
    ladrc_b0: float = 1.0
    accel_clip: float = 18.0


class ShipArmController:
    def __init__(self, robot=DEFAULT_ROBOT, dt: float = 1e-3, cfg: ControllerConfig = None,
                 nn_comp: ResidualCompensator = None, ladrc: LadrcCompensator = None,
                 gains=DEFAULT_GAINS, tsid_opts=DEFAULT_TSID_OPTS):
        self.robot = robot
        self.dt = dt
        self.cfg = cfg or ControllerConfig()
        self.tsid = __import__("ship_arm.control.tsid", fromlist=["TSIDController"]).TSIDController(
            robot, dt, gains, tsid_opts)

        # ---- NN 残差补偿 ----
        self.nn = nn_comp
        if self.nn is None and self.cfg.use_nn:
            if os.path.exists(self.cfg.nn_onnx):
                g = ROBOT_GATE.get(self.cfg.robot_kind, RobotGate())
                pick = lambda cfg_v, gate_v: gate_v if cfg_v is None else cfg_v  # noqa: E731
                self.nn = ResidualCompensator(
                    onnx_path=self.cfg.nn_onnx,
                    tau_max=robot.tau_max,
                    residual_frac=self.cfg.nn_residual_frac,
                    trust_in=pick(self.cfg.nn_trust_in, g.trust_in),
                    trust_out=pick(self.cfg.nn_trust_out, g.trust_out),
                    qdd_clip=self.cfg.nn_qdd_clip,
                    sev_in=pick(self.cfg.nn_sev_in, g.sev_in),
                    sev_out=pick(self.cfg.nn_sev_out, g.sev_out),
                    sev_tau=self.cfg.nn_sev_tau,
                    dt=self.dt)
            else:
                print(f"  [warn] NN onnx 不存在: {self.cfg.nn_onnx}; 仅用 TSID。")

        # ---- LADRC ----
        self.ladrc = ladrc
        if self.ladrc is None and self.cfg.use_ladrc:
            self.ladrc = LadrcCompensator(dt, wo=self.cfg.ladrc_wo, b0=self.cfg.ladrc_b0,
                                          accel_clip=self.cfg.accel_clip)

    # ------------------------------------------------------------------ #
    def reset(self):
        self.tsid.stats = {"feasible": 0, "fallback": 0, "active": 0}
        if self.ladrc is not None:
            self.ladrc.reset()

    def step(self, q: np.ndarray, dq: np.ndarray, ref: dict, est: dict,
             ee_pose_meas=None) -> tuple:
        """一个控制周期。

        参数
        ----
        q, dq        : 关节角/角速度 (来自编码器)
        ref          : 参考轨迹 {R_d, p_d, xd_dot(6), xd_ddot(6)}
        est          : ESKF 基座状态 {R_WB, p_B, omega_w, v_w, alpha_w, a_w}
        ee_pose_meas : 可选末端位姿测量 ((R_ee, p_ee)), LADRC 需要; 不给则不用 LADRC
        返回
        ----
        tau : 关节力矩指令 (已限幅)
        info: TSID 内部量 (qdd, e, status, ...)
        """
        out = self.tsid.compute(q, dq, est, ref, ladrc=self.ladrc, ee_pose_meas=ee_pose_meas)
        tau = out["tau"]
        if self.nn is not None:
            tau = tau + self.nn.compensate(q, dq, out["qdd"], est)
        tau = np.clip(tau, -self.robot.tau_max, self.robot.tau_max)
        return tau, out

    # ------------------------------------------------------------------ #
    @property
    def stats(self):
        return self.tsid.stats
