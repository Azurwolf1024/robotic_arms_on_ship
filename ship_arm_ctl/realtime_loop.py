"""
1 kHz 实时控制循环 —— 部署入口。

把 ESKF (基座状态估计) + ShipArmController (TSID+NN+LADRC) + 硬件接口 串成
一个闭环。两种运行模式:

  * --mode sim      用 SimBridge 把 ship_arm 仿真当"假硬件", 端到端验证整个部署
                    链路 (ESKF + 控制器 + 真实对象动力学) 是否闭环正确; 这同时也
                    是我们论文实验的可复现入口之一。
  * --mode real     接真机 (FrankaInterface 占位, 需先接入 libfranka/ROS2)。

典型用法
--------
  # 验证部署链路 (对比 开/关 NN 与 LADRC)
  python -m ship_arm_ctl.realtime_loop --mode sim --duration 15 --ship-scale 1.0
  python -m ship_arm_ctl.realtime_loop --mode sim --duration 15 --no-nn
  python -m ship_arm_ctl.realtime_loop --mode sim --duration 15 --ladrc

  # 真机 (填好 FrankaInterface 后)
  python -m ship_arm_ctl.realtime_loop --mode real --duration 60
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

# 让脚本既能被 `python realtime_loop.py` 也能被 `python -m ship_arm_ctl.realtime_loop` 运行
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ship_arm.core.lie import exp_so3, pose_error
from ship_arm.estimation.eskf import ESKF, ESKFConfig
from ship_arm.platform.ship import ShipMotion
from ship_arm.robot.model import Robot
from ship_arm.robot.panda import PANDA_HOME, make_panda, perturb_spec
from ship_arm.robot.nero import make_nero, NERO_HOME
from ship_arm_ctl.config import build_gains, build_robot
from ship_arm_ctl.controller import ControllerConfig, ShipArmController
from ship_arm_ctl.hardware import FrankaInterface, SimBridge


def _home_for(kind: str) -> np.ndarray:
    return NERO_HOME.copy() if kind == "nero" else PANDA_HOME.copy()


def _make_robot(kind: str, tool_mass=None, payload_mass=0.0, model_error=0.0, seed=7):
    robot = build_robot(kind, tool_mass=tool_mass, payload_mass=payload_mass)
    if model_error > 0.0:
        robot = Robot(perturb_spec(robot.spec, rel_mass=model_error,
                                   rel_inertia=1.6 * model_error, seed=seed))
    return robot


def _arm_info(robot: Robot, q, dq):
    terms = robot.state_terms((np.eye(3), np.zeros(3)), np.asarray(q, float),
                              np.asarray(dq, float), np.zeros((1, 12)),
                              want_M=False, want_J=False)
    R_EB, p_EB = terms["ee_pose"]
    vel = terms["ee_vel"][0]
    return dict(p_E_B=p_EB, R_E_B=R_EB, v_arm_B=vel[3:6], w_arm_B=vel[0:3])


def make_reference(kind: str, p0, R0, t, radius=0.075, period=8.0):
    """生成参考轨迹。fixed_point 保持 p0, circle 绕 p0 做水平圆。"""
    if kind == "fixed_point":
        return dict(R_d=R0.copy(), p_d=p0.copy(),
                    xd_dot=np.zeros(6), xd_ddot=np.zeros(6))
    # circle in x-y plane around p0
    ang = 2 * np.pi * t / period
    p_d = p0 + radius * np.array([np.cos(ang) - 1.0, np.sin(ang), 0.0])
    xd_dot = np.zeros(6)
    xd_ddot = np.zeros(6)
    xd_dot[3:5] = radius * 2 * np.pi / period * np.array([-np.sin(ang), np.cos(ang)])
    return dict(R_d=R0.copy(), p_d=p_d, xd_dot=xd_dot, xd_ddot=xd_ddot)


def run_sim(args):
    kind = getattr(args, "robot", "panda")
    robot_c = _make_robot(kind, tool_mass=None, payload_mass=0.0)
    robot_t = _make_robot(kind, tool_mass=None, payload_mass=0.0,
                          model_error=args.model_error, seed=7)
    ship = ShipMotion(); ship.scale = args.ship_scale
    hw = SimBridge(robot_c, robot_t, ship, dt=args.dt, model_error=args.model_error,
                   friction=args.friction, coulomb=args.coulomb, seed=args.seed)
    hw.q = _home_for(kind).copy()

    eskf = ESKF(ESKFConfig())
    st0 = ship.world_state(0.0)
    R_ee0, p_ee0 = robot_c.ee_pose(st0["R_WB"], st0["p_B"], hw.q)
    eskf.reset(st0["R_WB"], st0["p_B"], p_ee0, R_ee0)

    cfg = ControllerConfig(use_nn=not args.no_nn, use_ladrc=args.ladrc,
                           nn_onnx=args.nn_onnx, robot_kind=kind)
    # 增益必须按机械臂构造: 零空间目标位形 q_ns 要用该臂自己的 home
    ctrl = ShipArmController(robot_c, args.dt, cfg=cfg,
                             gains=build_gains(kind, robot_c))

    n = int(round(args.duration / args.dt))
    pe = np.zeros(n); t0 = time.perf_counter()
    for k in range(n):
        t = k * args.dt
        q, dq = hw.read_state()
        imu = hw.read()
        hw.step_sensors(t)
        arm = _arm_info(robot_c, q, dq)
        eskf.predict(args.dt, arm)
        if imu is not None:
            eskf.update_imu(imu["acc"], imu["gyro"])
        pm = hw.get_pose_meas()
        if pm is not None:
            eskf.update_pose(pm["p"], pm["R"], arm)
        est = eskf.as_controller_state(args.dt)
        ref = make_reference(args.task, p_ee0, R_ee0, t)
        ee_meas = (pm["R"], pm["p"]) if pm is not None else None
        tau, _ = ctrl.step(q, dq, ref, est, ee_pose_meas=ee_meas)
        hw.command_torque(tau)
        # 跟踪误差 (世界系末端 vs 参考)
        st_w = ship.world_state(t)
        R_ee, p_ee = robot_t.ee_pose(st_w["R_WB"], st_w["p_B"], hw.q)
        pe[k] = np.linalg.norm(pose_error((R_ee, p_ee), (ref["R_d"], ref["p_d"]))[3:6])

    wall = time.perf_counter() - t0
    pe_mm = pe * 1e3
    print(f"[sim] done  pos_mean={pe_mm.mean():.3f} mm  pos_max={pe_mm.max():.2f} mm"
          f"  realtime x{args.duration / wall:.1f}  qp_fallback={ctrl.stats['fallback']}")
    return dict(pos_mean=float(pe_mm.mean()), pos_max=float(pe_mm.max()),
                realtime=args.duration / wall)


def main():
    ap = argparse.ArgumentParser(description="Ship-arm deployable control loop")
    ap.add_argument("--mode", choices=["sim", "real"], default="sim")
    ap.add_argument("--robot", choices=["panda", "nero"], default="panda")
    ap.add_argument("--duration", type=float, default=15.0)
    ap.add_argument("--dt", type=float, default=1e-3)
    ap.add_argument("--ship-scale", type=float, default=1.0)
    ap.add_argument("--task", choices=["fixed_point", "circle"], default="fixed_point")
    ap.add_argument("--model-error", type=float, default=0.12)
    ap.add_argument("--friction", type=float, default=0.15)
    ap.add_argument("--coulomb", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--no-nn", action="store_true", help="关闭神经网络残差补偿")
    ap.add_argument("--ladrc", action="store_true", help="开启 LADRC 在线扰动补偿")
    ap.add_argument("--nn-onnx", default=None, help="自定义 NN 残差 ONNX 路径(默认按 --robot 选)")
    args = ap.parse_args()

    # 默认 NN 模型按机械臂分文件 (panda -> residual_net.onnx, nero -> residual_net_nero.onnx)
    if args.nn_onnx is None:
        args.nn_onnx = os.path.join(ROOT, "models",
                                    "residual_net_nero.onnx" if args.robot == "nero"
                                    else "residual_net.onnx")

    if args.mode == "sim":
        run_sim(args)
    else:
        hw = FrankaInterface()  # 占位: 接好驱动后即可运行
        raise NotImplementedError("real mode: 请在 hardware.FrankaInterface 接入驱动")


if __name__ == "__main__":
    main()
