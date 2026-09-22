"""
闭环仿真引擎: 6-DOF 船体 + 7-DOF 机械臂 + (TSID/基线)控制器 + ESKF。

一个控制周期的流程(默认 1 kHz, 与论文一致):

    1) 船体真值:  ShipMotion -> (R_WB, p_B, omega, v, alpha, a)
    2) 传感器:    IMU 100 Hz / 末端位姿 120 Hz (带噪声与零偏)
    3) 估计器:    ESKF 预测 + 异步观测更新 -> 基座状态估计
    4) 控制器:    TSID-QP -> tau*   (或基线: qdot_c -> 速度伺服 -> tau)
    5) 对象(真值): qdd = M_true^{-1}(tau - H_true - J^T F_ext - tau_d)
    6) 积分:      半隐式欧拉

论文实验中"控制器不知道真值", 因此:
    * 控制器使用 **ESKF 估计** 的基座状态与名义模型
    * 被控对象使用 **真值** 基座运动(可叠加未建模负载/摩擦)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from ..control.baselines import (
    PIEvolverController,
    PIController,
    PredictiveController,
    DisturbanceObserverController,
    VelocityServo,
)
from ..control.tsid import TSIDController, TSIDGains, TSIDOptions
from ..core.lie import exp_so3, log_so3, pose_error, rot_to_quat
from ..estimation.eskf import ESKF, ESKFConfig
from ..platform.sensors import BasePoseSensorSim, IMUSim, PoseSensorSim
from ..robot.model import Robot
from .tasks import Trajectory


@dataclass
class SimOptions:
    dt: float = 1e-3
    duration: float = 10.0
    controller: str = "tsid"            # tsid | pi | woolfrey | xu | evolver
    use_estimator: bool = True
    eskf_cfg: ESKFConfig = None
    payload_mass: float = 0.0           # 被控对象上的未建模负载 (kg)
    torque_disturbance: float = 0.0     # 关节未建模扰动力矩幅值 (Nm)
    friction: float = 0.0               # 粘性摩擦系数 (Nm/(rad/s))
    coulomb: float = 0.0                # 库仑摩擦幅值 (Nm)
    model_error: float = 0.0            # 质量特性相对误差(0 -> 名义模型即真值)
    model_error_seed: int = 7
    seed: int = 1
    log_every: int = 1
    velocity_servo_kp: float = 30.0
    contact: ContactModel = None
    # ---- 力控相关(论文 IV-F 动态插孔) ----
    force_noise: float = 0.0        # 腕部力传感器噪声标准差 (N / Nm)
    admittance: object = None       # 6 维导纳外环对象(见 control/admittance.py)
    # ---- 神经网络残差补偿 / LADRC 在线扰动补偿 (可部署方案) ----
    nn_comp: object = None         # ResidualCompensator, 叠加到力矩上
    ladrc: object = None           # LadrcCompensator, 叠加到任务加速度上
    verbose: bool = False


@dataclass
class SimLog:
    t: list = field(default_factory=list)
    q: list = field(default_factory=list)
    dq: list = field(default_factory=list)
    tau: list = field(default_factory=list)
    p_ee: list = field(default_factory=list)
    R_ee: list = field(default_factory=list)
    p_ref: list = field(default_factory=list)
    p_base_true: list = field(default_factory=list)
    R_base_true: list = field(default_factory=list)
    p_base_est: list = field(default_factory=list)
    R_base_est: list = field(default_factory=list)
    v_base_true: list = field(default_factory=list)
    v_base_est: list = field(default_factory=list)
    a_base_true: list = field(default_factory=list)
    a_base_est: list = field(default_factory=list)
    w_base_true: list = field(default_factory=list)
    w_base_est: list = field(default_factory=list)
    pos_err: list = field(default_factory=list)
    rot_err: list = field(default_factory=list)
    tau_base: list = field(default_factory=list)
    active: list = field(default_factory=list)
    fext: list = field(default_factory=list)
    cpu_ms: list = field(default_factory=list)

    def arrays(self) -> dict:
        return {k: np.asarray(v) for k, v in self.__dict__.items() if len(v) > 0}


class Simulator:
    def __init__(self, robot_ctrl: Robot, robot_true: Robot, ship, task: Trajectory,
                 opts: SimOptions, q0: np.ndarray = None, gains: TSIDGains = None,
                 tsid_opts: TSIDOptions = None):
        self.robot_c = robot_ctrl
        self.robot_t = robot_true
        self.ship = ship
        self.task = task
        self.opts = opts
        self.rng = np.random.default_rng(opts.seed)

        n = robot_ctrl.n
        self.q = np.array(q0, dtype=float).copy() if q0 is not None else np.zeros(n)
        self.dq = np.zeros(n)

        # ---- 控制器 ----
        self.tsid = TSIDController(robot_ctrl, opts.dt, gains or TSIDGains(), tsid_opts or TSIDOptions())
        self.servo = VelocityServo(kp=opts.velocity_servo_kp * np.ones(n))
        if opts.controller == "pi":
            self.baseline = PIController(robot_ctrl, opts.dt)
        elif opts.controller == "woolfrey":
            self.baseline = PredictiveController(robot_ctrl, opts.dt)
        elif opts.controller == "xu":
            self.baseline = DisturbanceObserverController(robot_ctrl, opts.dt)
        elif opts.controller == "evolver":
            self.baseline = PIEvolverController(robot_ctrl, opts.dt)
        else:
            self.baseline = None

        # ---- 估计器 & 传感器 ----
        self.eskf = ESKF(opts.eskf_cfg or ESKFConfig())
        self.imu = IMUSim(seed=opts.seed + 100)
        self.pose_sensor = PoseSensorSim(seed=opts.seed + 200)
        self.base_sensor = BasePoseSensorSim(seed=opts.seed + 300)
        self._imu_buf = None
        self._pose_buf = None
        # 腕部力传感器: 控制器只能拿到**上一周期**的读数(真实系统的 1 步滞后)
        self._f_meas = np.zeros(6)

    # ------------------------------------------------------------------ #
    def arm_info(self, robot: Robot, R_est: np.ndarray, p_est: np.ndarray) -> dict:
        """
        ESKF 需要的"机械臂相对基座的量": p_E^B, v_arm^B, w_arm^B, R_E^B。
        以基座系为参考系(基座姿态不影响这些相对量), 因此用 (I, 0) 做 FK。
        """
        terms = robot.state_terms((np.eye(3), np.zeros(3)), self.q, self.dq,
                                  np.zeros((1, 12)), want_M=False, want_J=False)
        R_EB, p_EB = terms["ee_pose"]
        vel = terms["ee_vel"][0]        # [omega_arm^B; v_arm^B]
        return dict(p_E_B=p_EB, R_E_B=R_EB, v_arm_B=vel[3:6], w_arm_B=vel[0:3])

    # ------------------------------------------------------------------ #
    def run(self) -> SimLog:
        dt = self.opts.dt
        T = self.opts.duration
        nsteps = int(round(T / dt))
        log = SimLog()

        # 初始: 用真值基座姿态初始化估计器与参考
        st0 = self.ship.world_state(0.0)
        R_ee0, p_ee0 = self.robot_c.ee_pose(st0["R_WB"], st0["p_B"], self.q)
        self.eskf.reset(st0["R_WB"], st0["p_B"], p_ee0, R_ee0)

        # 参考轨迹中心: 以"初始末端位姿"为起点, 保证 t=0 时误差为 0
        p_center = p_ee0.copy()
        if getattr(self.task, "kind", "") == "circle":
            p_center = p_ee0 - self.task.radius * self.task.e1
        R_ref = R_ee0.copy()
        self.task.p0 = np.asarray(p_center, dtype=float)
        self.task.R0 = np.asarray(R_ref, dtype=float)

        for k in range(nsteps):
            t = k * dt
            tcpu0 = time.perf_counter()

            # ---------- 1) 真值基座状态 ----------
            st_true_w = self.ship.world_state(t)
            st_true_b = self.ship.body_state(t)
            rows_true = np.array([[
                *st_true_w["omega_w"], *st_true_w["v_w"],
                *st_true_w["alpha_w"], *st_true_w["a_w"],
            ]])

            # ---------- 2) 传感器 ----------
            imu_meas = self.imu.sample(t, st_true_b)
            R_ee_true, p_ee_true = self.robot_t.ee_pose(st_true_w["R_WB"], st_true_w["p_B"], self.q)
            pose_meas = self.pose_sensor.sample(t, R_ee_true, p_ee_true)
            base_meas = self.base_sensor.sample(t, st_true_w["R_WB"], st_true_w["p_B"])

            # ---------- 3) ESKF ----------
            arm = self.arm_info(self.robot_c, None, None)
            self.eskf.predict(dt, arm)
            if imu_meas is not None:
                self.eskf.update_imu(imu_meas["acc"], imu_meas["gyro"])
            if pose_meas is not None:
                self.eskf.update_pose(pose_meas["p"], pose_meas["R"], arm)
            if base_meas is not None:
                self.eskf.update_base_pose(base_meas["p"], base_meas["R"])
            est = self.eskf.as_controller_state(dt) if self.opts.use_estimator else st_true_w

            # ---------- 4) 参考 ----------
            ref = self.task.sample(t)
            if self.opts.admittance is not None:
                # 导纳外环: 由腕部力测量生成笛卡尔位姿偏移(论文 IV-F / 式 51)
                xa = self.opts.admittance.step(self._f_meas, dt)
                ref = dict(ref)
                ref["p_d"] = ref["p_d"] + xa[0:3]
                ref["R_d"] = ref["R_d"] @ exp_so3(xa[3:6])

            # ---------- 5) 控制 ----------
            if self.opts.controller == "tsid":
                # LADRC 需要末端位姿测量(可用外部位姿传感器, 否则用 ESKF+FK 估计)
                if self.opts.ladrc is not None:
                    if pose_meas is not None:
                        ee_meas = (pose_meas["R"], pose_meas["p"])
                    else:
                        R_ee, p_ee = self.robot_c.ee_pose(est["R_WB"], est["p_B"], self.q)
                        ee_meas = (R_ee, p_ee)
                else:
                    ee_meas = None
                out = self.tsid.compute(self.q, self.dq, est, ref,
                                        ladrc=self.opts.ladrc, ee_pose_meas=ee_meas)
                tau = out["tau"]
                if self.opts.nn_comp is not None:
                    tau = tau + self.opts.nn_comp.compensate(self.q, self.dq, out["qdd"], est)
                terms = dict(J=out["J"], ee_pose=out["ee_pose"], ee_vel=out["ee_vel"])
                active = int(np.sum(out["active"]))
                tau_base = out["H"] - self.tsid.robot.bias(
                    (est["R_WB"], est["p_B"]), self.q, self.dq,
                    np.concatenate([np.zeros(6), np.zeros(6)]).reshape(1, 12),
                )
            else:
                base_pose = (est["R_WB"], est["p_B"])
                rows_est = np.array([[*est["omega_w"], *est["v_w"], *est["alpha_w"], *est["a_w"]]])
                terms = self.robot_c.state_terms(base_pose, self.q, self.dq, rows_est,
                                                 want_M=False, want_J=True)
                dq_cmd = self.baseline.command(self.q, self.dq, est, ref, terms)
                if isinstance(self.baseline, PIEvolverController):
                    self.baseline.observe(
                        p_ee_true if pose_meas is None else pose_meas["p"],
                        R_ee_true if pose_meas is None else pose_meas["R"],
                    )
                # 速度伺服含名义模型前馈(真实机器人速度接口亦含重力补偿)
                tau = self.servo.torque(self.robot_c, dq_cmd, self.dq, dt, H=terms["tau"][0])
                active = 0
                tau_base = np.zeros(self.robot_c.n)

            tau = np.clip(tau, -self.robot_t.tau_max, self.robot_t.tau_max)

            # ---------- 6) 接触力 ----------
            fext = np.zeros(6)
            if self.opts.contact is not None:
                R_ee_t, p_ee_t = self.robot_t.ee_pose(st_true_w["R_WB"], st_true_w["p_B"], self.q)
                # 末端世界速度(含基座运动)
                terms_t = self.robot_t.state_terms((st_true_w["R_WB"], st_true_w["p_B"]), self.q,
                                                   self.dq, rows_true, want_M=False, want_J=False)
                v_ee = terms_t["ee_vel"][0]
                fext, _ = self.opts.contact.wrench(R_ee_t, p_ee_t, v_ee)
                if self.opts.force_noise > 0.0:
                    fext = fext + self.rng.normal(0.0, self.opts.force_noise, 6)
                self._f_meas = fext.copy()

            # ---------- 7) 对象动力学 + 积分 ----------
            M_t = self.robot_t.mass_matrix(self.q)
            H_t = self.robot_t.bias((st_true_w["R_WB"], st_true_w["p_B"]), self.q, self.dq,
                                    rows_true[0], fext=fext.reshape(1, 6) if fext.any() else None)
            tau_d = (self.opts.torque_disturbance
                     * np.sin(2 * np.pi * np.array([0.7, 1.1, 1.7, 2.3, 2.9, 3.7, 4.3]) * t)) \
                if self.opts.torque_disturbance > 0 else np.zeros(self.robot_t.n)
            # 摩擦是**阻碍运动**的力矩, 下面统一用 qdd = M^{-1}(tau - H - tau_fric) 扣除,
            # 因此这里取正号。库仑项用宽度 0.02 rad/s 的平滑符号函数, 避免 1 kHz 下抖动。
            tau_fric = self.opts.friction * self.dq \
                + self.opts.coulomb * np.tanh(self.dq / 2e-2)
            qdd = np.linalg.solve(M_t, tau - H_t - tau_d - tau_fric)

            self.dq = np.clip(self.dq + qdd * dt, -self.robot_t.dq_max, self.robot_t.dq_max)
            self.q = self.q + self.dq * dt
            hit = False
            if np.any(self.q < self.robot_t.q_min) or np.any(self.q > self.robot_t.q_max):
                self.q = np.clip(self.q, self.robot_t.q_min, self.robot_t.q_max)
                hit = True

            # ---------- 8) 记录 ----------
            if k % self.opts.log_every == 0:
                st_ref = self.task.sample(t)
                R_ee, p_ee = self.robot_t.ee_pose(st_true_w["R_WB"], st_true_w["p_B"], self.q)
                e = pose_error((R_ee, p_ee), (st_ref["R_d"], st_ref["p_d"]))
                log.t.append(t)
                log.q.append(self.q.copy())
                log.dq.append(self.dq.copy())
                log.tau.append(tau.copy())
                log.p_ee.append(p_ee.copy())
                log.R_ee.append(R_ee.copy())
                log.p_ref.append(st_ref["p_d"].copy())
                log.p_base_true.append(st_true_w["p_B"].copy())
                log.R_base_true.append(st_true_w["R_WB"].copy())
                log.p_base_est.append(est["p_B"].copy())
                log.R_base_est.append(est["R_WB"].copy())
                log.v_base_true.append(st_true_w["v_w"].copy())
                log.v_base_est.append(est["v_w"].copy())
                log.a_base_true.append(st_true_w["a_w"].copy())
                log.a_base_est.append(est["a_w"].copy())
                log.w_base_true.append(st_true_w["omega_w"].copy())
                log.w_base_est.append(est["omega_w"].copy())
                log.pos_err.append(np.linalg.norm(e[3:6]))
                log.rot_err.append(np.linalg.norm(e[0:3]))
                log.tau_base.append(np.array(tau_base, dtype=float).copy())
                log.active.append(active)
                log.fext.append(fext.copy())
                log.cpu_ms.append((time.perf_counter() - tcpu0) * 1e3)

        return log


# --------------------------------------------------------------------------- #
def tracking_metrics(log: SimLog) -> dict:
    """论文 Table II / IV 的指标: 位置误差(mm)与姿态误差(deg)的 mean/std/max。"""
    pe = np.asarray(log.pos_err) * 1e3
    re = np.asarray(log.rot_err) * 180.0 / np.pi
    return dict(
        pos_mean=float(np.mean(pe)), pos_std=float(np.std(pe)), pos_max=float(np.max(pe)),
        rot_mean=float(np.mean(re)), rot_std=float(np.std(re)), rot_max=float(np.max(re)),
    )


def estimation_metrics(log: SimLog) -> dict:
    def rmse(a, b):
        return float(np.sqrt(np.mean(np.sum((np.asarray(a) - np.asarray(b)) ** 2, axis=-1))))

    out = {}
    out["pos_mm"] = rmse(log.p_base_est, log.p_base_true) * 1e3
    Rt = np.asarray(log.R_base_true)
    Re = np.asarray(log.R_base_est)
    ang = np.array([np.linalg.norm(log_so3(Rt[i].T @ Re[i])) for i in range(len(Rt))])
    out["rot_deg"] = float(np.sqrt(np.mean(ang ** 2))) * 180.0 / np.pi
    out["lin_vel"] = rmse(log.v_base_est, log.v_base_true)
    out["ang_vel"] = rmse(log.w_base_est, log.w_base_true)
    out["lin_acc"] = rmse(log.a_base_est, log.a_base_true)
    return out
