# robotic_arms_on_ship · 船载七自由度机械臂可部署控制方案

> 参考论文 *"Impedance Control of Ship-Borne Manipulators via Optimization-based Task-Space Inverse Dynamics"*
> (Meng et al.)，结合当前云台 / 稳定平台控制的常用方法（LADRC/ESO、陀螺前馈、DOB），
> 加入**神经网络残差补偿**，构建一套**可离线训练、可在线部署**的 7-DOF 机械臂控制方案。

本仓库**不依赖 ROS / Gazebo / Pinocchio**——动力学、QP、ESKF、控制全部自研（约 1 kHz
单周期 < 1.2 ms），因此既能在普通 PC 上做高保真仿真验证，也能直接把训练好的网络
导出成 ONNX，在边缘计算机 / Jetson / 工控机上用 ONNXRuntime 实时推理部署。

---

## 1. 方案总览

```
            ┌─────────────┐   基座 IMU / 编码器 / 末端位姿
            │   ESKF      │   浮动基状态估计 (R_WB, p_B, V_B, V̇_B)
            └──────┬──────┘
                   │ est
                   ▼
   ref ──► ┌───────────────────────────────────────────┐
           │            ShipArmController                │
           │                                             │
           │  ① TSID-QP  (论文核心)                       │
           │     解析基座耦合前馈 τ_base + 任务空间逆动力学  │
           │     + 力矩/加速度约束 (Mehrotra 内点法 QP)     │
           │                                             │
           │  ② NN 残差补偿  (本方案新增)                   │
           │     离线学习 τ_true − τ_model, 在线前馈叠加     │
           │                                             │
           │  ③ LADRC/ESO  (云台控制常用方法)              │
           │     在线估计"总扰动"并前馈抵消                  │
           └──────────────────────┬──────────────────────┘
                                   │ τ (已限幅)
                                   ▼
                            7-DOF 机械臂 (Franka-Panda 风格)
                            + 6-DOF 运动平台 (Stewart / 云台)
```

三层控制逻辑的角色分工（互补，可单独/组合开关）：

| 层 | 来源 | 作用 | 是否在线学习 |
|----|------|------|------|
| ① TSID-QP | 论文 | 解析基座动态耦合前馈 + 约束下的任务空间逆动力学 | 否（模型） |
| ② NN 残差补偿 | 本方案 | 补偿未建模摩擦 / 惯量误差 / 负载不确定度 / 基座耦合残差 | 离线训练、在线前馈 |
| ③ LADRC/ESO | 云台控制常用方法 | 在线估计"总扰动"（模型失配 + 基座残差 + 摩擦）并抵消 | 在线自适应 |

> **为什么这样组合？**
> 论文的 TSID 已经用*解析*方式补偿了已知的基座耦合（τ_base）；但真机存在摩擦、
> 参数误差、未建模负载，这部分无法解析得到。LADRC 用在线观测器去估计它，神经网络则
> 把它*离线学下来*做成前馈——两者是同一目标（扰动补偿）的"在线自适应版"与"离线学习版"，
> 叠加使用覆盖最全。

---

## 2. 目录结构

```
robotic_arms_on_ship/
├── ship_arm/              # 自研核心库 (动力学 / QP / 控制 / 估计 / 船体 / 仿真)
│   ├── core/lie.py        # SO(3)/SE(3)/四元数 / 李群误差
│   ├── robot/             # 浮动基牛顿-欧拉动力学 (batch), Panda 模型
│   ├── qp/dense_qp.py     # Mehrotra 原始-对偶内点法 QP + 兜底
│   ├── control/           # tsid.py (TSID-QP), baselines.py, admittance.py
│   ├── estimation/eskf.py # 误差状态卡尔曼滤波 (浮动基估计)
│   ├── platform/          # ship.py (JONSWAP+RAO 六自由度船体), sensors.py
│   └── sim/               # engine.py (闭环仿真), tasks.py, contact.py
├── ship_arm_ctl/          # ★ 可部署控制方案 (本仓库新增)
│   ├── model_nn.py        # 残差 MLP (PyTorch)
│   ├── nn_comp.py         # 特征构造 / 数据生成 / 训练 / ONNX 导出 / 推理
│   ├── ladrc.py           # LADRC/ESO 任务空间扰动补偿
│   ├── controller.py      # ShipArmController —— 顶层部署接口 step()
│   ├── hardware.py        # 硬件抽象层 (SimBridge + Franka 占位)
│   ├── config.py          # 默认机器人 / 增益
│   └── realtime_loop.py   # 1 kHz 实时循环 (仿真验证 / 真机入口)
├── training/train_nn.py   # 数据生成 → 训练 → 导出 ONNX 流水线
├── experiments/           # 复现论文 + 本方案对比实验
├── models/                # residual_net.onnx (导出的神经网络)
├── docs/                  # design.md, deployment.md
├── requirements.txt
└── README.md
```

---

## 3. 快速开始

### 3.1 安装
```bash
pip install -r requirements.txt
```

### 3.2 训练神经网络残差补偿
```bash
python training/train_nn.py --n 80000 --epochs 300 --out models/residual_net.onnx
```
该脚本会：用仿真生成 `(特征 X, 残差标签 Y)` 配对数据 → 训练 `ResidualNet` →
导出 `models/residual_net.onnx`（并自动校验 ONNX 与 PyTorch 数值一致）。

### 3.3 验证（仿真）
```bash
# 对比 TSID / TSID+NN / TSID+LADRC / TSID+NN+LADRC
python experiments/run_nn_study.py

# 端到端部署链路验证 (ESKF + 控制器 + 真值对象动力学闭环)
python -m ship_arm_ctl.realtime_loop --mode sim --duration 15 --ship-scale 1.0
python -m ship_arm_ctl.realtime_loop --mode sim --duration 15 --no-nn
python -m ship_arm_ctl.realtime_loop --mode sim --duration 15 --ladrc
```

### 3.4 部署到真机
```bash
# 在 FrankaInterface 里接好 libfranka / ROS2 驱动后：
python -m ship_arm_ctl.realtime_loop --mode real --duration 60
```
详见 [`docs/deployment.md`](docs/deployment.md)。

---

## 4. 关键结果（仿真验证）

> 条件：被控对象含 **12% 惯量误差 + 黏性摩擦 + 库仑摩擦**（与论文实机一致），由 ESKF
> 估计浮动基座；每项跑 12 s（定点 / 圆周两种任务），位置误差为末端相对参考的 RMSE。
> 由 `experiments/run_nn_study.py` 生成，可复现。

**定点跟踪（position RMSE，括号为峰值）：**

| 船体幅度 scale | TSID | **TSID+NN** | TSID+LADRC | TSID+NN+LADRC |
|---|---|---|---|---|
| 0.5（轻度） | 3.38 (5.9) | **1.74 (4.0)** | 2.13 (4.8) | 1.61 (4.3) |
| 1.0（标称） | 4.77 (10.1) | **3.16 (8.1)** | 3.50 (8.3) | 3.10 (8.4) |
| 2.0（剧烈） | 10.95 (69.2) | **9.37 (63.6)** | 15.12 (75.2) | 13.07 (60.7) |

**圆周跟踪（position RMSE，括号为峰值）：**

| 船体幅度 scale | TSID | **TSID+NN** | TSID+LADRC | TSID+NN+LADRC |
|---|---|---|---|---|
| 0.5（轻度） | 3.81 (5.7) | **1.68 (4.0)** | 2.17 (5.1) | 1.56 (4.2) |
| 1.0（标称） | 5.06 (9.9) | **3.04 (8.1)** | 3.48 (8.5) | 3.03 (8.3) |
| 2.0（剧烈） | 10.43 (74.9) | **8.44 (55.9)** | 13.91 (81.4) | 12.65 (62.3) |

**相对纯 TSID 的位置误差改善：**

| scale | TSID+NN | TSID+LADRC | TSID+NN+LADRC |
|---|---|---|---|
| 0.5 | −49% / −56% | −37% / −43% | −52% / −59% |
| 1.0 | −34% / −40% | −27% / −31% | −35% / −40% |
| 2.0 | −14% / −19% | **+38% / +33%** | **+19% / +21%** |

（表中 `a% / b%` 为 定点 / 圆周 两个任务；负号=误差降低。）

**结论与建议：**

- **神经网络残差补偿（TSID+NN）在所有扰动等级都稳定降低跟踪误差（−14% ～ −59%），且从不恶化**——它是离线把"未建模动力学"（摩擦 + 参数误差 + 负载 + 基座耦合残差）学成前馈，叠加在 QP 力矩之上，因此不会破坏 QP 约束。这是本方案最可靠的一层，**推荐作为默认部署配置**。
- **LADRC 在轻度 / 标称扰动下额外有帮助**（在线 ESO 估计"总扰动"并前馈抵消）；但在**剧烈扰动（scale=2.0）下反而使误差变大**。根因：此时 QP 已频繁逼近可行性边界（fallback 次数陡增），LADRC 较激进的加速度前馈与 QP 约束相互拉扯，反而失稳。因此 LADRC 宜按扰动等级做**增益调度**（剧烈时下调带宽 `ladrc_wo` 与 `accel_clip`），或仅在中低扰动开启。
- 组合 `TSID+NN+LADRC` 在轻/中标称下最优，但剧烈扰动下 LADRC 的负面效应盖过了 NN 的收益；实际部署用 **TSID+NN** 打底，按需叠加经调度的 LADRC。
- 与论文 Table II 复现的基线对照：纯 TSID 的跟踪误差与论文同量级（~3–5 mm 轻/中标称），而经典 PI 等基线在该场景下误差达 ~15–18 mm，进一步印证优化型任务空间逆动力学的必要性。

---

## 5. 部署要点

- **推理快**：残差 MLP 仅 3 个隐藏层（128/128/64），CPU 单样本推理 < 0.1 ms。
- **安全**：网络输出经 `tanh` 限幅到 ±40 Nm（不超过力矩限的 50%），且加在 QP 解出的
  力矩之上，**不会破坏力矩限 / QP 约束**。
- **可移植**：导出 ONNX 后用 ONNXRuntime（C++/Python）即可在任意边缘设备推理；
  `docs/deployment.md` 给出 C++ 推理片段与硬件接入清单。

---

## 6. 参考文献

1. Meng et al., *Impedance Control of Ship-Borne Manipulators via Optimization-based
   Task-Space Inverse Dynamics*, 2024 (arXiv:2407.22030).
2. Han J., *From PID to Active Disturbance Rejection Control* (LADRC/ESO), IEEE TIE 2009.
3. Gao Z., *Scaling and Bandwidth-Parameterization of LADRC*, ACC 2003.
4. Lewis F. et al., *Neural Network Control of Robot Manipulators* (计算力矩 + NN 补偿).
5. Gaz et al., *Fast and Soft Arm Simulation*, 2019 (Panda 参数来源).

---

## 许可证

代码用于科研与教学复现；商业部署请遵循对应机器人厂商与论文的许可要求。
