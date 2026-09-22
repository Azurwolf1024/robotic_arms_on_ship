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
                            7-DOF 机械臂 (Franka-Panda / AgileX NERO)
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

**支持的机械臂**：控制算法与机械臂型号解耦，目前内置两种 7-DOF 机型——
Franka-Panda（论文复现默认）与 **AgileX NERO**（厂商 URDF 高保真模型，见 §3.5）。
接入新机型只需提供一个 URDF（或等价的 `ArmSpec`），其余代码零改动。

---

## 2. 目录结构

```
robotic_arms_on_ship/
├── ship_arm/              # 自研核心库 (动力学 / QP / 控制 / 估计 / 船体 / 仿真)
│   ├── core/lie.py        # SO(3)/SE(3)/四元数 / 李群误差
│   ├── robot/             # 浮动基牛顿-欧拉动力学 (batch), Panda/NERO 模型, URDF 解析
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
# Panda (默认)
python training/train_nn.py --n 80000 --epochs 300 --out models/residual_net.onnx
# AgileX NERO —— 指定 --robot nero, 默认输出 models/residual_net_nero.onnx
python training/train_nn.py --robot nero --n 80000 --epochs 300 \
    --data-cache models/train_data_nero.npz --out models/residual_net_nero.onnx
```
该脚本会：用仿真生成 `(特征 X, 残差标签 Y)` 配对数据 → 训练 `ResidualNet` →
导出 `models/residual_net.onnx`（并自动校验 ONNX 与 PyTorch 数值一致）。

### 3.3 验证（仿真）
```bash
# 对比 TSID / TSID+NN / TSID+LADRC / TSID+NN+LADRC (Panda)
python experiments/run_nn_study.py

# 端到端部署链路验证 (ESKF + 控制器 + 真值对象动力学闭环)
python -m ship_arm_ctl.realtime_loop --mode sim --duration 15 --ship-scale 1.0
python -m ship_arm_ctl.realtime_loop --mode sim --duration 15 --no-nn
python -m ship_arm_ctl.realtime_loop --mode sim --duration 15 --ladrc

# 同样的命令，换 NERO 只需加 --robot nero
python -m ship_arm_ctl.realtime_loop --mode sim --robot nero --duration 15 --ship-scale 1.0
python -m ship_arm_ctl.realtime_loop --mode sim --robot nero --duration 15 --no-nn
python -m ship_arm_ctl.realtime_loop --mode sim --robot nero --duration 15 --ladrc
```

### 3.4 部署到真机
```bash
# 在 FrankaInterface 里接好 libfranka / ROS2 驱动后：
python -m ship_arm_ctl.realtime_loop --mode real --duration 60
# NERO 真机同样支持：在 hardware.FrankaInterface 接入对应驱动即可
python -m ship_arm_ctl.realtime_loop --mode real --robot nero --duration 60
```
详见 [`docs/deployment.md`](docs/deployment.md)。

### 3.5 AgileX NERO 适配说明

本仓库的控制算法与机械臂型号**解耦**：动力学、QP、ESKF、NN 残差、LADRC 都是型号无关的。
接入新机械臂只需提供它的**模型**——我们直接从厂商 **URDF** 解析，无需手写参数、无需 Pinocchio：

- 模型来源：`ship_arm/robot/urdf/nero_description.urdf`（取自官方
  [agx_arm_urdf](https://github.com/agilexrobotics/agx_arm_urdf)，SolidWorks 导出，含真实
  质量 / 质心 / 惯量，已随仓库自带）。
- 解析器：`ship_arm/robot/urdf_io.py`——纯 numpy + 标准库，把 URDF 的 `<link>` inertial 与
  `<joint>` origin/axis/limit 转成与 Panda 同构的 `ArmSpec`。
- 模型封装：`ship_arm/robot/nero.py`（`make_nero()`、`NERO_HOME`、`NERO_DQ_MAX`、`NERO_TAU_MAX`）。

`experiments/`、`training/`、`realtime_loop` 均已支持 `--robot {panda,nero}`，论文复现结果（§4）
仍以 Panda 为默认基线。NERO 的仿真闭环已验证：纯 TSID 在 ship-scale=1.0 定点任务下位置误差
约 **4.0 mm**（与 Panda 同量级），NN 残差进一步压低（见 §4 同款对比流程）。

> ⚠️ **NERO 力矩上限（部署前必改）**：URDF 里 `<limit effort>` 是 SolidWorks 导出的占位值（100），
> 并非 AgileX NERO 伺服的真实峰值扭矩。`make_nero()` 默认给了一组基于"持 3 kg@0.58 m + 自重"的
> 物理估计 `[80, 80, 60, 40, 40, 30, 30] Nm`，**真机部署前请替换为你机型的官方伺服峰值扭矩**
> （`make_nero(tau_max=...)` 或 `build_robot("nero", tau_max=...)`）。关节**角速度**上限已采用
> datasheet 权威值（J1–J3 = 180 °/s，J4–J7 = 225 °/s）。

---

## 4. 关键结果（仿真验证）

> 条件：被控对象含 **12% 惯量误差 + 黏性摩擦 + 库仑摩擦**（与论文实机一致），由 ESKF
> 估计浮动基座；每项跑 12 s（定点 / 圆周两种任务），位置误差为末端相对参考的 RMSE。
> 由 `experiments/run_nn_study.py` 生成，可复现。

（表中 `a% / b%` 为 定点 / 圆周 两个任务；负号=误差降低。数据由 `experiments/run_nn_study.py`
用 v3 流水线——训练量程 1500 + `‖q̈*‖` 幅值加权 + 每条臂各自的可信域——生成，可复现。）

### 4.1 Franka-Panda（论文复现基线，较重臂）

**定点跟踪（position RMSE mm，括号为峰值 mm）：**

| 船体幅度 scale | TSID | **TSID+NN** | TSID+LADRC | TSID+NN+LADRC |
|---|---|---|---|---|
| 0.5（轻度） | 3.38 (5.9) | **1.71 (4.3)** | 2.13 (4.8) | 1.56 (4.2) |
| 1.0（标称） | 4.77 (10.1) | **3.11 (8.4)** | 3.50 (8.3) | 3.07 (8.3) |
| 2.0（剧烈） | 10.95 (69.2) | **9.95 (73.1)** | 15.12 (75.2) | 15.75 (82.3) |

**圆周跟踪（position RMSE mm，括号为峰值 mm）：**

| 船体幅度 scale | TSID | **TSID+NN** | TSID+LADRC | TSID+NN+LADRC |
|---|---|---|---|---|
| 0.5（轻度） | 3.81 (5.7) | **1.57 (4.2)** | 2.17 (5.1) | 1.51 (4.2) |
| 1.0（标称） | 5.06 (9.9) | **3.08 (8.3)** | 3.48 (8.5) | 3.00 (8.2) |
| 2.0（剧烈） | 10.43 (74.9) | **9.28 (71.8)** | 13.91 (81.4) | 12.87 (78.2) |

**Panda 相对纯 TSID 的位置误差改善：**

| scale | TSID+NN | TSID+LADRC | TSID+NN+LADRC |
|---|---|---|---|
| 0.5 | −49% / −59% | −37% / −43% | −54% / −60% |
| 1.0 | −35% / −39% | −27% / −31% | −36% / −41% |
| 2.0 | −9% / −11% | **+38% / +33%** | **+44% / +23%** |

> Panda 闭环 `‖q̈*‖` 的 p99 仅 19~341 rad/s²（见 §5.1），远在训练域（1500）内，故可信域几乎
> 不触发，NN 在所有工况都全额生效。

### 4.2 AgileX NERO（轻量臂，7-DOF）

**定点跟踪（position RMSE mm，括号为峰值 mm）：**

| 船体幅度 scale | TSID | **TSID+NN** | TSID+LADRC | TSID+NN+LADRC |
|---|---|---|---|---|
| 0.5（轻度） | 3.43 (8.1) | **1.91 (5.2)** | 3.40 (11.4) | 1.77 (5.0) |
| 1.0（标称） | 5.06 (33.0) | **4.93 (49.5)** | 6.21 (39.6) | 5.27 (39.2) |
| 2.0（剧烈） | 10.93 (73.8) | **11.02 (79.4)** | 34.60 (268.1) | 29.64 (231.1) |

**圆周跟踪（position RMSE mm，括号为峰值 mm）：**

| 船体幅度 scale | TSID | **TSID+NN** | TSID+LADRC | TSID+NN+LADRC |
|---|---|---|---|---|
| 0.5（轻度） | 6.45 (66.0) | **5.21 (75.7)** | 7.33 (52.0) | 6.99 (57.3) |
| 1.0（标称） | 9.13 (67.3) | **8.85 (101.7)** | 13.03 (57.6) | 13.34 (83.5) |
| 2.0（剧烈） | 14.62 (117.4) | **14.82 (119.1)** | 19.38 (101.7) | 18.67 (107.9) |

**NERO 相对纯 TSID 的位置误差改善：**

| scale | TSID+NN | TSID+LADRC | TSID+NN+LADRC |
|---|---|---|---|
| 0.5 | −44% / −19% | −1% / +14% | −48% / +8% |
| 1.0 | −3% / −3% | **+23% / +43%** | +4% / +46% |
| 2.0 | +1% / +1% | **+217% / +33%** | **+171% / +28%** |

> NERO 闭环 `‖q̈*‖` 的 p99 高达 **1330~1767 rad/s²**（§5.1），是 Panda 的 ~7 倍、远超训练域。
> 因此其可信域被设得很紧（`trust_in/out = 30/150`），**只在轻度海况（scale≤0.5）开启 NN**；
> 中高海况下 NN 自动平滑退化为纯 TSID 基线（+1% 即"无恶化"）。这是有意为之的保守取舍：
> 不拿一台 4.8 kg 的轻臂去赌一个在其训练域外 7 倍的网络。

**结论与建议：**

- **对 Panda（重臂）：TSID+NN 在全部扰动等级稳定降误差（−9% ～ −60%），且从不恶化**，是默认配置；
  LADRC 在轻度/标称有帮助（−27%~−43%），但 scale=2.0 转负（误差 +33%~+44%），需增益调度或仅在中低扰动开。
- **对 NERO（轻臂）：NN 残差补偿只在轻度海况（scale=0.5）显著有效（−19% ～ −48%）**；
  中高海况被可信域自动关掉、退化为纯 TSID 基线，**保证不会比不用网络更差**。
- **NERO 上 LADRC 全面有害**（scale=2.0 误差暴涨 +217%，即 10.9→34.6 mm、系统濒临失稳），
  建议 **NERO 部署只用 TSID+NN（轻度海况）或直接纯 TSID，不要叠加 LADRC**。
- 两条臂的纯 TSID 基线误差都与论文同量级（~3–5 mm 轻/中标称），印证优化型任务空间逆动力学的必要性。
  共同结论：**神经网络残差补偿是最可靠的一层，且靠可信域做了安全兜底；LADRC 必须按扰动等级与机型谨慎调度。**

---

## 5. 部署要点

- **推理快**：残差 MLP 仅 3 个隐藏层（128/128/64），CPU 单样本推理 < 0.1 ms。
- **安全（五重保护）**：
  1. **输入钳位** `clamp_x=4`（归一化后特征夹到 ±4σ），**烘焙进 ONNX**；
  2. **输出限幅** `out_scale` 由训练数据统计给出（`2×|Δτ|` 的 99.9 分位，而非硬编码常量）；
  3. **残差占比限制** `residual_frac=0.25`，补偿量最多占关节力矩限 1/4；
  4. **海况包络门控（主保护）**：取 ESKF 估计的基座线加速度 `‖a_w‖` 的**滑动包络**
     （τ=5 s），超出设计海况（NERO：1.05~1.45 m/s²）时把 NN 增益平滑衰减到 0。
     包络冷启动初值取上限，让网络先"挣得"信任、避免开局白送。
  5. **`‖q̈*‖` 可信域（副保护）**：`‖q̈*‖` 超出训练域时再叠加一层增益衰减，
     **最坏情况退化为纯 TSID 基线 —— 保证不比不用网络更差。**
  因此即便网络因分布外输入失准，叠加的力矩也被物理地关在笼子里，**不会破坏力矩限 / QP 约束**。
- **可移植**：导出 ONNX 后用 ONNXRuntime（C++/Python）即可在任意边缘设备推理；
  `docs/deployment.md` 给出 C++ 推理片段与硬件接入清单。

### 5.1 换一条臂时踩到的两个坑（已修复，值得一看）

把 Panda 上验证好的方案迁到 NERO 时，闭环一度直接发散（误差 311 mm、QP 回退 294 次），
而且**换任务就换个地方炸**。逐时刻追踪后定位到两个根因，都不是网络结构问题：

**坑 1（主因）：零空间目标位形 `q_ns` 是"另一条臂"的位形，且越过了本臂的限位。**

TSID 的零空间任务 `q̈_ns = Kp_ns(q_ns − q) − Kd_ns·q̇` 会持续把关节拉向 `q_ns`。
`TSIDGains` 里的默认值是 Panda 的位形 `[0, −45°, 0, −135°, 0, +90°, 45°]`，而 NERO 的
关节 4 只有 `[−57.9°, 122.6°]`、关节 6 只有 `[−41.8°, 54.4°]` —— **第 4、6 关节的目标直接
落在限位之外**（分别偏 180.8° 和 72.8°）。于是零空间任务一刻不停地把臂往限位上顶：
表现为任务空间误差看着不大（2–3 mm），关节却在持续漂移，贴死限位后 QP 失配、系统发散。

> 修复：`config.build_gains(kind)` 让 `q_ns` 随机械臂走（取该臂 home，并夹进限位留 0.05 rad
> 余量）。Panda 的 `PANDA_HOME` 恰好等于旧默认值，所以**已发布的 Panda 结果完全不受影响**。

**坑 2（更深一层，一度以为是上面的问题）：训练时喂给网络的 `q̈*` 分布 ≠ 闭环真实的 `q̈*` 分布。**

这条坑排查得最久，因为它会伪装成别的故障。先把话说死：

> 不要靠"感觉"给训练数据定状态量程。**先量闭环里这个量到底跑到多少，再回头定训练分布。**

实测（`output/measure_qdd.py`，每周期从 TSID 里钩出真值 `‖q̈*‖_inf`）：

| 机械臂 / 任务 | scale | p50 | p90 | p99 | 峰值 |
|---|---|---|---|---|---|
| NERO 定点 | 0.5 / 1.0 / 2.0 | 32 / 61 / 135 | 89 / 227 / 420 | 153 / **1330** / **1312** | 174 / 1919 / **4000** |
| NERO 圆周 | 0.5 / 1.0 / 2.0 | 65 / 88 / 135 | 208 / 287 / 372 | 1044 / **1767** / **1639** | 1347 / 2153 / 3553 |
| Panda 定点 | 0.5 / 1.0 / 2.0 | 8 / 10 / 14 | 13 / 15 / 82 | 14 / 19 / **1213** | 15 / 21 / **3916** |
| Panda 圆周 | 0.5 / 1.0 / 2.0 | 13 / 14 / 15 | 31 / 40 / 55 | 53 / 73 / **341** | 59 / 81 / **2662** |

（单位 rad/s²。这张表本身也是成果 —— 它同时解释了为什么 Panda 对这套参数不敏感、
而 NERO 一动就炸：轻臂惯量小，同样的跟踪误差会被 TSID 换算成高一个量级的 `q̈*` 指令。）

早期训练按 `U(-50,50)^7` 采样，后又提到量程 200 —— **对 NERO 仍然连 p50 都没盖上**，
也就是说网络在**一半以上的工作时间里都在外推**。外推的残差经 `tanh` 饱和后会吐出满量程
假力矩，对 NERO 腕部（`tau_max` 仅 30 Nm）足以直接打飞系统。

期间走过一段弯路，值得记下来：当时用**硬裁剪输入**（把 `q̈*` 截到 ±`qdd_clip` 再喂进去）
去压这个现象，一度"能用"，但它在 scale=2.0 下是**非单调**的 —— NERO 定点实测
`clip=50→28.7mm`、`20→13.3mm`、`10→10.8mm`，**再往下 `5→65.3mm` 直接崩**。没有物理解释、
只能靠试, 换个任务就换个答案。这是典型的"用饱和掩盖外推误差",错的不是数值,是思路。

> 真正修复（三步，都可解释）：
> 1. **训练分布对齐实测闭环分布**：`--qdd-lim` 提到 1500，配合跨 3 个数量级的对数均匀幅度因子，
>    使 `q̈*≈0` 的定点工况与上千 rad/s² 的尖峰工况同时被覆盖；
> 2. **按 `‖q̈*‖` 给样本加权** `w = 1/(1+(‖q̈*‖/q_ref)²)`：残差里的 `ΔM·q̈*` 随 `‖q̈*‖` 线性增长，
>    而 MSE 是平方，若不加权，单个 `‖q̈*‖=1000` 的样本对梯度的贡献是 `‖q̈*‖=30` 样本的近千倍，
>    网络会被稀有的高加速度样本绑架，恰恰在最常用、决定平均误差的中低区间精度最差；
> 3. **可信域（trust region）取代输入裁剪**：`‖q̈*‖` 超出训练域时不再篡改输入，而是把**输出**
>    平滑衰减到 0，最坏情况退化为纯 TSID 基线。单调、可解释，且**保证不会比基线更差**。

**通用教训**：换机械臂时，先核三件事 —— ① 所有**位形相关的常量**（home、`q_ns`、限位）是否
属于这条臂；② 训练用的**状态分布**是否覆盖了真实闭环的量程（先量再定，别猜）；③ 任何"Distribution shift"
都用**输出端的保护**兜底，而不是在输入端糊一个饱和函数。三者都静默失败，且都表现为
"看起来快好了但会突然炸"。

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
