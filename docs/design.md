# 设计文档：船载七自由度机械臂控制方案

本文档给出方案的数学与架构细节，便于复现与二次开发。

---

## A. 问题建模

- **机械臂**：7-DOF，浮动基（基座为一个 6-DOF 运动平台 = "云台"）。控制算法与机械臂型号
  **解耦**，目前内置两种机型：Franka-Panda（论文复现默认）与 **AgileX NERO**。
  机械臂模型不手写参数——直接由厂商 **URDF** 解析（`ship_arm/robot/urdf_io.py` 纯 numpy
  解析 `<link>` inertial 与 `<joint>` origin/axis/limit → 与 Panda 同构的 `ArmSpec`；
  NERO 的 URDF 取自官方 [agx_arm_urdf](https://github.com/agilexrobotics/agx_arm_urdf)，
  含真实质量/质心/惯量）。接入新机型只需提供一个 URDF，其余代码零改动。详见 README §3.5。
- **基座运动**：JONSWAP 海浪谱 + RAO 生成解析可微的 6-DOF 基座位姿 / 速度 / 加速度，
  峰值标定到论文量级（线速度 ≈ 0.32 m/s，角速度 ≈ 24 °/s @ scale=1.0）。
- **控制频率**：1 kHz（与论文一致）。
- **状态来源**：基座 IMU（100 Hz）+ 末端位姿（120 Hz）经 ESKF 融合，估计
  `{R_WB, p_B, ω_w, v_w, α_w, a_w}`。控制器与估计器解耦。

---

## B. TSID-QP（论文核心，式 5–12）

任务空间 PD（世界系误差 `e = log(x⁻¹ x_d)`）：
```
xddot_c = xddot_d + Kp·e + Kd·edot
```
基座耦合项（论文关键，显式前馈补偿动态耦合）：
```
η = J̇q̇ + J_B·V̇_B + J̇_B·V_B          (式 7)
```
零空间与 QP（式 10）：
```
min   ½‖J qddot − xddot_c + η‖² + ½λ‖N(qddot − qddot_ns)‖²
s.t.  τ_min ≤ M qddot + H ≤ τ_max ,   qddot_min ≤ qddot ≤ qddot_max
τ* = M qddot* + H                       (式 12)
```
> **零空间目标位形 `q_ns` 必须与机械臂匹配**（`config.build_gains(kind)`）。
> `q̈_ns = Kp_ns(q_ns − q) − Kd_ns·q̇` 会持续把关节拉向 `q_ns`；若 `q_ns` 取自另一条臂
> 且落在本臂限位之外，零空间任务就会不停把臂顶向限位（任务空间误差看着不大，关节却在
> 漂移，贴死限位后 QP 失配发散）。Panda→NERO 迁移时正是踩到这点（关节 4 偏 180.8°、
> 关节 6 偏 72.8°）。详见 README §5.1。
其中 `H = C q̇ + g(R_WB) + τ_base − JᵀF_ext`，`τ_base` 即显式补偿的基座动态耦合项。
QP 用自研 **Mehrotra 原始-对偶内点法**（稠密），不可行时回退到"加速度盒 → 力矩饱和 →
再投影"的确定性兜底，保证 1 kHz 下输出永远有限且物理合理。

---

## C. 神经网络残差补偿（本方案新增）

### C.1 动机
TSID 用**名义模型 + ESKF 估计基座**算前馈力矩
```
τ_model = M(q) qddot* + C(q,q̇) q̇ + g(q,R̂_WB) + τ_base(̂V_B, V̂̇_B)
```
真机要产生同一 `qddot*` 实际需要的力矩为
```
τ_true = M_true qddot* + H_true + τ_friction
```
二者之差
```
Δτ = τ_true − τ_model
```
就是"未建模动力学"在力矩上的体现：**摩擦、惯量/质量参数误差、未建模负载、基座耦合项的
估计残差**。这正是论文 Table V 想要的鲁棒性来源，也是云台控制里 DOB/ADRC 想在线估计的
"总扰动"的离线可学习版本。

### C.2 网络
- 输入 `x ∈ ℝ⁴⁰`：`[sin q, cos q, dq, qddot*, ω, v, α, a]`（归一化在模型内部完成）。
- 结构：`Linear→LayerNorm→ReLU ×3` + `Linear→Tanh`，输出 `Δτ̂ ∈ ℝ⁷`。
- **四重有界性**（保证网络失准也不会破坏力矩限 / 约束）：
  1. 输入钳位 `clamp_x = 4`：归一化后特征夹到 ±4σ，**烘焙进 ONNX**；
  2. 输出限幅 `out_scale`：由训练数据统计给出（`2×|Δτ|` 的 99.9 分位），
     例如 NERO 为 48.3 Nm —— 不再是硬编码常量；
  3. 在线再按 `residual_frac = 0.25` 限到关节力矩限的 1/4；
  4. **可信域** `trust_in`/`trust_out`：以 `‖q̈*‖_inf` 为横坐标，
     增益 = `clip((trust_out − ‖q̈*‖)/(trust_out − trust_in), 0, 1)`，乘在网络**输出**上。
     注意这里刻意**不再裁剪输入** —— 早期版本把 `q̈*` 硬截到 ±50 再喂进去，实测在
     scale=2.0 下非单调（50→28.7mm、20→13.3mm、10→10.8mm、5→65.3mm 崩），
     本质是"用饱和掩盖外推误差"。改为输出侧增益后，最坏情况平滑退化为纯 TSID 基线。
- 归一化均值/标准差作为 buffer 烘焙进模型 → 导出的 ONNX 直接吃"原始特征"、输出"原始力矩"。

### C.3 训练数据（离线、用仿真）
对每个随机采样 `(q, dq, qddot*, 基座状态)`，分别计算
```
τ_model = M_nominal·qddot* + H_nominal
τ_true  = M_true·qddot* + H_true + friction·dq + coulomb·tanh(dq/2e-2)
Δτ      = τ_true − τ_model
```
其中 `true` 对象含 12% 惯量误差 + 0.15 粘性摩擦 + 0.3 Nm 库仑摩擦 + 0~2.5 倍船体运动，
与论文实机必然存在的失配一致。

**采样方式（重要）**：`q̈*` 与 `dq` 采用「方向均匀 × **对数均匀幅度因子**」
`10^U(−3,0)`，量程 `[−qdd_lim, qdd_lim]`，`qdd_lim` 默认 1500 rad/s²。

两条理由：
1. 纯均匀采样多维盒子时，幅度接近 0 的区域体积占比趋近于 0（`|q̈_i|<2` 在
   `U(−50,50)^7` 里的概率约 `1e-11`），"定点保持"这类 `q̈*≈0` 的工况在训练集里根本不存在；
2. `qdd_lim` 必须由**实测闭环分布**决定。逐周期钩取 TSID 真值得到 `‖q̈*‖_inf`
   （`output/measure_qdd.py`）：NERO 在船体 scale=1.0 时 p50 就有 61~88，p99 达
   1330~1767，峰值过 2000 —— 早先设的 200 连 p50 都没盖住，网络一半时间在外推。

**样本按 `‖q̈*‖` 加权**：`w = 1/(1+(‖q̈*‖_inf/q_ref)²)`，`q_ref = 50`。
残差中的 `ΔM·q̈*` 随 `‖q̈*‖` 线性增长，而 MSE 又是平方，若不加权，单个 `‖q̈*‖=1000`
的样本对梯度的贡献是 `‖q̈*‖=30` 样本的近千倍 —— 网络会被稀有的高加速度样本绑架，
恰在最常用、决定平均跟踪误差的中低加速度区间精度最差。该权重正好抵消这种增长，
使每个对数 decade 的贡献大致均等。

### C.4 部署
```
τ_cmd = τ_model(来自 TSID-QP) + Δτ̂(x)      # Δτ̂ 经限幅
```
推理用 ONNXRuntime，CPU 单样本 < 0.1 ms。

---

## D. LADRC / ESO（云台控制常用方法）

把 LADRC 放在**任务空间**（末端 6-DOF 位姿）上，与 TSID 框架天然契合：
```
y  = [log R_ee; p_ee]           被控输出 (世界系 6D 位姿)
r  = [log R_d;  p_d ]           参考
u_des = TSID 的 xddot_c         名义加速度指令
yddot ≈ z3 + b·u               把"总扰动" z3 扩张为第三个状态
```
二阶 ESO（每控制周期）：
```
eo  = z1 − y
z1 += dt·(z2 − β1·eo)
z2 += dt·(z3 − β2·eo + b·u_prev)
z3 += dt·(−β3·eo)
u   = u_des − z3/b          # 抵消扰动后的加速度指令
accel_ff = u − u_des = −z3/b   # 返回给 TSID 加进 xddot_c
```
`β1=3ω, β2=3ω², β3=ω³`。该补偿与 TSID 的**解析**基座耦合前馈互补：TSID 处理已知部分，
LADRC 在线补偿其估计残差；神经网络则把这部分离线学下来做成前馈。三者可叠加。

> 与论文方法的关系：论文用模型 + ESKF 做"已知扰动"的解析前馈；本方案的 NN 与 LADRC 处理
> "未知/未建模扰动"，是论文框架的自然扩展，而非替代。

---

## E. ESKF 浮动基估计（论文式 28–47）

- 误差状态卡尔曼滤波，30 维增广状态（含基座位姿/速度/加速度 + IMU 零偏 + 末端位姿）。
- 标称状态里 **IMU 是观测**而非传播输入：常加速度预测模型估计基座位姿/速度/加速度，
  并用末端 FK 导出的基座位姿约束做更新。
- 论文 Table V 的消融（关 IMU / 关 EE / 关 FK / 缩维）在 `experiments/exp_eskf.py` 复现。

---

## F. 实验与结果

| 实验 | 脚本 | 对应论文 |
|------|------|----------|
| 轨迹跟踪 (Table II/IV) | `experiments/run_tracking.py` | Table II |
| NN / LADRC 消融对比 | `experiments/run_nn_study.py` | 本方案新增 |
| ESKF 消融 (Table V) | `experiments/exp_eskf.py` | Table V |
| 约束激活 / 负载 / 时标 (Fig 6–8) | `experiments/exp_analysis.py` | Fig 6–8 |
| 动态插孔 (Fig 15–16) | `experiments/exp_peg.py` | Fig 15–16 |
| 计算耗时 (Table III) | `experiments/exp_analysis.py#table3` | Table III |

**典型结论**（含 12% 惯量误差 + 摩擦的被控对象，ship scale=1.0）：

- 定点 10 s：TSID ≈ 4.5 mm；圆周 24 s：TSID ≈ 5.4 mm；8 字：TSID ≈ 5.3 mm。
- NN 残差补偿在标称失配下进一步压低误差；LADRC 在失配更剧烈时提供额外鲁棒性。
- ESKF 全配置 ≈ 0.7 mm / 0.03°；缩维滤波恶化到 ≈ 23 mm。
- TSID+NN 单控制周期 < 1.2 ms，满足 1 kHz。
