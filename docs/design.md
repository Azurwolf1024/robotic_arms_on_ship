# 设计文档：船载七自由度机械臂控制方案

本文档给出方案的数学与架构细节，便于复现与二次开发。

---

## A. 问题建模

- **机械臂**：7-DOF（Franka-Panda 风格），浮动基（基座为一个 6-DOF 运动平台 = "云台"）。
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
- 输出经 `tanh` 限幅到 **±40 Nm**（≤ 力矩限 87 的 50%），叠加到 QP 力矩之上，
  **不会破坏力矩限 / 约束**。
- 归一化均值/标准差作为 buffer 烘焙进模型 → 导出的 ONNX 直接吃"原始特征"、输出"原始力矩"。

### C.3 训练数据（离线、用仿真）
对每个随机采样 `(q, dq, qddot*, 基座状态)`，分别计算
```
τ_model = M_nominal·qddot* + H_nominal
τ_true  = M_true·qddot* + H_true + friction·dq + coulomb·tanh(dq/2e-2)
Δτ      = τ_true − τ_model
```
其中 `true` 对象含 12% 惯量误差 + 0.15 粘性摩擦 + 0.3 Nm 库仑摩擦 + 0~2.5 倍船体运动，
与论文实机必然存在的失配一致。基座状态在 `qddot* ∈ [−50, 50] rad/s²`（与 TSID 实际指令
同量级）内采样，保证残差分布与部署一致。

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
