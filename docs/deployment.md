# 部署指南 (Deployment)

本文说明如何把 `robotic_arms_on_ship` 的控制方案真正落到真机。整体思路是
**"仿真即部署"**: 控制器、ESKF 估计器、NN 残差补偿、LADRC 的代码与真机完全一致,
只是状态/力矩的**来源**由 `SimBridge` (仿真) 换成真实硬件接口 (`FrankaInterface`
或你自己的实现)。因此离线用仿真验证过的链路, 上线几乎零改动。

---

## 1. 部署链路总览

```
  关节编码器 q, dq  ─┐
  基座 IMU (acc,gyro)├─► ESKF ─► est{R_WB,p_B,ω_w,v_w,α_w,a_w}
  平台编码器(可选)    ┘
                          │
  任务参考 p_d,R_d        ▼
                          TSID-QP ──┬─► q̈* ──► LADRC(在线) ─┐
                          (力矩)    │                       │
                                    └─► τ_model             │
                                                            ▼
                                          NN 残差补偿 Δτ̂(x) ─┤
                                                            ▼
                                              τ_cmd = τ_model + Δτ̂ + LADRC前馈
                                                            │
                                                    关节力矩接口
```

三层补偿各司其职、互不冲突:

| 层 | 来源 | 作用 | 是否在线学习 |
|----|------|------|--------------|
| ① TSID-QP | 名义模型 + ESKF | 约束下的最优力矩 + 解析基座耦合前馈 | 否 |
| ② NN 残差 | `residual_net.onnx` | 离线学出的 Δτ=τ_true−τ_model (摩擦/参数误差/负载/耦合残差) | 否 (离线训练) |
| ③ LADRC | 末端位姿误差 | 在线 ESO 估计"总扰动"并前馈抵消 | 是 (自适应) |

---

## 2. 已导出的模型

`models/residual_net.onnx` —— 已用 `training/train_nn.py` 训练并导出, 验证集
MSE ≈ 0.037, 与 PyTorch 数值一致 (parity < 3e-6)。

- **输入**: 节点名 `x`, 形状 `[N, 40]`, `float32` (单帧时 `N=1`)。
- **输出**: 节点名 `tau_res`, 形状 `[N, 7]`, `float32`, 单位 Nm。
- **归一化已烘焙**: 模型内部已含 `input_mean`/`input_std` buffer, 调用方只需喂
  **原始特征**, 输出即 **原始力矩**, 无需自己维护归一化。
- **输出限幅**: 网络末端 `tanh`, 乘以 `out_scale=40.0`, 因此单关节补偿恒在
  `±40 Nm` 内 (Panda 限 87 Nm), 即便网络失准也不会破坏 QP 力矩限。

### 40 维特征构造 (部署端必须逐元素一致)

特征 = `[sin q, cos q, dq/2, qdd*/15, ω/0.5, v/0.5, α/2, a/2]`, 共 `7+7+7+7+3+3+3+3 = 40`。

| 段 | 含义 | 除数 (s) | 维度 |
|----|------|----------|------|
| sin q | 关节角正弦 | — | 7 |
| cos q | 关节角余弦 | — | 7 |
| dq / 2.0 | 关节角速度 | 2.0 rad/s | 7 |
| qdd* / 15.0 | TSID 算出的期望关节加速度 | 15.0 rad/s² | 7 |
| ω / 0.5 | 基座角速度(世界系) | 0.5 rad/s | 3 |
| v / 0.5 | 基座线速度(世界系) | 0.5 m/s | 3 |
| α / 2.0 | 基座角加速度(世界系) | 2.0 rad/s² | 3 |
| a / 2.0 | 基座线加速度(世界系) | 2.0 m/s² | 3 |

> `qdd*` 是 TSID 解算出来的期望关节加速度 (即 `out["qdd"]`), 不是实测加速度。
> 基座量必须是 **世界系** (与 `eskf.as_controller_state` 输出一致)。

---

## 3. Python 端部署 (最快验证)

`ship_arm_ctl/realtime_loop.py` 是 1 kHz 实时闭环入口:

```bash
# 仿真验证整条部署链路 (开 NN)
python -m ship_arm_ctl.realtime_loop --mode sim --duration 15 --ship-scale 1.0
# 关 NN, 看基线误差
python -m ship_arm_ctl.realtime_loop --mode sim --duration 15 --no-nn
# 开 LADRC 在线扰动补偿
python -m ship_arm_ctl.realtime_loop --mode sim --duration 15 --ladrc
```

关键参数 (`ControllerConfig`):

```python
from ship_arm_ctl.controller import ControllerConfig, ShipArmController

cfg = ControllerConfig(
    use_nn=True,            # 是否加载 ONNX 残差补偿
    use_ladrc=False,        # 是否开启 LADRC
    nn_onnx="models/residual_net.onnx",
    ladrc_wo=15.0,          # LADRC 带宽 (rad/s), wo 越大观测/补偿越快
    ladrc_b0=1.0,           # 控制增益
    accel_clip=18.0,        # LADRC 前馈加速度限幅
)
ctrl = ShipArmController(robot, dt=1e-3, cfg=cfg)

# 每控制周期:
tau_cmd, info = ctrl.step(q, dq, ref, est, ee_pose_meas=(R_ee, p_ee))
# ref = {"R_d":(3,3), "p_d":(3,), "xd_dot":(6,), "xd_ddot":(6,)}
# est = eskf.as_controller_state(dt)
```

`ShipArmController.step` 内部顺序: ① TSID-QP 求 `τ_model` 与 `qdd*` → ② 若
`use_ladrc`, 用 `ee_pose_meas` 经 ESO 求加速度前馈并入 `qdd*` → ③ 若 `use_nn`,
用 `(q, dq, qdd*, est)` 过 ONNX 得 `Δτ̂`, 叠加到 `τ_model` → ④ 限幅到 `τ_max`。

---

## 4. C++ / ONNXRuntime 侧推理 (真机实时)

真机控制器常用 C++ (libfranka / ROS2)。下面给出**只替换 NN 推理**的最小片段
(ESKF、TSID-QP、LADRC 的 C++ 实现由你沿用同一数学, 见 `docs/design.md`)。
网络很小 (3×Linear+LayerNorm, <1 ms CPU), 用 CPU EP 即可。

```cpp
#include <onnxruntime_cxx_api.h>
#include <array>
#include <vector>
#include <cmath>
#include <cstring>

// 40 维特征 -> 7 维残差力矩 (Nm)。与 ship_arm_ctl/nn_comp.build_features 逐元素一致。
std::array<float,7> nn_residual(Ort::Session& sess, Ort::AllocatorWithDefaultOptions& alloc,
                                const std::array<double,7>& q,
                                const std::array<double,7>& dq,
                                const std::array<double,7>& qdd_star,   // TSID 输出的 q̈*
                                const std::array<double,3>& w,          // 基座角速度(世界系)
                                const std::array<double,3>& v,          // 基座线速度
                                const std::array<double,3>& alpha,      // 基座角加速度
                                const std::array<double,3>& a) {        // 基座线加速度
    std::array<float,40> x{};
    int k = 0;
    for (int i=0;i<7;i++){ x[k++] = static_cast<float>(std::sin(q[i])); }   // sin q
    for (int i=0;i<7;i++){ x[k++] = static_cast<float>(std::cos(q[i])); }   // cos q
    for (int i=0;i<7;i++){ x[k++] = static_cast<float>(dq[i]  / 2.0);   }   // dq /2
    for (int i=0;i<7;i++){ x[k++] = static_cast<float>(qdd_star[i]/15.0);}   // qdd*/15
    for (int i=0;i<3;i++){ x[k++] = static_cast<float>(w[i]  / 0.5);    }   // ω /0.5
    for (int i=0;i<3;i++){ x[k++] = static_cast<float>(v[i]  / 0.5);    }   // v /0.5
    for (int i=0;i<3;i++){ x[k++] = static_cast<float>(alpha[i]/2.0);   }   // α /2
    for (int i=0;i<3;i++){ x[k++] = static_cast<float>(a[i]  / 2.0);    }   // a /2

    std::array<int64_t,2> shape{1,40};
    Ort::Value in = Ort::Value::CreateTensor<float>(alloc, x.data(), x.size(),
                                                    shape.data(), shape.size());
    const char* in_names[]  = {"x"};
    const char* out_names[] = {"tau_res"};
    auto out = sess.Run(Ort::RunOptions{nullptr}, in_names, &in, 1, out_names, 1);
    float* y = out[0].GetTensorMutableData<float>();
    std::array<float,7> tau_res{};
    std::memcpy(tau_res.data(), y, 7*sizeof(float));
    return tau_res;   // 已含 tanh*40 限幅, 直接叠加到 τ_model
}

// 会话初始化 (1 kHz 循环外做一次):
//   Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "ship_arm"};
//   Ort::SessionOptions so; so.SetIntraOpNumThreads(1);
//   Ort::Session sess{env, "models/residual_net.onnx", so};
```

> **注意**: 输入节点名必须 `x`, 输出 `tau_res`, 形状 `[1,40]`→`[1,7]` (已配
> dynamic_axes, 也可批量)。归一化/限幅已在模型图内, C++ 只做上面的线性缩放。

---

## 5. 真机接入清单 (Hardware Bring-up)

`ship_arm_ctl/hardware.py` 已定义接口, 真机只需实现对应 `read_*` / `command_*`:

1. **坐标系标定 (最关键)**
   - 确定平台→世界系 (`R_WB`, `p_B`): 用平台编码器直读 (若有), 否则由 ESKF 估计。
   - 若用 Franka, 其 `O_F` 与本文世界系需做一次手眼/基坐标系标定, 把所有
     姿态量统一到同一世界系。
   - IMU 测得的是**本体坐标系** (acc, gyro), ESKF 内部会乘 `R_WB` 转到世界系,
     接入时务必分清本体/世界系。

2. **实现接口**
   - `JointInterface.read_state()` → `(q, dq)` 来自关节编码器。
   - `JointInterface.command_torque(tau)` → 下发关节力矩 (Nm), 用 libfranka
     力矩接口或 ROS2 `control_msgs` 力矩控制器。
   - `ImuInterface.read()` → `{"acc":(3,), "gyro":(3,)}` (本体系)。
   - `ForceTorqueInterface.read()` → 腕部 6D 力旋量 (用于导纳外环, 可选)。
   - `EndEffectorTracker.read()` → `(R_ee, p_ee)` 末端位姿外测量
     (用于 LADRC / 导纳; 没有则用 ESKF+FK 估计, 见 `engine.py`)。

3. **把 `FrankaInterface` 占位替换为真实实现**, 然后:
   ```bash
   python -m ship_arm_ctl.realtime_loop --mode real --duration 60
   ```

4. **安全**
   - 真机务必先开碰撞检测 / 力矩限幅 (Panda `TAU_MAX=[87,87,87,87,12,12,12]`)。
   - 首跑用小 `ship-scale`、小 `radius` 轨迹, 观察 `τ_cmd` 是否超出 `τ_max`、
     ESKF 是否发散 (看 `est` 与真值的位姿误差)。
   - NN 输出已 `tanh` 限幅, 但 TSID 的 QP 力矩限仍由 `robot.tau_max` 把关。

---

## 6. 实时性预算 (1 kHz)

| 模块 | 典型耗时 (Python, 单核) | 说明 |
|------|--------------------------|------|
| ESKF 一步 | ~0.05 ms | 预测 + IMU/Pose 更新 |
| TSID-QP | ~0.2–0.5 ms | Mehrotra 内点法, `n=7` 小问题 |
| NN 残差 (ONNX CPU) | <0.1 ms | 网络极小 |
| LADRC | <0.05 ms | 仅几个标量递推 |
| **合计** | **<1 ms** | 留足 1 kHz 余量 |

仿真端实测单帧 CPU 约 0.3 ms, 实时性裕度充足; C++ 实现只会更快。

---

## 7. 再训练 / 更新残差网

若换了工具、负载或机械臂型号, 残差分布会变, 重新生成数据并导出即可:

```bash
python training/train_nn.py --n 80000 --epochs 300 --hidden 128 \
    --data-cache models/train_data.npz --out models/residual_net.onnx
```

- `payload` (负载质量)、`model_error` (惯量失配比)、`friction`/`coulomb`
  (摩擦) 都可调, 见 `nn_comp.generate_training_data`。
- 导出后旧 ONNX 直接被覆盖, 部署端无需改代码。
- 若真机有实测数据, 可把 `generate_training_data` 换成"真机采集 (x, Δτ)"
  配对 (Δτ = τ_cmd − τ_model 在稳态附近的差值), 做在线微调。
