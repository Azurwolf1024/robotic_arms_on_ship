"""
神经网络残差补偿: 特征构造、数据生成、训练、ONNX 导出与在线推理。

设计思想
--------
部署时控制器用 *名义模型 + ESKF 估计基座* 算出前馈力矩
    τ_model = M(q) q̈* + C(q, q̇) q̇ + g(q, R̂_WB) + τ_base(̂V_B, V̂̇_B)
真实被控对象要产生同一 q̈* 实际需要的力矩是
    τ_true  = M_true q̈* + H_true + τ_friction
二者之差 Δτ = τ_true − τ_model 就是"未建模动力学"在力矩上的体现: 摩擦、惯量/质量
参数误差、未建模负载、基座耦合项的估计残差。

本模块离线用仿真生成 (x, Δτ) 配对数据训练一个小 MLP, 部署时把它作为前馈
补偿叠加到 τ_model 上:
    τ_cmd = τ_model + Δτ̂(x),   Δτ̂ 经 tanh 限幅到 ±out_scale。
这等价于"学习到的前馈/扰动补偿", 是 DOB/ADRC 的离线可部署版本, 且不会破坏
QP 力矩限与约束。
"""

from __future__ import annotations

import os
import numpy as np

from .model_nn import ResidualNet

# 特征归一化尺度 (仅用于训练时把输入拉到 ~O(1), 不进入 ONNX)
FEATURE_SCALES = dict(
    dq=2.0,        # rad/s
    qdd=15.0,      # rad/s^2
    w=0.5,         # rad/s
    v=0.5,         # m/s
    a_ang=2.0,     # rad/s^2
    a_lin=2.0,     # m/s^2
)


def build_features(q: np.ndarray, dq: np.ndarray, qdd_star: np.ndarray,
                   base: dict) -> np.ndarray:
    """构造 40 维原始特征向量 (未归一化; 归一化在模型内部完成)。

    base 需含 omega_w, v_w, alpha_w, a_w (世界系)。
    """
    s = FEATURE_SCALES
    x = np.concatenate([
        np.sin(np.asarray(q, dtype=float)),
        np.cos(np.asarray(q, dtype=float)),
        np.asarray(dq, dtype=float) / s["dq"],
        np.asarray(qdd_star, dtype=float) / s["qdd"],
        np.asarray(base["omega_w"], dtype=float) / s["w"],
        np.asarray(base["v_w"], dtype=float) / s["v"],
        np.asarray(base["alpha_w"], dtype=float) / s["a_ang"],
        np.asarray(base["a_w"], dtype=float) / s["a_lin"],
    ])
    return x.astype(np.float32)


# --------------------------------------------------------------------------- #
# 训练数据生成
# --------------------------------------------------------------------------- #
def _sample_configs(n_samples: int, robot_c, ship, rng, ship_scales=(0.3, 2.5),
                    base_noise: float = 0.05):
    """批量采样 (q, dq, q̈*, 基座状态), 返回用于逆动力学的数组。"""
    n = robot_c.n
    q = rng.uniform(robot_c.q_min * 0.95, robot_c.q_max * 0.95, size=(n_samples, n))
    dq = rng.uniform(-robot_c.dq_max, robot_c.dq_max, size=(n_samples, n)) * 0.8

    # 基座状态: 在不同尺度/时刻上从船体模型采样, 并叠加少量估计误差噪声
    scales = rng.uniform(*ship_scales, size=n_samples)
    ts = rng.uniform(0.0, 30.0, size=n_samples)
    omega = np.zeros((n_samples, 3))
    v = np.zeros((n_samples, 3))
    alpha = np.zeros((n_samples, 3))
    a = np.zeros((n_samples, 3))
    R = np.zeros((n_samples, 3, 3))
    p = np.zeros((n_samples, 3))
    for i in range(n_samples):
        ship.scale = float(scales[i])
        st = ship.world_state(float(ts[i]))
        R[i] = st["R_WB"]; p[i] = st["p_B"]
        omega[i] = st["omega_w"]; v[i] = st["v_w"]
        alpha[i] = st["alpha_w"]; a[i] = st["a_w"]
    # 估计误差: 角速度 ~5%、线加速度 ~5% 的相对高斯噪声
    omega += rng.normal(0.0, base_noise * 0.5, omega.shape)
    v += rng.normal(0.0, base_noise * 0.5, v.shape)
    alpha += rng.normal(0.0, base_noise * 2.0, alpha.shape)
    a += rng.normal(0.0, base_noise * 2.0, a.shape)

    # q̈*: 在 *真实可达* 的关节加速度范围内采样。注意 robot.accel_bounds 给的是
    # 由力矩限/惯量推出的极宽松运动学限 (可达 ±2000 rad/s²), 远超过实机与 TSID 的
    # 实际指令。这里用符合 Panda 实际工作范围的 ±qdd_lim 采样, 使残差与部署分布一致。
    qdd_lim = 50.0
    qdd = rng.uniform(-qdd_lim, qdd_lim, size=(n_samples, n))
    return q, dq, qdd, R, p, omega, v, alpha, a


def generate_training_data(n_samples: int = 80000, payload: float = 0.0,
                           model_error: float = 0.12, friction: float = 0.15,
                           coulomb: float = 0.3, seed: int = 0,
                           ship_scales=(0.3, 2.5)) -> dict:
    """生成 (特征 X, 残差标签 Y) 配对训练数据。

    返回 dict: X (N,40) float32, Y (N,7) float32, meta。
    """
    from ship_arm.robot.panda import make_panda
    from ship_arm.robot.model import Robot
    from ship_arm.platform.ship import ShipMotion

    rng = np.random.default_rng(seed)
    robot_c = Robot(make_panda(tool_mass=0.73, payload_mass=0.0))
    robot_t = Robot(make_panda(tool_mass=0.73, payload_mass=payload))
    if model_error > 0:
        from ship_arm.robot.panda import perturb_spec
        robot_t = Robot(perturb_spec(robot_t.spec, rel_mass=model_error,
                                     rel_inertia=1.6 * model_error, seed=7))
    ship = ShipMotion(); ship.scale = 1.0
    n = robot_c.n

    q, dq, qdd, R, p, omega, v, alpha, a = _sample_configs(
        n_samples, robot_c, ship, rng, ship_scales)

    X = np.zeros((n_samples, ResidualNet.IN_DIM), dtype=np.float32)
    Y = np.zeros((n_samples, n), dtype=np.float32)

    for i in range(n_samples):
        rows = np.array([[*omega[i], *v[i], *alpha[i], *a[i]]], dtype=float)
        base_pose = (R[i], p[i])
        tc = robot_c.state_terms(base_pose, q[i], dq[i], rows, want_M=True, want_J=False)
        tt = robot_t.state_terms(base_pose, q[i], dq[i], rows, want_M=True, want_J=False)
        tau_model = tc["M"] @ qdd[i] + tc["tau"][0]
        tau_true = tt["M"] @ qdd[i] + tt["tau"][0] \
            + friction * dq[i] + coulomb * np.tanh(dq[i] / 2e-2)
        base = dict(omega_w=omega[i], v_w=v[i], alpha_w=alpha[i], a_w=a[i])
        X[i] = build_features(q[i], dq[i], qdd[i], base)
        Y[i] = (tau_true - tau_model).astype(np.float32)

    return dict(X=X, Y=Y, meta=dict(
        n_samples=n_samples, payload=payload, model_error=model_error,
        friction=friction, coulomb=coulomb, ship_scales=ship_scales))


# --------------------------------------------------------------------------- #
# 训练
# --------------------------------------------------------------------------- #
def train_model(data: dict, hidden: int = 128, out_scale: float = 40.0,
                epochs: int = 300, batch_size: int = 1024, lr: float = 3e-3,
                val_frac: float = 0.1, seed: int = 0, verbose: bool = True):
    import torch
    from torch.utils.data import TensorDataset, DataLoader, random_split

    X = torch.tensor(data["X"])
    Y = torch.tensor(data["Y"])
    N = X.shape[0]
    nval = int(N * val_frac)
    gen = torch.Generator().manual_seed(seed)
    tr, va = random_split(range(N), [N - nval, nval], generator=gen)
    tr_ds = TensorDataset(X[tr], Y[tr])
    va_ds = TensorDataset(X[va], Y[va])
    tr_dl = DataLoader(tr_ds, batch_size=batch_size, shuffle=True)
    va_dl = DataLoader(va_ds, batch_size=batch_size)

    mean = X.mean(0); std = X.std(0).clamp(min=1e-3)
    model = ResidualNet(hidden=hidden, out_scale=out_scale)
    model.set_normalization(mean, std)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    mse = torch.nn.MSELoss()

    best_val = float("inf"); best_state = None
    for ep in range(epochs):
        model.train()
        for xb, yb in tr_dl:
            opt.zero_grad()
            loss = mse(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            val = float(torch.mean(torch.stack([mse(model(xb), yb) for xb, yb in va_dl])))
        if val < best_val:
            best_val = val; best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if verbose and (ep % 50 == 0 or ep == epochs - 1):
            print(f"  ep {ep:4d}  val MSE {val:.5f}  (best {best_val:.5f})")
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


# --------------------------------------------------------------------------- #
# ONNX 导出
# --------------------------------------------------------------------------- #
def export_onnx(model: ResidualNet, path: str, test_parity: bool = True) -> str:
    import torch
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    model.eval()
    dummy = torch.zeros(1, ResidualNet.IN_DIM, dtype=torch.float32)
    torch.onnx.export(
        model, dummy, path,
        input_names=["x"], output_names=["tau_res"],
        dynamic_axes={"x": {0: "N"}, "tau_res": {0: "N"}},
        opset_version=17,
    )
    if test_parity:
        import onnxruntime as ort
        import numpy as np
        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        rng = np.random.default_rng(0)
        x = rng.standard_normal((37, ResidualNet.IN_DIM)).astype(np.float32)
        with torch.no_grad():
            yt = model(torch.tensor(x)).numpy()
        yo = sess.run(None, {"x": x})[0]
        err = float(np.abs(yt - yo).max())
        assert err < 1e-4, f"ONNX parity failed: max err {err}"
        print(f"  [onnx] parity OK (max abs err {err:.2e})")
    print(f"  [onnx] exported -> {path}")
    return path


# --------------------------------------------------------------------------- #
# 在线推理封装
# --------------------------------------------------------------------------- #
class ResidualCompensator:
    """加载 ONNX 模型 (或 torch 权重) 做实时残差力矩补偿。"""

    def __init__(self, onnx_path: str = None, torch_model: ResidualNet = None,
                 tau_max: np.ndarray = None):
        self.tau_max = np.asarray(tau_max) if tau_max is not None else None
        self.out_scale = 40.0
        self._sess = None
        self._model = None
        if onnx_path is not None and os.path.exists(onnx_path):
            try:
                import onnxruntime as ort
                opts = ort.SessionOptions()
                opts.intra_op_num_threads = 1
                self._sess = ort.InferenceSession(onnx_path, sess_options=opts,
                                                  providers=["CPUExecutionProvider"])
                # 读 out_scale
                import onnx
                m = onnx.load(onnx_path)
                for init in m.graph.initializer:
                    if init.name.startswith("out_scale") or init.name == "4":
                        pass
                print(f"  [nn] loaded ONNX compensator: {onnx_path}")
            except Exception as e:  # 退回 torch
                print(f"  [nn] ONNX load failed ({e}); will need torch_model")
        if self._sess is None and torch_model is not None:
            self._model = torch_model.eval()
            self.out_scale = torch_model.out_scale
            print("  [nn] using torch fallback compensator")

    # 兼容: 同时保存训练时的 out_scale
    def set_out_scale(self, v: float):
        self.out_scale = float(v)

    def predict(self, features: np.ndarray) -> np.ndarray:
        """features: (40,) 或 (N,40) float32 -> 残差力矩 (7,) / (N,7)。"""
        x = np.asarray(features, dtype=np.float32)
        if x.ndim == 1:
            x = x[None, :]
        if self._sess is not None:
            y = self._sess.run(None, {"x": x})[0]
        elif self._model is not None:
            import torch
            with torch.no_grad():
                y = self._model(torch.tensor(x)).numpy()
        else:
            raise RuntimeError("compensator has no ONNX session and no torch model")
        y = np.asarray(y, dtype=float)
        if self.tau_max is not None:
            y = np.clip(y, -self.tau_max, self.tau_max)
        return y[0] if features.ndim == 1 else y

    def compensate(self, q, dq, qdd_star, est: dict) -> np.ndarray:
        """给定当前状态与 TSID 算出的 q̈*, 返回加性的残差力矩补偿。"""
        x = build_features(q, dq, qdd_star, est)
        return self.predict(x)
