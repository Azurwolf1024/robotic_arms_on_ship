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

# 特征向量布局: [sin q(7), cos q(7), dq(7), q̈*(7), ω(3), v(3), α(3), a(3)] = 40
_QDD_SLICE = slice(21, 28)


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
def _log_uniform_mag(rng, size, lo_dec: float = -3.0, hi_dec: float = 0.0):
    """10**U(lo_dec, hi_dec) —— 对数均匀幅度因子。

    纯均匀采样 7 维盒子的致命问题: 幅度接近 0 的区域体积占比趋近于 0
    (|q̈_i|<2 在 U(-50,50)^7 里的概率约 1e-11), 于是"定点保持"这类
    q̈*≈0、dq≈0 的工况在训练集里**根本不存在**, 部署时网络只能外推。
    乘上对数均匀幅度因子后, 样本幅度跨约 2 个数量级均匀分布, 近零与
    满量程都被覆盖。
    """
    return 10.0 ** rng.uniform(lo_dec, hi_dec, size=size)


def _sample_configs(n_samples: int, robot_c, ship, rng, ship_scales=(0.3, 2.5),
                    base_noise: float = 0.05, qdd_lim: float = 200.0):
    """批量采样 (q, dq, q̈*, 基座状态), 返回用于逆动力学的数组。"""
    n = robot_c.n
    q = rng.uniform(robot_c.q_min * 0.95, robot_c.q_max * 0.95, size=(n_samples, n))
    # dq: 方向均匀 × 对数均匀幅度 -> 覆盖近静止与高速两种工况
    dq = (rng.uniform(-1.0, 1.0, size=(n_samples, n))
          * (np.asarray(robot_c.dq_max, float)[None, :] * 0.8)
          * _log_uniform_mag(rng, (n_samples, 1)))

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
    # 实际指令。
    #
    # 量程必须由**实测闭环分布**决定, 不能拍脑袋。实测 TSID 闭环 ‖q̈*‖_inf
    # (output/measure_qdd.txt) 远大于早先假设的 50:
    #     NERO 定点  scale=0.5/1.0/2.0 : p50 32/61/135, p99 153/1330/1312, 峰值 3999
    #     NERO 圆周  scale=0.5/1.0     : p50 65/88,     p99 1044/1767,     峰值 2153
    # 也就是说, 训练量程若只到 200, 网络在**一半以上的工作时间里都在外推** ——
    # 外推的残差经 tanh 饱和后会给出满量程假力矩, 对 NERO 这类轻量臂足以直接打飞。
    # 幅度用跨 3 个数量级的对数均匀因子, 保证 q̈*≈0 的工况同样被充分采样。
    qdd = (rng.uniform(-1.0, 1.0, size=(n_samples, n)) * qdd_lim
           * _log_uniform_mag(rng, (n_samples, 1)))
    return q, dq, qdd, R, p, omega, v, alpha, a


def generate_training_data(n_samples: int = 80000, payload: float = 0.0,
                           model_error: float = 0.12, friction: float = 0.15,
                           coulomb: float = 0.3, seed: int = 0,
                           ship_scales=(0.3, 2.5), kind: str = "panda",
                           qdd_lim: float = 1500.0) -> dict:
    """生成 (特征 X, 残差标签 Y) 配对训练数据。

    返回 dict: X (N,40) float32, Y (N,7) float32, meta。

    kind : "panda" | "nero" —— 选哪条机械臂的运动学/动力学与真实对象。
    """
    from ship_arm_ctl.config import build_robot
    from ship_arm.robot.panda import perturb_spec
    from ship_arm.robot.model import Robot
    from ship_arm.platform.ship import ShipMotion

    rng = np.random.default_rng(seed)
    robot_c = build_robot(kind, payload_mass=0.0)
    robot_t = build_robot(kind, payload_mass=payload)
    if model_error > 0:
        robot_t = Robot(perturb_spec(robot_t.spec, rel_mass=model_error,
                                     rel_inertia=1.6 * model_error, seed=7))
    ship = ShipMotion(); ship.scale = 1.0
    n = robot_c.n
    q, dq, qdd, R, p, omega, v, alpha, a = _sample_configs(
        n_samples, robot_c, ship, rng, ship_scales, qdd_lim=qdd_lim)

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
def sample_weights_from_features(X: np.ndarray, q_ref: float = 50.0) -> np.ndarray:
    """按 ‖q̈*‖ 给样本赋权, 让各个加速度量级对损失的贡献可比。

    为什么需要: 残差标签里 ΔM·q̈* 这一项随 ‖q̈*‖ 线性增长, 而 MSE 又是平方,
    于是单个 ‖q̈*‖=1000 的样本对梯度的贡献是 ‖q̈*‖=30 样本的 ~1000 倍。结果网络
    被"大加速度"样本绑架, 而在**出现频率最高的中低加速度区间**精度很差 ——
    偏偏那里才是决定平均跟踪误差的地方。

    取 w = 1/(1+(‖q̈*‖_inf/q_ref)²) 正好抵消 |Δτ|²∝‖q̈*‖² 的增长, 使每个
    对数 decade 的贡献大致均等 (配合 _sample_configs 的对数均匀幅度采样)。
    """
    qdd = np.abs(np.asarray(X, dtype=float)[:, _QDD_SLICE]) * FEATURE_SCALES["qdd"]
    m = np.max(qdd, axis=1)
    w = 1.0 / (1.0 + (m / float(q_ref)) ** 2)
    return w * (len(w) / w.sum())          # 归一到均值 1, 不改变整体学习率


def train_model(data: dict, hidden: int = 128, out_scale: float = None,
                epochs: int = 300, batch_size: int = 1024, lr: float = 3e-3,
                val_frac: float = 0.1, seed: int = 0, verbose: bool = True,
                clamp_x: float = 4.0, mag_weight: bool = True,
                q_ref: float = 50.0):
    import torch
    from torch.utils.data import TensorDataset, DataLoader, random_split

    X = torch.tensor(data["X"])
    Y = torch.tensor(data["Y"])
    W = None
    if mag_weight:
        W = torch.tensor(sample_weights_from_features(data["X"], q_ref=q_ref),
                         dtype=torch.float32)
        if verbose:
            print(f"  [w] 幅值加权: q_ref={q_ref}, 权重 min/median/max = "
                  f"{W.min():.3f}/{W.median():.3f}/{W.max():.3f}")
    N = X.shape[0]
    nval = int(N * val_frac)
    gen = torch.Generator().manual_seed(seed)
    tr, va = random_split(range(N), [N - nval, nval], generator=gen)
    if W is not None:
        tr_ds = TensorDataset(X[tr], Y[tr], W[tr])
        va_ds = TensorDataset(X[va], Y[va], W[va])
    else:
        tr_ds = TensorDataset(X[tr], Y[tr])
        va_ds = TensorDataset(X[va], Y[va])
    tr_dl = DataLoader(tr_ds, batch_size=batch_size, shuffle=True)
    va_dl = DataLoader(va_ds, batch_size=batch_size)

    mean = X.mean(0); std = X.std(0).clamp(min=1e-3)
    # out_scale 由数据统计给出: tanh 的满量程要略大于真实残差峰值, 这样
    # (a) 正常工作区 tanh 不饱和、拟合精度高; (b) 万一失准, 最坏输出也就
    #     是这个量级, 不会像硬编码 40 Nm 那样对轻量臂 (NERO 腕部限 30 Nm) 致命。
    if out_scale is None:
        out_scale = max(1.0, 2.0 * float(np.percentile(np.abs(data["Y"]), 99.9)))
        if verbose:
            print(f"  [init] out_scale 由数据设定 = {out_scale:.2f} Nm "
                  f"(|Δτ| 99.9% 分位 = {np.percentile(np.abs(data['Y']), 99.9):.2f})")
    model = ResidualNet(hidden=hidden, out_scale=out_scale, clamp_x=clamp_x)
    model.set_normalization(mean, std)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    best_val = float("inf"); best_state = None
    for ep in range(epochs):
        model.train()
        for batch in tr_dl:
            xb, yb = batch[0], batch[1]
            wb = batch[2] if len(batch) > 2 else None
            opt.zero_grad()
            err = model(xb) - yb
            per_sample = (err ** 2).mean(1)              # 每样本的 7 维均方
            loss = (per_sample * wb).mean() if wb is not None else per_sample.mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            vv = []
            for batch in va_dl:
                xb, yb = batch[0], batch[1]
                wb = batch[2] if len(batch) > 2 else None
                per_sample = ((model(xb) - yb) ** 2).mean(1)
                vv.append(per_sample if wb is None else per_sample * wb)
            val = float(torch.mean(torch.cat(vv)))
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
                 tau_max: np.ndarray = None,                  residual_frac: float = 0.25,
                 trust_in: float = 250.0, trust_out: float = 1500.0,
                 qdd_clip: float = 4000.0,
                 sev_in: float = 1.05, sev_out: float = 1.45,
                 sev_tau: float = 5.0, dt: float = 1e-3):
        """参数
        ----
        residual_frac: 残差补偿最多占关节力矩限的比例。这是安全兜底: 即便网络
            因分布外输入失准, 叠加的补偿量也不会超过 ``residual_frac * tau_max``,
            剩下的力矩权限仍留给 TSID-QP 做闭环修正。

        trust_in / trust_out: **可信域 (trust region)**, 单位 rad/s², 作用在
            ``‖q̈*‖_inf`` (各关节指令加速度的最大绝对值) 上::

                gain = 1                                    ‖q̈*‖ ≤ trust_in
                gain = (trust_out-‖q̈*‖)/(trust_out-trust_in)  之间线性过渡
                gain = 0                                    ‖q̈*‖ ≥ trust_out

            输出的补偿力矩整体乘 gain。**这是稳定性的关键机制**, 理由见下。

            为什么需要: 残差标签里有一项 ΔM·q̈* 随指令加速度线性放大, 于是形成
            "跟踪误差↑ → TSID 抬高 q̈* → NN 补偿↑ → 实际加速度↑ → 误差↑" 的反馈通路。
            训练样本按对数均匀幅度采样, 大 |q̈*| 区域的样本密度低、学得最差,
            而那里恰恰是反馈增益最高的地方 —— 一旦预测偏大, 这条回路就会自激。

            早期版本用 **硬裁剪输入** (把 q̈* 截到 ±qdd_clip 再喂进去) 压这现象,
            但它有两个硬伤:
              1. 喂进去的是假输入, 网络在假点上求值, 输出同样是假的;
              2. **非单调** —— 实测 NERO scale=2.0 定点任务: clip=50→28.7mm,
                 20→13.3mm, 10→10.8mm, 但 5→65.3mm 直接崩。调参靠试, 没有物理解释。
            可信域把这些全部换成"输增益"而非"改输入": 域内原样使用学到的函数,
            越出训练域平滑衰减到 0, 最坏情况退化为纯 TSID 基线 —— 单调、可解释,
            且**保证不会比基线更差**。

        sev_in / sev_out / sev_tau: **海况包络门控** (主保护), 单位 m/s², 作用在
            ESKF 给出的基座线加速度 ‖a_w‖ 的**滑动包络**上 (``env = max(decay·env, ‖a_w‖)``,
            时间常数 ``sev_tau`` 秒)。

            为什么需要第二道门: ``‖q̈*‖`` 是不稳定的**结果**而非原因。用它做门控有两个毛病 ——
            (a) 它逐周期剧烈抖动, 增益跟着抖, 等于给自己注入高频力矩纹波; (b) 实测即便
            在 NERO scale=2.0 把可信域压到 (30,100), 误差仍有 65.6mm —— 因为在
            ``‖q̈*‖<100`` 的那部分时刻里补偿仍是全额的, 足以把系统踹进不稳定区。
            **海况烈度是原因**: 由 ESKF 直接给出、随船体尺度平滑变化、单调。
            取包络而非瞬时值, 是因为瞬时 ‖a_w‖ 会周期性回到小值, 直接门控区分不开
            scale=1.0 与 2.0 (两者瞬时值区间大幅重叠), 而包络严格随尺度单调:
                船体 scale = 0.5 / 1.0 / 1.5 / 2.0 → ‖a_w‖ 峰值 ≈ 0.51 / 1.03 / 1.54 / 2.06
            于是 ``sev_in=1.1`` (略高于标称海况峰值) 与 ``sev_out=1.7`` 能把
            "标称工况全额补偿" 与 "超出设计包络平滑关掉" 干净分开。

        qdd_clip: 纯数值哨兵 (默认 4000), 防止异常大值搞坏 ONNX 输入。
            它**不是**稳定性旋钮 —— 正常运行时不该触发, 触发说明工况已远超设计范围。
        """
        self.tau_max = np.asarray(tau_max) if tau_max is not None else None
        # 残差自身的限幅 (保守), 与 tau_max (总力矩限) 区分
        self.res_clip = (residual_frac * self.tau_max
                         if (self.tau_max is not None and residual_frac is not None)
                         else None)
        # 可信域: None 表示不启用 (退化为旧的输入裁剪行为之外的全信任)
        self.trust_in = float(trust_in) if trust_in is not None else None
        self.trust_out = float(trust_out) if trust_out is not None else None
        self.qdd_clip = float(qdd_clip) if qdd_clip is not None else None
        # 海况包络门控: 状态量 (滑动最大值)
        self.sev_in = float(sev_in) if sev_in is not None else None
        self.sev_out = float(sev_out) if sev_out is not None else None
        self.sev_decay = float(np.exp(-dt / max(sev_tau, 1e-6)))
        # 初值取 sev_out (即"最不信任"): 包络只能靠观测到的小烈度**慢慢降下来**,
        # 于是网络必须先"挣得"信任, 而不是开局白送。
        # 反例实测: 若从 0 开始, 剧烈海况下开局几秒增益=1、全额补偿, 包络还没升上来
        # 系统就已经发散 —— NERO 定点 scale=2.0 从 76.5mm(不门控) 恶化到 103.6mm。
        self._env = float(sev_out) if sev_out is not None else 0.0
        # 统计外部可见: 最近一次的各分量增益, 便于诊断/日志
        self.last_gain = 1.0
        self.last_sev_gain = 1.0
        self.last_env = 0.0
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
        # 安全限幅: 残差补偿本身被限制在 res_clip (tau_max 的一个比例) 内
        clip = self.res_clip if self.res_clip is not None else self.tau_max
        if clip is not None:
            y = np.clip(y, -clip, clip)
        return y[0] if features.ndim == 1 else y

    def trust_gain(self, qdd_star: np.ndarray) -> float:
        """由 ‖q̈*‖_inf 给出 [0,1] 的可信增益 (域内 1, 域外平滑到 0)。"""
        if self.trust_in is None or self.trust_out is None:
            return 1.0
        if self.trust_out <= self.trust_in:
            return 1.0
        m = float(np.max(np.abs(np.asarray(qdd_star, dtype=float))))
        return float(np.clip((self.trust_out - m) / (self.trust_out - self.trust_in),
                             0.0, 1.0))

    def severity_gain(self) -> float:
        """由海况包络 (‖a_w‖ 滑动最大值) 给出 [0,1] 的门控增益。

        设计包络内 = 1 (全额补偿); 超出设计包络平滑到 0 (退化为纯 TSID 基线)。
        """
        if self.sev_in is None or self.sev_out is None:
            return 1.0
        if self.sev_out <= self.sev_in:
            return 1.0
        return float(np.clip((self.sev_out - self._env) / (self.sev_out - self.sev_in),
                             0.0, 1.0))

    def compensate(self, q, dq, qdd_star, est: dict) -> np.ndarray:
        """给定当前状态与 TSID 算出的 q̈*, 返回加性的残差力矩补偿。

        注意这里的分工: q̈* 以**真值**喂给网络 (只在异常大时被数值哨兵截断),
        学习到的映射因此保持在自己的定义域上; 偏离训练域的后果由输出端的
        可信增益承担 —— 宁肯不补, 不肯瞎补。
        """
        qdd_raw = np.asarray(qdd_star, dtype=float)

        # --- 主门: 海况包络 ---
        a_w = np.asarray(est.get("a_w", np.zeros(3)), dtype=float)
        self._env = max(self.sev_decay * self._env, float(np.linalg.norm(a_w)))
        self.last_env = self._env
        self.last_sev_gain = self.severity_gain()

        # --- 副门: ‖q̈*‖ 可信域 (只挡极端离群, 不削标称收益) ---
        self.last_gain = self.last_sev_gain * self.trust_gain(qdd_raw)
        if self.last_gain <= 0.0:
            return np.zeros_like(qdd_raw)

        qdd_star = qdd_raw
        if self.qdd_clip is not None:
            qdd_star = np.clip(qdd_star, -self.qdd_clip, self.qdd_clip)
        x = build_features(q, dq, qdd_star, est)
        return self.last_gain * self.predict(x)
