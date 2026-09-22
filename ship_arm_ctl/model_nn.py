"""
神经网络残差补偿的模型定义 (PyTorch)。

网络任务
--------
给定可测量的状态特征 x, 预测"未建模动力学"引起的残差关节力矩
    Δτ = τ_true − τ_model  ∈ ℝ^7
其中 τ_model 是名义模型 + ESKF 估计基座给出的前馈力矩, τ_true 是真实被控对象
为产生同一 q̈* 实际所需的力矩。Δτ 包含了: 摩擦、惯量/质量参数误差、未建模负载、
以及基座耦合项的估计残差。

部署要点
--------
* 仅用全连接 + LayerNorm + ReLU, 体积小、推理快 (CPU 上 < 0.1 ms)。
* 归一化均值/标准差作为 buffer 烘焙进模型, 因此导出的 ONNX 直接吃"原始特征"
  输出"原始力矩", 部署端无需自己维护归一化。
* 输出经 tanh 限幅到 ±out_scale, 保证即便网络失准, 补偿量也不会破坏 QP 力矩限。
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ResidualNet(nn.Module):
    """残差力矩 MLP。输入 40 维特征, 输出 7 维残差力矩 (Nm)。"""

    IN_DIM = 40
    OUT_DIM = 7

    def __init__(self, hidden: int = 128, out_scale: float = 40.0):
        super().__init__()
        self.hidden = hidden
        self.out_scale = float(out_scale)
        # 烘焙进模型的归一化统计量 (导出 ONNX 后依旧生效)
        self.register_buffer("input_mean", torch.zeros(self.IN_DIM))
        self.register_buffer("input_std", torch.ones(self.IN_DIM))

        self.net = nn.Sequential(
            nn.Linear(self.IN_DIM, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.LayerNorm(hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, self.OUT_DIM),
            nn.Tanh(),
        )

    def set_normalization(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.input_mean.copy_(torch.as_tensor(mean, dtype=torch.float32))
        self.input_std.copy_(torch.as_tensor(std, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xn = (x - self.input_mean) / self.input_std
        return self.out_scale * self.net(xn)
