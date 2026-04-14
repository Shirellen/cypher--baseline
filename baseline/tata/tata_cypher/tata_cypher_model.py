"""
TATA Cypher Baseline 模型。

与原版 TATA 完全一致的架构：
  - BaoNet：三层 BinaryTreeConv（256→128→64）+ DynamicPooling，输出 64 维表示向量
  - Prediction：MLP（64→256→256→1，sigmoid 输出），与原版 cost_est.py 中的 Prediction 一致

关键差异（Cypher 版本）：
  - in_channels 动态计算（= len(encoding.type2idx) + 2），不硬编码为 12
  - BaoNet 和 Prediction 直接复用原版代码，不做任何修改
  - TreeConvolution 从 baseline/tata/algorithms/bao/TreeConvolution/ 复用
"""

import sys
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

_SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))                     # tata_cypher/
_TATA_DIR     = os.path.dirname(_SCRIPT_DIR)                                   # tata/
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_TATA_DIR))                   # research/
_TREECONV_DIR = os.path.join(_TATA_DIR, 'algorithms', 'bao', 'TreeConvolution')

for _p in [_PROJECT_ROOT, _TREECONV_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tcnn import BinaryTreeConv, TreeLayerNorm, TreeActivation, DynamicPooling


# ── BaoNet（与原版 tata/algorithms/bao/net.py 完全一致）──────────────────────

class BaoNet(nn.Module):
    """
    树卷积表示网络（与原版 TATA BaoNet 完全一致）。
    输入：TataBatch（flat_trees + idxes）
    输出：[batch_size, 64] 的表示向量
    """

    def __init__(self, in_channels: int):
        super().__init__()
        self._in_channels = in_channels

        self.tree_conv = nn.Sequential(
            BinaryTreeConv(in_channels, 256),
            TreeLayerNorm(),
            TreeActivation(nn.LeakyReLU()),
            BinaryTreeConv(256, 128),
            TreeLayerNorm(),
            TreeActivation(nn.LeakyReLU()),
            BinaryTreeConv(128, 64),
            TreeLayerNorm(),
            DynamicPooling(),
        )

    def forward(self, x):
        return self.tree_conv((x.trees, x.idxes))

    def in_channels(self) -> int:
        return self._in_channels


# ── Prediction（与原版 tata/training/cost_est.py 中的 Prediction 完全一致）──

class Prediction(nn.Module):
    """
    代价预测 MLP（与原版 TATA Prediction 完全一致）。
    输入：[batch_size, in_feature]（BaoNet 输出的 64 维表示向量）
    输出：[batch_size, 1]，sigmoid 归一化到 (0, 1)
    """

    def __init__(self, in_feature: int = 64, hid_units: int = 256,
                 contract: int = 1, mid_layers: bool = True, res_con: bool = True):
        super().__init__()
        self.mid_layers = mid_layers
        self.res_con    = res_con

        self.out_mlp1 = nn.Linear(in_feature, hid_units)
        self.mid_mlp1 = nn.Linear(hid_units, hid_units // contract)
        self.mid_mlp2 = nn.Linear(hid_units // contract, hid_units)
        self.out_mlp2 = nn.Linear(hid_units, 1)

    def forward(self, features):
        hid = F.relu(self.out_mlp1(features))
        if self.mid_layers:
            mid = F.relu(self.mid_mlp1(hid))
            mid = F.relu(self.mid_mlp2(mid))
            if self.res_con:
                hid = hid + mid
            else:
                hid = mid
        return torch.sigmoid(self.out_mlp2(hid))


# ── 完整模型（BaoNet + Prediction）──────────────────────────────────────────

class TataModel(nn.Module):
    """
    TATA Cypher Baseline 完整模型：BaoNet + Prediction。

    Args:
        in_channels: 节点特征维度（= len(encoding.type2idx) + 2，动态计算）
        hid_units:   Prediction MLP 隐藏层维度（默认 256，与原版一致）
    """

    def __init__(self, in_channels: int, hid_units: int = 256):
        super().__init__()
        self.bao_net    = BaoNet(in_channels)
        self.prediction = Prediction(in_feature=64, hid_units=hid_units)

    def forward(self, x):
        rep = self.bao_net(x)       # [B, 64]
        out = self.prediction(rep)  # [B, 1]
        return out
