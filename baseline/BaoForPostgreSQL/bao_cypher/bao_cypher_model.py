"""
Bao Cypher Baseline 模型（BaoNet：BinaryTreeConv，维度动态传入）。

直接复用原版 BaoForPostgreSQL/bao_server/net.py 的架构，
只改为 in_channels 必须显式传入（不再依赖硬编码的 ALL_TYPES）。

TreeConvolution 复用：baseline/BaoForPostgreSQL/bao_server/TreeConvolution/
"""

import sys
import os

import torch
import torch.nn as nn

_SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))                          # bao_cypher/
_BAO_SERVER   = os.path.join(os.path.dirname(_SCRIPT_DIR), 'bao_server')            # bao_server/
# _BAO_SERVER 必须用 append 而非 insert(0)，因为 bao_server/model.py 会遮蔽 research/model/ 包
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)
if _BAO_SERVER not in sys.path:
    sys.path.append(_BAO_SERVER)

from TreeConvolution.tcnn import BinaryTreeConv, TreeLayerNorm, TreeActivation, DynamicPooling
from TreeConvolution.util import prepare_trees


def left_child(x):
    """返回特征树的左子节点，叶子节点返回 None。"""
    if not isinstance(x, tuple) or len(x) != 3:
        return None
    return x[1]


def right_child(x):
    """返回特征树的右子节点，叶子节点返回 None。"""
    if not isinstance(x, tuple) or len(x) != 3:
        return None
    return x[2]


def features(x):
    """返回特征树节点的特征向量。"""
    if isinstance(x, tuple):
        return x[0]
    return x


class BaoCypherNet(nn.Module):
    """
    Bao Cypher Baseline 模型，与原版 BaoNet 完全一致。
    BinaryTreeConv → TreeLayerNorm → LeakyReLU（×3层）→ DynamicPooling → MLP → 输出

    Args:
        in_channels: 节点特征维度（由 compute_node_feature_dim(encoding) 动态计算）
    """

    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self._cuda = False

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
            nn.Linear(64, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, trees):
        """
        Args:
            trees: list of feature_tree（每个元素是叶子 np.ndarray 或 (vec, left, right) tuple）

        Returns:
            pred: Tensor [B, 1]，归一化后的 log(Execution Time)，范围 [0,1]
        """
        prepared = prepare_trees(trees, features, left_child, right_child,
                                 cuda=self._cuda)
        return self.tree_conv(prepared)

    def cuda(self):
        self._cuda = True
        return super().cuda()

    def to(self, device):
        if isinstance(device, str):
            self._cuda = device.startswith('cuda')
        elif hasattr(device, 'type'):
            self._cuda = device.type == 'cuda'
        return super().to(device)
