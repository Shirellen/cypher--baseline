"""
LEON Cypher Baseline 模型（方案 A：特征维度动态传入）。

架构：
  LEON TreeConvolution（plan encoder）→ tree_out_dim 维 plan embedding
  → Prediction MLP → 预测 log(Execution Time)

NODE_FEATURE_DIM 和 QUERY_FEATURE_DIM 不再使用常量，
必须在实例化时显式传入（由 LeonCypherDataset 的 node_feature_dim / query_feature_dim 提供）。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

# 把 baseline/LEON/ 加入 sys.path，使 LEON 内部的 `from util import ...` 能正确解析
_LEON_DIR = os.path.dirname(os.path.abspath(__file__))
if _LEON_DIR not in sys.path:
    sys.path.insert(0, _LEON_DIR)

from util.treeconv import TreeConvolution


class LeonCypherModel(nn.Module):
    """
    LEON TreeConv encoder + Prediction MLP。

    Args:
        query_feature_dim: 查询级全局特征维度（由 CypherQueryFeaturizer 动态计算，必须显式传入）。
        node_feature_dim:  计划树节点特征维度（由 CypherNodeFeaturizer 动态计算，必须显式传入）。
        tree_out_dim:      TreeConv 输出的 plan embedding 维度（TreeMaxPool 后）。
        mlp_hidden_dim:    Prediction MLP 隐藏层维度。
    """

    def __init__(
        self,
        query_feature_dim: int,
        node_feature_dim: int,
        tree_out_dim: int = 128,
        mlp_hidden_dim: int = 256,
    ):
        super().__init__()

        self.query_feature_dim = query_feature_dim
        self.node_feature_dim = node_feature_dim

        # LEON TreeConvolution 作为 plan encoder
        self.tree_encoder = TreeConvolution(
            feature_size=query_feature_dim,
            plan_size=node_feature_dim,
            label_size=tree_out_dim,
        )

        # Prediction MLP，与 model/model_transformer_job.py 中的 Prediction 结构一致
        self.predictor = Prediction(
            in_feature=tree_out_dim,
            hid_units=mlp_hidden_dim,
        )

    def forward(
        self,
        query_feats: torch.Tensor,
        trees: torch.Tensor,
        indexes: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            query_feats: [B, query_feature_dim]  查询级全局特征
            trees:       [B, node_feature_dim, max_nodes]  计划树节点特征
            indexes:     [B, max_nodes*3, 1]  树卷积索引

        Returns:
            pred: [B, 1]  预测值（归一化后的 log Execution Time，范围 [0,1]）
        """
        plan_emb = self.tree_encoder(query_feats, trees, indexes)  # [B, tree_out_dim]
        return self.predictor(plan_emb)                             # [B, 1]


class Prediction(nn.Module):
    """
    与 model/model_transformer_job.py 中 Prediction 结构完全一致的 MLP。

    结构：
      Linear(in_feature → hid_units) → ReLU
      → Linear(hid_units → hid_units) → ReLU  （残差连接）
      → Linear(hid_units → 1) → sigmoid
    """

    def __init__(self, in_feature: int = 128, hid_units: int = 256):
        super().__init__()
        self.out_mlp1 = nn.Linear(in_feature, hid_units)
        self.mid_mlp1 = nn.Linear(hid_units, hid_units)
        self.mid_mlp2 = nn.Linear(hid_units, hid_units)
        self.out_mlp2 = nn.Linear(hid_units, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        hid = F.relu(self.out_mlp1(features))
        mid = F.relu(self.mid_mlp1(hid))
        mid = F.relu(self.mid_mlp2(mid))
        hid = hid + mid  # 残差连接
        return torch.sigmoid(self.out_mlp2(hid))