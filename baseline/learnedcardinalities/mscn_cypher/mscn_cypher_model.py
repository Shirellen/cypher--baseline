"""
MSCN Cypher Baseline 模型（SetConv，维度动态传入）。

直接复用原版 SetConv 架构，只改为维度必须显式传入（不再硬编码）。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SetConv(nn.Module):
    """
    MSCN 的 SetConv 模型，与原版完全一致。
    三路集合输入 → 各自 MLP + mean pooling → concat → 输出预测。

    Args:
        sample_feats:    samples 路特征维度（Label one-hot + EstimatedRows）
        predicate_feats: predicates 路特征维度（属性 one-hot + 操作符 one-hot + 归一化值）
        join_feats:      joins 路特征维度（关系类型+方向 one-hot）
        hid_units:       隐藏层维度
    """

    def __init__(self, sample_feats: int, predicate_feats: int, join_feats: int, hid_units: int = 256):
        super().__init__()
        self.sample_mlp1    = nn.Linear(sample_feats,    hid_units)
        self.sample_mlp2    = nn.Linear(hid_units,       hid_units)
        self.predicate_mlp1 = nn.Linear(predicate_feats, hid_units)
        self.predicate_mlp2 = nn.Linear(hid_units,       hid_units)
        self.join_mlp1      = nn.Linear(join_feats,      hid_units)
        self.join_mlp2      = nn.Linear(hid_units,       hid_units)
        self.out_mlp1       = nn.Linear(hid_units * 3,   hid_units)
        self.out_mlp2       = nn.Linear(hid_units,       1)

    def forward(self, samples, predicates, joins, sample_mask, predicate_mask, join_mask):
        """
        Args:
            samples:          [B, max_num_labels, sample_feats]
            predicates:       [B, max_num_predicates, predicate_feats]
            joins:            [B, max_num_joins, join_feats]
            sample_mask:      [B, max_num_labels, 1]
            predicate_mask:   [B, max_num_predicates, 1]
            join_mask:        [B, max_num_joins, 1]

        Returns:
            pred: [B, 1]，归一化后的 log(Cardinality)，范围 [0,1]
        """
        # samples 路
        hid_sample = F.relu(self.sample_mlp1(samples))
        hid_sample = F.relu(self.sample_mlp2(hid_sample))
        hid_sample = hid_sample * sample_mask
        hid_sample = torch.sum(hid_sample, dim=1)
        sample_norm = sample_mask.sum(dim=1).clamp(min=1.0)
        hid_sample = hid_sample / sample_norm

        # predicates 路
        hid_pred = F.relu(self.predicate_mlp1(predicates))
        hid_pred = F.relu(self.predicate_mlp2(hid_pred))
        hid_pred = hid_pred * predicate_mask
        hid_pred = torch.sum(hid_pred, dim=1)
        pred_norm = predicate_mask.sum(dim=1).clamp(min=1.0)
        hid_pred = hid_pred / pred_norm

        # joins 路
        hid_join = F.relu(self.join_mlp1(joins))
        hid_join = F.relu(self.join_mlp2(hid_join))
        hid_join = hid_join * join_mask
        hid_join = torch.sum(hid_join, dim=1)
        join_norm = join_mask.sum(dim=1).clamp(min=1.0)
        hid_join = hid_join / join_norm

        hid = torch.cat((hid_sample, hid_pred, hid_join), dim=1)
        hid = F.relu(self.out_mlp1(hid))
        return torch.sigmoid(self.out_mlp2(hid))