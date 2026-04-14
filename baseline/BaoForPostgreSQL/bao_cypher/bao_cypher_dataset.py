"""
Bao Cypher Baseline 的 Dataset（对标 Bao 原版特征，公平 baseline）。

节点特征结构（严格对标原版 Bao featurize.py）：
  算子类型 one-hot(len(type2idx)) + EstimatedRows(1，log归一化)
  = len(type2idx) + 1 维

  原版 Bao：算子 one-hot(7) + Total Cost(1) + Plan Rows(1) = 9维
  Cypher 版：算子 one-hot(动态) + EstimatedRows(1)
  不编码 table(Label)、join(关系类型)、谓词(column/filter)，与原版 Bao 保持一致。

特征树结构（与原版 Bao 一致）：
  叶子节点：np.ndarray（特征向量）
  内部节点：(特征向量, 左子树, 右子树)

二叉树处理（Cypher 子节点数不固定）：
  0子（叶子）：直接作为叶子
  1子：透传，直接返回子节点（与原版 Bao 一致）
  2子：标准二叉内部节点
  3+子：右结合折叠成二叉树

监督信号：Execution Time（毫秒），log1p → Normalizer → [0,1]
"""

import json
import sys
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

_SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))                          # bao_cypher/
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_SCRIPT_DIR)))      # research/
for _p in [_PROJECT_ROOT, _SCRIPT_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from model.database_util import Encoding

def compute_node_feature_dim(encoding) -> int:
    """返回节点特征向量维度（encoding 词典稳定后调用）。"""
    return len(encoding.type2idx) + 1   # 算子类型 one-hot + EstimatedRows(1)

def _log_norm(value, min_val, max_val):
    """log(x+1) 归一化到 [0,1]。"""
    log_val = np.log(float(value) + 1.0)
    if max_val > min_val:
        return (log_val - min_val) / (max_val - min_val)
    return 0.0

def _encode_node_feature(type_id, est_rows, encoding, est_rows_min, est_rows_max) -> np.ndarray:
    """
    将节点编码为固定维度的特征向量（严格对标原版 Bao）。

    特征结构：
      算子类型 one-hot(num_types) + EstimatedRows(1，log归一化)

    原版 Bao 对应：
      算子 one-hot(7) + Total Cost(1，log归一化) + Plan Rows(1，log归一化)
    """
    num_types = len(encoding.type2idx)

    # 算子类型 one-hot
    type_vec = np.zeros(num_types, dtype=np.float32)
    if 0 <= type_id < num_types:
        type_vec[type_id] = 1.0

    # EstimatedRows（log 归一化，对应原版的 Total Cost + Plan Rows）
    est = np.array([_log_norm(est_rows or 0, est_rows_min, est_rows_max)],
                   dtype=np.float32)

    return np.concatenate([type_vec, est])


def _build_feature_tree(plan_node, encoding, est_rows_min, est_rows_max):
    """
    递归将 Plan JSON 转换为 Bao 风格的特征树。

    返回：
      叶子节点：np.ndarray
      内部节点：(np.ndarray, 左子树, 右子树)
    """
    node_type = plan_node['operatorType']
    type_id   = encoding.encode_type(node_type)
    est_rows  = plan_node['args']['EstimatedRows']

    feat_vec  = _encode_node_feature(type_id, est_rows, encoding, est_rows_min, est_rows_max)

    children = plan_node.get('children') or []

    if len(children) == 0:
        # 叶子节点
        return feat_vec

    if len(children) == 1:
        # 单子节点：透传（与原版 Bao 一致，直接返回子节点的特征树）
        return _build_feature_tree(children[0], encoding, est_rows_min, est_rows_max)

    if len(children) == 2:
        # 标准二叉内部节点
        left  = _build_feature_tree(children[0], encoding, est_rows_min, est_rows_max)
        right = _build_feature_tree(children[1], encoding, est_rows_min, est_rows_max)
        return (feat_vec, left, right)

    # 3+ 子节点：右结合折叠成二叉树
    subtrees = [
        _build_feature_tree(child, encoding, est_rows_min, est_rows_max)
        for child in children
    ]
    # 从右往左折叠
    right_tree = subtrees[-1]
    for subtree in reversed(subtrees[:-1]):
        right_tree = (feat_vec, subtree, right_tree)
    return right_tree


def _collect_est_rows(plan_node):
    """递归收集所有节点的 EstimatedRows，用于归一化。"""
    yield plan_node['args']['EstimatedRows']
    for child in plan_node.get('children') or []:
        yield from _collect_est_rows(child)


class BaoCypherDataset(Dataset):
    """
    将 Cypher 查询计划 CSV 转换为 Bao TreeConvolution 所需的特征树格式。

    每条样本返回：
        (feature_tree, label)
        - feature_tree: 递归嵌套的 tuple/np.ndarray，叶子为 np.ndarray，内部节点为 (vec, left, right)
        - label:        Tensor scalar，归一化后的 log(Execution Time)

    Args:
        json_df:    包含查询计划的 DataFrame，列：id, json
        encoding:   model.database_util.Encoding 实例
        cost_norm:  model.util.Normalizer 实例
        reset_norm: 是否用当前数据集重置 cost_norm 的 min/max（训练集为 True）
    """

    def __init__(
        self,
        json_df: pd.DataFrame,
        encoding: Encoding,
        cost_norm,
        reset_norm: bool = False,
        label_field: str = 'cost',
    ):
        self.encoding    = encoding
        self.label_field = label_field

        plan_jsons = [json.loads(s) for s in json_df['json']]
        plan_roots = [pj['Plan'] for pj in plan_jsons]
        idxs       = list(json_df['id'])

        # ── 标签提取（根据 label_field 切换）────────────────────────────────
        if label_field == 'card':
            # 根节点实际输出行数，clip to 1 avoid log(0)
            labels_raw = [max(float(pj['Plan']['args'].get('Rows', 1)), 1.0)
                          for pj in plan_jsons]
        else:
            # 默认：Execution Time (milliseconds)
            labels_raw = [pj['Execution Time'] for pj in plan_jsons]

        # 保存原始数据，供 rebuild_feature_trees 重新构建时使用（skill 文档 7.7）
        self._idxs       = idxs
        self._plan_roots = plan_roots

        # 收集所有 EstimatedRows，计算 log 归一化的 min/max
        all_est_rows = []
        for plan_root in plan_roots:
            all_est_rows.extend(list(_collect_est_rows(plan_root)))
        log_est = np.log(np.array(all_est_rows, dtype=np.float64) + 1.0)
        self.est_rows_min = float(log_est.min())
        self.est_rows_max = float(log_est.max())

        # 解析计划树，构建特征树（同时扩充 encoding 词典）
        self.feature_trees = []
        for plan_root in plan_roots:
            tree = _build_feature_tree(
                plan_root, encoding,
                self.est_rows_min, self.est_rows_max,
            )
            self.feature_trees.append(tree)

        # 归一化标签：log1p → Normalizer → [0,1]
        # label_field='cost': log1p(Execution Time ms)
        # label_field='card': log1p(Rows)，两者归一化方式相同
        log_labels = np.log1p(np.array(labels_raw, dtype=np.float64))
        self.labels = torch.from_numpy(
            cost_norm.normalize_labels(log_labels.tolist(), reset_min_max=reset_norm)
        ).float()

        # encoding 词典稳定后，计算节点特征维度
        self.node_feature_dim = compute_node_feature_dim(encoding)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return self.feature_trees[idx], self.labels[idx]

    def rebuild_feature_trees(self, est_rows_min: float, est_rows_max: float,
                               node_feature_dim: int):
        """
        用最终稳定的 encoding 词典和归一化参数重新构建特征树。
        在 train/val/test 全部解析完后调用（skill 文档 7.7）。
        """
        self.est_rows_min     = est_rows_min
        self.est_rows_max     = est_rows_max
        self.node_feature_dim = node_feature_dim

        self.feature_trees = []
        for plan_root in self._plan_roots:
            tree = _build_feature_tree(
                plan_root, self.encoding,
                self.est_rows_min, self.est_rows_max,
            )
            self.feature_trees.append(tree)


def bao_cypher_collate(batch):
    """
    DataLoader collate 函数。
    feature_tree 是不规则的嵌套结构，不能直接 stack，返回 list。
    """
    trees  = [item[0] for item in batch]
    labels = torch.stack([item[1] for item in batch])
    return trees, labels
