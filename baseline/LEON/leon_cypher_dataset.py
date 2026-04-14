"""
LEON Cypher Baseline 的 Dataset（方案 A：去掉直方图，对标 LEON 原版特征）。

节点特征 = typeId(1) + joinId(1) + 谓词(9) + mask(3) + table_id(N) + EstimatedRows(1)
查询特征 = 算子频次(num_op) + join频次(num_join) + 谓词列频次(num_col) + 树深度/节点数/根EstRows(3)

NODE_FEATURE_DIM 和 QUERY_FEATURE_DIM 依赖 encoding，在 Dataset 初始化后动态确定。
"""

import json
import sys
import os
import copy

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# ── 路径设置 ─────────────────────────────────────────────────────────────────
# 1. 项目根目录（research/）加入 sys.path，使 model/ 可被导入
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# 2. baseline/LEON/ 加入 sys.path，使 LEON 内部的 `from util import ...` 能正确解析
_LEON_DIR = os.path.dirname(os.path.abspath(__file__))
if _LEON_DIR not in sys.path:
    sys.path.insert(0, _LEON_DIR)

from model.database_util import TreeNode, Encoding
from model.cypher_format_filter import cypher_format_filter
from model.cypher_format_join import cypher_format_join
from util.cypher_featurizer import (
    CypherNodeFeaturizer,
    CypherQueryFeaturizer,
    compute_node_feature_dim,
    compute_query_feature_dim,
)
from util.cypher_treeconv_adapter import (
    cypher_tree_conv_featurize,
    cypher_query_featurize,
    _ensure_features,
)


class LeonCypherDataset(Dataset):
    """
    将 Cypher 查询计划 CSV 转换为 LEON TreeConvolution 所需的输入格式。

    每条样本返回：
        (query_feats, trees, indexes, label)
        - query_feats: Tensor [QUERY_FEATURE_DIM]
        - trees:       Tensor [NODE_FEATURE_DIM, max_nodes]
        - indexes:     Tensor [max_nodes*3, 1]
        - label:       Tensor scalar，归一化后的 log(Execution Time)

    Args:
        json_df:    包含查询计划的 DataFrame，列：id, json（Plan JSON 字符串）
        encoding:   model.database_util.Encoding 实例（算子/join/table 编码）
        cost_norm:  model.util.Normalizer 实例，用于归一化 Execution Time
        reset_norm: 是否用当前数据集重置 cost_norm 的 min/max（训练集为 True，验证/测试集为 False）
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
        self.cost_norm   = cost_norm
        self.label_field = label_field

        # 解析 Plan JSON
        plan_jsons     = [json.loads(plan_str) for plan_str in json_df['json']]
        plan_roots_raw = [pj['Plan'] for pj in plan_jsons]
        cypher_queries = [self._extract_cypher(pj['Plan']) for pj in plan_jsons]

        # ── 标签提取（根据 label_field 切换）────────────────────────────────
        if label_field == 'card':
            # 根节点实际输出行数，clip to 1 avoid log(0)
            labels_raw = [max(float(pj['Plan']['args'].get('Rows', 1)), 1.0)
                          for pj in plan_jsons]
        else:
            # 默认：Execution Time（毫秒）
            labels_raw = [pj['Execution Time'] for pj in plan_jsons]

        # 归一化：log1p → Normalizer → [0,1]（两种标签归一化方式相同）
        log_labels = np.log1p(np.array(labels_raw, dtype=np.float64))
        self.raw_labels = labels_raw
        self.labels = torch.from_numpy(
            cost_norm.normalize_labels(log_labels.tolist(), reset_min_max=reset_norm)
        ).float()

        # 解析每条查询的 TreeNode 树（会动态扩充 encoding 词典）
        # 注意：_traverse_plan 不赋值 feature，等 encoding 词典稳定后再统一赋值
        idxs = list(json_df['id'])
        self.tree_roots = []
        for idx, plan_root, cypher in zip(idxs, plan_roots_raw, cypher_queries):
            root = self._traverse_plan(plan_root, idx, encoding, cypher)
            self.tree_roots.append(root)

        # 在所有 _traverse_plan 完成后，encoding 词典已稳定，再初始化 featurizer
        # 这样 featurizer 的维度和 query_feats_list 的维度才能一致
        self.node_featurizer = CypherNodeFeaturizer(encoding)
        self.query_featurizer = CypherQueryFeaturizer(encoding)
        self.node_feature_dim = compute_node_feature_dim(encoding)
        self.query_feature_dim = compute_query_feature_dim(encoding)

        # encoding 词典稳定后，统一为所有节点赋值特征
        for root in self.tree_roots:
            self._assign_features(root)

        # 预计算查询级全局特征（在 encoding 词典稳定后计算，维度与 query_feature_dim 一致）
        self.query_feats_list = [
            torch.from_numpy(self.query_featurizer(root)).float()
            for root in self.tree_roots
        ]

        # 预计算 (trees, indexes)，每条查询单独处理（batch_size=1）
        self.trees_list = []
        self.indexes_list = []
        for root in self.tree_roots:
            # _ensure_features 确保二叉化前所有节点 feature 维度与当前 featurizer 一致
            _ensure_features(root, self.node_featurizer)
            trees, indexes = cypher_tree_conv_featurize([root], self.node_featurizer)
            self.trees_list.append(trees.squeeze(0))    # [NODE_FEATURE_DIM, max_nodes]
            self.indexes_list.append(indexes.squeeze(0))  # [max_nodes*3, 1]

    def __len__(self) -> int:
        return len(self.tree_roots)

    def pad_query_feats_to(self, target_dim: int):
        """
        将 query_feats_list 中所有向量 pad 或截断到 target_dim 维。
        用于对齐 train/val/test 集之间因 encoding 词典扩充导致的维度不一致问题。
        """
        padded = []
        for feat in self.query_feats_list:
            current_dim = feat.shape[0]
            if current_dim < target_dim:
                padding = torch.zeros(target_dim - current_dim, dtype=feat.dtype)
                feat = torch.cat([feat, padding], dim=0)
            elif current_dim > target_dim:
                feat = feat[:target_dim]
            padded.append(feat)
        self.query_feats_list = padded
        self.query_feature_dim = target_dim

    def __getitem__(self, idx: int):
        return (
            self.query_feats_list[idx],
            self.trees_list[idx],
            self.indexes_list[idx],
            self.labels[idx],
        )

    @staticmethod
    def _extract_cypher(plan_node: dict) -> str:
        """
        从 Plan JSON 中提取原始 Cypher 查询字符串。
        Neo4j 把 cypher 字段放在根节点（ProduceResults）的 args 里。
        """
        cypher = plan_node.get('args', {}).get('cypher', '')
        if cypher:
            return cypher
        # 兜底：递归找最深处的 cypher 字段
        for child in plan_node.get('children') or []:
            result = LeonCypherDataset._extract_cypher(child)
            if result:
                return result
        return ''

    def _traverse_plan(self, plan: dict, idx, encoding: Encoding, cypher: str) -> TreeNode:
        """
        递归将 Plan JSON 解析为 TreeNode 树，并赋值方案 A 节点特征。
        复用 model/dataset_schema_gat.py 中 traversePlan 的相同解析逻辑，
        但节点特征改用 CypherNodeFeaturizer（去掉直方图）。
        """
        node_type = plan['operatorType']
        type_id = encoding.encode_type(node_type)

        filters, alias = cypher_format_filter(plan, cypher)
        join = cypher_format_join(plan)
        join_id = encoding.encode_join(join)
        filters_encoded = encoding.encode_filters(filters, alias)

        root = TreeNode(
            nodeType=node_type,
            typeId=type_id,
            filt=filters,
            card=None,
            join=join_id,
            join_str=join,
            filterDict=filters_encoded,
            details=plan['args'].get('Details', ''),
            EstimatedRows=plan['args']['EstimatedRows'],
        )
        root.table = alias
        root.table_id = encoding.encode_table(alias)
        root.query_id = idx

        for subplan in plan.get('children') or []:
            child = self._traverse_plan(subplan, idx, encoding, cypher)
            child.parent = root
            root.addChild(child)

        # 注意：feature 不在这里赋值，等 encoding 词典稳定后由 _assign_features 统一赋值
        root.feature = None
        return root

    def _assign_features(self, node: TreeNode):
        """递归为树中所有节点赋值方案 A 节点特征（在 encoding 词典稳定后调用）。"""
        node.feature = self.node_featurizer(node)
        for child in node.children:
            self._assign_features(child)


def leon_cypher_collate(batch):
    """
    将 LeonCypherDataset 的一个 batch 拼接为 TreeConvolution 所需的张量格式。
    不同查询的计划树节点数不同，trees 和 indexes 需要 padding 到同一长度。

    Args:
        batch: list of (query_feats, trees, indexes, label)

    Returns:
        query_feats:    Tensor [B, QUERY_FEATURE_DIM]
        padded_trees:   Tensor [B, NODE_FEATURE_DIM, max_nodes]
        padded_indexes: Tensor [B, max_index_len, 1]
        labels:         Tensor [B]
    """
    query_feats_list, trees_list, indexes_list, labels_list = zip(*batch)

    query_feats = torch.stack(query_feats_list, dim=0)   # [B, Q_DIM]
    labels = torch.stack(labels_list, dim=0)              # [B]

    # trees padding
    node_dim = trees_list[0].shape[0]
    max_nodes = max(t.shape[1] for t in trees_list)
    padded_trees = torch.zeros(len(trees_list), node_dim, max_nodes)
    for i, tree in enumerate(trees_list):
        padded_trees[i, :, :tree.shape[1]] = tree

    # indexes padding
    max_index_len = max(idx_t.shape[0] for idx_t in indexes_list)
    padded_indexes = torch.zeros(len(indexes_list), max_index_len, 1, dtype=torch.long)
    for i, idx_t in enumerate(indexes_list):
        padded_indexes[i, :idx_t.shape[0], :] = idx_t

    return query_feats, padded_trees, padded_indexes, labels
