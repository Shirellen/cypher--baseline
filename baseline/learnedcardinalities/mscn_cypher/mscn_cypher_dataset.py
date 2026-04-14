"""
MSCN Cypher Baseline 的 Dataset（方案 A：用 Label one-hot + EstimatedRows 替代 bitmap）。

三路输入：
  samples:    每个涉及的 Label → [Label one-hot + EstimatedRows]
              sample_feats = len(encoding.table2idx) + 1
  predicates: 每个谓词 → [属性 one-hot + 操作符 one-hot + 归一化值]
              predicate_feats = len(encoding.col2idx) + len(encoding.op2idx) + 1
  joins:      每个关系类型 → [关系类型+方向 one-hot]
              join_feats = len(encoding.join2idx)

监督信号：plan['Plan']['args']['Rows']（根节点实际行数，Cardinality ground truth）
归一化：log(Rows) → min-max → [0,1]（与原版 MSCN 一致）
"""

import json
import sys
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

_SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))                          # mscn_cypher/
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_SCRIPT_DIR)))          # research/
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from model.database_util import TreeNode, Encoding
from model.cypher_format_filter import cypher_format_filter
from model.cypher_format_join import cypher_format_join


def compute_feature_dims(encoding):
    """返回三路特征维度 (sample_feats, predicate_feats, join_feats)"""
    num_labels   = len(encoding.table2idx)
    num_cols     = len(encoding.col2idx)
    num_ops      = len(encoding.op2idx)
    num_joins    = len(encoding.join2idx)
    sample_feats    = num_labels + 1          # Label one-hot + EstimatedRows
    predicate_feats = num_cols + num_ops + 1  # 属性 one-hot + 操作符 one-hot + 归一化值
    join_feats      = max(num_joins, 1)       # 关系类型+方向 one-hot（至少1维）
    return sample_feats, predicate_feats, join_feats


class MscnCypherDataset(Dataset):
    """
    将 Cypher 查询计划 CSV 转换为 MSCN SetConv 所需的三路集合输入格式。

    每条样本返回：
        (samples, predicates, joins, sample_mask, predicate_mask, join_mask, label)
        - samples:        Tensor [max_num_labels, sample_feats]
        - predicates:     Tensor [max_num_predicates, predicate_feats]
        - joins:          Tensor [max_num_joins, join_feats]
        - sample_mask:    Tensor [max_num_labels, 1]
        - predicate_mask: Tensor [max_num_predicates, 1]
        - join_mask:      Tensor [max_num_joins, 1]
        - label:          Tensor scalar，归一化后的 log(Cardinality)

    Args:
        json_df:    包含查询计划的 DataFrame，列：id, json
        encoding:   model.database_util.Encoding 实例
        card_norm:  归一化器（min_val/max_val），None 表示用当前数据集计算
        reset_norm: 是否重置 min_val/max_val（训练集为 True）
    """

    def __init__(
        self,
        json_df: pd.DataFrame,
        encoding: Encoding,
        card_norm=None,
        reset_norm: bool = False,
        label_field: str = 'card',
    ):
        self.encoding    = encoding
        self.label_field = label_field

        # 解析 Plan JSON
        plan_jsons     = [json.loads(s) for s in json_df['json']]
        cypher_queries = [self._extract_cypher(pj['Plan']) for pj in plan_jsons]

        # ── 标签提取（根据 label_field 切换）────────────────────────────────
        if label_field == 'cost':
            # Execution Time（毫秒），clip to 1 avoid log(0)
            cardinalities = [max(1.0, pj['Execution Time']) for pj in plan_jsons]
        else:
            # 默认：根节点实际行数（Cardinality ground truth）
            # Rows 可能为 0（查询结果为空），取 max(1, Rows) 避免 log(0)
            cardinalities = [max(1, pj['Plan']['args'].get('Rows', 1)) for pj in plan_jsons]

        # 解析计划树，提取三路集合（同时扩充 encoding 词典）
        idxs = list(json_df['id'])
        self.raw_samples_list    = []  # list of list of (label_str, est_rows)
        self.raw_predicates_list = []  # list of list of (col_id, op_id, val)
        self.raw_joins_list      = []  # list of list of join_id

        for idx, pj, cypher in zip(idxs, plan_jsons, cypher_queries):
            raw_s, raw_p, raw_j = self._extract_sets(pj['Plan'], idx, encoding, cypher)
            self.raw_samples_list.append(raw_s)
            self.raw_predicates_list.append(raw_p)
            self.raw_joins_list.append(raw_j)

        # encoding 词典稳定后，计算特征维度
        self.sample_feats, self.predicate_feats, self.join_feats = compute_feature_dims(encoding)

        # 归一化 Cardinality：log → min-max → [0,1]（与原版 MSCN 一致）
        log_cards = np.array([np.log(float(c)) for c in cardinalities], dtype=np.float64)
        if reset_norm or card_norm is None or card_norm.get('min_val') is None:
            self.min_val = float(log_cards.min())
            self.max_val = float(log_cards.max())
            if card_norm is not None:
                card_norm['min_val'] = self.min_val
                card_norm['max_val'] = self.max_val
        else:
            self.min_val = card_norm['min_val']
            self.max_val = card_norm['max_val']

        labels_norm = (log_cards - self.min_val) / max(self.max_val - self.min_val, 1e-8)
        labels_norm = np.clip(labels_norm, 0.0, 1.0)
        self.labels = torch.from_numpy(labels_norm).float()
        self.raw_cardinalities = cardinalities

        # 计算 padding 上限（在所有数据解析完后确定）
        self.max_num_labels     = max(max(len(s) for s in self.raw_samples_list), 1)
        self.max_num_predicates = max(max(len(p) for p in self.raw_predicates_list), 1)
        self.max_num_joins      = max(max(len(j) for j in self.raw_joins_list), 1)

        # 预计算 padding 后的张量
        self.samples_list    = []
        self.predicates_list = []
        self.joins_list      = []
        self.sample_masks    = []
        self.predicate_masks = []
        self.join_masks      = []

        for raw_s, raw_p, raw_j in zip(
            self.raw_samples_list, self.raw_predicates_list, self.raw_joins_list
        ):
            s_tensor, s_mask = self._encode_samples(raw_s)
            p_tensor, p_mask = self._encode_predicates(raw_p)
            j_tensor, j_mask = self._encode_joins(raw_j)
            self.samples_list.append(s_tensor)
            self.predicates_list.append(p_tensor)
            self.joins_list.append(j_tensor)
            self.sample_masks.append(s_mask)
            self.predicate_masks.append(p_mask)
            self.join_masks.append(j_mask)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return (
            self.samples_list[idx],
            self.predicates_list[idx],
            self.joins_list[idx],
            self.sample_masks[idx],
            self.predicate_masks[idx],
            self.join_masks[idx],
            self.labels[idx],
        )

    # ── 集合提取 ──────────────────────────────────────────────────────────────

    def _extract_sets(self, plan_node, idx, encoding, cypher):
        """
        遍历计划树，提取三路原始集合（在 encoding 词典扩充前调用，只做解析不做编码）。
        返回：
          raw_samples:    list of (label_str, est_rows)  每个 Label 一条
          raw_predicates: list of (col_id, op_id, val)   每个谓词一条
          raw_joins:      list of join_id                 每个关系类型一条（去重）
        """
        all_nodes = self._collect_all_nodes(plan_node, idx, encoding, cypher)

        # 1. Label 集合（去重，保留最大 EstimatedRows）
        label_est = {}
        for node in all_nodes:
            for label in (node.table if isinstance(node.table, list) else [node.table]):
                if label and label not in label_est:
                    label_est[label] = node.EstimatedRows or 0.0

        raw_samples = [(label, est) for label, est in label_est.items()]
        if not raw_samples:
            raw_samples = [('', 0.0)]  # 至少一条，避免空集合

        # 2. 谓词集合（所有节点的所有谓词，不去重）
        raw_predicates = []
        for node in all_nodes:
            fd = node.filterDict
            for col_id, op_id, val in zip(fd['colId'], fd['opId'], fd['val']):
                raw_predicates.append((col_id, op_id, val))
        if not raw_predicates:
            raw_predicates = [(0, 0, 0.0)]

        # 3. 关系类型集合（去重）
        join_ids = list({node.join for node in all_nodes if node.join > 0})
        if not join_ids:
            join_ids = [0]

        return raw_samples, raw_predicates, join_ids

    def _collect_all_nodes(self, plan_node, idx, encoding, cypher):
        """递归解析 Plan JSON 为 TreeNode 列表（扩充 encoding 词典）。"""
        node_type = plan_node['operatorType']
        type_id = encoding.encode_type(node_type)
        filters, alias = cypher_format_filter(plan_node, cypher)
        join = cypher_format_join(plan_node)
        join_id = encoding.encode_join(join)
        filters_encoded = encoding.encode_filters(filters, alias)

        node = TreeNode(
            nodeType=node_type,
            typeId=type_id,
            filt=filters,
            card=None,
            join=join_id,
            join_str=join,
            filterDict=filters_encoded,
            details=plan_node['args'].get('Details', ''),
            EstimatedRows=plan_node['args']['EstimatedRows'],
        )
        node.table = alias
        node.table_id = encoding.encode_table(alias)
        node.query_id = idx
        node.feature = None

        result = [node]
        for child in plan_node.get('children') or []:
            result.extend(self._collect_all_nodes(child, idx, encoding, cypher))
        return result

    # ── 编码（在 encoding 词典稳定后调用）────────────────────────────────────

    def _encode_samples(self, raw_samples):
        """将 Label 集合编码为 padding 后的张量。"""
        encoding_table = self.encoding.table2idx
        vecs = []
        for label, est_rows in raw_samples:
            vec = np.zeros(self.sample_feats, dtype=np.float32)
            label_idx = encoding_table.get(label)
            if label_idx is not None and label_idx < self.sample_feats - 1:
                vec[label_idx] = 1.0
            # EstimatedRows 做 log1p 归一化到 [0,1] 附近（log1p(1e8)≈18.4，除以 20 归一化）
            vec[-1] = float(np.log1p(est_rows)) / 20.0
            vecs.append(vec)

        # padding 到 max_num_labels
        num_pad = self.max_num_labels - len(vecs)
        mask = np.ones((len(vecs), 1), dtype=np.float32)
        tensor = np.array(vecs, dtype=np.float32)
        if num_pad > 0:
            tensor = np.vstack([tensor, np.zeros((num_pad, self.sample_feats), dtype=np.float32)])
            mask   = np.vstack([mask,   np.zeros((num_pad, 1), dtype=np.float32)])
        return torch.FloatTensor(tensor), torch.FloatTensor(mask)

    def _encode_predicates(self, raw_predicates):
        """将谓词集合编码为 padding 后的张量。"""
        num_cols = len(self.encoding.col2idx)
        num_ops  = len(self.encoding.op2idx)
        vecs = []
        for col_id, op_id, val in raw_predicates:
            vec = np.zeros(self.predicate_feats, dtype=np.float32)
            if 0 <= col_id < num_cols:
                vec[col_id] = 1.0
            if 0 <= op_id < num_ops:
                vec[num_cols + op_id] = 1.0
            vec[-1] = float(val)
            vecs.append(vec)

        num_pad = self.max_num_predicates - len(vecs)
        mask = np.ones((len(vecs), 1), dtype=np.float32)
        tensor = np.array(vecs, dtype=np.float32)
        if num_pad > 0:
            tensor = np.vstack([tensor, np.zeros((num_pad, self.predicate_feats), dtype=np.float32)])
            mask   = np.vstack([mask,   np.zeros((num_pad, 1), dtype=np.float32)])
        return torch.FloatTensor(tensor), torch.FloatTensor(mask)

    def _encode_joins(self, join_ids):
        """将关系类型集合编码为 padding 后的张量。"""
        num_joins = len(self.encoding.join2idx)
        vecs = []
        for join_id in join_ids:
            vec = np.zeros(self.join_feats, dtype=np.float32)
            if 0 <= join_id < num_joins:
                vec[join_id] = 1.0
            vecs.append(vec)

        num_pad = self.max_num_joins - len(vecs)
        mask = np.ones((len(vecs), 1), dtype=np.float32)
        tensor = np.array(vecs, dtype=np.float32)
        if num_pad > 0:
            tensor = np.vstack([tensor, np.zeros((num_pad, self.join_feats), dtype=np.float32)])
            mask   = np.vstack([mask,   np.zeros((num_pad, 1), dtype=np.float32)])
        return torch.FloatTensor(tensor), torch.FloatTensor(mask)

    def pad_to(self, max_num_labels, max_num_predicates, max_num_joins,
               sample_feats=None, predicate_feats=None, join_feats=None):
        """
        将所有张量 pad 到指定的最大长度，并可选地对齐特征维度（用于对齐 train/val/test 集）。

        Args:
            max_num_labels:     序列长度上限（samples 路）
            max_num_predicates: 序列长度上限（predicates 路）
            max_num_joins:      序列长度上限（joins 路）
            sample_feats:       强制指定 sample 特征维度（None 表示保持当前值）
            predicate_feats:    强制指定 predicate 特征维度（None 表示保持当前值）
            join_feats:         强制指定 join 特征维度（None 表示保持当前值）
        """
        # 更新序列长度
        self.max_num_labels     = max_num_labels
        self.max_num_predicates = max_num_predicates
        self.max_num_joins      = max_num_joins

        # 更新特征维度（val/test 强制对齐到 train 的维度，截断多余的 join 类型）
        if sample_feats is not None:
            self.sample_feats = sample_feats
        if predicate_feats is not None:
            self.predicate_feats = predicate_feats
        if join_feats is not None:
            self.join_feats = join_feats

        self.samples_list    = []
        self.predicates_list = []
        self.joins_list      = []
        self.sample_masks    = []
        self.predicate_masks = []
        self.join_masks      = []

        for raw_s, raw_p, raw_j in zip(
            self.raw_samples_list, self.raw_predicates_list, self.raw_joins_list
        ):
            s_tensor, s_mask = self._encode_samples(raw_s)
            p_tensor, p_mask = self._encode_predicates(raw_p)
            j_tensor, j_mask = self._encode_joins(raw_j)
            self.samples_list.append(s_tensor)
            self.predicates_list.append(p_tensor)
            self.joins_list.append(j_tensor)
            self.sample_masks.append(s_mask)
            self.predicate_masks.append(p_mask)
            self.join_masks.append(j_mask)

    @staticmethod
    def _extract_cypher(plan_node: dict) -> str:
        cypher = plan_node.get('args', {}).get('cypher', '')
        if cypher:
            return cypher
        for child in plan_node.get('children') or []:
            result = MscnCypherDataset._extract_cypher(child)
            if result:
                return result
        return ''


def mscn_cypher_collate(batch):
    """DataLoader collate 函数（张量维度已一致，直接 stack）。"""
    samples, predicates, joins, s_mask, p_mask, j_mask, labels = zip(*batch)
    return (
        torch.stack(samples),
        torch.stack(predicates),
        torch.stack(joins),
        torch.stack(s_mask),
        torch.stack(p_mask),
        torch.stack(j_mask),
        torch.stack(labels),
    )