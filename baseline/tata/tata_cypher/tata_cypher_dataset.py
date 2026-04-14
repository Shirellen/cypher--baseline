"""
TATA Cypher Baseline 的 Dataset。

TATA 使用 BaoNet（树卷积）作为表示层，与 Bao baseline 共享相同的树结构和特征编码方式。

节点特征（skill 文档 7.8，对标 Bao：无 table/column 编码）：
  算子类型 one-hot(len(type2idx)) + EstimatedRows_log_norm(1) + EstimatedRows_log_norm(1)
  = len(type2idx) + 2 维（动态，不硬编码）

  原版 Bao/TATA 特征：算子 one-hot(10) + cost_est_log_norm(1) + card_est_log_norm(1) = 12 维
  Cypher 替换：
    - 算子 one-hot：SQL 10种固定算子 → Cypher 动态算子词典（encoding.type2idx）
    - cost_est（Total Cost）：Cypher 无此字段，用 log(EstimatedRows+1) 替代
    - card_est（Plan Rows）：直接对应 EstimatedRows，用 log(EstimatedRows+1)

树结构：严格二叉树（BinaryTreeConv 要求），多子节点需要二叉化：
  - 0 子节点：叶子节点，直接返回特征向量（tuple 长度为 1 或 2）
  - 1 子节点：透传，返回子节点的特征树
  - 2 子节点：标准二叉节点 (feat, left, right)
  - 3+ 子节点：链式二叉化，将多余子节点嵌套为右子树

标签归一化：与原版 TATA 一致，使用 log(x+0.001) + min-max 归一化。
"""

import json
import sys
import os

import numpy as np
import torch
from torch.utils.data import Dataset

_SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))                     # tata_cypher/
_TATA_DIR     = os.path.dirname(_SCRIPT_DIR)                                   # tata/
_BASELINE_DIR = os.path.dirname(_TATA_DIR)                                     # baseline/
_PROJECT_ROOT = os.path.dirname(_BASELINE_DIR)                                 # research/
for _p in [_PROJECT_ROOT, _SCRIPT_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from model.database_util import Encoding
from model.cypher_format_filter import cypher_format_filter
from model.cypher_format_join import cypher_format_join


# ── 节点特征维度 ──────────────────────────────────────────────────────────────

def compute_node_feature_dim(encoding) -> int:
    """返回节点特征向量维度（encoding 词典稳定后调用）。"""
    return len(encoding.type2idx) + 2  # 算子 one-hot + EstimatedRows_log×2


# ── 节点特征编码 ──────────────────────────────────────────────────────────────

def _encode_node_feature(plan_node: dict, encoding,
                         est_log_min: float, est_log_max: float) -> np.ndarray:
    """
    将 plan_node dict 编码为节点特征向量（对标 Bao/TATA 原版，无 table/column）。

    特征结构：
      算子类型 one-hot(num_types) + EstimatedRows_log_norm(1) + EstimatedRows_log_norm(1)

    两个数值特征都用同一个 EstimatedRows，对应原版的 cost_est 和 card_est 两个槽位，
    保持特征结构与原版 Bao 一致（2个数值特征）。
    """
    num_types = len(encoding.type2idx)
    node_type = plan_node['operatorType']
    type_id   = encoding.encode_type(node_type)

    # 算子类型 one-hot
    type_vec = np.zeros(num_types, dtype=np.float32)
    if 0 <= type_id < num_types:
        type_vec[type_id] = 1.0

    # EstimatedRows log 归一化（对应原版 cost_est 和 card_est 两个槽位）
    est_rows = float(plan_node['args'].get('EstimatedRows') or 0)
    est_log  = float(np.log(est_rows + 1.0))
    log_range = est_log_max - est_log_min
    if log_range > 0:
        est_norm = (est_log - est_log_min) / log_range
    else:
        est_norm = 0.0
    est_norm = float(np.clip(est_norm, 0.0, 1.0))

    # 两个数值特征槽位（cost_est_norm, card_est_norm）均用 est_norm
    num_feats = np.array([est_norm, est_norm], dtype=np.float32)

    return np.concatenate([type_vec, num_feats])


# ── 计划树统计（用于归一化 EstimatedRows）────────────────────────────────────

def _collect_all_estimated_rows(plan_node: dict) -> list:
    """递归收集计划树中所有节点的 EstimatedRows。"""
    rows = [float(plan_node['args'].get('EstimatedRows') or 0)]
    for child in plan_node.get('children') or []:
        rows.extend(_collect_all_estimated_rows(child))
    return rows


def _compute_est_log_min_max(plan_roots: list) -> tuple:
    """计算所有计划树中 EstimatedRows 的 log min/max（用于归一化）。"""
    all_rows = []
    for root in plan_roots:
        all_rows.extend(_collect_all_estimated_rows(root))
    all_logs = np.log(np.array(all_rows, dtype=np.float64) + 1.0)
    return float(all_logs.min()), float(all_logs.max())


# ── 计划树 → 二叉特征树 ───────────────────────────────────────────────────────

def _extract_cypher(plan_node: dict) -> str:
    """从 Plan 根节点的 args 中提取原始 Cypher 查询字符串。"""
    cypher = plan_node.get('args', {}).get('cypher', '')
    if cypher:
        return cypher
    for child in plan_node.get('children') or []:
        result = _extract_cypher(child)
        if result:
            return result
    return ''


def _plan_to_binary_feature_tree(plan_node: dict, encoding,
                                  est_log_min: float, est_log_max: float):
    """
    递归将 Cypher 计划树节点转换为 BinaryTreeConv 所需的二叉特征树。

    返回格式（与原版 Bao TreeBuilder.plan_to_feature_tree 一致）：
      - 叶子节点：np.ndarray（特征向量）
      - 内部节点：(feat_vec, left_tree, right_tree)

    多子节点二叉化策略（与原版 Bao 一致）：
      - 0 子节点：直接返回特征向量（叶子）
      - 1 子节点：透传，返回子节点的特征树
      - 2 子节点：标准二叉节点 (feat, left, right)
      - 3+ 子节点：链式二叉化，将第3+个子节点嵌套为右子树的右子树
    """
    feat = _encode_node_feature(plan_node, encoding, est_log_min, est_log_max)
    children = plan_node.get('children') or []

    if len(children) == 0:
        return feat

    if len(children) == 1:
        return _plan_to_binary_feature_tree(children[0], encoding, est_log_min, est_log_max)

    if len(children) == 2:
        left  = _plan_to_binary_feature_tree(children[0], encoding, est_log_min, est_log_max)
        right = _plan_to_binary_feature_tree(children[1], encoding, est_log_min, est_log_max)
        return (feat, left, right)

    # 3+ 子节点：链式二叉化
    # 将 children[2:] 递归嵌套为右子树，children[0] 作为左子树
    left = _plan_to_binary_feature_tree(children[0], encoding, est_log_min, est_log_max)
    # 构造一个虚拟内部节点，将 children[1:] 继续二叉化
    pseudo_node = dict(plan_node)
    pseudo_node = {
        'operatorType': plan_node['operatorType'],
        'args': plan_node['args'],
        'children': children[1:],
    }
    right = _plan_to_binary_feature_tree(pseudo_node, encoding, est_log_min, est_log_max)
    return (feat, left, right)


# ── 标签归一化（与原版 TATA Normalizer 完全一致）─────────────────────────────

class TataLabelNormalizer:
    """
    与原版 TATA cost_est.py 中的 Normalizer 完全一致。
    log(x + 0.001) + min-max 归一化，输出 [0.001, 1]。
    """

    def __init__(self):
        self.mini = None
        self.maxi = None

    def normalize_labels(self, labels, reset_min_max: bool = False):
        labels_log = np.array([np.log(float(l) + 0.001) for l in labels])
        if self.mini is None or reset_min_max:
            self.mini = float(labels_log.min())
        if self.maxi is None or reset_min_max:
            self.maxi = float(labels_log.max())
        labels_norm = (labels_log - self.mini) / (self.maxi - self.mini)
        labels_norm = np.minimum(labels_norm, 1.0)
        labels_norm = np.maximum(labels_norm, 0.001)
        return labels_norm

    def unnormalize_labels(self, labels_norm):
        labels_norm = np.array(labels_norm, dtype=np.float32)
        labels_log  = labels_norm * (self.maxi - self.mini) + self.mini
        return np.exp(labels_log) - 0.001


# ── Dataset ───────────────────────────────────────────────────────────────────

def _left_child(x):
    if isinstance(x, tuple) and len(x) == 3:
        return x[1]
    return None


def _right_child(x):
    if isinstance(x, tuple) and len(x) == 3:
        return x[2]
    return None


def _features(x):
    if isinstance(x, tuple):
        return x[0]
    return x


class TataCypherDataset(Dataset):
    """
    将 Cypher 查询计划 CSV 转换为 TATA/BaoNet 所需的二叉特征树格式。

    每个样本为 (binary_feature_tree, label_normalized)。
    binary_feature_tree 是递归嵌套的 tuple，可直接传入 BaoNet 的 TreeConvolution。

    Attributes:
        feature_trees:   list of binary feature tree（每个查询一棵）
        labels_norm:     np.ndarray，归一化后的标签（[0.001, 1]）
        labels_raw:      np.ndarray，原始 Execution Time（毫秒）
        cost_norm:       TataLabelNormalizer 实例
        node_feature_dim: 节点特征维度（动态）
        row_ids:         list，原始 DataFrame 的 id 列
    """

    def __init__(self, json_df, encoding: Encoding,
                 cost_norm: TataLabelNormalizer = None,
                 est_log_min: float = None, est_log_max: float = None,
                 reset_norm: bool = False,
                 label_field: str = 'cost'):
        """
        Args:
            json_df:      包含查询计划的 DataFrame，必须有 'json' 列，可选 'id' 列
            encoding:     Encoding 实例（词典已稳定）
            cost_norm:    TataLabelNormalizer 实例（None 时自动创建）
            est_log_min:  EstimatedRows log 归一化的 min（None 时从当前数据集计算）
            est_log_max:  EstimatedRows log 归一化的 max（None 时从当前数据集计算）
            reset_norm:   是否用当前数据集重新计算标签归一化参数（训练集为 True）
            label_field:  标签来源，'cost' = Execution Time（ms），'card' = 根节点 Rows（行数）
        """
        self.encoding     = encoding
        self.label_field  = label_field
        self.node_feature_dim = compute_node_feature_dim(encoding)
        self._json_df = json_df  # 保存原始 DataFrame，供 rebuild() 使用

        plan_jsons = [json.loads(s) for s in json_df['json']]
        plan_roots = [pj['Plan'] for pj in plan_jsons]
        self.row_ids = list(json_df['id']) if 'id' in json_df.columns \
                       else list(range(len(json_df)))

        # ── 标签提取（根据 label_field 切换）────────────────────────────────
        if label_field == 'card':
            # 根节点实际输出行数，clip 到 1 避免 log(0)
            labels_raw = [max(float(pj['Plan']['args'].get('Rows', 1)), 1.0)
                          for pj in plan_jsons]
        else:
            # 默认：Execution Time（毫秒）
            labels_raw = [pj['Execution Time'] for pj in plan_jsons]

        # 计算 EstimatedRows log min/max（用于节点特征归一化）
        if est_log_min is None or est_log_max is None:
            self.est_log_min, self.est_log_max = _compute_est_log_min_max(plan_roots)
        else:
            self.est_log_min = est_log_min
            self.est_log_max = est_log_max

        # 构建二叉特征树（同时扩充 encoding 词典）
        self.feature_trees = [
            _plan_to_binary_feature_tree(pr, encoding, self.est_log_min, self.est_log_max)
            for pr in plan_roots
        ]

        # 标签归一化
        if cost_norm is None:
            cost_norm = TataLabelNormalizer()
        self.cost_norm   = cost_norm
        self.labels_raw  = np.array(labels_raw, dtype=np.float64)
        self.labels_norm = cost_norm.normalize_labels(labels_raw,
                                                      reset_min_max=reset_norm)

    def rebuild(self, node_feature_dim: int,
                est_log_min: float, est_log_max: float):
        """
        用最终稳定的 encoding 词典重新构建特征树（skill 文档 7.7）。
        在 train/val/test 全部解析完后调用。标签已在 __init__ 里计算好，不重新计算。
        """
        self.node_feature_dim = node_feature_dim
        self.est_log_min      = est_log_min
        self.est_log_max      = est_log_max

        plan_jsons = [json.loads(s) for s in self._json_df['json']]
        plan_roots = [pj['Plan'] for pj in plan_jsons]
        self.feature_trees = [
            _plan_to_binary_feature_tree(pr, self.encoding, est_log_min, est_log_max)
            for pr in plan_roots
        ]

    def __len__(self) -> int:
        return len(self.feature_trees)

    def __getitem__(self, idx):
        return self.feature_trees[idx], float(self.labels_norm[idx])


# ── DataLoader collate ────────────────────────────────────────────────────────

def _prepare_trees(trees, transformer, left_child_fn, right_child_fn):
    """
    将一批二叉特征树展平并 pad，返回 (flat_trees, indexes)。
    直接复用 baseline/tata/algorithms/bao/TreeConvolution/util.py 的逻辑。
    """
    _TREECONV_DIR = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'algorithms', 'bao', 'TreeConvolution'
    )
    if _TREECONV_DIR not in sys.path:
        sys.path.insert(0, _TREECONV_DIR)

    from util import flatten, pad_and_combine, tree_conv_indexes

    flat_trees = [flatten(x, transformer, left_child_fn, right_child_fn) for x in trees]
    flat_trees = pad_and_combine(flat_trees)
    flat_trees = torch.Tensor(flat_trees).transpose(1, 2)  # [B, channels, max_nodes]

    indexes = [tree_conv_indexes(x, left_child_fn, right_child_fn) for x in trees]
    indexes = pad_and_combine(indexes)
    indexes = torch.Tensor(indexes).long()

    return flat_trees, indexes


class TataBatch:
    """与原版 Bao Batch 完全一致的批次容器。"""

    def __init__(self, trees, idxes):
        self.trees = trees
        self.idxes = idxes

    def to(self, device):
        self.trees = self.trees.to(device)
        self.idxes = self.idxes.to(device)
        return self


def tata_collate(batch):
    """DataLoader 的 collate_fn，将一批 (feature_tree, label) 转换为 (TataBatch, labels)。"""
    trees   = [item[0] for item in batch]
    targets = [item[1] for item in batch]

    flat_trees, indexes = _prepare_trees(trees, _features, _left_child, _right_child)
    targets = torch.FloatTensor(targets).reshape(-1, 1)

    return TataBatch(flat_trees, indexes), targets
