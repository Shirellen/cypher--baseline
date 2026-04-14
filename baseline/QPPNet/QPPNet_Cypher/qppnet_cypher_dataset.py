"""
QPPNet Cypher Baseline 的 Dataset。

QPPNet 的核心数据结构是递归嵌套的 samp_dict，每层对应计划树的一个节点：
  {
    'node_type':     str,          # 算子类型（如 'Filter', 'Expand(All)'）
    'subbatch_size': int,          # 本组查询数量
    'feat_vec':      np.ndarray,   # [subbatch_size, node_feat_dim]，节点特征
    'children_plan': list,         # 子节点的 samp_dict 列表（递归）
    'total_time':    np.ndarray,   # [subbatch_size]，监督信号（Execution Time / SCALE）
    'is_subplan':    bool,         # 是否是子计划（Cypher 中统一为 False）
  }

节点特征（对标 LEON 方案 A，skill 文档 7.8）：
  typeId one-hot(len(type2idx)) + Label one-hot(len(table2idx))
  + join one-hot(len(join2idx)) + joinId scalar(1)
  + 谓词(colId×3+opId×3+val×3=9) + mask(3) + EstimatedRows(1，log1p/20归一化)
  总维度 = len(type2idx) + len(table2idx) + len(join2idx) + 14（动态）

dim_dict：每种算子的输入维度 = 节点自身特征维度 + 子节点输出维度之和
  （子节点输出 output_size=32 维，与原版 QPPNet 一致）

监督信号：Execution Time / SCALE（SCALE=1000），与原版 QPPNet 一致。

分组策略：同一查询模板（相同计划树结构）的查询分为一组，组内一起前向传播。

_groups_raw 中每条 item 为 (tree_node_root, exec_time, row_id)，
其中 tree_node_root 是已解析好的 TreeNode 根节点（不再存储原始 plan_node dict），
这样 rebuild 时不会再调用 _parse_node 扩充 encoding 词典，保证 node_feat_dim 稳定。
row_id 来自 DataFrame 的 'id' 列，用于结果 CSV 的 id 映射。
"""

import json
import sys
import os

import numpy as np
import pandas as pd

_SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))                              # QPPNet_Cypher/
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_SCRIPT_DIR)))          # research/
for _p in [_PROJECT_ROOT, _SCRIPT_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from model.database_util import TreeNode, Encoding
from model.cypher_format_filter import cypher_format_filter
from model.cypher_format_join import cypher_format_join

# 子节点输出维度（与原版 QPPNet NeuralUnit output_size 一致）
CHILD_OUTPUT_SIZE = 32


def compute_node_feature_dim(encoding) -> int:
    """返回节点特征向量维度（encoding 词典稳定后调用）。"""
    return (len(encoding.type2idx)    # 算子类型 one-hot
            + len(encoding.table2idx) # Label one-hot
            + len(encoding.join2idx)  # 关系类型 one-hot
            + 14)                     # joinId scalar(1) + 谓词(9) + mask(3) + EstimatedRows(1)


def _encode_node_feature(node, encoding) -> np.ndarray:
    """
    将 TreeNode 编码为固定维度的特征向量（对标 LEON 方案 A，skill 文档 7.8）。

    特征结构：
      typeId one-hot(num_types) + Label one-hot(num_labels)
      + joinId one-hot(num_joins) + joinId scalar(1)
      + colId×3 + opId×3 + val×3(9) + mask(3) + EstimatedRows(1，log1p/20归一化)
    """
    num_types  = len(encoding.type2idx)
    num_labels = len(encoding.table2idx)
    num_joins  = len(encoding.join2idx)

    # 算子类型 one-hot
    type_vec = np.zeros(num_types, dtype=np.float32)
    if 0 <= node.typeId < num_types:
        type_vec[node.typeId] = 1.0

    # Label one-hot（multi-hot，encode_table 返回长度为 num_labels 的向量）
    label_vec = np.asarray(node.table_id, dtype=np.float32)
    if len(label_vec) != num_labels:
        label_vec = np.zeros(num_labels, dtype=np.float32)

    # 关系类型 one-hot
    join_vec = np.zeros(num_joins, dtype=np.float32)
    if 0 <= node.join < num_joins:
        join_vec[node.join] = 1.0

    # joinId scalar（归一化到 [0,1]）
    join_scalar = np.array([float(node.join) / max(num_joins - 1, 1)], dtype=np.float32)

    # 谓词特征（最多3组，每组 colId/opId/val）
    fd = node.filterDict
    num_filter = min(3, len(fd['colId']))
    cols = np.asarray(fd['colId'], dtype=np.float32)[:num_filter]
    ops  = np.asarray(fd['opId'],  dtype=np.float32)[:num_filter]
    vals = np.asarray(fd['val'],   dtype=np.float32)[:num_filter]
    if num_filter > 0:
        filts = np.stack([cols, ops, vals], axis=0).flatten()
    else:
        filts = np.zeros(0, dtype=np.float32)
    filt_pad = np.zeros(9 - len(filts), dtype=np.float32)
    filts    = np.concatenate([filts, filt_pad])

    # 谓词 mask（标记哪些谓词槽位有效）
    mask = np.zeros(3, dtype=np.float32)
    mask[:num_filter] = 1.0

    # EstimatedRows（log1p/20 归一化到 [0,1] 附近）
    est = np.array([float(np.log1p(node.EstimatedRows or 0)) / 20.0], dtype=np.float32)

    return np.concatenate([type_vec, label_vec, join_vec, join_scalar, filts, mask, est])


def _plan_structure_hash(plan_node: dict) -> str:
    """递归计算计划树的结构哈希（用于分组，与原版 QPPNet grouping 一致）。"""
    res = plan_node['operatorType']
    for child in plan_node.get('children') or []:
        res += _plan_structure_hash(child)
    return res


def _parse_node(plan_node: dict, idx: int, encoding, cypher: str) -> TreeNode:
    """递归解析 Plan JSON 为 TreeNode（扩充 encoding 词典）。"""
    node_type = plan_node['operatorType']
    type_id   = encoding.encode_type(node_type)
    filters, alias = cypher_format_filter(plan_node, cypher)
    join_str  = cypher_format_join(plan_node)
    join_id   = encoding.encode_join(join_str)
    filters_encoded = encoding.encode_filters(filters, alias)

    node = TreeNode(
        nodeType=node_type,
        typeId=type_id,
        filt=filters,
        card=None,
        join=join_id,
        join_str=join_str,
        filterDict=filters_encoded,
        details=plan_node['args'].get('Details', ''),
        EstimatedRows=plan_node['args']['EstimatedRows'],
    )
    node.table    = alias
    node.table_id = encoding.encode_table(alias)
    node.query_id = idx

    for child_plan in plan_node.get('children') or []:
        child_node = _parse_node(child_plan, idx, encoding, cypher)
        child_node.parent = node
        node.addChild(child_node)

    return node


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


def _build_samp_dict_from_tree(tree_nodes: list, execution_times: list,
                               encoding, node_feat_dim: int,
                               mean_range_dict: dict) -> dict:
    """
    将同一查询模板的一组已解析 TreeNode（同结构）转换为 QPPNet 风格的 samp_dict。

    关键：接受 TreeNode 列表而不是原始 plan_node dict，不再调用 _parse_node，
    因此不会扩充 encoding 词典，保证 node_feat_dim 在 rebuild 时保持稳定。

    Args:
        tree_nodes:      list of TreeNode（同结构，同层，已解析好）
        execution_times: 对应的 Execution Time（毫秒）列表
        encoding:        Encoding 实例（词典已稳定，只用于 _encode_node_feature）
        node_feat_dim:   节点特征维度（encoding 词典稳定后的最终维度）
        mean_range_dict: {node_type: (mean_vec, range_vec)}，用于归一化

    Returns:
        samp_dict（递归嵌套）
    """
    node_type     = tree_nodes[0].nodeType
    subbatch_size = len(tree_nodes)

    # 直接用已解析好的 TreeNode 编码特征，不再调用 _parse_node
    raw_feat_vecs = np.array(
        [_encode_node_feature(tn, encoding) for tn in tree_nodes],
        dtype=np.float32
    )  # [subbatch_size, current_feat_dim]

    # 对齐到 node_feat_dim
    current_dim = raw_feat_vecs.shape[1]
    if current_dim < node_feat_dim:
        pad = np.zeros((subbatch_size, node_feat_dim - current_dim), dtype=np.float32)
        feat_vecs = np.concatenate([raw_feat_vecs, pad], axis=1)
    elif current_dim > node_feat_dim:
        feat_vecs = raw_feat_vecs[:, :node_feat_dim]
    else:
        feat_vecs = raw_feat_vecs

    # 归一化（mean-range normalization，与原版 QPPNet 一致）
    if node_type in mean_range_dict:
        mean_vec, range_vec = mean_range_dict[node_type]
        feat_dim = feat_vecs.shape[1]
        if len(mean_vec) < feat_dim:
            mean_vec  = np.concatenate([mean_vec,  np.zeros(feat_dim - len(mean_vec),  dtype=np.float32)])
            range_vec = np.concatenate([range_vec, np.ones(feat_dim - len(range_vec), dtype=np.float32)])
        elif len(mean_vec) > feat_dim:
            mean_vec  = mean_vec[:feat_dim]
            range_vec = range_vec[:feat_dim]
        feat_vecs = (feat_vecs - mean_vec) / range_vec

    # 递归构建子节点 samp_dict（从 TreeNode.children 获取，不访问原始 plan_node）
    children_plan = []
    num_children  = len(tree_nodes[0].children)
    for child_idx in range(num_children):
        child_tree_nodes = [tn.children[child_idx] for tn in tree_nodes]
        child_samp_dict  = _build_samp_dict_from_tree(
            child_tree_nodes, execution_times,
            encoding, node_feat_dim, mean_range_dict
        )
        children_plan.append(child_samp_dict)

    return {
        'node_type':      node_type,
        'real_node_type': node_type,
        'subbatch_size':  subbatch_size,
        'feat_vec':       feat_vecs,
        'children_plan':  children_plan,
        # log1p 归一化：target 值域 [0, ~8]，避免 target≈0 导致 dying ReLU
        'total_time':     np.log1p(np.array(execution_times, dtype=np.float32)),
        'is_subplan':     False,
    }


class QPPNetCypherDataset:
    """
    将 Cypher 查询计划 CSV 转换为 QPPNet 所需的分组 samp_dict 格式。

    QPPNet 不使用 DataLoader，而是按查询模板（相同计划树结构）分组，
    每组同结构查询一起前向传播。

    Attributes:
        groups:          list of samp_dict，每个 samp_dict 对应一个查询模板组
        encoding:        Encoding 实例
        node_feat_dim:   节点特征维度（encoding 词典稳定后计算）
        mean_range_dict: {node_type: (mean_vec, range_vec)}，用于特征归一化
        _groups_raw:     {struct_hash: [(plan_root, cypher, exec_time, row_id), ...]}
                         保存 row_id 供结果 CSV 使用
    """

    def __init__(self, json_df: pd.DataFrame, encoding: Encoding,
                 mean_range_dict: dict = None, fit_normalizer: bool = False,
                 label_field: str = 'cost'):
        """
        Args:
            json_df:         包含查询计划的 DataFrame，必须有 'json' 列，可选 'id' 列
            encoding:        Encoding 实例
            mean_range_dict: 特征归一化字典（None 表示不归一化，或由 fit_normalizer 计算）
            fit_normalizer:  是否用当前数据集计算 mean_range_dict（训练集为 True）
            label_field:     标签来源，'cost' = Execution Time（ms），'card' = 根节点 Rows（行数）
        """
        self.label_field = label_field
        self.encoding = encoding

        plan_jsons     = [json.loads(s) for s in json_df['json']]
        plan_roots     = [pj['Plan'] for pj in plan_jsons]
        cypher_queries = [_extract_cypher(pj['Plan']) for pj in plan_jsons]
        row_ids        = list(json_df['id']) if 'id' in json_df.columns \
                         else list(range(len(json_df)))

        # ── 标签提取（根据 label_field 切换）────────────────────────────────
        if label_field == 'card':
            # 根节点实际输出行数，clip to 1 avoid log(0)
            labels_raw = [max(float(pj['Plan']['args'].get('Rows', 1)), 1.0)
                          for pj in plan_jsons]
        else:
            # 默认：Execution Time（毫秒）
            labels_raw = [pj['Execution Time'] for pj in plan_jsons]

        # 解析所有计划树为 TreeNode（扩充 encoding 词典），按结构分组
        # item = (tree_node_root, label_value, row_id)
        # 存储 TreeNode 而非原始 plan_node dict，确保 rebuild 时不再扩充词典
        structure_hashes = [_plan_structure_hash(pr) for pr in plan_roots]
        groups_raw: dict = {}
        for plan_root, cypher, label_val, row_id, struct_hash in zip(
            plan_roots, cypher_queries, labels_raw, row_ids, structure_hashes
        ):
            if struct_hash not in groups_raw:
                groups_raw[struct_hash] = []
            tree_root = _parse_node(plan_root, 0, encoding, cypher)
            groups_raw[struct_hash].append((tree_root, label_val, row_id))

        self._groups_raw = groups_raw

        # 计算节点特征维度（encoding 词典扩充后）
        self.node_feat_dim = compute_node_feature_dim(encoding)

        # 计算或复用 mean_range_dict
        if fit_normalizer:
            self.mean_range_dict = self._compute_mean_range(groups_raw, encoding)
        else:
            self.mean_range_dict = mean_range_dict or {}

        # 构建 samp_dict 列表
        self.groups = self._build_groups(groups_raw, encoding)

    def rebuild(self, node_feat_dim: int, mean_range_dict: dict):
        """
        用最终稳定的 encoding 词典和归一化参数重新构建 samp_dict。
        在 train/val/test 全部解析完后调用（skill 文档 7.7）。
        """
        self.node_feat_dim   = node_feat_dim
        self.mean_range_dict = mean_range_dict
        self.groups = self._build_groups(self._groups_raw, self.encoding)

    def _build_groups(self, groups_raw: dict, encoding) -> list:
        """将分组原始数据转换为 samp_dict 列表。"""
        result = []
        for struct_hash, items in groups_raw.items():
            tree_roots = [item[0] for item in items]   # TreeNode 根节点
            exec_times = [item[1] for item in items]   # Execution Time（毫秒）
            samp_dict  = _build_samp_dict_from_tree(
                tree_roots, exec_times,
                encoding, self.node_feat_dim, self.mean_range_dict
            )
            result.append(samp_dict)
        return result

    def _compute_mean_range(self, groups_raw: dict, encoding) -> dict:
        """
        计算每种算子类型的特征均值和范围（与原版 QPPNet normalize() 一致）。
        只统计节点自身特征（不含子节点输出）。

        必须两步走，避免 encoding 词典在收集过程中不断扩充导致特征维度不一致：
          第一步：遍历所有组，只调用 _parse_node 解析 TreeNode（扩充词典），存储节点列表
          第二步：词典稳定后，统一调用 _encode_node_feature 编码特征，此时维度一致
        """
        # 第一步：从 _groups_raw 里已存储的 TreeNode 收集所有节点
        # item 结构：(tree_node_root, exec_time, row_id)，TreeNode 已在 __init__ 里解析好
        # node_pool: node_type → list of TreeNode
        node_pool: dict = {}

        def collect_tree_nodes(tree_nodes: list):
            """递归收集同层 TreeNode 到 node_pool。"""
            node_type = tree_nodes[0].nodeType
            if node_type not in node_pool:
                node_pool[node_type] = []
            node_pool[node_type].extend(tree_nodes)

            num_children = len(tree_nodes[0].children)
            for child_idx in range(num_children):
                child_nodes = [tn.children[child_idx] for tn in tree_nodes]
                collect_tree_nodes(child_nodes)

        for struct_hash, items in groups_raw.items():
            tree_roots = [item[0] for item in items]   # TreeNode 根节点
            collect_tree_nodes(tree_roots)

        # 第二步：encoding 词典已稳定，统一编码特征（所有节点维度一致）
        mean_range_dict = {}
        for node_type, tree_nodes in node_pool.items():
            vecs = np.array(
                [_encode_node_feature(tn, encoding) for tn in tree_nodes],
                dtype=np.float32
            )  # [N, feat_dim]，此时 feat_dim 对所有节点一致
            mean_vec  = np.mean(vecs, axis=0)
            range_vec = np.max(vecs, axis=0)
            # 对于全为0的特征列（如从未出现的 join/table），max=0，
            # 若直接用作除数会导致 NaN/Inf，将其设为 1（不归一化该维度）
            range_vec = np.where(range_vec < np.finfo(np.float32).eps, 1.0, range_vec)
            mean_range_dict[node_type] = (mean_vec, range_vec)

        return mean_range_dict

    def get_all_ids_in_group_order(self) -> list:
        """
        按 _groups_raw 的遍历顺序返回所有 row_id 列表。
        用于结果 CSV 的 id 对齐（与 model.evaluate(ds.groups) 返回顺序一致）。
        item 结构：(tree_node_root, exec_time, row_id)
        """
        ids = []
        for items in self._groups_raw.values():
            for item in items:
                ids.append(item[2])  # row_id（item[0]=TreeNode, item[1]=exec_time, item[2]=row_id）
        return ids

    def sample_batch(self, batch_size: int) -> list:
        """
        随机采样 batch_size 个查询，按模板分组后返回 samp_dict 列表。
        与原版 QPPNet sample_data() 接口一致。
        """
        all_items = []
        for struct_hash, items in self._groups_raw.items():
            for item in items:
                all_items.append((struct_hash, item))

        total   = len(all_items)
        indices = np.random.choice(total, min(batch_size, total), replace=False)
        sampled = [all_items[i] for i in indices]

        # 按结构哈希重新分组
        batch_groups: dict = {}
        for struct_hash, item in sampled:
            if struct_hash not in batch_groups:
                batch_groups[struct_hash] = []
            batch_groups[struct_hash].append(item)

        result = []
        for struct_hash, items in batch_groups.items():
            tree_roots = [item[0] for item in items]   # TreeNode 根节点
            exec_times = [item[1] for item in items]   # Execution Time（毫秒）
            samp_dict  = _build_samp_dict_from_tree(
                tree_roots, exec_times,
                self.encoding, self.node_feat_dim, self.mean_range_dict
            )
            result.append(samp_dict)

        return result

    def __len__(self) -> int:
        return sum(len(items) for items in self._groups_raw.values())
