"""
Cypher 计划树的特征编码器（方案 A：对标 LEON 原版特征，去掉直方图）。

节点级特征（NODE_FEATURE_DIM，动态）：
  typeId(1) + joinId(1) + colId×3 + opId×3 + val×3 + mask(3) + table_id(N) + EstimatedRows(1)
  = 15 + len(encoding.table2idx)

查询级特征（QUERY_FEATURE_DIM，动态）：
  各算子出现次数(num_op_types) + 各join类型出现次数(num_join_types)
  + 谓词列出现次数(num_cols) + 树深度(1) + 节点总数(1) + 根EstimatedRows(1)
  = num_op_types + num_join_types + num_cols + 3

设计原则：
  - 与 LEON 原版特征对齐（typeId/joinId/谓词/table），不加直方图
  - 直方图是 QueryFormer 的增强，是本方法的特征优势，baseline 不应包含
  - NODE_FEATURE_DIM / QUERY_FEATURE_DIM 依赖 encoding，需在 Dataset 初始化后动态确定
"""

import numpy as np


def compute_node_feature_dim(encoding) -> int:
    """
    计算节点特征维度。
    = typeId(1) + joinId(1) + filts(9) + mask(3) + table_id(len(table2idx)) + EstimatedRows(1)
    """
    return 15 + len(encoding.table2idx)


def compute_query_feature_dim(encoding) -> int:
    """
    计算查询级特征维度。
    = num_op_types + num_join_types + num_cols + 3
    """
    num_op_types = len(encoding.type2idx) if encoding.type2idx else 40
    num_join_types = len(encoding.join2idx) if encoding.join2idx else 1
    num_cols = len(encoding.col2idx) if encoding.col2idx else 1
    return num_op_types + num_join_types + num_cols + 3


class CypherNodeFeaturizer:
    """
    方案 A 节点特征编码器，对标 LEON 原版 PhysicalTreeNodeFeaturizer。

    特征结构（去掉直方图）：
      [0]          typeId（算子类型整数编码）
      [1]          joinId（关系类型整数编码）
      [2:11]       谓词编码：colId×3 + opId×3 + val×3（最多3个谓词）
      [11:14]      谓词 mask（哪些谓词槽有效）
      [14:14+N]    table_id（Label multi-hot，N = len(encoding.table2idx)）
      [14+N]       EstimatedRows（原始值，不取 log，与 node2feature 保持一致）

    实现 FeaturizeLeaf / Merge 接口，供 treeconv.make_and_featurize_trees 调用。
    """

    def __init__(self, encoding):
        self.encoding = encoding
        self.feature_dim = compute_node_feature_dim(encoding)

    def _encode_node(self, tree_node) -> np.ndarray:
        # 1. typeId + joinId（2维）
        type_join = np.array([
            float(tree_node.typeId),
            float(tree_node.join),
        ], dtype=np.float32)

        # 2. 谓词编码（9维）：colId×3 + opId×3 + val×3，最多3个谓词
        filter_dict = tree_node.filterDict
        num_filter = min(3, len(filter_dict['colId']))
        cols = np.asarray(filter_dict['colId'], dtype=np.float32)[:num_filter]
        ops  = np.asarray(filter_dict['opId'],  dtype=np.float32)[:num_filter]
        vals = np.asarray(filter_dict['val'],   dtype=np.float32)[:num_filter]
        if num_filter > 0:
            filts_arr = np.stack([cols, ops, vals], axis=0)  # [3, num_filter]
        else:
            filts_arr = np.zeros((3, 0), dtype=np.float32)
        pad   = np.zeros((3, 3 - num_filter), dtype=np.float32)
        filts = np.concatenate((filts_arr, pad), axis=1).flatten()  # 9维

        # 3. 谓词 mask（3维）
        mask = np.zeros(3, dtype=np.float32)
        mask[:num_filter] = 1.0

        # 4. Label multi-hot（len(table2idx)维）
        table = np.asarray(tree_node.table_id, dtype=np.float32)

        # 5. EstimatedRows（1维，原始值）
        est_val = 0.0 if tree_node.EstimatedRows is None else float(tree_node.EstimatedRows)
        est = np.array([est_val], dtype=np.float32)

        return np.concatenate([type_join, filts, mask, table, est])

    # ── treeconv 接口 ────────────────────────────────────────────────────────

    def FeaturizeLeaf(self, tree_node) -> np.ndarray:
        if tree_node.feature is not None and len(tree_node.feature) == self.feature_dim:
            return tree_node.feature.astype(np.float32)
        return self._encode_node(tree_node)

    def Merge(self, tree_node, left_vec: np.ndarray, right_vec: np.ndarray) -> np.ndarray:
        if tree_node.feature is not None and len(tree_node.feature) == self.feature_dim:
            return tree_node.feature.astype(np.float32)
        return self._encode_node(tree_node)

    def __call__(self, tree_node) -> np.ndarray:
        return self._encode_node(tree_node)


class CypherQueryFeaturizer:
    """
    方案 A 查询级特征编码器，对标 LEON 原版 QueryFeaturizer。

    特征结构：
      [0 : num_op_types]                     各算子出现次数（log 归一化）
      [num_op_types : num_op_types+num_joins] 各关系类型出现次数（log 归一化）
      [... : ...+num_cols]                   各谓词列出现次数（log 归一化）
      [-3]                                   树深度归一化（/20）
      [-2]                                   节点总数归一化（/30）
      [-1]                                   根节点 EstimatedRows（原始值）
    """

    def __init__(self, encoding):
        self.encoding = encoding
        self.feature_dim = compute_query_feature_dim(encoding)
        self.num_op_types  = len(encoding.type2idx) if encoding.type2idx else 40
        self.num_join_types = len(encoding.join2idx) if encoding.join2idx else 1
        self.num_cols      = len(encoding.col2idx) if encoding.col2idx else 1

    def __call__(self, root_node) -> np.ndarray:
        feat = np.zeros(self.feature_dim, dtype=np.float32)
        all_nodes = self._collect_all_nodes(root_node)

        # 1. 各算子出现次数
        for node in all_nodes:
            op_idx = node.typeId
            if 0 <= op_idx < self.num_op_types:
                feat[op_idx] += 1.0
        feat[:self.num_op_types] = np.log1p(feat[:self.num_op_types])

        # 2. 各关系类型出现次数
        offset_join = self.num_op_types
        for node in all_nodes:
            join_idx = node.join
            if 0 <= join_idx < self.num_join_types:
                feat[offset_join + join_idx] += 1.0
        feat[offset_join:offset_join + self.num_join_types] = np.log1p(
            feat[offset_join:offset_join + self.num_join_types]
        )

        # 3. 各谓词列出现次数
        offset_col = offset_join + self.num_join_types
        for node in all_nodes:
            for col_id in node.filterDict.get('colId', []):
                if 0 <= col_id < self.num_cols:
                    feat[offset_col + col_id] += 1.0
        feat[offset_col:offset_col + self.num_cols] = np.log1p(
            feat[offset_col:offset_col + self.num_cols]
        )

        # 4. 树深度归一化
        feat[-3] = min(self._get_depth(root_node), 20) / 20.0

        # 5. 节点总数归一化
        feat[-2] = min(len(all_nodes), 30) / 30.0

        # 6. 根节点 EstimatedRows
        est = getattr(root_node, 'EstimatedRows', None) or 0
        feat[-1] = float(est)

        return feat

    def _collect_all_nodes(self, node):
        result = [node]
        for child in node.children:
            result.extend(self._collect_all_nodes(child))
        return result

    def _get_depth(self, node, depth=0):
        if not node.children:
            return depth
        return max(self._get_depth(child, depth + 1) for child in node.children)
