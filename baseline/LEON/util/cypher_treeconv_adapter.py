"""
Cypher 计划树到 TreeConvolution 输入格式的适配层。

LEON 原版 treeconv.py 有两个关键约束：
  1. _make_preorder_ids_tree 假设每个节点最多 2 个子节点（children[0] / children[1]）
  2. _featurize_tree 调用 node_featurizer.FeaturizeLeaf / Merge 接口

Cypher 计划树节点可能有多个子节点（如 Apply 节点有 2 个，CartesianProduct 也有 2 个，
但 Filter 后接单子节点等情况也存在）。本模块通过「二叉化」将多叉树转为二叉树，
再复用 LEON 的 make_and_featurize_trees 生成 (trees, indexes) 张量。
"""

import sys
import os
import numpy as np
import torch

# 把 baseline/LEON/ 加入 sys.path，使 LEON 内部的 `from util import ...` 能正确解析
_LEON_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _LEON_DIR not in sys.path:
    sys.path.insert(0, _LEON_DIR)

from util.cypher_featurizer import CypherNodeFeaturizer, CypherQueryFeaturizer


# ── 二叉化工具 ────────────────────────────────────────────────────────────────

def _binarize_tree(node):
    """
    将多叉 TreeNode 树原地二叉化（右链式）。

    对于子节点数 > 2 的节点，将第 2 个及之后的子节点链接为右子树的右子树，
    形成右偏二叉树。对于子节点数 == 0 或 1 的节点，补充一个虚拟叶子节点
    使其满足二叉树结构（LEON 的 _make_preorder_ids_tree 对叶子节点返回 (id, 0, 0)，
    对非叶子节点访问 children[0] 和 children[1]）。

    注意：此函数修改的是节点的 children 列表，不影响原始 plan JSON。
    """
    # 先递归处理所有子节点
    for child in node.children:
        _binarize_tree(child)

    num_children = len(node.children)

    if num_children == 0:
        # 叶子节点，无需处理
        return

    if num_children == 1:
        # 单子节点：补充一个虚拟右子节点（零特征叶子）
        node.children.append(_make_dummy_leaf(node))
        return

    if num_children == 2:
        # 已经是二叉，无需处理
        return

    # 多于 2 个子节点：右链式折叠
    # children = [c0, c1, c2, c3, ...]
    # 变为：left=c0, right=_wrap(c1, c2, c3, ...)
    left_child = node.children[0]
    remaining = node.children[1:]
    right_subtree = _fold_right_chain(remaining, node)
    node.children = [left_child, right_subtree]


def _fold_right_chain(children_list, parent_node):
    """
    将 children_list 折叠为右链式二叉树。
    [c1, c2, c3] → 虚拟节点(left=c1, right=虚拟节点(left=c2, right=c3))
    """
    if len(children_list) == 1:
        return children_list[0]

    # 创建一个继承父节点类型的虚拟中间节点
    wrapper = _make_wrapper_node(parent_node)
    wrapper.children = [children_list[0], _fold_right_chain(children_list[1:], parent_node)]
    return wrapper


def _make_dummy_leaf(parent_node):
    """创建一个零特征的虚拟叶子节点，用于补全二叉树结构。"""
    # 从父节点 feature 推断维度；若父节点 feature 尚未赋值则用默认值 64
    feature_dim = len(parent_node.feature) if parent_node.feature is not None else 64
    dummy = _DummyNode(
        nodeType='Unknown',
        typeId=0,
        join=0,
        join_str=None,
        EstimatedRows=0.0,
        query_id=getattr(parent_node, 'query_id', None),
        feature_dim=feature_dim,
    )
    return dummy

def _make_wrapper_node(parent_node):
    """创建一个继承父节点类型的虚拟中间节点，用于右链式折叠。"""
    feature_dim = len(parent_node.feature) if parent_node.feature is not None else 64
    wrapper = _DummyNode(
        nodeType=parent_node.nodeType,
        typeId=getattr(parent_node, 'typeId', 0),
        join=getattr(parent_node, 'join', 0),
        join_str=getattr(parent_node, 'join_str', None),
        EstimatedRows=getattr(parent_node, 'EstimatedRows', 0.0),
        query_id=getattr(parent_node, 'query_id', None),
        feature_dim=feature_dim,
    )
    return wrapper


class _DummyNode:
    """
    轻量级虚拟节点，用于二叉化时补全树结构。
    不依赖 model.database_util.TreeNode，避免跨包导入问题。
    """
    def __init__(self, nodeType, typeId, join, join_str, EstimatedRows, query_id, feature_dim):
        self.nodeType = nodeType
        self.typeId = typeId
        self.join = join
        self.join_str = join_str
        self.EstimatedRows = EstimatedRows
        self.card = None
        self.children = []
        self.table = []
        self.table_id = np.zeros(1, dtype=np.float32)
        self.query_id = query_id
        self.feature = np.zeros(feature_dim, dtype=np.float32)
        self.filterDict = {'colId': [0], 'opId': [3], 'val': [0.0]}
        self.details = ''
        self.filter = []


# ── 特征化接口 ────────────────────────────────────────────────────────────────

class _CypherFeaturizerAdapter:
    """
    将 CypherNodeFeaturizer 包装为 treeconv._featurize_tree 所需的接口。

    treeconv._featurize_tree 要求：
      - node_featurizer.FeaturizeLeaf(node) → np.ndarray
      - node_featurizer.Merge(node, left_vec, right_vec) → np.ndarray
    """

    def __init__(self, node_featurizer: CypherNodeFeaturizer):
        self.node_featurizer = node_featurizer

    def FeaturizeLeaf(self, node) -> np.ndarray:
        # 叶子节点：直接用节点自身特征
        if node.feature is not None:
            return node.feature.astype(np.float32)
        return self.node_featurizer(node)

    def Merge(self, node, left_vec: np.ndarray, right_vec: np.ndarray) -> np.ndarray:
        # 非叶子节点：使用节点自身特征（子节点信息通过树卷积传播，无需在此合并）
        if node.feature is not None:
            return node.feature.astype(np.float32)
        return self.node_featurizer(node)


# ── 核心入口：复用 LEON treeconv 的 make_and_featurize_trees ─────────────────

def cypher_tree_conv_featurize(tree_nodes: list, node_featurizer: CypherNodeFeaturizer):
    """
    将 TreeNode 树列表转换为 TreeConvolution 所需的 (trees, indexes) 张量。

    步骤：
      1. 对每棵树进行二叉化（原地修改 children，不影响原始数据）
      2. 确保每个节点的 .feature 已赋值（维度 = NODE_FEATURE_DIM）
      3. 调用 LEON 的 make_and_featurize_trees 生成张量

    Args:
        tree_nodes: TreeNode 根节点列表（每个元素对应一条查询的计划树）
        node_featurizer: CypherNodeFeaturizer 实例

    Returns:
        trees:   Tensor [batch, NODE_FEATURE_DIM, max_nodes]
        indexes: Tensor [batch, max_nodes*3, 1]（树卷积索引）
    """
    import copy

    # 深拷贝，避免二叉化修改原始树结构
    binarized_nodes = [copy.deepcopy(root) for root in tree_nodes]

    # 确保所有节点的 feature 已赋值，并二叉化
    for root in binarized_nodes:
        _ensure_features(root, node_featurizer)
        _binarize_tree(root)

    # 使用 LEON 的 treeconv 工具生成张量
    # baseline/LEON/ 已在 sys.path 中，直接绝对导入
    from util import treeconv
    adapter = _CypherFeaturizerAdapter(node_featurizer)
    trees, indexes = treeconv.make_and_featurize_trees(binarized_nodes, adapter)
    return trees, indexes


def _ensure_features(node, node_featurizer: CypherNodeFeaturizer):
    """递归确保树中每个节点的 .feature 已赋值为正确维度的向量。"""
    expected_dim = node_featurizer.feature_dim
    if node.feature is None or len(node.feature) != expected_dim:
        node.feature = node_featurizer(node)
    for child in node.children:
        _ensure_features(child, node_featurizer)


def cypher_query_featurize(tree_nodes: list, query_featurizer: CypherQueryFeaturizer) -> torch.Tensor:
    """
    将 TreeNode 根节点列表转换为查询级全局特征张量。

    Args:
        tree_nodes: TreeNode 根节点列表
        query_featurizer: CypherQueryFeaturizer 实例

    Returns:
        query_feats: Tensor [batch, QUERY_FEATURE_DIM]
    """
    vecs = [query_featurizer(root) for root in tree_nodes]
    return torch.from_numpy(np.stack(vecs, axis=0)).float()
