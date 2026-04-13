import torch
from torch.utils.data import Dataset
import numpy as np
import json
import pandas as pd
import sys, os
from collections import deque
from .database_util import TreeNode, filterDict2Hist
from .database_util import *
from .cypher_format_join import cypher_format_join
from .cypher_format_filter import cypher_format_filter
import networkx as nx
import numpy as np
from node2vec import Node2Vec
from typing import Dict, List, Set
from .details_parser import DetailsParser
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GATv2Conv, global_mean_pool
from torch_geometric.data import Data, Batch

class SchemaGAT(nn.Module):
    """Schema图的GAT编码器"""
    def __init__(self, node_feature_dim=64, hidden_dim=128, output_dim=256, num_layers=2, use_gatv2=True):
        super().__init__()
        # GAT层 - 移除dropout参数，在forward中手动处理
        self.gat_layers = nn.ModuleList()
        conv_cls = GATv2Conv if use_gatv2 else GATConv
        # 第一层
        self.gat_layers.append(conv_cls(node_feature_dim, hidden_dim, heads=4, concat=True))
        # 中间层
        for _ in range(num_layers - 2):
            self.gat_layers.append(conv_cls(hidden_dim * 4, hidden_dim, heads=4, concat=True))
        # 最后一层
        self.gat_layers.append(conv_cls(hidden_dim * 4, hidden_dim, heads=1, concat=False))
        # 图级别投影层
        self.graph_proj = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.1), nn.Linear(hidden_dim, output_dim))
    
    def forward(self, x, edge_index, batch):
        # 统一在 CPU 上执行，避免设备不一致
        device = torch.device('cpu')
        x = torch.nan_to_num(x.to(device).float(), nan=0.0, posinf=1.0, neginf=-1.0)
        edge_index = edge_index.to(device=device, dtype=torch.long).contiguous().clone()
        batch = batch.to(device=device, dtype=torch.long)

        # print(x.shape, edge_index.shape, batch.shape, x.dtype, edge_index.dtype)

        # 无边图加自环
        if edge_index.numel() == 0:
            num_nodes = x.size(0)
            loop = torch.arange(num_nodes, device=device, dtype=torch.long)
            edge_index = torch.stack([loop, loop], dim=0)

        # 越界裁剪；若裁剪为空，再加自环
        if edge_index.numel() > 0:
            num_nodes = x.size(0)
            valid_mask = (edge_index[0] >= 0) & (edge_index[0] < num_nodes) & \
                         (edge_index[1] >= 0) & (edge_index[1] < num_nodes)
            if not torch.all(valid_mask):
                edge_index = edge_index[:, valid_mask]
                if edge_index.numel() == 0:
                    loop = torch.arange(num_nodes, device=device, dtype=torch.long)
                    edge_index = torch.stack([loop, loop], dim=0)

        # GATv2 前向
        for i, layer in enumerate(self.gat_layers):
            x = layer(x, edge_index)
            if i < len(self.gat_layers) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.1, training=self.training)
        graph_emb = global_mean_pool(x, batch)
        graph_emb = self.graph_proj(graph_emb)
        return graph_emb

class SubschemaGraphBuilder:
    def __init__(self, embedding_dim=256):
        self.embedding_dim = embedding_dim
        self.parser = DetailsParser()
        # 初始化GAT模型
        self.gat_model = SchemaGAT(
            node_feature_dim=64,
            hidden_dim=128,
            output_dim=self.embedding_dim,
            num_layers=2
        )
        self.gat_model.eval()
        # 缓存
        self._graph_cache = {}


    def build_all(self, node:TreeNode):
        """从根节点开始递归构建整个计划树的subschema图"""
        if (len(node.children) == 0):
            # 叶子节点，直接构建图
            return self.build_subschema_graph(node)
        else:
            # 非叶子节点，先递归构建子节点的图
            for child in node.children:
                self.build_all(child)
            # 然后构建当前节点的图
            return self.build_subschema_graph(node)
        
        
    def build_subschema_graph(self, node: TreeNode):
        """构建当前节点的subschema图"""
        # 1. 从子节点继承图结构
        subschema = nx.MultiDiGraph()
        for child in node.children:
            if hasattr(child, 'subschema_graph'):
                subschema = nx.compose_all([subschema, child.subschema_graph])
        # 2. 解析当前节点的Details
        parsed = self.parser.parse_details(node.nodeType, node.details) # TODO combine
        # print(node.nodeType)
        # print(node.details)
        # print(parsed)
        # print("----")
        # 3. 添加新的节点和边到图中
        self._add_parsed_elements_to_graph(subschema, parsed)
    
        # 4. 存储图并生成embedding
        node.subschema_graph = subschema
        # node.subschema_embedding = self._encode_graph_to_embedding(subschema)
        node.subschema_embedding = self._encode_graph_to_gnn_embedding(subschema)
        node.feature = np.concatenate([node.feature, node.subschema_embedding])
        return subschema
    
    def _add_parsed_elements_to_graph(self, graph: nx.MultiDiGraph, parsed: Dict):
        """将解析的元素添加到图中"""
        if not parsed or all(not parsed.get(key) for key in ['nodes', 'relationships', 'graph_structure']):
            return
        # 添加节点标签作为节点
        for (var,node_label) in parsed['nodes']:
            if not graph.has_node(var):
                graph.add_node(var, 
                             type='node', 
                             label=node_label,
                             properties=set())
            else:
                # 如果节点已存在，更新标签（如果不同）
                # 如果标签不存在，则添加
                if not graph.nodes[var].get('label'):
                    graph.nodes[var]['label'] = node_label
                elif graph.nodes[var]['label'] != node_label:
                    graph.nodes[var]['label'] += f";{node_label}"
        # 添加关系类型作为节点
        for (var,rel_type) in parsed['relationships']:
            if not graph.has_node(var):
                graph.add_node(var, 
                             type='relationship', 
                             label=rel_type,
                             properties=set())
        # 添加图结构（节点-关系-节点的连接）
        for structure in parsed['graph_structure']:
            if len(structure) == 3:
                source_var, rel_var, target_var = structure
                if source_var and target_var and rel_var:
                    # 添加 source -> relationship -> target 的路径
                    graph.add_edge(source_var, rel_var, edge_type='out')
                    graph.add_edge(rel_var, target_var, edge_type='in')
        # 添加属性信息
        for prop in parsed['properties']:
            if '.' in prop:
                var, prop_name = prop.split('.', 1)
                if graph.has_node(var):
                    graph.nodes[var].get('properties', set()).add(prop_name)
                if graph.has_node(var):
                    graph.nodes[var].get('properties', set()).add(prop_name)
    
    def _encode_graph_to_gnn_embedding(self, graph: nx.MultiDiGraph) -> np.ndarray:
        """使用GAT将图编码为嵌入向量"""
        # 空图直接返回零向量，避免前向异常
        if graph.number_of_nodes() == 0:
            return np.zeros(self.embedding_dim, dtype=np.float32)
        # 检查缓存
        # graph_hash = self._compute_graph_hash(graph)
        # if graph_hash in self._graph_cache:
        #     return self._graph_cache[graph_hash]
        # === 图转换部分 ===
        # 1. 创建节点映射
        node_mapping = {node: i for i, node in enumerate(graph.nodes())}
        num_nodes = len(node_mapping)
        # 2. 提取节点特征
        node_features = []
        for node in graph.nodes():
            node_data = graph.nodes[node]
            features = self._extract_node_features(node_data)
            node_features.append(features)
        x = torch.FloatTensor(node_features)
        x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        # 3. 构建边列表
        edge_index = []
        for u, v, edge_data in graph.edges(data=True):
            u_idx = node_mapping[u]
            v_idx = node_mapping[v]
            edge_index.append([u_idx, v_idx])
        if edge_index:
            edge_index = torch.LongTensor(edge_index).t()
        else:
            # 无边时使用自环，避免 GAT 静态图错误
            loop_index = torch.arange(num_nodes, dtype=torch.long)
            edge_index = torch.stack([loop_index, loop_index], dim=0)
        # 4. 创建Data对象并使用Batch处理（单图 -> batch）
        data = Data(x=x, edge_index=edge_index)
        batch = Batch.from_data_list([data])
        # === GAT编码部分 ===
        with torch.no_grad():
            # GAT前向传播
            graph_embedding = self.gat_model(batch.x, batch.edge_index, batch.batch)
            embedding = graph_embedding.squeeze().cpu().numpy().astype(np.float32)
        # 缓存结果
        # self._graph_cache[graph_hash] = embedding
        # if len(self._graph_cache) > 1000:  # 限制缓存大小
        #     self._graph_cache.clear()
        return embedding

    def _extract_node_features(self, node_data: dict) -> list:
        """提取节点特征 (64维)"""
        features = []
        # 1. 节点类型 (3维: node, relationship, unknown)
        node_type = node_data.get('type', 'unknown')
        type_features = [0.0, 0.0, 0.0]
        if node_type == 'node':
            type_features[0] = 1.0
        elif node_type == 'relationship':
            type_features[1] = 1.0
        else:
            type_features[2] = 1.0
        features.extend(type_features)
        # 2. 标签特征 (10维: hash编码)
        label = node_data.get('label', '')
        label_features = [0.0] * 10
        if label:
            # 处理多标签情况 (如 "Person;Actor")
            labels = label.split(';')
            for lbl in labels:
                label_hash = hash(lbl.strip()) % 10
                label_features[label_hash] = 1.0
        features.extend(label_features)
        # 3. 属性数量 (1维)
        properties = node_data.get('properties', set())
        prop_count = min(len(properties) / 10.0, 1.0)  # 归一化到[0,1]
        features.append(prop_count)
        # 4. 属性特征 (50维: hash编码)
        prop_features = [0.0] * 50
        for prop in properties:
            prop_hash = hash(prop) % 50
            prop_features[prop_hash] += 1.0
        # 属性特征归一化
        total_props = sum(prop_features)
        if total_props > 0:
            prop_features = [f / total_props for f in prop_features]
        features.extend(prop_features)
        # 总维度: 3 + 10 + 1 + 50 = 64
        return features

    def _compute_graph_hash(self, nx_graph: nx.MultiDiGraph) -> int:
        """计算图的哈希值用于缓存"""
        # 节点信息
        nodes_info = []
        for node in sorted(nx_graph.nodes()):
            data = nx_graph.nodes[node]
            node_str = f"{data.get('type', '')}-{data.get('label', '')}-{len(data.get('properties', set()))}"
            nodes_info.append(node_str)
        # 边信息
        edges_info = []
        for u, v, data in nx_graph.edges(data=True):
            edge_str = f"{u}-{v}-{data.get('edge_type', '')}"
            edges_info.append(edge_str)
        graph_str = "|".join(sorted(nodes_info)) + "||" + "|".join(sorted(edges_info))
        return hash(graph_str)
    
class PlanTreeDataset(Dataset):
    def __init__(self, json_df : pd.DataFrame, train : pd.DataFrame, encoding, hist_file, card_norm, cost_norm, to_predict, table_sample):
        self.table_sample = table_sample
        self.encoding = encoding
        self.hist_file = hist_file
        # 使用新的图构建器
        self.subschema_builder = SubschemaGraphBuilder(embedding_dim=256)
        
        self.length = len(json_df)
        # train = train.loc[json_df['id']]
        self.dataset = list(json_df['dataset']) * self.length
        self.src_file = list(json_df['src_file']) * self.length
        
        nodes = [json.loads(plan)['Plan'] for plan in json_df['json']]
        self.cards = [node['rows'] for node in nodes]
        self.costs = [json.loads(plan)['Execution Time'] for plan in json_df['json']]
        
        # 检查数据质量
        # print(f"Costs stats - min: {min(self.costs)}, max: {max(self.costs)}")

        # self.card_labels = torch.from_numpy(card_norm.normalize_labels(self.cards))
        # self.cost_labels = torch.from_numpy(cost_norm.normalize_labels(self.costs))
        self.cost_norm = cost_norm
        self.card_norm = card_norm

        # 统一初始化两类标签，避免 __getitem__ 取不到
        should_reset_cost = (self.cost_norm.mini is None or self.cost_norm.maxi is None)
        self.cost_labels = torch.from_numpy(
            self.cost_norm.normalize_labels(self.costs, reset_min_max=should_reset_cost)
        ).float()

        should_reset_card = (self.card_norm.mini is None or self.card_norm.maxi is None)
        self.card_labels = torch.from_numpy(
            self.card_norm.normalize_labels(self.cards, reset_min_max=should_reset_card)
        ).float()
        
        self.to_predict = to_predict
        if to_predict == 'cost':
            self.gts = self.costs
            self.labels = self.cost_labels
            # should_reset = (self.cost_norm.mini is None or self.cost_norm.maxi is None)
            # self.cost_labels = torch.from_numpy(self.cost_norm.normalize_labels(self.costs, reset_min_max=should_reset)).float()
        elif to_predict == 'card':
            self.gts = self.cards
            self.labels = self.card_labels
            # should_reset = (self.card_norm.mini is None or self.card_norm.maxi is None)
            # self.card_labels = torch.from_numpy(self.card_norm.normalize_labels(self.cards, reset_min_max=should_reset)).float()
        elif to_predict == 'both': ## try not to use, just in case
            self.gts = self.costs
            self.labels = self.cost_labels
        else:
            raise Exception('Unknown to_predict type')
            
        idxs = list(json_df['id'])
        
    
        self.treeNodes = [] ## for mem collection
        self.collated_dicts = [self.js_node2dict(i,node,node['cypher']) for i,node in zip(idxs, nodes)]

    def js_node2dict(self, idx, node, cypher):
        # print(cypher)
        treeNode = self.traversePlan(node, idx, self.encoding, cypher)
        self.subschema_builder.build_all(treeNode)
        _dict = self.node2dict(treeNode)
        collated_dict = self.pre_collate(_dict)
        
        self.treeNodes.clear()
        del self.treeNodes[:]

        return collated_dict

    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        
        return self.collated_dicts[idx], (self.cost_labels[idx], self.card_labels[idx])

    def old_getitem(self, idx):
        return self.dicts[idx], (self.cost_labels[idx], self.card_labels[idx])
      
    ## pre-process first half of old collator
    # 注意这里要横距数据集调整最大节点数
    def pre_collate(self, the_dict, max_node = 34, rel_pos_max = 20):
    # def pre_collate(self, the_dict, max_node = 33, rel_pos_max = 20):

        x = pad_2d_unsqueeze(the_dict['features'], max_node)
        N = len(the_dict['features'])
        attn_bias = torch.zeros([N+1,N+1], dtype=torch.float)
        
        edge_index = the_dict['adjacency_list'].t()
        if len(edge_index) == 0:
            shortest_path_result = np.array([[0]])
            path = np.array([[0]])
            adj = torch.tensor([[0]]).bool()
        else:
            adj = torch.zeros([N,N], dtype=torch.bool)
            adj[edge_index[0,:], edge_index[1,:]] = True
            
            shortest_path_result = floyd_warshall_rewrite(adj.numpy())
        
        rel_pos = torch.from_numpy((shortest_path_result)).long()

        
        attn_bias[1:, 1:][rel_pos >= rel_pos_max] = float('-inf')
        
        attn_bias = pad_attn_bias_unsqueeze(attn_bias, max_node + 1)
        rel_pos = pad_rel_pos_unsqueeze(rel_pos, max_node)

        heights = pad_1d_unsqueeze(the_dict['heights'], max_node)
        
        return {
            'x' : x,
            'attn_bias': attn_bias,
            'rel_pos': rel_pos,
            'heights': heights
        }


    def node2dict(self, treeNode):

        adj_list, num_child, features = self.topo_sort(treeNode)
        heights = self.calculate_height(adj_list, len(features))

        return {
            'features' : torch.FloatTensor(np.array(features)),
            'heights' : torch.LongTensor(heights),
            'adjacency_list' : torch.LongTensor(np.array(adj_list)),
          
        }
    
    def topo_sort(self, root_node):
        #nodes = []
        adj_list = [] #from parent to children
        num_child = []
        features = []

        toVisit = deque()
        toVisit.append((0,root_node))
        next_id = 1
        while toVisit:
            idx, node = toVisit.popleft()
            #            nodes.append(node)
            features.append(node.feature)
            num_child.append(len(node.children))
            for child in node.children:
                toVisit.append((next_id,child))
                adj_list.append((idx,next_id))
                next_id += 1
        
        return adj_list, num_child, features
    
    def traversePlan(self, plan, idx, encoding, cypher): # bfs accumulate plan
        
        # 目前提取的特征：标签（可以代替 table） 关系（即expand操作符可以代替 join） 操作符 谓词 直方图
        # 后续增强版可考虑加入的特征：
            # 关系类型 [CONTAINER_OF, HAS_CREATOR]、
            # 关系方向 (outgoing/incoming/both)、
            # Expand 操作的详细信息（ (f)-[anon_0:CONTAINER_OF]->(p) ）、
            # 节点 label 的统计信息（度分布、属性值范围）
            # 每个关系的 domain / range 类型
        nodeType = plan['operatorType']
        typeId = encoding.encode_type(nodeType)
        card = None #plan['Actual Rows']
        #filter条件 `f.creationDate > $autoint_0`, `p.length < $autoint_1`
        #filter：当前 plan 节点上的过滤条件 alias：当前 plan 节点对应的表别名（如 SQL 里的 
        filters, alias = cypher_format_filter(plan, cypher)
        # print(filters, alias) 
        #连接条件 当前 plan 节点上的连接条件 **没有传统 JOIN**，而是: `(f)-[anon_0:CONTAINER_OF]->(p)`
        join = cypher_format_join(plan) 
        joinId = encoding.encode_join(join)
        filters_encoded = encoding.encode_filters(filters, alias)
        
        # print(idx)
        # print(nodeType)
        # print(join)
        # print(plan['args'].get('Details', ''))
        root = TreeNode(nodeType, typeId, filters, card, joinId, join, filters_encoded, details=plan['args'].get('Details', ''),EstimatedRows=plan['args']['EstimatedRows'])
        
        self.treeNodes.append(root)

        root.table = alias
        root.table_id = encoding.encode_table(root.table)
        root.query_id = idx
        
        #    print(root)
        # Neo4j 计划末端节点可能没有 'children' 键；有时也使用 'plans'
        children = plan.get('children') or []
        if children:
            for subplan in children:
                subplan['parent'] = plan
                node = self.traversePlan(subplan, idx, encoding, cypher)
                node.parent = root
                root.addChild(node)
        # # 生成最终特征
        # root.feature = node2feature(root, encoding, self.hist_file, self.table_sample)
        root.feature = node2feature(root, encoding, self.hist_file)
        return root

    def calculate_height(self, adj_list,tree_size):
        if tree_size == 1:
            return np.array([0])

        adj_list = np.array(adj_list)
        node_ids = np.arange(tree_size, dtype=int)
        node_order = np.zeros(tree_size, dtype=int)
        uneval_nodes = np.ones(tree_size, dtype=bool)

        parent_nodes = adj_list[:,0]
        child_nodes = adj_list[:,1]

        n = 0
        while uneval_nodes.any():
            uneval_mask = uneval_nodes[child_nodes]
            unready_parents = parent_nodes[uneval_mask]

            node2eval = uneval_nodes & ~np.isin(node_ids, unready_parents)
            node_order[node2eval] = n
            uneval_nodes[node2eval] = False
            n += 1
        return node_order 

def node2feature(node, encoding, hist_file):
    """
    统一的特征提取函数
    特征结构: [node_type, join/relationship, filters..., mask, hist, table, sample]
    """
    # type, join, filter123, mask123
    # 1, 1, 3x3 (9), 3
    # TODO: add sample (or so-called table)
    # 限制最多3个过滤条件，超出部分截断，不足按原逻辑右侧补零
    num_filter = min(3, len(node.filterDict['colId']))
    cols = np.asarray(node.filterDict['colId'], dtype=np.float32)[:num_filter]
    ops = np.asarray(node.filterDict['opId'], dtype=np.float32)[:num_filter]
    vals = np.asarray(node.filterDict['val'], dtype=np.float32)[:num_filter]
    filts_arr = np.stack([cols, ops, vals], axis=0) if num_filter > 0 else np.zeros((3, 0), dtype=np.float32)
    pad = np.zeros((3, 3 - num_filter), dtype=np.float32)
    filts = np.concatenate((filts_arr, pad), axis=1).flatten()
    mask = np.zeros(3, dtype=np.float32)
    mask[:num_filter] = 1
    # 基本特征：节点类型 + join/relationship
    type_join = np.array([node.typeId, node.join])
    # 直方图特征
    hists = filterDict2Hist(hist_file, node.filterDict, encoding)

    # 表/标签特征 + 样本特征
    # table, bitmap, 1 + 1000 bits
    # print(node.table_id)
    table = node.table_id
    est_val = 0.0 if node.EstimatedRows is None else float(node.EstimatedRows)
    est_arr = np.array([est_val], dtype=np.float32)
    
    return np.concatenate((type_join, filts, mask, hists, table, est_arr)).astype(np.float32)

