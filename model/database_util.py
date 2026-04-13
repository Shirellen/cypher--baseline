import numpy as np
import pandas as pd
import csv
import torch

## bfs shld be enough
def floyd_warshall_rewrite(adjacency_matrix):
    (nrows, ncols) = adjacency_matrix.shape
    assert nrows == ncols
    M = adjacency_matrix.copy().astype('long')
    for i in range(nrows):
        for j in range(ncols):
            if i == j: 
                M[i][j] = 0
            elif M[i][j] == 0: 
                M[i][j] = 60
    
    for k in range(nrows):
        for i in range(nrows):
            for j in range(nrows):
                M[i][j] = min(M[i][j], M[i][k]+M[k][j])
    return M

def get_job_table_sample(workload_file_name, num_materialized_samples = 1000):

    tables = []
    samples = []

    # Load queries
    with open(workload_file_name + ".csv", 'r') as f:
        data_raw = list(list(rec) for rec in csv.reader(f, delimiter='#'))
        for row in data_raw:
            tables.append(row[0].split(','))

            if int(row[3]) < 1:
                print("Queries must have non-zero cardinalities")
                exit(1)

    print("Loaded queries with len ", len(tables))
    
    # Load bitmaps
    num_bytes_per_bitmap = int((num_materialized_samples + 7) >> 3)
    with open(workload_file_name + ".bitmaps", 'rb') as f:
        for i in range(len(tables)):
            four_bytes = f.read(4)
            if not four_bytes:
                print("Error while reading 'four_bytes'")
                exit(1)
            num_bitmaps_curr_query = int.from_bytes(four_bytes, byteorder='little')
            bitmaps = np.empty((num_bitmaps_curr_query, num_bytes_per_bitmap * 8), dtype=np.uint8)
            for j in range(num_bitmaps_curr_query):
                # Read bitmap
                bitmap_bytes = f.read(num_bytes_per_bitmap)
                if not bitmap_bytes:
                    print("Error while reading 'bitmap_bytes'")
                    exit(1)
                bitmaps[j] = np.unpackbits(np.frombuffer(bitmap_bytes, dtype=np.uint8))
            samples.append(bitmaps)
    print("Loaded bitmaps")
    table_sample = []
    for ts, ss in zip(tables,samples):
        d = {}
        for t, s in zip(ts,ss):
            tf = t.split(' ')[0] # remove alias
            d[tf] = s
        table_sample.append(d)
    
    return table_sample


def get_hist_file(hist_path, bin_number = 50):
    hist_file = pd.read_csv(hist_path)
    for i in range(len(hist_file)):
        freq = hist_file['freq'][i]
        freq_np = np.frombuffer(bytes.fromhex(freq), dtype=float)
        hist_file['freq'][i] = freq_np

    # 若 CSV 已包含 "table_column"（例如 Cypher 直方图生成脚本写入的 Label.Prop），直接使用；
    # 否则用原始表名与列名拼接为 Label.Prop，不再做别名压缩，避免冲突（如 C.id）。
    # if 'table_column' not in hist_file.columns:
    #     table_column = []
    #     for i in range(len(hist_file)):
    #         table = hist_file['table'][i]
    #         col = hist_file['column'][i]
    #         combine = '.'.join([str(table), str(col)])
    #         table_column.append(combine)
    #     hist_file['table_column'] = table_column

    for rid in range(len(hist_file)):
        hist_file['bins'][rid] = \
            [int(i) for i in hist_file['bins'][rid][1:-1].split(' ') if len(i)>0]

    if bin_number != 50:
        hist_file = re_bin(hist_file, bin_number)

    return hist_file

def re_bin(hist_file, target_number):
    for i in range(len(hist_file)):
        freq = hist_file['freq'][i]
        bins = freq2bin(freq,target_number)
        hist_file['bins'][i] = bins
    return hist_file

def freq2bin(freqs, target_number):
    freq = freqs.copy()
    maxi = len(freq)-1
    
    step = 1. / target_number
    mini = 0
    while freq[mini+1]==0:
        mini+=1
    pointer = mini+1
    cur_sum = 0
    res_pos = [mini]
    residue = 0
    while pointer < maxi+1:
        cur_sum += freq[pointer]
        freq[pointer] = 0
        if cur_sum >= step:
            cur_sum -= step
            res_pos.append(pointer)
        else:
            pointer += 1
    
    if len(res_pos)==target_number: res_pos.append(maxi)
    
    return res_pos



class Batch():
    def __init__(self, attn_bias, rel_pos, heights, x, y=None):
        super(Batch, self).__init__()

        self.heights = heights
        self.x, self.y = x, y
        self.attn_bias = attn_bias
        self.rel_pos = rel_pos
        
    def to(self, device):

        self.heights = self.heights.to(device)
        self.x = self.x.to(device)

        self.attn_bias, self.rel_pos = self.attn_bias.to(device), self.rel_pos.to(device)

        return self

    def __len__(self):
        return self.in_degree.size(0)


def pad_1d_unsqueeze(x, padlen):
    x = x + 1 # pad id = 0
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen], dtype=x.dtype)
        new_x[:xlen] = x
        x = new_x
    return x.unsqueeze(0)


def pad_2d_unsqueeze(x, padlen):
    # dont know why add 1, comment out first
#    x = x + 1 # pad id = 0
    xlen, xdim = x.size()
    if xlen < padlen:
        new_x = x.new_zeros([padlen, xdim], dtype=x.dtype) + 1
        new_x[:xlen, :] = x
        x = new_x
    return x.unsqueeze(0)

def pad_rel_pos_unsqueeze(x, padlen):
    x = x + 1
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen], dtype=x.dtype)
        new_x[:xlen, :xlen] = x
        x = new_x
    return x.unsqueeze(0)

def pad_attn_bias_unsqueeze(x, padlen):
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen], dtype=x.dtype).fill_(float('-inf'))
        new_x[:xlen, :xlen] = x
        new_x[xlen:, :xlen] = 0
        x = new_x
    return x.unsqueeze(0)


def collator(small_set):
    y = small_set[1]
    xs = [s['x'] for s in small_set[0]]
    # 添加调试信息
    # print(f"=== Collator Debug Info ===")
    # print(f"Number of samples: {len(xs)}")
    # for i, x in enumerate(xs):
    #     print(f"Sample {i}: shape = {x.shape}")
    #     if i >= 5:  # 只打印前5个样本
    #         break
    
    # # 检查所有张量的形状
    # shapes = [x.shape for x in xs]
    # unique_shapes = list(set(shapes))
    # print(f"Unique shapes: {unique_shapes}")
    
    # if len(unique_shapes) > 1:
    #     print("=== Shape Mismatch Details ===")
    #     for i, (x, shape) in enumerate(zip(xs, shapes)):
    #         if i == 117 or shape != shapes[0]:  # 特别关注第117个或形状不匹配的
    #             print(f"Tensor {i}: shape = {shape}")
    num_graph = len(y)
    x = torch.cat(xs)
    attn_bias = torch.cat([s['attn_bias'] for s in small_set[0]])
    rel_pos = torch.cat([s['rel_pos'] for s in small_set[0]])
    heights = torch.cat([s['heights'] for s in small_set[0]])
    
    return Batch(attn_bias, rel_pos, heights, x), y

def filterDict2Hist(hist_file, filterDict, encoding):
    buckets = len(hist_file['bins'][0]) 
    empty = np.zeros(buckets - 1)
    ress = np.zeros((3, buckets-1))
    col_ids = list(filterDict.get('colId', []))
    op_ids = list(filterDict.get('opId', []))
    vals = list(filterDict.get('val', []))
    k = min(3, len(col_ids))
    for i in range(k):
        colId = col_ids[i]
        try:
            col = encoding.idx2col[colId]
        except Exception:
            col = 'NA'
        if col == 'NA':
            ress[i] = empty
            continue
        # 若该列未出现在直方图（非数值列或未统计），则忽略
        if (col not in encoding.column_min_max_vals) or (hist_file['table_column'].eq(col).sum() == 0):
            ress[i] = empty
            continue
        # 读取 bins（首条匹配）
        bins = hist_file.loc[hist_file['table_column']==col, 'bins'].iloc[0]
        # 取该条件对应的 op 和 val（缺失时给默认）
        opId = op_ids[i] if i < len(op_ids) else (op_ids[0] if op_ids else 3)
        try:
            op = encoding.idx2op[opId]
        except Exception:
            op = 'NA'
        val = vals[i] if i < len(vals) else 0.0
        mini, maxi = encoding.column_min_max_vals[col]
        val_unnorm = val * (maxi-mini) + mini
        left = 0
        right = len(bins)-1
        for j in range(len(bins)):
            if bins[j] < val_unnorm:
                left = j
            if bins[j] > val_unnorm:
                right = j
                break
        res = np.zeros(len(bins)-1)
        if op == '=':
            res[left:right] = 1
        elif op == '<':
            res[:left] = 1
        elif op == '>':
            res[right:] = 1
        ress[i] = res
    # 其余行保持为零
    return ress.flatten()

class Encoding:
    def __init__(self, column_min_max_vals, 
                 col2idx, op2idx={'>':0, '=':1, '<':2, 'NA':3, '>=':4, '<=':5}):
                #  col2idx, op2idx={'>':0, '=':1, '<':2, 'NA':3}):
        self.column_min_max_vals = column_min_max_vals
        self.col2idx = col2idx
        self.op2idx = op2idx
        
        idx2col = {}
        for k,v in col2idx.items():
            idx2col[v] = k
        self.idx2col = idx2col
        self.idx2op = {0:'>', 1:'=', 2:'<', 3:'NA', 4:'>=', 5:'<='}
        # self.idx2op = {0:'>', 1:'=', 2:'<', 3:'NA'}
        
        self.type2idx = {}
        self.idx2type = {}
        
        self.join2idx = {}
        self.idx2join = {}
        
        self.table2idx = {'NA':0}
        self.idx2table = {0:'NA'}
    
    def normalize_val(self, column, val, log=False):
        mini, maxi = self.column_min_max_vals[column]
        
        val_norm = 0.0
        if maxi > mini:
            val_norm = (val-mini) / (maxi-mini)
        return val_norm
    
    def encode_filters(self, filters=[], alias=None): 
        ## filters: list of dict 

        # print(filters)
        if len(filters) == 0:
            return {'colId':[self.col2idx['NA']],
                   'opId': [self.op2idx['NA']],
                   'val': [0.0]} 
        res = {'colId':[],'opId': [],'val': []}
        for filt in filters:
            filt = ''.join(c for c in filt if c not in '()')
            fs = filt.split(' AND ')
            for f in fs:
     #           print(filters)
                parts = f.split(' ')
                if len(parts) < 3:
                    continue
                # 支持直接传入 Label.Property 形式
                if '.' in parts[0]:
                    column, op, num = parts[0], parts[1], parts[2]
                else:
                    col, op, num = parts[0], parts[1], parts[2]
                    # column = (alias[0] + '.' + col)
                    if alias and len(alias) > 0:
                        column = (alias[0] + '.' + col)
                    else:
                        column = ('NA.' + col)
                if column not in self.col2idx:
                    res['colId'].append(self.col2idx['NA'])
                    res['opId'].append(self.op2idx.get(op, self.op2idx['NA']))
                    res['val'].append(0.0)
                    continue
                res['colId'].append(self.col2idx[column])
                res['opId'].append(self.op2idx.get(op, self.op2idx['NA']))
                # 这里要改一下！！！！！！！！
                try:
                    valf = float(num)
                except Exception:
                    valf = 0.0
                res['val'].append(self.normalize_val(column, valf))
        return res
    
    def encode_join(self, join):
        if join not in self.join2idx:
            if join is None:
                return 0
            self.join2idx[join] = len(self.join2idx)
            self.idx2join[self.join2idx[join]] = join
        return self.join2idx[join]
    
    def encode_table(self, table):
        # print(table)
        # 涉及到多个label，所以使用多热编码
        multihot = np.zeros(len(self.table2idx))# 设置对应位置为1
        for tid in table:
            idx = self.table2idx.get(tid)  # 未知表将返回 None
            if idx is not None:
                multihot[idx] = 1
        return multihot


    def encode_type(self, nodeType):
        if nodeType not in self.type2idx:
            self.type2idx[nodeType] = len(self.type2idx)
            self.idx2type[self.type2idx[nodeType]] = nodeType
        return self.type2idx[nodeType]


class TreeNode:
    def __init__(self, nodeType, typeId, filt, card, join, join_str, filterDict, details, EstimatedRows):
        self.nodeType = nodeType
        self.typeId = typeId
        self.filter = filt
        
        self.table = 'NA'
        self.table_id = 0
        self.query_id = None ## so that sample bitmap can recognise
        
        self.join = join
        self.join_str = join_str
        self.card = card #'Actual Rows'
        self.children = []
        self.rounds = 0
        
        self.filterDict = filterDict
        
        self.parent = None
        
        self.feature = None

        self.details = details
        self.EstimatedRows = EstimatedRows
        
    def addChild(self,treeNode):
        self.children.append(treeNode)

    def inherit_subschema_from_children(self):
        """从子节点继承subschema"""
        for child in self.children:
            # 合并节点标签
            self.subschema['nodes'].update(child.subschema['nodes'])
            # 合并关系类型
            self.subschema['relationships'].update(child.subschema['relationships'])
            # 合并属性
            self.subschema['properties'].update(child.subschema['properties'])
            # 合并节点属性映射
            for node, props in child.subschema['node_properties'].items():
                if node not in self.subschema['node_properties']:
                    self.subschema['node_properties'][node] = set()
                self.subschema['node_properties'][node].update(props)
            # 合并关系属性映射
            for rel, props in child.subschema['rel_properties'].items():
                if rel not in self.subschema['rel_properties']:
                    self.subschema['rel_properties'][rel] = set()
                self.subschema['rel_properties'][rel].update(props)
            # 合并图结构
            self.subschema['graph_structure'].extend(child.subschema['graph_structure'])
    
    def __str__(self):
#        return TreeNode.print_nested(self)
        return '{} with {}, {}, {} children'.format(self.nodeType, self.filter, self.join_str, len(self.children))

    def __repr__(self):
        return self.__str__()
    
    @staticmethod
    def print_nested(node, indent = 0): 
        print('--'*indent+ '{} with {} and {}, {} childs'.format(node.nodeType, node.filter, node.join_str, len(node.children)))
        for k in node.children: 
            TreeNode.print_nested(k, indent+1)






