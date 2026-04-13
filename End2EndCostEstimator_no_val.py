#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import json
import re
import math
# import psycopg2
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import time
from itertools import chain
from random import shuffle
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
torch.set_printoptions(precision=10, threshold=None, edgeitems=None, linewidth=None, profile=None)

def prepare_dataset(prefix_path):
    data={}
    data["aka_name"] = pd.read_csv(prefix_path+'/imdb_data_csv/aka_name.csv',header=None,on_bad_lines='skip',)
    data["aka_title"] = pd.read_csv(prefix_path+'/imdb_data_csv/aka_title.csv',header=None,on_bad_lines='skip',)
    data["cast_info"] = pd.read_csv(prefix_path+'/imdb_data_csv/cast_info.csv',header=None,on_bad_lines='skip',)
    data["char_name"] = pd.read_csv(prefix_path+'/imdb_data_csv/char_name.csv',header=None,on_bad_lines='skip',)
    data["company_name"] = pd.read_csv(prefix_path+'/imdb_data_csv/company_name.csv',header=None,on_bad_lines='skip',)
    data["company_type"] = pd.read_csv(prefix_path+'/imdb_data_csv/company_type.csv',header=None,on_bad_lines='skip',)
    data["comp_cast_type"] = pd.read_csv(prefix_path+'/imdb_data_csv/comp_cast_type.csv',header=None,on_bad_lines='skip',)
    data["complete_cast"] = pd.read_csv(prefix_path+'/imdb_data_csv/complete_cast.csv',header=None,on_bad_lines='skip',)
    data["info_type"] = pd.read_csv(prefix_path+'/imdb_data_csv/info_type.csv',header=None,on_bad_lines='skip',)
    data["keyword"] = pd.read_csv(prefix_path+'/imdb_data_csv/keyword.csv',header=None,on_bad_lines='skip',)
    data["kind_type"] = pd.read_csv(prefix_path+'/imdb_data_csv/kind_type.csv',header=None,on_bad_lines='skip',)
    data["link_type"] = pd.read_csv(prefix_path+'/imdb_data_csv/link_type.csv',header=None,on_bad_lines='skip',)
    data["movie_companies"] = pd.read_csv(prefix_path+'/imdb_data_csv/movie_companies.csv',header=None,on_bad_lines='skip',)
    data["movie_info"] = pd.read_csv(prefix_path+'/imdb_data_csv/movie_info.csv',header=None,on_bad_lines='skip',)
    data["movie_info_idx"] = pd.read_csv(prefix_path+'/imdb_data_csv/movie_info_idx.csv',header=None,on_bad_lines='skip',)
    data["movie_keyword"] = pd.read_csv(prefix_path+'/imdb_data_csv/movie_keyword.csv',header=None,on_bad_lines='skip',)
    data["movie_link"] = pd.read_csv(prefix_path+'/imdb_data_csv/movie_link.csv',header=None,on_bad_lines='skip',)
    data["name"] = pd.read_csv(prefix_path+'/imdb_data_csv/name.csv',header=None,on_bad_lines='skip',)
    data["person_info"] = pd.read_csv(prefix_path+'/imdb_data_csv/person_info.csv',header=None,on_bad_lines='skip',)
    data["role_type"] = pd.read_csv(prefix_path+'/imdb_data_csv/role_type.csv',header=None,on_bad_lines='skip',)
    data["title"] = pd.read_csv(prefix_path+'/imdb_data_csv/title.csv',header=None,on_bad_lines='skip',)
    aka_name_column = [
        'id',
        'person_id',
        'name',
        'imdb_index',
        'name_pcode_cf',
        'name_pcode_nf',
        'surname_pcode',
        'md5sum'
    ]
    aka_title_column = [
        'id',
        'movie_id',
        'title',
        'imdb_index',
        'kind_id',
        'production_year',
        'phonetic_code',
        'episode_of_id',
        'season_nr',
        'episode_nr',
        'note',
        'md5sum'
    ]
    cast_info_column = [
        'id',
        'person_id',
        'movie_id',
        'person_role_id',
        'note',
        'nr_order',
        'role_id'
    ]
    char_name_column = [
        'id',
        'name',
        'imdb_index',
        'imdb_id',
        'name_pcode_nf',
        'surname_pcode',
        'md5sum'
    ]
    comp_cast_type_column = [
        'id',
        'kind'
    ]
    company_name_column = [
        'id',
        'name',
        'country_code',
        'imdb_id',
        'name_pcode_nf',
        'name_pcode_sf',
        'md5sum'
    ]
    company_type_column = [
        'id',
        'kind'
    ]
    complete_cast_column = [
        'id',
        'movie_id',
        'subject_id',
        'status_id'
    ]
    info_type_column = [
        'id',
        'info'
    ]
    keyword_column = [
        'id',
        'keyword',
        'phonetic_code'
    ]
    kind_type_column = [
        'id',
        'kind'
    ]
    link_type_column = [
        'id',
        'link'
    ]
    movie_companies_column = [
        'id',
        'movie_id',
        'company_id',
        'company_type_id',
        'note'
    ]
    movie_info_idx_column = [
        'id',
        'movie_id',
        'info_type_id',
        'info',
        'note'
    ]
    # movie_keyword_column = [
    #     'id',
    #     'movie_id',
    #     'keyword_id'
    # ]
    movie_link_column = [
        'id',
        'movie_id',
        'linked_movie_id',
        'link_type_id'
    ]
    name_column = [
        'id',
        'name',
        'imdb_index',
        'imdb_id',
        'gender',
        'name_pcode_cf',
        'name_pcode_nf',
        'surname_pcode',
        'md5sum'
    ]
    role_type_column = [
        'id',
        'role'
    ]
    title_column = [
        'id',
        'title',
        'imdb_index',
        'kind_id',
        'production_year',
        'imdb_id',
        'phonetic_code',
        'episode_of_id',
        'season_nr',
        'episode_nr',
        'series_years',
        'md5sum'
    ]
    movie_info_column = [
        'id',
        'movie_id',
        'info_type_id',
        'info',
        'note'
    ]
    person_info_column = [
        'id',
        'person_id',
        'info_type_id',
        'info',
        'note'
    ]
    data["aka_name"].columns = aka_name_column
    data["aka_title"].columns = aka_title_column
    data["cast_info"].columns = cast_info_column
    data["char_name"].columns = char_name_column
    data["company_name"].columns = company_name_column
    data["company_type"].columns = company_type_column
    data["comp_cast_type"].columns = comp_cast_type_column
    data["complete_cast"].columns = complete_cast_column
    data["info_type"].columns = info_type_column
    data["keyword"].columns = keyword_column
    data["kind_type"].columns = kind_type_column
    data["link_type"].columns = link_type_column
    data["movie_companies"].columns = movie_companies_column
    data["movie_info"].columns = movie_info_column
    data["movie_info_idx"].columns = movie_info_idx_column
    # data["movie_keyword"].columns = movie_keyword_column
    data["movie_link"].columns = movie_link_column
    data["name"].columns = name_column
    data["person_info"].columns = person_info_column
    data["role_type"].columns = role_type_column
    data["title"].columns = title_column

    column2pos = {}
    column2pos['aka_name'] = aka_name_column
    column2pos['aka_title'] = aka_title_column
    column2pos['cast_info'] = cast_info_column
    column2pos['char_name'] = char_name_column
    column2pos['company_name'] = company_name_column
    column2pos['company_type'] = company_type_column
    column2pos['comp_cast_type'] = comp_cast_type_column
    column2pos['complete_cast'] = complete_cast_column
    column2pos['info_type'] = info_type_column
    column2pos['keyword'] = keyword_column
    column2pos['kind_type'] = kind_type_column
    column2pos['link_type'] = link_type_column
    column2pos['movie_companies'] = movie_companies_column
    column2pos['movie_info'] = movie_info_column
    column2pos['movie_info_idx'] = movie_info_idx_column
    # column2pos['movie_keyword'] = movie_keyword_column
    column2pos['movie_link'] = movie_link_column
    column2pos['name'] = name_column
    column2pos['person_info'] = person_info_column
    column2pos['role_type'] = role_type_column
    column2pos['title'] = title_column
    tables = ['aka_name', 'aka_title', 'cast_info', 'char_name', 'company_name', 'company_type', 'comp_cast_type', 'complete_cast', 'info_type', 'keyword', 'kind_type', 'link_type', 'movie_companies', 'movie_info', 'movie_info_idx',
            #   'movie_keyword',
                'movie_link', 'name', 'person_info', 'role_type', 'title']
    indexes = ['aka_name_pkey', 'aka_title_pkey', 'cast_info_pkey', 'char_name_pkey',
               'comp_cast_type_pkey', 'company_name_pkey', 'company_type_pkey', 'complete_cast_pkey',
               'info_type_pkey', 'keyword_pkey', 'kind_type_pkey', 'link_type_pkey', 'movie_companies_pkey',
               'movie_info_idx_pkey', 'movie_keyword_pkey', 'movie_link_pkey', 'name_pkey', 'role_type_pkey',
               'title_pkey', 'movie_info_pkey', 'person_info_pkey', 'company_id_movie_companies',
               'company_type_id_movie_companies', 'info_type_id_movie_info_idx', 'info_type_id_movie_info',
               'info_type_id_person_info', 'keyword_id_movie_keyword', 'kind_id_aka_title', 'kind_id_title',
               'linked_movie_id_movie_link', 'link_type_id_movie_link', 'movie_id_aka_title', 'movie_id_cast_info',
               'movie_id_complete_cast', 'movie_id_movie_ companies', 'movie_id_movie_info_idx',
               'movie_id_movie_keyword', 'movie_id_movie_link', 'movie_id_movie_info', 'person_id_aka_name',
               'person_id_cast_info', 'person_id_person_info', 'person_role_id_cast_info', 'role_id_cast_info']
    indexes_id = {}
    for idx, index in enumerate(indexes):
        indexes_id[index] = idx + 1
    # physic_ops_id = {'Materialize':1, 'Sort':2, 'Hash':3, 'Merge Join':4, 'Bitmap Index Scan':5,
    #  'Index Only Scan':6, 'BitmapAnd':7, 'Nested Loop':8, 'Aggregate':9, 'Result':10,
    #  'Hash Join':11, 'Seq Scan':12, 'Bitmap Heap Scan':13, 'Index Scan':14, 'BitmapOr':15}
    physic_ops_id={'ProduceResults':1,'Projection':2,'Filter':3,'Expand(All)':4,'NodeUniqueIndexSeekByRange':5,'NodeByLabelScan':6,'CacheProperties':7,'NodeUniqueIndexSeek':8,'NodeHashJoin':9,'CartesianProduct':10}
    strategy_id = {'Plain':1}
    compare_ops_id = {'=':1, '>':2, '<':3, '!=':4, '~~':5, '!~~':6, '!Null': 7, '>=':8, '<=':9}
    bool_ops_id = {'AND':1,'OR':2}
    tables_id = {}
    columns_id = {}
    table_id = 1
    column_id = 1
    for table_name in tables:
        tables_id[table_name] = table_id
        table_id += 1
        for column in column2pos[table_name]:
            columns_id[table_name+'.'+column] = column_id
            column_id += 1 
    return data, indexes_id, tables_id, columns_id, physic_ops_id, compare_ops_id, bool_ops_id



def load_dictionary(path): 
    word_vectors = KeyedVectors.load(path, mmap='r')
    return word_vectors

def load_numeric_min_max(path):
    with open(path,'r') as f:
        min_max_column = json.loads(f.read())
    return min_max_column

def determine_prefix(column):
    relation_name = column.split('.')[0]
    column_name = column.split('.')[1]
    if relation_name == 'aka_title':
        if column_name == 'title':
            return 'title_'
        else:
            print (column)
            raise
    elif relation_name == 'char_name':
        if column_name == 'name':
            return 'name_'
        elif column_name == 'name_pcode_nf':
            return 'nf_'
        elif column_name == 'surname_pcode':
            return 'surname_'
        else:
            print (column)
            raise
    elif relation_name == 'movie_info_idx':
        if column_name == 'info':
            return 'info_'
        else:
            print (column)
            raise
    elif relation_name == 'title':
        if column_name == 'title':
            return 'title_'
        else:
            print (column)
            raise
    elif relation_name == 'role_type':
        if column_name == 'role':
            return 'role_'
        else:
            print (column)
            raise
    elif relation_name == 'movie_companies':
        if column_name == 'note':
            return 'note_'
        else:
            print (column)
            raise
    elif relation_name == 'info_type':
        if column_name == 'info':
            return 'info_'
        else:
            print (column)
            raise
    elif relation_name == 'company_type':
        if column_name == 'kind':
            return ''
        else:
            print (column)
            raise
    elif relation_name == 'company_name':
        if column_name == 'name':
            return 'cn_name_'
        elif column_name == 'country_code':
            return 'country_'
        else:
            print (column)
            raise
    elif relation_name == 'keyword':
        if column_name == 'keyword':
            return 'keyword_'
        else:
            print (column)
            raise

    elif relation_name == 'movie_info':
        if column_name == 'info':
            return ''
        elif column_name == 'note':
            return 'note_'
        else:
            print (column)
            raise
    elif relation_name == 'name':
        if column_name == 'gender':
            return 'gender_'
        elif column_name == 'name':
            return 'name_'
        elif column_name == 'name_pcode_cf':
            return 'cf_'
        elif column_name == 'name_pcode_nf':
            return 'nf_'
        elif column_name == 'surname_pcode':
            return 'surname_'
        else:
            print (column)
            raise
    elif relation_name == 'aka_name':
        if column_name == 'name':
            return 'name_'
        elif column_name == 'name_pcode_cf':
            return 'cf_'
        elif column_name == 'name_pcode_nf':
            return 'nf_'
        elif column_name == 'surname_pcode':
            return 'surname_'
        else:
            print (column)
            raise
    elif relation_name == 'link_type':
        if column_name == 'link':
            return 'link_'
        else:
            print (column)
            raise
    elif relation_name == 'person_info':
        if column_name == 'note':
            return 'note_'
        else:
            print (column)
            raise
    elif relation_name == 'cast_info':
        if column_name == 'note':
            return 'note_'
        else:
            print (column)
            raise
    elif relation_name == 'comp_cast_type':
        if column_name == 'kind':
            return 'kind_'
        else:
            print (column)
            raise
    elif relation_name == 'kind_type':
        if column_name == 'kind':
            return 'kind_'
        else:
            print (column)
            raise
    else:
        print (column)
        raise

def get_representation(value):
    if value in word_vectors:
        embedded_result = np.array(list(word_vectors[value]))
    else:
        embedded_result = np.array([0.0 for _ in range(500)])
    hash_result = np.array([0.0 for _ in range(500)])
    for t in value:
        hash_result[hash(t) % 500] = 1.0
    return np.concatenate((embedded_result, hash_result), 0)

def get_str_representation(value, column):
    vec = np.array([])
    count = 0
    prefix = determine_prefix(column)
    for v in value.split('%'):
        if len(v) > 0:
            if len(vec) == 0:
                vec = get_representation(prefix+v)
                count = 1
            else:
                new_vec = get_representation(prefix+v)
                vec = vec + new_vec
                count += 1
    if count > 0:
        vec = vec / float(count)
    return vec

def encode_condition_op(condition_op, relation_name, index_name):
    # bool_operator + left_value + compare_operator + right_value
    vec = []
    try:
        if condition_op is None:
            vec = [0.0 for _ in range(condition_op_dim)]
        elif condition_op.get('op_type') == 'Bool':
            idx = bool_ops_id.get(condition_op.get('operator'), None)
            vec = [0.0 for _ in range(bool_ops_total_num)]
            if idx is not None and idx - 1 < len(vec):
                vec[idx - 1] = 1.0
        else:
            operator = condition_op.get('operator')
            left_value = condition_op.get('left_value')

            if left_value is None:
                vec = [0.0 for _ in range(condition_op_dim)]
            else:
                if re.match(r'.+\..+', left_value) is None:
                    if relation_name is None and index_name is not None:
                        # 防止 split 异常
                        try:
                            relation_name = index_name.split(left_value)[1].strip('_')
                        except Exception:
                            relation_name = None
                    if relation_name is not None:
                        left_value = relation_name + '.' + left_value

                if (left_value not in columns_id) or (relation_name not in data):
                    vec = [0.0 for _ in range(condition_op_dim)]
                else:
                    left_value_idx = columns_id[left_value]
                    left_value_vec = [0.0 for _ in range(column_total_num)]
                    left_value_vec[left_value_idx - 1] = 1.0

                    operator_idx = compare_ops_id.get(operator, compare_ops_id.get('='))
                    operator_vec = [0.0 for _ in range(compare_ops_total_num)]
                    if operator_idx is not None and operator_idx - 1 < len(operator_vec):
                        operator_vec[operator_idx - 1] = 1.0

                    # 右值统一做“最小可用”编码，避免空向量
                    right_value = condition_op.get('right_value', None)
                    right_value_vec = [0.0]  # 至少1维，后续统一pad到condition_op_dim

                    # 数值列时尝试归一化
                    try:
                        column_name = left_value.split('.')[1]
                        if relation_name in data and column_name in data[relation_name].columns:
                            dtype_name = str(data[relation_name].dtypes[column_name])
                            if dtype_name in ('int64', 'float64') and right_value not in [None, 'None']:
                                rv = float(right_value)
                                value_max = min_max_column[relation_name][column_name]['max']
                                value_min = min_max_column[relation_name][column_name]['min']
                                if value_max != value_min:
                                    right_value_vec = [(rv - value_min) / (value_max - value_min)]
                                else:
                                    right_value_vec = [0.0]
                    except Exception:
                        pass

                    vec = [0.0 for _ in range(bool_ops_total_num)] + left_value_vec + operator_vec + right_value_vec

    except Exception:
        # 任意解析异常都兜底为固定长度零向量
        vec = [0.0 for _ in range(condition_op_dim)]

    # ===== 关键：强制固定长度，杜绝空宽度 =====
    if vec is None:
        vec = []
    vec = list(vec)

    if len(vec) >= condition_op_dim:
        result = np.array(vec[:condition_op_dim], dtype=np.float32)
    else:
        num_pad = condition_op_dim - len(vec)
        result = np.pad(np.array(vec, dtype=np.float32), (0, num_pad), 'constant')

    return result

def encode_sample(sample):
    return np.array([int(i) for i in sample])

def bitand(sample1, sample2):
    return np.minimum(sample1, sample2)

def encode_node_job(node, condition_max_num):
    # operator + first_condition + second_condition + relation
    extra_info_num = max(column_total_num, table_total_num, index_total_num)
    operator_vec = np.array([0 for _ in range(physic_op_total_num)])

    extra_info_vec = np.array([0 for _ in range(extra_info_num)])
    condition1_vec = np.array([[0 for _ in range(condition_op_dim)] for _ in range(condition_max_num)])
    condition2_vec = np.array([[0 for _ in range(condition_op_dim)] for _ in range(condition_max_num)])
    ### Samples Starts
    sample_vec = np.array([1 for _ in range(1000)])
    ### Samples Ends
    has_condition = 0
    if node != None:
        operator = node['node_type']
        operator_idx = physic_ops_id[operator]
        operator_vec[operator_idx-1] = 1
        if operator == 'Materialize' or operator == 'BitmapAnd' or operator == 'Result':
            pass
        elif operator == 'Sort':
            for key in node['sort_keys']:
                extra_info_inx = columns_id[key]
                extra_info_vec[extra_info_inx-1] = 1
        elif operator == 'Hash Join' or operator == 'Merge Join' or operator == 'Nested Loop':
            condition1_vec = encode_condition(node['condition'], None, None, condition_max_num)
        elif operator == 'Aggregate':
            for key in node['group_keys']:
                extra_info_inx = columns_id[key]
                extra_info_vec[extra_info_inx-1] = 1
        elif operator == 'Seq Scan' or operator == 'Bitmap Heap Scan' or operator == 'Index Scan' or operator == 'Bitmap Index Scan' or operator == 'Index Only Scan':
            relation_name = node['relation_name']
            index_name = node['index_name']
            if relation_name != None:
                extra_info_inx = tables_id[relation_name]
            else:
                extra_info_inx = indexes_id[index_name]
            extra_info_vec[extra_info_inx-1] = 1
            condition1_vec = encode_condition(node['condition_filter'], relation_name, index_name, condition_max_num)
            condition2_vec = encode_condition(node['condition_index'], relation_name, index_name, condition_max_num)
            if 'bitmap' in node:
                ### Samples Starts
                sample_vec = encode_sample(node['bitmap'])
                ### Samples Ends
                has_condition = 1
            if 'bitmap_filter' in node:
                ### Samples Starts
                sample_vec = bitand(encode_sample(node['bitmap_filter']), sample_vec)
                ### Samples Ends
                has_condition = 1
            if 'bitmap_index' in node:
                ### Samples Starts
                sample_vec = bitand(encode_sample(node['bitmap_index']), sample_vec)
                ### Samples Ends
                has_condition = 1

#     print 'operator: ', operator_vec
#     print 'extra_infos: ', extra_info_vec
    return operator_vec, extra_info_vec, condition1_vec, condition2_vec, sample_vec, has_condition

def encode_condition(condition, relation_name, index_name, condition_max_num):
    # condition 可能是 None
    if condition is None or len(condition) == 0:
        vecs = [np.zeros(condition_op_dim, dtype=np.float32)]
    else:
        vecs = []
        for condition_op in condition:
            v = encode_condition_op(condition_op, relation_name, index_name)
            # 双保险：保证每条都是 1D 且长度=condition_op_dim
            v = np.asarray(v, dtype=np.float32).reshape(-1)
            if v.shape[0] != condition_op_dim:
                if v.shape[0] > condition_op_dim:
                    v = v[:condition_op_dim]
                else:
                    v = np.pad(v, (0, condition_op_dim - v.shape[0]), 'constant')
            vecs.append(v)

    vecs = np.asarray(vecs, dtype=np.float32)  # [num_cond, condition_op_dim]
    num_pad = max(0, condition_max_num - len(vecs))
    result = np.pad(vecs, ((0, num_pad), (0, 0)), 'constant')
    return result

class TreeNode(object):
    def __init__(self, current_vec, parent, idx, level_id):
        self.item = current_vec
        self.idx = idx
        self.level_id = level_id
        self.parent = parent
        self.children = []
    def get_parent(self):
        return self.parent
    def get_item(self):
        return self.item
    def get_children(self):
        return self.children
    def add_child(self, child):
        self.children.append(child)
    def get_idx(self):
        return self.idx
    def __str__(self):
        return 'level_id: ' + self.level_id + '; idx: ' + self.idx

def recover_tree(vecs, parent, start_idx):
    if len(vecs) == 0:
        return vecs, start_idx
    if vecs[0] == None:
        return vecs[1:], start_idx+1
    node = TreeNode(vecs[0], parent, start_idx, -1)
    while True:
        vecs, start_idx = recover_tree(vecs[1:], node, start_idx+1)
        parent.add_child(node)
        if len(vecs) == 0:
            return vecs, start_idx
        if vecs[0] == None:
            return vecs[1:], start_idx+1
        node = TreeNode(vecs[0], parent, start_idx, -1)

def dfs_tree_to_level(root, level_id, nodes_by_level):
    root.level_id = level_id
    if len(nodes_by_level) <= level_id:
        nodes_by_level.append([])
    nodes_by_level[level_id].append(root)
    root.idx = len(nodes_by_level[level_id])
    for c in root.get_children():
        dfs_tree_to_level(c, level_id+1, nodes_by_level)

def debug_nodes_by_level(nodes_by_level):
    for nodes in nodes_by_level:
        for node in nodes:
            whitespace = ''
            for i in range(node.level_id):
                whitespace += ' '
            print (whitespace + 'level_id: ' + str(node.level_id))
            print (whitespace + 'idx: ' + str(node.idx))

def encode_plan_job(plan, condition_max_num):
    original_cypher = plan['cypher']
    
    operators, extra_infos, condition1s, condition2s, samples, condition_masks = [], [], [], [], [], []
    mapping = []
    nodes_by_level = []

    # node = TreeNode(plan[0], None, 0, -1)
    # recover_tree(plan[1:], node, 1)
    # 递归构建树，传入 cypher 以便解析
    node = cypher_json_to_tree_node(plan, None, 0, -1, original_cypher)
    dfs_tree_to_level(node, 0, nodes_by_level)

#     print (plan)
#     debug_nodes_by_level(nodes_by_level)


    for level in nodes_by_level:
        operators.append([])
        extra_infos.append([])
        condition1s.append([])
        condition2s.append([])
        samples.append([])
        condition_masks.append([])
        mapping.append([])
        for node in level:
            operator, extra_info, condition1, condition2, sample, condition_mask = encode_node_job(node.item, condition_max_num)
            operators[-1].append(operator)
            extra_infos[-1].append(extra_info)
            condition1s[-1].append(condition1)
            condition2s[-1].append(condition2)
            samples[-1].append(sample)
            condition_masks[-1].append(condition_mask)
            if len(node.children) == 2:
                mapping[-1].append([n.idx for n in node.children])
            elif len(node.children) == 1:
                mapping[-1].append([node.children[0].idx, 0])
            else:
                mapping[-1].append([0, 0])
#     num_pad = plan_node_max_num - len(operators)
#     masks = [0 for _ in range(plan_node_max_num)]
#     for i in range(len(operators)):
#         if operators[i].sum() > 0:
#             masks[i] = 1
#         else:
#             masks[i] = 0
#     masks = np.array(masks)
#     condition_masks = np.array(condition_masks)
#     operators, extra_infos, condition1s, condition2s = np.pad(operators, ((0, num_pad), (0,0)), 'constant'), np.pad(extra_infos, ((0, num_pad), (0,0)), 'constant'),np.pad(condition1s, ((0, num_pad), (0,0), (0,0)), 'constant'),np.pad(condition2s, ((0, num_pad), (0,0), (0,0)), 'constant')
#     samples = np.pad(samples, ((0, num_pad), (0,0)), 'constant')
#     condition_masks = np.pad(condition_masks, (0, num_pad), 'constant')
    return operators, extra_infos, condition1s, condition2s, samples, condition_masks, mapping

def normalize_label(labels, mini, maxi):
    labels_norm = (torch.log(labels) - mini) / (maxi - mini)
    labels_norm = torch.min(labels_norm, torch.ones_like(labels_norm))
    labels_norm = torch.max(labels_norm, torch.zeros_like(labels_norm))
    return labels_norm

def unnormalize(vecs, mini, maxi):
    return torch.exp(vecs * (maxi - mini) + mini)

def obtain_upper_bound_query_size(path):
    plan_node_max_num = 0
    condition_max_num = 0
    cost_label_max = 0.0
    cost_label_min = float('inf')
    card_label_max = 0.0
    card_label_min = float('inf')
    plans = []
    
    def traverse_plan_tree(node):
        """递归遍历 Plan 树，统计节点数和条件数"""
        nonlocal plan_node_max_num, condition_max_num
        
        # 统计当前节点
        plan_node_max_num += 1
        
        # 统计条件数（解析 Details 字段）
        details = None
        if 'Details' in node:
            details = node['Details']
        elif 'args' in node and 'Details' in node['args']:
            details = node['args']['Details']
        
        if details:
            # 简单统计条件数：通过 AND/OR 分割
            # 注意：这只是粗略统计，实际条件解析需要更复杂的逻辑
            import re
            # 匹配比较操作符（>, <, =, >=, <=, !=, =~）
            conditions = re.findall(r'[><=!]=?|~=|AND|OR', details, re.IGNORECASE)
            condition_count = len(conditions)
            if condition_count > condition_max_num:
                condition_max_num = condition_count
        
        # 递归处理子节点
        if 'children' in node and node['children']:
            for child in node['children']:
                traverse_plan_tree(child)
    
    df = pd.read_csv(path)
    # 获取 json 列（包含 Plan 信息的 JSON 字符串）
    json_col = df['json']

    for json_str in json_col:
        # 解析最外层的 JSON
        plan_data = json.loads(json_str) 
        cost = plan_data['Execution Time']
        
        # 更新 cost Min/Max
        if cost > cost_label_max:
            cost_label_max = cost
        if cost < cost_label_min:
            cost_label_min = cost
        
        # 获取最顶层 Plan 的 rows（即查询的最终基数）
        cardinality = None
        if 'rows' in plan_data['Plan'] and plan_data['Plan']['rows'] is not None:
            cardinality = plan_data['Plan']['rows']
        elif 'args' in plan_data['Plan'] and 'Rows' in plan_data['Plan']['args']:
            cardinality = plan_data['Plan']['args']['Rows']
        
        if cardinality is not None:
            if cardinality > card_label_max:
                card_label_max = cardinality
            if cardinality < card_label_min:
                card_label_min = cardinality
        
        # 递归遍历 Plan 树统计节点数和条件数
        traverse_plan_tree(plan_data['Plan'])

    cost_label_min, cost_label_max = math.log(cost_label_min), math.log(cost_label_max)
    card_label_min, card_label_max = math.log(card_label_min), math.log(card_label_max)
    print (plan_node_max_num, condition_max_num)
    print (cost_label_min, cost_label_max)
    print (card_label_min, card_label_max)
    return plan_node_max_num, condition_max_num, cost_label_min, cost_label_max, card_label_min, card_label_max

def merge_plans_level(level1, level2, isMapping=False):
    for idx, level in enumerate(level2):
        if idx >= len(level1):
            level1.append([])
        if isMapping:
            if idx < len(level1) - 1:
                base = len(level1[idx+1])
                for i in range(len(level)):
                    if level[i][0] > 0:
                        level[i][0] += base
                    if level[i][1] > 0:
                        level[i][1] += base
        level1[idx] += level
    return level1

from model.details_parser import DetailsParser
from model.cypher_format_join import cypher_format_join
from model.cypher_format_filter import cypher_format_filter, _cypher_tokens
def cypher_json_to_tree_node(json_node, parent, idx, level_id, original_cypher):
    """
    递归将 Cypher Plan JSON 转为 E2E TreeNode
    增加 original_cypher 参数以便解析 filter 时能查找参数值
    """
    # args = json_node['args']
    # op_type = json_node['operatorType']

    # 2. 调用现有的解析器逻辑
    # parser = DetailsParser()
    # parsed = parser.parse_details(op_type, json_node['args'].get('Details',''))

    filters, alias = cypher_format_filter(json_node, original_cypher)
    join = cypher_format_join(json_node)

    # 1. 构造 E2E Item 字典
    item = {
        'node_type': json_node['operatorType'],
        'condition_filter': filters,
        'condition_index': join,
        'relation_name': alias,
        # 'index_name': None,
        # 'bitmap': None  # 如果需要 bitmap 支持，需额外透传
    }
    # 3. 创建节点
    node = TreeNode(item, parent, idx, level_id)
    
    # 4. 递归处理子节点
    children = json_node.get('children', [])
    for child_json in children:
        child_node = cypher_json_to_tree_node(child_json, node, -1, -1,original_cypher)
        node.add_child(child_node)
        
    return node

def make_data_job(plans):
    target_cost_batch = []
    target_card_batch = []
    operators_batch = []
    extra_infos_batch = []
    condition1s_batch = []
    condition2s_batch = []
    node_masks_batch = []
    samples_batch = []
    condition_masks_batch = []
    mapping_batch = []

    for plan in plans:
        # target_cost = plan['cost']
        # target_cardinality = plan['cardinality']
        target_cost = plan['Execution Time']
        target_cardinality = plan['Plan']['rows']
        target_cost_batch.append(target_cost)
        target_card_batch.append(target_cardinality)
        plan = plan['Plan']
        # plan = plan['seq']
        operators, extra_infos, condition1s, condition2s, samples, condition_masks, mapping = encode_plan_job(plan, condition_max_num)

        operators_batch = merge_plans_level(operators_batch, operators)
        extra_infos_batch = merge_plans_level(extra_infos_batch, extra_infos)
        condition1s_batch = merge_plans_level(condition1s_batch, condition1s)
        condition2s_batch = merge_plans_level(condition2s_batch, condition2s)
        samples_batch = merge_plans_level(samples_batch, samples)
        condition_masks_batch = merge_plans_level(condition_masks_batch, condition_masks)
        mapping_batch = merge_plans_level(mapping_batch, mapping, True)
    max_nodes = 0
    for o in operators_batch:
        if len(o) > max_nodes:
            max_nodes = len(o)
    print (max_nodes)
    print (len(condition2s_batch))
    operators_batch = np.array([np.pad(v, ((0, max_nodes - len(v)),(0,0)), 'constant') for v in operators_batch])
    extra_infos_batch = np.array([np.pad(v, ((0, max_nodes - len(v)),(0,0)), 'constant') for v in extra_infos_batch])
    
    # 新增：调试打印（定位2维/异常shape）
    for i, v in enumerate(condition1s_batch):
        a = np.asarray(v)
        if a.ndim != 3:
            print(f"[WARN][condition1] sample_idx={i}, ndim={a.ndim}, shape={a.shape}")
    for i, v in enumerate(condition2s_batch):
        a = np.asarray(v)
        if a.ndim != 3:
            print(f"[WARN][condition2] sample_idx={i}, ndim={a.ndim}, shape={a.shape}")
    condition1s_batch = np.array([np.pad(v, ((0, max_nodes - len(v)),(0,0),(0,0)), 'constant') for v in condition1s_batch])
    condition2s_batch = np.array([np.pad(v, ((0, max_nodes - len(v)),(0,0),(0,0)), 'constant') for v in condition2s_batch])
    # condition1s_batch = np.array([
    #     np.pad((np.asarray(v)[:, None, :] if np.asarray(v).ndim == 2 else np.asarray(v)),
    #         ((0, max_nodes - len(v)), (0, 0), (0, 0)),
    #         'constant')
    #     for v in condition1s_batch
    # ], dtype=np.float32)
    # condition2s_batch = np.array([
    # np.pad((np.asarray(v)[:, None, :] if np.asarray(v).ndim == 2 else np.asarray(v)),
    #        ((0, max_nodes - len(v)), (0, 0), (0, 0)),
    #        'constant')
    #     for v in condition2s_batch
    # ], dtype=np.float32)
    samples_batch = np.array([np.pad(v, ((0, max_nodes - len(v)),(0,0)), 'constant') for v in samples_batch])
    condition_masks_batch = np.array([np.pad(v, (0, max_nodes - len(v)), 'constant') for v in condition_masks_batch])
    mapping_batch = np.array([np.pad(v, ((0, max_nodes - len(v)),(0,0)), 'constant') for v in mapping_batch])

    print ('operators_batch: ', operators_batch.shape)

    target_cost_batch = torch.FloatTensor(target_cost_batch)
    target_card_batch = torch.FloatTensor(target_card_batch)
    operators_batch = torch.FloatTensor([operators_batch])
    extra_infos_batch = torch.FloatTensor([extra_infos_batch])
    condition1s_batch = torch.FloatTensor([condition1s_batch])
    condition2s_batch = torch.FloatTensor([condition2s_batch])
    samples_batch = torch.FloatTensor([samples_batch])
    condition_masks_batch = torch.FloatTensor([condition_masks_batch])
    mapping_batch = torch.FloatTensor([mapping_batch])

    target_cost_batch = normalize_label(target_cost_batch, cost_label_min, cost_label_max)
    target_card_batch = normalize_label(target_card_batch, card_label_min, card_label_max)

    return (target_cost_batch, target_card_batch, operators_batch, extra_infos_batch, condition1s_batch, condition2s_batch, samples_batch, condition_masks_batch, mapping_batch)

def chunks(arr, batch_size):
    return [arr[i:i+batch_size] for i in range(0, len(arr), batch_size)]

def save_data_job(plans, istest = True, batch_size=64, directory='/home/sunji/learnedcardinality/job'):
    if istest:
        suffix = 'test_'
    else:
        suffix = ''
    batch_id = 0
    for batch_id, plans_batch in enumerate(chunks(plans, batch_size)):
        print ('batch_id', batch_id, len(plans_batch))
        target_cost_batch, target_cardinality_batch, operators_batch, extra_infos_batch, condition1s_batch, condition2s_batch, samples_batch, condition_masks_batch, mapping_batch = make_data_job(plans_batch)
        np.save(directory+'/target_cost_'+suffix+str(batch_id)+'.np', target_cost_batch.numpy())
        np.save(directory+'/target_cardinality_'+suffix+str(batch_id)+'.np', target_cardinality_batch.numpy())
        np.save(directory+'/operators_'+suffix+str(batch_id)+'.np', operators_batch.numpy())
        np.save(directory+'/extra_infos_'+suffix+str(batch_id)+'.np', extra_infos_batch.numpy())
        np.save(directory+'/condition1s_'+suffix+str(batch_id)+'.np', condition1s_batch.numpy())
        np.save(directory+'/condition2s_'+suffix+str(batch_id)+'.np', condition2s_batch.numpy())
        np.save(directory+'/samples_'+suffix+str(batch_id)+'.np', samples_batch.numpy())
        np.save(directory+'/condition_masks_'+suffix+str(batch_id)+'.np', condition_masks_batch.numpy())
        np.save(directory+'/mapping_'+suffix+str(batch_id)+'.np', mapping_batch.numpy())
        print ('saved: ', str(batch_id))

def get_batch_job(batch_id, istest=False, directory='Learning-based-cost-estimator/job'):
    if istest:
        suffix = 'test_'
    else:
        suffix = ''
    target_cost_batch = np.load(directory+'/target_cost_'+suffix+str(batch_id)+'.np.npy')
    target_cardinality_batch = np.load(directory+'/target_cardinality_'+suffix+str(batch_id)+'.np.npy')
    operators_batch = np.load(directory+'/operators_'+suffix+str(batch_id)+'.np.npy')
    extra_infos_batch = np.load(directory+'/extra_infos_'+suffix+str(batch_id)+'.np.npy')
    condition1s_batch = np.load(directory+'/condition1s_'+suffix+str(batch_id)+'.np.npy')
    condition2s_batch = np.load(directory+'/condition2s_'+suffix+str(batch_id)+'.np.npy')
    samples_batch = np.load(directory+'/samples_'+suffix+str(batch_id)+'.np.npy')
    condition_masks_batch = np.load(directory+'/condition_masks_'+suffix+str(batch_id)+'.np.npy')
    mapping_batch = np.load(directory+'/mapping_'+suffix+str(batch_id)+'.np.npy')
    return target_cost_batch, target_cardinality_batch, operators_batch, extra_infos_batch, condition1s_batch, condition2s_batch, samples_batch, condition_masks_batch, mapping_batch

def encode_train_plan_seq_save(path, batch_size=64, directory='Learning-based-cost-estimator/job'):
    test_plans = []
    # with open(path, 'r') as f:
    #     for idx, seq in enumerate(f.readlines()):
    #         plan = json.loads(seq)
    #         test_plans.append(plan)
#     shuffle(test_plans)
    df = pd.read_csv(path)
    plans_json_str = df['json'].tolist()
    for idx, seq in enumerate(plans_json_str):
        plan = json.loads(seq)
        test_plans.append(plan)
    return save_data_job(plans=test_plans, batch_size=batch_size, directory=directory)

def encode_test_plan_seq_save(path, batch_size=64, directory='Learning-based-cost-estimator/job'):
    test_plans = []
    df = pd.read_csv(path)
    plans_json_str = df['json'].tolist()
    for idx, seq in enumerate(plans_json_str):
        plan = json.loads(seq)
        test_plans.append(plan)
    save_data_job(plans=test_plans, istest=True, batch_size=batch_size, directory=directory)

class Representation(nn.Module):
    def __init__(self, input_dim, hidden_dim, hid_dim, middle_result_dim, task_num):
        super(Representation, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm1 = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.batch_norm1 = nn.BatchNorm1d(hid_dim)
        # The linear layer that maps from hidden state space to tag space

        self.sample_mlp = nn.Linear(1000, hid_dim)
        self.condition_mlp = nn.Linear(hidden_dim, hid_dim)
#         self.out_mlp1 = nn.Linear(hidden_dim, middle_result_dim)
#         self.hid_mlp1 = nn.Linear(15+108+2*hid_dim, hid_dim)
#         self.out_mlp1 = nn.Linear(hid_dim, middle_result_dim)

        self.lstm2 = nn.LSTM(10+105+2*hid_dim, hidden_dim, batch_first=True)
#         self.lstm2_binary = nn.LSTM(15+108+2*hid_dim, hidden_dim, batch_first=True)
#         self.lstm2_binary = nn.LSTM(15+108+2*hid_dim, hidden_dim, batch_first=True)
        self.batch_norm2 = nn.BatchNorm1d(hidden_dim)
        # The linear layer that maps from hidden state space to tag space
        self.hid_mlp2_task1 = nn.Linear(hidden_dim, hid_dim)
        self.hid_mlp2_task2 = nn.Linear(hidden_dim, hid_dim)
        self.batch_norm3 = nn.BatchNorm1d(hid_dim)
        self.hid_mlp3_task1 = nn.Linear(hid_dim, hid_dim)
        self.hid_mlp3_task2 = nn.Linear(hid_dim, hid_dim)
        self.out_mlp2_task1 = nn.Linear(hid_dim, 1)
        self.out_mlp2_task2 = nn.Linear(hid_dim, 1)
    #         self.hidden2values2 = nn.Linear(hidden_dim, action_num)

    def init_hidden(self, hidden_dim, batch_size=1):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(1, batch_size, hidden_dim),
                torch.zeros(1, batch_size, hidden_dim))

    def forward(self, operators, extra_infos, condition1s, condition2s, samples, condition_masks, mapping):
        # condition1
        batch_size = 0
        for i in range(operators.size()[1]):
            if operators[0][i].sum(0) != 0:
                batch_size += 1
            else:
                break
        print ('batch_size: ', batch_size)

#         print (operators.size())
#         print (extra_infos.size())
#         print (condition1s.size())
#         print (condition2s.size())
#         print (samples.size())
#         print (condition_masks.size())
#         print (mapping.size())

#         torch.Size([14, 133, 15])
#         torch.Size([14, 133, 108])
#         torch.Size([14, 133, 13, 1119])
#         torch.Size([14, 133, 13, 1119])
#         torch.Size([14, 133, 1000])
#         torch.Size([14, 133, 1])
#         torch.Size([14, 133, 2])

        num_level = condition1s.size()[0]
        num_node_per_level = condition1s.size()[1]
        num_condition_per_node = condition1s.size()[2]
        condition_op_length = condition1s.size()[3]

        inputs = condition1s.view(num_level * num_node_per_level, num_condition_per_node, condition_op_length)
        hidden = self.init_hidden(self.hidden_dim, num_level * num_node_per_level)

        out, hid = self.lstm1(inputs, hidden)
        last_output1 = hid[0].view(num_level * num_node_per_level, -1)

        # condition2
        num_level = condition2s.size()[0]
        num_node_per_level = condition2s.size()[1]
        num_condition_per_node = condition2s.size()[2]
        condition_op_length = condition2s.size()[3]

        inputs = condition2s.view(num_level * num_node_per_level, num_condition_per_node, condition_op_length)
        hidden = self.init_hidden(self.hidden_dim, num_level * num_node_per_level)

        out, hid = self.lstm1(inputs, hidden)
        last_output2 = hid[0].view(num_level * num_node_per_level, -1)

        last_output1 = F.relu(self.condition_mlp(last_output1))
        last_output2 = F.relu(self.condition_mlp(last_output2))
        last_output = (last_output1 + last_output2) / 2
        last_output = self.batch_norm1(last_output).view(num_level, num_node_per_level, -1)

#         print (last_output.size())
#         torch.Size([14, 133, 256])

        sample_output = F.relu(self.sample_mlp(samples))
        sample_output = sample_output * condition_masks

        out = torch.cat((operators, extra_infos, last_output, sample_output), 2)
#         print (out.size())
#         torch.Size([14, 133, 635])
#         out = out * node_masks
        start = time.time()
        hidden = self.init_hidden(self.hidden_dim, num_node_per_level)
        last_level = out[num_level-1].view(num_node_per_level, 1, -1)
#         torch.Size([133, 1, 635])
        _, (hid, cid) = self.lstm2(last_level, hidden)
        mapping = mapping.long()
        for idx in reversed(range(0, num_level-1)):
            mapp_left = mapping[idx][:,0]
            mapp_right = mapping[idx][:,1]
            pad = torch.zeros_like(hid)[:,0].unsqueeze(1)
            next_hid = torch.cat((pad, hid), 1)
            pad = torch.zeros_like(cid)[:,0].unsqueeze(1)
            next_cid = torch.cat((pad, cid), 1)
            hid_left = torch.index_select(next_hid, 1, mapp_left)
            cid_left = torch.index_select(next_cid, 1, mapp_left)
            hid_right = torch.index_select(next_hid, 1, mapp_right)
            cid_right = torch.index_select(next_cid, 1, mapp_right)
            hid = (hid_left + hid_right) / 2
            cid = (cid_left + cid_right) / 2
            last_level = out[idx].view(num_node_per_level, 1, -1)
            _, (hid, cid) = self.lstm2(last_level, (hid, cid))
        output = hid[0]
#         print (output.size())
#         torch.Size([133, 128])
        end = time.time()
        print ('Forest Evaluate Running Time: ', end - start)
        last_output = output[0:batch_size]
        out = self.batch_norm2(last_output)

        out_task1 = F.relu(self.hid_mlp2_task1(out))
        out_task1 = self.batch_norm3(out_task1)
        out_task1 = F.relu(self.hid_mlp3_task1(out_task1))
        out_task1 = self.out_mlp2_task1(out_task1)
        out_task1 = F.sigmoid(out_task1)

        out_task2 = F.relu(self.hid_mlp2_task2(out))
        out_task2 = self.batch_norm3(out_task2)
        out_task2 = F.relu(self.hid_mlp3_task2(out_task2))
        out_task2 = self.out_mlp2_task2(out_task2)
        out_task2 = F.sigmoid(out_task2)
#         print 'out: ', out.size()
        # batch_size * task_num
        return out_task1, out_task2

def qerror_loss(preds, targets, mini, maxi):
    qerror = []
    preds = unnormalize(preds, mini, maxi)
    targets = unnormalize(targets, mini, maxi)
    for i in range(len(targets)):
        if (preds[i] > targets[i]).cpu().data.numpy()[0]:
            qerror.append(preds[i]/targets[i])
        else:
            qerror.append(targets[i]/preds[i])
    return torch.mean(torch.cat(qerror)), torch.median(torch.cat(qerror)), torch.max(torch.cat(qerror)), torch.argmax(torch.cat(qerror))

def train(train_start, train_end, validate_start, validate_end, num_epochs):
    input_dim = condition_op_dim
    hidden_dim = 128
    hid_dim = 256
    middle_result_dim = 128
    task_num = 2
    model = Representation(input_dim, hidden_dim, hid_dim, middle_result_dim, task_num)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.train()

    # 新增：读取测试集 src_file（顺序与编码一致）
    test_meta_df = pd.read_csv('data/job/plan_and_cost/test_by_para_v2_same_500.csv')
    test_src_files = test_meta_df['src_file'].tolist() if 'src_file' in test_meta_df.columns else ['unknown'] * len(test_meta_df)
    # 新增：记录 loss 历史用于绘图
    train_cost_losses = []
    train_card_losses = []
    val_cost_losses = []
    val_card_losses = []
    epochs_list = []

    start = time.time()
    for epoch in range(num_epochs):
        cost_loss_total = 0.
        card_loss_total = 0.
        model.train()
        for batch_idx in range(train_start, train_end):
            print ('batch_idx: ', batch_idx)
            target_cost, target_cardinality, operatorss, extra_infoss, condition1ss, condition2ss, sampless, condition_maskss, mapping = get_batch_job(batch_idx)
            target_cost, target_cardinality, operatorss, extra_infoss, condition1ss, condition2ss, sampless, condition_maskss, mapping = torch.FloatTensor(target_cost), torch.FloatTensor(target_cardinality),torch.FloatTensor(operatorss),torch.FloatTensor(extra_infoss),torch.FloatTensor(condition1ss),torch.FloatTensor(condition2ss), torch.FloatTensor(sampless), torch.FloatTensor(condition_maskss), torch.FloatTensor(mapping)
            operatorss, extra_infoss, condition1ss, condition2ss, condition_maskss = operatorss.squeeze(0), extra_infoss.squeeze(0), condition1ss.squeeze(0), condition2ss.squeeze(0), condition_maskss.squeeze(0).unsqueeze(2)
            sampless = sampless.squeeze(0)
            mapping = mapping.squeeze(0)
            target_cost, target_cardinality, operatorss, extra_infoss, condition1ss, condition2ss = Variable(target_cost), Variable(target_cardinality), Variable(operatorss), Variable(extra_infoss), Variable(condition1ss), Variable(condition2ss)
            sampless = Variable(sampless)
            optimizer.zero_grad()
            estimate_cost,estimate_cardinality = model(operatorss, extra_infoss, condition1ss, condition2ss, sampless, condition_maskss, mapping)
            target_cost = target_cost
            target_cardinality = target_cardinality
            cost_loss,cost_loss_median,cost_loss_max,cost_max_idx = qerror_loss(estimate_cost, target_cost, cost_label_min, cost_label_max)
            card_loss,card_loss_median,card_loss_max,card_max_idx = qerror_loss(estimate_cardinality, target_cardinality, card_label_min, card_label_max)
            # print (card_loss.item(),card_loss_median.item(),card_loss_max.item(),card_max_idx.item())
            loss = cost_loss + card_loss
            cost_loss_total += cost_loss.item()
            card_loss_total += card_loss.item()
            start = time.time()
            loss.backward()
            optimizer.step()
            end = time.time()
            print ('batchward time: ',end - start)
        batch_num = train_end - train_start

        # 画图
        train_cost_avg = cost_loss_total/batch_num
        train_card_avg = card_loss_total/batch_num
        print("Epoch {}, training cost loss: {}, training card loss: {}".format(epoch, train_cost_avg, train_card_avg))
        # 记录训练 loss
        train_cost_losses.append(train_cost_avg)
        train_card_losses.append(train_card_avg)

        final_results_cost = [] # 用列表收集 cost 预测结果
        final_results_card = [] # 用列表收集 cardinality 预测结果
        cost_loss_total = 0.
        card_loss_total = 0.

        # 新增：验证阶段的全局偏移
        sample_offset = 0

        for batch_idx in range(validate_start, validate_end):
            print ('batch_idx: ', batch_idx)
            target_cost, target_cardinality, operatorss, extra_infoss, condition1ss, condition2ss, sampless, condition_maskss, mapping = get_batch_job(batch_idx,istest=True)
            target_cost, target_cardinality, operatorss, extra_infoss, condition1ss, condition2ss, sampless, condition_maskss, mapping = torch.FloatTensor(target_cost), torch.FloatTensor(target_cardinality),torch.FloatTensor(operatorss),torch.FloatTensor(extra_infoss),torch.FloatTensor(condition1ss),torch.FloatTensor(condition2ss), torch.FloatTensor(sampless), torch.FloatTensor(condition_maskss), torch.FloatTensor(mapping)
            operatorss, extra_infoss, condition1ss, condition2ss, condition_maskss = operatorss.squeeze(0), extra_infoss.squeeze(0), condition1ss.squeeze(0), condition2ss.squeeze(0), condition_maskss.squeeze(0).unsqueeze(2)
            sampless = sampless.squeeze(0)
            mapping = mapping.squeeze(0)
            target_cost, target_cardinality, operatorss, extra_infoss, condition1ss, condition2ss = Variable(target_cost), Variable(target_cardinality), Variable(operatorss), Variable(extra_infoss), Variable(condition1ss), Variable(condition2ss)
            sampless = Variable(sampless)
            estimate_cost,estimate_cardinality = model(operatorss, extra_infoss, condition1ss, condition2ss, sampless, condition_maskss, mapping)
            target_cost = target_cost
            target_cardinality = target_cardinality
            cost_loss,cost_loss_median,cost_loss_max,cost_max_idx = qerror_loss(estimate_cost, target_cost, cost_label_min, cost_label_max)
            card_loss,card_loss_median,card_loss_max,card_max_idx = qerror_loss(estimate_cardinality, target_cardinality, card_label_min, card_label_max)
            # print (card_loss.item(),card_loss_median.item(),card_loss_max.item(),card_max_idx.item())
            loss = cost_loss + card_loss
            cost_loss_total += cost_loss.item()
            card_loss_total += card_loss.item()

            # 5. 收集数据 - 同时收集 cost 和 cardinality 的预测结果
            pred_cost = unnormalize(estimate_cost, cost_label_min, cost_label_max).detach().cpu().numpy().flatten()
            target_cost_real = unnormalize(target_cost, cost_label_min, cost_label_max).detach().cpu().numpy().flatten()
            pred_card = unnormalize(estimate_cardinality, card_label_min, card_label_max).detach().cpu().numpy().flatten()
            target_card_real = unnormalize(target_cardinality, card_label_min, card_label_max).detach().cpu().numpy().flatten()
            
            batch_n = len(pred_cost)
            batch_src = test_src_files[sample_offset: sample_offset + batch_n]
            sample_offset += batch_n
            
            # 收集 cost 预测结果
            for p, t, s in zip(pred_cost, target_cost_real, batch_src):
                q_err = max(float(p)/float(t), float(t)/float(p))
                final_results_cost.append({
                    'pred': float(p),
                    'cost_real': float(t),
                    'qerror': q_err,
                    'dataset': dataset,
                    'src_file': s
                })
            
            # 收集 cardinality 预测结果
            for p, t, s in zip(pred_card, target_card_real, batch_src):
                q_err = max(float(p)/float(t), float(t)/float(p))
                final_results_card.append({
                    'pred': float(p),
                    'card_real': float(t),
                    'qerror': q_err,
                    'dataset': dataset,
                    'src_file': s
                })
        
        # 保存 cost 预测结果
        if final_results_cost:
            df_cost = pd.DataFrame(final_results_cost)
            df_cost.to_csv('results/E2E/E2E-cost-job.csv', index=False)
            print(f"Saved {len(df_cost)} cost predictions to results/E2E/E2E-cost-job.csv")
        
        # 保存 cardinality 预测结果
        if final_results_card:
            df_card = pd.DataFrame(final_results_card)
            df_card.to_csv('results/E2E/E2E-card-job.csv', index=False)
            print(f"Saved {len(df_card)} cardinality predictions to results/E2E/E2E-card-job.csv")
        batch_num = validate_end - validate_start
        print("Epoch {}, validating cost loss: {}, validating card loss: {}".format(epoch, cost_loss_total/batch_num, card_loss_total/batch_num))

        # 画图
        val_cost_avg = cost_loss_total/batch_num
        val_card_avg = card_loss_total/batch_num
        print("Epoch {}, validating cost loss: {}, validating card loss: {}".format(epoch, val_cost_avg, val_card_avg))
        
        # 记录验证 loss
        val_cost_losses.append(val_cost_avg)
        val_card_losses.append(val_card_avg)
        epochs_list.append(epoch)
    
    # 训练结束后绘制 loss 曲线
    plt.figure(figsize=(12, 5))
    
    # 绘制 Cost Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs_list, train_cost_losses, 'b-', label='Training Cost Loss', marker='o')
    plt.plot(epochs_list, val_cost_losses, 'r-', label='Validation Cost Loss', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Q-Error Loss')
    plt.title('Cost Loss over Epochs')
    plt.legend()
    plt.grid(True)
    
    # 绘制 Cardinality Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs_list, train_card_losses, 'b-', label='Training Cardinality Loss', marker='o')
    plt.plot(epochs_list, val_card_losses, 'r-', label='Validation Cardinality Loss', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Q-Error Loss')
    plt.title('Cardinality Loss over Epochs')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('results/E2E/E2E-training_loss_curves.png', dpi=300, bbox_inches='tight')
    print("Training loss curves saved to 'results/E2E/E2E-training_loss_curves.png'")
    plt.show()

    end = time.time()
    print (end-start)
    return model

# from model.util import Normalizer
# from model.database_util import get_hist_file
# from model.model_transformer_job import QueryFormer
# from model.dataset_schema_gat import PlanTreeDataset
# from model.trainer_cost_job import train, evaluate
data, indexes_id, tables_id, columns_id, physic_ops_id, compare_ops_id, bool_ops_id = prepare_dataset('Learning-based-cost-estimator/test_files_open_source')
print ('data prepared')
word_vectors = load_dictionary('Learning-based-cost-estimator/test_files_open_source/wordvectors_updated.kv')
print ('word_vectors loaded')
min_max_column = load_numeric_min_max('Learning-based-cost-estimator/test_files_open_source/min_max_vals.json')
encoding_ckpt = torch.load('/data/yuanyihui/cypher-emb/data/job/encoding.pt')
encoding = encoding_ckpt['encoding'] # 把其中的 encoding 取出保存到变量 encoding。
min_max_column=encoding.column_min_max_vals
print ('min_max loaded')
index_total_num = len(indexes_id)
table_total_num = len(tables_id)
column_total_num = len(columns_id)
physic_op_total_num = len(physic_ops_id)
compare_ops_total_num = len(compare_ops_id)
bool_ops_total_num = len(bool_ops_id)
condition_op_dim = bool_ops_total_num + compare_ops_total_num+column_total_num+1000
condition_op_dim_pro = bool_ops_total_num + column_total_num + 3
plan_node_max_num, condition_max_num, cost_label_min, cost_label_max, card_label_min, card_label_max = obtain_upper_bound_query_size('data/job/plan_and_cost/train_by_para_v2_same_500.csv')
print ('query upper size prepared')

# encode_train_plan_seq_save('Learning-based-cost-estimator/test_files_open_source/plans_seq_sample.json')
encode_test_plan_seq_save('data/job/plan_and_cost/test_by_para_v2.csv')
# print ('data encoded')

print ('start training')
dataset='job'
# model = train(0, total_train_batches, total_train_batches, total_train_batches, 20)

model = train(0, 62, 0, 39, 150)