import pandas as pd
import numpy as np
import json 
from .feature_extractor import traversePlan
import time

## collate labels
CLIP_VAL = 1e4
CUT_OFF = 0.2
IMPROVE = 1
REGRESS = 0
NODIFF = 2

def collate_labels(df):
    labels = []
    pair_diffs = []
    pair_diff_ratios = []
    for idx, row in df.iterrows():

        relative_time = row['Right_Cost'] / row['Left_Cost']
        if relative_time > (1+CUT_OFF):
            labels.append(REGRESS)
        elif relative_time < (1-CUT_OFF):
            labels.append(IMPROVE)
        else: 
            labels.append(NODIFF)
    
    return labels


## splitting

def split_grouped_ids(raw_df, threshold):
    # group dataset by their plan ids, and split into train ids and test ids by given threshold
    add_plan_id(raw_df)
    #### need to do with imdb's query id
    template2plan_id = dict(raw_df.groupby('qid')['Plan_id'].unique())
    train_plan_ids = set()
    test_plan_ids = set()
    for k, v in template2plan_id.items():
        if len(v) <= threshold:
            train_plan_ids.update(v)
        else:
            train_plan_ids.update(v[:len(v)*(threshold-1)//threshold])
            test_plan_ids.update(v[len(v)*(threshold-1)//threshold:])
    data_raw_train_ids = set(raw_df.loc[raw_df['Plan_id'].isin(train_plan_ids)].index)
    data_raw_test_ids = set(raw_df.loc[raw_df['Plan_id'].isin(test_plan_ids)].index)
    return data_raw_train_ids, data_raw_test_ids

def split_train_test(df, raw_df, method = 'query', threshold = 5, qid = None):
    np.random.seed(42)
    length = len(df)
    if method == 'pair':
        order = np.random.permutation(length)
        train_idxs = order[:length*(threshold-1)//threshold]
        test_idxs = order[length*(threshold-1)//threshold:]
        train_pairs = df.loc[train_idxs]
        test_pairs = df.loc[test_idxs]
    
    elif method == 'query':
        if qid is not None:
            test_run_id = qid
            test_pairs = df.loc[df['qid'].isin(test_run_id)]
            train_pairs = df.loc[~df['qid'].isin(test_run_id)]
        else:
            max_run_id = max(df['Query_id'])
            order = np.random.permutation(max_run_id)
            train_run_id = order[:max_run_id*(threshold-1)//threshold]
            test_run_id = order[max_run_id*(threshold-1)//threshold:]
            train_pairs = df.loc[df['Query_id'].isin(train_run_id)]
            test_pairs = df.loc[df['Query_id'].isin(test_run_id)]
    
    elif method == 'plan':
        data_raw_train_ids, data_raw_test_ids = split_grouped_ids(raw_df, threshold)
        train_pairs = df.loc[
            (df['Left'].isin(data_raw_train_ids)) & 
            df['Right'].isin(data_raw_train_ids)
        ]
        test_pairs = df.loc[
            ((df['Left'].isin(data_raw_train_ids)) &
            (df['Right'].isin(data_raw_test_ids))) |
            ((df['Right'].isin(data_raw_train_ids)) &
            (df['Left'].isin(data_raw_test_ids)))            
        ]
  
    else:
        raise NotImplementedError('unknown method')
    return train_pairs, test_pairs


def df2nodes(df):
    t0 = time.time()
    idxs = []
    roots = []
    js_nodes = []
    for i, row in df.iterrows():
        
        idx = i
        js_str = row['plan']
            
        if js_str == 'failed':
            continue
        js_node = json.loads(js_str)
        js_nodes.append(js_node)
        root = traversePlan(js_node)
        roots.append(root)
        idxs.append(idx)
    print('length: ', len(idxs), ', Time: ',  time.time()-t0)
    return roots, js_nodes, idxs
