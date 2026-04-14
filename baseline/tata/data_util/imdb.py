import glob, re
import pandas as pd
import json
from .utils import df2nodes, get_costs, traversePlan, get_ds_info
from .feature_extractor import DatasetInfo
import numpy as np

## using adaptive vs learned paper queries
job_slow_queries = [
    '16b.sql', '17a.sql', '17e.sql', '17f.sql', '17b.sql', '19d.sql', '17d.sql',
    '17c.sql', '10c.sql', '26c.sql', '25c.sql', '6d.sql', '6f.sql', '8c.sql',
    '18c.sql', '9d.sql', '30a.sql', '19c.sql', '20a.sql'
]
job_rand_queries = [
    '8a.sql', '16a.sql', '2a.sql', '30c.sql', '17e.sql', '20a.sql', '26b.sql',
    '12b.sql', '15b.sql', '15d.sql', '10b.sql', '15a.sql', '4c.sql', '4b.sql',
    '22b.sql', '17c.sql', '24b.sql', '10a.sql', '22c.sql'
]

all_JOB_queries = [
    '30a.sql', '1d.sql', '25a.sql', '6b.sql', '19c.sql', '28b.sql', 
    '9b.sql', '33b.sql', '32a.sql', '27a.sql', '20b.sql', '17d.sql', 
    '16b.sql', '10b.sql', '6a.sql', '17b.sql', '25b.sql', '8b.sql', 
    '31b.sql', '33c.sql', '23a.sql', '15a.sql', '21b.sql', '11d.sql', 
    '9a.sql', '6d.sql', '24a.sql', '17e.sql', '19a.sql', '1c.sql', 
    '25c.sql', '31a.sql', '23c.sql', '17f.sql', '19b.sql', '9d.sql', 
    '27b.sql', '11b.sql', '7b.sql', '19d.sql', '1a.sql', '11c.sql', 
    '31c.sql', '28a.sql', '3c.sql', '6c.sql', '26b.sql', '16a.sql', 
    '7a.sql', '29b.sql', '29a.sql', '1b.sql', '6e.sql', '13d.sql', 
    '10c.sql', '26a.sql', '4b.sql', '10a.sql', '15b.sql', '15d.sql', 
    '26c.sql', '14b.sql', '15c.sql', '14c.sql', '13b.sql', '3a.sql', 
    '24b.sql', '5a.sql', '8c.sql', '6f.sql', '8a.sql', '27c.sql', 
    '12a.sql', '22a.sql', '17a.sql', '13a.sql', '21a.sql', '12b.sql', 
    '13c.sql', '8d.sql', '21c.sql', '7c.sql', '2a.sql', '3b.sql', 
    '16c.sql', '9c.sql', '32b.sql', '28c.sql', '33a.sql', '11a.sql', 
    '18a.sql', '5c.sql', '22d.sql', '18c.sql', '5b.sql', '2c.sql', 
    '16d.sql', '4a.sql', '22c.sql', '12c.sql', '29c.sql', '30b.sql', 
    '2d.sql', '14a.sql', '17c.sql', '22b.sql', '30c.sql', '20a.sql', 
    '20c.sql', '2b.sql', '4c.sql', '18b.sql', '23b.sql'
]



def get_job_queries(pat = 'imdb/bao/job_queries/'):
    sql_files = glob.glob(pat + '*.sql')  # Adjust path
    # sql_identifiers = [a.split('\\')[-1] for a in sql_files]
    sql_identifiers = [re.split(r'[\\/]', a)[-1] for a in sql_files]
    sql_queries = []
    
    for i, sql_file in enumerate(sql_files):
        with open(sql_file) as f:
            sql_query = f.read()
            sql_queries.append(sql_query)
    return sql_identifiers, sql_queries


from .utils import get_col_min_max
def get_min_max(pat = 'imdb/'):
    minmax = pd.read_csv(pat+ 'column_min_max_vals.csv')
    col_min_max = get_col_min_max(minmax)
    return col_min_max


def get_imdb_pretrain_data(dat_path='../imdb/', seed=42, frac = 0.3):
    files = sorted(glob.glob(dat_path+'job_pqo_run/collated_*'))
    seeded_gen = np.random.default_rng(seed)
    df = pd.DataFrame()
    for file in files:
        tmp_df = pd.read_csv(file).sample(frac=frac, random_state = seeded_gen)
        df = pd.concat([df,tmp_df])
    # df.reset_index(drop=True, inplace=True)

    df = df.sample(frac=1, random_state=seeded_gen).reset_index()

    roots, js_nodes, idxs = df2nodes(df, 'plan')
    costs = df['Avg Cost'].tolist()
    # costs = get_costs(js_nodes)

    minmax = pd.read_csv(dat_path + 'column_min_max_vals.csv')
    col_min_max = get_col_min_max(minmax)
    ds_info = DatasetInfo({})
    ds_info.construct_from_plans(roots)
    ds_info.get_columns(col_min_max)

    split_point = len(df)//5*4
    train_roots, val_roots = roots[:split_point], roots[split_point:]
    train_costs, val_costs = costs[:split_point], costs[split_point:]
    train_js_nodes, val_js_nodes = js_nodes[:split_point], js_nodes[split_point:]


    # original JOB
    # planss = get_planss(dat_path + 'server/job_direct_run/collated_plans.csv')
    df_tmp = pd.read_csv(dat_path + 'server/job_direct_run/collated_plans.csv')
    js_nodes = [json.loads(p) for p in df_tmp['plan']]
    job_roots = [traversePlan(js_node) for js_node in js_nodes]
    # job_costs = get_costs(js_nodes)
    job_costs = df_tmp['Avg Cost'].tolist()

    return {
        'ds_info' : ds_info,
        'data_raw' : df,
        'col_min_max' : col_min_max,
        'total_roots' : roots,
        'total_costs' : costs,
        'train_roots' : train_roots,
        'train_costs' : train_costs,
        'train_js_nodes' : train_js_nodes,
        'val_roots' : val_roots,
        'val_costs' : val_costs,
        'val_js_nodes' : val_js_nodes,
        'test_roots': job_roots,
        'test_costs': job_costs
    }