
import random
from .utils import traversePlan
import pandas as pd
import json

def get_planss(pat = 'imdb/bao/job/', keyword='plan'):
    df_list = []
    for arm in range(49):
        df_list.append(pd.read_csv(pat + 'arm{}/collated_plan.csv'.format(arm)))

    # costss = []
    planss = []
    for i in range(49):
        pls = [json.loads(plan) for plan in df_list[i][keyword]]
        
        ## overwrite with average time across runs 
        costs = df_list[i]['Avg Cost'].tolist()
        for j in range(len(pls)):
            pls[j]['Execution Time'] = costs[j]

        planss.append(pls)
        # costss.append(df_list[i]['Avg Cost'].tolist())

    return planss#, costss



def get_prl(some_planss):
    # assert(len(some_planss)==49)
    planss = some_planss
    latencies = []
    rootss = []
    for i in range(len(some_planss)):
        plans = planss[i]
        roots = [traversePlan(pl) for pl in plans]
        rootss.append(roots)
        latency = [plan['Execution Time'] for plan in plans]
        latencies.append(latency)
    return planss, rootss, latencies


def sample_train_test(planss, sql_identifiers, test_group, train_size, seed):
    full_size = len(planss[0])
    test_ids = [sql_identifiers.index(a) for a in test_group]
    other_ids = [a for a in list(range(full_size)) if a not in test_ids]
    
    random.seed(seed)
    assert(train_size <= len(other_ids))
    train_ids = random.sample(other_ids, train_size)
    train_planss = []
    for i in range(49):
        train_planss.append([])
        for j in train_ids:
            train_planss[i].append(planss[i][j])
    train_set = get_prl(train_planss)

    test_planss = []
    for i in range(49):
        test_planss.append([])
        for j in test_ids:
            test_planss[i].append(planss[i][j])
    test_set = get_prl(test_planss)

    return train_set, test_set