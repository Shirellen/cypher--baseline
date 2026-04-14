from collections import Counter, defaultdict
import pandas as pd    
import numpy as np 

def extract_root(root):
    result = {
        "operators": [],
        "columns": [],
        "tables": [],
        "number_of_nodes": 0,
        "max_depth": 0,
    }
    
    def dfs(nd, depth):
        nonlocal result
        result["number_of_nodes"] += 1
        result["max_depth"] = max(result["max_depth"], depth)
        result["operators"].append(nd.nodeType)
        if nd.table:
            result["tables"].append(nd.table)
        for filter in nd.filters:
            result["columns"].append(filter[0])
        for child in nd.children:
            dfs(child, depth + 1)

    dfs(root, 0)
    return result

def extract_workload(features):
    plen = len(features)

    combined_operator_freqs = Counter()
    combined_column_freqs = Counter()
    combined_table_freqs = Counter()
    depths = []
    node_counts = []

    for result in features:
        operators = result["operators"]
        columns = result["columns"]
        tables = result["tables"]
        max_depth = result["max_depth"]
        number_of_nodes = result["number_of_nodes"]

        # Update frequency counts 
        ## count once / n times for each plan
        # combined_operator_freqs += Counter(operators)
        # combined_column_freqs += Counter(columns)
        # combined_table_freqs += Counter(tables)
        
        combined_operator_freqs += Counter(set(operators))
        combined_column_freqs += Counter(set(columns))
        combined_table_freqs += Counter(set(tables))
        
        depths.append(max_depth)
        node_counts.append(number_of_nodes)

    def normalize_counter(counter):
        return {k: v / plen for k, v in counter.items()}
    
    normalized_operator_freqs = {k: v / plen for k, v in combined_operator_freqs.items()}
    normalized_column_freqs = {k: v / plen for k, v in combined_column_freqs.items()}
    normalized_table_freqs = {k: v / plen for k, v in combined_table_freqs.items()}
    normalized_depths = [depth for depth in depths]
    normalized_node_counts = [node_count for node_count in node_counts]

    distribution = {
        "full_features": features,
        "operators": normalized_operator_freqs,
        "columns": normalized_column_freqs,
        "tables": normalized_table_freqs,
        "depths": normalized_depths,
        "node_counts": normalized_node_counts,
    }

    return distribution

def workload_diff(base_dist, new_dist, threshold = 0.1):
    columns = {}
    tables = {}

    base_cols = base_dist['columns']
    new_cols = new_dist['columns']

    col_diffs = []
    for col, freq in base_cols.items():
        if col not in new_cols:
            columns[col] = freq
            col_diffs.append(freq)
        else:
            diff = freq - new_cols[col]
            if diff > threshold:
                columns[col] = diff
                col_diffs.append(diff)
                
    base_tabs = base_dist['tables']
    new_tabs = new_dist['tables']

    tab_diffs = []
    for tab, freq in base_tabs.items():
        if tab not in new_tabs:
            tables[tab] = freq
            tab_diffs.append(freq)
        else:
            diff = freq - new_tabs[tab]
            if diff > threshold:
                tables[tab] = diff
                tab_diffs.append(diff)
    
    return {
        'columns':columns,
        'tables':tables,
    }

def score_query(feature, requirements):
    score = 0
    for cat,content in requirements.items():
        # print(cat)
        desired_set = set(content.keys())
        # print(desired_set.intersection(feature[cat]))
        score += len(desired_set.intersection(feature[cat])) ** 2
    return score

def requirement_satisfied(requirements):
    for cat, content in requirements.items():
        if len(content) > 0:
            return False
    return True

import numpy as np
def sample_queries(features, requirements, number=5, seed = 42, iteration = 0):
    if requirement_satisfied(requirements):
        print('Matched at iter: {}'.format(iteration))
        return random_sample_queries(features, 
            number, seed+iteration)
    scores = np.array([score_query(feature, requirements) for feature in features])
    probs = scores / np.sum(scores)
    np.random.seed(seed)
    sels = np.random.choice(list(range(len(features))), size = number, replace = False, p=probs)
    return sels


def random_sample_queries(features, number, seed):
    np.random.seed(seed)
    return np.random.choice(list(range(len(features))), 
        size = number, replace = False)

def sample_iters(base_features, target_features, iterations=10, num_per_iter=3, threshold=0.05, to_print=False):
    ## base: to sample queries from
    ## target: the small one
    new_features = target_features.copy()
    new_dist = extract_workload(new_features)
    base_dist = extract_workload(base_features)
    requirements = workload_diff(base_dist, new_dist, threshold)
    if to_print:
        print('Init Requirement: ', requirements)

    to_add_ids = []
    for iter in range(iterations):
        sample_ids = sample_queries(base_features, requirements, 
            number=num_per_iter, seed = 42, iteration=iter)
        to_add_ids.extend(sample_ids)
        new_features.extend([base_features[i] for i in sample_ids])
        new_dist = extract_workload(new_features)
        requirements = workload_diff(base_dist, new_dist, threshold)
        # print(requirements)

    return to_add_ids