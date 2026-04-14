import numpy as np
from sklearn.linear_model import LinearRegression, Lasso

def get_features(node, context):
    # Retrieve the feature function based on the node type
    feature_func = feature_functions.get(node['nodeType'])
    if feature_func:
        return feature_func(node, context)
    else:
        # Handle unknown node type or return default features
        return default_feature(node, context)

def default_feature(node, context):
    # Default feature extractor if node type is unknown
    if len(node['cards']) == 0:
        return (node['card'],)
    elif len(node['cards']) == 1:
        return (
            node['card'], 
            node['cards'][0]
        )
    else:
        return (
            node['card'], 
            node['cards'][0],
            node['cards'][1]
        )    


def nest_loop_feature(node, context):
    return (
        node['cards'][0] * node['cards'][1], 
        node['cards'][0], 
        node['cards'][1]
        )

def gather_feature(node, context):
    return (
        # node['cards'][0] * node['cards'][1], 
        node['cards'][0], 
        # node['cards'][1]
        )

def merge_join_feature(node, context):
    # linear for join, NlogN for sorting
    return (
        node['cards'][0] + node['cards'][1],
        node['cards'][0] * np.log(node['cards'][0]+1),
        node['cards'][1] * np.log(node['cards'][1]+1),
    )

def hash_join_feature(node, context):
    # linear for hash and probe
    return (
        node['cards'][0],
        node['cards'][1],
    )


# TODO: per-table record, if not found, use general
def seq_scan_feature(node, ds_info):
    # relation size
    return (
        ds_info.table_size[node['table']],
    )

def index_scan_feature(node, context):
    # relation size
    return (
        node['card']
    )


feature_functions = {
    'Nested Loop': nest_loop_feature,
    'Merge Join': merge_join_feature,
    'Hash Join': hash_join_feature,
    'Seq Scan': seq_scan_feature,
    'Gather' : gather_feature,
}


## use card_est
def extract_node_inference(node):
    """ Traverse the node tree and collect information in a structured form. """
    node_data = {
        'nodeType': node.nodeType,
        'cost': node.cost,
        'card': node.card_est,
        'cards': [],
        'costs': [],
        'table': getattr(node, 'table', None)  # Handle nodes without 'table' attribute safely
    }

    if len(node.children) > 0:
        for child in node.children:
            node_data['cards'].append(child.card_est)
            node_data['costs'].append(child.cost)

    return node_data

## use card
def extract_node(node):
    """ Traverse the node tree and collect information in a structured form. """
    node_data = {
        'nodeType': node.nodeType,
        'cost': node.cost,
        'card': node.card,
        'cards': [],
        'costs': [],
        'table': getattr(node, 'table', None)  # Handle nodes without 'table' attribute safely
    }

    if len(node.children) > 0:
        for child in node.children:
            node_data['cards'].append(child.card)
            node_data['costs'].append(child.cost)

    return node_data



from scipy.stats import pearsonr
def print_qerror(ps, ls, prints=True):
    ps = np.array(ps)+0.001
    ls = np.array(ls)+0.001
    qerror = []
    for i in range(len(ps)):
        if ps[i] > float(ls[i]):
            qerror.append(float(ps[i]) / float(ls[i]))
        else:
            qerror.append(float(ls[i]) / float(ps[i]))

    e_50, e_90, e_95, e_99, e_max = np.median(qerror), np.percentile(qerror,90), \
                np.percentile(qerror,95), np.percentile(qerror,99), np.max(qerror)
    e_mean = np.mean(qerror)
    res = {
        'q_median' : e_50,
        'q_90' : e_90,
        'q_95' : e_95,
        'q_99' : e_99,
        'q_max' : e_max,
        'q_mean' : e_mean,
    }
    if prints:
        print("Median: {}".format(e_50))
        print("90th percentile: {}".format(e_90))
        print("Max: {}".format(e_max))
        print("Mean: {}".format(e_mean))

    return res


def get_abs_errors(ps, ls): # unnormalised
    ps = np.array(ps).flatten()+0.001
#     if len(set(ps)) == 1:
#         print(ps, ls)
    ls = np.array(ls).flatten()+0.001
    abs_diff = np.abs(ps-ls)

    if not np.isfinite(ps).all(): ## contains nan or inf
#         print(ps, ls)
        corr = np.nan
        log_corr = np.nan
    else:  
        corr, _ = pearsonr(ps, ls)
#         if np.isnan(corr):
#             print(ps, ls, corr)
        # log_corr, _ = pearsonr(np.log(ps), np.log(ls))
    res = {
        'rmse' : (abs_diff ** 2).mean() ** 0.5,
        'corr' : corr,
        # 'log_corr': log_corr,
        'abs_median' : np.median(abs_diff),
        'abs_90' : np.percentile(abs_diff, 90),
        'abs_95' : np.percentile(abs_diff, 95),
        'abs_99' : np.percentile(abs_diff, 99),
        'abs_max' : np.max(abs_diff)
    }

    # print("abs err: ", (e_50, e_90, e_95, e_99, e_max))
    return res

class CostFormula():
    def __init__(self, ds_info):

        self.ds_info = ds_info
        self.node_dict = {}
        self.models = {}

    ## TODO whether to use actual card, or card_est
    ## blocking and nonblocking

    def fit_all(self):
        for node_type in self.node_dict.keys():
            print(node_type)
            self.fit_type(node_type)
    
    def fit_type(self, nodeType='Gather'):
        node_history = self.node_dict[nodeType]
        X = [get_features(node,self.ds_info) for node in node_history]
        y = []
        for node in node_history:
            c = node['cost']
            for cc in node['costs']:
                c -= cc
            y.append(c)
        lr = LinearRegression()
        # lr = Lasso(alpha=0.1)
        lr.fit(X,y)
        predictions = lr.predict(X)
        self.models[nodeType] = lr
        print_qerror(predictions, y)
        abs_err = get_abs_errors(predictions, y)
        print(abs_err)
        return X, y, predictions, abs_err

    def fit_workload(self, plans):
        node_dict = {}

        def dfs(node):
            nonlocal node_dict
            if node.nodeType not in node_dict:
                node_dict[node.nodeType] = []

            node_dict[node.nodeType].append(extract_node(node))
            if len(node.children) > 0:
                for child in node.children:
                    dfs(child)
            return
        for plan in plans:
            dfs(plan)

        self.node_dict = node_dict
    

    def predict_node_cost(self, node, use_est=False):
        if node is None:
            return 0

        ## TODO
        if use_est:
            nd = extract_node_inference(node)
        else:
            nd = extract_node(node)
        features = get_features(nd, self.ds_info)
        if node.nodeType in self.models:
            model = self.models[node.nodeType]
            cost = model.predict([features])[0]
        else:
            cost = 0
        return max(0, cost)

    def predict_plan_cost(self, plan, use_est=False):

        def dfs(node):
            node_cost = self.predict_node_cost(node, use_est)
            if len(node.children)>0:
                for child in node.children:
                    node_cost += dfs(child)
            return node_cost

        return max(0, dfs(plan))

    def inference(self, root):
        pass
