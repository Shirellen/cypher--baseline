import pandas as pd
import numpy as np

from algorithms.bao.hyperparams import get_hyperparams
from algorithms.bao.featurize import collate, construct_loader, construct_seeded_loader
from algorithms.bao.net import BaoNet
from training.utils import seed_all
import torch
import torch.nn as nn
from training.bao_optimizer import BanditOptimizer

class Args:
    device = 'cuda:0'
    task_name = 'imdb_bao'

    method = 'bao'
    result_path = 'results/bao/bao/imdb_enriched/'
    
    
    def set_attr(self, d):
        for key, value in d.items():
            setattr(self, key, value)
            
args = Args()
args.set_attr(get_hyperparams('imdb'))
import os

from data_util.utils import get_ds_info
from data_util.imdb import get_imdb_pretrain_data
ds_info = get_ds_info('../data/imdb/')
imdb_set = get_imdb_pretrain_data('../data/imdb/')

from algorithms.bao.featurize import TreeFeaturizer
tree_transform = TreeFeaturizer()
tree_transform.fit_from_ds_info(ds_info)

from training.bao_optimizer import Prediction, train, evaluate

from data_util.imdb import get_min_max
from data_util.bao_data import get_planss

planss = get_planss('../imdb/job_bao_run/')

from data_util.imdb import get_job_queries
sql_identifiers, sql_queries = get_job_queries('../data/imdb/')

col_min_max = get_min_max('../data/imdb/')

from data_util.bao_data import get_prl
from data_util.bao_data import sample_train_test


from data_util.imdb import job_rand_queries

train_sizes = [2,3,4,5,10,15] 
num_per_iter = 3
threshold = 0.05

from data_enrich.sample_gen import extract_root, sample_iters
pretrain_features = [extract_root(root) for root in stats_set['train_roots']]

from data_enrich.dbms_util import imdb_config, get_bao_plans
from data_util.feature_extractor import traversePlan

from data_enrich.opr_costing import CostFormula
checkpoint = torch.load('../data_enrich/imdb_pretrain_cost_form.pth')
cost_form = checkpoint['cost_form']

pretrain = 'joint_pretrain'
log_file = pd.read_csv(f'../results/cost/{pretrain}/imdb/log.csv')
lambdaas = log_file['lambdaa'].tolist()
print(lambdaas)
checkpoint_file = log_file.loc[log_file['lambdaa']==0.5]['model'].item()
checkpoint = torch.load('../saved_models/joint_pretrain/imdb/' + checkpoint_file)


def one_round(train_size, lambdaa, iteration, freeze, log_file, seed = 42):
    split_seed = 42
    train_set, test_set = sample_train_test(planss, sql_identifiers, 
                                            job_rand_queries, train_size, split_seed)

    ## DATA ENRICH
    # sample and add data
    train_roots = [item for sublist in train_set[1] for item in sublist]
    train_latencies = [item for sublist in train_set[2] for item in sublist]
    train_features = [extract_root(root) for root in train_roots]
    to_add_ids = sample_iters(pretrain_features, train_features, iteration, num_per_iter=num_per_iter, threshold=threshold)
    to_add_sqls = [stats_set['data_raw']['sql'][idx] for idx in to_add_ids]
    to_add_sqls = [to_add_sql[2:-1].replace("\\n", " ") for to_add_sql in to_add_sqls]

    
    # get plans
    plans = get_bao_plans(imdb_config, to_add_sqls)
    added_roots = [traversePlan(plan) for plan in plans]
    
    # make pseudo labels
    pseudo_labels = [cost_form.predict_plan_cost(a, True) for a in added_roots]

    ## DATA ENRICH END

    ## LOAD MODEL
    checkpoint_file = log_file.loc[log_file['lambdaa']==lambdaa]['model'].item()
    checkpoint = torch.load('../saved_models/joint_pretrain/imdb/' + checkpoint_file)
    args = checkpoint['args']
    rep_model = BaoNet(args.in_channel).to(args.device)
    rep_model.load_state_dict(checkpoint['model0'])
    prediction = Prediction(args.hid).to(args.device)
    prediction.load_state_dict(checkpoint['model1'])
    ## LOAD MODEL END
    
    
    ## training loop
    seed_all(seed)
    get_loader = construct_loader(ds_info, tree_transform)
    enriched_train_loader = get_loader(train_roots+added_roots, \
                train_latencies+pseudo_labels, args.bs, shuffle=True)
    val_loader = get_loader(train_roots, train_latencies, args.bs, shuffle=False)

    ress = []
    
    args.epochs = 100
    model = nn.Sequential(rep_model, prediction)
    if freeze:
        for param in rep_model.parameters():
            param.requires_grad = False
        optimizer = torch.optim.Adam(prediction.parameters(), lr=args.lr)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    model = train(model, enriched_train_loader, val_loader, train_latencies, ds_info, args, record=False, prints=False, optimizer = optimizer)
    
    bo_train = BanditOptimizer(train_set[0],train_set[1], train_set[2])
    bo_test = BanditOptimizer(test_set[0],test_set[1], test_set[2])
    
    from algorithms.bao.featurize import construct_batch
    get_loader = construct_loader(ds_info, tree_transform)
    get_batch = construct_batch(get_loader,args)
    
    selections, best_time, post_time, sel_time = bo_train.select_plans(model,get_batch)
    res = {
        'from': checkpoint_file,
        'train_size': train_size,
        'epochs': epochs,
        'train_best_time': best_time,
        'train_post_time': post_time,
        'train_sel_time': sel_time
    }
    selections, best_time, post_time, sel_time = bo_test.select_plans(model,get_batch)
    res['test_best_time'] = best_time
    res['test_post_time'] = post_time
    res['test_sel_time'] = sel_time

    res['lambdaa'] = lambdaa
    res['enrich_size'] = iteration * num_per_iter
    res['freeze'] = freeze


    return res

train_size = 3
lambdaa = 0.6
iteration = 5
freeze = False
res = one_round(train_size, lambdaa, iteration, \
                        freeze, log_file, 42)