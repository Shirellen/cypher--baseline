import psycopg2
import time
import pandas as pd
import csv
import json
import os
import collections, re
import glob

import sys
sys.path.append('../')
from data_util.utils import *
import torch
from data_util.imdb import get_min_max
from data_util.imdb import get_imdb_pretrain_data
from data_util.utils import get_ds_info

data_set = get_imdb_pretrain_data('../data/imdb/')
ds_info = get_ds_info('../data/imdb/')

from data_enrich.opr_costing import CostFormula, print_qerror, get_abs_errors

cost_form = CostFormula(ds_info)
cost_form.fit_workload(stats_set['train_roots'])
cost_form.fit_all()


predictions = [cost_form.predict_plan_cost(a, True) for a in stats_set['train_roots']]
print_qerror(predictions, stats_set['train_costs'])
get_abs_errors(predictions, stats_set['train_costs'])


predictions = [cost_form.predict_plan_cost(a, True) for a in stats_set['test_roots']]
print_qerror(predictions, stats_set['test_costs'])
get_abs_errors(predictions, stats_set['test_costs'])

torch.save({'cost_form': cost_form}, 'imdb_pretrain_cost_form.pth')