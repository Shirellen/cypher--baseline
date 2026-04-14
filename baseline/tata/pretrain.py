from data_util.utils import *
import torch
from data_util.imdb import get_min_max
from data_util.imdb import get_imdb_pretrain_data, get_ds_info

data_set = get_imdb_pretrain_data('../data/imdb/')
data_set['ds_info'] = get_ds_info('../data/imdb/')
ds_info = data_set['ds_info']

from algorithms.bao.hyperparams import get_hyperparams
from algorithms.bao.featurize import collate, construct_loader, construct_seeded_loader
from algorithms.bao.net import BaoNet
from training.utils import seed_all

class Args:
    device = 'cuda'
    task_name = 'imdb'
    result_path = '../results/cost/joint_pretrain/imdb/'
    model_path = '../saved_models/joint_pretrain/imdb/'
    method = 'bao'

    lambdaa = 0.5 # joint_train
    
    
    def set_attr(self, d):
        for key, value in d.items():
            setattr(self, key, value)
            
args = Args()
args.set_attr(get_hyperparams('imdb'))
import os
if not os.path.exists(args.result_path):
    os.makedirs(args.result_path)
if not os.path.exists(args.model_path):
    os.makedirs(args.model_path)

seed_all(42)

for lambdaa in lambdas:
    args.lambdaa = lambdaa
    
    seed_all(42)
    
    from algorithms.bao.featurize import TreeFeaturizer
    tree_transform = TreeFeaturizer()
    tree_transform.fit_from_ds_info(ds_info)
    
    import torch
    get_loader = construct_loader(ds_info, tree_transform)
    # train_loader = get_loader(data_set['train_roots'], data_set['train_costs'], args.bs, shuffle=True)
    val_loader = get_loader(data_set['val_roots'], data_set['val_costs'], args.bs, shuffle=False)
    
    from training.cost_est import Prediction, evaluate
    from training.pretrain import pretrain_joint
    import scipy
    prediction = Prediction(args.hid)
    
    from torch import nn
    device = args.device
    hid = args.hid
    
    rep_model = BaoNet(args.in_channel)
    
    from plan_decoder.data_util import SeededSampler
    from algorithms.bao.featurize import construct_seeded_loader
    sampler = SeededSampler(len(data_set['train_roots']))
    get_seed_loader = construct_seeded_loader(ds_info, tree_transform)
    encoder_loader = get_seed_loader(data_set['train_roots'], data_set['train_costs'], sampler, args.bs)
    
    from plan_decoder.data_util import Encoding, DecoderDataset, collate as decoder_collate
    from torch.utils.data import DataLoader
    encoding = Encoding(ds_info)
    dds = DecoderDataset(data_set['train_roots'], encoding)
    decoder_loader = DataLoader(dds, batch_size=args.bs, sampler=sampler, collate_fn=decoder_collate)
    
    from plan_decoder.model import PlanDecoder
    rep_model = BaoNet(args.in_channel).to(args.device)
    decoder = PlanDecoder().to(args.device)
    
    models = pretrain_joint(rep_model, prediction, decoder, encoder_loader, decoder_loader,
                val_loader, sampler, data_set['val_costs'], ds_info, args,
               record=True)

   