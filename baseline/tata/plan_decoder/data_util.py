import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from scipy.stats import pearsonr
import time
import os
import math

from torch.utils.data import DataLoader

def sort_plan(root):
    res = []
    def dfs(node):
        nonlocal res
        res.append(node.nodeType)
        if len(node.children) > 0:
            res.append('(')
            for child in node.children:
                dfs(child)
            #     res.append(';')
            # res = res[:-1]
            res.append(')')
    dfs(root)
    res.append('END')
    return res

class Encoding():
    def __init__(self, ds_info):
        self.node2idx = {}
        self.nodeTypes = ds_info.nodeTypes
        for i,nd in enumerate(ds_info.nodeTypes):
            self.node2idx[nd] = i
        offset = len(self.node2idx)
        self.syntax2idx = {
            '(':offset, 
            ')':offset+1,
            'END':offset+2
        }

    def encode_plan(self, plan):
        flat_plan = sort_plan(plan)
        out = []
        for ele in flat_plan:
            if ele in self.node2idx:
                out.append(self.node2idx[ele])
            else:
                out.append(self.syntax2idx[ele])
        return out

    def decode_plan(self, idxs):
        idx2node = {value: key for key, value in self.node2idx.items()}
        idx2syntax = {value: key for key, value in self.syntax2idx.items()}

        out = []
        for idx in idxs:
            if idx in idx2node:
                out.append(idx2node[idx])
            else:
                out.append(idx2syntax[idx])
        return out           


from torch.utils.data import Dataset
from plan_decoder.data_util import Encoding
class DecoderDataset(Dataset):
    def __init__(self, roots, encoding):
        self.encoding = encoding
        self.roots = roots
        self.plans = [encoding.encode_plan(root) for root in roots]
        # self.lens = [len(plan) for plan in self.plans]
    
    def __len__(self):
        return len(self.roots)
    
    def __getitem__(self,idx):
        return self.plans[idx]#, self.lens[idx]

class Batch():
    def __init__(self, targets, lens):
        self.targets = targets
        self.lens = lens

    def to(self, device):
        self.targets = self.targets.to(device)
#         self.feat_lens = self.feat_lens.to(device)
#         self.node_lens = self.node_lens.to(device)
        return self

def collate(small_set):
    lens = [len(lst) for lst in small_set]
    max_len = max(lens)
    padded_indexes = [lst + [0] * (max_len - len(lst)) for lst in small_set]
    return Batch(torch.LongTensor(padded_indexes), torch.LongTensor(lens))


from torch.utils.data import Sampler
class SeededSampler(Sampler):
    def __init__(self, length, base_seed = 42):
        self.length = length
        self.base_seed = base_seed
        self.epoch_offset = 0

    def __iter__(self):
        np.random.seed(self.base_seed + self.epoch_offset)
        indices = np.random.permutation(self.length)
        return iter(indices.tolist())

    def __len__(self):
        return len(self.data_source)

    def set_epoch(self, epoch):
        self.epoch_offset = epoch

        