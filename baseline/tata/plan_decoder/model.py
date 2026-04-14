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

from .model_util import *

class PlanDecoder(nn.Module):
    def __init__(self, types = 20, hidden_size = 256,
                 mid_size = 32, out_size = 64, input_size = 64,
                 type_size = 64, syntax_embed = 64):
        super(PlanDecoder, self).__init__()

        self.out_size = out_size
        self.node_type_embed = nn.Embedding(types, type_size)
        # (, ), END
        # self.syntax_embed = nn.Embedding(5, syntax_embed)

        ## Multihead-attn or Linear
        self.apply_action = lambda x: F.linear(x, self.node_type_embed.weight)
        # self.apply_action = MultiHeadAttention(
        #     out_size, out_size, 
        #     out_size, out_size,
        #     num_heads=4, feat_drop=0.1)
        
        self.get_h0 = nn.Sequential(
            nn.Linear(input_size, mid_size),
            nn.Linear(mid_size, out_size)
        )
        # self.encoding_emb = nn.Linear(input_size, hidden_size)
        self.encoding_emb = nn.Linear(input_size, input_size)

        self.context_attn = MultiHeadAttention(out_size, out_size, 
            out_size, out_size,
            num_heads=4, feat_drop=0.1)

        self.decoder_lstm = ONLSTM(out_size, out_size, 
            num_layers=1, chunk_num=4, 
            dropout=0.1, dropconnect=0.2)
        
        self.att_vec_linear = nn.Sequential(
            nn.Linear(out_size + out_size, 
            out_size), nn.Tanh())

    
    def pool(self, encodings):
        pass
    
    # batch of 1
    def beam_search(self, encoding, beam_size = 5):
        pass

    def score(self, encodings, targets, target_lengths):
        batch_size = encodings.size(0)

        # labels = self.node_type_embed(targets) ## use the same first
        encs = self.encoding_emb(encodings).view(batch_size, -1, self.out_size)
        h0 = self.get_h0(encodings)
        # print(encs.size(), h0.size())
        context_0, _ = self.context_attn(encs, h0)
        # print(context_0.size())
        ## steps at len(target) first, actually is max len
        h_c = (h0, h0.new_zeros(h0.size()))
        # action_probs = [[] for _ in range(encodings.size(0))]

        log_probs = []
        for step in range(len(targets[0])):
            out, (h, c) = self.decoder_lstm(encs, h_c)
            # print(out.size(), h.size(), c.size())
            context, _ = self.context_attn(encs, out)
            att_vec = self.att_vec_linear(torch.cat([out, context], dim=-1))
            action_logprob = F.log_softmax(self.apply_action(att_vec), dim=-1) # bsize x prod_num
            log_probs.append(action_logprob)
            # print(action_logprob.size())
            h_c = h, c

        log_probs = torch.cat(log_probs, dim=1).transpose(1,2)
        ## TODO: mask by seq len
        mask = (torch.arange(len(targets[0]))[None, :] < target_lengths[:, None]).float()
        log_probs = log_probs * mask.unsqueeze(1).to(encodings.device)

        loss_fn = nn.NLLLoss()
        loss = loss_fn(log_probs, targets)
        # loss is negative sum of all the action probabilities
        # loss = - torch.stack([torch.stack(logprob_i).sum() for logprob_i in action_probs]).sum()
        return loss