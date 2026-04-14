import pandas as pd
import numpy as np
import time
import torch, torch.nn as nn, torch.nn.functional as F
# from ..data_util.utils import get_prl
from .cost_est import print_qerror, evaluate, Prediction

def train(model, train_loader, val_loader, val_labels, \
    ds_info, args, crit=None, optimizer=None, scheduler=None, prints=True, record=True):
    
    bs, device, epochs = \
        args.bs, args.device, args.epochs
    lr = args.lr

    if not optimizer:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if not scheduler:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 20, 0.9)
    if not crit:
        crit = torch.nn.MSELoss()

    t0 = time.time()


    best_prev = 999999

    model.to(device)
    
    best_model_path = None
    
    for epoch in range(epochs):
        model.train()
        losses = 0
        predss = np.empty(0)
        labels = np.empty(0)

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            preds = model(x)
            loss = crit(preds, y)

            loss.backward(retain_graph=True)

            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)

            optimizer.step()
            losses += loss.item()
            predss = np.append(predss, preds.detach().cpu().numpy())
            # for debugging
            # if None in predss or len(set(predss)) == 1:
            #     print(predss)
            labels = np.append(labels, y.detach().cpu().numpy())
            

        if epoch % 20 == 0 and prints:
            print('Epoch: {}  Avg Loss: {}, Time: {}'.format(epoch,losses/len(predss), time.time()-t0))
            print_qerror(ds_info.cost_norm.unnormalize_labels(predss), ds_info.cost_norm.unnormalize_labels(labels))

        scheduler.step()   

    return model


import random
class BanditOptimizer():
    def __init__(self, planss, rootss, latencies):

        self.planss = planss
        self.rootss = rootss
        self.latencies = latencies
        
        self.arms = len(self.latencies)
        self.total = len(self.latencies[0])
        self.selections = []
        self.tm = [] # inference
        self.tl = [] # pre-process
        self.tr = [] # train
        self.exe_time = []
        ## record results
        # training time
        # inference time
        # query execution time
        # creating ds time

        self.spl = 0
    
    def get_execution_time(self):
        exe_time = []
        for i,sel in enumerate(self.selections):
            exe_time[i] = self.latencies[i][sel]
        self.exe_time = exe_time
        return exe_time

    def select_plans(self, model, get_batch):
        sels = []
        right = self.total
        qids = range(0, right)
        tm = []
        tl = []
        for qid in qids:
            roots = [self.rootss[i][qid] for i in range(self.arms)]
            lats = [self.latencies[i][qid] for i in range(self.arms)]
            
            t0 = time.time()
            batch = get_batch(roots, lats)
            t1 = time.time()
            out = model(batch).squeeze()
            t2 = time.time()
            tm.append(t2-t1)
            tl.append(t1-t0)
            
            sels.append( out.detach().cpu().argmin().numpy().item() )
            
            del batch
            
        self.selections += sels
        self.cur_query = right
        self.tm += tm
        self.tl += tl
        print('Model Time: {}, Preprocessing Time: {}'.format(sum(tm), sum(tl)))

        ## to get some reference numbers
        latss = [[self.latencies[i][qid] for i in range(self.arms)] for qid in qids]
        best_lats = 0
        post_lats = 0
        sel_lats = 0
        for i,qid in enumerate(qids):
            lats = [self.latencies[k][qid] for k in range(self.arms)]
            post_lats += self.latencies[0][qid] / 1000
            best_lats += min(lats) / 1000
            sel_lats += self.latencies[sels[i]][qid] / 1000
        print('Best Time: {}, Post Time: {}, Sel Time: {}'.format(best_lats, \
                                                post_lats, sel_lats))
        
        return self.selections, best_lats, post_lats, sel_lats
    
    def train_time(self, tr):
        print(len(self.tr))
        self.tr.append(tr)
        remain_len = min(self.freq-1, self.total-len(self.tr)) 
        self.tr += [0 for a in range(remain_len)]

    def total_lat(self, selections):
        return sum([self.latencies[selections[i]][i] 
                for i in range(len(selections))]) / 1000