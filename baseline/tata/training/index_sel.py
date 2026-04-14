import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
import torch
import torch.nn as nn
import torch.nn.functional as F



def compute_score(gt, pred):
    gt = np.array(gt)
    pred = np.array(pred)
    acc = sum(pred == gt) / len(gt)
    f1 = f1_score(gt, pred, average = None)
    avg_f1 = np.mean(f1)
    return acc, f1, avg_f1


def append_log(log_df, cutoff, feature, split, repeat, group, acc, f1):
    while len(f1) < 3:
        f1 = np.append(f1, None)
    
    line = {
        'group' : group,
        'cutoff' : cutoff,
        'feature' : feature,
        'split' : split,
        'repeat' : repeat,
        'acc' : acc,
        'f1_REGRESS' : f1[0],
        'f1_IMPROVE' : f1[1],
        'f1_NODIFF' : f1[2]
    }
    if log_df is None:
        return pd.DataFrame(line, index=[0])
    else:
        return log_df.append(line, ignore_index = True)  


class Classifier(nn.Module):
    def __init__(self, in_feat, hid_unit=64, classes=3):
        super(Classifier, self).__init__()
        self.mlp1 = nn.Linear(in_feat, hid_unit)
        self.mlp2 = nn.Linear(hid_unit, hid_unit)
        self.mlp3 = nn.Linear(hid_unit, hid_unit)
        self.mlp4 = nn.Linear(hid_unit, classes)
    def forward(self, lefts, rights):
        features = rights - lefts
        hid = F.relu(self.mlp1(features))
        mid = F.relu(self.mlp2(hid))
        mid = F.relu(self.mlp3(mid))
        out = self.mlp4(hid+mid)
        return out
    
    
# training functions

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def train(model, rep, full_ds, train_df,y_train, ds_info, args, loader):
    
    bs, device, epochs = args.bs, args.device, args.epochs
    lr = args.lr
    
    if args.freeze:
        optimizer = torch.optim.Adam(list(model.parameters()),lr = args.lr)
    else:
        optimizer = torch.optim.Adam(list(model.parameters())+ list(rep.parameters()),lr = args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 30, 0.8)
    crit = nn.CrossEntropyLoss()
    best_acc = 0
    rng = np.random.default_rng()

#     t0 = time.time()
    best_prev_f1 = 0

    model = model.to(device)
    if rep != 'NA':
        rep = rep.to(device)
    
    for epoch in range(epochs):
        losses = 0
        model.train()
        predlables = []
        gt = []
        if rep != 'NA':
            rep.train()
        train_idxs = rng.permutation(len(train_df))
        for idxs in chunks(train_idxs, bs):
            optimizer.zero_grad()

            lefts = train_df.loc[idxs, 'Left'].to_numpy()
            rights = train_df.loc[idxs, 'Right'].to_numpy()

            left_ds = torch.utils.data.Subset(full_ds, lefts)
            right_ds = torch.utils.data.Subset(full_ds, rights)
            
            left_batch, right_batch = loader(left_ds, right_ds, args)
            
            if rep == 'NA':
                preds = model(left_batch, right_batch)
            else:
                preds = model(rep(left_batch), rep(right_batch))
                
            _, pred_labels = torch.max(preds, 1)

            predlables = np.append(predlables, pred_labels.cpu().detach().numpy())
            
            batch_labels = y_train[idxs].to(device)
            gt = np.append(gt, batch_labels.cpu().detach().numpy())

            loss = crit(preds, batch_labels)
            loss.backward(retain_graph=True)

            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)

            optimizer.step()
            losses += loss.item()
            
        if epoch % 20 == 0:
            print('training epoch: ', epoch)
            
    return model

def predict(model, rep, ds, pair_df, args, loader, method = 'NaN', record = False):
    model = model.to(args.device)
    if rep != 'NA':
        rep = rep.to(args.device)
    model.eval()
    res = np.empty(0)
    if rep == 'NA':
        optimizer = torch.optim.Adam(list(model.parameters()),lr = args.lr)
    else:
        optimizer = torch.optim.Adam(list(model.parameters())+ list(rep.parameters()),lr = args.lr)
    for idxs in chunks(range(len(pair_df)), args.bs):
        
        optimizer.zero_grad()

        lefts = pair_df.loc[idxs, 'Left'].to_numpy()
        rights = pair_df.loc[idxs, 'Right'].to_numpy()
        
        left_ds = torch.utils.data.Subset(ds,lefts)
        right_ds = torch.utils.data.Subset(ds,rights)
        
        left_batch, right_batch = loader(left_ds, right_ds, args)
        
        if rep == 'NA':
            preds = model(left_batch, right_batch)
        else:
            preds = model(rep(left_batch), rep(right_batch))
            
        _, pred_labels = torch.max(preds, 1)
        
        res = np.append(res, pred_labels.cpu().detach().numpy())
        
    if record == True:
        fname = 'results/index/test_log_' + 'stats_' + args.splitting + '.csv'
        
        file = open(fname, 'a+')
        file.close()

        try:
            df = pd.read_csv(fname)
        except:
            df = pd.DataFrame()
        df[method] = res
        df.to_csv(fname)
    
    return res