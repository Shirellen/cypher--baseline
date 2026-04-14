import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from scipy.stats import pearsonr
import time
import os
import csv



## cost prediction MLP model
class Prediction(nn.Module):
    def __init__(self, in_feature = 69, hid_units = 256, contract = 1, mid_layers = True, res_con = True):
        super(Prediction, self).__init__()
        self.mid_layers = mid_layers
        self.res_con = res_con
        
        self.out_mlp1 = nn.Linear(in_feature, hid_units)

        self.mid_mlp1 = nn.Linear(hid_units, hid_units//contract)
        self.mid_mlp2 = nn.Linear(hid_units//contract, hid_units)

        self.out_mlp2 = nn.Linear(hid_units, 1)

    def forward(self, features):
        
        hid = F.relu(self.out_mlp1(features))
        if self.mid_layers:
            mid = F.relu(self.mid_mlp1(hid))
            mid = F.relu(self.mid_mlp2(mid))
            if self.res_con:
                hid = hid + mid
            else:
                hid = mid
        out = torch.sigmoid(self.out_mlp2(hid))

        return out


def print_qerror(ps, ls, prints=True):
    ps = np.array(ps)
    ls = np.array(ls)
    qerror = []
    for i in range(len(ps)):
        if ps[i] > float(ls[i]):
            qerror.append(ps[i] / float(ls[i]))
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
    #    print("95th percentile: {}".format(e_95))
    #    print("99th percentile: {}".format(e_99))
        print("Max: {}".format(e_max))
        print("Mean: {}".format(e_mean))

    return res


def get_abs_errors(ps, ls): # unnormalised
    ps = np.array(ps).flatten()
#     if len(set(ps)) == 1:
#         print(ps, ls)
    ls = np.array(ls).flatten()
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

def evaluate(model, loader, labels, bs, norm, device, prints=True):
    model.to(device)
    model.eval()
    predss = np.empty(0)

    with torch.no_grad():
        for x,y in loader:
            x = x.to(device)           
            preds = model(x).squeeze()
            predss = np.append(predss, preds.cpu().detach().numpy())


    q_errors = print_qerror(norm.unnormalize_labels(predss), labels, prints)
    abs_errors = get_abs_errors(norm.unnormalize_labels(predss), labels)

    return q_errors, abs_errors
#     return q_errors, abs_errors, predss


def get_record(model, loader, labels, bs, norm, device):
    model.to(device)
    model.eval()
    predss = np.empty(0)

    with torch.no_grad():
        for x,y in loader:
            x = x.to(device)           
            preds = model(x).squeeze()
            predss = np.append(predss, preds.cpu().detach().numpy())
    predss = norm.unnormalize_labels(predss)
    predss = predss.tolist()
    return predss, labels

def collate_record(predss, labels, method, save_file):
    if os.path.isfile(save_file):
        df = pd.read_csv(save_file)
    else:
        df = pd.DataFrame(data={'label':labels})
    df[method] = predss
    df.to_csv(save_file,index=False)
    return df

def eval_record(model, loader, labels, bs, norm, device, save_path, method, dataset):
    model.to(device)
    model.eval()
    predss = np.empty(0)

    with torch.no_grad():
        for x,y in loader:
            x = x.to(device)           
            preds = model(x).squeeze()
            predss = np.append(predss, preds.cpu().detach().numpy())
    predss = norm.unnormalize_labels(predss)
    predss = predss.tolist()
    title = [method, dataset]
    title += predss
    file = open('results/cost/test_log.csv', 'a+', newline ='')
    with file:   
        write = csv.writer(file)
        write.writerow(title)
    file.close()



def logging(args, epoch, qscores, absscores, time, filename = None, save_model = False, models = None):
    arg_keys = [attr for attr in dir(args) if not attr.startswith('__')]
    arg_vals = [getattr(args, attr) for attr in arg_keys]
    
    res = dict(zip(arg_keys, arg_vals))
    model_checkpoint = str(hash(tuple(arg_vals))) + '.pt'

    res['epoch'] = epoch
    res['model'] = model_checkpoint 

    res = {**res, **qscores, **absscores}
    
    res['time'] = time


    filename = args.result_path + filename

    model_checkpoint = args.model_path + model_checkpoint


    if os.path.isfile(filename):
        df = pd.read_csv(filename)
        df = pd.concat([df,pd.DataFrame(res,index=[0])], ignore_index=True)
        # df = df.append(res, ignore_index=True)
        df.to_csv(filename, index=False)
    else:
        if not os.path.exists(args.result_path):
            os.mkdir(args.result_path)
        df = pd.DataFrame(res, index=[0])
        df.to_csv(filename, index=False)

    if save_model:
        d = {}
        for i, model in enumerate(models):
            d['model{}'.format(i)] = model.state_dict()
        d['args'] = args
        torch.save(d, model_checkpoint)
    
    return res['model']  


def pretrain(rep_model, est_model, train_loader, val_loader, val_labels,
        ds_info, args, record = True, prints = True):
    
    model = nn.Sequential(rep_model, est_model)

    bs, device, epochs = \
        args.bs, args.device, args.epochs
    lr = args.lr

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 20, 0.9)
    crit = torch.nn.MSELoss()

    t0 = time.time()

    best_prev = 999999

    model.to(device)
    
    best_model_file = None
    
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
            labels = np.append(labels, y.detach().cpu().numpy())
            

        if epoch % 20 == 0 and prints:
            print('Epoch: {}  Avg Loss: {}, Time: {}'.format(epoch,losses/len(predss), time.time()-t0))
            print_qerror(ds_info.cost_norm.unnormalize_labels(predss), ds_info.cost_norm.unnormalize_labels(labels))

        if record:
            q_errors, abs_errors = evaluate(model, val_loader, val_labels, bs, ds_info.cost_norm, device, prints=False)

            if (q_errors['q_median'] + q_errors['q_mean']) < best_prev: ## mean mse
                best_model_file = logging(args, epoch, q_errors, abs_errors, time.time()-t0, filename = 'log.csv', save_model = True, models = [rep_model, est_model])
                best_prev = (q_errors['q_median'] + q_errors['q_mean'])
            # resq = print_qerror(ds_info.cost_norm.unnormalize_labels(predss), ds_info.cost_norm.unnormalize_labels(labels), prints=False)
            # filename = args.result_path + 'loss_log_' + str(args.lr) + '_' + str(args.bs) + '.csv'
            # res = {}
            # res['Epoch'] = epoch
            # res['Avg Loss'] = losses/len(predss)
            # res['Time'] = time.time()-t0
            # res['Mean Train Error'] = resq['q_mean']
            
            # if os.path.isfile(filename):
            #     df = pd.read_csv(filename)
            #     df = pd.concat([df,res], ignore_index=True)
            #     # df = df.append(res, ignore_index=True)
            #     df.to_csv(filename, index=False)
            # else:
            #     if not os.path.exists(args.save_path):
            #         os.mkdir(args.save_path)
            #     df = pd.DataFrame(res, index=[0])
            #     df.to_csv(filename, index=False)

        scheduler.step()
    if best_model_file is not None:
        model_file = args.model_path + best_model_file
        checkpoint = torch.load(model_file)

        rep_model.load_state_dict(checkpoint['model0'])
        est_model.load_state_dict(checkpoint['model1'])
        model = nn.Sequential(rep_model,est_model)
    return [rep_model, est_model]


def joint_train(rep_model, cost_model, decoder_model, 
    train_loader, decoder_loader, val_loader, sampler, val_labels, 
    ds_info, args, crit=None, optimizer=None, 
    scheduler=None, prints=True, record=True,):

    models = [rep_model, cost_model, decoder_model]
    bs, device, epochs = \
        args.bs, args.device, args.epochs
    lr = args.lr


    all_parameters = []
    for m in models:
        all_parameters.extend(m.parameters())
    optimizer = torch.optim.Adam(all_parameters, lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 20, 0.9)
    crit = torch.nn.MSELoss()

    t0 = time.time()

    best_prev = 999999

    for m in models:
        m.to(device)
    
    best_model_file = None
    
    for epoch in range(epochs):
        sampler.set_epoch(epoch)
        for m in models:
            m.train()
 
        losses = 0
        predss = np.empty(0)
        labels = np.empty(0)

        decoder_it = iter(decoder_loader)
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            d_batch = next(decoder_it)
            targets = d_batch.targets.to(args.device)
            lens = d_batch.lens#.to(args.device)
            
            optimizer.zero_grad()

            intermediate_reps = rep_model(x)
            decoding_loss = decoder_model.score(intermediate_reps, targets, lens)
            
            cost_pred = cost_model(intermediate_reps)
            cost_loss = crit(cost_pred, y)

            loss = (1-args.lambdaa) * cost_loss + args.lambdaa * decoding_loss
            loss.backward(retain_graph=True)

            for m in models:
                torch.nn.utils.clip_grad_norm_(m.parameters(), 5)

            optimizer.step()
            losses += loss.item()
            predss = np.append(predss, cost_pred.detach().cpu().numpy())
            # for debugging
            # if None in predss or len(set(predss)) == 1:
            #     print(predss)
            labels = np.append(labels, y.detach().cpu().numpy())
            

        if epoch % 20 == 0 and prints:
            print('Epoch: {}  Avg Loss: {}, Time: {}'.format(epoch,losses/len(predss), time.time()-t0))
            print_qerror(ds_info.cost_norm.unnormalize_labels(predss), ds_info.cost_norm.unnormalize_labels(labels))
        ##############
        model = nn.Sequential(rep_model, cost_model)
        if record:
            q_errors, abs_errors = evaluate(model, val_loader, val_labels, bs, ds_info.cost_norm, device, prints=False)
            if (q_errors['q_median'] + q_errors['q_mean']) < best_prev: ## mean mse
                best_model_file = logging(args, epoch, q_errors, abs_errors, time.time()-t0, filename = 'log.csv', save_model = True, models = models)
                best_prev = (q_errors['q_median'] + q_errors['q_mean'])
        scheduler.step()   
        
        if best_model_file is not None:
            model_file = args.model_path + best_model_file
            checkpoint = torch.load(model_file)
    
            rep_model.load_state_dict(checkpoint['model0'])
            cost_model.load_state_dict(checkpoint['model1'])
            decoder_model.load_state_dict(checkpoint['model2'])
    return [rep_model, cost_model, decoder_model]

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
        ##############
        if record:
            q_errors, abs_errors = evaluate(model, val_loader, val_labels, bs, ds_info.cost_norm, device, prints=False)

            if q_errors['q_mean'] < best_prev: ## mean mse
                best_model_path = logging(args, epoch, q_errors, abs_errors, time.time()-t0, filename = 'log.csv', save_model = True, models = model)
                best_prev = q_errors['q_mean']
            resq = print_qerror(ds_info.cost_norm.unnormalize_labels(predss), ds_info.cost_norm.unnormalize_labels(labels), prints=False)
            filename = args.save_path + 'loss_log_' + str(args.lr) + '_' + str(args.bs) + '.csv'
            res = {}
            res['Epoch'] = epoch
            res['Avg Loss'] = losses/len(predss)
            res['Time'] = time.time()-t0
            res['Mean Train Error'] = resq['q_mean']
            
            if os.path.isfile(filename):
                df = pd.read_csv(filename)
                df = pd.concat([df,res], ignore_index=True)
                # df = df.append(res, ignore_index=True)
                df.to_csv(filename, index=False)
            else:
                if not os.path.exists(args.save_path):
                    os.mkdir(args.save_path)
                df = pd.DataFrame(res, index=[0])
                df.to_csv(filename, index=False)
        ##############
        scheduler.step()   

    return model, best_model_path
