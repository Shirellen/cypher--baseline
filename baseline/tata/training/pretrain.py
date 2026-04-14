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

from .cost_est import evaluate, print_qerror




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
        ds_info, args, record = True, prints = True, save_every_epoch = False):
    
    model = nn.Sequential(rep_model, est_model)

    bs, device, epochs = \
        args.bs, args.device, args.epochs
    lr = args.lr

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 20, 0.9)
    crit = torch.nn.MSELoss()

    t0 = time.time()

    model.to(device)
    
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

        scheduler.step()            

        if epoch % 20 == 0 and prints:
            print('Epoch: {}  Avg Loss: {}, Time: {}'.format(epoch,losses/len(predss), time.time()-t0))
            print_qerror(ds_info.cost_norm.unnormalize_labels(predss), ds_info.cost_norm.unnormalize_labels(labels))

        if record and save_every_epoch and (epoch+1 != epochs):
            q_errors, abs_errors = evaluate(model, val_loader, val_labels, bs, ds_info.cost_norm, device, prints=False)
            best_model_file = logging(args, epoch, q_errors, abs_errors, 
                time.time()-t0, filename = 'log.csv', 
                save_model = True, models = [rep_model, est_model])

    if record:
        q_errors, abs_errors = evaluate(model, val_loader, val_labels, bs, ds_info.cost_norm, device, prints=False)
        best_model_file = logging(args, epoch, q_errors, abs_errors, 
            time.time()-t0, filename = 'log.csv', 
            save_model = True, models = [rep_model, est_model])

    return [rep_model, est_model]


def pretrain_joint(rep_model, cost_model, decoder_model, 
    train_loader, decoder_loader, val_loader, sampler, val_labels, 
    ds_info, args, record = True, prints = True, save_every_epoch = False):

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

            labels = np.append(labels, y.detach().cpu().numpy())
        scheduler.step()              

        if epoch % 20 == 0 and prints:
            print('Epoch: {}  Avg Loss: {}, Time: {}'.format(epoch,losses/len(predss), time.time()-t0))
            print_qerror(ds_info.cost_norm.unnormalize_labels(predss), ds_info.cost_norm.unnormalize_labels(labels))

        model = nn.Sequential(rep_model, cost_model)
        if record and save_every_epoch and (epoch+1 != epochs):
            q_errors, abs_errors = evaluate(model, val_loader, val_labels, bs, ds_info.cost_norm, device, prints=False)
            best_model_file = logging(args, epoch, q_errors, 
                abs_errors, time.time()-t0, 
                filename = 'log.csv', save_model = True, 
                models = [rep_model, cost_model, decoder_model])

    if record:
        q_errors, abs_errors = evaluate(model, val_loader, val_labels, bs, ds_info.cost_norm, device, prints=False)
        best_model_file = logging(args, epoch, q_errors, 
            abs_errors, time.time()-t0, 
            filename = 'log.csv', save_model = True, 
            models = [rep_model, cost_model, decoder_model])        


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
