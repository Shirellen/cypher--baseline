import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr

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


    # print(len(predss))
    q_errors = print_qerror(norm.unnormalize_labels(predss), labels, prints)
    abs_errors = get_abs_errors(norm.unnormalize_labels(predss), labels)

    return q_errors, abs_errors


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

class Normalizer(): # in log scale
    def __init__(self, mini=None,maxi=None):
        self.mini = mini
        self.maxi = maxi
        
    def normalize_labels(self, labels, reset_min_max = False):
        ## added 0.001 for numerical stability
        labels = np.array([np.log(float(l) + 0.001) for l in labels])
        if self.mini is None or reset_min_max:
            self.mini = labels.min()
            print("min log(label): {}".format(self.mini))
        if self.maxi is None or reset_min_max:
            self.maxi = labels.max()
            print("max log(label): {}".format(self.maxi))
        labels_norm = (labels - self.mini) / (self.maxi - self.mini)
        # Threshold labels <-- but why...
        labels_norm = np.minimum(labels_norm, 1)
        labels_norm = np.maximum(labels_norm, 0.001)

        return labels_norm

    def normalize_label(self,label):
        label_norm = ((np.log(float(label)+0.001)) - self.mini) / (self.maxi - self.mini)
        label_norm = np.minimum(label_norm, 1)
        label_norm = np.maximum(label_norm, 0.001)        
        return label_norm
    
    def unnormalize_labels(self, labels_norm):
        labels_norm = np.array(labels_norm, dtype=np.float32)
        labels = (labels_norm * (self.maxi - self.mini)) + self.mini
#         return np.array(np.round(np.exp(labels) - 0.001), dtype=np.int64)
        return np.array(np.exp(labels) - 0.001)
