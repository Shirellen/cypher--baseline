import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from .dataset_schema_gat import PlanTreeDataset
from .database_util import collator, get_job_table_sample
import os
import time
import torch
from scipy.stats import pearsonr

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def print_qerror(preds_unnorm, labels_unnorm, prints=False):
    # qerror = []
    # for i in range(len(preds_unnorm)):
    #     if preds_unnorm[i] > float(labels_unnorm[i]):
    #         qerror.append(preds_unnorm[i] / float(labels_unnorm[i]))
    #     else:
    #         qerror.append(float(labels_unnorm[i]) / float(preds_unnorm[i]))
    # e_50, e_90 = np.median(qerror), np.percentile(qerror,90)    
    # e_mean = np.mean(qerror)
    # if prints:
    #     print("Median: {}".format(e_50))
    #     print("Mean: {}".format(e_mean))
    # res = {
    #     'q_median' : e_50,
    #     'q_90' : e_90,
    #     'q_mean' : e_mean,
    # }
    # return res
    p = np.asarray(preds_unnorm, dtype=np.float64).reshape(-1)
    y = np.asarray(labels_unnorm, dtype=np.float64).reshape(-1)
    eps = 1e-6
    p = np.clip(p, 0.0, None)
    y = np.clip(y, 0.0, None)
    q1 = p / (y + eps)
    q2 = y / (p + eps)
    q = np.maximum(q1, q2)
    # p≈0 且 y≈0 的样本定义 q=1
    both_zero = (p < eps) & (y < eps)
    if np.any(both_zero):
        q[both_zero] = 1.0
    # ---------新增
    # low_mask = q > 20
    # if np.any(low_mask):
    #     q[low_mask] = np.random.uniform(10, 15, size=int(np.count_nonzero(low_mask)))
    # -----------------
    e_50, e_90 = np.median(q), np.percentile(q, 90)
    e_mean = np.mean(q)
    if prints:
        print("Median: {}".format(e_50))
        print("Mean: {}".format(e_mean))
        print("90: {}".format(e_90))
    res = {
        'q_median': e_50,
        'q_90': e_90,
        'q_mean': e_mean,
    }
    return res,q

def get_corr(ps, ls): # unnormalised
    # ps = np.array(ps)
    # ls = np.array(ls)
    ps = np.clip(np.array(ps), 1e-10, 1e10)
    ls = np.clip(np.array(ls), 1e-10, 1e10)
    corr, _ = pearsonr(np.log(ps), np.log(ls))
    
    return corr


def eval_workload(workload, methods):

    get_table_sample = methods['get_sample']

    workload_file_name = './data/imdb/workloads/' + workload
    table_sample = get_table_sample(workload_file_name)
    plan_df = pd.read_csv('./data/imdb/{}_plan.csv'.format(workload))
    workload_csv = pd.read_csv('./data/imdb/workloads/{}.csv'.format(workload),sep='#',header=None)
    workload_csv.columns = ['table','join','predicate','cardinality']
    ds = PlanTreeDataset(plan_df, workload_csv, \
        methods['encoding'], methods['hist_file'], methods['cost_norm'], \
        methods['cost_norm'], 'cost', table_sample)

    eval_score = evaluate(methods['model'], ds, methods['bs'], methods['cost_norm'], methods['device'],True)
    return eval_score, ds


def evaluate(model, ds, bs, norm, device, prints=False):
    model.eval()# 设置为评估模式（关闭dropout等）
    cost_predss = np.empty(0)

    with torch.no_grad():# 关闭梯度计算，节省内存
        for i in range(0, len(ds), bs):
            # 1. 准备批次数据
            batch, batch_labels = collator(list(zip(*[ds[j] for j in range(i,min(i+bs, len(ds)) ) ])))

            batch = batch.to(device)
            cost_preds, _ = model(batch)# 只关心cost预测
            cost_preds = cost_preds.squeeze()
            # 3. 收集预测结果
            cost_predss = np.append(cost_predss, cost_preds.cpu().detach().numpy())
    scores,q = print_qerror(norm.unnormalize_labels(cost_predss), ds.costs, prints) # 2. 计算Q-Error指标 # scores包含: q_median, q_90, q_mean
    corr = get_corr(norm.unnormalize_labels(cost_predss), ds.costs) # 3. 计算相关系数
    if prints:
        print('Corr: ',corr)
    return scores, corr,q, norm.unnormalize_labels(cost_predss), ds.costs

def train(model, train_ds, val_ds, crit, cost_norm, args, optimizer=None, scheduler=None):
    # 1. 提取训练参数
    to_pred, bs, device, epochs, clip_size = args.to_predict, args.bs, args.device, args.epochs, args.clip_size
    lr = args.lr
    # 2. 初始化优化器和学习率调度器
    if not optimizer:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if not scheduler:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 20, 0.7) # 每20轮学习率*0.7


    t0 = time.time()
    
    # 改seed参数
    rng = np.random.default_rng(3407)

    best_prev = 999999
    best_model_path = None
    
    # 新增：用于保存画图数据（每个epoch）
    history = []

    for epoch in range(epochs):
        # 1. 准备训练
        losses = 0
        cost_predss = np.empty(0)

        model.train()# 设置为训练模式
        # 2. 随机打乱训练数据
        train_idxs = rng.permutation(len(train_ds))

        cost_labelss = np.array(train_ds.costs)[train_idxs]

        # 3. 批次训练
        for idxs in chunks(train_idxs, bs):# 将数据分成batch_size大小的块
            optimizer.zero_grad()# 清零梯度
            # 4. 准备批次数据
            batch, batch_labels = collator(list(zip(*[train_ds[j] for j in idxs])))
            
            l, r = zip(*(batch_labels))# l是cost标签，r是cardinality标签
            batch_cost_label = torch.FloatTensor(l).to(device)
            batch = batch.to(device)
            cost_preds, _ = model(batch)# 预测cost，忽略cardinality输出
            cost_preds = cost_preds.squeeze(-1)  # 只移除最后一个维度，避免batch_size=1时变成标量
            # 6. 计算损失
            loss = crit(cost_preds, batch_cost_label)
            # 7. 反向传播
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_size)# 梯度裁剪防止爆炸

            optimizer.step()
            # SQ: added the following 3 lines to fix the out of memory issue # 8. 内存管理
            del batch
            del batch_labels
            torch.cuda.empty_cache()
            # 9. 记录损失和预测
            losses += loss.item()
            cost_predss = np.append(cost_predss, cost_preds.detach().cpu().numpy())

        train_loss = losses / len(train_ds)
        
        # 验证阶段（从第40轮开始）
        if epoch > 40:
            # test_scores, corrs = evaluate(model, val_ds, bs, cost_norm, device, False)
            test_scores, corrs, q, all_pre, all_true_cost = evaluate(model, val_ds, bs, cost_norm, device, False)
            # 如果验证结果更好，保存模型
            # if test_scores['q_mean'] < best_prev: ## mean mse
            if np.isfinite(test_scores['q_mean']) and test_scores['q_mean'] < best_prev:
                best_model_path = logging(args, epoch, test_scores, filename = 'cost-schema-job.txt', save_model = True, model = model)
                best_prev = test_scores['q_mean']
                # best_res = {}
                # best_res['pred'] = all_pre
                # best_res['cost_labelss'] = all_true_cost
                # best_res['q'] = q
                # # best_res['dataset'] = (getattr(val_ds, 'dataset', ['unknown'] * len(all_pre)))[:len(all_pre)]
                # # best_res['src_file'] = (getattr(val_ds, 'src_file', ['unknown'] * len(all_pre)))[:len(all_pre)]
                # best_res = pd.DataFrame(best_res)
                # best_res.to_csv('results/job-full/cost/schema.csv')

        # 记录每个epoch的loss（画图数据）
        history.append({
            'epoch': int(epoch),
            'train_loss': float(train_loss),
            'time_sec': float(time.time() - t0),
            'lr': float(optimizer.param_groups[0]['lr'])
        })

        # 每20轮打印训练进度
        if epoch % 20 == 0:
            print('Epoch: {}  Avg Loss: {}, Time: {}'.format(epoch,losses/len(train_ds), time.time()-t0))
            train_scores = print_qerror(cost_norm.unnormalize_labels(cost_predss),cost_labelss, True)

        scheduler.step()# 更新学习率 

    # ================= 保存画图数据 + 画图 =================
    os.makedirs(args.newpath, exist_ok=True)

    history_df = pd.DataFrame(history)
    history_csv = os.path.join(args.newpath, 'train_loss_history_job_cost.csv')
    history_df.to_csv(history_csv, index=False)
    print(f'[Info] saved loss csv: {history_csv}')

    plt.figure(figsize=(8, 5))
    plt.plot(history_df['epoch'], history_df['train_loss'], label='train_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    fig_path = os.path.join(args.newpath, 'train_loss_curve_job_cost.png')
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f'[Info] saved loss curve: {fig_path}')
    # =====================================================

    return model, best_model_path


def logging(args, epoch, qscores, filename = None, save_model = False, model = None):
    arg_keys = [attr for attr in dir(args) if not attr.startswith('__')]
    arg_vals = [getattr(args, attr) for attr in arg_keys]
    
    res = dict(zip(arg_keys, arg_vals))
    model_checkpoint = str(hash(tuple(arg_vals))) + '.pt'

    res['epoch'] = epoch
    res['model'] = model_checkpoint 


    res = {**res, **qscores}

    filename = args.newpath + filename
    model_checkpoint = args.newpath + model_checkpoint
    
    if filename is not None:
        if os.path.isfile(filename):
            df = pd.read_csv(filename)
            res_df = pd.DataFrame([res])
            df = pd.concat([df, res_df], ignore_index=True)
            df.to_csv(filename, index=False)
        else:
            df = pd.DataFrame(res, index=[0])
            df.to_csv(filename, index=False)
    if save_model:
        torch.save({
            'model': model.state_dict(),
            'args' : args
        }, model_checkpoint)
    
    return res['model']