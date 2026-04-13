# %%
import numpy as np
import os
import torch
import torch.nn as nn
import time
import pandas as pd
from scipy.stats import pearsonr

# %%
from model.util import Normalizer #数值归一化/反归一化工具（常用于把原始 cost/card 转到训练尺度）
from model.database_util import get_hist_file  #获取直方图文件和获取表样本 与数据库统计/采样/数据整理相关工具
from model.model_transformer_job import QueryFormer #模型类（这是脚本要训练的模型）
from model.dataset_schema_gat import PlanTreeDataset #把 plan 数据组织成 PyTorch Dataset 的类
from model.trainer_cost_job import train,evaluate #训练与评估工作流函数

# %%
data_path = './data/job/'

# %%
class Args:
    # bs = 1024
    # SQ: smaller batch size
    bs = 128
    # lr = 1
    lr = 0.001 # 当前最好
    # epochs = 200
    epochs = 100
    clip_size = 50
    embed_size = 64
    pred_hid = 128
    ffn_dim = 128
    head_size = 12
    n_layers = 8
    dropout = 0.1
    sch_decay = 0.6
    device = 'cuda:0'
    newpath = './results/job-full/cost/'
    to_predict = 'cost'
args = Args()

import os
if not os.path.exists(args.newpath):
    os.makedirs(args.newpath)

# %%
hist_file = get_hist_file(data_path + 'histogram_string.csv') # 从指定路径加载 histogram 文件（图数据库/表的统计摘要），供模型/数据集使用以获得统计信息
# cost_norm = Normalizer(-3.61192, 12.290855) #用于 cost 的标准化（具体参数是 mean/std 或其他方式，需看 Normalizer 实现）
# card_norm = Normalizer(1,100) #用于 cardinality（基数）标准化
cost_norm = Normalizer()
card_norm = Normalizer()

# %%
# 从磁盘加载两个 checkpoint：encoding.pt：包含 encoding（例如 token→id、属性映射等）并
encoding_ckpt = torch.load('data/job/encoding.pt')
encoding = encoding_ckpt['encoding'] # 把其中的 encoding 取出保存到变量 encoding。
checkpoint = torch.load('checkpoints/cost_model.pt', map_location='cpu') # cost_model.pt：把模型 checkpoint 加载到 checkpoint（注意：此脚本里并没有把 checkpoint 应用到 model（比如 model.load_state_dict(...)）——所以它被加载但并未被使用，可能是作者临时保留的）。
# map_location='cpu' 保证在没有 GPU 时也能加载到 CPU。

# %%
from model.util import seed_everything
seed_everything()

# %%
# 创建 QueryFormer 模型实例，传入超参数：
# use_sample = True, use_hist = True 表明模型会使用表采样信息和 histogram 统计作为输入特征（加强估计能力）。
# 其他参数定义了模型结构（embedding/transformer 等）
model = QueryFormer(emb_size = args.embed_size ,ffn_dim = args.ffn_dim, head_size = args.head_size, \
                 dropout = args.dropout, n_layers = args.n_layers, \
                 use_hist = False, \
                 pred_hid = args.pred_hid
                )
# %%
# 将模型移动到指定设备（GPU 'cuda:0' 或 CPU），并把返回值赋给 _（只为了调用 .to()）
# _ = model.to(args.device)
try:
    _ = model.to(args.device)
except RuntimeError as e:
    # 驱动或 CUDA 版本不匹配时，回退到 CPU
    if "NVIDIA driver" in str(e) or "CUDA" in str(e):
        print(f"[Warn] CUDA 初始化失败，回退到 CPU：{e}")
        args.device = 'cpu'
        _ = model.to(args.device)
    else:
        raise

# %%
to_predict = 'cost' #再次明确要预测的目标是 cost
# to_predict = 'card' #再次明确要预测的目标是 cost

# %%
imdb_path = './data/job/' #再一次定义数据根路径
dfs = []  # list to hold DataFrames
# SQ: added
# 读取训练数据分片 CSV 文件，原本注释显示可能本来打算读取更多分片（range(18)），但当前只读取 train_plan_part0.csv 和 train_plan_part1.csv（range(2)）。将每个 CSV 加入 dfs 列表
for i in range(1):
#for i in range(18):
    # file = imdb_path + 'plan_and_cost/update_train_plan_part{}.csv'.format(i)
    file = imdb_path + 'plan_and_cost/train_by_para_v2_same_500.csv'.format(i)
    # file = imdb_path + 'update/queries_4b_118.csv'.format(i)
    df = pd.read_csv(file)
    dfs.append(df)

full_train_df = pd.concat(dfs) #将读取到的训练 DataFrame 列表合并成一个大 DataFrame

# 这里把 train_plan_part18.csv 和 train_plan_part19.csv 作为验证集（val_df）。注意分片编号的选择表明数据被预先分成很多部分，脚本选择部分来训练/验证
val_dfs = []  # list to hold DataFrames
# for i in range(18,20):
for i in range(1, 2):
    # file = imdb_path + 'plan_and_cost/update_train_plan_part{}.csv'.format(i)
    # file = imdb_path + 'plan_and_cost/queries_ldbc_Parametric.csv'.format(i)
    file = imdb_path + 'plan_and_cost/val_by_para_v2_same_500.csv'.format(i)
    # file = imdb_path + 'update/queries_13b_117.csv'.format(i)
    df = pd.read_csv(file)
    val_dfs.append(df)

val_df = pd.concat(val_dfs)

# %%
# 读取 / 生成表采样（table samples），供 dataset 或模型在构建特征时使用（例如从表中抽样以估计 predicate 选择性）??????????????????????
# table_sample = get_job_table_sample(imdb_path+'train')
table_sample = None
# %%
db_id='job'
# db_id='ldbc'
# 用 PlanTreeDataset 把 pandas DataFrame 转换为 PyTorch Dataset 对象????????????????????????
train_ds = PlanTreeDataset(full_train_df, None, encoding, hist_file, card_norm, cost_norm, to_predict, table_sample)
print(f"[Norm] train cost min/max: {cost_norm.mini}, {cost_norm.maxi}")
val_ds = PlanTreeDataset(val_df, None, encoding, hist_file, card_norm, cost_norm, to_predict, table_sample)
# print(f"[Norm] val   cost min/max(should be same): {cost_norm.mini}, {cost_norm.maxi}")
# %%
# 使用均方误差损失（MSE），常用于回归任务 —— 这里用于预测 cost 值（回归）
crit = nn.MSELoss()

# ######
start_time = time.time()
model, best_path = train(model, train_ds, val_ds, crit, cost_norm, args)
end_time = time.time()
print(f"Training completed in {end_time - start_time:.2f} seconds.")

# 使用最优模型评估验证集与测试集
best_ckpt_path = os.path.join(args.newpath, best_path) if best_path else None
# best_ckpt_path = '/data/yuanyihui/cypher-emb/results/job-full/cost/5930485619218814098.pt'
ckpt = torch.load(best_ckpt_path, map_location='cpu')
model.load_state_dict(ckpt['model'])
_ = model.to(args.device)
torch.cuda.empty_cache()
print(f'[Info] Loaded best checkpoint: {best_ckpt_path}')
# 验证集评估（可选）
# evaluate(model, val_ds, args.bs, cost_norm, args.device, prints=True)
test_csv = '/data/yuanyihui/cypher-emb/data/job/plan_and_cost/test_by_para_v2_same_500.csv'
# test_csv = '/data/yuanyihui/cypher-emb/data/cypher/plan_and_cost/ldbc_Parametric.csv'
test_df = pd.read_csv(test_csv)
test_ds = PlanTreeDataset(test_df, None, encoding, hist_file, card_norm, cost_norm, args.to_predict, table_sample)
test_scores, corrs,q, all_pre, all_true_cost = evaluate(model, test_ds, args.bs, cost_norm, args.device, prints=True)
best_res = {}
best_res['id'] = test_df['id']
best_res['q'] = q
best_res['pred'] = all_pre
best_res['cost_labelss'] = all_true_cost
best_res['dataset'] = (getattr(test_ds, 'dataset', ['unknown'] * len(all_pre)))[:len(all_pre)]
best_res['src_file'] = (getattr(test_ds, 'src_file', ['unknown'] * len(all_pre)))[:len(all_pre)]
best_res = pd.DataFrame(best_res)
best_res.to_csv('results/job-full/cost/cost-schema-job.csv')