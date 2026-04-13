"""调试 QPPNet 预测值是否为常数。"""
import sys
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, '/Users/yyh/Desktop/coding/research')
sys.path.insert(0, '/Users/yyh/Desktop/coding/research/baseline/QPPNet/QPPNet_Cypher')

encoding_ckpt = torch.load('/Users/yyh/Desktop/coding/research/data/encoding.pt', weights_only=False)
encoding = encoding_ckpt['encoding']

train_df = pd.read_csv('/Users/yyh/Desktop/coding/research/data/train_by_para_v2_same_500.csv')

from qppnet_cypher_dataset import QPPNetCypherDataset, compute_node_feature_dim
from qppnet_cypher_model import QPPNetCypher

# 只用前 300 条快速调试
small_df = train_df.head(300)
train_ds = QPPNetCypherDataset(small_df, encoding, fit_normalizer=True)
node_feat_dim = compute_node_feature_dim(encoding)
final_mrd = train_ds._compute_mean_range(train_ds._groups_raw, encoding)
train_ds.rebuild(node_feat_dim, final_mrd)

print(f'node_feat_dim={node_feat_dim}, groups={len(train_ds.groups)}')

# 检查 feat_vec
g = train_ds.groups[0]
print(f'Group0: node_type={g["node_type"]}, subbatch={g["subbatch_size"]}')
print(f'feat_vec shape={g["feat_vec"].shape}, min={g["feat_vec"].min():.4f}, max={g["feat_vec"].max():.4f}')
print(f'total_time[:5]={g["total_time"][:5]}')
print(f'feat nan={np.isnan(g["feat_vec"]).any()}, inf={np.isinf(g["feat_vec"]).any()}')

# 初始化模型
operator_types = list(encoding.type2idx.keys())
model = QPPNetCypher(node_feat_dim=node_feat_dim, operator_types=operator_types, lr=1e-3)

# 训练前：直接看 NeuralUnit 输出
samp = train_ds.sample_batch(8)
with torch.no_grad():
    _, pred_before, true_before = model.forward(samp)
print(f'\n[Before] pred[:5]={pred_before[:5]}')
print(f'[Before] true[:5]={true_before[:5]}')

# 训练 50 步
for step in range(50):
    samp = train_ds.sample_batch(32)
    loss_val, _, _ = model.optimize_parameters(samp)
    if step % 10 == 0:
        print(f'  step {step}: loss={loss_val:.6f}')

# 训练后
samp = train_ds.sample_batch(8)
with torch.no_grad():
    _, pred_after, true_after = model.forward(samp)
print(f'\n[After 50 steps] pred[:5]={pred_after[:5]}')
print(f'[After 50 steps] true[:5]={true_after[:5]}')
print(f'pred unique: {len(set(pred_after))}')
