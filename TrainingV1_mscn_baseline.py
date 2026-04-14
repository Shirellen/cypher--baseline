"""
MSCN Cypher Baseline 训练脚本。

TASK = 'card'：预测 Cardinality（根节点实际行数），标签 = 根节点 Rows（默认）
TASK = 'cost'：预测 Execution Time（毫秒），标签 = Execution Time
切换只需修改下方 TASK 变量，输出路径、文件名、列名全部自动切换。

归一化：log → min-max → [0,1]（与原版 MSCN 一致）
评估指标：Q-error（与 TrainingV1_leon_baseline.py 保持一致）
"""

# ── 任务开关（'card' 或 'cost'）──────────────────────────────────────────────
TASK = 'card'   # ← 改这里切换任务（MSCN 默认预测 card）

import os, sys, time, datetime, logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from scipy.stats import pearsonr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))                              # research/
_MSCN_DIR     = os.path.join(_PROJECT_ROOT, 'baseline', 'learnedcardinalities', 'mscn_cypher')
for _p in [_PROJECT_ROOT, _MSCN_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from model.util import seed_everything
from mscn_cypher_dataset import MscnCypherDataset, mscn_cypher_collate, compute_feature_dims
from mscn_cypher_model import SetConv


class Args:
    bs          = 128
    lr          = 0.001
    epochs      = 100
    clip_size   = 50
    hid_units   = 256
    device      = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    newpath     = os.path.join(_PROJECT_ROOT, 'results', 'job-full', f'mscn_{TASK}_baseline')
    sch_decay   = 0.6

args = Args()
os.makedirs(args.newpath, exist_ok=True)

# ── 日志 ──────────────────────────────────────────────────────────────────────
timestamp = datetime.datetime.now().strftime('%m%d-%H%M%S')

def get_logger(log_path):
    logger = logging.getLogger('mscn_cypher_baseline')
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('[%(asctime)s] %(message)s', datefmt='%H:%M:%S')
    logger.addHandler(logging.FileHandler(log_path, mode='w'))
    logger.addHandler(logging.StreamHandler())
    for h in logger.handlers:
        h.setFormatter(fmt)
    return logger

logger = get_logger(os.path.join(args.newpath, f'running_log_{timestamp}.txt'))
logger.info(f'Args: {vars(args)}')

# ── Q-error ───────────────────────────────────────────────────────────────────

def compute_qerror(preds, labels):
    preds  = np.clip(np.asarray(preds,  dtype=np.float64), 1.0, None)
    labels = np.clip(np.asarray(labels, dtype=np.float64), 1.0, None)
    q = np.maximum(preds / labels, labels / preds)
    return {'q_median': float(np.median(q)), 'q_mean': float(np.mean(q)),
            'q_90': float(np.percentile(q, 90))}, q

def unnormalize(vals_norm, min_val, max_val):
    """[0,1] → log 空间 → exp → 实际基数"""
    log_vals = vals_norm * (max_val - min_val) + min_val
    return np.round(np.exp(log_vals)).astype(np.int64)

# ── 训练 ──────────────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, loss_fn, device, clip_size):
    model.train()
    total_loss = 0.0
    for samples, predicates, joins, s_mask, p_mask, j_mask, labels in loader:
        samples, predicates, joins = samples.to(device), predicates.to(device), joins.to(device)
        s_mask, p_mask, j_mask    = s_mask.to(device), p_mask.to(device), j_mask.to(device)
        labels = labels.to(device).float().unsqueeze(1)

        preds = model(samples, predicates, joins, s_mask, p_mask, j_mask)
        loss  = loss_fn(preds, labels)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip_size)
        optimizer.step()
        total_loss += loss.item() * len(labels)
    return total_loss / len(loader.dataset)

def _compute_val_loss(model, ds, bs, loss_fn, device):
    model.eval()
    total_loss = 0.0
    loader = DataLoader(ds, batch_size=bs, shuffle=False, collate_fn=mscn_cypher_collate)
    with torch.no_grad():
        for samples, predicates, joins, s_mask, p_mask, j_mask, labels in loader:
            samples, predicates, joins = samples.to(device), predicates.to(device), joins.to(device)
            s_mask, p_mask, j_mask    = s_mask.to(device), p_mask.to(device), j_mask.to(device)
            labels = labels.to(device).float().unsqueeze(1)
            preds  = model(samples, predicates, joins, s_mask, p_mask, j_mask)
            total_loss += loss_fn(preds, labels).item() * len(labels)
    return total_loss / len(ds)

def evaluate(model, ds, bs, min_val, max_val, device, prints=False):
    model.eval()
    all_preds_norm, all_labels_norm = [], []
    loader = DataLoader(ds, batch_size=bs, shuffle=False, collate_fn=mscn_cypher_collate)
    with torch.no_grad():
        for samples, predicates, joins, s_mask, p_mask, j_mask, labels in loader:
            samples, predicates, joins = samples.to(device), predicates.to(device), joins.to(device)
            s_mask, p_mask, j_mask    = s_mask.to(device), p_mask.to(device), j_mask.to(device)
            preds = model(samples, predicates, joins, s_mask, p_mask, j_mask).squeeze(1).cpu().numpy()
            all_preds_norm.extend(preds.tolist())
            all_labels_norm.extend(labels.numpy().tolist())

    all_preds  = unnormalize(np.array(all_preds_norm),  min_val, max_val)
    all_labels = unnormalize(np.array(all_labels_norm), min_val, max_val)
    qerror_stats, per_q = compute_qerror(all_preds, all_labels)
    corr, _ = pearsonr(np.log(np.clip(all_preds, 1, None)), np.log(np.clip(all_labels, 1, None)))

    if prints:
        logger.info(f'  Q-error median={qerror_stats["q_median"]:.4f}  '
                    f'mean={qerror_stats["q_mean"]:.4f}  '
                    f'90th={qerror_stats["q_90"]:.4f}  '
                    f'Pearson={corr:.4f}')
    return qerror_stats, float(corr), per_q, all_preds, all_labels

def train(model, train_ds, val_ds, loss_fn, args):
    loader = DataLoader(train_ds, batch_size=args.bs, shuffle=True, collate_fn=mscn_cypher_collate)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=args.sch_decay, patience=5
    )
    best_val_loss, best_ckpt_name = float('inf'), None
    train_losses, val_losses = [], []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(model, loader, optimizer, loss_fn, args.device, args.clip_size)
        val_loss   = _compute_val_loss(model, val_ds, args.bs, loss_fn, args.device)
        scheduler.step(val_loss)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        logger.info(f'Epoch {epoch:3d}/{args.epochs}  train={train_loss:.6f}  val={val_loss:.6f}  {time.time()-t0:.1f}s')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_name = f'mscn_baseline_best_{timestamp}.pt'
            torch.save({'model': model.state_dict(), 'epoch': epoch, 'val_loss': val_loss},
                       os.path.join(args.newpath, ckpt_name))
            best_ckpt_name = ckpt_name
            logger.info(f'  → Saved best checkpoint: {ckpt_name}')

    # 保存 loss 曲线
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(1, len(train_losses)+1), train_losses, label='Train Loss')
    ax.plot(range(1, len(val_losses)+1),   val_losses,   label='Val Loss')
    ax.set_xlabel('Epoch'); ax.set_ylabel('MSE Loss')
    ax.set_title('MSCN Cypher Baseline - Training Loss')
    ax.legend(); ax.grid(True, alpha=0.3); fig.tight_layout()
    fig.savefig(os.path.join(args.newpath, f'loss_curve_{timestamp}.png'), dpi=150)
    plt.close(fig)
    logger.info(f'Loss curve saved.')
    return model, best_ckpt_name

# ── 主流程 ────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    seed_everything()
    data_path = os.path.join(_PROJECT_ROOT, 'data') + os.sep

    encoding_ckpt = torch.load(data_path + 'encoding.pt', weights_only=False)
    encoding = encoding_ckpt['encoding']

    card_norm = {}  # 用 dict 传递 min_val/max_val（可变对象，方便跨 Dataset 共享）

    train_df = pd.read_csv(data_path + 'train_by_para_v2_same_500.csv')
    val_df   = pd.read_csv(data_path + 'val_by_para_v2_same_500.csv')
    test_df  = pd.read_csv(data_path + 'test_by_para_v2_same_500.csv')
    logger.info(f'Train size: {len(train_df)}, Val size: {len(val_df)}, Test size: {len(test_df)}')

    # ── 第一步：先把 train/val/test 全部解析，让 encoding 词典完全稳定 ──────
    # 这样三个集合的特征维度天然一致，不需要截断
    train_ds = MscnCypherDataset(train_df, encoding, card_norm, reset_norm=True,
                                 label_field=TASK)
    val_ds   = MscnCypherDataset(val_df,   encoding, card_norm, reset_norm=False,
                                 label_field=TASK)
    test_ds  = MscnCypherDataset(test_df,  encoding, card_norm, reset_norm=False,
                                 label_field=TASK)

    # encoding 词典完全稳定后，重新计算最终特征维度
    final_sample_feats, final_predicate_feats, final_join_feats = compute_feature_dims(encoding)
    logger.info(f'[Norm] log(Card) min={card_norm["min_val"]:.4f}, max={card_norm["max_val"]:.4f}')
    logger.info(f'[Dim] sample_feats={final_sample_feats}, '
                f'predicate_feats={final_predicate_feats}, '
                f'join_feats={final_join_feats}')

    # ── 第二步：统一 pad 到三个集合的最大序列长度，并对齐到最终特征维度 ──────
    max_labels = max(train_ds.max_num_labels,     val_ds.max_num_labels,     test_ds.max_num_labels)
    max_preds  = max(train_ds.max_num_predicates, val_ds.max_num_predicates, test_ds.max_num_predicates)
    max_joins  = max(train_ds.max_num_joins,      val_ds.max_num_joins,      test_ds.max_num_joins)
    for ds in [train_ds, val_ds, test_ds]:
        ds.pad_to(max_labels, max_preds, max_joins,
                  sample_feats=final_sample_feats,
                  predicate_feats=final_predicate_feats,
                  join_feats=final_join_feats)

    # ── 第三步：用最终稳定的特征维度初始化模型 ───────────────────────────────
    model = SetConv(
        sample_feats=final_sample_feats,
        predicate_feats=final_predicate_feats,
        join_feats=final_join_feats,
        hid_units=args.hid_units,
    ).to(args.device)

    loss_fn = nn.MSELoss()
    logger.info('Training started.')
    t0 = time.time()
    model, best_ckpt_name = train(model, train_ds, val_ds, loss_fn, args)
    logger.info(f'Training completed in {time.time()-t0:.1f}s.')

    # 加载最优 checkpoint 评估测试集
    ckpt = torch.load(os.path.join(args.newpath, best_ckpt_name), map_location='cpu')
    model.load_state_dict(ckpt['model'])
    model = model.to(args.device)

    logger.info('Evaluating on test set:')
    test_scores, corr, per_q, all_preds, all_labels = evaluate(
        model, test_ds, args.bs, card_norm['min_val'], card_norm['max_val'], args.device, prints=True
    )

    true_col_name = 'true_card' if TASK == 'card' else 'true_time'
    result_df = pd.DataFrame({
        'id':           list(test_df['id']),
        'pred':         all_preds,
        true_col_name:  all_labels,
        'q_error':      per_q,
        'dataset':      list(test_df['dataset']) if 'dataset' in test_df.columns else ['unknown'] * len(all_preds),
        'src_file':     list(test_df['src_file']) if 'src_file' in test_df.columns else ['unknown'] * len(all_preds),
    })
    result_csv = os.path.join(args.newpath, f'mscn_{TASK}_baseline_test_results_{timestamp}.csv')
    result_df.to_csv(result_csv, index=False)
    logger.info(f'Test results saved to: {result_csv}')