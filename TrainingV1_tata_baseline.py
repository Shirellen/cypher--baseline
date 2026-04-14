# %%
"""
TATA Cypher Baseline 训练脚本。

TASK = 'cost'：预测 Execution Time（毫秒），标签 = Execution Time
TASK = 'cost'：预测 Cardinality（行数），标签 = 根节点 Rows
切换只需修改下方 TASK 变量，输出路径、文件名、列名全部自动切换。

代价信号归一化：log(x+0.001) + min-max（与原版 TATA 一致）。
评估指标：Q-error（与其他 baseline 保持一致）。

关键流程（skill 文档 7.7）：
  先解析 train/val/test 全部，让 encoding 词典完全稳定，
  再用最终维度重建所有数据集的特征树，然后初始化模型。
"""

# ── 任务开关（'cost' 或 'card'）──────────────────────────────────────────────
TASK = 'cost'   # ← 改这里切换任务

import os
import sys
import time
import datetime
import logging

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from scipy.stats import pearsonr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── 路径设置（从项目根目录运行）────────────────────────────────────────────────
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

_TATA_CYPHER_DIR = os.path.join(_PROJECT_ROOT, 'baseline', 'tata', 'tata_cypher')
if _TATA_CYPHER_DIR not in sys.path:
    sys.path.insert(0, _TATA_CYPHER_DIR)

from model.util import seed_everything
from baseline.tata.tata_cypher.tata_cypher_dataset import (
    TataCypherDataset, TataLabelNormalizer, tata_collate,
    compute_node_feature_dim, _compute_est_log_min_max
)
from baseline.tata.tata_cypher.tata_cypher_model import TataModel

# ── 超参数配置 ────────────────────────────────────────────────────────────────

class Args:
    batch_size = 128
    lr         = 1e-4
    epochs     = 200
    hid_units  = 256       # Prediction MLP 隐藏层维度
    newpath    = f'./results/job-full/tata_{TASK}_baseline/'
    save_freq  = 50        # 每隔多少 epoch 保存一次 checkpoint

args = Args()
os.makedirs(args.newpath, exist_ok=True)

# ── 日志 ──────────────────────────────────────────────────────────────────────

def get_logger(log_path: str) -> logging.Logger:
    logger = logging.getLogger('tata_cypher_baseline')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(asctime)s] %(message)s', datefmt='%H:%M:%S')
    file_handler = logging.FileHandler(log_path, mode='w')
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger

timestamp = datetime.datetime.now().strftime('%m%d-%H%M%S')
logger = get_logger(os.path.join(args.newpath, f'running_log_{timestamp}.txt'))
logger.info(f'Args: {vars(args)}')

# ── Q-error 计算（与其他 baseline 保持一致）──────────────────────────────────

def compute_qerror(preds: np.ndarray, labels: np.ndarray):
    preds  = np.clip(np.asarray(preds,  dtype=np.float64), 1e-6, None)
    labels = np.clip(np.asarray(labels, dtype=np.float64), 1e-6, None)
    q = np.maximum(preds / labels, labels / preds)
    both_zero = (preds < 1e-6) & (labels < 1e-6)
    q[both_zero] = 1.0
    return {
        'q_median': float(np.median(q)),
        'q_mean':   float(np.mean(q)),
        'q_90':     float(np.percentile(q, 90)),
    }, q

def get_corr(preds: np.ndarray, labels: np.ndarray) -> float:
    preds  = np.clip(preds,  1e-10, 1e10)
    labels = np.clip(labels, 1e-10, 1e10)
    corr, _ = pearsonr(np.log(preds), np.log(labels))
    return float(corr)

# ── 训练与评估 ────────────────────────────────────────────────────────────────

def train_epoch(model: TataModel, loader: DataLoader,
                optimizer, criterion, device: torch.device) -> float:
    """训练一个 epoch，返回平均 loss。"""
    model.train()
    total_loss = 0.0
    total_samples = 0

    for batch, targets in loader:
        batch   = batch.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        preds = model(batch)
        loss  = criterion(preds, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        total_loss    += loss.item() * targets.size(0)
        total_samples += targets.size(0)

    return total_loss / max(total_samples, 1)


def evaluate(model: TataModel, loader: DataLoader,
             cost_norm: TataLabelNormalizer, device: torch.device,
             prints: bool = False):
    """
    在整个数据集上评估，返回 Q-error 统计、Pearson 相关系数、各样本 Q-error、预测值、真实值。
    预测值和真实值均为原始毫秒单位（反归一化后）。
    """
    model.eval()
    all_preds_norm = []
    all_labels_raw = []

    with torch.no_grad():
        for batch, targets in loader:
            batch = batch.to(device)
            preds = model(batch).squeeze().cpu().numpy()
            all_preds_norm.extend(preds.tolist() if preds.ndim > 0 else [float(preds)])
            all_labels_raw.extend(targets.squeeze().numpy().tolist())

    # 反归一化：归一化空间 → 原始毫秒
    preds_ms  = cost_norm.unnormalize_labels(np.array(all_preds_norm))
    labels_ms = cost_norm.unnormalize_labels(np.array(all_labels_raw))

    qerror_stats, per_sample_q = compute_qerror(preds_ms, labels_ms)
    corr = get_corr(preds_ms, labels_ms)

    if prints:
        logger.info(f'  Q-error median={qerror_stats["q_median"]:.4f}  '
                    f'mean={qerror_stats["q_mean"]:.4f}  '
                    f'90th={qerror_stats["q_90"]:.4f}  '
                    f'Pearson={corr:.4f}')

    return qerror_stats, corr, per_sample_q, preds_ms, labels_ms


def _save_loss_plot(train_losses, val_qmedians, save_dir, timestamp):
    """保存训练 loss 和验证集 Q-error 曲线图。"""
    epochs = range(1, len(train_losses) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(epochs, train_losses, linewidth=1.5)
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Train Loss (MSE)')
    ax1.set_title('TATA Cypher Baseline - Train Loss')
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, val_qmedians, linewidth=1.5, color='orange')
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Val Q-error (median)')
    ax2.set_title('TATA Cypher Baseline - Val Q-error')
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    plot_path = os.path.join(save_dir, f'loss_curve_{timestamp}.png')
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    logger.info(f'Loss curve saved to: {plot_path}')


def train(model: TataModel, train_loader: DataLoader, val_loader: DataLoader,
          cost_norm: TataLabelNormalizer, device: torch.device, args):
    """训练主循环，保存验证集最优 checkpoint，返回最优 checkpoint 路径。"""
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)
    criterion = nn.MSELoss()

    best_val_qmedian = float('inf')
    best_ckpt_path   = None
    train_losses     = []
    val_qmedians     = []

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        train_loss  = train_epoch(model, train_loader, optimizer, criterion, device)
        train_losses.append(train_loss)
        scheduler.step()

        # 每 10 个 epoch 评估一次验证集
        if epoch % 10 == 0 or epoch == 1:
            val_scores, val_corr, _, _, _ = evaluate(model, val_loader, cost_norm, device)
            val_qmedian = val_scores['q_median']
            val_qmedians.append(val_qmedian)

            elapsed = time.time() - epoch_start
            logger.info(f'Epoch {epoch:3d}/{args.epochs}  '
                        f'train_loss={train_loss:.6f}  '
                        f'val_q_median={val_qmedian:.4f}  '
                        f'time={elapsed:.1f}s')

            if val_qmedian < best_val_qmedian:
                best_val_qmedian = val_qmedian
                ckpt_path = os.path.join(args.newpath, f'best_{timestamp}.pt')
                torch.save(model.state_dict(), ckpt_path)
                best_ckpt_path = ckpt_path
                logger.info(f'  → Saved best checkpoint: {ckpt_path}')
        else:
            val_qmedians.append(val_qmedians[-1] if val_qmedians else float('inf'))

        if epoch % args.save_freq == 0:
            ckpt_path = os.path.join(args.newpath, f'epoch{epoch}_{timestamp}.pt')
            torch.save(model.state_dict(), ckpt_path)

    _save_loss_plot(train_losses, val_qmedians, args.newpath, timestamp)
    return best_ckpt_path

# ── 主流程 ────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    seed_everything()

    device    = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    data_path = 'data/'

    # 加载 encoding（与其他 baseline 保持一致）
    encoding_ckpt = torch.load(data_path + 'encoding.pt', weights_only=False)
    encoding      = encoding_ckpt['encoding']

    # ── 加载数据集（skill 文档 7.7：先全部解析，让 encoding 词典稳定）────────
    train_df = pd.read_csv(data_path + 'train_by_para_v2_same_500.csv')
    val_df   = pd.read_csv(data_path + 'val_by_para_v2_same_500.csv')
    test_df  = pd.read_csv(data_path + 'test_by_para_v2_same_500.csv')

    logger.info(f'Train size: {len(train_df)}, Val size: {len(val_df)}, '
                f'Test size: {len(test_df)}')

        # 第一步：创建标签归一化器
    cost_norm = TataLabelNormalizer()

    # 第二步：解析所有数据集，让 encoding 词典完全稳定
    # 训练集：reset_norm=True，确定标签归一化参数
    train_ds = TataCypherDataset(train_df, encoding, cost_norm=cost_norm,
                                 reset_norm=True, label_field=TASK)
    logger.info(f'[Train] {len(train_ds)} queries, '
                f'node_feature_dim={train_ds.node_feature_dim}')

    # val/test：reset_norm=False，复用训练集的归一化参数和 EstimatedRows 统计
    val_ds  = TataCypherDataset(val_df,  encoding, cost_norm=cost_norm,
                                est_log_min=train_ds.est_log_min,
                                est_log_max=train_ds.est_log_max,
                                reset_norm=False, label_field=TASK)
    test_ds = TataCypherDataset(test_df, encoding, cost_norm=cost_norm,
                                est_log_min=train_ds.est_log_min,
                                est_log_max=train_ds.est_log_max,
                                reset_norm=False, label_field=TASK)

    # 第三步：encoding 词典已稳定，计算最终节点特征维度
    node_feature_dim = compute_node_feature_dim(encoding)
    logger.info(f'[Dim] node_feature_dim={node_feature_dim} '
                f'(type={len(encoding.type2idx)}, +2 for EstimatedRows×2)')

    # 第四步：用最终维度和全局 EstimatedRows 统计重建所有数据集的特征树
    # 计算全局 EstimatedRows log min/max（train+val+test 合并）
    import json as _json
    all_plan_roots = []
    for df in [train_df, val_df, test_df]:
        for s in df['json']:
            pj = _json.loads(s)
            all_plan_roots.append(pj['Plan'])
    global_est_log_min, global_est_log_max = _compute_est_log_min_max(all_plan_roots)
    logger.info(f'[EstRows] global log min={global_est_log_min:.4f}, '
                f'max={global_est_log_max:.4f}')

    for ds_name, ds in [('train', train_ds), ('val', val_ds), ('test', test_ds)]:
        ds.rebuild(node_feature_dim, global_est_log_min, global_est_log_max)
        logger.info(f'[Rebuild] {ds_name} rebuilt, {len(ds)} queries')

    # ── 构建 DataLoader ───────────────────────────────────────────────────────
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  collate_fn=tata_collate)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, collate_fn=tata_collate)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size,
                              shuffle=False, collate_fn=tata_collate)

    # ── 构建模型 ──────────────────────────────────────────────────────────────
    model = TataModel(in_channels=node_feature_dim, hid_units=args.hid_units).to(device)
    logger.info(f'Model: TataModel(in_channels={node_feature_dim}, hid_units={args.hid_units})')

    # ── 训练 ──────────────────────────────────────────────────────────────────
    logger.info('Training started.')
    start_time = time.time()
    best_ckpt_path = train(model, train_loader, val_loader, cost_norm, device, args)
    logger.info(f'Training completed in {time.time() - start_time:.1f}s.')

    # ── 加载最优 checkpoint 并在测试集上评估 ──────────────────────────────────
    if best_ckpt_path and os.path.exists(best_ckpt_path):
        model.load_state_dict(torch.load(best_ckpt_path, map_location=device))
        logger.info(f'Loaded best checkpoint: {best_ckpt_path}')

    logger.info('Evaluating on test set:')
    test_scores, corr, per_sample_q, all_preds, all_true = evaluate(
        model, test_loader, cost_norm, device, prints=True
    )

    # 保存预测结果 CSV（与其他 baseline 格式一致）
    all_ids = list(test_ds.row_ids)

    if 'dataset' in test_df.columns:
        id_to_dataset = dict(zip(test_df['id'], test_df['dataset']))
        all_datasets  = [id_to_dataset.get(rid, 'unknown') for rid in all_ids]
    else:
        all_datasets = ['unknown'] * len(all_ids)

    if 'src_file' in test_df.columns:
        id_to_src_file = dict(zip(test_df['id'], test_df['src_file']))
        all_src_files  = [id_to_src_file.get(rid, 'unknown') for rid in all_ids]
    else:
        all_src_files = ['unknown'] * len(all_ids)

    true_col_name = 'true_time' if TASK == 'cost' else 'true_card'
    result_df = pd.DataFrame({
        'id':           all_ids[:len(all_preds)],
        'pred':         all_preds,
        true_col_name:  all_true,
        'q_error':      per_sample_q,
        'dataset':      all_datasets[:len(all_preds)],
        'src_file':     all_src_files[:len(all_preds)],
    })
    result_csv_path = os.path.join(args.newpath, f'tata_{TASK}_baseline_test_results_{timestamp}.csv')
    result_df.to_csv(result_csv_path, index=False)
    logger.info(f'Test results saved to: {result_csv_path}')
