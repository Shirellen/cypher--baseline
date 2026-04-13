# %%
"""
LEON Cypher Baseline 训练脚本。

使用 LEON 的 TreeConvolution 作为 plan encoder，接 Prediction MLP 预测 Execution Time。
代价信号：Execution Time（秒），取 log1p 后用 Normalizer 归一化到 [0,1]。
评估指标：Q-error（与 TrainingV1_cypher_cost_job.py 保持一致）。
"""

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
matplotlib.use('Agg')  # 非交互式后端，避免无 GUI 环境报错
import matplotlib.pyplot as plt

# ── 路径设置（从项目根目录运行）────────────────────────────────────────────────
# 1. 项目根目录加入 sys.path，使 model/ 可被导入
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# 2. baseline/LEON/ 加入 sys.path，使 LEON 内部的 `from util import ...` 能正确解析
_LEON_DIR = os.path.join(_PROJECT_ROOT, 'baseline', 'LEON')
if _LEON_DIR not in sys.path:
    sys.path.insert(0, _LEON_DIR)

from model.util import Normalizer, seed_everything
from baseline.LEON.leon_cypher_dataset import LeonCypherDataset, leon_cypher_collate
from baseline.LEON.leon_cypher_model import LeonCypherModel


# ── 超参数配置 ────────────────────────────────────────────────────────────────

class Args:
    bs = 64                    # batch size（TreeConv 计算量大，比 QueryFormer 小）
    lr = 0.001
    epochs = 100
    clip_size = 50             # 梯度裁剪阈值
    tree_out_dim = 128         # TreeConv 输出的 plan embedding 维度
    mlp_hidden_dim = 256       # Prediction MLP 隐藏层维度
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    newpath = './results/job-full/leon_baseline/'
    to_predict = 'cost'        # 预测目标（此处为 Execution Time，与 cost 流程一致）
    sch_decay = 0.6            # 学习率衰减因子（每 epoch 后按 val loss 调整）

args = Args()

if not os.path.exists(args.newpath):
    os.makedirs(args.newpath)


# ── 日志 ──────────────────────────────────────────────────────────────────────

def get_logger(log_path: str) -> logging.Logger:
    logger = logging.getLogger('leon_cypher_baseline')
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


# ── Q-error 计算（与 trainer_cost_job.py 保持一致）───────────────────────────

def compute_qerror(preds_unnorm: np.ndarray, labels_unnorm: np.ndarray):
    preds = np.clip(np.asarray(preds_unnorm, dtype=np.float64), 1e-6, None)
    labels = np.clip(np.asarray(labels_unnorm, dtype=np.float64), 1e-6, None)
    q = np.maximum(preds / labels, labels / preds)
    both_zero = (preds < 1e-6) & (labels < 1e-6)
    q[both_zero] = 1.0
    return {
        'q_median': float(np.median(q)),
        'q_mean':   float(np.mean(q)),
        'q_90':     float(np.percentile(q, 90)),
    }, q


def get_corr(preds: np.ndarray, labels: np.ndarray) -> float:
    preds = np.clip(preds, 1e-10, 1e10)
    labels = np.clip(labels, 1e-10, 1e10)
    corr, _ = pearsonr(np.log(preds), np.log(labels))
    return float(corr)


# ── 训练与评估 ────────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, loss_fn, device, clip_size):
    model.train()
    total_loss = 0.0
    for query_feats, trees, indexes, labels in loader:
        query_feats = query_feats.to(device)
        trees = trees.to(device)
        indexes = indexes.to(device)
        labels = labels.to(device).float().unsqueeze(1)  # [B, 1]

        preds = model(query_feats, trees, indexes)        # [B, 1]
        loss = loss_fn(preds, labels)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip_size)
        optimizer.step()

        total_loss += loss.item() * len(labels)

    return total_loss / len(loader.dataset)


def evaluate(model, ds: LeonCypherDataset, bs: int, cost_norm: Normalizer, device: str, prints: bool = False):
    """
    评估模型，返回 Q-error 统计、Pearson 相关系数、各样本 Q-error、预测值、真实值。
    与 trainer_cost_job.py 中的 evaluate 接口对齐。
    """
    model.eval()
    all_preds_norm = []
    all_labels_norm = []

    loader = DataLoader(
        ds,
        batch_size=bs,
        shuffle=False,
        collate_fn=leon_cypher_collate,
    )

    with torch.no_grad():
        for query_feats, trees, indexes, labels in loader:
            query_feats = query_feats.to(device)
            trees = trees.to(device)
            indexes = indexes.to(device)

            preds = model(query_feats, trees, indexes).squeeze(1).cpu().numpy()
            all_preds_norm.extend(preds.tolist())
            all_labels_norm.extend(labels.numpy().tolist())

    # 反归一化：先 unnormalize（[0,1] → log1p 空间），再 expm1（→ 秒）
    all_preds_log = cost_norm.unnormalize_labels(np.array(all_preds_norm))
    all_labels_log = cost_norm.unnormalize_labels(np.array(all_labels_norm))
    all_preds_sec = np.expm1(all_preds_log)
    all_labels_sec = np.expm1(all_labels_log)

    qerror_stats, per_sample_q = compute_qerror(all_preds_sec, all_labels_sec)
    corr = get_corr(all_preds_sec, all_labels_sec)

    if prints:
        logger.info(f'  Q-error median={qerror_stats["q_median"]:.4f}  '
                    f'mean={qerror_stats["q_mean"]:.4f}  '
                    f'90th={qerror_stats["q_90"]:.4f}  '
                    f'Pearson={corr:.4f}')

    return qerror_stats, corr, per_sample_q, all_preds_sec, all_labels_sec


def train(model, train_ds, val_ds, loss_fn, cost_norm, args):
    """
    训练主循环，保存验证集最优 checkpoint 和 loss 曲线图，返回最优 checkpoint 文件名。
    """
    train_loader = DataLoader(
        train_ds,
        batch_size=args.bs,
        shuffle=True,
        collate_fn=leon_cypher_collate,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=args.sch_decay, patience=5, verbose=False
    )

    best_val_loss = float('inf')
    best_ckpt_name = None
    train_losses = []
    val_losses = []

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, args.device, args.clip_size)

        # 验证集 loss（归一化空间）
        val_loss = _compute_val_loss(model, val_ds, args.bs, loss_fn, args.device)
        scheduler.step(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        elapsed = time.time() - epoch_start
        logger.info(f'Epoch {epoch:3d}/{args.epochs}  '
                    f'train_loss={train_loss:.6f}  val_loss={val_loss:.6f}  '
                    f'time={elapsed:.1f}s')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_name = f'leon_baseline_best_{timestamp}.pt'
            ckpt_path = os.path.join(args.newpath, ckpt_name)
            torch.save({'model': model.state_dict(), 'epoch': epoch, 'val_loss': val_loss}, ckpt_path)
            best_ckpt_name = ckpt_name
            logger.info(f'  → Saved best checkpoint: {ckpt_name}')

    # 保存 loss 曲线图
    _save_loss_plot(train_losses, val_losses, args.newpath, timestamp)

    return model, best_ckpt_name


def _save_loss_plot(train_losses, val_losses, save_dir, timestamp):
    """保存训练/验证 loss 曲线图到 save_dir。"""
    epochs = range(1, len(train_losses) + 1)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs, train_losses, label='Train Loss', linewidth=1.5)
    ax.plot(epochs, val_losses,   label='Val Loss',   linewidth=1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax.set_title('LEON Cypher Baseline - Training Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plot_path = os.path.join(save_dir, f'loss_curve_{timestamp}.png')
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    logger.info(f'Loss curve saved to: {plot_path}')


def _compute_val_loss(model, val_ds, bs, loss_fn, device) -> float:
    model.eval()
    total_loss = 0.0
    loader = DataLoader(val_ds, batch_size=bs, shuffle=False, collate_fn=leon_cypher_collate)
    with torch.no_grad():
        for query_feats, trees, indexes, labels in loader:
            query_feats = query_feats.to(device)
            trees = trees.to(device)
            indexes = indexes.to(device)
            labels = labels.to(device).float().unsqueeze(1)
            preds = model(query_feats, trees, indexes)
            total_loss += loss_fn(preds, labels).item() * len(labels)
    return total_loss / len(val_ds)


# ── 主流程 ────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    seed_everything()

    data_path = 'data/'

    # 加载 encoding（与 TrainingV1_cypher_cost_job.py 保持一致）
    encoding_ckpt = torch.load('data/encoding.pt', weights_only=False)
    encoding = encoding_ckpt['encoding']

    cost_norm = Normalizer()

    # ── 加载数据集 ────────────────────────────────────────────────────────────
    train_df = pd.read_csv(data_path + 'train_by_para_v2_same_500.csv')
    val_df   = pd.read_csv(data_path + 'val_by_para_v2_same_500.csv')

    logger.info(f'Train size: {len(train_df)}, Val size: {len(val_df)}')

    # 训练集：reset_norm=True，用训练集数据确定 Normalizer 的 min/max
    train_ds = LeonCypherDataset(train_df, encoding, cost_norm, reset_norm=True)
    logger.info(f'[Norm] log(ExecTime) min={cost_norm.mini:.4f}, max={cost_norm.maxi:.4f}')
    logger.info(f'[Dim] node_feature_dim={train_ds.node_feature_dim}, query_feature_dim={train_ds.query_feature_dim}')

    # 验证集：reset_norm=False，复用训练集的 min/max
    val_ds = LeonCypherDataset(val_df, encoding, cost_norm, reset_norm=False)
    # encoding 词典在 val_ds 解析后可能继续扩充，统一 pad 到 train_ds 的维度
    val_ds.pad_query_feats_to(train_ds.query_feature_dim)

    # ── 构建模型 ──────────────────────────────────────────────────────────────
    # node_feature_dim / query_feature_dim 依赖 encoding，从 train_ds 动态获取
    model = LeonCypherModel(
        query_feature_dim=train_ds.query_feature_dim,
        node_feature_dim=train_ds.node_feature_dim,
        tree_out_dim=args.tree_out_dim,
        mlp_hidden_dim=args.mlp_hidden_dim,
    )

    # 设备回退：CUDA 不可用时自动切换到 CPU
    try:
        model = model.to(args.device)
    except RuntimeError as cuda_error:
        if 'CUDA' in str(cuda_error) or 'NVIDIA' in str(cuda_error):
            logger.warning(f'CUDA 初始化失败，回退到 CPU：{cuda_error}')
            args.device = 'cpu'
            model = model.to(args.device)
        else:
            raise

    loss_fn = nn.MSELoss()

    # ── 训练 ──────────────────────────────────────────────────────────────────
    logger.info('Training started.')
    start_time = time.time()
    model, best_ckpt_name = train(model, train_ds, val_ds, loss_fn, cost_norm, args)
    logger.info(f'Training completed in {time.time() - start_time:.1f}s.')

    # ── 加载最优 checkpoint 并在测试集上评估 ──────────────────────────────────
    best_ckpt_path = os.path.join(args.newpath, best_ckpt_name)
    ckpt = torch.load(best_ckpt_path, map_location='cpu')
    model.load_state_dict(ckpt['model'])
    model = model.to(args.device)
    torch.cuda.empty_cache()
    logger.info(f'Loaded best checkpoint: {best_ckpt_path}')

    test_csv = data_path + 'test_by_para_v2_same_500.csv'
    test_df = pd.read_csv(test_csv)
    test_ds = LeonCypherDataset(test_df, encoding, cost_norm, reset_norm=False)
    test_ds.pad_query_feats_to(train_ds.query_feature_dim)

    logger.info('Evaluating on test set:')
    test_scores, corr, per_sample_q, all_preds, all_true = evaluate(
        model, test_ds, args.bs, cost_norm, args.device, prints=True
    )

    # 保存预测结果 CSV（与 TrainingV1_cypher_cost_job.py 格式一致）
    result_df = pd.DataFrame({
        'id':          list(test_df['id']),
        'pred':        all_preds,
        'true_time':   all_true,
        'q_error':     per_sample_q,
        'dataset':     list(test_df.get('dataset', ['unknown'] * len(all_preds))),
        'src_file':    list(test_df.get('src_file', ['unknown'] * len(all_preds))),
    })
    result_csv_path = os.path.join(args.newpath, f'leon_baseline_test_results_{timestamp}.csv')
    result_df.to_csv(result_csv_path, index=False)
    logger.info(f'Test results saved to: {result_csv_path}')
