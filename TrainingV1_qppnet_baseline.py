# %%
"""
QPPNet Cypher Baseline 训练脚本。

使用 QPPNet 的每算子 NeuralUnit 架构预测 Execution Time。
代价信号：Execution Time（毫秒），不额外归一化（QPPNet 原版风格）。
评估指标：Q-error（与其他 baseline 保持一致）。

关键流程（skill 文档 7.7）：
  先解析 train/val/test 全部，让 encoding 词典完全稳定，
  再用最终维度初始化模型，并对所有数据集调用 rebuild。
"""

import os
import sys
import time
import datetime
import logging

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── 路径设置（从项目根目录运行）────────────────────────────────────────────────
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

_QPPNET_CYPHER_DIR = os.path.join(_PROJECT_ROOT, 'baseline', 'QPPNet', 'QPPNet_Cypher')
if _QPPNET_CYPHER_DIR not in sys.path:
    sys.path.insert(0, _QPPNET_CYPHER_DIR)

from model.util import seed_everything
from baseline.QPPNet.QPPNet_Cypher.qppnet_cypher_dataset import (
    QPPNetCypherDataset, compute_node_feature_dim
)
from baseline.QPPNet.QPPNet_Cypher.qppnet_cypher_model import QPPNetCypher

# ── 超参数配置 ────────────────────────────────────────────────────────────────

class Args:
    batch_size    = 32        # 每次训练采样的查询数量
    lr            = 1e-3
    epochs        = 200
    use_scheduler = False
    step_size     = 1000
    gamma         = 0.95
    device        = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    newpath       = './results/job-full/qppnet_baseline/'
    save_freq     = 50        # 每隔多少 epoch 保存一次 checkpoint

args = Args()
os.makedirs(args.newpath, exist_ok=True)

# ── 日志 ──────────────────────────────────────────────────────────────────────

def get_logger(log_path: str) -> logging.Logger:
    logger = logging.getLogger('qppnet_cypher_baseline')
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

def train_epoch(model: QPPNetCypher, train_ds: QPPNetCypherDataset,
                batch_size: int):
    """采样一批查询，执行一步训练，返回 loss 和预测/真实值。"""
    samp_dicts = train_ds.sample_batch(batch_size)
    loss, pred_times, true_times = model.optimize_parameters(samp_dicts)
    return loss, np.array(pred_times), np.array(true_times)


def evaluate(model: QPPNetCypher, ds: QPPNetCypherDataset,
             prints: bool = False):
    """在整个数据集上评估，返回 Q-error 统计、Pearson 相关系数、各样本 Q-error、预测值、真实值。
    
    pred_times 是归一化空间的预测值（NeuralUnit 输出，已除以 SCALE=1000），
    true_times 也是归一化空间的值（total_time / 1000），
    Q-error 计算前均乘以 SCALE 反归一化回原始毫秒单位。
    """
    loss, pred_times, true_times = model.evaluate(ds.groups)
    # 反归一化：log1p 空间 → 原始毫秒（expm1）
    pred_times  = np.expm1(np.array(pred_times,  dtype=np.float64))
    true_times  = np.expm1(np.array(true_times,  dtype=np.float64))

    qerror_stats, per_sample_q = compute_qerror(pred_times, true_times)
    corr = get_corr(pred_times, true_times)

    if prints:
        logger.info(f'  Q-error median={qerror_stats["q_median"]:.4f}  '
                    f'mean={qerror_stats["q_mean"]:.4f}  '
                    f'90th={qerror_stats["q_90"]:.4f}  '
                    f'Pearson={corr:.4f}')

    return qerror_stats, corr, per_sample_q, pred_times, true_times


def _save_loss_plot(train_losses, val_qmedians, save_dir, timestamp):
    """保存训练 loss 和验证集 Q-error 曲线图。"""
    epochs = range(1, len(train_losses) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(epochs, train_losses, linewidth=1.5)
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Train Loss')
    ax1.set_title('QPPNet Cypher Baseline - Train Loss')
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, val_qmedians, linewidth=1.5, color='orange')
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Val Q-error (median)')
    ax2.set_title('QPPNet Cypher Baseline - Val Q-error')
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    plot_path = os.path.join(save_dir, f'loss_curve_{timestamp}.png')
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    logger.info(f'Loss curve saved to: {plot_path}')


def train(model: QPPNetCypher, train_ds: QPPNetCypherDataset,
          val_ds: QPPNetCypherDataset, args):
    """训练主循环，保存验证集最优 checkpoint，返回最优 checkpoint tag。"""
    best_val_qmedian = float('inf')
    best_ckpt_tag    = None
    train_losses     = []
    val_qmedians     = []

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        train_loss, _, _ = train_epoch(model, train_ds, args.batch_size)
        train_losses.append(train_loss)

        # 每 10 个 epoch 评估一次验证集
        if epoch % 10 == 0 or epoch == 1:
            val_scores, val_corr, _, _, _ = evaluate(model, val_ds)
            val_qmedian = val_scores['q_median']
            val_qmedians.append(val_qmedian)

            elapsed = time.time() - epoch_start
            logger.info(f'Epoch {epoch:3d}/{args.epochs}  '
                        f'train_loss={train_loss:.6f}  '
                        f'val_q_median={val_qmedian:.4f}  '
                        f'time={elapsed:.1f}s')

            if val_qmedian < best_val_qmedian:
                best_val_qmedian = val_qmedian
                ckpt_tag = f'best_{timestamp}'
                model.save(ckpt_tag)
                best_ckpt_tag = ckpt_tag
                logger.info(f'  → Saved best checkpoint: {ckpt_tag}')
        else:
            val_qmedians.append(val_qmedians[-1] if val_qmedians else float('inf'))

        if epoch % args.save_freq == 0:
            model.save(f'epoch{epoch}_{timestamp}')

    _save_loss_plot(train_losses, val_qmedians, args.newpath, timestamp)
    return model, best_ckpt_tag

# ── 主流程 ────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    seed_everything()

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

    # 第一步：解析所有数据集，让 encoding 词典完全稳定
    # 训练集：fit_normalizer=True，计算 mean_range_dict
    train_ds = QPPNetCypherDataset(train_df, encoding, fit_normalizer=True)
    logger.info(f'[Train] {len(train_ds)} queries, '
                f'{len(train_ds.groups)} template groups')

    # val/test：fit_normalizer=False，复用训练集的 mean_range_dict
    val_ds  = QPPNetCypherDataset(val_df,  encoding,
                                  mean_range_dict=train_ds.mean_range_dict,
                                  fit_normalizer=False)
    test_ds = QPPNetCypherDataset(test_df, encoding,
                                  mean_range_dict=train_ds.mean_range_dict,
                                  fit_normalizer=False)

    # 第二步：encoding 词典已稳定，计算最终节点特征维度
    node_feat_dim = compute_node_feature_dim(encoding)
    logger.info(f'[Dim] node_feat_dim={node_feat_dim} '
                f'(type={len(encoding.type2idx)}, table={len(encoding.table2idx)}, '
                f'join={len(encoding.join2idx)})')

    # 第三步：encoding 词典已稳定，重新计算 mean_range_dict（基于最终维度）
    # 必须在 encoding 词典稳定后重新计算，否则 mean_vec/range_vec 维度是旧的，
    # pad 后的 range_vec=1 导致部分特征未归一化，值过大引发 dying ReLU
    final_mean_range_dict = train_ds._compute_mean_range(train_ds._groups_raw, encoding)
    logger.info(f'[Normalizer] Recomputed mean_range_dict with final encoding '
                f'(node_feat_dim={node_feat_dim})')

    # 第四步：用最终维度和重新计算的 mean_range_dict 重建所有数据集的 samp_dict
    for ds_name, ds in [('train', train_ds), ('val', val_ds), ('test', test_ds)]:
        ds.rebuild(node_feat_dim, final_mean_range_dict)
        logger.info(f'[Rebuild] {ds_name} rebuilt, '
                    f'{len(ds.groups)} template groups')

    # ── 构建模型 ──────────────────────────────────────────────────────────────
    operator_types = list(encoding.type2idx.keys())
    model = QPPNetCypher(
        node_feat_dim  = node_feat_dim,
        operator_types = operator_types,
        lr             = args.lr,
        use_scheduler  = args.use_scheduler,
        step_size      = args.step_size,
        gamma          = args.gamma,
        save_dir       = args.newpath,
    )
    logger.info(f'Model initialized with {len(operator_types)} operator types.')

    # ── 训练 ──────────────────────────────────────────────────────────────────
    logger.info('Training started.')
    start_time = time.time()
    model, best_ckpt_tag = train(model, train_ds, val_ds, args)
    logger.info(f'Training completed in {time.time() - start_time:.1f}s.')

    # ── 加载最优 checkpoint 并在测试集上评估 ──────────────────────────────────
    if best_ckpt_tag:
        model.load(best_ckpt_tag)
        logger.info(f'Loaded best checkpoint: {best_ckpt_tag}')

    logger.info('Evaluating on test set:')
    test_scores, corr, per_sample_q, all_preds, all_true = evaluate(
        model, test_ds, prints=True
    )

    # 保存预测结果 CSV（与其他 baseline 格式一致）
    # model.evaluate(ds.groups) 按 _groups_raw 的遍历顺序输出预测值，
    # get_all_ids_in_group_order() 返回相同顺序的 row_id 列表，保证对齐。
    all_ids = test_ds.get_all_ids_in_group_order()

    # 按 row_id 从 test_df 中查找 dataset / src_file 列（如果存在）
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

    result_df = pd.DataFrame({
        'id':        all_ids[:len(all_preds)],
        'pred':      all_preds,
        'true_time': all_true,
        'q_error':   per_sample_q,
        'dataset':   all_datasets[:len(all_preds)],
        'src_file':  all_src_files[:len(all_preds)],
    })
    result_csv_path = os.path.join(args.newpath, f'qppnet_baseline_test_results_{timestamp}.csv')
    result_df.to_csv(result_csv_path, index=False)
    logger.info(f'Test results saved to: {result_csv_path}')
