"""
查询计划聚类实验 —— 对齐 PreQR 论文评估框架
评估三种查询计划表示方法的聚类质量：
  - QueryFormer (GAT Schema)  ← 你的模型
  - QueryFormer (Random Schema)
  - QueryFormer (QF)

评估指标（对齐 PreQR）：
  - BetaCV（越小越好）：类内平均距离 / 类间平均距离
  - NDCG@10 / NDCG@20（越大越好）：相似性排序准确率

Ground truth：src_file 列（查询模板）
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
from tqdm import tqdm

# 将 myfiles 目录加入路径，使 model 包可以被导入
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model.util import Normalizer
from model.database_util import get_hist_file, collator

# ============================================================
# ★ 你只需要修改这里 ★
# ============================================================
DATA_PATH  = "./data/job/"
TEST_CSV   = "./myfiles/test_by_para_v2_same_500.csv"
ENCODING_PT   = "data/job/encoding.pt"
HISTOGRAM_CSV = "data/job/histogram_string.csv"

# 三个模型的 checkpoint 路径（填写你训练好的 .pt 文件路径）
CHECKPOINT_QUERYFORMER_GAT = "checkpoints/your_queryformer_gat.pt"   # GAT Schema 模型
CHECKPOINT_RANDOMSCHEMA    = "checkpoints/your_randomschema.pt"       # Random Schema 模型
CHECKPOINT_QF              = "checkpoints/your_qf.pt"                 # QF 模型

# 过滤掉样本数少于此值的模板（避免评估偏差）
MIN_TEMPLATE_SAMPLES = 10

# 评估 NDCG 的 K 值
NDCG_K_LIST = [10, 20]

# 批量提取 embedding 时的 batch size
BATCH_SIZE = 128

DEVICE = "cpu"  # 本地无 GPU 时用 cpu；有 GPU 改为 "cuda:0"
# ============================================================


# ============================================================
# 模型超参数（与训练时保持一致）
# ============================================================
class Args:
    embed_size = 64
    ffn_dim    = 128
    head_size  = 12
    n_layers   = 8
    dropout    = 0.1
    pred_hid   = 128


# ============================================================
# Step 1：加载数据
# ============================================================
def load_test_data(test_csv: str, min_samples: int) -> pd.DataFrame:
    df = pd.read_csv(test_csv)
    template_counts = df["src_file"].value_counts()
    valid_templates = template_counts[template_counts >= min_samples].index
    df = df[df["src_file"].isin(valid_templates)].reset_index(drop=True)
    print(f"[Data] 过滤后：{len(df)} 条查询，{df['src_file'].nunique()} 个模板")
    print(f"[Data] 模板分布：\n{df['src_file'].value_counts().to_string()}")
    return df


# ============================================================
# Step 2：提取 embedding
# collator 的签名是 collator((list_of_dicts, labels))，
# 不是标准 DataLoader collate_fn，需要手动分批调用。
# ============================================================
def extract_embeddings_with_hook(model, dataset, batch_size: int, device: str) -> np.ndarray:
    """
    通过 forward hook 截取 QueryFormer 的 super token embedding（output[:,0,:]），
    即 Transformer encoder 最后一层 LayerNorm 之后、pred 头之前的表示向量。

    注意：collator 期望接收 (list_of_dicts, labels) 格式，
    dataset.__getitem__ 返回 (collated_dict, (cost_label, card_label))，
    因此手动分批，不使用 DataLoader。
    """
    model.eval()
    model.to(device)

    captured_embeddings = []

    def hook_fn(module, input, output):
        # output shape: [batch, n_node+1, hidden_dim]
        # output[:,0,:] 是 super token 的表示
        captured_embeddings.append(output[:, 0, :].detach().cpu())

    hook_handle = model.final_ln.register_forward_hook(hook_fn)

    num_samples = len(dataset)
    indices = list(range(num_samples))

    with torch.no_grad():
        for start in tqdm(range(0, num_samples, batch_size), desc="  提取 embedding", leave=False):
            batch_indices = indices[start: start + batch_size]
            # 手动收集每个样本，格式与 collator 期望一致
            samples = [dataset[i] for i in batch_indices]
            # dataset.__getitem__ 返回 (collated_dict, (cost_label, card_label))
            dicts  = [s[0] for s in samples]
            labels = [s[1] for s in samples]
            # collator 接收 (list_of_dicts, labels) 这个 tuple
            batch_data, _ = collator((dicts, labels))
            batch_data = batch_data.to(device)
            _ = model(batch_data)

    hook_handle.remove()

    all_embeddings = torch.cat(captured_embeddings, dim=0).numpy()
    return all_embeddings


def _load_ckpt_state_dict(checkpoint_path: str) -> dict:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    return ckpt["model"] if "model" in ckpt else ckpt


def load_queryformer_gat_model(checkpoint_path: str, args: Args):
    """加载 QueryFormer + GAT Schema 模型（你的模型）"""
    from model.model_transformer_job import QueryFormer
    model = QueryFormer(
        emb_size  = args.embed_size,
        ffn_dim   = args.ffn_dim,
        head_size = args.head_size,
        dropout   = args.dropout,
        n_layers  = args.n_layers,
        use_hist  = False,
        pred_hid  = args.pred_hid,
    )
    model.load_state_dict(_load_ckpt_state_dict(checkpoint_path))
    print(f"  [✓] 加载 checkpoint: {checkpoint_path}")
    return model


def load_randomschema_model(checkpoint_path: str, args: Args):
    """加载 QueryFormer + Random Schema 模型"""
    from model.model_transformer_random import QueryFormer
    model = QueryFormer(
        emb_size  = args.embed_size,
        ffn_dim   = args.ffn_dim,
        head_size = args.head_size,
        dropout   = args.dropout,
        n_layers  = args.n_layers,
        use_hist  = False,
        pred_hid  = args.pred_hid,
    )
    model.load_state_dict(_load_ckpt_state_dict(checkpoint_path))
    print(f"  [✓] 加载 checkpoint: {checkpoint_path}")
    return model


def load_qf_model(checkpoint_path: str, args: Args):
    """加载 QueryFormer (QF) 模型"""
    from model.model_transformer_qf import QueryFormer
    model = QueryFormer(
        emb_size  = args.embed_size,
        ffn_dim   = args.ffn_dim,
        head_size = args.head_size,
        dropout   = args.dropout,
        n_layers  = args.n_layers,
        use_hist  = True,
        pred_hid  = args.pred_hid,
    )
    model.load_state_dict(_load_ckpt_state_dict(checkpoint_path))
    print(f"  [✓] 加载 checkpoint: {checkpoint_path}")
    return model


# ============================================================
# Step 3：评估指标
# ============================================================
def compute_beta_cv(similarity_matrix: np.ndarray, labels: np.ndarray) -> float:
    """
    BetaCV = 类内平均距离 / 类间平均距离（越小越好）
    距离 = 1 - cosine_similarity
    """
    distance_matrix = 1.0 - similarity_matrix
    unique_labels = np.unique(labels)

    intra_distances = []
    inter_distances = []

    for label in unique_labels:
        mask = labels == label
        indices_in  = np.where(mask)[0]
        indices_out = np.where(~mask)[0]

        # 类内距离（上三角，避免重复计算）
        for i in range(len(indices_in)):
            for j in range(i + 1, len(indices_in)):
                intra_distances.append(distance_matrix[indices_in[i], indices_in[j]])

        # 类间距离
        for i in indices_in:
            for j in indices_out:
                inter_distances.append(distance_matrix[i, j])

    mean_intra = np.mean(intra_distances) if intra_distances else 0.0
    mean_inter = np.mean(inter_distances) if inter_distances else 1.0
    beta_cv = mean_intra / mean_inter if mean_inter > 0 else float("inf")
    return float(beta_cv)


def compute_ndcg_at_k(similarity_matrix: np.ndarray, labels: np.ndarray, k: int) -> float:
    """
    NDCG@K（越大越好）
    对每条查询，按相似度排序其他查询，同模板的视为相关（relevance=1），计算 NDCG@K
    """
    n = len(labels)
    ndcg_scores = []

    for i in range(n):
        sims = similarity_matrix[i].copy()
        sims[i] = -1.0  # 排除自身

        sorted_indices = np.argsort(sims)[::-1][:k]

        num_relevant = int(np.sum(labels == labels[i])) - 1  # 排除自身
        if num_relevant == 0:
            continue

        # DCG
        dcg = 0.0
        for rank, idx in enumerate(sorted_indices):
            if labels[idx] == labels[i]:
                dcg += 1.0 / np.log2(rank + 2)

        # IDCG（理想情况：前 min(num_relevant, k) 个都是相关的）
        ideal_k = min(num_relevant, k)
        idcg = sum(1.0 / np.log2(rank + 2) for rank in range(ideal_k))

        ndcg_scores.append(dcg / idcg if idcg > 0 else 0.0)

    return float(np.mean(ndcg_scores)) if ndcg_scores else 0.0


def evaluate_method(method_name: str, embeddings: np.ndarray, labels: np.ndarray, ndcg_k_list: list) -> dict:
    """计算一种方法的所有评估指标"""
    print(f"\n[Eval] {method_name}（embedding shape: {embeddings.shape}）")

    sim_matrix = cosine_similarity(embeddings)

    beta_cv = compute_beta_cv(sim_matrix, labels)
    print(f"  BetaCV  = {beta_cv:.4f}")

    result = {"Method": method_name, "BetaCV": round(beta_cv, 4)}
    for k in ndcg_k_list:
        ndcg = compute_ndcg_at_k(sim_matrix, labels, k)
        result[f"NDCG@{k}"] = round(ndcg, 4)
        print(f"  NDCG@{k} = {ndcg:.4f}")

    return result


# ============================================================
# Step 4：t-SNE 可视化
# ============================================================
def plot_tsne(embeddings: np.ndarray, labels: np.ndarray, method_name: str, output_path: str):
    print(f"\n[Viz] 生成 t-SNE 可视化：{method_name}")
    unique_labels = sorted(np.unique(labels))
    num_classes = len(unique_labels)
    label_to_int = {label: i for i, label in enumerate(unique_labels)}
    int_labels = np.array([label_to_int[l] for l in labels])

    perplexity = min(30, len(embeddings) // 4)
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    reduced = tsne.fit_transform(embeddings)

    colors = cm.get_cmap("tab20", num_classes)
    plt.figure(figsize=(14, 10))
    for i, label in enumerate(unique_labels):
        mask = int_labels == i
        plt.scatter(
            reduced[mask, 0], reduced[mask, 1],
            c=[colors(i)],
            label=label.replace("queries_", "").replace(".csv", ""),
            alpha=0.6, s=20,
        )

    plt.title(f"t-SNE Visualization — {method_name}", fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [✓] 保存至 {output_path}")


# ============================================================
# 主流程
# ============================================================
def main():
    args = Args()
    os.makedirs("results/clustering", exist_ok=True)

    # ---------- Step 1：加载数据 ----------
    print("\n" + "=" * 60)
    print("Step 1: 加载测试集")
    print("=" * 60)
    df = load_test_data(TEST_CSV, MIN_TEMPLATE_SAMPLES)
    labels = df["src_file"].values

    # 加载公共依赖
    encoding_ckpt = torch.load(ENCODING_PT, map_location="cpu")
    encoding = encoding_ckpt["encoding"]
    hist_file = get_hist_file(HISTOGRAM_CSV)
    table_sample = None

    all_results = []
    all_embeddings = {}

    # ---- 方法 1：QueryFormer + GAT Schema（你的模型）----
    print("\n" + "=" * 60)
    print("Step 2a: QueryFormer (GAT Schema) — 你的模型")
    print("=" * 60)
    from model.dataset_schema_gat import PlanTreeDataset as PlanTreeDataset_GAT
    cost_norm_gat = Normalizer()
    card_norm_gat = Normalizer()
    test_ds_gat = PlanTreeDataset_GAT(df, None, encoding, hist_file, card_norm_gat, cost_norm_gat, "cost", table_sample)
    model_gat = load_queryformer_gat_model(CHECKPOINT_QUERYFORMER_GAT, args)
    emb_gat = extract_embeddings_with_hook(model_gat, test_ds_gat, BATCH_SIZE, DEVICE)
    all_embeddings["QueryFormer (GAT Schema)"] = emb_gat

    # ---- 方法 2：QueryFormer + Random Schema ----
    print("\n" + "=" * 60)
    print("Step 2b: QueryFormer (Random Schema)")
    print("=" * 60)
    from model.dataset_schema_random import PlanTreeDataset as PlanTreeDataset_Random
    cost_norm_rand = Normalizer()
    card_norm_rand = Normalizer()
    test_ds_random = PlanTreeDataset_Random(df, None, encoding, hist_file, card_norm_rand, cost_norm_rand, "cost", table_sample)
    model_random = load_randomschema_model(CHECKPOINT_RANDOMSCHEMA, args)
    emb_random = extract_embeddings_with_hook(model_random, test_ds_random, BATCH_SIZE, DEVICE)
    all_embeddings["QueryFormer (Random Schema)"] = emb_random

    # ---- 方法 3：QueryFormer (QF) ----
    print("\n" + "=" * 60)
    print("Step 2c: QueryFormer (QF)")
    print("=" * 60)
    from model.dataset_qf import PlanTreeDataset as PlanTreeDataset_QF
    cost_norm_qf = Normalizer()
    card_norm_qf = Normalizer()
    test_ds_qf = PlanTreeDataset_QF(df, None, encoding, hist_file, card_norm_qf, cost_norm_qf, "cost", table_sample)
    model_qf = load_qf_model(CHECKPOINT_QF, args)
    emb_qf = extract_embeddings_with_hook(model_qf, test_ds_qf, BATCH_SIZE, DEVICE)
    all_embeddings["QueryFormer (QF)"] = emb_qf

    # ---------- Step 3：统一评估 ----------
    print("\n" + "=" * 60)
    print("Step 3: 评估指标计算")
    print("=" * 60)
    for method_name, embeddings in all_embeddings.items():
        result = evaluate_method(method_name, embeddings, labels, NDCG_K_LIST)
        all_results.append(result)

    # ---------- Step 4：打印对比表格 ----------
    print("\n" + "=" * 60)
    print("Step 4: 对比结果汇总（对齐 PreQR Table 7）")
    print("=" * 60)
    results_df = pd.DataFrame(all_results)
    print(results_df.to_string(index=False))
    results_df.to_csv("results/clustering/clustering_results.csv", index=False)
    print("\n[✓] 结果已保存至 results/clustering/clustering_results.csv")

    # ---------- Step 5：t-SNE 可视化 ----------
    print("\n" + "=" * 60)
    print("Step 5: t-SNE 可视化")
    print("=" * 60)
    for method_name, embeddings in all_embeddings.items():
        safe_name = method_name.replace(" ", "_").replace("(", "").replace(")", "")
        output_path = f"results/clustering/tsne_{safe_name}.png"
        plot_tsne(embeddings, labels, method_name, output_path)

    print("\n[完成] 所有实验结束！")


if __name__ == "__main__":
    main()
