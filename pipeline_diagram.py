"""
论文Pipeline图生成脚本
生成查询表示学习模型的完整架构图
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

def draw_pipeline():
    """绘制论文pipeline图"""
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # 定义颜色方案
    color_input = '#E8F4F8'
    color_encoding = '#B3E5FC'
    color_core = '#4FC3F7'
    color_embedding = '#0288D1'
    color_downstream = '#01579B'
    color_output = '#FFA726'
    
    # ==================== 标题 ====================
    ax.text(8, 9.5, 'Query Representation Learning Pipeline', 
            ha='center', va='center', fontsize=20, fontweight='bold')
    
    # ==================== 第一层：输入层 ====================
    # Cypher查询输入
    input_box = FancyBboxPatch((0.5, 7.5), 3, 1.2, 
                               boxstyle="round,pad=0.1", 
                               facecolor=color_input, 
                               edgecolor='black', linewidth=2)
    ax.add_patch(input_box)
    ax.text(2, 8.1, 'Cypher Query', ha='center', va='center', 
            fontsize=14, fontweight='bold')
    
    # ==================== 第二层：编码层 ====================
    # 查询计划解析
    plan_box = FancyBboxPatch((0.5, 5.5), 3, 1.2, 
                              boxstyle="round,pad=0.1", 
                              facecolor=color_encoding, 
                              edgecolor='black', linewidth=2)
    ax.add_patch(plan_box)
    ax.text(2, 6.1, 'Query Plan\nParser', ha='center', va='center', 
            fontsize=12, fontweight='bold')
    
    # Schema提取
    schema_box = FancyBboxPatch((4.5, 5.5), 3, 1.2, 
                                boxstyle="round,pad=0.1", 
                                facecolor=color_encoding, 
                                edgecolor='black', linewidth=2)
    ax.add_patch(schema_box)
    ax.text(6, 6.1, 'Schema\nExtractor', ha='center', va='center', 
            fontsize=12, fontweight='bold')
    
    # ==================== 第三层：核心模型层 ====================
    # 查询计划树编码器
    tree_encoder = FancyBboxPatch((0.5, 3), 3.5, 1.5, 
                                  boxstyle="round,pad=0.1", 
                                  facecolor=color_core, 
                                  edgecolor='black', linewidth=2)
    ax.add_patch(tree_encoder)
    ax.text(2.25, 4.0, 'Plan Tree Encoder', ha='center', va='center', 
            fontsize=13, fontweight='bold')
    ax.text(2.25, 3.5, '(Transformer)', ha='center', va='center', 
            fontsize=11, style='italic')
    
    # Schema图编码器
    schema_encoder = FancyBboxPatch((5, 3), 3.5, 1.5, 
                                    boxstyle="round,pad=0.1", 
                                    facecolor=color_core, 
                                    edgecolor='black', linewidth=2)
    ax.add_patch(schema_encoder)
    ax.text(6.75, 4.0, 'Schema Graph Encoder', ha='center', va='center', 
            fontsize=13, fontweight='bold')
    ax.text(6.75, 3.5, '(GAT)', ha='center', va='center', 
            fontsize=11, style='italic')
    
    # ==================== 第四层：融合层 ====================
    fusion_box = FancyBboxPatch((3, 1), 3, 1.2, 
                                boxstyle="round,pad=0.1", 
                                facecolor=color_embedding, 
                                edgecolor='black', linewidth=2)
    ax.add_patch(fusion_box)
    ax.text(4.5, 1.6, 'Feature Fusion', ha='center', va='center', 
            fontsize=13, fontweight='bold', color='white')
    
    # ==================== 第五层：表示输出 ====================
    embedding_box = FancyBboxPatch((7, 1), 3.5, 1.2, 
                                   boxstyle="round,pad=0.1", 
                                   facecolor=color_embedding, 
                                   edgecolor='black', linewidth=3)
    ax.add_patch(embedding_box)
    ax.text(8.75, 1.6, 'Query Embedding', ha='center', va='center', 
            fontsize=14, fontweight='bold', color='white')
    
    # ==================== 第六层：下游任务 ====================
    # 任务1：时间估计
    task1_box = FancyBboxPatch((11.5, 7.5), 3.5, 1.2, 
                               boxstyle="round,pad=0.1", 
                               facecolor=color_downstream, 
                               edgecolor='black', linewidth=2)
    ax.add_patch(task1_box)
    ax.text(13.25, 8.1, 'Cost Estimation', ha='center', va='center', 
            fontsize=12, fontweight='bold', color='white')
    
    # 任务2：基数估计
    task2_box = FancyBboxPatch((11.5, 5.5), 3.5, 1.2, 
                               boxstyle="round,pad=0.1", 
                               facecolor=color_downstream, 
                               edgecolor='black', linewidth=2)
    ax.add_patch(task2_box)
    ax.text(13.25, 6.1, 'Cardinality\nEstimation', ha='center', va='center', 
            fontsize=12, fontweight='bold', color='white')
    
    # 任务3：查询聚类
    task3_box = FancyBboxPatch((11.5, 3.5), 3.5, 1.2, 
                               boxstyle="round,pad=0.1", 
                               facecolor=color_downstream, 
                               edgecolor='black', linewidth=2)
    ax.add_patch(task3_box)
    ax.text(13.25, 4.1, 'Query Clustering', ha='center', va='center', 
            fontsize=12, fontweight='bold', color='white')
    
    # ==================== 绘制箭头连接 ====================
    arrow_style = "Simple, tail_width=0.5, head_width=4, head_length=8"
    
    # 输入 -> 查询计划解析
    ax.annotate('', xy=(2, 6.7), xytext=(2, 7.5),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    # 输入 -> Schema提取
    ax.annotate('', xy=(6, 6.7), xytext=(6, 7.5),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    ax.plot([2, 6], [8.1, 8.1], 'k-', lw=2)
    ax.plot([6, 6], [8.1, 7.5], 'k-', lw=2)
    
    # 查询计划解析 -> Plan Tree Encoder
    ax.annotate('', xy=(2.25, 4.5), xytext=(2, 5.5),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    # Schema提取 -> Schema Graph Encoder
    ax.annotate('', xy=(6.75, 4.5), xytext=(6, 5.5),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    # Plan Tree Encoder -> Fusion
    ax.annotate('', xy=(3.5, 2.2), xytext=(2.25, 3),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    # Schema Graph Encoder -> Fusion
    ax.annotate('', xy=(5.5, 2.2), xytext=(6.75, 3),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    # Fusion -> Query Embedding
    ax.annotate('', xy=(7, 1.6), xytext=(6, 1.6),
                arrowprops=dict(arrowstyle='->', lw=3, color='black'))
    
    # Query Embedding -> 下游任务
    ax.annotate('', xy=(11.5, 8.1), xytext=(10.5, 1.6),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    ax.plot([10.5, 11], [1.6, 1.6], 'k-', lw=2)
    ax.plot([11, 11], [1.6, 8.1], 'k-', lw=2)
    
    ax.annotate('', xy=(11.5, 6.1), xytext=(11, 6.1),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    ax.annotate('', xy=(11.5, 4.1), xytext=(11, 4.1),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    # ==================== 添加说明文字 ====================
    # 左侧说明
    ax.text(0.3, 8.1, 'Input', ha='left', va='center', 
            fontsize=10, style='italic', color='gray')
    ax.text(0.3, 6.1, 'Encoding', ha='left', va='center', 
            fontsize=10, style='italic', color='gray')
    ax.text(0.3, 3.75, 'Core Model', ha='left', va='center', 
            fontsize=10, style='italic', color='gray')
    ax.text(0.3, 1.6, 'Fusion', ha='left', va='center', 
            fontsize=10, style='italic', color='gray')
    
    # 右侧说明
    ax.text(15.5, 8.1, 'Regression', ha='left', va='center', 
            fontsize=10, style='italic', color='gray')
    ax.text(15.5, 6.1, 'Regression', ha='left', va='center', 
            fontsize=10, style='italic', color='gray')
    ax.text(15.5, 4.1, 'Unsupervised', ha='left', va='center', 
            fontsize=10, style='italic', color='gray')
    
    # 底部说明
    ax.text(8.75, 0.5, 'Unified Query Representation', 
            ha='center', va='center', fontsize=12, 
            style='italic', color=color_embedding)
    
    plt.tight_layout()
    
    # 保存图像
    save_path = '/Users/yyh/Desktop/coding/research/pipeline.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print(f"[Info] Pipeline图已保存: {save_path}")
    
    plt.show()
    
    return save_path

if __name__ == "__main__":
    draw_pipeline()
