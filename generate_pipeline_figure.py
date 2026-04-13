"""
论文Pipeline图生成脚本 - 专业学术风格
参考SIGMOD/VLDB等顶会论文风格
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, Polygon, FancyArrowPatch, Arc
from matplotlib.lines import Line2D
import numpy as np

# 设置字体
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'stix'

# 专业配色方案 - 简洁学术风格
COLORS = {
    'blue': '#4A90D9',        # 主色调蓝
    'light_blue': '#E8F1FA',  # 浅蓝背景
    'orange': '#F5A623',      # 橙色强调
    'light_orange': '#FFF4E6',
    'green': '#7ED321',       # 绿色
    'light_green': '#F0F9E8',
    'purple': '#9B59B6',      # 紫色
    'light_purple': '#F5EEF8',
    'gray': '#9B9B9B',        # 灰色边框
    'dark_gray': '#4A4A4A',   # 深灰文字
    'white': '#FFFFFF',
    'arrow': '#666666',
}


def draw_box(ax, x, y, width, height, color, border_color='#9B9B9B', text='',
             fontsize=9, fontweight='normal', text_color='#4A4A4A', linewidth=1.5):
    """绘制圆角矩形框"""
    box = FancyBboxPatch((x, y), width, height,
                         boxstyle="round,pad=0.02,rounding_size=0.15",
                         facecolor=color, edgecolor=border_color,
                         linewidth=linewidth)
    ax.add_patch(box)
    if text:
        ax.text(x + width/2, y + height/2, text,
                ha='center', va='center', fontsize=fontsize,
                fontweight=fontweight, color=text_color)
    return box


def draw_dashed_box(ax, x, y, width, height, label='', label_pos='top'):
    """绘制虚线框（用于分组）"""
    box = FancyBboxPatch((x, y), width, height,
                         boxstyle="round,pad=0.02,rounding_size=0.1",
                         facecolor='none', edgecolor=COLORS['gray'],
                         linewidth=1.5, linestyle='--')
    ax.add_patch(box)
    if label:
        if label_pos == 'top':
            ax.text(x + width/2, y + height + 0.15, label,
                    ha='center', va='bottom', fontsize=10, fontweight='bold',
                    color=COLORS['dark_gray'])
        else:
            ax.text(x - 0.1, y + height/2, label,
                    ha='right', va='center', fontsize=10, fontweight='bold',
                    color=COLORS['dark_gray'], rotation=90)
    return box


def draw_cylinder(ax, x, y, width, height, color, text='', fontsize=8):
    """绘制数据库圆柱图标"""
    # 主体矩形
    rect = FancyBboxPatch((x, y + height*0.1), width, height*0.8,
                          boxstyle="round,pad=0,rounding_size=0.05",
                          facecolor=color, edgecolor=COLORS['gray'],
                          linewidth=1)
    ax.add_patch(rect)

    # 顶部椭圆
    ellipse_top = mpatches.Ellipse((x + width/2, y + height*0.9), width, height*0.2,
                                    facecolor=color, edgecolor=COLORS['gray'], linewidth=1)
    ax.add_patch(ellipse_top)

    # 底部椭圆（只显示前半部分）
    ellipse_bottom = Arc((x + width/2, y + height*0.1), width, height*0.2,
                         theta1=180, theta2=360, color=COLORS['gray'], linewidth=1)
    ax.add_patch(ellipse_bottom)

    if text:
        ax.text(x + width/2, y + height/2, text, ha='center', va='center',
                fontsize=fontsize, color=COLORS['dark_gray'])


def draw_tree_node(ax, x, y, radius=0.15, color=COLORS['blue'], text=''):
    """绘制树节点"""
    circle = Circle((x, y), radius, facecolor=color, edgecolor='white', linewidth=1)
    ax.add_patch(circle)
    if text:
        ax.text(x, y, text, ha='center', va='center', fontsize=6, color='white', fontweight='bold')


def draw_tree_structure(ax, x, y, scale=0.3):
    """绘制小型树结构示意"""
    # 根节点
    draw_tree_node(ax, x, y + 0.8*scale, scale*0.4, COLORS['blue'])
    # 第二层
    draw_tree_node(ax, x - 0.4*scale, y + 0.3*scale, scale*0.35, COLORS['blue'])
    draw_tree_node(ax, x + 0.4*scale, y + 0.3*scale, scale*0.35, COLORS['blue'])
    # 第三层
    draw_tree_node(ax, x - 0.6*scale, y - 0.2*scale, scale*0.3, COLORS['orange'])
    draw_tree_node(ax, x - 0.2*scale, y - 0.2*scale, scale*0.3, COLORS['orange'])
    draw_tree_node(ax, x + 0.2*scale, y - 0.2*scale, scale*0.3, COLORS['orange'])
    draw_tree_node(ax, x + 0.6*scale, y - 0.2*scale, scale*0.3, COLORS['orange'])

    # 连接线
    ax.plot([x, x-0.4*scale], [y+0.5*scale, y+0.55*scale], 'k-', lw=0.8)
    ax.plot([x, x+0.4*scale], [y+0.5*scale, y+0.55*scale], 'k-', lw=0.8)
    ax.plot([x-0.4*scale, x-0.6*scale], [y+0.1*scale, y-0.0*scale], 'k-', lw=0.8)
    ax.plot([x-0.4*scale, x-0.2*scale], [y+0.1*scale, y-0.0*scale], 'k-', lw=0.8)
    ax.plot([x+0.4*scale, x+0.2*scale], [y+0.1*scale, y-0.0*scale], 'k-', lw=0.8)
    ax.plot([x+0.4*scale, x+0.6*scale], [y+0.1*scale, y-0.0*scale], 'k-', lw=0.8)


def draw_graph_structure(ax, x, y, scale=0.3):
    """绘制图结构示意"""
    # 中心节点
    draw_tree_node(ax, x, y, scale*0.35, COLORS['purple'])
    # 周围节点
    positions = [(0, 0.5), (0.43, 0.25), (0.43, -0.25), (0, -0.5), (-0.43, -0.25), (-0.43, 0.25)]
    for px, py in positions:
        draw_tree_node(ax, x + px*scale, y + py*scale, scale*0.25, COLORS['green'])
        # 连接线
        ax.plot([x, x + px*scale], [y, y + py*scale], 'k-', lw=0.6, alpha=0.5)


def draw_transformer_block(ax, x, y, width, height):
    """绘制Transformer块"""
    # 外框
    box = FancyBboxPatch((x, y), width, height,
                         boxstyle="round,pad=0.02,rounding_size=0.1",
                         facecolor=COLORS['light_purple'], edgecolor=COLORS['purple'],
                         linewidth=1.5)
    ax.add_patch(box)

    # 内部组件
    block_h = height / 5
    components = ['Multi-Head\nAttention', 'Add & Norm', 'Feed Forward', 'Add & Norm']
    colors = [COLORS['light_blue'], COLORS['white'], COLORS['light_orange'], COLORS['white']]

    for i, (comp, c) in enumerate(zip(components, colors)):
        y_pos = y + height - (i+1)*block_h
        inner_box = FancyBboxPatch((x + 0.05, y_pos + 0.02), width - 0.1, block_h - 0.04,
                                   boxstyle="round,pad=0,rounding_size=0.05",
                                   facecolor=c, edgecolor=COLORS['gray'], linewidth=0.5)
        ax.add_patch(inner_box)
        ax.text(x + width/2, y_pos + block_h/2, comp, ha='center', va='center',
                fontsize=6, color=COLORS['dark_gray'])


def draw_arrow(ax, start, end, color=None, style='->', lw=1.5, label=''):
    """绘制箭头"""
    if color is None:
        color = COLORS['arrow']
    ax.annotate('', xy=end, xytext=start,
                arrowprops=dict(arrowstyle=style, color=color, lw=lw),
                zorder=1)
    if label:
        mid_x = (start[0] + end[0]) / 2
        mid_y = (start[1] + end[1]) / 2
        ax.text(mid_x, mid_y + 0.1, label, ha='center', va='bottom',
                fontsize=7, color=COLORS['dark_gray'])


def create_main_figure():
    """创建主框架图"""
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')

    # ==================== 标题 ====================
    ax.text(7, 7.7, 'Figure 1: Framework of QueryFormer',
            ha='center', va='center', fontsize=12, fontweight='bold')

    # ==================== 左侧：输入部分 ====================
    # 数据库圆柱
    draw_cylinder(ax, 0.3, 5.5, 1.2, 1.0, COLORS['light_blue'], 'DBMS', fontsize=8)

    # 输入框
    draw_box(ax, 0.3, 4.0, 1.2, 0.8, COLORS['light_orange'], COLORS['orange'],
             'Cypher\nQuery', fontsize=8)

    # 直方图文件
    draw_box(ax, 0.3, 2.8, 1.2, 0.8, COLORS['light_green'], COLORS['green'],
             'Histogram\nFile', fontsize=8)

    # ==================== 解析模块 ====================
    draw_dashed_box(ax, 1.9, 2.3, 2.0, 4.5, 'Parser')

    # Query Plan Parser
    draw_box(ax, 2.1, 5.5, 1.6, 1.0, COLORS['light_blue'], COLORS['blue'],
             'Query Plan\nParser', fontsize=8)

    # Plan Tree 结构示意
    ax.text(2.9, 5.2, 'Plan Tree', ha='center', va='top', fontsize=7, color=COLORS['gray'])
    draw_tree_structure(ax, 2.9, 4.2, scale=0.5)

    # Schema Extractor
    draw_box(ax, 2.1, 2.5, 1.6, 1.0, COLORS['light_green'], COLORS['green'],
             'Schema\nExtractor', fontsize=8)

    # 箭头：输入 -> 解析
    draw_arrow(ax, (1.5, 6.0), (2.1, 6.0))
    draw_arrow(ax, (1.5, 4.4), (2.1, 5.8))
    draw_arrow(ax, (1.5, 3.2), (2.1, 3.0))

    # ==================== 特征提取模块 ====================
    draw_dashed_box(ax, 4.2, 2.3, 2.0, 4.5, 'Feature Extraction')

    # 特征列表
    features = [
        ('Node Type', 5.8),
        ('Join Info', 5.2),
        ('Filters', 4.6),
        ('Histogram', 4.0),
        ('Table Label', 3.4),
        ('Subschema', 2.8),
    ]
    for feat, y in features:
        draw_box(ax, 4.4, y - 0.25, 1.6, 0.5, COLORS['white'], COLORS['gray'],
                 feat, fontsize=7, linewidth=1)

    # 箭头：解析 -> 特征
    draw_arrow(ax, (3.9, 4.5), (4.2, 4.5))

    # ==================== 嵌入模块 ====================
    draw_dashed_box(ax, 6.5, 2.3, 2.0, 4.5, 'Embedding Layer')

    # FeatureEmbed
    draw_box(ax, 6.7, 5.5, 1.6, 1.0, COLORS['light_purple'], COLORS['purple'],
             'Feature\nEmbedding', fontsize=8)

    # 嵌入类型
    embeddings = ['Type Emb', 'Join Emb', 'Filter Emb', 'Hist Emb', 'Table Emb', 'Schema Emb']
    for i, emb in enumerate(embeddings):
        y_pos = 5.0 - i * 0.45
        ax.text(7.5, y_pos, emb, ha='center', va='center', fontsize=6,
                color=COLORS['dark_gray'])

    # 箭头：特征 -> 嵌入
    draw_arrow(ax, (6.2, 4.5), (6.5, 4.5))

    # ==================== Transformer编码器 ====================
    draw_dashed_box(ax, 8.8, 2.3, 2.5, 4.5, 'Transformer Encoder')

    # Super Token
    draw_box(ax, 9.0, 5.8, 2.1, 0.6, COLORS['light_orange'], COLORS['orange'],
             '[CLS] Token', fontsize=7)

    # Transformer块 ×8
    draw_transformer_block(ax, 9.0, 3.0, 2.1, 2.6)

    # ×8 标注
    ax.text(11.2, 4.3, '×8', ha='left', va='center', fontsize=9, fontweight='bold',
            color=COLORS['purple'])

    # 循环箭头
    ax.annotate('', xy=(11.1, 5.4), xytext=(11.1, 3.2),
                arrowprops=dict(arrowstyle='<->', color=COLORS['purple'],
                               lw=1.2, connectionstyle='arc3,rad=0.3'))

    # 箭头：嵌入 -> Transformer
    draw_arrow(ax, (8.5, 4.5), (8.8, 4.5))

    # ==================== 预测头 ====================
    draw_box(ax, 9.3, 2.5, 1.5, 0.6, COLORS['light_blue'], COLORS['blue'],
             'Prediction\nMLP', fontsize=7)

    # 箭头：Transformer -> 预测
    draw_arrow(ax, (10.05, 3.0), (10.05, 3.1))

    # ==================== 输出 ====================
    draw_box(ax, 11.8, 3.5, 1.5, 1.5, COLORS['light_orange'], COLORS['orange'],
             'Cost\nEstimation', fontsize=9, fontweight='bold')

    # 箭头：预测 -> 输出
    draw_arrow(ax, (10.8, 2.8), (11.8, 4.0))

    # ==================== Subschema GAT模块（底部） ====================
    draw_dashed_box(ax, 0.3, 0.3, 5.5, 1.6, 'Subschema Graph Encoder')

    # Graph Construction
    draw_box(ax, 0.5, 0.6, 1.4, 1.0, COLORS['light_green'], COLORS['green'],
             'Graph\nConstruction', fontsize=7)

    # 图结构示意
    draw_graph_structure(ax, 2.8, 1.1, scale=0.4)

    # GATv2
    draw_box(ax, 3.8, 0.6, 1.0, 1.0, COLORS['light_purple'], COLORS['purple'],
             'GATv2', fontsize=7)

    # Pooling
    draw_box(ax, 5.0, 0.6, 0.6, 1.0, COLORS['light_blue'], COLORS['blue'],
             'Pool', fontsize=6)

    # 箭头
    draw_arrow(ax, (1.9, 1.1), (2.3, 1.1))
    draw_arrow(ax, (3.3, 1.1), (3.8, 1.1))
    draw_arrow(ax, (4.8, 1.1), (5.0, 1.1))

    # 虚线连接到特征提取
    ax.plot([5.6, 6.0], [1.6, 2.3], 'k--', lw=1, alpha=0.6)
    ax.text(5.8, 2.0, '256-d', ha='left', va='bottom', fontsize=6, color=COLORS['gray'])

    # ==================== 图例 ====================
    legend_elements = [
        mpatches.Patch(facecolor=COLORS['light_blue'], edgecolor=COLORS['blue'], label='Processing'),
        mpatches.Patch(facecolor=COLORS['light_orange'], edgecolor=COLORS['orange'], label='Input/Output'),
        mpatches.Patch(facecolor=COLORS['light_green'], edgecolor=COLORS['green'], label='Extraction'),
        mpatches.Patch(facecolor=COLORS['light_purple'], edgecolor=COLORS['purple'], label='Encoding'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=7,
              framealpha=0.9, edgecolor=COLORS['gray'])

    plt.tight_layout()
    return fig


def create_compact_figure():
    """创建紧凑版框架图"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis('off')

    # 标题
    ax.text(6, 4.7, 'Figure 1: QueryFormer Architecture',
            ha='center', va='center', fontsize=11, fontweight='bold')

    # ===== 输入 =====
    draw_cylinder(ax, 0.3, 2.8, 1.0, 0.9, COLORS['light_blue'], 'DB', fontsize=7)
    draw_box(ax, 0.3, 1.6, 1.0, 0.8, COLORS['light_orange'], COLORS['orange'],
             'Query', fontsize=7)

    # ===== Parser =====
    draw_dashed_box(ax, 1.6, 1.2, 1.8, 2.8, 'Parser')
    draw_tree_structure(ax, 2.5, 2.3, scale=0.4)

    # ===== Feature =====
    draw_dashed_box(ax, 3.7, 1.2, 1.8, 2.8, 'Features')
    ax.text(4.6, 3.6, 'Type | Join | Filter\nHist | Table | Schema',
            ha='center', va='center', fontsize=6, color=COLORS['dark_gray'])

    # ===== Embedding =====
    draw_dashed_box(ax, 5.8, 1.2, 1.8, 2.8, 'Embedding')
    draw_box(ax, 6.0, 2.5, 1.4, 1.0, COLORS['light_purple'], COLORS['purple'],
             'Feature\nEmbed', fontsize=7)

    # ===== Transformer =====
    draw_dashed_box(ax, 7.9, 1.2, 2.2, 2.8, 'Encoder')
    draw_transformer_block(ax, 8.1, 1.5, 1.8, 2.2)
    ax.text(10.0, 2.6, '×8', ha='left', va='center', fontsize=8, fontweight='bold',
            color=COLORS['purple'])

    # ===== Output =====
    draw_box(ax, 10.4, 2.0, 1.2, 1.2, COLORS['light_orange'], COLORS['orange'],
             'Cost\nEst.', fontsize=8, fontweight='bold')

    # 箭头
    draw_arrow(ax, (1.3, 3.2), (1.6, 3.0))
    draw_arrow(ax, (3.4, 2.6), (3.7, 2.6))
    draw_arrow(ax, (5.5, 2.6), (5.8, 2.6))
    draw_arrow(ax, (7.6, 2.6), (7.9, 2.6))
    draw_arrow(ax, (10.1, 2.6), (10.4, 2.6))

    plt.tight_layout()
    return fig


def create_detailed_figure():
    """创建详细技术图"""
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # 标题
    ax.text(7.5, 9.6, 'Figure 2: Detailed Architecture of QueryFormer',
            ha='center', va='center', fontsize=12, fontweight='bold')

    # ==================== (a) Data Input ====================
    ax.text(0.5, 9.0, '(a) Data Input', ha='left', va='center',
            fontsize=10, fontweight='bold', color=COLORS['dark_gray'])

    draw_cylinder(ax, 0.5, 7.5, 1.3, 1.2, COLORS['light_blue'], 'Neo4j\nDBMS', fontsize=7)
    draw_box(ax, 0.5, 6.0, 1.3, 0.9, COLORS['light_orange'], COLORS['orange'],
             'Cypher\nQuery', fontsize=7)
    draw_box(ax, 0.5, 4.8, 1.3, 0.9, COLORS['light_green'], COLORS['green'],
             'Histogram\nStats', fontsize=7)
    draw_box(ax, 0.5, 3.6, 1.3, 0.9, COLORS['white'], COLORS['gray'],
             'Encoding\nDict', fontsize=7)

    # ==================== (b) Plan Tree Construction ====================
    ax.text(2.5, 9.0, '(b) Plan Tree Construction', ha='left', va='center',
            fontsize=10, fontweight='bold', color=COLORS['dark_gray'])

    draw_dashed_box(ax, 2.3, 3.2, 2.2, 5.4, '')

    # Parser
    draw_box(ax, 2.5, 7.5, 1.8, 0.9, COLORS['light_blue'], COLORS['blue'],
             'Plan Parser', fontsize=8)

    # 树结构详细展示
    ax.text(3.4, 7.2, 'TreeNode Structure:', ha='center', va='top',
            fontsize=7, color=COLORS['gray'])

    # 示例树
    draw_tree_node(ax, 3.4, 6.5, 0.2, COLORS['blue'], 'R')
    draw_tree_node(ax, 2.8, 5.8, 0.18, COLORS['blue'], 'J')
    draw_tree_node(ax, 4.0, 5.8, 0.18, COLORS['blue'], 'S')
    draw_tree_node(ax, 2.4, 5.1, 0.15, COLORS['orange'], 'F')
    draw_tree_node(ax, 3.2, 5.1, 0.15, COLORS['orange'], 'F')
    draw_tree_node(ax, 3.8, 5.1, 0.15, COLORS['green'], 'T')
    draw_tree_node(ax, 4.2, 5.1, 0.15, COLORS['green'], 'T')

    # 连接线
    for (x1, y1), (x2, y2) in [((3.4, 6.3), (2.8, 5.98)), ((3.4, 6.3), (4.0, 5.98)),
                               ((2.8, 5.62), (2.4, 5.25)), ((2.8, 5.62), (3.2, 5.25)),
                               ((4.0, 5.62), (3.8, 5.25)), ((4.0, 5.62), (4.2, 5.25))]:
        ax.plot([x1, x2], [y1, y2], 'k-', lw=0.8)

    # 节点特征说明
    features_text = 'R: Root  J: Join  S: Scan\nF: Filter  T: Table'
    ax.text(3.4, 4.5, features_text, ha='center', va='top',
            fontsize=6, color=COLORS['gray'])

    # Subschema Graph Builder
    draw_box(ax, 2.5, 3.4, 1.8, 0.9, COLORS['light_green'], COLORS['green'],
             'Subschema\nGraph Builder', fontsize=7)

    # ==================== (c) Feature Encoding ====================
    ax.text(5.0, 9.0, '(c) Feature Encoding', ha='left', va='center',
            fontsize=10, fontweight='bold', color=COLORS['dark_gray'])

    draw_dashed_box(ax, 4.8, 3.2, 2.5, 5.4, '')

    # Feature Embedding
    draw_box(ax, 5.0, 7.5, 2.1, 0.9, COLORS['light_purple'], COLORS['purple'],
             'Feature Embedding', fontsize=8)

    # 特征维度说明
    feature_dims = [
        ('Type Embedding', '64-d'),
        ('Join Embedding', '64-d'),
        ('Filter Embedding', '72-d'),
        ('Histogram Embedding', '64-d'),
        ('Table Embedding', '64-d'),
        ('Subschema Embedding', '64-d'),
    ]
    for i, (name, dim) in enumerate(feature_dims):
        y_pos = 6.8 - i * 0.55
        ax.text(5.1, y_pos, name, ha='left', va='center', fontsize=6, color=COLORS['dark_gray'])
        ax.text(7.0, y_pos, dim, ha='right', va='center', fontsize=6, color=COLORS['gray'])

    # Concatenation
    draw_box(ax, 5.0, 3.4, 2.1, 0.7, COLORS['white'], COLORS['gray'],
             'Concatenation\n(392-d)', fontsize=7)

    # ==================== (d) Transformer Encoder ====================
    ax.text(7.8, 9.0, '(d) Transformer Encoder', ha='left', va='center',
            fontsize=10, fontweight='bold', color=COLORS['dark_gray'])

    draw_dashed_box(ax, 7.6, 3.2, 3.0, 5.4, '')

    # Super Token
    draw_box(ax, 7.8, 7.8, 2.6, 0.6, COLORS['light_orange'], COLORS['orange'],
             '[CLS] Token (Learnable)', fontsize=7)

    # Position & Height Encoding
    draw_box(ax, 7.8, 7.0, 1.25, 0.6, COLORS['light_blue'], COLORS['blue'],
             'Rel. Pos.', fontsize=6)
    draw_box(ax, 9.15, 7.0, 1.25, 0.6, COLORS['light_blue'], COLORS['blue'],
             'Height', fontsize=6)

    # Transformer Layers
    draw_transformer_block(ax, 7.8, 3.8, 2.6, 3.0)

    # ×8 标注
    ax.annotate('', xy=(10.5, 6.5), xytext=(10.5, 4.0),
                arrowprops=dict(arrowstyle='<->', color=COLORS['purple'],
                               lw=1.5, connectionstyle='arc3,rad=0.3'))
    ax.text(10.7, 5.25, '×8\nlayers', ha='left', va='center', fontsize=7,
            fontweight='bold', color=COLORS['purple'])

    # ==================== (e) Prediction ====================
    ax.text(11.0, 9.0, '(e) Prediction', ha='left', va='center',
            fontsize=10, fontweight='bold', color=COLORS['dark_gray'])

    draw_dashed_box(ax, 10.9, 3.2, 3.5, 5.4, '')

    # Prediction MLP
    draw_box(ax, 11.1, 7.0, 3.1, 1.4, COLORS['light_purple'], COLORS['purple'],
             'Prediction MLP\n(392 → 128 → 1)\nSigmoid Output', fontsize=7)

    # 输出
    draw_box(ax, 11.1, 4.8, 1.4, 1.0, COLORS['light_orange'], COLORS['orange'],
             'Cost\nPrediction', fontsize=8, fontweight='bold')
    draw_box(ax, 12.6, 4.8, 1.4, 1.0, COLORS['light_orange'], COLORS['orange'],
             'Card\nPrediction', fontsize=8, fontweight='bold')

    # Loss
    draw_box(ax, 11.1, 3.4, 3.1, 1.0, COLORS['white'], COLORS['gray'],
             'MSE Loss + Adam Optimizer\nLearning Rate: 0.001', fontsize=6)

    # ==================== 箭头连接 ====================
    # (a) -> (b)
    draw_arrow(ax, (1.8, 7.0), (2.3, 7.0))
    draw_arrow(ax, (1.8, 5.5), (2.3, 6.0))
    draw_arrow(ax, (1.8, 4.3), (2.3, 4.0))

    # (b) -> (c)
    draw_arrow(ax, (4.5, 5.5), (4.8, 5.5))

    # (c) -> (d)
    draw_arrow(ax, (7.3, 5.5), (7.6, 5.5))

    # (d) -> (e)
    draw_arrow(ax, (10.6, 5.5), (10.9, 5.5))

    # 内部箭头
    draw_arrow(ax, (9.1, 7.8), (9.1, 7.0))
    draw_arrow(ax, (9.1, 3.8), (9.1, 3.5))
    draw_arrow(ax, (9.1, 3.5), (10.9, 5.3))
    draw_arrow(ax, (12.65, 7.0), (12.65, 5.8))
    draw_arrow(ax, (11.8, 7.0), (11.8, 5.8))

    plt.tight_layout()
    return fig


if __name__ == '__main__':
    import os

    output_dir = '/Users/yyh/Desktop/coding/research/pipeline_figures'
    os.makedirs(output_dir, exist_ok=True)

    # 生成主图
    print("Generating main framework figure...")
    fig1 = create_main_figure()
    fig1.savefig(os.path.join(output_dir, 'framework_main.png'), dpi=300, bbox_inches='tight',
                 facecolor='white', edgecolor='none')
    fig1.savefig(os.path.join(output_dir, 'framework_main.pdf'), bbox_inches='tight',
                 facecolor='white', edgecolor='none')
    plt.close(fig1)

    # 生成紧凑版
    print("Generating compact figure...")
    fig2 = create_compact_figure()
    fig2.savefig(os.path.join(output_dir, 'framework_compact.png'), dpi=300, bbox_inches='tight',
                 facecolor='white', edgecolor='none')
    fig2.savefig(os.path.join(output_dir, 'framework_compact.pdf'), bbox_inches='tight',
                 facecolor='white', edgecolor='none')
    plt.close(fig2)

    # 生成详细版
    print("Generating detailed figure...")
    fig3 = create_detailed_figure()
    fig3.savefig(os.path.join(output_dir, 'framework_detailed.png'), dpi=300, bbox_inches='tight',
                 facecolor='white', edgecolor='none')
    fig3.savefig(os.path.join(output_dir, 'framework_detailed.pdf'), bbox_inches='tight',
                 facecolor='white', edgecolor='none')
    plt.close(fig3)

    print(f"\nAll figures saved to: {output_dir}")
    print("Generated files:")
    print("  - framework_main.png/pdf      (Main framework)")
    print("  - framework_compact.png/pdf   (Compact version)")
    print("  - framework_detailed.png/pdf  (Detailed architecture)")
