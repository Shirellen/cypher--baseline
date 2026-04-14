"""
QPPNet Cypher Baseline 模型。

与原版 QPPNet 完全一致的架构：
  - 每种算子类型对应一个独立的 NeuralUnit（MLP）
  - NeuralUnit 输入 = 节点自身特征 + 所有子节点输出（拼接）
  - NeuralUnit 输出 = output_size=32 维向量，第 0 维为预测时间
  - 损失函数 = squared_diff（与原版一致）

关键差异（Cypher 版本）：
  - dim_dict 动态构建（依赖 encoding 词典），不硬编码
  - 所有算子共用同一节点特征维度（node_feat_dim），不像原版按算子类型分别定义
  - 每种算子的输入维度 = node_feat_dim + num_children × CHILD_OUTPUT_SIZE
"""

import sys
import os

import torch
import torch.nn as nn
from torch.optim import lr_scheduler

_SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_SCRIPT_DIR)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# 子节点输出维度（与 dataset 中 CHILD_OUTPUT_SIZE 保持一致）
CHILD_OUTPUT_SIZE = 32


def squared_diff(output, target):
    """QPPNet 原版损失函数。"""
    return torch.sum((output - target) ** 2)


class NeuralUnit(nn.Module):
    """
    QPPNet 的算子神经单元（与原版完全一致）。
    输入维度 = 节点自身特征维度 + 子节点输出维度之和。
    输出维度 = output_size（默认 32），第 0 维为预测时间。
    """

    def __init__(self, input_dim: int, num_layers: int = 5,
                 hidden_size: int = 128, output_size: int = CHILD_OUTPUT_SIZE):
        super().__init__()
        assert num_layers >= 2
        layers = [nn.Linear(input_dim, hidden_size), nn.ReLU()]
        for _ in range(num_layers - 2):
            layers += [nn.Linear(hidden_size, hidden_size), nn.ReLU()]
        # 输出层不加 ReLU：让输出可以是任意实数，避免 dying ReLU
        # pred_time 在 log1p 空间，理论上 ≥ 0，但训练初期可能为负，ReLU 会截断梯度
        layers += [nn.Linear(hidden_size, output_size)]

        for layer in layers:
            if hasattr(layer, 'weight'):
                nn.init.xavier_uniform_(layer.weight)

        self.dense_block = nn.Sequential(*layers)

    def forward(self, x):
        return self.dense_block(x)


class QPPNetCypher:
    """
    QPPNet Cypher Baseline，与原版 QPPNet 接口完全一致。

    每种算子类型对应一个独立的 NeuralUnit，dim_dict 动态构建。
    前向传播时递归处理 samp_dict 嵌套结构。

    Args:
        node_feat_dim:  节点自身特征维度（由 compute_node_feature_dim(encoding) 计算）
        operator_types: 所有出现过的算子类型列表（从 encoding.type2idx 获取）
        lr:             学习率
        use_scheduler:  是否使用 StepLR 学习率调度
        step_size:      StepLR 的 step_size
        gamma:          StepLR 的 gamma
        save_dir:       checkpoint 保存目录
    """

    def __init__(self, node_feat_dim: int, operator_types: list,
                 lr: float = 1e-3, use_scheduler: bool = False,
                 step_size: int = 1000, gamma: float = 0.95,
                 save_dir: str = './saved_model'):
        self.device = torch.device('cuda:0') if torch.cuda.is_available() \
                      else torch.device('cpu')
        self.node_feat_dim  = node_feat_dim
        self.operator_types = operator_types
        self.save_dir       = save_dir
        os.makedirs(save_dir, exist_ok=True)

        # 为每种算子类型动态构建 NeuralUnit
        # 输入维度在第一次遇到该算子时确定（取决于子节点数量）
        self.units      = {}   # node_type → NeuralUnit
        self.optimizers = {}
        self.schedulers = {}

        # 预先为已知算子类型创建 NeuralUnit（子节点数量未知时先用 0 子节点）
        # 实际使用时会在 _get_or_create_unit 中按需创建
        self._input_dims = {}  # node_type → input_dim（已创建的）

        self.lr            = lr
        self.use_scheduler = use_scheduler
        self.step_size     = step_size
        self.gamma         = gamma

        self.loss_fn   = squared_diff
        self.dummy     = torch.zeros(1).to(self.device)
        self.acc_loss  = {}
        self.best      = float('inf')
        self.total_loss = None

    def _get_or_create_unit(self, node_type: str, num_children: int) -> NeuralUnit:
        """
        按需创建 NeuralUnit（输入维度 = node_feat_dim + num_children × CHILD_OUTPUT_SIZE）。

        同一算子类型在不同查询模板中可能出现不同的子节点数量（如 Filter 可能是叶子也可能有子节点）。
        一旦为某个 node_type 创建了 NeuralUnit，就固定其 input_dim，不再重建，
        避免丢失已训练的权重。input_vec 的维度对齐由调用方（_forward_one_batch）负责。
        """
        if node_type not in self.units:
            input_dim = self.node_feat_dim + num_children * CHILD_OUTPUT_SIZE
            unit = NeuralUnit(input_dim).to(self.device)
            self.units[node_type]      = unit
            self._input_dims[node_type] = input_dim
            optimizer = torch.optim.Adam(unit.parameters(), lr=self.lr)
            self.optimizers[node_type] = optimizer
            if self.use_scheduler:
                self.schedulers[node_type] = lr_scheduler.StepLR(
                    optimizer, step_size=self.step_size, gamma=self.gamma
                )
            if node_type not in self.acc_loss:
                self.acc_loss[node_type] = [self.dummy]

        return self.units[node_type]

    def _forward_one_batch(self, samp_dict: dict):
        """
        递归前向传播一个 samp_dict（与原版 _forward_oneQ_batch 完全一致）。

        Returns:
            output_vec: Tensor [subbatch_size, CHILD_OUTPUT_SIZE]
            pred_time:  Tensor [subbatch_size]
        """
        node_type     = samp_dict['node_type']
        feat_vec      = samp_dict['feat_vec']
        children_plan = samp_dict['children_plan']

        input_vec = torch.from_numpy(feat_vec).to(self.device)  # [B, node_feat_dim]

        # 递归处理子节点，拼接子节点输出
        subplans_time = []
        for child_dict in children_plan:
            child_output, _ = self._forward_one_batch(child_dict)
            if not child_dict['is_subplan']:
                input_vec = torch.cat((input_vec, child_output), dim=1)
            else:
                subplans_time.append(
                    torch.index_select(child_output, 1,
                                       torch.zeros(1, dtype=torch.long).to(self.device))
                )

        # 按需创建 NeuralUnit
        num_children = len(children_plan)
        unit = self._get_or_create_unit(node_type, num_children)

        # 维度对齐（防止 input_vec 维度与 unit 期望不符）
        expected_dim = self._input_dims[node_type]
        if input_vec.size(1) < expected_dim:
            pad = torch.zeros(input_vec.size(0), expected_dim - input_vec.size(1)).to(self.device)
            input_vec = torch.cat((input_vec, pad), dim=1)
        elif input_vec.size(1) > expected_dim:
            input_vec = input_vec[:, :expected_dim]

        output_vec = unit(input_vec)  # [B, CHILD_OUTPUT_SIZE]

        # 第 0 维为预测时间（log1p 空间，理论上 ≥ 0）
        # 用 softplus 保证非负，同时梯度不会像 ReLU 那样在负值区域截断为 0
        pred_time = torch.index_select(output_vec, 1,
                                       torch.zeros(1, dtype=torch.long).to(self.device))
        pred_time = torch.nn.functional.softplus(pred_time)
        if subplans_time:
            subplans_time = [torch.nn.functional.softplus(t) for t in subplans_time]
            pred_time = torch.sum(torch.cat([pred_time] + subplans_time, dim=1), dim=1)
        else:
            pred_time = pred_time.squeeze(1)

        # 计算损失
        target = torch.from_numpy(samp_dict['total_time']).to(self.device)
        loss   = (pred_time - target) ** 2
        if node_type not in self.acc_loss:
            self.acc_loss[node_type] = [self.dummy]
        self.acc_loss[node_type].append(loss)

        return output_vec, pred_time

    def forward(self, samp_dicts: list, is_train: bool = True):
        """
        对一批 samp_dict 进行前向传播，计算总损失。

        Args:
            samp_dicts: list of samp_dict（每个对应一个查询模板组）
            is_train:   是否训练模式

        Returns:
            total_loss: Tensor scalar（训练时）
            all_pred_times, all_true_times: list（评估时）
        """
        total_loss = torch.zeros(1).to(self.device)
        all_pred_times = []
        all_true_times = []

        for samp_dict in samp_dicts:
            # 清空 acc_loss
            self.acc_loss = {op: [self.dummy] for op in self.units}

            _, pred_time = self._forward_one_batch(samp_dict)

            all_pred_times.extend(pred_time.detach().cpu().numpy().tolist())
            all_true_times.extend(samp_dict['total_time'].tolist())

            # 计算本组损失
            d_size = 0
            subbatch_loss = torch.zeros(1).to(self.device)
            for op in self.acc_loss:
                all_loss = torch.cat(self.acc_loss[op])
                d_size   += all_loss.shape[0]
                subbatch_loss += torch.sum(all_loss)

            if d_size > 0:
                subbatch_loss = torch.mean(torch.sqrt(subbatch_loss / d_size))
                total_loss   += subbatch_loss * samp_dict['subbatch_size']

        self.total_loss = total_loss / max(len(samp_dicts), 1)
        return self.total_loss, all_pred_times, all_true_times

    def optimize_parameters(self, samp_dicts: list):
        """训练一步：前向 → 反向 → 更新参数。"""
        for op in self.optimizers:
            self.optimizers[op].zero_grad()

        loss, pred_times, true_times = self.forward(samp_dicts, is_train=True)

        if self.best > loss.item():
            self.best = loss.item()

        loss.backward()

        for op in self.optimizers:
            self.optimizers[op].step()
            if op in self.schedulers:
                self.schedulers[op].step()

        return loss.item(), pred_times, true_times

    def evaluate(self, samp_dicts: list):
        """评估模式前向传播（不计算梯度）。"""
        with torch.no_grad():
            loss, pred_times, true_times = self.forward(samp_dicts, is_train=False)
        return loss.item(), pred_times, true_times

    def save(self, tag: str):
        """保存所有 NeuralUnit 的权重。"""
        for name, unit in self.units.items():
            save_path = os.path.join(self.save_dir, f'{tag}_net_{name}.pth')
            torch.save(unit.cpu().state_dict(), save_path)
            unit.to(self.device)

    def load(self, tag: str):
        """加载所有 NeuralUnit 的权重。"""
        for name in self.units:
            save_path = os.path.join(self.save_dir, f'{tag}_net_{name}.pth')
            if os.path.exists(save_path):
                self.units[name].load_state_dict(
                    torch.load(save_path, map_location=self.device)
                )
