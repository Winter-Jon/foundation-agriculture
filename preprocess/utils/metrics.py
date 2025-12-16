#!/usr/bin/env python3
"""长尾分布评估指标模块 (高性能优化版)

包含用于处理类别不平衡数据集的评估指标：
- Balanced Accuracy: 平衡准确率，考虑每个类别的召回率
- Macro-F1 Score: 宏平均F1分数，对所有类别平等对待

优化说明:
内部实现已重构为基于混淆矩阵(Confusion Matrix)的向量化计算。
在保持接口完全兼容的前提下，大幅提升了在多类别(如ImageNet-1k)下的计算速度。

作者: AI Assistant (Optimized)
日期: 2024
"""

import torch
import torch.nn.functional as F
from collections import OrderedDict
from typing import Dict, Tuple, Optional, List
import warnings


def _compute_confusion_matrix(output: torch.Tensor, target: torch.Tensor, num_classes: int) -> torch.Tensor:
    """内部辅助函数：高效计算混淆矩阵"""
    # 确保预测值是标签格式
    if output.dim() > 1:
        predictions = torch.argmax(output, dim=1)
    else:
        predictions = output

    # 确保数据在同一设备
    if predictions.device != target.device:
        predictions = predictions.to(target.device)

    # 过滤掉非法索引 (以防万一)
    mask = (target >= 0) & (target < num_classes) & (predictions >= 0) & (predictions < num_classes)
    target = target[mask]
    predictions = predictions[mask]

    # 使用 bincount 技巧向量化计算混淆矩阵
    # index = target * num_classes + prediction
    indices = target * num_classes + predictions
    # minlength 确保矩阵大小固定为 num_classes^2
    conf_mat = torch.bincount(indices, minlength=num_classes**2).reshape(num_classes, num_classes).float()
    
    return conf_mat


def balanced_accuracy(output: torch.Tensor, target: torch.Tensor, num_classes: Optional[int] = None) -> float:
    """计算平衡准确率 (Balanced Accuracy) - 向量化优化版"""
    try:
        # 自动推断类别数
        if num_classes is None:
            preds = torch.argmax(output, dim=1) if output.dim() > 1 else output
            num_classes = max(target.max().item(), preds.max().item()) + 1
            
        if num_classes <= 0:
            raise ValueError(f"类别数量必须为正数，得到: {num_classes}")

        # 1. 计算混淆矩阵
        cm = _compute_confusion_matrix(output, target, num_classes)
        
        # 2. 计算每个类别的 Recall
        # TP: 对角线元素
        tp = cm.diag()
        # True Targets (TP + FN): 每一行的和
        true_targets = cm.sum(dim=1)
        
        # 避免除以零
        # 逻辑与原代码保持一致：如果某类无真实样本，其 Recall 为 0
        recall_per_class = tp / (true_targets + 1e-8)
        recall_per_class[true_targets == 0] = 0.0
        
        # 3. 计算平均值
        balanced_acc = recall_per_class.sum() / num_classes
        return balanced_acc.item()
        
    except Exception as e:
        warnings.warn(f"计算平衡准确率时出错: {str(e)}，返回0.0")
        return 0.0


def macro_f1_score(output: torch.Tensor, target: torch.Tensor, num_classes: Optional[int] = None,
                   average: str = 'macro') -> float:
    """计算宏平均F1分数 (Macro-F1 Score) - 向量化优化版"""
    try:
        # 自动推断类别数
        if num_classes is None:
            preds = torch.argmax(output, dim=1) if output.dim() > 1 else output
            num_classes = max(target.max().item(), preds.max().item()) + 1

        if num_classes <= 0:
            raise ValueError(f"类别数量必须为正数，得到: {num_classes}")

        # 1. 计算混淆矩阵
        cm = _compute_confusion_matrix(output, target, num_classes)
        
        # 2. 计算 TP, FP, FN
        tp = cm.diag()
        # Predicted Positives (TP + FP): 每一列的和
        pred_positives = cm.sum(dim=0)
        # True Positives (TP + FN): 每一行的和
        true_positives = cm.sum(dim=1)
        
        # 3. 计算 Precision & Recall
        precision = tp / (pred_positives + 1e-8)
        recall = tp / (true_positives + 1e-8)
        
        # 4. 计算 F1
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        
        # 处理可能的 NaN (如 precision 和 recall 都是 0)
        f1_scores = torch.nan_to_num(f1_scores, 0.0)
        
        # 5. 宏平均
        macro_f1 = f1_scores.sum() / num_classes
        return macro_f1.item()
        
    except Exception as e:
        warnings.warn(f"计算宏平均F1分数时出错: {str(e)}，返回0.0")
        return 0.0


def compute_long_tail_metrics(output: torch.Tensor, target: torch.Tensor, 
                             num_classes: Optional[int] = None) -> Dict[str, float]:
    """计算长尾分布相关的评估指标 - 向量化优化版"""
    
    # 如果传入的是 logits，转换为 label (为了复用 logic，虽然 _compute_confusion_matrix 也会做)
    # 这里主要为了推断 num_classes
    if output.dim() > 1:
        preds = torch.argmax(output, dim=1)
    else:
        preds = output

    if num_classes is None:
        num_classes = max(target.max().item(), preds.max().item()) + 1
    
    # 复用混淆矩阵计算所有指标，避免重复计算
    try:
        cm = _compute_confusion_matrix(preds, target, num_classes)
        
        # --- 计算 Balanced Accuracy ---
        tp = cm.diag()
        true_targets = cm.sum(dim=1)
        recall_per_class = tp / (true_targets + 1e-8)
        recall_per_class[true_targets == 0] = 0.0
        balanced_acc = recall_per_class.sum() / num_classes
        
        # --- 计算 Macro F1 ---
        pred_positives = cm.sum(dim=0)
        precision = tp / (pred_positives + 1e-8)
        f1_scores = 2 * (precision * recall_per_class) / (precision + recall_per_class + 1e-8)
        f1_scores = torch.nan_to_num(f1_scores, 0.0)
        macro_f1 = f1_scores.sum() / num_classes
        
        # --- 计算基尼系数 ---
        class_counts = true_targets # 这就是每个类别的真实样本数
        total_samples = class_counts.sum().item()
        if total_samples > 0:
            class_probs = class_counts / total_samples
            gini_coefficient = 1.0 - torch.sum(class_probs ** 2).item()
        else:
            gini_coefficient = 0.0
            
        metrics = OrderedDict([
            ('balanced_accuracy', balanced_acc.item()),
            ('macro_f1', macro_f1.item()),
            ('gini_coefficient', gini_coefficient),
        ])
        
        # 添加每个类别的样本数量信息 (保持接口兼容)
        for i, count in enumerate(class_counts):
            metrics[f'class_{i}_count'] = count.item()
            
        return metrics
        
    except Exception as e:
        warnings.warn(f"计算长尾指标时出错: {str(e)}")
        return OrderedDict()


class LongTailMetricsTracker:
    """长尾分布指标跟踪器 (内存优化版)
    
    优化: 不再存储所有的预测列表 (total_predictions)，改为流式累加混淆矩阵。
    大幅降低内存占用，并加快 compute_metrics 的速度。
    """
    
    def __init__(self, num_classes: int):
        if num_classes <= 0:
            raise ValueError(f"类别数量必须为正数，得到: {num_classes}")
        
        self.num_classes = num_classes
        self.reset()
    
    def reset(self):
        """重置所有累积的指标"""
        # 核心改变：使用混淆矩阵作为累积器，而不是列表
        # 这将内存复杂度从 O(N) 降低到 O(C^2)
        # 对于 1000 类，仅需约 4MB 内存，而不是存储百万张量
        self.conf_mat = torch.zeros((self.num_classes, self.num_classes), dtype=torch.float32)
        
        # 保持对 loss 的跟踪
        self.total_loss = 0.0
        self.total_samples = 0
        
        # 为了保持严格的接口兼容性（防止有代码直接访问这些属性），
        # 我们保留这两个属性，但设置为 None 或空列表。
        # 如果外部代码强依赖这些属性，它们可能需要调整，
        # 但标准的 usage pattern (update -> compute) 不受影响。
        self.total_predictions = [] 
        self.total_targets = []
    
    def update(self, output: torch.Tensor, target: torch.Tensor, loss: Optional[float] = None):
        """更新累积状态"""
        try:
            # 计算当前 batch 的混淆矩阵
            # 建议传入 CPU tensor 以避免 GPU 显存累积，但此函数也支持 GPU tensor
            batch_cm = _compute_confusion_matrix(output, target, self.num_classes)
            
            # 如果累积矩阵在 CPU，确保 batch 也在 CPU
            if self.conf_mat.device != batch_cm.device:
                batch_cm = batch_cm.to(self.conf_mat.device)
                
            self.conf_mat += batch_cm
            
            if loss is not None:
                batch_size = target.size(0)
                self.total_loss += loss * batch_size
                self.total_samples += batch_size
                
        except Exception as e:
            warnings.warn(f"更新指标跟踪器时出错: {str(e)}，跳过此次更新")
    
    def compute_metrics(self) -> Dict[str, float]:
        """基于累积的混淆矩阵计算指标"""
        if self.conf_mat.sum() == 0:
            warnings.warn("没有累积任何数据，返回空指标")
            return OrderedDict()
            
        try:
            # 基于 self.conf_mat 计算所有指标
            cm = self.conf_mat
            
            # --- Balanced Accuracy ---
            tp = cm.diag()
            true_targets = cm.sum(dim=1)
            recall_per_class = tp / (true_targets + 1e-8)
            recall_per_class[true_targets == 0] = 0.0
            balanced_acc = recall_per_class.sum() / self.num_classes
            
            # --- Macro F1 ---
            pred_positives = cm.sum(dim=0)
            precision = tp / (pred_positives + 1e-8)
            f1_scores = 2 * (precision * recall_per_class) / (precision + recall_per_class + 1e-8)
            f1_scores = torch.nan_to_num(f1_scores, 0.0)
            macro_f1 = f1_scores.sum() / self.num_classes
            
            # --- Gini ---
            total_samples = true_targets.sum().item()
            if total_samples > 0:
                class_probs = true_targets / total_samples
                gini = 1.0 - torch.sum(class_probs ** 2).item()
            else:
                gini = 0.0
            
            metrics = OrderedDict([
                ('balanced_accuracy', balanced_acc.item()),
                ('macro_f1', macro_f1.item()),
                ('gini_coefficient', gini),
            ])
            
            if self.total_samples > 0:
                metrics['loss'] = self.total_loss / self.total_samples
                
            # 添加每个类别的样本数量信息
            for i, count in enumerate(true_targets):
                metrics[f'class_{i}_count'] = count.item()
                
            return metrics
            
        except Exception as e:
            warnings.warn(f"计算指标时出错: {str(e)}，返回空指标")
            return OrderedDict()


# 保持接口兼容
def accuracy_long_tail(output: torch.Tensor, target: torch.Tensor, 
                       num_classes: Optional[int] = None) -> Tuple[float, float]:
    """计算平衡准确率和宏平均F1分数"""
    balanced_acc = balanced_accuracy(output, target, num_classes)
    macro_f1 = macro_f1_score(output, target, num_classes)
    return balanced_acc, macro_f1