#!/usr/bin/env python3
"""长尾分布评估指标模块

包含用于处理类别不平衡数据集的评估指标：
- Balanced Accuracy: 平衡准确率，考虑每个类别的召回率
- Macro-F1 Score: 宏平均F1分数，对所有类别平等对待

作者: AI Assistant
日期: 2024
"""

import torch
import torch.nn.functional as F
from collections import OrderedDict
from typing import Dict, Tuple, Optional
import warnings


def balanced_accuracy(output: torch.Tensor, target: torch.Tensor, num_classes: Optional[int] = None) -> float:
    """计算平衡准确率 (Balanced Accuracy)
    
    平衡准确率是每个类别召回率的平均值，对类别不平衡问题更加鲁棒。
    公式: BA = (1/C) * Σ(TP_i / (TP_i + FN_i))，其中C是类别数
    
    Args:
        output: 模型输出，形状为 (batch_size, num_classes) 或 (batch_size,)
        target: 真实标签，形状为 (batch_size,)
        num_classes: 类别数量，如果为None则自动推断
    
    Returns:
        平衡准确率 (0-1之间的浮点数)
        
    Raises:
        ValueError: 当输入张量维度不匹配时
        RuntimeError: 当计算过程中出现错误时
    """
    try:
        if output.dim() > 1:
            # 多分类情况，取argmax
            predictions = torch.argmax(output, dim=1)
        else:
            # 已经是预测标签
            predictions = output
        
        # 输入验证
        if predictions.shape != target.shape:
            raise ValueError(f"预测值和标签的形状不匹配: {predictions.shape} vs {target.shape}")
        
        if num_classes is None:
            num_classes = max(target.max().item(), predictions.max().item()) + 1
        
        if num_classes <= 0:
            raise ValueError(f"类别数量必须为正数，得到: {num_classes}")
        
        # 计算每个类别的召回率
        recall_per_class = []
        for class_id in range(num_classes):
            # 当前类别的真实样本
            true_positives = ((predictions == class_id) & (target == class_id)).sum().float()
            false_negatives = ((predictions != class_id) & (target == class_id)).sum().float()
            
            # 避免除零
            if true_positives + false_negatives > 0:
                recall = true_positives / (true_positives + false_negatives)
            else:
                recall = 0.0
            recall_per_class.append(recall.item())
        
        # 计算平均召回率（平衡准确率）
        balanced_acc = sum(recall_per_class) / num_classes
        return balanced_acc
        
    except Exception as e:
        warnings.warn(f"计算平衡准确率时出错: {str(e)}，返回0.0")
        return 0.0


def macro_f1_score(output: torch.Tensor, target: torch.Tensor, num_classes: Optional[int] = None,
                   average: str = 'macro') -> float:
    """计算宏平均F1分数 (Macro-F1 Score)
    
    Macro-F1是所有类别F1分数的未加权平均，对少数类别和多数类别同等重视。
    对于每个类别：F1 = 2 * (Precision * Recall) / (Precision + Recall)
    
    Args:
        output: 模型输出，形状为 (batch_size, num_classes) 或 (batch_size,)
        target: 真实标签，形状为 (batch_size,)
        num_classes: 类别数量，如果为None则自动推断
        average: 平均方式，'macro' 表示宏平均
    
    Returns:
        宏平均F1分数 (0-1之间的浮点数)
        
    Raises:
        ValueError: 当输入张量维度不匹配时
        RuntimeError: 当计算过程中出现错误时
    """
    try:
        if output.dim() > 1:
            # 多分类情况，取argmax
            predictions = torch.argmax(output, dim=1)
        else:
            # 已经是预测标签
            predictions = output
        
        # 输入验证
        if predictions.shape != target.shape:
            raise ValueError(f"预测值和标签的形状不匹配: {predictions.shape} vs {target.shape}")
        
        if num_classes is None:
            num_classes = max(target.max().item(), predictions.max().item()) + 1
        
        if num_classes <= 0:
            raise ValueError(f"类别数量必须为正数，得到: {num_classes}")
        
        # 计算每个类别的精确率和召回率
        f1_scores = []
        for class_id in range(num_classes):
            # 真正例：预测为class_id且实际为class_id
            true_positives = ((predictions == class_id) & (target == class_id)).sum().float()
            
            # 假正例：预测为class_id但实际不为class_id
            false_positives = ((predictions == class_id) & (target != class_id)).sum().float()
            
            # 假负例：预测不为class_id但实际为class_id
            false_negatives = ((predictions != class_id) & (target == class_id)).sum().float()
            
            # 计算精确率和召回率
            precision = true_positives / (true_positives + false_positives + 1e-8)
            recall = true_positives / (true_positives + false_negatives + 1e-8)
            
            # 计算F1分数
            if precision + recall > 0:
                f1 = 2 * (precision * recall) / (precision + recall)
            else:
                f1 = 0.0
            
            f1_scores.append(f1.item())
        
        # 宏平均F1分数
        macro_f1 = sum(f1_scores) / num_classes
        return macro_f1
        
    except Exception as e:
        warnings.warn(f"计算宏平均F1分数时出错: {str(e)}，返回0.0")
        return 0.0


def compute_long_tail_metrics(output: torch.Tensor, target: torch.Tensor, 
                             num_classes: Optional[int] = None) -> Dict[str, float]:
    """计算长尾分布相关的评估指标
    
    一次性计算多个长尾分布评估指标，提高效率
    
    Args:
        output: 模型输出，形状为 (batch_size, num_classes)
        target: 真实标签，形状为 (batch_size,)
        num_classes: 类别数量，如果为None则自动推断
    
    Returns:
        包含各种指标的字典
    """
    if num_classes is None:
        num_classes = max(target.max().item(), output.argmax(dim=1).max().item()) + 1
    
    # 计算平衡准确率
    balanced_acc = balanced_accuracy(output, target, num_classes)
    
    # 计算宏平均F1分数
    macro_f1 = macro_f1_score(output, target, num_classes)
    
    # 计算每个类别的样本数量（用于分析）
    class_counts = torch.bincount(target, minlength=num_classes)
    
    # 计算类别分布的基尼系数（衡量不平衡程度）
    total_samples = class_counts.sum().float()
    if total_samples > 0:
        class_probs = class_counts.float() / total_samples
        gini_coefficient = 1.0 - torch.sum(class_probs ** 2).item()
    else:
        gini_coefficient = 0.0
    
    metrics = OrderedDict([
        ('balanced_accuracy', balanced_acc),
        ('macro_f1', macro_f1),
        ('gini_coefficient', gini_coefficient),
    ])
    
    # 添加每个类别的样本数量信息
    for i, count in enumerate(class_counts):
        metrics[f'class_{i}_count'] = count.item()
    
    return metrics


class LongTailMetricsTracker:
    """长尾分布指标跟踪器
    
    用于在训练过程中累积和计算长尾分布评估指标
    
    Attributes:
        num_classes: 类别数量
        total_predictions: 累积的预测结果列表
        total_targets: 累积的真实标签列表
        total_loss: 累积的损失值
        total_samples: 累积的样本数量
    """
    
    def __init__(self, num_classes: int):
        """初始化跟踪器
        
        Args:
            num_classes: 类别数量
        """
        if num_classes <= 0:
            raise ValueError(f"类别数量必须为正数，得到: {num_classes}")
        
        self.num_classes = num_classes
        self.reset()
    
    def reset(self):
        """重置所有累积的指标"""
        self.total_predictions = []
        self.total_targets = []
        self.total_loss = 0.0
        self.total_samples = 0
    
    def update(self, output: torch.Tensor, target: torch.Tensor, loss: Optional[float] = None):
        """更新累积的预测和目标
        
        Args:
            output: 模型输出，形状为 (batch_size, num_classes) 或 (batch_size,)
            target: 真实标签，形状为 (batch_size,)
            loss: 损失值（可选）
            
        Raises:
            ValueError: 当输入维度不匹配时
        """
        try:
            if output.dim() > 1:
                predictions = torch.argmax(output, dim=1)
            else:
                predictions = output
            
            # 输入验证
            if predictions.shape != target.shape:
                raise ValueError(f"预测值和标签的形状不匹配: {predictions.shape} vs {target.shape}")
            
            self.total_predictions.append(predictions.cpu())
            self.total_targets.append(target.cpu())
            
            if loss is not None:
                batch_size = target.size(0)
                self.total_loss += loss * batch_size
                self.total_samples += batch_size
                
        except Exception as e:
            warnings.warn(f"更新指标跟踪器时出错: {str(e)}，跳过此次更新")
    
    def compute_metrics(self) -> Dict[str, float]:
        """计算累积的长尾分布指标
        
        Returns:
            包含所有指标的字典，如果没有任何数据则返回空字典
        """
        if not self.total_predictions:
            warnings.warn("没有累积任何预测数据，返回空指标")
            return OrderedDict()
        
        try:
            # 合并所有批次的数据
            all_predictions = torch.cat(self.total_predictions)
            all_targets = torch.cat(self.total_targets)
            
            # 计算长尾分布指标
            metrics = compute_long_tail_metrics(all_predictions, all_targets, self.num_classes)
            
            # 添加平均损失
            if self.total_samples > 0:
                metrics['loss'] = self.total_loss / self.total_samples
            
            return metrics
            
        except Exception as e:
            warnings.warn(f"计算指标时出错: {str(e)}，返回空指标")
            return OrderedDict()


# 为了与timm库的指标计算保持一致，提供类似的接口
def accuracy_long_tail(output: torch.Tensor, target: torch.Tensor, 
                      num_classes: Optional[int] = None) -> Tuple[float, float]:
    """计算平衡准确率和宏平均F1分数
    
    Args:
        output: 模型输出
        target: 真实标签
        num_classes: 类别数量
    
    Returns:
        (balanced_accuracy, macro_f1) 元组
    """
    balanced_acc = balanced_accuracy(output, target, num_classes)
    macro_f1 = macro_f1_score(output, target, num_classes)
    return balanced_acc, macro_f1