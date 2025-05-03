import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Tuple, Dict, Any


class DetectionLoss(nn.Module):
    """不可见光目标检测的损失函数
    
    该损失函数包含三个主要部分：
    1. 目标检测损失（边界框回归、目标性、分类）
    2. 时序一致性损失（自监督时序建模）
    3. 特征融合损失（确保时序信息的有效融合）
    """
    
    def __init__(self, config: Dict[str, Any]):
        """初始化损失函数
        
        Args:
            config: 配置字典，包含各个损失项的权重
        """
        super().__init__()
        self.config = config
        
        # 从配置中获取损失权重，如果未指定则使用默认值
        self.lambda_box = config.get('lambda_box', 5.0)  # 边界框回归损失权重
        self.lambda_obj = config.get('lambda_obj', 1.0)  # 目标性损失权重
        self.lambda_cls = config.get('lambda_cls', 1.0)  # 分类损失权重
        self.lambda_temporal = config.get('lambda_temporal', 0.1)  # 时序一致性损失权重
        self.lambda_feature = config.get('lambda_feature', 0.05)  # 特征融合损失权重
        
        # 损失计算相关的参数
        self.eps = 1e-16  # 数值稳定性参数
        self.neg_pos_ratio = config.get('neg_pos_ratio', 3)  # 负样本比例

    def compute_box_loss(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor, 
                        obj_mask: torch.Tensor) -> torch.Tensor:
        """计算边界框回归损失
        
        Args:
            pred_boxes: 预测的边界框 [B, A*4, H, W]
            target_boxes: 目标边界框 [B, A*4, H, W]
            obj_mask: 目标掩码 [B, A, H, W]
            
        Returns:
            torch.Tensor: 边界框损失
        """
        # 将预测和目标框重塑为 [B, A, 4, H, W]
        pred_boxes = pred_boxes.view(pred_boxes.size(0), -1, 4, pred_boxes.size(2), pred_boxes.size(3))
        target_boxes = target_boxes.view(target_boxes.size(0), -1, 4, target_boxes.size(2), target_boxes.size(3))
        
        # 计算CIoU损失
        loss = 0
        for b in range(pred_boxes.size(0)):
            for a in range(pred_boxes.size(1)):
                if obj_mask[b, a].sum() > 0:
                    pred = pred_boxes[b, a].permute(1, 2, 0)[obj_mask[b, a]]  # [N, 4]
                    target = target_boxes[b, a].permute(1, 2, 0)[obj_mask[b, a]]  # [N, 4]
                    loss += self.ciou_loss(pred, target)
        
        return loss / (obj_mask.sum() + self.eps)

    def compute_obj_loss(self, pred_obj: torch.Tensor, target_obj: torch.Tensor, 
                        obj_mask: torch.Tensor) -> torch.Tensor:
        """计算目标性损失
        
        Args:
            pred_obj: 预测的目标性 [B, A, H, W]
            target_obj: 目标目标性 [B, A, H, W]
            obj_mask: 目标掩码 [B, A, H, W]
            
        Returns:
            torch.Tensor: 目标性损失
        """
        # 计算正样本损失
        pos_loss = F.binary_cross_entropy_with_logits(
            pred_obj[obj_mask], 
            target_obj[obj_mask], 
            reduction='none'
        )
        
        # 计算负样本损失
        neg_mask = ~obj_mask
        neg_loss = F.binary_cross_entropy_with_logits(
            pred_obj[neg_mask], 
            target_obj[neg_mask], 
            reduction='none'
        )
        
        # 选择最难的负样本
        num_pos = obj_mask.sum()
        num_neg = min(neg_mask.sum(), num_pos * self.neg_pos_ratio)
        if num_neg > 0:
            neg_loss, _ = torch.topk(neg_loss, num_neg)
        
        return (pos_loss.sum() + neg_loss.sum()) / (num_pos + num_neg + self.eps)

    def compute_cls_loss(self, pred_cls: torch.Tensor, target_cls: torch.Tensor, 
                        obj_mask: torch.Tensor) -> torch.Tensor:
        """计算分类损失
        
        Args:
            pred_cls: 预测的类别 [B, A*C, H, W]
            target_cls: 目标类别 [B, A*C, H, W]
            obj_mask: 目标掩码 [B, A, H, W]
            
        Returns:
            torch.Tensor: 分类损失
        """
        pred_cls = pred_cls.view(pred_cls.size(0), -1, pred_cls.size(2), pred_cls.size(3))
        target_cls = target_cls.view(target_cls.size(0), -1, target_cls.size(2), target_cls.size(3))
        
        # 使用focal loss处理类别不平衡
        bce = F.binary_cross_entropy_with_logits(pred_cls, target_cls, reduction='none')
        pt = torch.exp(-bce)
        focal_loss = ((1 - pt) ** 2 * bce)
        
        return (focal_loss * obj_mask.unsqueeze(1)).sum() / (obj_mask.sum() + self.eps)

    def compute_temporal_consistency_loss(self, features: torch.Tensor) -> torch.Tensor:
        """计算时序一致性损失
        
        Args:
            features: 特征序列 [B, T, C, H, W]
            
        Returns:
            torch.Tensor: 时序一致性损失
        """
        B, T, C, H, W = features.shape
        loss = 0
        
        # 计算相邻帧之间的特征差异
        for t in range(T-1):
            curr_feat = features[:, t]
            next_feat = features[:, t+1]
            
            # 计算特征差异
            diff = F.mse_loss(curr_feat, next_feat, reduction='none')
            
            # 使用注意力机制加权
            attention = torch.sigmoid(torch.mean(diff, dim=1, keepdim=True))
            loss += (diff * attention).mean()
        
        return loss / (T-1)

    def compute_feature_fusion_loss(self, temporal_features: torch.Tensor, 
                                  original_features: torch.Tensor) -> torch.Tensor:
        """计算特征融合损失
        
        Args:
            temporal_features: 时序融合后的特征 [B, C, H, W]
            original_features: 原始特征序列 [B, T, C, H, W]
            
        Returns:
            torch.Tensor: 特征融合损失
        """
        # 确保时序特征保留了原始特征的关键信息
        original_mean = original_features.mean(dim=1)  # [B, C, H, W]
        
        # 使用L1损失，对异常值更鲁棒
        return F.l1_loss(temporal_features, original_mean)

    def ciou_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """计算CIoU损失
        
        Args:
            pred: 预测框 [N, 4] (x, y, w, h)
            target: 目标框 [N, 4] (x, y, w, h)
            
        Returns:
            torch.Tensor: CIoU损失
        """
        # 转换为 (x1, y1, x2, y2) 格式
        pred_x1 = pred[:, 0] - pred[:, 2] / 2
        pred_y1 = pred[:, 1] - pred[:, 3] / 2
        pred_x2 = pred[:, 0] + pred[:, 2] / 2
        pred_y2 = pred[:, 1] + pred[:, 3] / 2
        
        target_x1 = target[:, 0] - target[:, 2] / 2
        target_y1 = target[:, 1] - target[:, 3] / 2
        target_x2 = target[:, 0] + target[:, 2] / 2
        target_y2 = target[:, 1] + target[:, 3] / 2
        
        # 计算交集面积
        inter_x1 = torch.max(pred_x1, target_x1)
        inter_y1 = torch.max(pred_y1, target_y1)
        inter_x2 = torch.min(pred_x2, target_x2)
        inter_y2 = torch.min(pred_y2, target_y2)
        
        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
        
        # 计算并集面积
        pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
        target_area = (target_x2 - target_x1) * (target_y2 - target_y1)
        union_area = pred_area + target_area - inter_area
        
        # 计算IoU
        iou = inter_area / (union_area + self.eps)
        
        # 计算中心点距离
        pred_center = torch.stack([pred[:, 0], pred[:, 1]], dim=1)
        target_center = torch.stack([target[:, 0], target[:, 1]], dim=1)
        center_distance = torch.sum((pred_center - target_center) ** 2, dim=1)
        
        # 计算对角线距离
        enclose_x1 = torch.min(pred_x1, target_x1)
        enclose_y1 = torch.min(pred_y1, target_y1)
        enclose_x2 = torch.max(pred_x2, target_x2)
        enclose_y2 = torch.max(pred_y2, target_y2)
        enclose_diagonal = torch.sum((torch.stack([enclose_x2, enclose_y2], dim=1) - 
                                    torch.stack([enclose_x1, enclose_y1], dim=1)) ** 2, dim=1)
        
        # 计算长宽比一致性
        pred_wh = torch.stack([pred[:, 2], pred[:, 3]], dim=1)
        target_wh = torch.stack([target[:, 2], target[:, 3]], dim=1)
        v = (4 / (math.pi ** 2)) * torch.pow(torch.atan(target_wh[:, 0] / target_wh[:, 1]) - 
                                           torch.atan(pred_wh[:, 0] / pred_wh[:, 1]), 2)
        
        # 计算CIoU
        alpha = v / (v - iou + self.eps)
        ciou = iou - (center_distance / enclose_diagonal) - alpha * v
        
        return (1 - ciou).mean()

    def forward(self, predictions: List[torch.Tensor], targets: List[torch.Tensor], 
                features: torch.Tensor, temporal_features: torch.Tensor) -> torch.Tensor:
        """计算总损失
        
        Args:
            predictions: 多尺度预测结果列表 [p1, p2, p3]
            targets: 多尺度目标列表 [t1, t2, t3]
            features: 原始特征序列 [B, T, C, H, W]
            temporal_features: 时序融合后的特征 [B, C, H, W]
            
        Returns:
            torch.Tensor: 总损失
        """
        total_loss = 0
        
        # 计算多尺度检测损失
        for pred, target in zip(predictions, targets):
            # 分离预测结果
            pred_boxes = pred[:, :4]
            pred_obj = pred[:, 4]
            pred_cls = pred[:, 5:]
            
            # 分离目标
            target_boxes = target[:, :4]
            target_obj = target[:, 4]
            target_cls = target[:, 5:]
            obj_mask = target_obj > 0
            
            # 计算各项损失
            box_loss = self.compute_box_loss(pred_boxes, target_boxes, obj_mask)
            obj_loss = self.compute_obj_loss(pred_obj, target_obj, obj_mask)
            cls_loss = self.compute_cls_loss(pred_cls, target_cls, obj_mask)
            
            # 加权求和
            scale_loss = (self.lambda_box * box_loss + 
                         self.lambda_obj * obj_loss + 
                         self.lambda_cls * cls_loss)
            total_loss += scale_loss
        
        # 计算时序一致性损失
        temporal_loss = self.compute_temporal_consistency_loss(features)
        
        # 计算特征融合损失
        feature_loss = self.compute_feature_fusion_loss(temporal_features, features)
        
        # 总损失
        total_loss = (total_loss + 
                     self.lambda_temporal * temporal_loss + 
                     self.lambda_feature * feature_loss)
        
        return total_loss
