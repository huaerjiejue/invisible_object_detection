import torch
import torch.nn as nn
from .backbone import FeatureExtractor
from .detector import DetectionModel
from .temporal_modeula import TemporalTransformer


class InvisibleObjectDetector(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.backbone = FeatureExtractor(config)
        self.temporal = TemporalTransformer(config)
        self.detector = DetectionModel(config)

    def forward(self, x):
        # x: [B, T, C, H, W]
        B, T, C, H, W = x.shape
        
        # 处理每一帧
        features = []
        for t in range(T):
            feat = self.backbone(x[:, t])  # [B, C, H/8, W/8]
            features.append(feat)
        
        # 堆叠特征
        features = torch.stack(features, dim=1)  # [B, T, C, H/8, W/8]
        
        # 时序处理
        temporal_features = self.temporal(features)  # [B, C, H/8, W/8]
        
        # 检测
        predictions = self.detector(temporal_features)  # [p1, p2, p3]
        
        return predictions 