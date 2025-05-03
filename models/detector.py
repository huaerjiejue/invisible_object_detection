import torch
import torch.nn as nn


class DetectionHead(nn.Module):
    def __init__(self, in_channels, num_classes, num_anchors=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        # 输出 (x, y, w, h, obj, class_probs...)
        self.pred = nn.Conv2d(in_channels, num_anchors * (5 + num_classes), kernel_size=1)

    def forward(self, x):
        x = self.conv(x)
        return self.pred(x)  # [B, A*(5+C), H, W]


class SimpleFPN(nn.Module):
    def __init__(self, in_channels, fpn_channels=128):
        super().__init__()
        self.down1 = nn.Sequential(
            nn.Conv2d(in_channels, fpn_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(fpn_channels),
            nn.ReLU(inplace=True)
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(fpn_channels, fpn_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(fpn_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        f1 = x                   # 原始分辨率特征（H/8）
        f2 = self.down1(f1)      # 下采样一次（H/16）
        f3 = self.down2(f2)      # 下采样两次（H/32）
        return [f1, f2, f3]


class DetectionModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        detector_config = config.get_detector_config()
        fpn_config = detector_config['fpn']
        
        self.fpn = SimpleFPN(
            in_channels=fpn_config['in_channels'],
            fpn_channels=fpn_config['fpn_channels']
        )

        self.head1 = DetectionHead(
            detector_config['in_channels'],
            detector_config['num_classes'],
            detector_config['num_anchors']
        )
        self.head2 = DetectionHead(
            detector_config['in_channels'],
            detector_config['num_classes'],
            detector_config['num_anchors']
        )
        self.head3 = DetectionHead(
            detector_config['in_channels'],
            detector_config['num_classes'],
            detector_config['num_anchors']
        )

    def forward(self, x):
        f1, f2, f3 = self.fpn(x)               # 多尺度特征图
        p1 = self.head1(f1)  # [B, A*(5+C), H/8, W/8]
        p2 = self.head2(f2)  # [B, A*(5+C), H/16, W/16]
        p3 = self.head3(f3)  # [B, A*(5+C), H/32, W/32]
        return [p1, p2, p3]
