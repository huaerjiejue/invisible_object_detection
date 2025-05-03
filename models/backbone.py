import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=None):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = ConvBNReLU(in_channels, out_channels, kernel_size=3, stride=stride)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.downsample = None
        if in_channels != out_channels or stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out += identity
        return self.relu(out)


class MultiScaleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv3 = ConvBNReLU(in_channels, out_channels, kernel_size=3)
        self.conv5 = ConvBNReLU(in_channels, out_channels, kernel_size=5)
        self.merge = ConvBNReLU(out_channels * 2, out_channels, kernel_size=1)

    def forward(self, x):
        x3 = self.conv3(x)
        x5 = self.conv5(x)
        merged = torch.cat([x3, x5], dim=1)
        return self.merge(merged)


class FeatureExtractor(nn.Module):
    def __init__(self, config):
        super().__init__()
        backbone_config = config.get_backbone_config()
        
        # 从配置中获取参数
        init_conv_config = backbone_config['init_conv']
        multi_scale_config = backbone_config['multi_scale']
        layer1_config = backbone_config['layer1']
        layer2_config = backbone_config['layer2']
        
        self.init_conv = ConvBNReLU(
            in_channels=init_conv_config['in_channels'],
            out_channels=init_conv_config['out_channels'],
            kernel_size=init_conv_config['kernel_size'],
            stride=init_conv_config['stride']
        )

        self.multi_scale = MultiScaleConv(
            in_channels=multi_scale_config['in_channels'],
            out_channels=multi_scale_config['out_channels']
        )

        self.layer1 = nn.Sequential(
            ResidualBlock(layer1_config['in_channels'], layer1_config['out_channels'], stride=2),
            ResidualBlock(layer1_config['out_channels'], layer1_config['out_channels'])
        )
        
        self.layer2 = nn.Sequential(
            ResidualBlock(layer2_config['in_channels'], layer2_config['out_channels'], stride=2),
            ResidualBlock(layer2_config['out_channels'], layer2_config['out_channels'])
        )

        self.out_proj = ConvBNReLU(
            layer2_config['out_channels'],
            backbone_config['out_channels'],
            kernel_size=1
        )

    def forward(self, x):
        x = self.init_conv(x)      # -> [B, 64, H/2, W/2]
        x = self.multi_scale(x)    # -> [B, 64, H/2, W/2]
        x = self.layer1(x)         # -> [B, 128, H/4, W/4]
        x = self.layer2(x)         # -> [B, 256, H/8, W/8]
        x = self.out_proj(x)       # -> [B, out_channels, H/8, W/8]
        return x
