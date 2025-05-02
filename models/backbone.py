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
    def __init__(self, out_channels=256):
        super().__init__()
        self.init_conv = ConvBNReLU(1, 64, kernel_size=7, stride=2)

        self.multi_scale = MultiScaleConv(64, 64)

        # 模拟 ResNet 的前两层
        self.layer1 = nn.Sequential(
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 128)
        )
        self.layer2 = nn.Sequential(
            ResidualBlock(128, 256, stride=2),
            ResidualBlock(256, 256)
        )

        self.out_proj = ConvBNReLU(256, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.init_conv(x)      # -> [B, 64, H/2, W/2]
        x = self.multi_scale(x)    # -> [B, 64, H/2, W/2]
        x = self.layer1(x)         # -> [B, 128, H/4, W/4]
        x = self.layer2(x)         # -> [B, 256, H/8, W/8]
        x = self.out_proj(x)       # -> [B, out_channels, H/8, W/8]
        return x
