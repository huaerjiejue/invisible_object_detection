import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)  # [T, C]
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数位置
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数位置

        self.pe = pe.unsqueeze(0)  # shape: [1, T, C]

    def forward(self, x):
        # x: [B, T, C, H, W]
        T = x.size(1)
        pe = self.pe[:, :T].to(x.device)  # [1, T, C]
        return x + pe.unsqueeze(-1).unsqueeze(-1)  # 广播到 [B, T, C, H, W]


class TemporalTransformer(nn.Module):
    def __init__(self, in_channels=256, num_heads=8, num_layers=2, dropout=0.1):
        super().__init__()
        self.in_channels = in_channels
        self.embed_dim = in_channels  # 不改变通道数
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=num_heads,
            dim_feedforward=2 * self.embed_dim,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pos_encoder = PositionalEncoding(self.embed_dim)

    def forward(self, x):
        # x: [B, T, C, H, W]
        B, T, C, H, W = x.shape
        x = x.view(B, T, C, -1).permute(0, 3, 1, 2)  # [B, HW, T, C]
        x = x.contiguous().view(B * H * W, T, C)     # -> [B*H*W, T, C]

        x = self.pos_encoder(x)                      # 加位置编码
        out = self.encoder(x)                        # [B*H*W, T, C]

        fused = out[:, -1, :]                        # 取最后一帧或平均均可
        fused = fused.view(B, H, W, C).permute(0, 3, 1, 2)  # -> [B, C, H, W]
        return fused
