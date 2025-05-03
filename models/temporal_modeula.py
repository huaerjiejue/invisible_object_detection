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
    def __init__(self, config):
        super().__init__()
        temporal_config = config.get_temporal_config()
        pos_encoding_config = temporal_config['positional_encoding']
        
        self.in_channels = temporal_config['in_channels']
        self.embed_dim = self.in_channels  # 不改变通道数
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=temporal_config['num_heads'],
            dim_feedforward=2 * self.embed_dim,
            dropout=temporal_config['dropout'],
            activation='relu',
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=temporal_config['num_layers']
        )
        self.pos_encoder = PositionalEncoding(
            d_model=pos_encoding_config['d_model'],
            max_len=pos_encoding_config['max_len']
        )

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
