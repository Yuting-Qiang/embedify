import math

import torch
from torch import nn


# 定义Transformer模型用于生成时序数据Embedding
class TimeSeriesTransformerEmbedding(nn.Module):
    def __init__(self, input_size, d_model, nhead, num_layers, dim_feedforward=2048, dropout=0.1):
        super(TimeSeriesTransformerEmbedding, self).__init__()
        self.embedding = nn.Linear(input_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

    def forward(self, x):
        # x的形状为(batch_size, sequence_length, input_size)
        x = self.embedding(x)
        x = self.positional_encoding(x.permute(1, 0, 2)).permute(1, 0, 2)
        output = self.transformer_encoder(x)
        return output


# 位置编码层
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return x
