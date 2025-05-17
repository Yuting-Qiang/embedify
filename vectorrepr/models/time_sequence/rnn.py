import math

import torch
from torch import nn


# 定义LSTM模型用于生成时序数据Embedding
class LSTMEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(LSTMEncoder, self).__init__()
        self.lstm = nn.LSTM(input_dim, latent_dim, batch_first=True)

    def forward(self, x):
        outputs, (h, c) = self.lstm(x)
        return h
