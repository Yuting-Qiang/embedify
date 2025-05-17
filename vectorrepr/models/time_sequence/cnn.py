import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeSeriesCNN(nn.Module):
    def __init__(self, input_dim=7, seq_len=30, output_dim=256):
        super().__init__()

        # 多尺度卷积并行分支
        self.branch1 = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.MaxPool1d(2),
        )

        self.branch2 = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.MaxPool1d(2),
        )

        # 残差主路径
        self.res_block = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Conv1d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )

        # 维度自适应模块
        self.dim_adapter = nn.Sequential(
            nn.Linear(128, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, output_dim),
            nn.LayerNorm(output_dim),
        )

        # 初始化参数
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # 输入形状: (batch_size, seq_len, input_dim)
        x = x.permute(0, 2, 1)  # 转换为 (batch, channels, seq)

        # 多尺度特征融合
        branch1_out = self.branch1(x)
        branch2_out = self.branch2(x)
        fused = torch.cat([branch1_out, branch2_out], dim=1)

        # 残差连接
        residual = fused
        out = self.res_block(fused) + residual.mean(dim=-1, keepdim=True)

        # 维度调整
        out = out.squeeze(-1)
        out = self.dim_adapter(out)

        # L2标准化
        return F.normalize(out, p=2, dim=-1)


# 测试用例
if __name__ == "__main__":
    model = TimeSeriesCNN()
    dummy_input = torch.randn(32, 30, 7)  # (batch, seq_len, features)
    output = model(dummy_input)
    print(f"输出形状: {output.shape}")  # 应为 (32, 256)
    print(f"特征范数: {torch.norm(output[0], p=2)}")  # 应接近1.0
