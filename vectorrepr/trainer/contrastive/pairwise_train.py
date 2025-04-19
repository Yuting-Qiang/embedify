import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

from vectorrepr.datasets import TabularDataset
from vectorrepr.sampler import ConfigurableSampler
from vectorrepr.models import TimeSeriesTransformerEmbedding


parser = argparse.ArgumentParser()
parser.add_argument("--d_model", type=int, default=512)
parser.add_argument("--nhead", type=int, default=8)
parser.add_argument("--num_layers", type=int, default=6)
parser.add_argument("--dim_feedforward", type=int, default=2048)
parser.add_argument("--dropout", type=float, default=0.1)
parser.add_argument("--label_thr", type=float, default=1.10)
args = parser.parse_args()


# 计算余弦相似度
def cosine_similarity(x, y):
    return torch.nn.functional.cosine_similarity(x, y, dim=2)


# 训练函数
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for batch_idx, (anchor, candidate, score) in enumerate(train_loader):
        anchor, candidate, score = (anchor.to(device), candidate.to(device), score.to(device))

        optimizer.zero_grad()

        anchor_embedding = model(anchor)
        anchor_embedding = torch.mean(anchor_embedding, dim=1)

        candidate_embedding = model(candidate)
        candidate_embedding = torch.mean(candidate_embedding, dim=1)

        similarities = cosine_similarity(anchor_embedding, candidate_embedding)

        each_loss = criterion(similarities, score)

        # loss = (each_loss.mean(dim=1)*(1-anchor_y[:, 0])*positive_ratio).sum() + (each_loss.mean(dim=1)*anchor_y[:, 0]*negative_ratio).sum()
        loss = each_loss.mean()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        print(loss.item())

    return running_loss / len(train_loader)


data = np.random.rand(100, 11, 7)
sampler = ConfigurableSampler([0, 1, 2], [[1, 2], [0, 1, 2], [1]], [[1, 1], [0, 1, 1], [1]])
dataset = TabularDataset(
    data_source=data,
    columns=["label", "feat1", "feat2", "feat3", "feat4"],
    na_handling="drop",
    na_values=[np.inf, np.nan],
    sampler=sampler,
)
train_loader = DataLoader(dataset, batch_size=128, shuffle=True)

# 初始化模型、损失函数和优化器
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TimeSeriesTransformerEmbedding(7, args.d_model, args.nhead, args.num_layers).to(device)
criterion = nn.MSELoss(reduction="none")
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# 训练模型
num_epochs = 10
for epoch in range(num_epochs):
    train_loss = train(model, train_loader, criterion, optimizer, device)
    print(f"Epoch {epoch + 1}, Train Loss: {train_loss}")
    # 存储模型
    torch.save(model.state_dict(), f"model_{epoch}.pth")
