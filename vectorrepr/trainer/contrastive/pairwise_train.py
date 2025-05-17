import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from loguru import logger
from tqdm import tqdm
from pathlib import Path
from vectorrepr.models import TimeSeriesTransformerEmbedding, LSTMEncoder
import time


def configure_device(use_cuda: bool = True) -> torch.device:
    """配置训练设备"""
    if use_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")
    return device


# 计算余弦相似度
def cosine_similarity(x, y):
    return torch.nn.functional.cosine_similarity(x, y)


# 训练函数
def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    # 训练循环
    with tqdm(train_loader, desc="Training", unit="batch") as progress:
        for anchor, candidate, score in progress:
            # start = time.time()
            # 数据改成torch.float32
            anchor, candidate, score = (
                anchor.to(device, dtype=torch.float32),
                candidate.to(device, dtype=torch.float32),
                score.to(device, dtype=torch.float32),
            )
            # print("time cost for load data:", time.time() - start)
            # start = time.time()
            # 数据改成torch.float32
            optimizer.zero_grad()

            anchor_embedding = model(anchor)

            candidate_embedding = model(candidate)
            if isinstance(model, TimeSeriesTransformerEmbedding):
                anchor_embedding = torch.mean(anchor_embedding, dim=1)
                candidate_embedding = torch.mean(candidate_embedding, dim=1)
            elif isinstance(model, LSTMEncoder):
                anchor_embedding = torch.mean(anchor_embedding, dim=0)
                candidate_embedding = torch.mean(candidate_embedding, dim=0)

            similarities = cosine_similarity(anchor_embedding, candidate_embedding)

            each_loss = criterion(similarities, score)
            # loss = (each_loss.mean(dim=1)*(1-anchor_y[:, 0])*positive_ratio).sum() + (each_loss.mean(dim=1)*anchor_y[:, 0]*negative_ratio).sum()
            loss = each_loss.mean()
            # print("time cost for forward:", time.time() - start)
            # start = time.time()
            loss.backward()
            optimizer.step()
            # print("time cost for backward:", time.time() - start)
            running_loss += loss.item()
            progress.set_postfix(loss=loss.item())
            # print(loss.item())

    return running_loss / len(train_loader)


def train(
    train_loader,
    model,
    criterion,
    # 训练参数
    epochs: int,
    learning_rate: float,
    use_cuda: bool = True,
    optimizer: str = "adam",
    model_dir: Path = Path("./checkpoints"),
):
    """模型训练主入口"""
    # 初始化配置
    device = configure_device(use_cuda)

    if optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # 训练循环
    logger.info("Starting training...")
    for epoch in range(1, epochs + 1):
        with logger.contextualize(epoch=epoch):
            avg_loss = train_epoch(model, train_loader, criterion, optimizer, device)
            logger.info(f"Epoch {epoch} | Avg Loss: {avg_loss:.4f}")

            # 保存checkpoint
            checkpoint_path = model_dir / f"model_epoch{epoch}.pth"
            torch.save(model.state_dict(), checkpoint_path)
            logger.debug(f"Checkpoint saved to {checkpoint_path}")

    logger.success("Training completed successfully")


# 使用示例
if __name__ == "__main__":

    from vectorrepr.datasets import TabularDataset
    from vectorrepr.sampler import ConfigurableSampler
    from vectorrepr.models import TimeSeriesTransformerEmbedding

    data = np.random.rand(100, 11, 7)
    sampler = ConfigurableSampler(
        [0, 1, 2], [[1, 2], [0, 1, 2], [1]], [[1.0, 1.0], [0.0, 1.0, 1.0], [1.0]]
    )
    dataset = TabularDataset(
        data_source=data,
        na_handling="drop",
        na_values=[np.inf, np.nan],
        sampler=sampler,
    )
    train_loader = DataLoader(dataset, batch_size=128, shuffle=True)

    # 初始化模型、损失函数和优化器
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TimeSeriesTransformerEmbedding(7, 512, 8, 6).to(device)
    criterion = nn.MSELoss(reduction="none")
    optimizer = optim.Adam(model.parameters(), lr=0.00001)
    train(train_loader, model, criterion, epochs=10, learning_rate=0.00001)
