from pathlib import Path
from typing import Optional, Tuple

from loguru import logger
import typer
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from vectorrepr.config import MODELS_DIR
from vectorrepr.datasets import TabularDataset
from vectorrepr.sampler import ConfigurableSampler
from vectorrepr.models import TimeSeriesTransformerEmbedding  # 假设模型定义在单独模块

app = typer.Typer()


def configure_device(use_cuda: bool = True) -> torch.device:
    """配置训练设备"""
    if use_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")
    return device


def initialize_model(
    input_dim: int, d_model: int, nhead: int, num_layers: int, device: torch.device
) -> TimeSeriesTransformerEmbedding:
    """初始化Transformer模型"""
    model = TimeSeriesTransformerEmbedding(
        input_dim=input_dim, d_model=d_model, nhead=nhead, num_layers=num_layers
    ).to(device)
    logger.info(f"Model initialized with {num_layers} layers and {nhead} attention heads")
    return model


def train_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """单epoch训练循环"""
    model.train()
    total_loss = 0.0
    with tqdm(loader, desc="Training", unit="batch") as progress:
        for batch in progress:
            inputs = batch.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, inputs).mean()  # 假设是自监督任务
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress.set_postfix(loss=loss.item())

    return total_loss / len(loader)


@app.command()
def train(
    model_dir: Path = MODELS_DIR,
    # 模型超参数
    input_dim: int = typer.Option(7, help="输入特征维度"),
    d_model: int = typer.Option(512, help="Transformer隐层维度"),
    nhead: int = typer.Option(8, help="注意力头数"),
    num_layers: int = typer.Option(6, help="Transformer层数"),
    # 训练参数
    epochs: int = typer.Option(10, help="训练轮数"),
    batch_size: int = typer.Option(128, help="批大小"),
    learning_rate: float = typer.Option(1e-4, help="学习率"),
    # 设备选项
    use_cuda: bool = typer.Option(True, help="是否使用GPU"),
):
    """模型训练主入口"""
    # 初始化配置
    device = configure_device(use_cuda)
    model = initialize_model(input_dim, d_model, nhead, num_layers, device)

    # 准备数据
    data = np.random.rand(100, 11, 7)
    sampler = ConfigurableSampler(
        categories=[0, 1, 2],
        combinations=[[1, 2], [0, 1, 2], [1]],
        weights=[[1, 1], [0, 1, 1], [1]],
    )
    dataset = TabularDataset(
        data_source=data,
        na_handling="drop",
        na_values=[np.inf, np.nan],
        sampler=sampler,
    )
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 初始化训练组件
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

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


if __name__ == "__main__":
    app()
