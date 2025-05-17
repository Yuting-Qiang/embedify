import numpy as np
import pickle as pkl
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

from vectorrepr.datasets.timeseries import TimeSeriesDataset
from vectorrepr.sampler import ConfigurableSampler
from vectorrepr.models import TimeSeriesTransformerEmbedding, LSTMEncoder
from vectorrepr.trainer.contrastive.pairwise_train import train

dataset = TimeSeriesDataset(
    "data/processed/stock500_2023_2024_ts.parquet",
    time_idx="DateIdx",
    group_ids="Ticker",
    feature_columns=["Open", "High", "Low", "Close", "Volume", "Adj Close"],
    na_handling=-1,
    input_steps=30,
    predict_steps=5,
)
print(len(dataset))
print("Shape of sample: ", dataset[0][0].shape)
print("Shape of label: ", dataset[0][1].shape)
sampled_candidates = pkl.load(
    open("data/processed/stock500_2023_2024_sampled_candidates_v2_clean.pkl", "rb")
)
anchors = [x[0] for x in sampled_candidates]
candidates = [x[1] for x in sampled_candidates]
scores = [x[2] for x in sampled_candidates]
sampler = ConfigurableSampler(dataset, anchors[:10000], candidates[:10000], scores[:10000])

loader = DataLoader(sampler, batch_size=512, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMEncoder(6, 512).to(device, dtype=torch.float32)
criterion = nn.MSELoss(reduction="none")
optimizer = optim.Adam(model.parameters(), lr=0.00001)
train(loader, model, criterion, epochs=10, learning_rate=0.00001)
