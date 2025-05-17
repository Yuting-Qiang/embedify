import numpy as np
import pickle as pkl
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

from vectorrepr.datasets.timeseries import TimeSeriesDataset
from vectorrepr.datasets.fast_dataset import FastCachedDataset
from vectorrepr.sampler import ConfigurableSampler
from vectorrepr.models import TimeSeriesTransformerEmbedding, LSTMEncoder, TimeSeriesCNN
from vectorrepr.trainer.contrastive.pairwise_train import train

samples = np.load("data/interim/stock500_2023_2024_samples.npz")["samples"]
labels = np.load("data/interim/stock500_2023_2024_labels.npz")["labels"]
dataset = FastCachedDataset(samples, labels)
# dataset = TimeSeriesDataset(
#     "data/processed/stock500_2023_2024_ts.parquet",
#     time_idx="DateIdx",
#     group_ids="Ticker",
#     feature_columns=["Open", "High", "Low", "Close", "Volume", "Adj Close"],
#     na_handling=-1,
#     input_steps=30,
#     predict_steps=5,
# )
print(len(dataset))
print("Shape of sample: ", dataset[0][0].shape)
print("Shape of label: ", dataset[0][1].shape)
sampled_candidates = pkl.load(
    open("data/processed/stock500_2023_2024_sampled_candidates_v2_clean.pkl", "rb")
)
anchors = [x[0] for x in sampled_candidates]
candidates = [x[1] for x in sampled_candidates]
scores = [x[2] for x in sampled_candidates]
sampler = ConfigurableSampler(dataset, anchors, candidates, scores)

loader = DataLoader(sampler, batch_size=1024, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = TimeSeriesTransformerEmbedding(6, 256, 4, 2).to(device, dtype=torch.float32)

# model = TimeSeriesCNN(6, 30, 256).to(device, dtype=torch.float32)
# model = LSTMEncoder(6, 256).to(device, dtype=torch.float32)
criterion = nn.MSELoss(reduction="none")
optimizer = optim.Adam(model.parameters(), lr=0.00001)
train(loader, model, criterion, epochs=10, learning_rate=0.00001)
