import torch
import numpy as np
import faiss

from vectorrepr.datasets.timeseries import TimeSeriesDataset
from vectorrepr.models import TimeSeriesTransformerEmbedding


# dataset = TimeSeriesDataset(
#     "data/processed/stock500_2023_2024_ts.parquet",
#     time_idx="DateIdx",
#     group_ids="Ticker",
#     feature_columns=["Open", "High", "Low", "Close", "Volume", "Adj Close"],
#     na_handling=-1,
#     input_steps=30,
#     predict_steps=5,
# )
# print(len(dataset))
# print("Shape of sample: ", dataset[0][0].shape)
# print("Shape of label: ", dataset[0][1].shape)


# samples = np.load("data/interim/stock500_2023_2024_samples.npz")["samples"]
# labels = np.load("data/interim/stock500_2023_2024_labels.npz")["labels"]
samples = np.load("data/interim/stock500_20250101_20250430_samples.npz")["data"]
labels = np.load("data/interim/stock500_20250101_20250430_labels.npz")["data"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TimeSeriesTransformerEmbedding(6, 512, 8, 6).to(device, dtype=torch.float32)

state_dict = torch.load("expr_ts/model_epoch6.pth", map_location=device)
# 移除 `module.` 前缀
state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
model.load_state_dict(state_dict)

# Generate embeddings for each sample in the dataset
model.eval()
embeddings = []
start_idx = 0
with torch.no_grad():
    while start_idx < len(samples):
        end_idx = min(start_idx + 1000, len(samples))
        sample = torch.tensor(samples[start_idx:end_idx], dtype=torch.float32)
        embedding = model(sample.to(device)).cpu().numpy()
        embeddings.append(embedding.mean(axis=1))
        start_idx = end_idx
        if start_idx % 10000 == 0:
            print(start_idx)
    # for i in range(0, len(dataset), 1000):
    #     sample = torch.tensor(dataset[i][0], dtype=torch.float32)
    #     embedding = model(sample.unsqueeze(0)).squeeze(0).cpu().numpy()
    #     embeddings.append(embedding)
embeddings = np.concatenate(embeddings, axis=0)
# np.savez_compressed("data/interim/stock500_2023_2024_embeddings.npz", embeddings=embeddings)
np.savez_compressed(
    "data/interim/stock500_20250101_20250430_embeddings.npz", embeddings=embeddings
)
# index = faiss.IndexFlatL2(embeddings[0].shape[0])
# index.add(np.array(embeddings))

# # Save the FAISS index to disk
# faiss.write_index(index, "faiss_index_epoch2_train.bin")
