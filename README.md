# Embedify

A flexible framework to generate vector representation of different data using deep learning models. Designed for both research and production use cases. 
** still work in progress **

## Features
- 🕒 **Multiple Model Architectures**
  - LSTM/GRU with attention mechanisms
  - Transformer with temporal position encoding
  - TCN (Temporal Convolutional Networks)
  
- 🎯 **Training Paradigms**
  - Self-supervised learning (Masked Reconstruction, Contrastive Learning)
  - Supervised learning (Triplet Loss, Classification-guided)
  
- 📊 **Production-ready Features**
  - Seamless Pandas DataFrame integration
  - Built-in visualization tools (t-SNE/UMAP projections)
  - GPU acceleration support

## TODO
### data
- 实现基础CSV加载+归一化
- 添加对Parquet/Excel格式的支持
- 实现内存映射文件支持大型数据集
- 集成自动特征工程（如自动编码分类列）

## Quick Start
### basic usage
```python

from torch.utils.data import DataLoader
from vectorrepr.datasets.timeseries import TimeSeriesDataset
from vectorrepr.sampler import ConfigurableSampler
from vectorrepr.models import TimeSeriesTransformerEmbedding
from vectorrepr.trainer.contrastive.pairwise_train import train

# create dataloader with sampler

dataset = TimeSeriesDataset(
    "Your_dataframe",
    time_idx="DateIdx",
    group_ids="group",
    feature_columns=["col1", "col2"],
    na_handling=-1,
    input_steps=30,
    predict_steps=5,
)
sampler = ConfigurableSampler(
    dataset, 
    anchors = [0, 1, 2], 
    candidates = [[1, 2], [0, 1, 2], [1]], 
    scores = [[1.0, 1.0], [0.0, 1.0, 1.0], [1.0]]
)

train_loader = DataLoader(dataset, batch_size=128, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TimeSeriesTransformerEmbedding(6, 512, 8, 6).to(device)

train(train_loader, model, criterion, epochs=10, learning_rate=0.0001)
```
### visualization
WIP

## Documentation
WIP

## Citation
WIP

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         vectorrepr and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── vectorrepr   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes vectorrepr a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------

