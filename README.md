# VectorRepr

A flexible framework to generate vector representation of different data using deep learning models. Designed for both research and production use cases. 

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

## Quick Start
### basic usage
```python
from tsembed import TemporalEmbedder

# 初始化模型
embedder = TemporalEmbedder(model="transformer", input_dims=3)

# 自监督训练
embedder.fit(your_data)

# 生成Embedding
embeddings = embedder.transform(your_data)
```
### visualization
```python
from tsembed.visualization import plot_embeddings
```

## Documentation
Full documentation is available at https://yourusername.github.io/tsembed (when deployed) or:
```bash
python -m pydoc tsembed.TemporalEmbedder
```

## Citation
Citation
If you use TSEmbed in your research, please cite:

```bibtex
@software{TSEmbed,
  author = {Your Name},
  title = {TSEmbed: A Flexible Time Series Embedding Library},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/yourusername/tsembed}}
}
```

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

