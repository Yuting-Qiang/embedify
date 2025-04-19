import pytest
from vectorrepr.datasets.tabular.csv_dataset import CSVDataset
from vectorrepr.datasets.tabular.tabular_dataset import TabularDataset
from vectorrepr.sampler.configurable_sampler import ConfigurableSampler
import pandas as pd
import numpy as np


@pytest.fixture
def mock_config_sampler():
    return ConfigurableSampler([0, 1, 2], [[1, 2], [0, 1, 2], [1]], [[1, 1], [0, 1, 1], [1]])


@pytest.fixture
def sample_csv():
    origin_csv_path = "data/tmp/train_dataset_2023-01-01_2025-01-31.csv"
    new_csv_path = "data/tmp/clean_dataset.csv"

    feature_cols = ["High_n1"]
    for i in range(10):
        feature_cols += [
            f"Volume_p{i}",
            f"OpenMinMax_p{i}",
            f"CloseMinMax_p{i}",
            f"maxIncreaseRatio_p{i}",
            f"increaseRatio_p{i}",
        ]
    print(feature_cols)

    df = pd.read_csv(origin_csv_path, usecols=feature_cols)
    df.to_csv(new_csv_path, index=False)
    return str(new_csv_path), df


def test_csv_dataset_loading(sample_csv):
    _, df = sample_csv
    dataset = CSVDataset(df, label_column="High_n1")
    assert len(dataset) == 119778, "Dataset should have 119778 samples"
    assert dataset[0][0].shape == (50,), "Sample should have 50 features"
    assert dataset[0][1] is not None, "Target should not be None"


def test_csv_dataset_with_sampler(sample_csv, mock_config_sampler):
    _, df = sample_csv
    dataset = CSVDataset(df, label_column="High_n1", sampler=mock_config_sampler)
    assert len(dataset) == 6, "Dataset with sampler should have 6 samples"
    assert dataset[0][0].shape == (50,), f"anchor sample should have 50 features, {dataset[0][0]}"
    assert dataset[0][1].shape == (
        50,
    ), f"candidate sample should have 50 features, {dataset[0][0]}"


def test_tabular_dataset():
    data = np.random.rand(100, 5)
    dataset = TabularDataset(
        data_source=data,
        label_column=0,
        columns=["label", "feat1", "feat2", "feat3", "feat4"],
        na_handling="mean",
        na_values=[np.inf, np.nan],
    )
    assert len(dataset) == 100


def test_tabular_dataset_with_sampler():
    data = np.random.rand(100, 5)
    sampler = ConfigurableSampler([0, 1, 2], [[1, 2], [0, 1, 2], [1]], [[1, 1], [0, 1, 1], [1]])
    dataset = TabularDataset(
        data_source=data,
        columns=["label", "feat1", "feat2", "feat3", "feat4"],
        na_handling="mean",
        na_values=[np.inf, np.nan],
        sampler=sampler,
    )
    assert len(dataset) == 6
    assert dataset[0][0].shape == (5,), f"anchor sample should have 5 features, {dataset[0][0]}"
    assert dataset[0][1].shape == (5,), f"candidate sample should have 5 features, {dataset[0][0]}"
