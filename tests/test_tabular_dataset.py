import pytest
from vectorrepr.datasets.tabular.csv_dataset import CSVDataset
import pandas as pd


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
    assert dataset[0][0].shape == (50, ), "Sample should have 50 features"
    assert dataset[0][1] is not None, "Target should not be None"
