# tabular_dataset.py
from typing import Union, Optional, List, Callable
from pathlib import Path
import numpy as np
import pandas as pd
import torch

# import pyarrow.parquet as pq
import h5py

from vectorrepr.sampler import BaseSampler, ConfigurableSampler
from vectorrepr.datasets.base import BaseDataset

SUPPORTED_FORMATS = {
    ".csv": "pandas",
    ".parquet": "pandas",
    ".feather": "pandas",
    ".h5": "hdf5",
    ".hdf5": "hdf5",
    ".npz": "numpy",
    ".npy": "numpy",
    ".pkl": "pickle",
    ".joblib": "pickle",
}


class TabularDataset(BaseDataset):
    def __init__(
        self,
        data_source: Union[str, pd.DataFrame, np.ndarray, dict],
        label_column: Optional[Union[str, int]] = None,
        columns: Optional[List[str]] = None,
        na_values: Union[str, List[str]] = "NA",
        na_handling: str = "mean",
        sampler: Optional[BaseSampler] = None,
        transforms: Optional[List[Callable]] = None,
        **loader_kwargs,  # 各格式的特定参数
    ):
        """
        参数增强说明：
        data_source: 支持多种输入形式
            - 文件路径 (str)
            - pandas DataFrame
            - numpy数组 (需配合columns参数)
            - 字典形式 {'data': array, 'columns': list}
        label_column: 支持列名或列索引
        columns: 为numpy等无列名数据指定列名
        loader_kwargs: 各文件格式的加载参数
            - hdf5: key='data' (指定数据集路径)
            - parquet: use_pandas_metadata=True (保留元数据)
        """
        self.na_handling = na_handling
        self.transforms = transforms or []
        self.sampler = sampler
        self.na_values = na_values

        # 统一数据加载
        self.data, self.columns, self.labels = self._load_data(
            data_source, label_column, columns, **loader_kwargs
        )

        # 数据预处理管道
        self._preprocess()

    def _load_data(self, source, label_col, columns, **kwargs):
        """多格式数据加载核心逻辑"""
        if isinstance(source, pd.DataFrame):
            return self._load_pandas(source, label_col)

        elif isinstance(source, (np.ndarray, dict)):
            return self._process_array(source, label_col, columns)

        elif isinstance(source, str):
            return self._load_file(source, label_col, columns, **kwargs)

        else:
            raise TypeError(f"Unsupported data source type: {type(source)}")

    def _load_file(self, path, label_col, columns, **kwargs):
        """处理文件路径加载"""
        suffix = Path(path).suffix.lower()

        if suffix not in SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported file format: {suffix}")

        loader = getattr(self, f"_load_{SUPPORTED_FORMATS[suffix]}")
        return loader(path, label_col, columns, **kwargs)

    def _load_pandas(self, path, label_col, columns, **kwargs):
        """加载pandas兼容格式"""
        if path.endswith(".parquet"):
            df = pd.read_parquet(path, **kwargs)
        else:  # csv, feather等
            df = pd.read_csv(path, **kwargs)

        return self._process_pandas(df, label_col)

    def _load_hdf5(self, path, label_col, columns, key="data", **kwargs):
        """加载HDF5格式"""
        with h5py.File(path, "r") as f:
            data = f[key][()]
        return self._process_array(data, label_col, columns)

    def _load_numpy(self, path, label_col, columns, **kwargs):
        """加载numpy格式"""
        if path.suffix == ".npz":
            data = np.load(path, **kwargs)["arr_0"]
        else:
            data = np.load(path, **kwargs)
        return self._process_array(data, label_col, columns)

    def _load_pickle(self, path, label_col, columns, **kwargs):
        """加载序列化对象"""
        obj = pd.read_pickle(path, **kwargs)
        if isinstance(obj, pd.DataFrame):
            return self._process_pandas(obj, label_col)
        else:
            return self._process_array(obj, label_col, columns)

    def _process_pandas(self, df, label_col):
        """处理DataFrame数据"""
        labels = None
        if label_col is not None:
            if isinstance(label_col, str):
                labels = df.pop(label_col).values
            elif isinstance(label_col, int):
                labels = df.iloc[:, label_col].values
                df = df.drop(df.columns[label_col], axis=1)
        return df.to_numpy(), df.columns.tolist(), labels

    def _process_array(self, array, label_col, columns):
        """处理数组类数据"""
        if isinstance(label_col, int):
            labels = array[:, label_col]
            data = np.delete(array, label_col, axis=1)
            label_col_plus1 = label_col + 1
            columns = columns[:label_col] + columns[label_col_plus1:] if columns else None
        else:
            labels = None
            data = array

        # 自动生成列名
        if columns is None:
            columns = [f"feature_{i}" for i in range(data.shape[1])]
        elif len(columns) != data.shape[1]:
            raise ValueError("Columns length mismatch with data shape")

        return data, columns, labels

    def _preprocess(self):
        """统一预处理流程"""
        # 缺失值处理
        self._handle_missing_values()

        # 转换为Tensor
        self.data = torch.tensor(self.data.astype(np.float32))

        # 应用转换管道
        for transform in self.transforms:
            self.data = transform(self.data)

    def _handle_missing_values(self):
        """跨格式缺失值处理"""
        # 识别NaN位置（兼容不同格式的缺失表示）
        nan_mask = pd.isna(self.data) | np.isin(self.data, self.na_values)

        if self.na_handling == "mean":
            col_means = np.nanmean(np.where(nan_mask, np.nan, self.data), axis=0)
            self.data = np.where(nan_mask, col_means, self.data)
        elif self.na_handling == "drop":
            valid_rows = ~np.any(nan_mask, axis=tuple([i for i in range(1, len(self.data.shape))]))
            self.data = self.data[valid_rows]
            if self.labels is not None:
                self.labels = self.labels[valid_rows]

    def __getitem__(self, idx):
        if self.sampler is None:
            sample = self.data[idx]

            # 应用预处理管道
            for transform in self.preprocessing:
                sample = transform(sample)

            if self.has_labels:
                return sample, self.labels[idx]
            return sample, None
        else:
            pair, score = self.sampler.sample(idx)
            anchor_sample = self.data[pair[0]]
            candidate_sample = self.data[pair[1]]

            # # 应用预处理管道
            # for transform in self.preprocessing:
            #     anchor_sample = transform(anchor_sample)
            #     candidate_sample = transform(candidate_sample)

            return anchor_sample, candidate_sample, score

    def __len__(self):
        if self.sampler is None:
            return len(self.data)
        else:
            return len(self.sampler)


# 使用示例
if __name__ == "__main__":
    # 加载numpy数组
    data = np.random.rand(100, 5)
    dataset = TabularDataset(
        data_source=data,
        columns=["label", "feat1", "feat2", "feat3", "feat4"],
        na_handling="mean",
        na_values=[np.inf, np.nan],
    )
    print(len(dataset))

    sampler = ConfigurableSampler([0, 1, 2], [[1, 2], [0, 1, 2], [1]], [[1, 1], [0, 1, 1], [1]])
    dataset = TabularDataset(
        data, columns=["label", "feat1", "feat2", "feat3", "feat4"], sampler=sampler
    )
    print(len(dataset))
    print(dataset[0][0].shape)
    # for i, (anchor_sample, candidate_sample, score) in enumerate(dataset):
    #     print(i)
    #     print(anchor_sample)
    #     print(candidate_sample)
    #     print(score)
