from typing import Union, Optional, List, Callable
from pathlib import Path
from copy import deepcopy

import pandas as pd
import numpy as np
from loguru import logger

from .base import BaseDataset


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


class TimeSeriesDataset(BaseDataset):
    def __init__(
        self,
        data_source: Union[str, pd.DataFrame, np.ndarray, dict],
        time_idx: str,
        group_ids: str,
        feature_columns: Optional[List[str]],
        label_column: Optional[Union[str, int]] = None,
        input_steps: int = 1,
        predict_steps: int = 1,
        na_handling: str = "drop",
        transforms: Optional[List[Callable]] = None,
        return_group_time: bool = False,
        **loader_kwargs,  # 各格式的特定参数
    ):
        """
        仿照pytorch-forcasting接口实现
        参数说明：
        data_source: 支持多种输入形式
            - 文件路径 (str)
            - pandas DataFrame
            - numpy数组 (需配合columns参数)
            - 字典形式 {'data': array, 'columns': list}
        time_idx: 时间索引列名, 该列的类型是整型
        group_ids: 分组序列列名,
        feature_columns: 特征列名
        label_column: 支持列名或列索引
        input_steps: 输入窗口大小
        predict_steps: 预测窗口大小
        na_handling: 缺失值处理方式,
            - "drop": 去除包含缺失值的行
            - "mean": 缺失值用全局平均值填充
            - "mean_by_group": 缺失值用每组的平均值填充
            - "mean_by_time": 缺失值用每个时间步的平均值填充
            - 其它: 用na_handling参数指定的值填充
        loader_kwargs: 各文件格式的加载参数
            - hdf5: key='data' (指定数据集路径)
            - parquet: use_pandas_metadata=True (保留元数据)
        """
        self.na_handling = na_handling
        self.transforms = transforms or []
        self.group_ids = deepcopy(group_ids)
        self.time_idx = time_idx
        self.feature_columns = deepcopy(feature_columns)
        self.input_steps = input_steps
        self.predict_steps = predict_steps
        self.return_group_time = return_group_time

        # 统一数据加载
        self.data, self.labels = self._load_data(
            data_source, time_idx, group_ids, label_column, feature_columns, **loader_kwargs
        )

        # 数据预处理管道
        self._preprocess()

    def _load_data(self, source, time_idx, group_ids, label_col, columns, **kwargs):
        """多格式数据加载核心逻辑"""
        if isinstance(source, pd.DataFrame):
            return self._load_pandas(source, time_idx, group_ids, label_col, columns)

        elif isinstance(source, (np.ndarray, dict)):
            return self._process_array(source, time_idx, group_ids, label_col, columns)

        elif isinstance(source, str):
            return self._load_file(source, time_idx, group_ids, label_col, columns, **kwargs)

        else:
            raise TypeError(f"Unsupported data source type: {type(source)}")

    def _load_file(self, path, time_idx, group_ids, label_col, columns, **kwargs):
        """处理文件路径加载"""
        suffix = Path(path).suffix.lower()

        if suffix not in SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported file format: {suffix}")

        loader = getattr(self, f"_load_{SUPPORTED_FORMATS[suffix]}")
        return loader(path, time_idx, group_ids, label_col, columns, **kwargs)

    def _load_pandas(self, path_or_df, time_idx, group_ids, label_col, columns, **kwargs):
        """加载pandas兼容格式"""

        columns.append(time_idx)
        columns.append(group_ids)
        if label_col is not None:
            columns.append(label_col)

        if isinstance(path_or_df, pd.DataFrame):
            df = path_or_df
        elif path_or_df.endswith(".parquet"):
            df = pd.read_parquet(path_or_df, columns=columns, **kwargs)
        else:  # csv, feather等
            df = pd.read_csv(path_or_df, columns=columns, **kwargs)

        return self._process_pandas(df, time_idx, group_ids, label_col)

    def _process_pandas(self, df, time_idx, group_ids, label_col):
        """处理DataFrame数据"""
        labels = None
        if label_col is not None:
            labels = df.pop(label_col).values
        df.sort_values([group_ids, time_idx], inplace=True)
        # self.group_counter = Counter(df[group_ids].values.tolist())
        self.group_counter = (
            df.groupby(group_ids, sort=True).size().reset_index(name="count").values
        )
        return df, labels

    def _preprocess(self):
        """统一预处理流程"""
        # 缺失值处理
        self._handle_missing_values()

        # 应用转换管道
        for transform in self.transforms:
            self.data = transform(self.data)

    def _handle_missing_values(self):
        """跨格式缺失值处理"""
        if self.na_handling == "drop":
            self.data.dropna(inplace=True)
        elif self.na_handling == "mean":
            self.data[self.feature_columns] = self.data[self.feature_columns].fillna(
                self.data[self.feature_columns].mean()
            )
        elif self.na_handling == "mean_by_group":
            self.data[self.feature_columns] = self.data[self.feature_columns].fillna(
                self.data.groupby(self.group_ids)[self.feature_columns].transform("mean")
            )
        elif self.na_handling == "mean_by_time":
            self.data[self.feature_columns] = self.data[self.feature_columns].fillna(
                self.data.groupby(self.time_idx)[self.feature_columns].transform("mean")
            )
        else:
            self.data.fillna(self.na_handling, inplace=True)

    def __len__(self):
        res = 0
        for key, length in self.group_counter:
            if length < self.input_steps + self.predict_steps:
                logger.info(
                    "Group {} has less than {} steps, skipped.".format(
                        key, self.input_steps + self.predict_steps
                    )
                )
            else:
                res += length - self.input_steps - self.predict_steps + 1
        return res

    def __getitem__(self, idx):
        cur_idx = idx
        start_row_idx = 0
        for key, length in self.group_counter:
            if length < self.input_steps + self.predict_steps:
                continue
            if cur_idx < length - self.input_steps - self.predict_steps + 1:
                start_row_idx += cur_idx
                break
            cur_idx -= length - self.input_steps - self.predict_steps + 1
            start_row_idx += length
        assert start_row_idx < len(self.data)
        input = (
            self.data[self.feature_columns]
            .iloc[start_row_idx : start_row_idx + self.input_steps]
            .values.astype(np.float32)
        )
        output = (
            self.data[self.feature_columns]
            .iloc[
                start_row_idx
                + self.input_steps : start_row_idx
                + self.input_steps
                + self.predict_steps
            ]
            .values.astype(np.float32)
        )
        if self.return_group_time:
            return (
                input,
                output,
                self.data[self.group_ids].iloc[start_row_idx + self.input_steps - 1],
                self.data[self.time_idx].iloc[start_row_idx + self.input_steps - 1],
            )
        else:
            return input, output


if __name__ == "__main__":
    # from datetime import datetime

    # 数据配置
    # start_date = "2023-01-01"
    # end_date = "2024-12-31"
    # pre_days = 10
    # post_days = 4
    # stock_datapath = "data/external/stock.parquet.gz"
    # df = pd.read_parquet(f"{stock_datapath}", engine="pyarrow")
    # df = df[(df.index.astype("str") >= start_date) & (df.index.astype("str") <= end_date)]
    # df = df.T.reset_index().melt(
    #     id_vars=["Ticker", "Price"],
    #     value_vars=list(df.T.columns),
    #     var_name="Date",
    #     value_name="value",
    # )
    # df = df.pivot(index=["Ticker", "Date"], columns="Price", values=df.columns[3:])
    # df.columns = df.columns.droplevel(0)
    # df.reset_index(inplace=True)
    # df["Date"] = pd.to_datetime(df["Date"])
    # df["DateIdx"] = (
    #     pd.to_datetime(df["Date"]) - datetime.strptime("2023-01-01", "%Y-%m-%d")
    # ).apply(lambda x: x.days)
    # print(len(df))
    # df.to_parquet("data/processed/stock500_2023_2024_ts.parquet")
    df = pd.read_parquet("data/processed/stock500_2023_2024_ts.parquet")
    dataset = TimeSeriesDataset(
        # "data/processed/stock500_2023_2024_ts.parquet",
        df,
        time_idx="DateIdx",
        group_ids="Ticker",
        feature_columns=["Open", "High", "Low", "Close", "Volume", "Adj Close"],
        na_handling=-1,
        input_steps=10,
        predict_steps=4,
    )

    print(len(dataset))
    print(dataset[0])
    print(dataset)
