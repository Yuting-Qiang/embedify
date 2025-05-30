import pandas as pd
import torch
from ..base import BaseDataset, DataType
from ...sampler import ConfigurableSampler, BaseSampler


class CSVDataset(BaseDataset):
    def __init__(
        self,
        data_source: str,
        label_column: str = None,
        delimiter: str = ",",
        na_values: str = "NA",
        na_handling: str = "drop",
        sampler: BaseSampler = None,
        transforms: list = None,
    ):
        """
        Parameters
        ----------
        data_source: str or pd.DataFrame
            - Path to the CSV file
            - loaded pandas DataFrame
        label_column: str, optional
            The column name for the label. If None, the dataset is assumed to be unlabelled.
        delimiter: str, optional
            The separator used in the CSV file. Defaults to ','
        na_values: str or list, optional
            The values to be recognized as NA/NaN. Defaults to 'NA'
        transforms: list, optional
            A list of transforms to apply to the data
        """
        super().__init__(DataType.TABULAR, transforms)

        # 处理数据加载
        if isinstance(data_source, str):
            self.df = pd.read_csv(
                data_source,
                delimiter=delimiter,
                na_values=na_values,
                engine="python",  # 增强编码兼容性
            )
        elif isinstance(data_source, pd.DataFrame):
            self.df = data_source.copy()  # 防止修改原始DF
        else:
            raise TypeError(f"Unsupported data source type: {type(data_source)}")

        # 在初始化时处理缺失值
        if na_handling == "mean":
            self.df.fillna(self.df.mean(), inplace=True)
        elif na_handling == "drop":
            self.df.dropna(inplace=True)

        # 处理标签
        self.has_labels = label_column is not None
        if self.has_labels:
            self.labels = self.df.pop(label_column).values
        else:
            self.labels = None

        # 转换为Tensor
        self.data = torch.tensor(self.df.values.astype("float32"))

        # 初始化Sampler
        self.sampler = sampler

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

            # 应用预处理管道
            for transform in self.preprocessing:
                anchor_sample = transform(anchor_sample)
                candidate_sample = transform(candidate_sample)

            return anchor_sample, candidate_sample, score

    def __len__(self):
        if self.sampler is None:
            return len(self.data)
        else:
            return len(self.sampler)


class LazyCSVDataset(CSVDataset):
    def __init__(self, file_path, chunk_size=1000):
        self.file_path = file_path
        self.chunk_size = chunk_size
        self.current_chunk = None

    def __getitem__(self, idx):
        chunk_num = idx // self.chunk_size
        if chunk_num != self.current_chunk:
            self._load_chunk(chunk_num)
        return super().__getitem__(idx % self.chunk_size)


if __name__ == "__main__":
    dataset1 = CSVDataset("data/tmp/clean_dataset.csv", label_column="High_n1")
    print(len(dataset1))
    # for i, (sample, target) in enumerate(dataset1):
    #     print(i, sample, target)

    sampler = ConfigurableSampler([0, 1, 2], [[1, 2], [0, 1, 2], [1]], [[1, 1], [0, 1, 1], [1]])
    dataset = CSVDataset("data/tmp/clean_dataset.csv", label_column="High_n1", sampler=sampler)
    for i, (anchor_sample, candidate_sample, score) in enumerate(dataset):
        print(i)
        print(anchor_sample)
        print(candidate_sample)
        print(score)
