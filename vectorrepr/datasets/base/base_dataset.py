from torch.utils.data import Dataset
from enum import Enum
from abc import abstractmethod


class DataType(Enum):
    TABULAR = "tabular"
    IMAGE = "image"  # TODO
    TEXT = "text"  # TODO
    AUDIO = "audio"  # TODO
    VIDEO = "video"  # TODO


class BaseDataset(Dataset):
    def __init__(self, data_type: DataType, preprocessing_pipeline: list = None):
        """
        Parameters
        ----------
        data_type: DataType
            The type of data to be stored in this dataset.
        preprocessing_pipeline: list, optional
            A list of transforms to apply to the data when it is retrieved.
            If not supplied, no transforms will be applied.
        """
        self.data_type = data_type
        self.preprocessing = preprocessing_pipeline or []

    @abstractmethod
    def __getitem__(self, idx):
        """return (sample, target)"""
        pass

    @abstractmethod
    def __len__(self):
        """return the number of samples in the dataset"""
        pass

    def add_transform(self, transform):
        """Add a transform to the preprocessing pipeline"""
        self.preprocessing.append(transform)
