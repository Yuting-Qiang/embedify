from torch.utils.data import Dataset
from enum import Enum

class DataType(Enum):
    TABULAR = "tabular"
    IMAGE = "image" # TODO
    TEXT = "text" # TODO
    AUDIO = "audio" # TODO
    VIDEO = "video" # TODO

class BaseDataset(Dataset, ABC):
    def __init__(self, 
                data_type: DataType,
                preprocessing_pipeline: list = None):
        self.data_type = data_type
        self.preprocessing = preprocessing_pipeline or []

    @abstractmethod
    def __getitem__(self, idx):
        """必须返回 (sample, target) 的元组"""
        pass

    @abstractmethod
    def __len__(self):
        pass

    def add_transform(self, transform):
        self.preprocessing.append(transform)