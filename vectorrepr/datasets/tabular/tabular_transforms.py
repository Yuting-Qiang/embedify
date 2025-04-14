from sklearn.preprocessing import StandardScaler, MinMaxScaler
import torch

class TabularNormalizer:
    def __init__(self, mode='standard'):
        self.mode = mode
        self.scaler = None

    def __call__(self, sample):
        if not self.scaler:
            self._fit(sample)
        return self.scaler.transform(sample.reshape(1, -1)).flatten()

    def _fit(self, sample):
        if self.mode == 'standard':
            self.scaler = StandardScaler()
        elif self.mode == 'minmax':
            self.scaler = MinMaxScaler()
        self.scaler.fit(sample.reshape(1, -1))

class ToTensor:
    """示例转换器（已内置在CSVDataset中）"""
    def __call__(self, sample):
        return torch.as_tensor(sample)