from typing import Union, Optional, List, Callable
from pathlib import Path
from copy import deepcopy

import pandas as pd
import numpy as np
from loguru import logger

from .base import BaseDataset


class FastCachedDataset(BaseDataset):
    def __init__(
        self,
        samples: np.ndarray,
        labels: Optional[np.ndarray] = None,
    ):
        self.samples = samples
        self.labels = labels

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx]
