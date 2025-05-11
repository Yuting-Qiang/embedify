import numpy as np
from ..datasets.base import BaseDataset
from .base_sampler import BaseSampler


class ConfigurableSampler(BaseSampler):
    def __init__(
        self,
        dataset: BaseDataset,
        anchors: list[int],
        candidates: list[list[int]],
        scores: list[float] = None,
    ):
        """
        Initialize the ConfigurableSampler with anchors, candidates, and scores.

        Parameters
        ----------
            anchors : list[int]
                A list of anchor indices.
            candidates : list[list[int]]
                A list of lists containing candidate indices for each anchor.
            scores : list[float]
                A list of scores corresponding to each candidate list.
        """

        super().__init__()
        assert len(anchors) == len(candidates) == len(scores)
        for i in range(len(anchors)):
            assert len(candidates[i]) == len(scores[i])

        self.dataset = dataset
        self.sample_pairs = [[a, c] for a, x in zip(anchors, candidates) for c in x]
        self.scores = None
        if scores is not None:
            self.scores = np.concatenate(scores)
            assert len(self.sample_pairs) == len(self.scores)

    def sample(self, idx):
        if self.scores is not None:
            return self.sample_pairs[idx], self.scores[idx]
        else:
            return self.sample_pairs[idx], None

    def __len__(self):
        return len(self.sample_pairs)

    def __getitem__(self, idx):
        pair, score = self.sample(idx)
        anchor_sample = self.dataset[pair[0]]
        candidate_sample = self.dataset[pair[1]]
        if len(anchor_sample) > 1:
            anchor_sample = anchor_sample[0]
        if len(candidate_sample) > 1:
            candidate_sample = candidate_sample[0]

        return anchor_sample, candidate_sample, score.astype(np.float32)


if __name__ == "__main__":
    from ..datasets.timeseries import TimeSeriesDataset
    from torch.utils.data import DataLoader

    dataset = TimeSeriesDataset(
        "data/processed/stock500_2023_2024_ts.parquet",
        time_idx="DateIdx",
        group_ids="Ticker",
        feature_columns=["Open", "High", "Low", "Close", "Volume", "Adj Close"],
        na_handling=-1,
        input_steps=10,
        predict_steps=4,
    )
    print(len(dataset))
    print("Shape of sample: ", dataset[0][0].shape)
    print("Shape of label: ", dataset[0][1].shape)
    sampler = ConfigurableSampler(
        dataset, [0, 1, 2], [[1, 2], [0, 1, 2], [1]], [[1, 1], [0, 1, 1], [1]]
    )
    print(len(sampler))
    print(sampler[0])

    loader = DataLoader(sampler, batch_size=2, shuffle=True)
    for i, (anchor_sample, candidate_sample, score) in enumerate(loader):
        print(i)
        print(anchor_sample)
        print(candidate_sample)
        print(score)
