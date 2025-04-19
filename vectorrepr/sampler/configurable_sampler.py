import numpy as np
from .base_sampler import BaseSampler


class ConfigurableSampler(BaseSampler):
    def __init__(self, anchors: list[int], candidates: list[list[int]], scores: list[float]):
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

        self.sample_pairs = [[a, c] for a, x in zip(anchors, candidates) for c in x]
        self.scores = np.concatenate(scores, axis=0, dtype=np.float32)
        assert len(self.sample_pairs) == len(self.scores)

    def sample(self, idx):
        return self.sample_pairs[idx], self.scores[idx]

    def __len__(self):
        return len(self.sample_pairs)


if __name__ == "__main__":
    sampler = ConfigurableSampler([0, 1, 2], [[1, 2], [0, 1, 2], [1]], [[1, 1], [0, 1, 1], [1]])
    print(len(sampler))
