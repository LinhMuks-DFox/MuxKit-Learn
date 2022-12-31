import numpy as np
from mklearn.core.mk_types import *
from typing import Tuple


class TrainTestSplitter:
    def __init__(self, test_ratio: float = 0.2, seed: int = None):
        self.train_test_split_ratio_ = test_ratio
        assert 0.0 <= self.train_test_split_ratio_ <= 1.0
        if seed:
            np.random.seed(seed)

    def split(self, data: NDArray, label: NDArray) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
        shuffled_idx = np.random.permutation(len(data))
        test_set_size = int(len(data) * self.train_test_split_ratio_)
        test_indices = shuffled_idx[:test_set_size]
        train_indices = shuffled_idx[test_set_size:]

        return data[train_indices], data[test_indices], label[train_indices], label[test_indices]

    def __call__(self, data: NDArray, label: NDArray) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
        return self.split(data, label)

    def forward(self, data: NDArray, label: NDArray) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
        return self.split(data, label)


def train_test_split(data: NDArray, label: NDArray, test_ratio: float = 0.2, seed: int = None) -> Tuple[
    NDArray, NDArray, NDArray, NDArray]:
    splitter = TrainTestSplitter(test_ratio, seed)
    return splitter(data, label)
