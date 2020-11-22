import abc
import os
import pathlib
from typing import List, Tuple

import numpy as np


class RepositoryBase:
    path: str
    EXT: str

    def __init__(self, compress_level: int = 9) -> None:
        """
        :param compress_level: Compression level for gzip
        """
        self.compress_level = compress_level

    def set_path(self, path: str = None) -> "RepositoryBase":
        if path is None:
            path = self._default_path

        if path.endswith(os.path.sep) or path.endswith('/'):
            path = os.path.abspath(path)
            pathlib.Path(path).mkdir(exist_ok=True, parents=True)
        else:
            path = f"{path}.{self.EXT}"

        self.path = path

        return self

    @property
    @abc.abstractmethod
    def _default_path(self) -> str:
        pass

    @property
    @abc.abstractmethod
    def available_episodes(self) -> List[int]:
        pass

    @classmethod
    def split_episodes(cls, episode_ids: List[int], train_prop: float = 0.5,
                       max_group_n: int = np.inf) -> Tuple[List[int], List[int]]:
        n_episodes = len(episode_ids)
        n_train = int(train_prop * n_episodes)
        np.random.shuffle(episode_ids)
        train_eps = episode_ids[0:n_train]
        test_eps = episode_ids[n_train:]

        train_eps = train_eps[0:min(len(train_eps), max_group_n)]
        test_eps = test_eps[0:min(len(test_eps), max_group_n)]

        return train_eps, test_eps

    def split(self, train_prop: float = 0.5, max_group_n: int = np.inf) -> Tuple[List[int], List[int]]:
        return self.split_episodes(self.available_episodes, train_prop=train_prop, max_group_n=max_group_n)
