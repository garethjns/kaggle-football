from typing import List, Union, Tuple

import h5py
import hdf5plugin
import numpy as np
import pandas as pd

from kaggle_football.api.data.repository_base import RepositoryBase


class HDFRepository(RepositoryBase):
    """Handle saving, loading of hdfs for downloaded and collected game data."""
    features: List[str]
    EXT: str = "hdf"
    non_features = ["Unnamed: 0", "episode_id", "agent_id"]
    target: str = "action"
    episode_id_key: str = 'episode_id'
    agent_id_key: str = 'agent_id'

    _default_path: str = 'hdf_repo'
    actions_key = 'actions'
    s115_key = 's115_obs'
    smm_key = 'smm_obs'
    raw_key = 'raw_obs'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._available_episodes: Union[None, List[int]] = None
        self._episodes_index: Union[None, pd.DataFrame] = None

    def add(self, arr: Union[np.ndarray, pd.DataFrame], key: str) -> None:
        if isinstance(arr, (pd.DataFrame, pd.Series)):
            arr.to_hdf(self.path, key=key, mode='a', complevel=self.compress_level, complib='blosc', format='fixed')
        else:
            with h5py.File(self.path, 'a') as f:
                if key in f.keys():
                    # If key already exists in file, remove
                    del f[key]
                try:
                    f.create_dataset(key, data=arr,
                                     **hdf5plugin.Blosc(cname='blosclz', clevel=2, shuffle=hdf5plugin.Blosc.SHUFFLE))
                except ValueError:
                    # ValueError: Unknown compression filter number: 32001 (no blosc :( )
                    f.create_dataset(key, data=arr, chunks=True, compression='lzf')

    @property
    def episodes_index(self) -> pd.Series:
        if self._episodes_index is None:
            try:
                self._episodes_index = pd.read_hdf(self.path, key=self.actions_key,
                                                   columns=[self.episode_id_key])  # noqa
            except TypeError:
                # Fixed format store doesn't allow loading individual columns
                self._episodes_index = pd.read_hdf(self.path, key=self.actions_key)[["episode_id"]]  # noqa

        return self._episodes_index  # noqa

    @property
    def available_episodes(self) -> List[int]:
        if self._available_episodes is None:
            self._available_episodes = list(np.unique(self.episodes_index))

        return self._available_episodes

    def _load_df_or_array(self, key: str, idx: Union[None, np.ndarray] = None) -> Union[pd.DataFrame, np.ndarray]:
        """Actions key loads a df, others are arrays."""
        if key == self.actions_key:
            if idx is None:
                data = pd.read_hdf(self.path, key=key)
            else:
                data = pd.read_hdf(self.path, key=key, where=idx)
        else:
            with h5py.File(self.path, 'r') as f:
                if idx is None:
                    data = f[key][...]
                else:
                    data = f[key][idx, ...]

        return data

    def load_episodes(self, episodes: List[int] = None,
                      keys: List[str] = None) -> Tuple[Union[pd.DataFrame, np.ndarray, None], ...]:
        """
        Load selected episode keys ('table' format only) or all episodes ('table' or 'fixed')

        :param episodes: Episodes to load.
        :param keys: Keys to load from [self.actions_key, self.s115_key, self.smm_key, self.raw_key].
        :return: Selected keys, always in same order (None if not loaded): (actions, s115, smm, raw).
        """

        all_keys = [self.actions_key, self.s115_key, self.smm_key, self.raw_key]
        if keys is None:
            keys = all_keys

        if episodes is None:
            idx = None
        else:
            idx = self.episodes_index[self.episode_id_key].isin(episodes).values

        loaded = []
        for k in all_keys:
            if k in keys:
                data = self._load_df_or_array(key=k, idx=idx)
            else:
                data = None
            loaded.append(data)

        return tuple(loaded)
