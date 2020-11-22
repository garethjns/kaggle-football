import numpy as np


class TimeDimension:
    array_len_: int
    idx_to_drop_: np.ndarray

    def __init__(self, n_episode_steps: int, n_roll_steps: int) -> None:
        self.n_episode_steps = n_episode_steps
        self.n_roll_steps = n_roll_steps

    def _validate_fit_input(self, x_rows: int) -> None:
        if x_rows % self.n_episode_steps:
            raise ValueError(f"n rows in x is not a multiple of n_episode_steps.")

    def fit(self, x_rows: int) -> "TimeDimension":
        self._validate_fit_input(x_rows)
        self.array_len_ = x_rows
        self.idx_to_drop_ = np.arange(0, x_rows, self.n_episode_steps)

        return self

    def _validate_transform_input(self, x: np.ndarray) -> None:
        if x.shape[0] != self.array_len_:
            raise ValueError(f"x shape does not match fitted shape.")

    def transform_1d(self, x: np.ndarray) -> np.ndarray:
        x_offset = np.roll(x, self.n_roll_steps, axis=0)

        x = np.delete(x, self.idx_to_drop_, axis=0)
        x_offset = np.delete(x_offset, self.idx_to_drop_, axis=0)

        return np.concatenate([x_offset, x], axis=1)
