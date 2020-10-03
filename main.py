import collections
from typing import Tuple, Dict, Any, Union

import gym
import numpy as np
from gfootball.env import observation_preprocessing
from tensorflow import keras


class SMMFrameProcessWrapper(gym.Wrapper):
    """
    Wrapper for processing frames from SMM observation wrapper from football env.

    Input is (72, 96, 4), where last dim is (team 1 pos, team 2 pos, ball pos,
    active player pos). Range 0 -> 255.
    Output is (72, 96, 4) as difference to last frame for all. Range -1 -> 1
    """

    def __init__(self, env: Union[None, gym.Env] = None,
                 obs_shape: Tuple[int, int] = (72, 96, 4)) -> None:
        """
        :param env: Gym env, or None. Allowing None here is unusual,
                    but we'll reuse the buffer functunality later in
                    the submission, when we won't be using the gym API.
        :param obs_shape: Expected shape of single observation.
        """
        if env is not None:
            super().__init__(env)
        self._buffer_length = 2
        self._obs_shape = obs_shape
        self._prepare_obs_buffer()

    @staticmethod
    def _normalise_frame(frame: np.ndarray):
        return frame / 255.0

    def _prepare_obs_buffer(self) -> None:
        """Create buffer and preallocate with empty arrays of expected shape."""

        self._obs_buffer = collections.deque(maxlen=self._buffer_length)

        for _ in range(self._buffer_length):
            self._obs_buffer.append(np.zeros(shape=self._obs_shape))

    def build_buffered_obs(self) -> np.ndarray:
        """
        Iterate over the last dimenion, and take the difference between this obs
        and the last obs for each.
        """
        agg_buff = np.empty(self._obs_shape)
        for f in range(self._obs_shape[-1]):
            agg_buff[..., f] = self._obs_buffer[1][..., f] - self._obs_buffer[0][..., f]

        return agg_buff

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[Any, Any]]:
        """Step env, add new obs to buffer, return buffer."""
        obs, reward, done, info = self.env.step(action)

        obs = self._normalise_frame(obs)
        self._obs_buffer.append(obs)

        return self.build_buffered_obs(), reward, done, info

    def reset(self) -> np.ndarray:
        """Add initial obs to end of pre-allocated buffer.

        :return: Buffered observation
        """
        self._prepare_obs_buffer()
        obs = self.env.reset()
        self._obs_buffer.append(obs)

        return self.build_buffered_obs()

try:
    tf_mod = keras.models.load_model('/kaggle_simulations/agent/test_model')
except:
    tf_mod = keras.models.load_model('test_model')

obs_buffer = SMMFrameProcessWrapper()


def agent(obs):
    global tf_mod
    global obs_buffer

    obs = obs['players_raw'][0]
    obs = observation_preprocessing.generate_smm([obs])
    obs_buffer._obs_buffer.append(obs)

    actions = tf_mod.predict(obs)
    action = np.argmax(actions)

    return [action]
