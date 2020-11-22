import collections
from typing import Any, Dict, Tuple, Union, List

import gym


class BufferWrapper(gym.Wrapper):
    """General buffer wrapper to handle raw observations."""

    def __init__(self, env: Union[None, gym.Env] = None, buffer_length: int = 2) -> None:
        if env is not None:
            super().__init__(env)
        else:
            self.env = None

        self._buffer_length = buffer_length
        self._prepare_obs_buffer()

    def _prepare_obs_buffer(self) -> None:
        """Create buffer and preallocate with empty arrays of expected shape."""
        self._obs_buffer = collections.deque(maxlen=self._buffer_length)

        for _ in range(self._buffer_length):
            self._obs_buffer.append(None)

    def add(self, obs: Any):
        self._obs_buffer.append(obs)

    def get(self) -> List[Any]:
        return [self._obs_buffer[n] for n in range(self._buffer_length)]

    def step(self, action: int) -> Tuple[List[Any], float, bool, Dict[Any, Any]]:
        """Step env, add new obs to buffer, return buffer."""
        obs, reward, done, info = self.env.step(action)

        self.add(obs)

        return self.get(), reward, done, info

    def reset(self) -> List[Any]:
        """Add initial obs to end of pre-allocated buffer.

        :return: Buffered observation
        """
        self._prepare_obs_buffer()
        obs = self.env.reset()
        self.add(obs)
        return self.get()
