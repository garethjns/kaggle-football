from typing import Tuple, Dict, Any, List, Union

import gym
import numpy as np


class ProcessSimple115Wrapper(gym.Wrapper):
    """
    22 - (x,y) coordinates of left team players
    22 - (x,y) direction of left team players
    22 - (x,y) coordinates of right team players
    22 - (x, y) direction of right team players
    3 - (x, y and z) - ball position
    3 - ball direction
    3 - one hot encoding of ball ownership (noone, left, right)
    11 - one hot encoding of which player is active
    7 - one hot encoding of game_mode
    """

    def __init__(self, env: gym.Env = None):
        """
        :param env: A gym env, or None.
        """
        if env is not None:
            super().__init__(env)

        self.simple_obs_shape = (115,)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=self.simple_obs_shape,
                                                dtype=np.float32)

    @staticmethod
    def process_obs(obs: Union[Dict[str, Any], List[Any]]) -> Tuple[np.ndarray, np.ndarray]:
        """
        22 - (x,y) coordinates of left team players  x_min = -1, y_min=-0.42, x_max = 1, y_max = 0.42
        22 - (x,y) direction of left team players
        22 - (x,y) coordinates of right team players
        22 - (x, y) direction of right team players
        3 - (x, y and z) - ball position
        3 - ball direction
        3 - one hot encoding of ball ownership (noone, left, right)
        11 - one hot encoding of which player is active
        7 - one hot encoding of game_mode
        """
        pass

    def step(self, action: int) -> Tuple[Tuple[np.ndarray, np.ndarray], float, bool, Dict[Any, Any]]:
        obs, reward, done, info = self.env.step(action)

        return self.process_obs(obs), reward, done, info

    def reset(self) -> Tuple[np.ndarray, np.ndarray]:
        obs = self.env.reset()
        return self.process_obs(obs)
