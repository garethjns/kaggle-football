import collections
from typing import Tuple, Dict, Any, List, Union

import gym
import numpy as np
from gfootball.env import observation_preprocessing
from gfootball.env.config import Config
from gfootball.env.wrappers import Simple115StateWrapper, SMMWrapper


class SimpleAndSMMObsWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env = None):
        if env is not None:
            super().__init__(env)

        self.observation_space = gym.spaces.Tuple(
            [gym.spaces.Box(low=0, high=255, shape=(1, 72, 96, 4), dtype=np.uint8),
             gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1, 115), dtype=np.float32)])

    @staticmethod
    def process_obs(obs: Union[Dict[str, Any], List[Any]]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Obs can be from gym env or the version passed from Kaggle runner.

        We need to extract this dict to generate simple and SMM obs:
        dict_keys(['left_team_tired_factor', 'left_team_yellow_card', 'right_team_tired_factor', 'left_team',
                    'ball_owned_player', 'right_team_yellow_card', 'ball_rotation', 'ball_owned_team', 'ball',
                    'right_team_roles', 'right_team_active', 'steps_left', 'score', 'right_team', 'left_team_roles',
                    'ball_direction', 'left_team_active', 'left_team_direction', 'right_team_direction', 'game_mode',
                    'designated', 'active', 'sticky_actions'])

        Which is located in:
         - Kag obs: obs_kag_env['players_raw'][0].keys():
         - Gym obs: obs_gym_env[0].keys()
        """

        if isinstance(obs, dict):
            obs = obs['players_raw']

        simple_obs = Simple115StateWrapper.convert_observation(obs, fixed_positions=False)
        smm_obs = observation_preprocessing.generate_smm([obs[0]])

        return smm_obs, simple_obs

    def step(self, action: int) -> Tuple[Tuple[np.ndarray, np.ndarray], float, bool, Dict[Any, Any]]:
        obs, reward, done, info = self.env.step(action)

        return self.process_obs(obs), reward, done, info

    def reset(self) -> Tuple[np.ndarray, np.ndarray]:
        obs = self.env.reset()
        return self.process_obs(obs)


class SMMFrameProcessWrapper(gym.Wrapper):
    """
    Wrapper for processing frames from SMM observation wrapper from football env.

    Input is (72, 96, 4), where last dim is (team 1 pos, team 2 pos, ball pos,
    active player pos). Range 0 -> 255.
    Output is (72, 96, 4) as difference to last frame for all. Range -1 -> 1

    If the observation passed is a Tuple, assumes SMM component is [0].
    """

    def __init__(self, env: Union[None, gym.Env] = None,
                 obs_shape: Tuple[int, int] = (72, 96, 4)) -> None:
        """
        :param env: Gym env, or None. Allowing None here is unusual,
                    but we'll reuse the buffer functionality later in
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
        Iterate over the last dimension, and take the difference between this obs
        and the last obs for each.
        """
        agg_buff = np.empty(self._obs_shape)
        for f in range(self._obs_shape[-1]):
            agg_buff[..., f] = self._obs_buffer[1][..., f] - self._obs_buffer[0][..., f]

        return agg_buff

    def process_obs(self, obs: Union[np.ndarray, List[np.ndarray]]) -> Union[np.ndarray, List[np.ndarray]]:
        if isinstance(obs, tuple):
            smm_obs = obs[0]
            other_obs = obs[1]
        else:
            smm_obs = obs
            other_obs = None

        smm_obs_norm = self._normalise_frame(smm_obs)
        self._obs_buffer.append(smm_obs_norm)
        buffered_smm_obs = self.build_buffered_obs()

        if other_obs is None:
            return buffered_smm_obs
        else:
            return [buffered_smm_obs, other_obs]

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[Any, Any]]:
        """Step env, add new obs to buffer, return buffer."""
        obs, reward, done, info = self.env.step(action)

        return self.process_obs(obs), reward, done, info

    def reset(self) -> np.ndarray:
        """Add initial obs to end of pre-allocated buffer.

        :return: Buffered observation
        """
        self._prepare_obs_buffer()
        obs = self.env.reset()

        return self.process_obs(obs)


if __name__ == "__main__":
    import gfootball  # noqa

    from kaggle_environments import make

    env = make("football", configuration={"save_video": True, "scenario_name": "11_vs_11_kaggle"})

    # This is the observation that is passed on agent function.
    obs_kag_env = env.state[0]['observation']

    print(obs_kag_env.keys())

    simple_obs_ = Simple115StateWrapper.convert_observation(obs_kag_env['players_raw'], fixed_positions=False)
    smm_obs_ = observation_preprocessing.generate_smm([obs_kag_env['players_raw'][0]])

    base_env = gym.make("GFootball-11_vs_11_kaggle-SMM-v0").unwrapped
    obs_gym_env = base_env.reset()

    wrapped_env = SimpleAndSMMObsWrapper(base_env.unwrapped)
    wrapped_env.reset()

    SimpleAndSMMObsWrapper.process_obs(obs_kag_env)
    SimpleAndSMMObsWrapper.process_obs(obs_gym_env)

    buff_wrapped_env = SMMFrameProcessWrapper(wrapped_env)
    buff_obs = buff_wrapped_env.reset()
    buff_obs = buff_wrapped_env.step(1)

    buffed_smm = SMMFrameProcessWrapper(SMMWrapper(gym.make("GFootball-11_vs_11_kaggle-SMM-v0").unwrapped))
    buffed_smm_obs = buffed_smm.reset()
    buffed_smm_obs = buffed_smm.step(1)

    import json

    kaggle_config = json.load(open('../football.json', 'r'))

    config = Config()
    config.update(kaggle_config)
    config._values['level']

    from gfootball.env import create_environment

    env_from_kaggle_config = create_environment(env_name='11_vs_11_kaggle',
                                                stacked=False,
                                                representation='raw',
                                                write_goal_dumps=False,
                                                write_full_episode_dumps=False,
                                                write_video=False,
                                                render=False,
                                                number_of_left_players_agent_controls=1,
                                                number_of_right_players_agent_controls=1)

    wrapped_env_from_kaggle_config = SMMFrameProcessWrapper(SimpleAndSMMObsWrapper(env_from_kaggle_config))
