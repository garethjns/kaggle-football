import json
import urllib.request
from http.client import IncompleteRead, HTTPResponse
from typing import Dict, Any, Tuple, Union

import numpy as np
import pandas as pd

from kaggle_football.api.side import Side
from kaggle_football.envs.raw_obs import RawObs

try:
    from gfootball.env import observation_preprocessing
    from gfootball.env.wrappers import Simple115StateWrapper
except ImportError:
    pass


class AgentEpisode:
    """
    Handle data for agent in episode.

    Downloaded episodes contain 2 agents.
    """
    _base_url = f'https://www.kaggleusercontent.com/episodes'
    _expected_steps: int = 3000

    def __init__(self, episode_id: int = None, side: int = 0, agent_id: int = None, score: int = 0):
        """

        :param episode_id: Numeric ID of this episode.
        :param side: Side the agent of interest is playing on in this episode.
        """
        self.episode_id = episode_id
        self.url = f'{self._base_url}/{episode_id}.json'
        self.downloaded = False
        self.side = Side(side)
        self.data: Dict[str, Any] = {}
        self.agent_id: Union[int, None] = agent_id
        self.s115_obs: Union[None, pd.DataFrame] = None
        self.smm_obs: Union[None, pd.DataFrame] = None
        self.actions_df: Union[None, pd.DataFrame] = None
        self.score = score

    def _multi_try_download(self, attempts: int = 5) -> Tuple[Union[HTTPResponse, None],
                                                              Union[Dict[str, Any], None]]:
        done = False
        attempt = 0
        resp, resp_data = None, None
        while (not done) and (attempt < attempts):
            try:
                resp = urllib.request.urlopen(self.url)
                resp_data = json.loads(resp.read())
                done = True
            except IncompleteRead:
                attempt += 1

        return resp, resp_data

    def download(self) -> bool:
        resp, resp_data = self._multi_try_download()

        if resp is None:
            return False

        resp_data["episode_id"] = self.episode_id
        success = resp.getcode() == 200
        if success:
            self.downloaded = True
            self.data = resp_data
            self.data['updatedScore'] = self.score

        return success

    def get_episode_data(self,
                         get_s115: bool = True, get_smm: bool = True,
                         get_raw: bool = True) -> Tuple[Union[None, np.ndarray],
                                                        Union[None, np.ndarray],
                                                        Union[None, np.ndarray],
                                                        Union[None, pd.DataFrame]]:
        """Convert episode dict to a df with shape (3000 (x2), n_obs_dims + meta columns)."""
        if not self.downloaded:
            self.download()

        try:
            steps = pd.Series(np.arange(1, len(self.data['steps']) - 1).astype(np.uint16), name='step')
            agent_s115_obs, agent_smm_obs, agent_raw_obs, agent_actions = self._side_to_df(get_s115=get_s115,
                                                                                           get_smm=get_smm,
                                                                                           get_raw=get_raw)

            actions_df = pd.concat((pd.Series(agent_actions.squeeze(), name='action'),
                                    steps), axis=1)
            actions_df.loc[:, 'episode_id'] = np.uint32(self.data["episode_id"])
            actions_df.loc[:, 'agent_id'] = np.uint32(self.agent_id)
            actions_df.loc[:, 'updated_score'] = np.int16(self.score)

            return agent_s115_obs, agent_smm_obs, agent_raw_obs, actions_df

        except (KeyError, IndexError, TypeError):
            # Incomplete JSON for episode?
            self.downloaded = False

    def _side_to_df(self, get_s115: bool = True,
                    get_smm: bool = True, get_raw: bool = True) -> Tuple[Union[None, np.ndarray],
                                                                         Union[None, np.ndarray],
                                                                         Union[None, np.ndarray],
                                                                         np.ndarray]:
        """Select the correct side/agent from the downloaded data and return as df."""
        raw_obs = []
        s115_obs = []
        smm_obs = []
        actions = []
        for step in np.arange(1, self._expected_steps + 1):
            players_raw = self.data['steps'][step][self.side.value]['observation']['players_raw']

            if get_s115:
                s115_obs.append(Simple115StateWrapper.convert_observation(players_raw, fixed_positions=True))

            if get_smm:
                smm_obs.append(observation_preprocessing.generate_smm([players_raw[0]]))

            if get_raw:
                raw_obs.append(RawObs.convert_observation(players_raw))

            actions.append(self.data['steps'][step][self.side.value]['action'][0])

        s115_obs = np.concatenate(s115_obs, axis=0) if get_s115 else None
        smm_obs = np.concatenate(smm_obs, axis=0) if get_smm else None
        raw_obs = np.concatenate(raw_obs, axis=0).astype(np.float32) if get_raw else None

        return s115_obs, smm_obs, raw_obs, np.expand_dims(np.array(actions, dtype=np.uint8), axis=1)
