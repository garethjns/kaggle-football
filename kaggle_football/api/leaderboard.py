import pprint
from itertools import chain
from typing import List, Union, Dict, Tuple, Any

import numpy as np
import pandas as pd
import requests
from joblib import Parallel, delayed
from tqdm.auto import tqdm

from kaggle_football.api.agent_episode import AgentEpisode
from kaggle_football.api.data.hdf_repository import HDFRepository
from kaggle_football.api.data.json_repository import JSONRepository
from kaggle_football.api.side import Side


class LeaderBoard:
    json_repo: JSONRepository
    hdf_repo: HDFRepository
    url: str = "https://www.kaggle.com/requests/EpisodeService/ListEpisodes"

    _get_s115: bool
    _get_smm: bool
    _get_raw: bool

    def __init__(self, team_min_rank: int = 100, episode_min_score: float = 600,
                 n_download_jobs: int = 4, n_process_jobs: int = 8):
        """

        :param team_min_rank:
        :param episode_min_score: When downloading, only bother saving json of episodes score more than this value.
                                  When collecting, only include episodes above this score.
        :param n_download_jobs:
        :param n_process_jobs:
        """
        self.team_min_rank = team_min_rank
        self.episode_min_score = episode_min_score
        self.episode_wins_only = True  # Currently not used, always True

        self.df: Union[None, pd.DataFrame] = None
        self.actions_df: Union[None, pd.DataFrame] = None
        self.s115_obs: Union[None, np.ndarray] = None
        self.smm_obs: Union[None, np.ndarray] = None
        self.raw_obs: Union[None, np.ndarray] = None

        self.top_teams: Union[None, List[int]] = None
        self.episodes: Union[None, List[AgentEpisode]] = None
        self.n_download_jobs = n_download_jobs
        self.n_process_jobs = n_process_jobs

        self._download_pool = Parallel(n_jobs=self.n_download_jobs)
        self._process_pool = Parallel(n_jobs=self.n_process_jobs)

    def using(self, download_path: str = None, get_s115: bool = True, get_smm: bool = True, get_raw: bool = True):
        self.json_repo = JSONRepository().set_path(download_path)
        self._get_s115 = get_s115
        self._get_smm = get_smm
        self._get_raw = get_raw

    def get_top_teams(self) -> None:
        """Get top teams from Kaggle api

         If this fails with 404, manually set top teams with lb.set_top_teams(json.loads(top_teams.json))
        use existing top_teams.json"""
        resp = requests.post(self.url, json={"teamId": 5673355})

        self.set_top_teams(resp.json())

    def set_top_teams(self, resp_dict: Dict[str, Any]):
        self.df = pd.DataFrame(resp_dict['result']['teams'])
        self.top_teams = self.df.loc[self.df.publicLeaderboardRank <= self.team_min_rank, "id"].values.tolist()

    @staticmethod
    def list_team_episodes(url: str, team_id: int,
                           episode_min_score: int = 0) -> List[AgentEpisode]:
        """
        Get episodes for a team where the agents score after the episode is > min_score.

        team_id doesn't seem to be included in json response, so for each game, collect winner - may be from a different
        team. Ignore ties for now. Ultimately collecting multiple teams and returning just winners should produce list
        of unique episodes, but even if it doesn't this shouldn't matter.

        Example json response:
        {'id': 4871817,  < -- episode id
         'competitionId': 21723,
         'createTime': {'seconds': 1605532962, 'nanos': 211233000},
         'endTime': {'seconds': 1605533506, 'nanos': 891038300},
         'replayUrl': 'gs://kaggle-episode-replays/4871817.json',
         'adminNotes': None,
         'state': 'completed',
         'type': 'public',
         'agents': [{'id': 13440364,  < -- agent id
           'submissionId': 17691035,
           'reward': 0.0,  < -- reward
           'index': 0,
           'initialScore': 1008.757902299554,
           'initialConfidence': 35.0,
           'updatedScore': 1008.0689896621802,
           'updatedConfidence': 35.0,
           'state': 'unspecified',
           'submission': None},
          {'id': 13440366,  < -- agent id
           'submissionId': 17678702,
           'reward': 0.0,  < -- reward
           'index': 1,
           'initialScore': 970.6367307037693,
           'initialConfidence': 35.0,
           'updatedScore': 971.3256433411431,
           'updatedConfidence': 35.0,
           'state': 'unspecified',
           'submission': None}]}
        team_id
        Out[5]: 5587417


        :param team_id: Numeric ID of team to get
        :return: List of episode IDs
        """
        resp = requests.post(url, json={"teamId": team_id})

        log = {'None reward': 0,
               'None updatedScore': 0,
               'Score too low': 0,
               'Tie, ignoring': 0,
               'Left won': 0,
               'Right won': 0,
               'Valid episodes': 0}

        selected_episodes = []
        for ep in resp.json()['result']['episodes']:
            # Check if this agent is playing on the left or right
            # Special case for first episode - agent plays itself, but treat in same way (ie. just take left).

            # Check reward available
            if ep['agents'][0]['reward'] is None or ep['agents'][1]['reward'] is None:
                # If None reward (why??)
                log['None reward'] += 1
                continue

            # Find winner
            if ep['agents'][0]['reward'] == ep['agents'][1]['reward']:
                log['Tie, ignoring'] += 1
                continue
            elif ep['agents'][0]['reward'] > ep['agents'][1]['reward']:
                log['Left won'] += 1
                side = Side(0)
            else:
                log['Right won'] += 1
                # side = Side(1)
                continue

            # Check score threshold
            if ep['agents'][side.value]['updatedScore'] is None:
                log['None updatedScore'] += 1
                continue

            if not ep['agents'][side.value]['updatedScore'] >= episode_min_score:
                # If None score (why??) or score below threshold
                log['Score too low'] += 1
                continue

            selected_episodes.append(AgentEpisode(episode_id=ep['id'], side=side.value, agent_id=team_id,
                                                  score=ep['agents'][side.value]['updatedScore']))

        log["Valid episodes"] = len(selected_episodes)
        print(f'Team id: {team_id}')
        pprint.pprint(log)

        return selected_episodes

    @staticmethod
    def _download_and_cache_episode(json_repo: JSONRepository, episode: AgentEpisode) -> None:
        """Download episode or collect from cache of downloaded episodes."""

        if not json_repo.episode_available(episode):
            episode.download()
            json_repo.add_episode(episode)

    def list_top_team_episodes(self) -> Union[Dict[str, List[AgentEpisode]], List[AgentEpisode]]:

        if self.top_teams is None:
            self.get_top_teams()

        if self.episodes is None:
            jobs = (delayed(self.list_team_episodes)(url=self.url, team_id=team_id,
                                                     episode_min_score=self.episode_min_score)
                    for team_id in tqdm(self.top_teams, desc='Listing top teams'))
            selected_episodes = self._download_pool(jobs)

            self.episodes = list(chain.from_iterable(selected_episodes))

        return self.episodes

    def download_episodes(self) -> None:
        if self.episodes is None:
            self.list_top_team_episodes()

        jobs = (delayed(self._download_and_cache_episode)(self.json_repo, ep_) for ep_ in
                tqdm(self.episodes, desc='Downloading'))
        self._download_pool(jobs)

    @staticmethod
    def _episodes_to_df(json_repo, episode_json: str,
                        get_s115: bool, get_smm: bool, get_raw: bool) -> Tuple[Union[None, np.ndarray],
                                                                               Union[None, np.ndarray],
                                                                               Union[None, np.ndarray],
                                                                               Union[None, pd.DataFrame]]:
        ep_ag = episode_json.split(f".{json_repo.EXT}")[0].split('AgentEpisode')[1]
        episode_id = int(ep_ag.split('_ep')[1].split('_ag')[0])
        agent_id = int(ep_ag.split('_ag')[1].split('_sc')[0])
        agent_sc = int(ep_ag.split('_sc')[1])

        ep = AgentEpisode(episode_id=episode_id, agent_id=agent_id, score=agent_sc)
        ep.data = json_repo.load_episode(ep)
        ep.downloaded = True
        try:
            s115_obs, smm_obs, raw_obs, actions_df = ep.get_episode_data(get_s115=get_s115, get_smm=get_smm,
                                                                         get_raw=get_raw)
            return s115_obs, smm_obs, raw_obs, actions_df
        except (IndexError, TypeError):
            pass

            return None, None, None, None

    def collect(self, episode_min_score: int = None):
        # TODO: Get episode score for filtering
        # TODO: Add useful raw features

        if episode_min_score is not None:
            eps_to_collect = [ep for ep in self.json_repo.available_episodes
                              if int(ep.split('_sc')[1].split(f".{self.json_repo.EXT}")[0]) > episode_min_score]
        else:
            eps_to_collect = self.json_repo.available_episodes

        jobs = (delayed(self._episodes_to_df)(json_repo=self.json_repo, episode_json=episode_path,
                                              get_s115=self._get_s115, get_smm=self._get_smm, get_raw=self._get_raw)
                for episode_path in tqdm(eps_to_collect, desc='Collecting'))
        loaded_episodes = self._process_pool(jobs)

        s115s = [ep[0] for ep in loaded_episodes if ep[0] is not None]
        if len(s115s) > 0:
            self.s115_obs = np.concatenate(s115s, axis=0)

        smms = [ep[1] for ep in loaded_episodes if ep[1] is not None]
        if len(smms) > 0:
            self.smm_obs = np.concatenate(smms, axis=0)

        raw_obs = [ep[2] for ep in loaded_episodes if ep[2] is not None]
        if len(raw_obs) > 0:
            self.raw_obs = np.concatenate(raw_obs, axis=0)

        self.actions_df = pd.concat((ep[3] for ep in loaded_episodes if ep[3] is not None), axis=0)

    def save(self, fn: str = "downloaded_games", recollect: bool = False) -> HDFRepository:
        if recollect:
            self.collect()

        self.hdf_repo = HDFRepository().set_path(fn)
        if self.s115_obs is not None:
            self.hdf_repo.add(self.s115_obs, key=self.hdf_repo.s115_key)

        if self.smm_obs is not None:
            self.hdf_repo.add(self.smm_obs, key=self.hdf_repo.smm_key)

        if self.raw_obs is not None:
            self.hdf_repo.add(self.raw_obs, key=self.hdf_repo.raw_key)

        self.hdf_repo.add(self.actions_df, key=self.hdf_repo.actions_key)

        return self.hdf_repo
