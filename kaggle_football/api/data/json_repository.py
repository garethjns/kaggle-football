import glob
import gzip
import json
import os
from typing import Dict, Any, List

from kaggle_football.api.agent_episode import AgentEpisode
from kaggle_football.api.data.repository_base import RepositoryBase


class JSONRepository(RepositoryBase):
    EXT: str = "json.gzip"
    _default_path: str = 'json_repo/'

    def _fn_from_agent_episode(self, episode: AgentEpisode) -> str:
        return os.path.join(self.path, f"AgentEpisode_ep{episode.episode_id}"
                                       f"_ag{episode.agent_id}"
                                       f"_sc{int(episode.score)}.{self.EXT}")

    @staticmethod
    def _dict_to_bytes(d: Dict[str, Any]) -> bytes:
        return json.dumps(d).encode()

    @staticmethod
    def _bytes_to_dict(b: bytes) -> Dict[str, Any]:
        return json.loads(b.decode())

    def add_episode(self, episode: AgentEpisode):
        if episode.data is None:
            episode.download()

        with gzip.GzipFile(self._fn_from_agent_episode(episode), 'wb', compresslevel=self.compress_level) as f:
            f.write(self._dict_to_bytes(episode.data))

    def load_episode(self, episode: AgentEpisode) -> Dict[str, Any]:
        try:
            with gzip.GzipFile(self._fn_from_agent_episode(episode), 'rb', ) as f:
                loaded_json = self._bytes_to_dict(f.read())
        except json.decoder.JSONDecodeError:
            loaded_json = {}

        return loaded_json

    def episode_available(self, episode: AgentEpisode) -> bool:
        return os.path.exists(self._fn_from_agent_episode(episode))

    @property
    def available_episodes(self) -> List[str]:
        return glob.glob(os.path.join(self.path, f"*.{self.EXT}"))
