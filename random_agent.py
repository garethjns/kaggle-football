from typing import Any
from typing import List

import numpy as np


class RandomAgent:
    def get_action(self, obs: Any) -> int:
        return np.random.randint(19)


AGENT = RandomAgent()


def agent(obs) -> List[int]:
    return [AGENT.get_action(obs)]
