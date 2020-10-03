from typing import Any

import numpy as np


class RandomAgent:
    def get_action(self, obs: Any) -> int:
        return [np.random.randint(19)]
