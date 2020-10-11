"""
Run agent defined in main.py with an observation similar to those that it will receive when run by the Kaggle runner.

Useful for debugging into main.py.
"""
import sys
from typing import List

import numpy as np
from kaggle_environments import make

sys.path.append("../")  # Assuming running from scripts/

from main import agent  # noqa

if __name__ == "__main__":
    env = make("football", configuration={"save_video": True, "scenario_name": "11_vs_11_kaggle"})
    env.reset()

    # This is the observation that is passed to agent function
    obs_kag_env = env.state[0]['observation']

    for _ in range(100):
        action = agent(obs_kag_env)

        try:
            assert isinstance(action, List)
            assert len(action) == 1
            assert isinstance(action[0], int) or isinstance(action[0], np.integer)

        except AssertionError:
            print("Debug here for errors.")

        # Environment step is list of agent actions, ie [[agent_1], [agent_2]], here there is 1 action per agent.
        other_agent_action = [0]
        full_obs = env.step([action, other_agent_action])
        obs_kag_env = full_obs[0]['observation']
