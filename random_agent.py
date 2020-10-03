from kaggle_football.random_agent.random_agent import RandomAgent

AGENT = RandomAgent()


def agent(obs):
    return AGENT.get_action(obs)
