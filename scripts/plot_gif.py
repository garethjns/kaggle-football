import gfootball  # noqa
import gym
from IPython.display import Image

from kaggle_football.viz import generate_gif

if __name__ == "__main__":
    smm_env = gym.make("GFootball-11_vs_11_kaggle-SMM-v0")
    generate_gif(smm_env)
    Image(filename='smm_env_replay.gif', format='png')
