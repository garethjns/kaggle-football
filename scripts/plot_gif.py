import glob
import os
import tempfile
from typing import Tuple

import gfootball  # noqa
import gym
import imageio as imageio
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Image
from tqdm import tqdm


def plot_smm_obs(obs: np.ndarray) -> Tuple[plt.Figure, plt.Axes]:
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(8, 6))
    for i, (ax, title) in enumerate(zip(axs.flatten(), ['left team', 'right team', 'ball', 'active player'])):
        ax.imshow(obs[..., i], animated=True)
        ax.set_title(title)
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])

    return fig, ax


def generate_gif(env: gym.Env, suffix: str = "smm_env"):
    """Plot a few steps of an env and generate a .gif."""

    tmp_dir = tempfile.TemporaryDirectory()
    _ = env.reset()

    for s in tqdm(range(200)):
        obs, _, _, _ = env.step(5)
        fig, ax = plot_smm_obs(obs)
        fig.suptitle(f"Step: {s}")
        fig.tight_layout()
        fig.savefig(f'{os.path.join(tmp_dir.name, str(s))}.png')
        plt.close('all')

    fns = glob.glob(f'{tmp_dir.name}/*.png')
    sorted_idx = np.argsort([int(f.split(tmp_dir.name)[1].split('.png')[0].split('/')[1])
                             for f in fns])
    fns = np.array(fns)[sorted_idx]
    output_path = f"{suffix}replay.gif"
    images = [imageio.imread(f) for f in fns]
    imageio.mimsave(output_path, images,
                    duration=0.1,
                    subrectangles=True)

    tmp_dir.cleanup()


if __name__ == "__main__":
    smm_env = gym.make("GFootball-11_vs_11_kaggle-SMM-v0")
    generate_gif(smm_env)
    Image(filename='smm_env_replay.gif', format='png')

