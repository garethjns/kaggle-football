import glob
import os
import tempfile
from typing import Tuple

import gym
import imageio
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def plot_smm_obs(obs: np.ndarray, expected_min: int = 0, expected_max: int = 255) -> Tuple[plt.Figure, plt.Axes]:
    """Plot the 4 "frames" in a single observation."""

    cmap = plt.cm.gray
    norm = plt.Normalize(expected_min, expected_max)

    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))
    for i, (ax, title) in enumerate(zip(axs.flatten(),
                                        ['left team', 'right team',
                                         'ball', 'active player'])):
        ax.imshow(cmap(norm(obs[..., i])), animated=True)
        ax.set_title(title)
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])

    return fig, ax


def generate_gif(env: gym.Env, n_steps: int = 20, suffix: str = "smm_env_", **kwargs) -> None:
    """Plot a few steps of an env and generate a .gif."""

    tmp_dir = tempfile.TemporaryDirectory()
    _ = env.reset()

    for s in tqdm(range(20)):
        obs, _, _, _ = env.step(n_steps)
        fig, ax = plot_smm_obs(obs, **kwargs)
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
