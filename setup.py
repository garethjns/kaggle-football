import setuptools

from kaggle_football import __version__

setuptools.setup(
    name="kaggle-football",
    version=__version__,
    author="Gareth Jones",
    author_email="garethgithub@gmail.com",
    description="",
    long_description='',
    long_description_content_type="text/markdown",
    url="https://github.com/garethjns/reinforcement-learning-keras",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent"],
    python_requires='>=3.6',
    install_requires=["gfootball", "kaggle-environments", "kaggle", "IPython", "numpy", "imageio", "tqdm", "pandas",
                      "joblib", "hdf5plugin", "h5py", "tables"])
