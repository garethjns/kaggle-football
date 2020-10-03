#!/bin/bash

apt-get update
apt-get install -y libsdl2-gfx-dev libsdl2-ttf-dev
pip3 install tensorflow==2.2
pip3 install tensorflow_probability==0.9.0

git clone -b v2.5 https://github.com/google-research/football.git
mkdir -p football/third_party/gfootball_engine/lib

wget https://storage.googleapis.com/gfootball/prebuilt_gameplayfootball_v2.5.so -O football/third_party/gfootball_engine/lib/prebuilt_gameplayfootball.so
cd football && GFOOTBALL_USE_PREBUILT_SO=1 pip3 install .
