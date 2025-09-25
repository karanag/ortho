#!/bin/bash
set -e

echo ">>> Updating apt..."
apt-get update -y && apt-get upgrade -y

echo ">>> Installing base tools..."
apt-get install -y git cmake build-essential pkg-config \
    libboost-all-dev libeigen3-dev libsuitesparse-dev \
    libfreeimage-dev libgoogle-glog-dev libgflags-dev \
    libglew-dev qtbase5-dev libqt5opengl5-dev libcgal-dev \
    libsqlite3-dev libceres-dev libmetis-dev wget unzip \
    python3-dev python3-pip python3-venv

echo ">>> Installing Python deps..."
# optional: create venv
# python3 -m venv colmap_env && source colmap_env/bin/activate
pip install --upgrade pip
pip install pycolmap opencv-contrib-python

echo ">>> Cloning COLMAP..."
cd /
if [ -d "colmap" ]; then rm -rf colmap; fi
git clone https://github.com/colmap/colmap.git
cd colmap
git fetch --all
git checkout 81ea1784   # pinned stable commit

echo ">>> Building COLMAP..."
rm -rf build && mkdir build && cd build
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DCUDA_ENABLED=ON \
  -DGUI_ENABLED=OFF \
  -DCMAKE_CUDA_ARCHITECTURES=80

make -j"$(nproc)"
make install

echo ">>> Done! Test with: colmap --help"
