#!/usr/bin/env bash
# =============================================================================
#  Bare-RunPod setup path (no Docker). Use this when you spin up a RunPod
#  pod from one of the stock CUDA-11.x Ubuntu 20.04 templates and want to
#  install everything inside the running pod.
#
#  Tested on RunPod template:  "Ubuntu 22.04 + CUDA 11.8" (community)
#  Wall-clock to first runnable: ~6 minutes.
#
#  Usage on the pod (after `git clone <your-repo> /workspace/COOM`):
#    cd /workspace/COOM && bash CL/scripts/gpu/setup_runpod.sh
# =============================================================================
set -euo pipefail

# Make tf 2.11 happy with whatever CUDA the pod ships. TF 2.11 was built for
# CUDA 11.2; on 11.8 hosts it works only because libcudart is forward
# compatible. If you hit "Could not load dynamic library 'libcudnn.so.8'"
# the pod is on the wrong CUDA major version -- pick a CUDA 11.x template,
# not 12.x.
echo "[1/5] sanity check: CUDA + driver"
nvidia-smi | head -15

echo "[2/5] system deps for ViZDoom"
sudo apt-get update -qq
sudo apt-get install -y --no-install-recommends \
    build-essential cmake git pkg-config \
    libboost-all-dev libsdl2-dev libfreetype-dev libgl1-mesa-dev \
    libopenal-dev nasm libjpeg-dev tar zlib1g-dev libbz2-dev \
    liblzma-dev wget rsync

echo "[3/5] python venv"
python3 -m venv /workspace/venv
# shellcheck disable=SC1091
source /workspace/venv/bin/activate
pip install --upgrade pip setuptools wheel

echo "[4/5] python deps"
pip install --no-cache-dir \
    "tensorflow==2.11" \
    "tensorflow-probability==0.19" \
    "numpy<2" \
    "pandas" "matplotlib" "seaborn" "scipy==1.11.4" \
    "gymnasium==0.28.1" "opencv-python" \
    "wandb"
pip install --no-cache-dir "vizdoom"

echo "[5/5] COOM in editable mode"
cd /workspace/COOM
pip install --no-cache-dir -e .

echo
echo "verifying TF can see the GPU..."
python - <<'PY'
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
print(f"GPUs visible to TF: {len(gpus)}")
for g in gpus:
    print(f"  - {g}")
assert gpus, "no GPU visible to TF; abort before launching expensive runs"
PY

echo
echo "setup OK. Activate the venv in future shells with:"
echo "    source /workspace/venv/bin/activate"
echo
echo "next: run a smoke test with"
echo "    bash CL/scripts/gpu/smoke_test.sh"
