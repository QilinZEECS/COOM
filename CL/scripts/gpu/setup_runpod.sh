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
# `head -15` would close the pipe early and trigger SIGPIPE on nvidia-smi
# under `set -o pipefail`; emit the device list directly instead.
nvidia-smi -L
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true

echo "[2/5] system deps for ViZDoom"
# RunPod containers run as root by default, in which case `sudo` is not
# installed; fall back to plain apt-get when we already have root.
if [[ $EUID -eq 0 ]]; then SUDO=""; else SUDO="sudo"; fi
$SUDO apt-get update -qq
$SUDO apt-get install -y --no-install-recommends \
    build-essential cmake git pkg-config \
    libboost-all-dev libsdl2-dev libfreetype-dev libgl1-mesa-dev \
    libopenal-dev nasm libjpeg-dev tar zlib1g-dev libbz2-dev \
    liblzma-dev wget rsync \
    python3-venv python3-dev

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
# NVIDIA libs that TF 2.11 dlopens at runtime. Most RunPod templates ship
# the NVIDIA driver (libnvidia-ml, libcuda) but NOT cuDNN or cuBLAS, so
# TF falls back to CPU. Pulling these wheels into the venv puts the .so
# files in a predictable place that we add to LD_LIBRARY_PATH below.
pip install --no-cache-dir \
    "nvidia-cudnn-cu11==8.6.0.163" \
    "nvidia-cublas-cu11==11.10.3.66" \
    "nvidia-cuda-runtime-cu11==11.7.99" \
    "nvidia-cuda-cupti-cu11==11.7.101" \
    "nvidia-cufft-cu11==10.9.0.58" \
    "nvidia-curand-cu11==10.2.10.91" \
    "nvidia-cusolver-cu11==11.4.0.1" \
    "nvidia-cusparse-cu11==11.7.4.91" \
    "nvidia-nccl-cu11==2.14.3"
pip install --no-cache-dir "vizdoom"

echo "[5/5] COOM in editable mode"
cd /workspace/COOM
pip install --no-cache-dir -e .

# Wire NVIDIA shared libs into LD_LIBRARY_PATH so TF can dlopen them.
# Each `nvidia-*-cu11` wheel installs into site-packages/nvidia/<pkg>/lib;
# we glob those dirs at activation time.
NVIDIA_LIB_DIRS=$(python -c "
import os, glob
base = os.path.join(os.environ['VIRTUAL_ENV'],
                    'lib', 'python3.10', 'site-packages', 'nvidia')
print(':'.join(sorted(glob.glob(os.path.join(base, '*', 'lib')))))
")
ACT="/workspace/venv/bin/activate"
if ! grep -q "NVIDIA_LIB_DIRS" "$ACT"; then
    cat <<EOSH >> "$ACT"

# --- COOM: NVIDIA shared-lib path for TF 2.11 -----------------------------
export NVIDIA_LIB_DIRS="${NVIDIA_LIB_DIRS}"
export LD_LIBRARY_PATH="\${NVIDIA_LIB_DIRS}:\${LD_LIBRARY_PATH:-}"
EOSH
fi
# Apply to the current shell as well (we're still inside the same setup
# invocation; later scripts will pick it up via venv activate).
export NVIDIA_LIB_DIRS
export LD_LIBRARY_PATH="${NVIDIA_LIB_DIRS}:${LD_LIBRARY_PATH:-}"

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
