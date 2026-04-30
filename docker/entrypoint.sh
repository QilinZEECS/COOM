#!/usr/bin/env bash
# =============================================================================
#  Container entrypoint. Verifies the GPU is visible to TensorFlow before
#  handing control to the user's command. If you see "GPU not available" the
#  container was launched without --gpus all (Docker) or without a GPU
#  selected (RunPod custom template).
# =============================================================================
set -e

echo "============================================================"
echo "COOM GPU container"
echo "  CUDA visible devices : ${NVIDIA_VISIBLE_DEVICES:-(unset)}"
echo "  Python              : $(python --version 2>&1)"
echo "  TensorFlow GPU check:"
python - <<'PY'
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
print(f"  -> {len(gpus)} GPU(s) found: {gpus}")
if not gpus:
    print("  -> WARNING: TF cannot see a GPU. Runs will fall back to CPU.")
PY
echo "============================================================"

exec "$@"
