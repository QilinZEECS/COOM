#!/usr/bin/env bash
# =============================================================================
#  Pull GPU pod logs back to the local laptop. Run on the LAPTOP, not the
#  pod. Replace POD_HOST and POD_PORT with the values RunPod gives you in
#  the Connect dialog (look for "TCP Port Mappings" -> 22:xxxxx).
#
#  Usage:
#    POD_HOST=...   POD_PORT=...  bash CL/scripts/gpu/sync_logs.sh
#
#  This rsyncs /workspace/COOM/logs/* on the pod into ./logs/gpu_sync/* on
#  the laptop, preserving the run timestamp directory tree so existing
#  analysis scripts (paper/scripts/make_figures.py) can read them with no
#  changes.
# =============================================================================
set -euo pipefail

: "${POD_HOST:?set POD_HOST=root@<host> to the pod's SSH host}"
: "${POD_PORT:?set POD_PORT=<ssh port> to the pod's SSH port}"

LOCAL_OUT=logs/gpu_sync
mkdir -p "${LOCAL_OUT}"

echo "[sync] pulling GPU logs from ${POD_HOST}:${POD_PORT} ..."
rsync -avh --progress \
    -e "ssh -p ${POD_PORT} -o StrictHostKeyChecking=no" \
    "${POD_HOST}:/workspace/COOM/logs/" \
    "${LOCAL_OUT}/"

echo "[sync] done. logs land in ${LOCAL_OUT}/"
echo "[sync] re-run paper/scripts/make_figures.py to incorporate them."
