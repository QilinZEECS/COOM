#!/usr/bin/env bash
# =============================================================================
#  GPU smoke test. ~10 minutes on an RTX 3090. The point is to verify that
#  the full pipeline (ViZDoom + TF GPU + multi-task wrapping + log writer)
#  runs end-to-end before you commit to a paper-budget queue. If this run
#  finishes with a non-empty progress.tsv and rc=0, the long queue is safe
#  to start.
#
#  Configuration deliberately minimal: 2 tasks of CO4, 5000 steps each,
#  EWC. If it works, the same code path will work for the full queue.
#
#  Usage on the pod:
#    bash CL/scripts/gpu/smoke_test.sh
# =============================================================================
set -euo pipefail

cd /workspace/COOM

mkdir -p logs/_smoke
GROUP="_smoke_ewc_5k"

echo "[smoke] launching ${GROUP} ..."
PYTHONPATH=. python CL/run_cl.py \
    --sequence CO4 \
    --seed 1 \
    --steps_per_env 5000 \
    --no_test \
    --cl_method ewc \
    --cl_reg_coef 250 \
    --gpu 0 \
    --group_id "${GROUP}" \
    --start_from 0 \
    2>&1 | tee logs/_smoke/run.log

run_dir=$(ls -dt logs/${GROUP}/2* | head -1)
rows=$(wc -l < "${run_dir}/progress.tsv")
echo "[smoke] log dir: ${run_dir}"
echo "[smoke] progress.tsv rows: ${rows}"

if [ "${rows}" -lt 10 ]; then
    echo "[smoke] FAILED: progress.tsv has fewer than 10 rows"
    exit 1
fi

echo "[smoke] OK -- pipeline works on GPU. Next: bash CL/scripts/gpu/queue_paper.sh"
