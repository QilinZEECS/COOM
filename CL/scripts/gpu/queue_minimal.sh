#!/usr/bin/env bash
# =============================================================================
#  Minimal-budget paper-reproduction queue. EWC only, three seeds, CO4,
#  200,000 steps per task with held-out evaluation. This is the cheapest
#  possible "real" reproduction of the published EWC number; any conclusion
#  about whether our CO4 pipeline matches the COOM paper depends on this
#  one queue.
#
#  Wall-clock estimate on a single RTX 3090 (community spot tier):
#      ~25-40 min per seed run.
#      3 seeds -> 75-120 minutes total.
#      At $0.34/h spot the GPU bill is roughly $0.50-0.70.
#
#  Usage on the pod:
#    bash CL/scripts/gpu/queue_minimal.sh
# =============================================================================
set -uo pipefail

cd /workspace/COOM
mkdir -p logs/gpu

STATUS=logs/gpu/queue_status.txt
LOG=logs/gpu/queue_run.log
echo "[$(date '+%F %T')] START gpu_queue_minimal" | tee -a "$STATUS"

run() {
    local tag="$1"; shift
    local t0=$(date +%s)
    echo "[$(date '+%F %T')] START ${tag} :: $*" | tee -a "$STATUS"
    "$@" >> "$LOG" 2>&1
    local rc=$?
    local t1=$(date +%s)
    echo "[$(date '+%F %T')] DONE  ${tag} :: rc=${rc} duration=$((t1 - t0))s" \
        | tee -a "$STATUS"
    return $rc
}

for SEED in 1 2 3; do
    run ewc_seed${SEED} python CL/run_cl.py \
        --sequence CO4 --seed ${SEED} \
        --steps_per_env 200000 --gpu 0 \
        --cl_method ewc --cl_reg_coef 250 \
        --group_id ewc_co4_200k_seed${SEED}
done

echo "[$(date '+%F %T')] QUEUE COMPLETE" | tee -a "$STATUS"
