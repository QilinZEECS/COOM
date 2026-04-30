#!/usr/bin/env bash
# =============================================================================
#  Paper-budget production queue. Reproduces the COOM baseline numbers on
#  CO4 at 200,000 steps per task with held-out evaluation enabled. Three
#  seeds per method gives a defensible inter-seed band without exhausting
#  the rental budget.
#
#  Wall-clock estimate on a single RTX 3090 (community spot tier):
#      ~25-40 min per (method, seed) run with --test enabled.
#      4 methods * 3 seeds = 12 runs -> 5-8 hours total
#      At $0.34/h spot the GPU bill comes in around $2-3.
#
#  Usage on the pod:
#    bash CL/scripts/gpu/queue_paper.sh
#
#  Re-running this script is safe: each run lives in its own group_id /
#  timestamp directory, so previous results are not overwritten.
# =============================================================================
set -uo pipefail

cd /workspace/COOM
mkdir -p logs/gpu

STATUS=logs/gpu/queue_status.txt
LOG=logs/gpu/queue_run.log
echo "[$(date '+%F %T')] START gpu_queue_paper" | tee -a "$STATUS"

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

STEPS=200000
SEQUENCE=CO4

# ---- four methods x three seeds -------------------------------------------
for SEED in 1 2 3; do
    run ft_seed${SEED}  python CL/run_cl.py \
        --sequence ${SEQUENCE} --seed ${SEED} \
        --steps_per_env ${STEPS} --gpu 0 \
        --group_id ft_${SEQUENCE,,}_${STEPS}_seed${SEED}

    run l2_seed${SEED}  python CL/run_cl.py \
        --sequence ${SEQUENCE} --seed ${SEED} \
        --steps_per_env ${STEPS} --gpu 0 \
        --cl_method l2 --cl_reg_coef 100000 \
        --group_id l2_${SEQUENCE,,}_${STEPS}_seed${SEED}

    run mas_seed${SEED} python CL/run_cl.py \
        --sequence ${SEQUENCE} --seed ${SEED} \
        --steps_per_env ${STEPS} --gpu 0 \
        --cl_method mas --cl_reg_coef 10000 \
        --group_id mas_${SEQUENCE,,}_${STEPS}_seed${SEED}

    run ewc_seed${SEED} python CL/run_cl.py \
        --sequence ${SEQUENCE} --seed ${SEED} \
        --steps_per_env ${STEPS} --gpu 0 \
        --cl_method ewc --cl_reg_coef 250 \
        --group_id ewc_${SEQUENCE,,}_${STEPS}_seed${SEED}
done

echo "[$(date '+%F %T')] QUEUE COMPLETE" | tee -a "$STATUS"
