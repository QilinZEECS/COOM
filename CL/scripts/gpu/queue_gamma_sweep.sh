#!/usr/bin/env bash
# =============================================================================
#  Recommended GPU queue (Option B): Online EWC gamma sweep at the
#  paper-budget of 200,000 steps per task, plus an original-EWC reference
#  run for the head-to-head. Three runs, single seed each.
#
#  This queue is designed to answer two questions in one batch:
#    (1) Does the pipeline reproduce EWC ~ 0.54 at the published budget?
#    (2) At what gamma does Online EWC start to beat (or match) EWC?
#
#  Runs:
#    - ewc      seed 1 200k        (baseline; expected ~ 0.54 paper avg)
#    - ewc_online gamma=0.50 seed 1 200k   (rapid-decay; weights F_0 at 0.25)
#    - ewc_online gamma=0.95 seed 1 200k   (matches our CPU 20k run; direct)
#
#  Wall-clock estimate on a single RTX 3090 spot tier:
#      ~50-80 min per run, depending on env stepping throughput
#      3 runs total -> 2.5-4 hours
#  Spot cost at $0.34/h: ~$1-1.50
#
#  Usage on the pod:
#    cd /workspace/COOM && nohup bash CL/scripts/gpu/queue_gamma_sweep.sh \
#        > gpu_queue.out 2>&1 & disown
#  Watch:
#    tail -f /workspace/COOM/logs/gpu/queue_status.txt
# =============================================================================
set -uo pipefail

cd /workspace/COOM
mkdir -p logs/gpu

STATUS=logs/gpu/queue_status.txt
LOG=logs/gpu/queue_run.log
echo "[$(date '+%F %T')] START gpu_queue_gamma_sweep" | tee -a "$STATUS"

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

# Reference: original EWC at the published budget. The headline number to
# compare against the COOM paper's EWC = 0.54.
run ewc_seed1_200k python CL/run_cl.py \
    --sequence CO4 --seed 1 \
    --steps_per_env ${STEPS} --gpu 0 \
    --cl_method ewc --cl_reg_coef 250 \
    --group_id ewc_co4_200k_seed1

# Improved variant: gamma = 0.50 weights the very first (noisiest) Fisher
# at gamma^2 = 0.25 by the time the third anchor is added, which directly
# tests the "Fisher on noise is preserved by gamma=0.95" hypothesis from
# the dissertation's Outcome C analysis.
run ewconline_g050_seed1_200k python CL/run_cl.py \
    --sequence CO4 --seed 1 \
    --steps_per_env ${STEPS} --gpu 0 \
    --cl_method ewc_online --ewc_gamma 0.50 --cl_reg_coef 250 \
    --group_id ewc_online_g050_co4_200k_seed1

# Same gamma = 0.95 as the CPU 20k run, now at the paper budget. If the
# Outcome C result was driven by under-converged source tasks rather than
# by gamma itself, this run should look much closer to (or beat) the EWC
# reference above.
run ewconline_g095_seed1_200k python CL/run_cl.py \
    --sequence CO4 --seed 1 \
    --steps_per_env ${STEPS} --gpu 0 \
    --cl_method ewc_online --ewc_gamma 0.95 --cl_reg_coef 250 \
    --group_id ewc_online_g095_co4_200k_seed1

echo "[$(date '+%F %T')] QUEUE COMPLETE" | tee -a "$STATUS"
