#!/usr/bin/env bash
# =============================================================================
#  Phase-2 extension: 6 paper-budget jobs run in BATCHES OF 2 (not all 6
#  at once). The earlier all-6-parallel attempt OOM'd a ~32 GB pod
#  because each TF process holds ~5-10 GB RAM in addition to GPU memory;
#  even 6 with TF_FORCE_GPU_ALLOW_GROWTH=true is too many on a community
#  RTX 3090 host. 2 concurrent processes fit safely.
#
#  Wall-clock estimate: 3 batches x ~8 h = ~24 h on RTX 3090 spot.
#  Spot cost at $0.34/h: ~$8.
#
#  Runs (single seed per run unless noted):
#    Batch 1 (highest priority -- multi-seed validation of headline):
#      - EWC seed 2 (multi-seed band on the paper-budget EWC reference)
#      - Online EWC gamma=0.95 seed 2 (multi-seed on the *winning* gamma)
#    Batch 2 (gamma sweep):
#      - Online EWC gamma=0.70 seed 1 (fills gap between 0.50 and 0.95)
#      - Online EWC gamma=0.30 seed 1 (aggressive Fisher decay)
#    Batch 3 (control + intermediate):
#      - FT seed 1 (paper-budget control)
#      - Online EWC gamma=0.85 seed 1
#
#  Implementation note: spawn was previously a function returning $! via
#  command substitution. That ran the `&` inside a subshell, orphaning
#  the child as soon as the subshell exited; main-shell `wait` then
#  returned immediately and ALL six runs launched at once. This version
#  launches each python invocation directly in the main shell so that
#  $! and `wait` are bound correctly.
#
#  Usage on the pod:
#    cd /workspace/COOM
#    nohup bash CL/scripts/gpu/queue_extension_parallel.sh \
#        > /tmp/extension.log 2>&1 & disown
# =============================================================================
set -uo pipefail

cd /workspace/COOM
mkdir -p logs/gpu

if [[ -f /workspace/venv/bin/activate ]]; then
    # shellcheck disable=SC1091
    source /workspace/venv/bin/activate
else
    echo "ERROR: /workspace/venv missing. Re-run setup_runpod.sh first." >&2
    exit 1
fi

export PYTHONPATH="/workspace/COOM:${PYTHONPATH:-}"
export TF_FORCE_GPU_ALLOW_GROWTH=true

STATUS=logs/gpu/extension_status.txt
echo "[$(date '+%F %T')] START extension batched x2" | tee -a "$STATUS"

stamp() {
    echo "[$(date '+%F %T')] $*" | tee -a "$STATUS"
}

run_batch() {
    # Args: <batch label> <pid1_var=cmd1...> <pid2_var=cmd2...>
    # Both children are launched IN THE MAIN SHELL of this script so
    # `wait` is happy.
    local label="$1"; shift
    stamp "BATCH ${label} START"
}

STEPS=200000

# ---- Batch 1 -------------------------------------------------------------
stamp "BATCH 1 START (EWC s2 + Online g=0.95 s2)"

python CL/run_cl.py --sequence CO4 --seed 2 \
    --steps_per_env ${STEPS} --gpu 0 \
    --cl_method ewc --cl_reg_coef 250 \
    --group_id ewc_co4_200k_seed2 \
    > logs/gpu/ewc_co4_200k_seed2.log 2>&1 &
P1=$!
stamp "LAUNCH ewc_co4_200k_seed2 pid=${P1}"

python CL/run_cl.py --sequence CO4 --seed 2 \
    --steps_per_env ${STEPS} --gpu 0 \
    --cl_method ewc_online --ewc_gamma 0.95 --cl_reg_coef 250 \
    --group_id ewc_online_g095_co4_200k_seed2 \
    > logs/gpu/ewc_online_g095_co4_200k_seed2.log 2>&1 &
P2=$!
stamp "LAUNCH ewc_online_g095_co4_200k_seed2 pid=${P2}"

wait $P1; rc1=$?
stamp "DONE pid=${P1} rc=${rc1} ewc_co4_200k_seed2"
wait $P2; rc2=$?
stamp "DONE pid=${P2} rc=${rc2} ewc_online_g095_co4_200k_seed2"
stamp "BATCH 1 COMPLETE"

# ---- Batch 2 -------------------------------------------------------------
stamp "BATCH 2 START (gamma=0.70 + gamma=0.30)"

python CL/run_cl.py --sequence CO4 --seed 1 \
    --steps_per_env ${STEPS} --gpu 0 \
    --cl_method ewc_online --ewc_gamma 0.70 --cl_reg_coef 250 \
    --group_id ewc_online_g070_co4_200k_seed1 \
    > logs/gpu/ewc_online_g070_co4_200k_seed1.log 2>&1 &
P3=$!
stamp "LAUNCH ewc_online_g070_co4_200k_seed1 pid=${P3}"

python CL/run_cl.py --sequence CO4 --seed 1 \
    --steps_per_env ${STEPS} --gpu 0 \
    --cl_method ewc_online --ewc_gamma 0.30 --cl_reg_coef 250 \
    --group_id ewc_online_g030_co4_200k_seed1 \
    > logs/gpu/ewc_online_g030_co4_200k_seed1.log 2>&1 &
P4=$!
stamp "LAUNCH ewc_online_g030_co4_200k_seed1 pid=${P4}"

wait $P3; rc3=$?
stamp "DONE pid=${P3} rc=${rc3} ewc_online_g070_co4_200k_seed1"
wait $P4; rc4=$?
stamp "DONE pid=${P4} rc=${rc4} ewc_online_g030_co4_200k_seed1"
stamp "BATCH 2 COMPLETE"

# ---- Batch 3 -------------------------------------------------------------
stamp "BATCH 3 START (FT s1 + gamma=0.85)"

python CL/run_cl.py --sequence CO4 --seed 1 \
    --steps_per_env ${STEPS} --gpu 0 \
    --group_id ft_co4_200k_seed1 \
    > logs/gpu/ft_co4_200k_seed1.log 2>&1 &
P5=$!
stamp "LAUNCH ft_co4_200k_seed1 pid=${P5}"

python CL/run_cl.py --sequence CO4 --seed 1 \
    --steps_per_env ${STEPS} --gpu 0 \
    --cl_method ewc_online --ewc_gamma 0.85 --cl_reg_coef 250 \
    --group_id ewc_online_g085_co4_200k_seed1 \
    > logs/gpu/ewc_online_g085_co4_200k_seed1.log 2>&1 &
P6=$!
stamp "LAUNCH ewc_online_g085_co4_200k_seed1 pid=${P6}"

wait $P5; rc5=$?
stamp "DONE pid=${P5} rc=${rc5} ft_co4_200k_seed1"
wait $P6; rc6=$?
stamp "DONE pid=${P6} rc=${rc6} ewc_online_g085_co4_200k_seed1"
stamp "BATCH 3 COMPLETE"

stamp "EXTENSION QUEUE COMPLETE"
