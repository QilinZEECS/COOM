#!/usr/bin/env bash
# =============================================================================
#  Phase-2 extension: 6 paper-budget jobs run in *batches of 2* (not all
#  6 at once). The earlier all-6-parallel attempt OOM'd the pod because
#  each TF process holds ~5-10 GB RAM in addition to GPU memory; on a
#  community-tier RTX 3090 pod with ~32 GB RAM, only 2 concurrent
#  processes fit safely.
#
#  Wall-clock estimate: 3 batches x ~8 h = ~24 h on RTX 3090 spot.
#  Spot cost at $0.34/h: ~$8.
#
#  Runs (single seed per run unless noted):
#    Batch 1 (highest priority):
#      - EWC seed 2 (multi-seed band on the paper-budget EWC reference)
#      - Online EWC gamma=0.95 seed 2 (multi-seed on the *winning* gamma)
#    Batch 2:
#      - Online EWC gamma=0.70 seed 1 (fills gap between 0.50 and 0.95)
#      - Online EWC gamma=0.30 seed 1 (aggressive Fisher decay)
#    Batch 3:
#      - FT seed 1 (paper-budget control)
#      - Online EWC gamma=0.85 seed 1 (intermediate)
#
#  Usage on the pod:
#    cd /workspace/COOM
#    nohup bash CL/scripts/gpu/queue_extension_parallel.sh \
#        > /tmp/extension.log 2>&1 & disown
#
#  Watch:
#    tail -f /workspace/COOM/logs/gpu/extension_status.txt
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

# Run a single python invocation in the background; return its PID and
# its per-run log path. Caller waits on PID and inspects log path.
spawn() {
    local tag="$1"; shift
    local outlog="logs/gpu/${tag}.log"
    "$@" > "$outlog" 2>&1 &
    local pid=$!
    echo "[$(date '+%F %T')] LAUNCH ${tag} pid=${pid} :: $*" | tee -a "$STATUS"
    echo "$pid"
}

# Wait for all PIDs, recording each completion. Tolerates `wait` returning
# nonzero on already-orphaned PIDs.
wait_batch() {
    local tag_label="$1"; shift
    for pid in "$@"; do
        wait "$pid" 2>/dev/null
        local rc=$?
        echo "[$(date '+%F %T')] WAIT pid=${pid} rc=${rc} (${tag_label})" \
            | tee -a "$STATUS"
    done
    echo "[$(date '+%F %T')] BATCH COMPLETE: ${tag_label}" | tee -a "$STATUS"
}

STEPS=200000

# ---- Batch 1: multi-seed validation of the headline numbers ---------------
echo "[$(date '+%F %T')] BATCH 1 START (EWC s2 + Online g=0.95 s2)" | tee -a "$STATUS"
P1=$(spawn ewc_co4_200k_seed2 \
    python CL/run_cl.py --sequence CO4 --seed 2 \
        --steps_per_env ${STEPS} --gpu 0 \
        --cl_method ewc --cl_reg_coef 250 \
        --group_id ewc_co4_200k_seed2)
P2=$(spawn ewc_online_g095_co4_200k_seed2 \
    python CL/run_cl.py --sequence CO4 --seed 2 \
        --steps_per_env ${STEPS} --gpu 0 \
        --cl_method ewc_online --ewc_gamma 0.95 --cl_reg_coef 250 \
        --group_id ewc_online_g095_co4_200k_seed2)
wait_batch "batch1" "$P1" "$P2"

# ---- Batch 2: gamma sweep ------------------------------------------------
echo "[$(date '+%F %T')] BATCH 2 START (gamma=0.70 + gamma=0.30)" | tee -a "$STATUS"
P3=$(spawn ewc_online_g070_co4_200k_seed1 \
    python CL/run_cl.py --sequence CO4 --seed 1 \
        --steps_per_env ${STEPS} --gpu 0 \
        --cl_method ewc_online --ewc_gamma 0.70 --cl_reg_coef 250 \
        --group_id ewc_online_g070_co4_200k_seed1)
P4=$(spawn ewc_online_g030_co4_200k_seed1 \
    python CL/run_cl.py --sequence CO4 --seed 1 \
        --steps_per_env ${STEPS} --gpu 0 \
        --cl_method ewc_online --ewc_gamma 0.30 --cl_reg_coef 250 \
        --group_id ewc_online_g030_co4_200k_seed1)
wait_batch "batch2" "$P3" "$P4"

# ---- Batch 3: FT control + gamma=0.85 ------------------------------------
echo "[$(date '+%F %T')] BATCH 3 START (FT s1 + gamma=0.85)" | tee -a "$STATUS"
P5=$(spawn ft_co4_200k_seed1 \
    python CL/run_cl.py --sequence CO4 --seed 1 \
        --steps_per_env ${STEPS} --gpu 0 \
        --group_id ft_co4_200k_seed1)
P6=$(spawn ewc_online_g085_co4_200k_seed1 \
    python CL/run_cl.py --sequence CO4 --seed 1 \
        --steps_per_env ${STEPS} --gpu 0 \
        --cl_method ewc_online --ewc_gamma 0.85 --cl_reg_coef 250 \
        --group_id ewc_online_g085_co4_200k_seed1)
wait_batch "batch3" "$P5" "$P6"

echo "[$(date '+%F %T')] EXTENSION QUEUE COMPLETE" | tee -a "$STATUS"
