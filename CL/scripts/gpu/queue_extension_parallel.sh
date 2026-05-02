#!/usr/bin/env bash
# =============================================================================
#  Parallel extension queue (Phase 2). Runs 6 paper-budget jobs
#  concurrently on a single GPU + multi-core pod, exploiting the fact that
#  ViZDoom is CPU-bound and barely uses the GPU. Each job sets
#  TF_FORCE_GPU_ALLOW_GROWTH=true so multiple TF processes share the GPU
#  without each grabbing the full 24 GB. RunPod community pods have 256
#  CPU cores; six concurrent runs uses ~6 cores and ~18 GB GPU.
#
#  Wall-clock estimate: 7-9 h on RTX 3090 spot (same per-run wall time as
#  serial, run in parallel rather than sequentially).
#  Spot cost at $0.34/h: ~$2.5.
#
#  Runs (single seed per run unless noted):
#    - EWC seed 2 200k                   (multi-seed band on EWC paper-budget)
#    - Fine-Tuning seed 1 200k           (paper-budget control)
#    - Online EWC gamma=0.30 seed 1 200k (aggressive Fisher decay)
#    - Online EWC gamma=0.70 seed 1 200k (intermediate)
#    - Online EWC gamma=0.85 seed 1 200k (intermediate)
#    - Online EWC gamma=0.95 seed 1 200k (matches the gamma we picked
#                                         from Schwarz 2018, paper budget)
#
#  Usage on the pod (after the gamma sweep queue ends, or alongside if
#  GPU memory is free):
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

# Activate the venv that setup_runpod.sh provisioned. Pod restarts wipe
# /tmp and reset the shell environment, so we must re-source the venv
# every time this script is invoked rather than rely on inherited PATH.
if [[ -f /workspace/venv/bin/activate ]]; then
    # shellcheck disable=SC1091
    source /workspace/venv/bin/activate
else
    echo "ERROR: /workspace/venv missing. Re-run setup_runpod.sh first." >&2
    exit 1
fi

export PYTHONPATH="/workspace/COOM:${PYTHONPATH:-}"
# Multiple TF processes will share the same GPU if each opts into memory
# growth. Without this, TF defaults to allocating 95% of GPU memory at
# startup which would prevent any second TF process from initialising.
export TF_FORCE_GPU_ALLOW_GROWTH=true

STATUS=logs/gpu/extension_status.txt
LOG=logs/gpu/extension_run.log
echo "[$(date '+%F %T')] START parallel extension batch" | tee -a "$STATUS"

STEPS=200000

# Launch every run in the background. Each writes its own per-run log so
# stdouts don't interleave, and TF_FORCE_GPU_ALLOW_GROWTH is exported
# already so all children inherit it.
launch() {
    local tag="$1"; shift
    local outlog="logs/gpu/${tag}.log"
    echo "[$(date '+%F %T')] LAUNCH ${tag} :: $*" | tee -a "$STATUS"
    "$@" > "$outlog" 2>&1 &
    echo $!
}

declare -A PIDS

PIDS[ewc_s2]=$(launch ewc_co4_200k_seed2 \
    python CL/run_cl.py --sequence CO4 --seed 2 \
        --steps_per_env ${STEPS} --gpu 0 \
        --cl_method ewc --cl_reg_coef 250 \
        --group_id ewc_co4_200k_seed2)

PIDS[ft_s1]=$(launch ft_co4_200k_seed1 \
    python CL/run_cl.py --sequence CO4 --seed 1 \
        --steps_per_env ${STEPS} --gpu 0 \
        --group_id ft_co4_200k_seed1)

PIDS[ewconline_g030]=$(launch ewc_online_g030_co4_200k_seed1 \
    python CL/run_cl.py --sequence CO4 --seed 1 \
        --steps_per_env ${STEPS} --gpu 0 \
        --cl_method ewc_online --ewc_gamma 0.30 --cl_reg_coef 250 \
        --group_id ewc_online_g030_co4_200k_seed1)

PIDS[ewconline_g070]=$(launch ewc_online_g070_co4_200k_seed1 \
    python CL/run_cl.py --sequence CO4 --seed 1 \
        --steps_per_env ${STEPS} --gpu 0 \
        --cl_method ewc_online --ewc_gamma 0.70 --cl_reg_coef 250 \
        --group_id ewc_online_g070_co4_200k_seed1)

PIDS[ewconline_g085]=$(launch ewc_online_g085_co4_200k_seed1 \
    python CL/run_cl.py --sequence CO4 --seed 1 \
        --steps_per_env ${STEPS} --gpu 0 \
        --cl_method ewc_online --ewc_gamma 0.85 --cl_reg_coef 250 \
        --group_id ewc_online_g085_co4_200k_seed1)

# Multi-seed Online EWC at the most-promising gamma (the gamma=0.5
# paper-budget run hinted at Outcome A on the held-out metric, so a
# second seed is the natural extension).
PIDS[ewconline_g050_s2]=$(launch ewc_online_g050_co4_200k_seed2 \
    python CL/run_cl.py --sequence CO4 --seed 2 \
        --steps_per_env ${STEPS} --gpu 0 \
        --cl_method ewc_online --ewc_gamma 0.50 --cl_reg_coef 250 \
        --group_id ewc_online_g050_co4_200k_seed2)

echo "[$(date '+%F %T')] All 6 runs launched: ${PIDS[*]}" | tee -a "$STATUS"

# Wait for each, recording per-run completion in the status file. We
# don't pipefail here because individual run failures should not abort
# the wait on the others.
for tag in "${!PIDS[@]}"; do
    pid="${PIDS[$tag]}"
    wait "$pid"
    rc=$?
    echo "[$(date '+%F %T')] DONE ${tag} :: rc=${rc}" | tee -a "$STATUS"
done

echo "[$(date '+%F %T')] EXTENSION QUEUE COMPLETE" | tee -a "$STATUS"
