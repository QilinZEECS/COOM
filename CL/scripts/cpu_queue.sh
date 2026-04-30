#!/usr/bin/env bash
# Sequential CPU experiment queue for FYP CL comparison.
# Each run is CO4, 20k steps/task, seed 1 (or labelled), --no_test, ~2h on Apple Silicon CPU.
# Status is written to CL/logs/queue_status.txt after each run.

set -u
cd "$(dirname "$0")/../.."   # → project root (parent of CL/)

source /opt/homebrew/Caskroom/miniforge/base/etc/profile.d/conda.sh
conda activate coom

# Project root must be on PYTHONPATH so `from CL.methods...` works when running CL/run_cl.py.
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

STATUS=CL/logs/queue_status.txt
mkdir -p CL/logs
: > "$STATUS"

log() { echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*" | tee -a "$STATUS"; }

run() {
  local tag="$1"; shift
  log "START $tag :: $*"
  local t0=$(date +%s)
  "$@"
  local rc=$?
  local t1=$(date +%s)
  local dur=$((t1 - t0))
  if [ $rc -eq 0 ]; then
    log "DONE  $tag :: rc=0  duration=${dur}s"
  else
    log "FAIL  $tag :: rc=$rc duration=${dur}s"
  fi
  return $rc
}

# 1. Fine-Tuning baseline (no CL method)
run ft_seed1 python CL/run_cl.py --sequence CO4 --seed 1 --steps_per_env 20000 --no_test \
  --group_id ft_co4_20k_seed1

# 2. L2 baseline
run l2_seed1 python CL/run_cl.py --sequence CO4 --seed 1 --steps_per_env 20000 --no_test \
  --cl_method l2 --cl_reg_coef 100000 --group_id l2_co4_20k_seed1

# 3. MAS baseline
run mas_seed1 python CL/run_cl.py --sequence CO4 --seed 1 --steps_per_env 20000 --no_test \
  --cl_method mas --cl_reg_coef 10000 --group_id mas_co4_20k_seed1

# 4. EWC seed 2 (variance)
run ewc_seed2 python CL/run_cl.py --sequence CO4 --seed 2 --steps_per_env 20000 --no_test \
  --cl_method ewc --cl_reg_coef 250 --group_id ewc_co4_20k_seed2

# 5. EWC seed 3 (variance)
run ewc_seed3 python CL/run_cl.py --sequence CO4 --seed 3 --steps_per_env 20000 --no_test \
  --cl_method ewc --cl_reg_coef 250 --group_id ewc_co4_20k_seed3

log "QUEUE COMPLETE"
