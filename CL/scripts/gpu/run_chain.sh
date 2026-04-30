#!/usr/bin/env bash
# =============================================================================
#  Autonomous setup -> smoke -> queue chain. Designed to run unattended on
#  a freshly deployed RunPod pod. The user kicks this off once via nohup,
#  then is free to close the laptop and walk away.
#
#  Phases:
#    1. setup_runpod.sh  (~6 min)  -- system + python deps + COOM install
#    2. smoke_test.sh    (~10 min) -- 5k-step EWC pipeline check
#    3. queue_gamma_sweep.sh (~3-4 h) -- paper-budget reproduction queue
#
#  Status is written to /tmp/chain_status.txt with a timestamped line per
#  phase. On any non-zero return code from setup or smoke the chain
#  aborts; the queue phase is allowed to soft-fail per-run (its own
#  status file is logs/gpu/queue_status.txt).
#
#  Usage on the pod (run by hand the FIRST time):
#    cd /workspace/COOM
#    nohup bash CL/scripts/gpu/run_chain.sh > /tmp/chain.log 2>&1 & disown
#
#  Watch from the laptop:
#    ssh <pod> tail -f /tmp/chain_status.txt
# =============================================================================
set -uo pipefail
cd /workspace/COOM

STATUS=/tmp/chain_status.txt
: > "$STATUS"   # truncate

stamp() {
    echo "[$(date '+%F %T')] $*" | tee -a "$STATUS"
}

stamp "CHAIN START on host $(hostname)"

# Phase 1: setup. Skip if /workspace/venv already exists (resumable).
if [[ -d /workspace/venv && -f /workspace/venv/bin/activate ]]; then
    stamp "PHASE setup skipped (venv already present)"
else
    stamp "PHASE setup START"
    bash CL/scripts/gpu/setup_runpod.sh > /tmp/setup.log 2>&1
    rc=$?
    stamp "PHASE setup DONE rc=$rc"
    if [[ $rc -ne 0 ]]; then
        stamp "ABORT: setup failed; see /tmp/setup.log"
        exit 1
    fi
fi

# Activate venv for downstream phases
# shellcheck disable=SC1091
source /workspace/venv/bin/activate

# Phase 2: smoke test
stamp "PHASE smoke START"
bash CL/scripts/gpu/smoke_test.sh > /tmp/smoke.log 2>&1
rc=$?
stamp "PHASE smoke DONE rc=$rc"
if [[ $rc -ne 0 ]]; then
    stamp "ABORT: smoke failed; see /tmp/smoke.log"
    exit 1
fi

# Phase 3: production queue (gamma sweep at paper budget)
stamp "PHASE queue START"
bash CL/scripts/gpu/queue_gamma_sweep.sh
rc=$?
stamp "PHASE queue DONE rc=$rc"

stamp "CHAIN COMPLETE rc=$rc"
exit $rc
