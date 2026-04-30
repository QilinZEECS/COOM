# GPU Reproduction Runbook

This document is the step-by-step procedure for reproducing the COOM
paper baseline numbers on a rented GPU. Two paths are documented:

1. **Bare RunPod path (recommended for first attempt).** No Docker. You
   spin up a stock RunPod template, clone the repo, run a setup script,
   then launch the experiment queue. ~20 min from "credit card entered"
   to "first run launched".

2. **Docker path (recommended once the runs are stable).** You build the
   image locally, push to Docker Hub, and point a RunPod custom template
   at it. Useful if you intend to reproduce repeatedly or if you want
   the image as supplementary material with the dissertation.

The compute target is **CO4 / 200,000 steps per task / 3 seeds** for
EWC, with optional extension to FT, L2 and MAS.

---

## 0. What you should have before you start

- A RunPod account with $20 of credit (real cost will be $5--$15).
- An SSH client on your laptop (macOS Terminal is fine).
- This repo pushed to a Git host the pod can `git clone` from. If your
  fork is private, use a GitHub PAT or deploy key on the pod.
- A clear answer to the question: \"do I want **EWC only** or **all four
  methods**?\" The minimal queue takes ~2 hours, the full queue ~6--8.

---

## 1. RunPod -- spin up the pod

1. Go to <https://www.runpod.io/console/pods>.
2. Click **+ Deploy**.
3. **GPU type**: pick `RTX 3090` from the *Community Cloud* tab. The spot
   price is ~$0.34/h. RTX 4090 is faster but $0.60+/h --- not worth it
   for this workload because ViZDoom is CPU-bound, not GPU-bound.
4. **Template**: pick the *RunPod TensorFlow* template (it ships
   Ubuntu 20.04 + CUDA 11.8). Despite the name we install our own TF; we
   only need the OS+CUDA layer.
5. **Volume**: the default 50 GB is plenty.
6. **Custom start command**: leave blank.
7. Click **Deploy**. Wait ~60 s for the pod to enter the *Running* state.

Note the **SSH details** in the *Connect* dialog: there is a host
(typically `root@ssh.runpod.io`) and a port number per pod. You need
both.

---

## 2. Bare-pod setup (~6 minutes)

SSH into the pod:

```bash
ssh -p <POD_PORT> root@ssh.runpod.io
```

Inside the pod:

```bash
mkdir -p /workspace
cd /workspace
git clone https://github.com/<your-handle>/COOM.git
cd COOM
bash CL/scripts/gpu/setup_runpod.sh
```

The setup script:

- runs `nvidia-smi` to verify the driver is wired up,
- installs the system packages ViZDoom needs (`libsdl2-dev`, `libboost-all-dev`, etc.),
- creates `/workspace/venv` and installs TF 2.11, ViZDoom 1.3 and the COOM repo,
- ends by importing TF and asserting that at least one GPU is visible.

If the final TF assertion fails, the pod is on the wrong CUDA major
version. Stop the pod (you do not want to be charged for it) and pick a
different template; CUDA 12.x will not work for TF 2.11.

Once setup completes, activate the venv in any future shell on the pod:

```bash
source /workspace/venv/bin/activate
```

---

## 3. Smoke test (~10 minutes)

Before launching the long queue, run the smoke test. It runs EWC at
5,000 steps per task on CO4 --- enough to verify the full code path
including ViZDoom rendering, multi-task wrapping, and TF GPU --- but
short enough that you can interrupt it without guilt.

```bash
cd /workspace/COOM
bash CL/scripts/gpu/smoke_test.sh
```

Acceptance criterion: the script ends with `[smoke] OK` and the
`progress.tsv` it created has at least 10 rows. If either fails, do
not run the long queue --- diagnose first.

---

## 4. Production queue

### Option A -- minimal: EWC × 3 seeds (~2 h, ~$0.70)

This is what you need to write \"I reproduced the EWC number on CO4\":

```bash
cd /workspace/COOM
nohup bash CL/scripts/gpu/queue_minimal.sh > gpu_queue.out 2>&1 &
disown
```

`nohup` + `disown` lets you log out of SSH without killing the queue.

### Option B -- full: 4 methods × 3 seeds (~6--8 h, ~$3)

For the full comparison table:

```bash
cd /workspace/COOM
nohup bash CL/scripts/gpu/queue_paper.sh > gpu_queue.out 2>&1 &
disown
```

Either way, the live status is in `logs/gpu/queue_status.txt`. Tail it
from your laptop (over SSH) any time:

```bash
ssh -p <POD_PORT> root@ssh.runpod.io \
    'tail -f /workspace/COOM/logs/gpu/queue_status.txt'
```

---

## 5. While the queue runs

You can leave RunPod and shut your laptop. The pod keeps running as long
as your credit holds out. Cost ticks at the spot rate per minute.

Useful sanity checks during the run:

```bash
# is TF still pinned to the GPU?
ssh -p <POD_PORT> root@ssh.runpod.io 'nvidia-smi'

# how many epochs has the current run produced?
ssh -p <POD_PORT> root@ssh.runpod.io \
    'wc -l /workspace/COOM/logs/*_co4_200000_seed*/2*/progress.tsv'
```

---

## 6. After the queue completes

Pull the logs back to your laptop. From the LAPTOP, not the pod:

```bash
cd /Users/qilinz/Desktop/coom/COOM
POD_HOST=root@ssh.runpod.io POD_PORT=<your_port> \
    bash CL/scripts/gpu/sync_logs.sh
```

This rsyncs the entire `logs/` tree on the pod into `logs/gpu_sync/`
locally, preserving directory structure. The make-figures pipeline reads
TSV files via a glob pattern that already matches the new layout; no
code changes needed.

Re-run the figure / table regeneration:

```bash
PYTHONPATH=. python paper/scripts/make_figures.py
```

The new outputs land in `paper/figs/` and `paper/tab*.tex`. Compare the
three-seed `EWC` row in `paper/tab1_avg_perf.tex` against the COOM
paper's reported $0.54$ value.

---

## 7. Stop the pod

Critical step --- this is what stops the meter.

1. Go to the RunPod console.
2. Click the running pod.
3. Click **Stop** (preserves the volume; you can resume) or **Delete**
   (frees the volume; cheaper if you are done).

Verify on the *Billing* page that the meter has stopped within the next
five minutes. If it has not, file a support ticket --- RunPod has been
known to bill stopped pods through orphaned proxies.

---

## 8. Docker path (optional)

If you intend to reproduce more than once, build and push the image:

```bash
cd /Users/qilinz/Desktop/coom/COOM
docker build -f docker/Dockerfile.gpu -t <your-handle>/coom-gpu:tf211 .
docker push <your-handle>/coom-gpu:tf211
```

On RunPod, create a *Custom Template* pointing at
`<your-handle>/coom-gpu:tf211`, set the volume mount to `/workspace`,
and skip steps 2--3 of this runbook --- the entrypoint already verifies
GPU visibility on container start.

---

## 9. Cost ledger (filled in as you go)

| Date | Action                | Pod hours | Cost |
|------|-----------------------|-----------|------|
|      | smoke test            |           |      |
|      | minimal queue (EWC×3) |           |      |
|      | full queue (4×3)      |           |      |
|      | re-runs / debug       |           |      |
|      | **total**             |           |      |

Keep this honest --- the dissertation's reproducibility appendix can
cite the actual cost if the audit committee asks.
