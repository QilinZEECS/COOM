# FYP — Notes on Improving the COOM EWC Baseline

2026-04-22

---

## 1. Task

COOM (Tomilin 2023) is a continual RL benchmark built on ViZDoom. It has 8 scenarios and a handful of task sequence families (CD, CO, COC, etc.), and provides 9 CL methods on top of SAC as baselines.

The FYP has three parts:

1. Pick one baseline (EWC), reproduce it with the paper's config, and get numbers close to the paper. This is the control.
2. Modify EWC in some well-motivated way.
3. Run the modified version alongside the original and analyse whether the change actually helps. If my numbers don't match the paper, I need to be able to explain why.

Evaluation follows the three metrics from the paper:

- **Average Performance** — mean success rate across all tasks after training
- **Forgetting** — performance on task i when its training ended, minus performance on task i at the end of the whole sequence
- **Forward Transfer** — learning-efficiency gain relative to SAC trained from scratch on each single task

The last two require evaluating on past tasks during training (i.e. `--test=True`). My current run has test turned off, so I can only read off a rough Average Performance from training curves.

### Where things stand

Done:

- Python 3.10 + TF 2.11 (CPU-only) + ViZDoom environment set up
- EWC pipeline verified end-to-end
- One full run on CO4, `steps_per_env=20k`, `seed=1`, training curves plotted
- Plotting script + comparison writeup (`ewc_co4_20k_seed1.png`, `comparison_ewc_co4_vs_paper.md`)

Not yet done:

- No eval — no Forgetting / FT numbers
- No Fine-Tuning control run
- Haven't run the paper's full `steps_per_env=2e5` config

---

## 2. How EWC is implemented in this repo

### The original EWC (Kirkpatrick 2017)

$$\mathcal{L}(\theta) = \mathcal{L}_{\text{current}}(\theta) + \lambda \sum_i F_i (\theta_i - \theta^*_i)^2$$

$F_i$ is the diagonal of the Fisher information matrix, measuring how important parameter i was for the old task. $\theta^*$ is the parameter snapshot at the end of the previous task.

### The RL adaptation in `CL/methods/ewc.py`

A few things are different from the supervised version:

- **How Fisher is computed**: Jacobian of (sum of actor logits) and (Q1/Q2 outputs) w.r.t. parameters, squared, averaged over the batch. Strictly speaking this is a variant of empirical Fisher, not the full expected Fisher.
- **What data**: 10 batches of 32 samples from the replay buffer at the end of the current task.
- **Which parameters are regularised**: only `common_variables` — the shared backbone in the multi-head architecture. Per-task output heads are free.
- **How Fishers combine across tasks**: element-wise sum (`_merge_weights`). Fisher grows monotonically as more tasks are added.
- **Floor**: Fisher is clipped to `1e-5` from below so no parameter is ever fully abandoned.
- **Critic regularisation**: off by default (`regularize_critic=False`). Only the actor is constrained.

Paper's recommended config:

```bash
python run_cl.py --cl_method ewc --cl_reg_coef 250 \
                 --sequence CO8 --seed {1..10}
```

Everything else uses the defaults in `config.py`: lr=1e-3, γ=0.99, α=auto, replay_size=5e4, batch_size=128, steps_per_env=2e5.

### Why EWC probably scores only 0.54 in COOM

In the paper's baseline table EWC is 5th (PackNet 0.74 > ClonEx 0.73 > L2 0.64 > MAS 0.56 > EWC 0.54), which means EWC actually isn't strong on COOM. Plausible reasons:

1. The Fisher approximation isn't rigorous — jacobian-of-sum is not expected Fisher.
2. The magnitude of summed Fisher gets out of hand on long sequences, so either λ ends up crushing the policy or Fisher drowns λ out.
3. The critic isn't constrained. Q-value drift feeds back into the actor loss and perturbs the actor indirectly.
4. SAC's α is auto-tuned per task, which adds another source of parameter drift at task boundaries.
5. The replay buffer resets on task change by default, so Fisher is estimated from just the tail of the new task's rollouts.

All five are plausible places to intervene.

---

## 3. Candidate improvements

Eight directions I've thought about, roughly from cleanest/cheapest to most ambitious.

**Online EWC** replaces the running sum of Fisher matrices with an EMA (`F ← γF + (1-γ)F_new`). The motivation is that summed Fisher grows monotonically with the number of tasks, and on long sequences either the constraint crushes the policy or λ gets drowned out. Low implementation effort, there's prior work to cite, and the story writes itself.

**Per-layer λ** gives separate regularisation strengths to the CNN encoder, the shared MLP, and the output heads. The CNN learns low-level visual features that should transfer across tasks (strong constraint), while the heads are task-specific and shouldn't be held back (weak or zero constraint). Low effort, easy to motivate, easy to write up.

**True Fisher** swaps the current "sum-of-logits jacobian" for gradients of the action log-probability, which is closer to the textbook definition of the Fisher information matrix. More rigorous, but the story is a technical detail rather than a methodological contribution.

**Fisher percentile clipping** only regularises the top-k% most important parameters and ignores the rest. The observation is that most Fisher values are near zero, so constraining those parameters wastes capacity without protecting anything meaningful. Low effort, story is OK.

**Dynamic λ** scales the regularisation strength per task, either by difficulty or by Fisher magnitude. The fixed λ=250 from the paper treats easy and hard tasks the same, which probably isn't optimal. Medium effort, story is OK.

**`regularize_critic` on/off** is a trivial ablation — the flag already exists, it just defaults to off. Not a contribution on its own, but worth running for the ablation table.

**EWC + reservoir replay hybrid** combines regularisation with a small episodic memory. The case is that pure regularisation methods consistently underperform memory-based methods in the COOM table (EWC 0.54 vs ClonEx 0.73). Ambitious and the most likely to actually move the number, but high implementation effort and higher risk of not working.

**Uncertainty-aware Fisher** uses MC Dropout (or ensembles) to estimate parameter importance instead of Fisher. More academically interesting, but heavy engineering and the payoff isn't obvious.

I'd start with **Online EWC + Per-layer λ** because:

- They're orthogonal, so I can ablate them individually and together.
- Both only touch `regularization.py` / `ewc.py`, nothing in the SAC core.
- The narrative works: "EWC breaks down on long sequences → fix Fisher management (EMA) and spatial allocation (per-layer) → EWC comes back."

### Online EWC — what changes

In `regularization.py`, change `_merge_weights`:

```python
# current: accumulate
merged = F_old + F_current_task

# proposed: EMA
merged = γ * F_old + (1 - γ) * F_current_task    # γ ∈ [0.9, 0.99]
```

Sweep γ ∈ {0.9, 0.95, 0.99}. Expect Forgetting to be slightly worse (older tasks constrained less), but plasticity on new tasks to improve. Net effect depends on the trade-off.

### Per-layer λ — what changes

Bucket `reg_weights` by layer (CNN encoder / shared MLP / actor head / critic head), assign each bucket its own λ:

```python
λ_cnn  = 500   # strong: shared features
λ_mlp  = 100   # middle
λ_head = 0     # free: task heads are supposed to change
```

Try a few hand-picked combinations first, then grid search if any of them look promising.

### Ablations worth stacking

- Default `regularize_critic=False` vs `True`
- `reset_buffer_on_task_change=True` vs `False`
- Bump Fisher sample budget from 10×32 to 50×64 and see if stability improves

---

## 4. Roadmap

### Done
Environment + pipeline verification + one 20k-step run + plots.

### Local, next 2–3 days
1. Run the Fine-Tuning baseline (same CO4 × 20k × seed=1 with `cl_method=None`) and plot it alongside EWC. This is the most direct way to see if EWC is actually helping.
2. Re-run EWC with eval on, to get real Forgetting numbers.
3. One ablation run with `regularize_critic=True`.

Deliverable: one plot comparing Fine-Tuning / EWC / EWC+critic curves, plus a table with Forgetting.

### Implement the improvement, 3–5 days
4. Pick Online EWC, Per-layer λ, or both as the main contribution.
5. Add `CL/methods/ewc_online.py` (and/or `ewc_layerwise.py`) that subclasses `Regularization_SAC` and overrides just what's needed.
6. Register it in `run_cl.py`'s `CLMethod` enum.
7. Smoke-test locally on CO4 × 20k.

### Scale up on a cloud GPU, 3–5 days, ~$15–30
8. Rent an RTX 3090 on RunPod.
9. Run at the paper's `steps_per_env=2e5`:
   - Original EWC × 3 seeds on CO4
   - Improved EWC × 3 seeds on CO4
   - Fine-Tuning × 3 seeds on CO4
10. If budget allows, extend to CO8.

### Analysis and writing, 3–5 days
11. All three metrics (Avg Perf / Forgetting / FT) side by side.
12. Discuss whether the improved version reaches the paper's 0.54; if not, whether it's seed variance or an implementation difference; whether the original hypothesis held up.
13. Final figures, update `comparison.md`.

---

## 5. Things I need you to decide before I can move on

1. **Direction**: go with Online EWC + Per-layer λ as I've scoped, or try something else (e.g. the EWC + replay hybrid)?
2. **Hardware**: willing to spend $15–30 on RunPod for Phase 3? Any university GPU resources available?
3. **Final deliverable**: paper (LaTeX), report (Markdown/Word), codebase only, plus slides?
4. **Deadlines**: FYP final deadline? Any interim check-ins?

Once I know these I can turn the roadmap into concrete experiment scripts.

---

## Appendix — where to look

```
COOM/
├── CL/methods/ewc.py                 the core to read
├── CL/methods/regularization.py      Fisher merge lives here
├── CL/rl/sac.py                      SAC main loop
├── CL/run_cl.py                      experiment entry point
├── CL/config.py                      CLI args
├── CL/scripts/cl.sh                  paper's run commands
├── CL/logs/ewc_co4_20k_seed1/        this run's logs
├── COOM/env/builder.py               env construction
├── COOM/utils/config.py              sequence / task defs
├── results/ewc_co4_20k_seed1.png     training curves
├── results/comparison_ewc_co4_vs_paper.md
└── results/fyp_ewc_plan_en.md        this file
```
