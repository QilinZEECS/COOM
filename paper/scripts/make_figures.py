"""Generate all paper figures + tables from the COOM-CL CPU runs.

Outputs (paper/figs/):
  fig1_success_curves.pdf    : 4-task x 5-method success-rate training curves
  fig2_return_curves.pdf     : 4-task x 5-method return curves
  fig3_loss_reg.pdf          : EWC/L2/MAS regularization-loss dynamics
  fig4_ewc_seed_band.pdf     : EWC mean+-std across 3 seeds per task
  tab1_avg_perf.tex          : LaTeX table - Average Performance (last-5-epoch mean)
  tab2_forgetting.tex        : LaTeX table - rough Forgetting (peak-during minus end)

Run from project root:
  PYTHONPATH=. python paper/scripts/make_figures.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = ROOT / "logs"
EWC_LEGACY_DIR = ROOT / "CL" / "logs" / "ewc_co4_20k_seed1"
OUT = ROOT / "paper" / "figs"
OUT.mkdir(parents=True, exist_ok=True)

CO4_TASKS = ["Chainsaw", "Raise the Roof", "Run and Gun", "Health Gathering"]
N_TASKS = 4
STEPS_PER_TASK = 20_000
EPOCHS_PER_TASK = 30  # log_every=1000, steps_per_env=20000; extra rows from task-switch epochs


def latest_progress(group_dir: Path) -> Path:
    runs = sorted([p for p in group_dir.iterdir() if p.is_dir()])
    if not runs:
        raise FileNotFoundError(f"no run subdir under {group_dir}")
    return runs[-1] / "progress.tsv"


def _has_complete_run(group_dir: Path) -> bool:
    """True iff the latest run subdirectory under `group_dir` has training
    rows for all four CO4 tasks. Uses the same trailing-tab-tolerant
    loader as `load_run` so it doesn't silently truncate at task
    boundaries where the COOM logger appends new metric columns."""
    if not group_dir.is_dir():
        return False
    runs = sorted([p for p in group_dir.iterdir() if p.is_dir()])
    if not runs:
        return False
    progress = runs[-1] / "progress.tsv"
    if not progress.exists():
        return False
    try:
        df = load_run(group_dir)
        return df["train/active_env"].nunique() == N_TASKS
    except Exception:
        return False


def _existing_dirs(*candidates: Path) -> list[Path]:
    """Filter out incomplete runs so the script can be re-run
    while a new method is still being collected."""
    return [c for c in candidates if _has_complete_run(c)]


RUNS: dict[str, list[Path]] = {
    "FT":  [LOG_DIR / "ft_co4_20k_seed1"],
    "L2":  [LOG_DIR / "l2_co4_20k_seed1"],
    "MAS": [LOG_DIR / "mas_co4_20k_seed1"],
    "EWC": [
        EWC_LEGACY_DIR,
        LOG_DIR / "ewc_co4_20k_seed2",
        LOG_DIR / "ewc_co4_20k_seed3",
    ],
}
# Online EWC: improved variant; the gate that adds it to RUNS lives at the
# bottom of this file (after `load_run` is defined) because the
# completeness check needs the trailing-tab-tolerant loader.

COLORS = {
    "FT": "#888888", "EWC": "#1f77b4", "L2": "#d62728",
    "MAS": "#2ca02c", "EWC-Online": "#ff7f0e",
}


def load_run(group_dir: Path) -> pd.DataFrame:
    """The TSV grows extra trailing columns at each task switch (39 -> 40 ->
    41 -> ...). We read the file with a header padded with enough placeholder
    names to absorb up to 20 extra columns, then drop them."""
    path = latest_progress(group_dir)
    with path.open() as f:
        header = f.readline().rstrip("\n").split("\t")
    names = header + [f"_pad{i}" for i in range(20)]
    df = pd.read_csv(path, sep="\t", skiprows=1, names=names,
                     engine="c", index_col=False)
    df = df[header]
    # `train/active_env` can be NaN on eval-only epochs (the paper-budget
    # runs interleave held-out evaluation rows). Drop those rows for the
    # task-indexed views; the original df is preserved for full-sequence
    # plots that don't require a task assignment.
    if df["train/active_env"].notna().any():
        df = df[df["train/active_env"].notna()].copy()
        df["task"] = df["train/active_env"].astype(int)
        df["epoch_in_task"] = df.groupby("task").cumcount() + 1
    else:
        df["task"] = -1
        df["epoch_in_task"] = 0
    return df


def stack_method(group_dirs: list[Path]) -> dict[int, pd.DataFrame]:
    """Return {seed_idx: df} for one method (single-element dict if single seed)."""
    return {i: load_run(d) for i, d in enumerate(group_dirs)}


def per_task_arr(df: pd.DataFrame, col: str) -> np.ndarray:
    """Return shape (N_TASKS, EPOCHS_PER_TASK) array, NaN-padded."""
    arr = np.full((N_TASKS, EPOCHS_PER_TASK), np.nan)
    for t in range(N_TASKS):
        sub = df[df["task"] == t][col].to_numpy()
        arr[t, : len(sub)] = sub
    return arr


def last_k_per_task(df: pd.DataFrame, col: str, k: int = 5) -> np.ndarray:
    """Mean of the last k non-NaN epochs of each task. Shape (N_TASKS,)."""
    out = np.full(N_TASKS, np.nan)
    for t in range(N_TASKS):
        sub = df[df["task"] == t][col].dropna().to_numpy()
        if len(sub) >= 1:
            out[t] = sub[-k:].mean()
    return out


def peak_per_task(df: pd.DataFrame, col: str) -> np.ndarray:
    out = np.full(N_TASKS, np.nan)
    for t in range(N_TASKS):
        sub = df[df["task"] == t][col].dropna().to_numpy()
        if len(sub) >= 1:
            out[t] = sub.max()
    return out


def smooth(y: np.ndarray, k: int = 3) -> np.ndarray:
    """Simple centered moving average ignoring NaN."""
    if y.ndim == 1:
        s = pd.Series(y).rolling(k, min_periods=1, center=True).mean().to_numpy()
        return s
    return np.vstack([smooth(row, k) for row in y])


def fig_curves(metric: str, ylabel: str, fname: str, smooth_k: int = 3) -> None:
    fig, axes = plt.subplots(1, N_TASKS, figsize=(13, 2.8), sharey=True)
    for t, (ax, name) in enumerate(zip(axes, CO4_TASKS)):
        for method, dirs in RUNS.items():
            seed_runs = stack_method(dirs)
            arrs = np.stack([per_task_arr(df, metric) for df in seed_runs.values()])
            mean = np.nanmean(arrs, axis=0)
            x = np.arange(1, EPOCHS_PER_TASK + 1)
            y = smooth(mean[t], smooth_k)
            ax.plot(x, y, label=method, color=COLORS[method], lw=1.6)
            if arrs.shape[0] > 1:  # band only if multiple seeds (EWC)
                std = np.nanstd(arrs, axis=0)
                ax.fill_between(
                    x, smooth(mean[t] - std[t], smooth_k),
                    smooth(mean[t] + std[t], smooth_k),
                    color=COLORS[method], alpha=0.18, lw=0,
                )
        ax.set_title(f"Task {t}: {name}", fontsize=10)
        ax.set_xlabel("Epoch within task")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel(ylabel)
    axes[-1].legend(loc="upper left", fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / fname, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT / fname}")


def fig_loss_reg() -> None:
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    for method in ["EWC", "L2", "MAS"]:
        dirs = RUNS[method]
        arrs = []
        for d in dirs:
            df = load_run(d)
            v = df["train/loss_reg"].to_numpy()
            arrs.append(v[:N_TASKS * EPOCHS_PER_TASK])
        L = max(len(a) for a in arrs)
        padded = np.full((len(arrs), L), np.nan)
        for i, a in enumerate(arrs):
            padded[i, : len(a)] = a
        mean = np.nanmean(padded, axis=0)
        ax.plot(np.arange(1, len(mean) + 1), smooth(mean, 3),
                label=method, color=COLORS[method], lw=1.6)
    for t in range(1, N_TASKS):
        ax.axvline(t * EPOCHS_PER_TASK, color="k", alpha=0.25, lw=0.7, ls="--")
    ax.set_yscale("symlog", linthresh=1.0)
    ax.set_xlabel("Training epoch (across all 4 tasks)")
    ax.set_ylabel(r"$\mathcal{L}_{\mathrm{reg}}$")
    ax.set_title("Regularization loss across the CO4 sequence")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig3_loss_reg.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT / 'fig3_loss_reg.pdf'}")


def fig_ewc_seed_band() -> None:
    seeds = stack_method(RUNS["EWC"])
    means = np.stack([last_k_per_task(df, "train/success", 5)
                      for df in seeds.values()])  # (seeds, tasks)
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    width = 0.22
    x = np.arange(N_TASKS)
    for i, (k, _) in enumerate(seeds.items()):
        ax.bar(x + (i - 1) * width, means[i], width=width, label=f"seed {k+1}",
               color=plt.cm.Blues(0.4 + 0.2 * i))
    ax.bar(x + 2 * width, means.mean(axis=0), width=width, label="mean",
           color="#1f77b4", edgecolor="black", lw=1.0)
    ax.set_xticks(x + 0.5 * width)
    ax.set_xticklabels(CO4_TASKS, rotation=15, ha="right")
    ax.set_ylabel("Last-5-epoch success rate")
    ax.set_title("EWC seed-level variance per task")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig4_ewc_seed_band.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT / 'fig4_ewc_seed_band.pdf'}")


def make_table_avg_perf() -> None:
    rows = []
    for method, dirs in RUNS.items():
        seed_runs = stack_method(dirs)
        last5 = np.stack([last_k_per_task(df, "train/success", 5)
                          for df in seed_runs.values()])  # (seeds, tasks)
        per_task_mean = np.nanmean(last5, axis=0)
        per_task_std = np.nanstd(last5, axis=0) if last5.shape[0] > 1 else None
        avg_perf = per_task_mean.mean()
        rows.append((method, per_task_mean, per_task_std, avg_perf, last5.shape[0]))

    lines = [
        r"\begin{tabular}{lcccc|c}",
        r"\toprule",
        r"Method & " + " & ".join(f"T{i}" for i in range(N_TASKS)) +
        r" & \textbf{Avg.\ Perf.} \\",
        r"\midrule",
    ]
    for method, m, s, ap, n in rows:
        cells = []
        for i in range(N_TASKS):
            if s is not None:
                cells.append(f"{m[i]:.3f} $\\pm$ {s[i]:.3f}")
            else:
                cells.append(f"{m[i]:.3f}")
        lines.append(f"{method} ({n} seed{'s' if n > 1 else ''}) & " +
                     " & ".join(cells) + f" & \\textbf{{{ap:.3f}}} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT.parent / "tab1_avg_perf.tex").write_text("\n".join(lines))
    print("wrote tab1_avg_perf.tex")


def make_table_forgetting() -> None:
    """Within-task plasticity proxy: peak success during task i minus end of
    its own training phase (last-5-epoch mean). Without held-out eval we
    cannot measure cross-task forgetting directly; this captures only how
    much a method drifts inside its own training window — a lower bound on
    the regularization tax."""
    rows = []
    for method, dirs in RUNS.items():
        seed_runs = stack_method(dirs)
        per_seed = []
        for df in seed_runs.values():
            peak = peak_per_task(df, "train/success")
            end = last_k_per_task(df, "train/success", 5)
            per_seed.append(peak - end)
        arrs = np.stack(per_seed)
        m = np.nanmean(arrs, axis=0)
        avg = np.nanmean(m)
        rows.append((method, m, avg, arrs.shape[0]))

    lines = [
        r"\begin{tabular}{lcccc|c}",
        r"\toprule",
        r"Method & " + " & ".join(f"T{i}" for i in range(N_TASKS)) +
        r" & \textbf{Avg.} \\",
        r"\midrule",
    ]
    for method, m, avg, n in rows:
        cells = " & ".join(f"{m[i]:.3f}" for i in range(N_TASKS))
        lines.append(f"{method} ({n}) & {cells} & \\textbf{{{avg:.3f}}} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT.parent / "tab2_forgetting.tex").write_text("\n".join(lines))
    print("wrote tab2_forgetting.tex")


def _full_sequence_arr(df: pd.DataFrame, col: str) -> np.ndarray:
    """Return values of `col` ordered as the training sequence
    (concatenated across the 4 tasks, NaNs preserved)."""
    return df[col].to_numpy()


def fig_training_dynamics() -> None:
    """Three-panel: actor loss, Q1 loss, and (per-task) SAC alpha across
    the full CO4 sequence for FT and EWC. Supports the Fisher-on-noise
    discussion: at each task switch the alpha for the new task is reset
    and the Q loss spikes, evidence that the source-task end-state is
    far from converged."""
    fig, axes = plt.subplots(3, 1, figsize=(7.0, 6.6), sharex=True)
    methods_to_show = ["FT", "EWC"]
    for method in methods_to_show:
        seed_runs = stack_method(RUNS[method])
        df0 = next(iter(seed_runs.values()))
        x = np.arange(1, len(df0) + 1)
        # actor loss
        arrs = np.stack([_full_sequence_arr(d, "train/loss_pi") for d in seed_runs.values()])
        axes[0].plot(x, smooth(np.nanmean(arrs, axis=0), 3), label=method,
                     color=COLORS[method], lw=1.6)
        # Q1 loss
        arrs = np.stack([_full_sequence_arr(d, "train/loss_q1") for d in seed_runs.values()])
        axes[1].plot(x, smooth(np.nanmean(arrs, axis=0), 3), label=method,
                     color=COLORS[method], lw=1.6)
    # alpha for the active task only (4 columns, one per multi-head)
    for method in methods_to_show:
        seed_runs = stack_method(RUNS[method])
        df0 = next(iter(seed_runs.values()))
        active_alpha = np.full(len(df0), np.nan)
        for i, row in df0.iterrows():
            t = int(row["task"])
            active_alpha[i] = row[f"train/alpha/{t}"]
        x = np.arange(1, len(df0) + 1)
        axes[2].plot(x, smooth(active_alpha, 3), label=method,
                     color=COLORS[method], lw=1.6)
    for ax in axes:
        for t in range(1, N_TASKS):
            ax.axvline(EPOCHS_PER_TASK * t, color="k", alpha=0.25,
                       lw=0.7, ls="--")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel(r"$\mathcal{L}_{\pi}$ (actor loss)")
    axes[1].set_ylabel(r"$\mathcal{L}_{Q_1}$")
    axes[2].set_ylabel(r"SAC $\alpha$ (active task)")
    axes[2].set_xlabel("Training epoch (concatenated across the 4 tasks)")
    axes[0].legend(frameon=False, loc="upper right", fontsize=9)
    fig.suptitle("Training dynamics across the CO4 sequence",
                 fontsize=11, y=0.99)
    fig.tight_layout()
    fig.savefig(OUT / "fig5_training_dynamics.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT / 'fig5_training_dynamics.pdf'}")


def fig_action_distribution() -> None:
    """Stacked bar of mean action proportions in the last 5 epochs of
    each task, per method. Reveals policy collapse modes."""
    action_groups = {
        "NO-OP / EXECUTE": [0, 1],
        "MOVE_FORWARD (+EXECUTE)": [2, 3],
        "TURN_RIGHT (+combo)": [4, 5, 6, 7],
        "TURN_LEFT (+combo)": [8, 9, 10, 11],
    }
    group_colors = ["#cccccc", "#1f77b4", "#ff7f0e", "#2ca02c"]
    fig, axes = plt.subplots(1, N_TASKS, figsize=(13, 2.8), sharey=True)
    methods = list(RUNS.keys())
    width = 0.7
    for t, (ax, name) in enumerate(zip(axes, CO4_TASKS)):
        bottoms = np.zeros(len(methods))
        for gname, idxs, gcolor in zip(action_groups.keys(),
                                        action_groups.values(), group_colors):
            heights = []
            for method in methods:
                seed_runs = stack_method(RUNS[method])
                vals = []
                for df in seed_runs.values():
                    sub = df[df["task"] == t].iloc[-5:]
                    grp = sub[[f"train/actions/{i}" for i in idxs]].sum(axis=1).mean()
                    total = sub[[f"train/actions/{i}" for i in range(12)]].sum(axis=1).mean()
                    vals.append(grp / total if total > 0 else 0.0)
                heights.append(np.mean(vals))
            ax.bar(np.arange(len(methods)), heights, bottom=bottoms,
                   width=width, color=gcolor, label=gname if t == 0 else None,
                   edgecolor="white", lw=0.6)
            bottoms += np.array(heights)
        ax.set_xticks(np.arange(len(methods)))
        ax.set_xticklabels(methods, fontsize=9)
        ax.set_title(f"Task {t}: {name}", fontsize=10)
        if t == 0:
            ax.set_ylabel("Action share (last 5 epochs)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.04), fontsize=9)
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(OUT / "fig6_action_dist.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT / 'fig6_action_dist.pdf'}")


def make_table_learning_speed() -> None:
    """Epochs-within-task at which the success rate first reaches half
    of the in-task peak. Lower is faster learning."""
    rows = []
    for method, dirs in RUNS.items():
        seed_runs = stack_method(dirs)
        per_seed = []
        for df in seed_runs.values():
            t_to_half = np.full(N_TASKS, np.nan)
            for t in range(N_TASKS):
                sub = df[df["task"] == t]["train/success"].dropna().to_numpy()
                if len(sub) == 0:
                    continue
                pk = sub.max()
                if pk <= 0:
                    continue
                hit = np.where(sub >= 0.5 * pk)[0]
                if len(hit) > 0:
                    t_to_half[t] = hit[0] + 1
            per_seed.append(t_to_half)
        arr = np.stack(per_seed)
        m = np.nanmean(arr, axis=0)
        rows.append((method, m, arr.shape[0]))
    lines = [
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Method & T0 & T1 & T2 & T3 \\",
        r"\midrule",
    ]
    for method, m, n in rows:
        cells = [f"{m[i]:.1f}" if not np.isnan(m[i]) else "---"
                 for i in range(N_TASKS)]
        lines.append(f"{method} ({n}) & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT.parent / "tab3_learning_speed.tex").write_text("\n".join(lines))
    print("wrote tab3_learning_speed.tex")


def make_table_walltime() -> None:
    """Total wall-clock time per method (single-machine CPU)."""
    rows = []
    for method, dirs in RUNS.items():
        seed_runs = stack_method(dirs)
        per_seed = []
        for df in seed_runs.values():
            wt = df["walltime"].to_numpy()
            per_seed.append((wt[-1] - wt[0]) / 60.0)
        per_seed = np.array(per_seed)
        rows.append((method, per_seed.mean(), per_seed.std(),
                     per_seed.shape[0]))
    lines = [
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Method & Wall-clock per CO4 run (min) & Seeds \\",
        r"\midrule",
    ]
    for method, m, s, n in rows:
        cell = f"{m:.0f} $\\pm$ {s:.0f}" if n > 1 else f"{m:.0f}"
        lines.append(f"{method} & {cell} & {n} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT.parent / "tab4_walltime.tex").write_text("\n".join(lines))
    print("wrote tab4_walltime.tex")


def make_table_domain_metrics() -> None:
    """Last-5-epoch mean of the per-task domain-specific metric.
    Chainsaw / Run-and-Gun use kills (combat objective); Raise-the-Roof
    and Health Gathering use ep_length / health, the survival proxies
    used by the COOM paper's per-task success criteria."""
    metric_per_task = ["train/kills", "train/ep_length",
                       "train/kills", "train/ep_length"]
    rows = []
    for method, dirs in RUNS.items():
        seed_runs = stack_method(dirs)
        per_seed = []
        for df in seed_runs.values():
            vec = np.full(N_TASKS, np.nan)
            for t in range(N_TASKS):
                sub = df[df["task"] == t][metric_per_task[t]].dropna().to_numpy()
                if len(sub) > 0:
                    vec[t] = sub[-5:].mean()
            per_seed.append(vec)
        arr = np.stack(per_seed)
        m = np.nanmean(arr, axis=0)
        rows.append((method, m, arr.shape[0]))
    lines = [
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Method & Chainsaw (kills) & Raise-the-Roof (frames) "
        r"& Run-and-Gun (kills) & Health Gathering (frames) \\",
        r"\midrule",
    ]
    for method, m, n in rows:
        cells = [f"{m[i]:.1f}" for i in range(N_TASKS)]
        lines.append(f"{method} ({n}) & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT.parent / "tab5_domain_metrics.tex").write_text("\n".join(lines))
    print("wrote tab5_domain_metrics.tex")


def fig_action_entropy() -> None:
    """Shannon entropy of the 12-action distribution per epoch, across
    the full sequence, for FT and EWC. Low entropy = decisive policy;
    high entropy = exploring or uncertain. Supports the diagnosis that
    the regularised methods are *not* collapsing to a deterministic
    bad policy --- the entropy stays well above one bit throughout."""
    fig, ax = plt.subplots(figsize=(7.0, 3.0))
    methods_to_show = ["FT", "EWC"]
    for method in methods_to_show:
        seed_runs = stack_method(RUNS[method])
        per_seed_h = []
        for df in seed_runs.values():
            cols = [f"train/actions/{i}" for i in range(12)]
            counts = df[cols].to_numpy()
            counts = np.clip(counts, 1e-9, None)
            p = counts / counts.sum(axis=1, keepdims=True)
            h = -(p * np.log2(p)).sum(axis=1)
            per_seed_h.append(h)
        h_arr = np.stack(per_seed_h)
        mean = np.nanmean(h_arr, axis=0)
        x = np.arange(1, len(mean) + 1)
        ax.plot(x, smooth(mean, 3), color=COLORS[method],
                label=method, lw=1.6)
        if h_arr.shape[0] > 1:
            std = np.nanstd(h_arr, axis=0)
            ax.fill_between(x, smooth(mean - std, 3),
                            smooth(mean + std, 3),
                            color=COLORS[method], alpha=0.18, lw=0)
    ax.axhline(np.log2(12), color="k", alpha=0.4, lw=0.7, ls=":")
    ax.text(2, np.log2(12) - 0.05, "uniform 12-action policy",
            fontsize=8, color="k", alpha=0.6, va="top")
    for t in range(1, N_TASKS):
        ax.axvline(EPOCHS_PER_TASK * t, color="k", alpha=0.25,
                   lw=0.7, ls="--")
    ax.set_xlabel("Training epoch (concatenated across the 4 tasks)")
    ax.set_ylabel("Action entropy (bits)")
    ax.set_title("Policy decisiveness across the CO4 sequence")
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig8_action_entropy.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT / 'fig8_action_entropy.pdf'}")


def _ewc_online_complete() -> bool:
    """True iff at least one Online EWC run on disk has training data
    for all 4 CO4 tasks (i.e. has not been interrupted partway)."""
    if "EWC-Online" not in RUNS:
        return False
    for d in RUNS["EWC-Online"]:
        try:
            df = load_run(d)
        except Exception:
            continue
        if df["train/active_env"].nunique() == N_TASKS:
            return True
    return False


def fig_ewc_vs_online() -> None:
    """Side-by-side per-task curves of original EWC vs Online EWC.
    Only emits a figure if a completed Online EWC run is available."""
    if not _ewc_online_complete():
        print("skip fig_ewc_vs_online (Online EWC not finished yet)")
        return
    fig, axes = plt.subplots(1, N_TASKS, figsize=(13, 2.8), sharey=True)
    for t, (ax, name) in enumerate(zip(axes, CO4_TASKS)):
        for method in ["EWC", "EWC-Online"]:
            seed_runs = stack_method(RUNS[method])
            arrs = np.stack([per_task_arr(df, "train/success")
                             for df in seed_runs.values()])
            mean = np.nanmean(arrs, axis=0)
            x = np.arange(1, EPOCHS_PER_TASK + 1)
            ax.plot(x, smooth(mean[t], 3), label=method,
                    color=COLORS[method], lw=1.6,
                    linestyle="-" if method == "EWC" else "--")
            if arrs.shape[0] > 1:
                std = np.nanstd(arrs, axis=0)
                ax.fill_between(
                    x, smooth(mean[t] - std[t], 3),
                    smooth(mean[t] + std[t], 3),
                    color=COLORS[method], alpha=0.18, lw=0,
                )
        ax.set_title(f"Task {t}: {name}", fontsize=10)
        ax.set_xlabel("Epoch within task")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Success rate")
    axes[-1].legend(loc="upper left", fontsize=9, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "fig7_ewc_vs_online.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT / 'fig7_ewc_vs_online.pdf'}")


def _paper_budget_dir() -> Path | None:
    """Locate the GPU-synced paper-budget run, if present. Layout is
    `logs/gpu_sync/ewc_co4_200k_seed1/<timestamp>/`."""
    cand = ROOT / "logs" / "gpu_sync" / "ewc_co4_200k_seed1"
    if not cand.is_dir():
        return None
    runs = sorted([p for p in cand.iterdir() if p.is_dir()])
    if not runs:
        return None
    if not (runs[-1] / "progress.tsv").exists():
        return None
    return cand


PAPER_BUDGET_RUNS = {
    "EWC": ROOT / "logs" / "gpu_sync" / "ewc_co4_200k_seed1",
    "EWC-Online ($\\gamma$=0.50)": ROOT / "logs" / "gpu_sync" / "ewc_online_g050_co4_200k_seed1",
    "EWC-Online ($\gamma$=0.95)": ROOT / "logs" / "gpu_sync" / "ewc_online_g095_co4_200k_seed1",
    "EWC seed 2": ROOT / "logs" / "gpu_sync" / "ewc_co4_200k_seed2",
    "FT": ROOT / "logs" / "gpu_sync" / "ft_co4_200k_seed1",
    "EWC-Online ($\gamma$=0.30)": ROOT / "logs" / "gpu_sync" / "ewc_online_g030_co4_200k_seed1",
    "EWC-Online ($\gamma$=0.70)": ROOT / "logs" / "gpu_sync" / "ewc_online_g070_co4_200k_seed1",
    "EWC-Online ($\gamma$=0.85)": ROOT / "logs" / "gpu_sync" / "ewc_online_g085_co4_200k_seed1",
    "EWC-Online (γ=0.50) seed 2": ROOT / "logs" / "gpu_sync" / "ewc_online_g050_co4_200k_seed2",
}


def _heldout_last5(df: pd.DataFrame) -> np.ndarray:
    test_cols = [
        "test/stochastic/0/chainsaw-default/success",
        "test/stochastic/1/raise_the_roof-default/success",
        "test/stochastic/2/run_and_gun-default/success",
        "test/stochastic/3/health_gathering-default/success",
    ]
    out = np.full(N_TASKS, np.nan)
    for i, c in enumerate(test_cols):
        if c in df.columns:
            sub = df[c].dropna()
            if len(sub) > 0:
                out[i] = float(sub.tail(5).mean())
    return out


def fig_gamma_sweep() -> None:
    """Online EWC CO4 average vs the EMA decay factor gamma, with the
    additive-Fisher EWC baseline as a horizontal reference. Picks up
    every Online-EWC paper-budget run on disk and plots them as
    individual points with seed labels where multi-seed exists."""
    points: list[tuple[float, float, str]] = []  # (gamma, avg, label)
    ewc_baseline_avg: float | None = None
    ft_avg: float | None = None
    for label, group_dir in PAPER_BUDGET_RUNS.items():
        if not (_has_complete_run(group_dir) or _has_partial_run(group_dir)):
            continue
        try:
            df = load_run(group_dir)
        except Exception:
            continue
        ho = _heldout_last5(df)
        if np.all(np.isnan(ho)):
            continue
        avg = float(np.nanmean(ho))
        if "EWC-Online" in label:
            # extract gamma from label like "EWC-Online ($\gamma$=0.50)"
            import re
            m = re.search(r"=([\d.]+)", label)
            if m:
                points.append((float(m.group(1)), avg, label))
        elif label == "EWC":
            ewc_baseline_avg = avg
        elif label == "FT":
            ft_avg = avg
    if not points:
        print("skip fig_gamma_sweep (no Online EWC paper-budget runs)")
        return
    points.sort()
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    ax.plot(xs, ys, "o-", color="#ff7f0e", lw=1.6, markersize=8,
            label="Online EWC")
    if ewc_baseline_avg is not None:
        ax.axhline(ewc_baseline_avg, color="#1f77b4", lw=1.4,
                   linestyle="--",
                   label=f"EWC additive (baseline): {ewc_baseline_avg:.3f}")
    if ft_avg is not None:
        ax.axhline(ft_avg, color="#888888", lw=1.4, linestyle=":",
                   label=f"Fine-Tuning: {ft_avg:.3f}")
    ax.set_xlabel(r"EMA decay $\gamma$")
    ax.set_ylabel("Held-out CO4 average")
    ax.set_title("Online EWC: CO4 average vs $\\gamma$ at paper budget (200k)")
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig10_gamma_sweep.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT / 'fig10_gamma_sweep.pdf'}")


def make_table_unified() -> None:
    """Unified comparison: EWC and Online EWC at both budgets in a single
    table. Designed to make the budget-flip (Outcome C at 20k -> Outcome
    A at 200k) visible at a glance."""
    rows = []
    # 20k CPU rows: train/success last-5 averages, no held-out available
    cpu_seed_runs_ewc = stack_method(RUNS["EWC"])
    cpu_ewc_means = np.stack([
        last_k_per_task(df, "train/success", 5)
        for df in cpu_seed_runs_ewc.values()
    ])
    rows.append((
        "EWC additive (CPU 20k, train-time)",
        np.nanmean(cpu_ewc_means, axis=0),
        np.nanmean(np.nanmean(cpu_ewc_means, axis=0)),
        f"{cpu_ewc_means.shape[0]} seeds",
    ))
    if "EWC-Online" in RUNS:
        oe = stack_method(RUNS["EWC-Online"])
        oe_means = np.stack([
            last_k_per_task(df, "train/success", 5)
            for df in oe.values()
        ])
        rows.append((
            r"Online EWC ($\gamma$=0.95, CPU 20k, train-time)",
            np.nanmean(oe_means, axis=0),
            np.nanmean(np.nanmean(oe_means, axis=0)),
            f"{oe_means.shape[0]} seed",
        ))
    # 200k held-out rows
    for label, group_dir in PAPER_BUDGET_RUNS.items():
        if not (_has_complete_run(group_dir) or _has_partial_run(group_dir)):
            continue
        try:
            df = load_run(group_dir)
        except Exception:
            continue
        ho = _heldout_last5(df)
        if np.all(np.isnan(ho)):
            continue
        avg = float(np.nanmean(ho))
        rows.append((
            f"{label} (GPU 200k, held-out)",
            ho,
            avg,
            "1 seed",
        ))

    lines = [
        r"\begin{tabular}{lcccc|cl}",
        r"\toprule",
        r"Method (budget, metric) & T0 & T1 & T2 & T3 & "
        r"\textbf{Avg.} & Seeds \\",
        r"\midrule",
    ]
    for label, vec, avg, seeds in rows:
        cells = " & ".join(
            "---" if np.isnan(v) else f"{v:.3f}" for v in vec
        )
        avg_cell = "---" if np.isnan(avg) else f"\\textbf{{{avg:.3f}}}"
        lines.append(f"{label} & {cells} & {avg_cell} & {seeds} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT.parent / "tab8_unified.tex").write_text("\n".join(lines))
    print(f"wrote tab8_unified.tex ({len(rows)} rows)")


def make_table_paper_budget() -> None:
    """Paper-budget reproduction table. Reports held-out (test/stochastic)
    success per task plus the CO4 average for every method whose
    GPU-synced run is on disk. Training-time numbers are dropped here
    because the held-out version is the metric the COOM paper itself
    reports."""
    rows = []
    for label, group_dir in PAPER_BUDGET_RUNS.items():
        if not _has_complete_run(group_dir) and not _has_partial_run(group_dir):
            continue
        try:
            df = load_run(group_dir)
        except Exception:
            continue
        ho = _heldout_last5(df)
        if np.all(np.isnan(ho)):
            continue
        rows.append((label, ho))
    if not rows:
        print("skip tab7_paper_budget (no paper-budget runs on disk)")
        return

    lines = [
        r"\begin{tabular}{lcccc|c}",
        r"\toprule",
        r"Method & T0 & T1 & T2 & T3 & \textbf{Avg.} \\",
        r"\midrule",
    ]
    for label, ho in rows:
        cells = " & ".join(
            "---" if np.isnan(v) else f"{v:.3f}" for v in ho
        )
        avg = np.nanmean(ho)
        avg_cell = "---" if np.isnan(avg) else f"\\textbf{{{avg:.3f}}}"
        lines.append(f"{label} & {cells} & {avg_cell} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT.parent / "tab7_paper_budget.tex").write_text("\n".join(lines))
    print(f"wrote tab7_paper_budget.tex ({len(rows)} methods)")


def _has_partial_run(group_dir: Path) -> bool:
    """True iff the latest run subdirectory exists with a non-empty
    progress.tsv (regardless of whether all 4 tasks are present)."""
    if not group_dir.is_dir():
        return False
    runs = sorted([p for p in group_dir.iterdir() if p.is_dir()])
    if not runs:
        return False
    p = runs[-1] / "progress.tsv"
    return p.exists() and p.stat().st_size > 200


def fig_paper_budget_curves() -> None:
    """Held-out per-task success across the full paper-budget sequence,
    overlaid on the training-time success of the same run. The held-out
    line shows catastrophic forgetting: every task sees its success
    rate climb during its own training phase and then drop, sometimes
    to zero, once subsequent tasks are introduced."""
    pb_dir = _paper_budget_dir()
    if pb_dir is None:
        print("skip fig_paper_budget_curves (no GPU paper-budget run)")
        return
    df = load_run(pb_dir)
    fig, axes = plt.subplots(1, N_TASKS, figsize=(13, 2.8), sharey=True)
    test_cols = [
        "test/stochastic/0/chainsaw-default/success",
        "test/stochastic/1/raise_the_roof-default/success",
        "test/stochastic/2/run_and_gun-default/success",
        "test/stochastic/3/health_gathering-default/success",
    ]
    x = np.arange(1, len(df) + 1)
    for t, (ax, name) in enumerate(zip(axes, CO4_TASKS)):
        if test_cols[t] in df.columns:
            y = df[test_cols[t]].to_numpy()
            ax.plot(x, smooth(y, 5), color="#1f77b4",
                    label="Held-out", lw=1.6)
        # Training-time success only meaningful when active_env == t
        ts = df["train/success"].where(df["train/active_env"] == t).to_numpy()
        ax.plot(x, smooth(ts, 5), color="#888888",
                label="Training-time", lw=1.2, linestyle="--", alpha=0.7)
        for tt in range(1, N_TASKS):
            ax.axvline(tt * (len(df) // N_TASKS), color="k",
                       alpha=0.25, lw=0.7, ls=":")
        ax.set_title(f"Task {t}: {name}", fontsize=10)
        ax.set_xlabel("Training epoch")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Success rate")
    axes[-1].legend(loc="upper right", fontsize=8, frameon=False)
    fig.suptitle("Paper-budget EWC (200k steps/task, held-out eval)",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "fig9_paper_budget.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT / 'fig9_paper_budget.pdf'}")


def make_table_ewc_online() -> None:
    """Head-to-head summary table: original EWC vs Online EWC.
    Includes the unregularised Fine-Tuning control as the reference
    point that both regularised variants are trying to recover."""
    if not _ewc_online_complete():
        print("skip tab6_ewc_online (Online EWC not finished yet)")
        return
    methods = ["FT", "EWC", "EWC-Online"]
    rows = []
    for method in methods:
        seed_runs = stack_method(RUNS[method])
        last5 = np.stack([last_k_per_task(df, "train/success", 5)
                          for df in seed_runs.values()])
        per_task_mean = np.nanmean(last5, axis=0)
        per_task_std = np.nanstd(last5, axis=0) if last5.shape[0] > 1 else None
        rows.append((method, per_task_mean, per_task_std,
                     per_task_mean.mean(), last5.shape[0]))
    lines = [
        r"\begin{tabular}{lcccc|c}",
        r"\toprule",
        r"Method & " + " & ".join(f"T{i}" for i in range(N_TASKS)) +
        r" & \textbf{Avg.\ Perf.} \\",
        r"\midrule",
    ]
    for method, m, s, ap, n in rows:
        cells = []
        for i in range(N_TASKS):
            if s is not None:
                cells.append(f"{m[i]:.3f} $\\pm$ {s[i]:.3f}")
            else:
                cells.append(f"{m[i]:.3f}")
        lines.append(f"{method} ({n} seed{'s' if n > 1 else ''}) & " +
                     " & ".join(cells) + f" & \\textbf{{{ap:.3f}}} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT.parent / "tab6_ewc_online.tex").write_text("\n".join(lines))
    print("wrote tab6_ewc_online.tex")


# Now that `load_run` is defined, gate Online EWC into RUNS if a complete
# run is on disk. Order matters: do this *after* every helper function this
# block depends on has been bound.
_ewc_online_dirs = _existing_dirs(LOG_DIR / "ewc_online_co4_20k_seed1")
if _ewc_online_dirs:
    RUNS["EWC-Online"] = _ewc_online_dirs


if __name__ == "__main__":
    fig_curves("train/success", "Success rate", "fig1_success_curves.pdf")
    fig_curves("train/return/avg", "Episode return", "fig2_return_curves.pdf")
    fig_loss_reg()
    fig_ewc_seed_band()
    fig_training_dynamics()
    fig_action_distribution()
    make_table_avg_perf()
    make_table_forgetting()
    make_table_learning_speed()
    make_table_walltime()
    make_table_domain_metrics()
    fig_action_entropy()
    fig_ewc_vs_online()
    make_table_ewc_online()
    make_table_paper_budget()
    fig_paper_budget_curves()
    make_table_unified()
    fig_gamma_sweep()
    print("done.")
