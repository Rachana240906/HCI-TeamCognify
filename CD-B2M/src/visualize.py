"""
visualize.py
============
Phase 9 – Publication-quality plots for CD-B2M.

Generates all figures and saves them + underlying CSV data to results/plots/.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import yaml

CONDITION_COLORS = {
    "baseline":    "#555555",
    "real_brain":  "#2196F3",
    "synthetic":   "#FF9800",
    "shuffled":    "#9C27B0",
    "random":      "#F44336",
}
CONDITION_LABELS = {
    "baseline":    "A. Baseline",
    "real_brain":  "B. Real Brain",
    "synthetic":   "C. Synthetic (matched)",
    "shuffled":    "D. Shuffled Brain",
    "random":      "E. Random",
}


def load_data(cfg: dict, project_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    metrics_dir = Path(cfg["paths"]["results_dir"]) / "metrics"
    if not metrics_dir.is_absolute():
        metrics_dir = project_root / metrics_dir

    eval_csv  = metrics_dir / "evaluation_results.csv"
    stats_csv = metrics_dir / "stats_summary.csv"
    train_json = metrics_dir / "training_results.json"

    eval_df  = pd.read_csv(eval_csv)  if eval_csv.exists()  else pd.DataFrame()
    stats_df = pd.read_csv(stats_csv) if stats_csv.exists() else pd.DataFrame()

    import json
    training = json.load(open(train_json)) if train_json.exists() else []

    return eval_df, stats_df, training


def save_fig(fig: plt.Figure, path: Path, data: pd.DataFrame | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path.name}")
    if data is not None:
        data.to_csv(path.with_suffix(".csv"), index=False)


# ─── Fig 1: Model performance comparison (bar chart with error bars) ──────────
def fig_performance_comparison(eval_df: pd.DataFrame, plots_dir: Path) -> None:
    if eval_df.empty:
        return
    conditions = [c for c in CONDITION_LABELS if c in eval_df["condition"].values]
    means, stds, labels, colors = [], [], [], []
    rows = []
    for cond in conditions:
        grp = eval_df[eval_df["condition"] == cond]
        # Use best alpha (highest mean bal acc across seeds)
        best_alpha = (grp.groupby("alpha")["id_balanced_accuracy"]
                      .mean().idxmax() if "alpha" in grp.columns else 0.0)
        sub = grp[grp["alpha"] == best_alpha]["id_balanced_accuracy"]
        m, s = float(sub.mean()), float(sub.std())
        means.append(m); stds.append(s if not np.isnan(s) else 0)
        labels.append(CONDITION_LABELS[cond])
        colors.append(CONDITION_COLORS[cond])
        rows.append({"condition": cond, "best_alpha": best_alpha,
                     "bal_acc_mean": m, "bal_acc_std": s})

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(conditions))
    bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors, alpha=0.85,
                  error_kw={"elinewidth": 1.5})
    ax.axhline(1/3, color="gray", ls="--", lw=1.2, label="Chance (33.3%)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=10)
    ax.set_ylabel("Balanced Accuracy", fontsize=12)
    ax.set_title("CD-B2M: Model Performance Comparison\n"
                 "(SetSize 3-class, test set, mean ± SD across seeds)", fontsize=12)
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{m:.3f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    save_fig(fig, plots_dir / "fig1_performance_comparison.png", pd.DataFrame(rows))


# ─── Fig 2: Learning curves ────────────────────────────────────────────────────
def fig_learning_curves(training: list, plots_dir: Path) -> None:
    if not training:
        return
    # One seed per condition for clarity; pick seed=42
    cond_data = {}
    for run in training:
        cond = run["condition"]
        if run["seed"] == 42 and cond not in cond_data:
            cond_data[cond] = run["history"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    rows = []
    for cond, hist in cond_data.items():
        epochs   = [h["epoch"]     for h in hist]
        tr_loss  = [h["train_loss"] for h in hist]
        val_loss = [h["val_loss"]   for h in hist]
        val_acc  = [h["val_acc"]    for h in hist]
        color    = CONDITION_COLORS.get(cond, "black")
        label    = CONDITION_LABELS.get(cond, cond)
        axes[0].plot(epochs, val_loss, color=color, label=label, lw=1.8)
        axes[1].plot(epochs, val_acc,  color=color, label=label, lw=1.8)
        for e, tl, vl, va in zip(epochs, tr_loss, val_loss, val_acc):
            rows.append({"condition": cond, "epoch": e,
                         "train_loss": tl, "val_loss": vl, "val_acc": va})

    for ax, title, ylabel in zip(
        axes,
        ["Validation Loss", "Validation Accuracy"],
        ["Cross-entropy loss", "Accuracy"]
    ):
        ax.set_xlabel("Epoch"); ax.set_ylabel(ylabel)
        ax.set_title(title); ax.legend(fontsize=9); ax.grid(alpha=0.3)
    axes[1].axhline(1/3, color="gray", ls="--", lw=1, label="Chance")
    axes[1].set_ylim(0, 1)
    fig.suptitle("CD-B2M: Learning Curves (seed=42)", fontsize=13)
    fig.tight_layout()
    save_fig(fig, plots_dir / "fig2_learning_curves.png", pd.DataFrame(rows))


# ─── Fig 3: Noise robustness ───────────────────────────────────────────────────
def fig_noise_robustness(eval_df: pd.DataFrame, plots_dir: Path) -> None:
    if eval_df.empty:
        return
    noise_cols = [c for c in eval_df.columns if c.startswith("noise_sigma_") and c.endswith("_bal_acc")]
    if not noise_cols:
        return

    sigmas = sorted(set(
        float(c.replace("noise_sigma_", "").replace("_bal_acc", "")) for c in noise_cols
    ))

    fig, ax = plt.subplots(figsize=(8, 5))
    rows = []
    for cond in CONDITION_LABELS:
        grp = eval_df[eval_df["condition"] == cond]
        if grp.empty:
            continue
        ys, errs = [], []
        for sig in sigmas:
            col = f"noise_sigma_{sig:.1f}_bal_acc"
            if col in grp.columns:
                vals = grp[col].dropna()
                ys.append(float(vals.mean()))
                errs.append(float(vals.std()) if len(vals) > 1 else 0)
                rows.append({"condition": cond, "noise_sigma": sig,
                             "bal_acc_mean": ys[-1], "bal_acc_std": errs[-1]})
            else:
                ys.append(np.nan); errs.append(0)
        ax.errorbar(sigmas, ys, yerr=errs,
                    color=CONDITION_COLORS[cond],
                    label=CONDITION_LABELS[cond], marker="o", lw=1.8, capsize=4)

    ax.axhline(1/3, color="gray", ls="--", lw=1, label="Chance")
    ax.set_xlabel("Noise level σ (× data std)", fontsize=12)
    ax.set_ylabel("Balanced Accuracy", fontsize=12)
    ax.set_title("CD-B2M: Noise Robustness", fontsize=12)
    ax.legend(fontsize=9); ax.grid(alpha=0.3); ax.set_ylim(0, 1)
    fig.tight_layout()
    save_fig(fig, plots_dir / "fig3_noise_robustness.png", pd.DataFrame(rows))


# ─── Fig 4: Confusion matrices ─────────────────────────────────────────────────
def fig_confusion_matrices(metrics_dir: Path, plots_dir: Path) -> None:
    import json
    full_json = metrics_dir / "evaluation_results_full.json"
    if not full_json.exists():
        return
    with open(full_json) as fh:
        full = json.load(fh)

    conditions = list(CONDITION_LABELS.keys())
    # One entry per condition (best alpha, seed=42)
    selected = {}
    for row in full:
        cond = row["condition"]
        if row["seed"] == 42 and cond not in selected:
            selected[cond] = row

    n_cond = len([c for c in conditions if c in selected])
    if n_cond == 0:
        return

    fig, axes = plt.subplots(1, n_cond, figsize=(4 * n_cond, 4))
    if n_cond == 1:
        axes = [axes]

    class_labels = ["SS=4", "SS=6", "SS=8"]
    for ax, cond in zip(axes, [c for c in conditions if c in selected]):
        row = selected[cond]
        cm_key = "id_confusion_matrix"
        # confusion matrix may be stored as a list
        cm = None
        for k, v in row.items():
            if "confusion_matrix" in k and isinstance(v, list):
                cm = np.array(v)
                break
        if cm is None:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center")
            ax.set_title(CONDITION_LABELS[cond], fontsize=9)
            continue
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(3)); ax.set_yticks(range(3))
        ax.set_xticklabels(class_labels, fontsize=9)
        ax.set_yticklabels(class_labels, fontsize=9)
        ax.set_xlabel("Predicted", fontsize=9)
        ax.set_ylabel("True", fontsize=9)
        ax.set_title(CONDITION_LABELS[cond], fontsize=9)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() * 0.6 else "black",
                        fontsize=10, fontweight="bold")

    fig.suptitle("CD-B2M: Confusion Matrices (test set, seed=42)", fontsize=12)
    fig.tight_layout()
    save_fig(fig, plots_dir / "fig4_confusion_matrices.png")


# ─── Fig 5: Brain vs Synthetic rep statistics ─────────────────────────────────
def fig_rep_statistics(processed_dir: Path, plots_dir: Path) -> None:
    brain   = np.load(processed_dir / "brain_train.npz")["X"]
    syn     = np.load(processed_dir / "brain_ctrl_synthetic_train.npz")["X"]
    shuf    = np.load(processed_dir / "brain_ctrl_shuffled_train.npz")["X"]
    rand    = np.load(processed_dir / "brain_ctrl_random_train.npz")["X"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Feature means
    axes[0].plot(brain.mean(axis=0), label="Real Brain",  color=CONDITION_COLORS["real_brain"], lw=1.5)
    axes[0].plot(syn.mean(axis=0),   label="Synthetic",   color=CONDITION_COLORS["synthetic"],  lw=1.5, ls="--")
    axes[0].plot(shuf.mean(axis=0),  label="Shuffled",    color=CONDITION_COLORS["shuffled"],   lw=1.5, ls=":")
    axes[0].plot(rand.mean(axis=0),  label="Random",      color=CONDITION_COLORS["random"],     lw=1.2, ls="-.")
    axes[0].set_title("Per-feature Mean"); axes[0].set_xlabel("Feature index")
    axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)

    # Feature std
    axes[1].plot(brain.std(axis=0), color=CONDITION_COLORS["real_brain"], lw=1.5, label="Real Brain")
    axes[1].plot(syn.std(axis=0),   color=CONDITION_COLORS["synthetic"],  lw=1.5, ls="--", label="Synthetic")
    axes[1].plot(shuf.std(axis=0),  color=CONDITION_COLORS["shuffled"],   lw=1.5, ls=":",  label="Shuffled")
    axes[1].plot(rand.std(axis=0),  color=CONDITION_COLORS["random"],     lw=1.2, ls="-.", label="Random")
    axes[1].set_title("Per-feature Std"); axes[1].set_xlabel("Feature index")
    axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)

    # Distribution overlay (first feature as example)
    bins = np.linspace(brain[:, 0].min() - 1, brain[:, 0].max() + 1, 30)
    axes[2].hist(brain[:, 0], bins=bins, alpha=0.5, color=CONDITION_COLORS["real_brain"], label="Real Brain", density=True)
    axes[2].hist(syn[:, 0],   bins=bins, alpha=0.5, color=CONDITION_COLORS["synthetic"],  label="Synthetic",  density=True)
    axes[2].set_title("Feature[0] Distribution (train)"); axes[2].set_xlabel("Value")
    axes[2].legend(fontsize=8); axes[2].grid(alpha=0.3)

    fig.suptitle("CD-B2M: Brain Representation Statistics", fontsize=12)
    fig.tight_layout()
    rows = pd.DataFrame({
        "feature_idx": range(brain.shape[1]),
        "real_mean": brain.mean(axis=0), "real_std": brain.std(axis=0),
        "syn_mean":  syn.mean(axis=0),   "syn_std":  syn.std(axis=0),
    })
    save_fig(fig, plots_dir / "fig5_rep_statistics.png", rows)


# ─── Fig 6: RSA / CKA comparison ──────────────────────────────────────────────
def fig_alignment_metrics(eval_df: pd.DataFrame, plots_dir: Path) -> None:
    if eval_df.empty:
        return
    rsa_col = "id_rsa_rho"
    cka_col = "id_cka_linear"
    if rsa_col not in eval_df.columns:
        return

    brain_conds = [c for c in ["real_brain", "synthetic", "shuffled", "random"]
                   if c in eval_df["condition"].values]
    if not brain_conds:
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    rows = []
    for ax, col, title in zip(axes, [rsa_col, cka_col], ["RSA ρ", "Linear CKA"]):
        means, errs, lbls, cols = [], [], [], []
        for cond in brain_conds:
            grp = eval_df[eval_df["condition"] == cond][col].dropna()
            if grp.empty:
                continue
            m, s = float(grp.mean()), float(grp.std()) if len(grp) > 1 else 0
            means.append(m); errs.append(s)
            lbls.append(CONDITION_LABELS.get(cond, cond))
            cols.append(CONDITION_COLORS.get(cond, "gray"))
            rows.append({"condition": cond, "metric": col, "mean": m, "std": s})
        x = np.arange(len(means))
        ax.bar(x, means, yerr=errs, color=cols, alpha=0.85, capsize=4)
        ax.set_xticks(x); ax.set_xticklabels(lbls, rotation=20, ha="right", fontsize=9)
        ax.set_title(title); ax.set_ylabel(title); ax.grid(axis="y", alpha=0.3)
        if col == rsa_col:
            ax.axhline(0, color="gray", ls="--", lw=1)
    fig.suptitle("CD-B2M: EEG–Brain Representation Alignment", fontsize=12)
    fig.tight_layout()
    save_fig(fig, plots_dir / "fig6_alignment_metrics.png", pd.DataFrame(rows))


# ─── Fig 7: Per-subject performance ───────────────────────────────────────────
def fig_per_subject(eval_df: pd.DataFrame, plots_dir: Path) -> None:
    if eval_df.empty:
        return
    sub_cols = [c for c in eval_df.columns if c.startswith("sub_") and c.endswith("_bal_acc")]
    if not sub_cols:
        return

    subjects = sorted(set(c.replace("sub_", "").replace("_bal_acc", "") for c in sub_cols))
    conditions = [c for c in CONDITION_LABELS if c in eval_df["condition"].values]

    x = np.arange(len(subjects))
    width = 0.8 / len(conditions)
    fig, ax = plt.subplots(figsize=(max(6, 3 * len(subjects)), 5))
    rows = []
    for i, cond in enumerate(conditions):
        grp = eval_df[eval_df["condition"] == cond]
        vals = []
        for sub in subjects:
            col = f"sub_{sub}_bal_acc"
            if col in grp.columns:
                vals.append(float(grp[col].mean()))
            else:
                vals.append(np.nan)
            rows.append({"condition": cond, "subject": sub, "bal_acc": vals[-1]})
        offset = (i - len(conditions) / 2 + 0.5) * width
        ax.bar(x + offset, vals, width=width * 0.9,
               color=CONDITION_COLORS[cond], label=CONDITION_LABELS[cond], alpha=0.85)

    ax.set_xticks(x); ax.set_xticklabels(subjects, fontsize=11)
    ax.set_ylabel("Balanced Accuracy"); ax.set_title("CD-B2M: Per-Subject Performance")
    ax.axhline(1/3, color="gray", ls="--", lw=1, label="Chance")
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3); ax.set_ylim(0, 1)
    fig.tight_layout()
    save_fig(fig, plots_dir / "fig7_per_subject.png", pd.DataFrame(rows))


# ─── Main ──────────────────────────────────────────────────────────────────────
def main(args: argparse.Namespace) -> int:
    project_root = Path(__file__).parent.parent

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = project_root / cfg_path
    with open(cfg_path) as fh:
        cfg = yaml.safe_load(fh)

    metrics_dir = Path(cfg["paths"]["results_dir"]) / "metrics"
    if not metrics_dir.is_absolute():
        metrics_dir = project_root / metrics_dir

    plots_dir = Path(cfg["paths"]["results_dir"]) / "plots"
    if not plots_dir.is_absolute():
        plots_dir = project_root / plots_dir
    plots_dir.mkdir(parents=True, exist_ok=True)

    processed_dir = Path(cfg["paths"]["processed_dir"])
    if not processed_dir.is_absolute():
        processed_dir = project_root / processed_dir

    eval_df, stats_df, training = load_data(cfg, project_root)

    print("Generating figures ...")
    fig_performance_comparison(eval_df, plots_dir)
    fig_learning_curves(training, plots_dir)
    fig_noise_robustness(eval_df, plots_dir)
    fig_confusion_matrices(metrics_dir, plots_dir)
    fig_rep_statistics(processed_dir, plots_dir)
    fig_alignment_metrics(eval_df, plots_dir)
    fig_per_subject(eval_df, plots_dir)
    print(f"All figures saved to {plots_dir}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    sys.exit(main(parser.parse_args()))
