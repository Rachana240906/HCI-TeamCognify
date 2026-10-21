"""
evaluate.py
===========
Phases 7–8 – Evaluation: in-distribution, OOD, zero-shot, and robustness.

Loads trained checkpoints, evaluates on test split, computes all metrics,
saves CSV and JSON results.

Usage
-----
    python src/evaluate.py --config configs/config.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Core evaluation
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_model(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    is_brain_guided: bool = False,
) -> dict:
    """Run inference on a DataLoader and return full metric dict."""
    model.eval()
    all_logits, all_labels, all_projs, all_brains = [], [], [], []

    with torch.no_grad():
        for batch in loader:
            if is_brain_guided:
                x, b, y = batch
                x, b, y = x.to(device), b.to(device), y.to(device)
            else:
                x, y = batch
                b = None
                x, y = x.to(device), y.to(device)

            logits, proj = model(x)
            all_logits.append(logits.cpu())
            all_labels.append(y.cpu())
            if proj is not None:
                all_projs.append(proj.cpu())
            if b is not None:
                all_brains.append(b.cpu())

    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    preds  = logits.argmax(dim=-1).numpy()
    y_true = labels.numpy()

    n_classes = logits.shape[1]
    results = {
        "accuracy":          float(accuracy_score(y_true, preds)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, preds)),
        "macro_f1":          float(f1_score(y_true, preds, average="macro", zero_division=0)),
        "confusion_matrix":  confusion_matrix(y_true, preds, labels=list(range(n_classes))).tolist(),
        "n_samples":         int(len(y_true)),
        "class_counts":      {str(c): int((y_true == c).sum()) for c in range(n_classes)},
    }

    # Per-class accuracy
    for c in range(n_classes):
        mask = y_true == c
        if mask.sum() > 0:
            results[f"acc_class_{c}"] = float((preds[mask] == c).mean())

    # Representation similarity (if proj and brain available)
    if all_projs and all_brains:
        from alignment import cka_linear, rsa_correlation
        proj_np  = torch.cat(all_projs).numpy()
        brain_np = torch.cat(all_brains).numpy()
        rsa_rho, rsa_p = rsa_correlation(proj_np, brain_np)
        cka = cka_linear(proj_np, brain_np)
        results["rsa_rho"]   = rsa_rho
        results["rsa_pvalue"] = rsa_p
        results["cka_linear"] = cka

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Noise robustness
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_noise_robustness(
    model: torch.nn.Module,
    dataset,
    device: torch.device,
    noise_levels: list[float],
    batch_size: int = 16,
) -> list[dict]:
    """
    Evaluate EEG classification accuracy at multiple Gaussian noise levels.

    For each sigma, add N(0, sigma^2) to the EEG input and evaluate.
    Sigma is expressed as a fraction of the per-channel std of the data.

    Parameters
    ----------
    noise_levels : list of sigma values (multiples of data std)

    Returns
    -------
    list of dicts with noise_sigma, accuracy, balanced_accuracy
    """
    # Compute data std for scaling
    X = dataset.X.numpy()
    data_std = X.std()

    results = []
    for sigma in noise_levels:
        noisy_ds = type(dataset).__new__(type(dataset))
        noisy_ds.__dict__.update(dataset.__dict__)
        noise = torch.randn_like(dataset.X) * sigma * data_std
        noisy_ds.X = dataset.X + noise

        loader = DataLoader(noisy_ds, batch_size=batch_size, shuffle=False)
        metrics = evaluate_model(model, loader, device,
                                 is_brain_guided=hasattr(dataset, "B"))
        results.append({
            "noise_sigma": sigma,
            "noise_std_abs": float(sigma * data_std),
            "accuracy":          metrics["accuracy"],
            "balanced_accuracy": metrics["balanced_accuracy"],
        })
    return results


# ──────────────────────────────────────────────────────────────────────────────
# Main evaluation loop
# ──────────────────────────────────────────────────────────────────────────────

CONDITIONS = ["baseline", "real_brain", "synthetic", "shuffled", "random"]


def main(args: argparse.Namespace) -> int:
    project_root = Path(__file__).parent.parent

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = project_root / cfg_path
    with open(cfg_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    processed_dir = Path(cfg["paths"]["processed_dir"])
    if not processed_dir.is_absolute():
        processed_dir = project_root / processed_dir

    ckpt_dir = Path(cfg["paths"]["checkpoints_dir"])
    if not ckpt_dir.is_absolute():
        ckpt_dir = project_root / ckpt_dir

    metrics_dir = Path(cfg["paths"]["results_dir"]) / "metrics"
    if not metrics_dir.is_absolute():
        metrics_dir = project_root / metrics_dir
    metrics_dir.mkdir(parents=True, exist_ok=True)

    reports_dir = Path(cfg["paths"]["reports_dir"])
    if not reports_dir.is_absolute():
        reports_dir = project_root / reports_dir

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    smoke_test = args.smoke_test or cfg["smoke_test"]["enabled"]
    seeds = [42] if smoke_test else cfg["training"]["seeds"]
    alpha_values = cfg["brain_guidance"]["alpha_values"]
    if smoke_test:
        alpha_values = [alpha_values[0]]

    # Load training results to know which checkpoints exist
    tr_results_path = metrics_dir / "training_results.json"
    if not tr_results_path.exists():
        log.error("Training results not found at %s. Run train.py first.", tr_results_path)
        return 1
    with open(tr_results_path, "r") as fh:
        training_results = json.load(fh)

    noise_levels = [0.0, 0.5, 1.0, 2.0, 4.0]

    all_eval_rows = []

    for tr in training_results:
        condition = tr["condition"]
        alpha     = tr["alpha"]
        seed      = tr["seed"]
        ckpt_path = Path(tr["checkpoint"])

        if not ckpt_path.exists():
            log.warning("Checkpoint not found: %s", ckpt_path)
            continue

        log.info("Evaluating %s alpha=%.2f seed=%d ...", condition, alpha, seed)

        # Reload model from checkpoint
        from dataset import load_split
        from model import build_model

        is_brain = condition != "baseline"
        brain_override_map = {
            "synthetic": "brain_ctrl_synthetic",
            "shuffled":  "brain_ctrl_shuffled",
            "random":    "brain_ctrl_random",
        }

        def get_brain_path(split: str) -> Path | None:
            if not is_brain:
                return None
            if condition == "real_brain":
                return processed_dir / f"brain_{split}.npz"
            prefix = brain_override_map[condition]
            return processed_dir / f"{prefix}_{split}.npz"

        test_ds = load_split(
            processed_dir, "test",
            brain_guided=is_brain,
            brain_npz_override=get_brain_path("test"),
            stride=4,
        )

        if smoke_test:
            mt = cfg["smoke_test"]["max_trials"]
            test_ds.X = test_ds.X[:mt]
            test_ds.y = test_ds.y[:mt]
            if is_brain:
                test_ds.B = test_ds.B[:mt]

        brain_dim = test_ds.brain_dim if is_brain else None
        cfg["model"]["input_size"] = test_ds.input_size

        model = build_model(cfg, n_classes=3, brain_dim=brain_dim).to(device)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state"])

        test_loader = DataLoader(test_ds, batch_size=cfg["training"]["batch_size"],
                                  shuffle=False)

        # ── In-distribution (test split = held-out sessions) ──────────────────
        id_metrics = evaluate_model(model, test_loader, device, is_brain)
        row = {
            "condition": condition, "alpha": alpha, "seed": seed,
            "split": "test_id",
            **{f"id_{k}": v for k, v in id_metrics.items()
               if isinstance(v, (int, float))},
        }

        # ── Per-subject breakdown ─────────────────────────────────────────────
        test_data = np.load(processed_dir / "eeg_test.npz", allow_pickle=True)
        subjects  = test_data["subject"][:len(test_ds)]  # respect smoke-test truncation

        for sub in np.unique(subjects):
            mask = subjects == sub
            sub_x = test_ds.X[mask]
            sub_y = test_ds.y[mask]
            sub_b = test_ds.B[mask] if is_brain else None

            if len(sub_x) == 0:
                continue

            class TmpDS(torch.utils.data.Dataset):
                def __init__(self, x, y, b=None):
                    self.X, self.y, self.B = x, y, b
                def __len__(self): return len(self.y)
                def __getitem__(self, i):
                    return (self.X[i], self.B[i], self.y[i]) if self.B is not None \
                        else (self.X[i], self.y[i])

            sub_loader = DataLoader(
                TmpDS(sub_x, sub_y, sub_b),
                batch_size=cfg["training"]["batch_size"]
            )
            sub_m = evaluate_model(model, sub_loader, device, is_brain)
            row[f"sub_{sub}_acc"]     = sub_m["accuracy"]
            row[f"sub_{sub}_bal_acc"] = sub_m["balanced_accuracy"]

        # ── Noise robustness ──────────────────────────────────────────────────
        noise_results = evaluate_noise_robustness(
            model, test_ds, device, noise_levels, cfg["training"]["batch_size"]
        )
        for nr in noise_results:
            sigma_key = f"noise_sigma_{nr['noise_sigma']:.1f}"
            row[f"{sigma_key}_acc"]     = nr["accuracy"]
            row[f"{sigma_key}_bal_acc"] = nr["balanced_accuracy"]

        all_eval_rows.append(row)
        log.info("  → ID acc=%.3f  bal=%.3f  rsa=%s  cka=%s",
                 id_metrics["accuracy"],
                 id_metrics["balanced_accuracy"],
                 f"{id_metrics.get('rsa_rho', float('nan')):.3f}",
                 f"{id_metrics.get('cka_linear', float('nan')):.3f}")

    # Save results
    eval_df = pd.DataFrame(all_eval_rows)
    csv_path = metrics_dir / "evaluation_results.csv"
    eval_df.to_csv(csv_path, index=False)
    log.info("Evaluation results → %s", csv_path)

    # JSON version (includes confusion matrices)
    json_path = metrics_dir / "evaluation_results_full.json"
    # Rebuild with confusion matrices
    full_results = []
    for row in all_eval_rows:
        full_results.append(row)
    with open(json_path, "w") as fh:
        json.dump(full_results, fh, indent=2, default=str)

    # Print summary
    log.info("=" * 80)
    log.info("EVALUATION SUMMARY  (test set)")
    log.info("%-20s %-7s %-6s  %-10s %-12s %-10s %-10s",
             "Condition", "Alpha", "Seed", "ID_Acc", "ID_BalAcc", "RSA_rho", "CKA")
    for row in all_eval_rows:
        log.info("%-20s %-7.2f %-6d  %-10.3f %-12.3f %-10s %-10s",
                 row["condition"], row["alpha"], row["seed"],
                 row.get("id_accuracy", float("nan")),
                 row.get("id_balanced_accuracy", float("nan")),
                 f"{row.get('id_rsa_rho', float('nan')):.3f}",
                 f"{row.get('id_cka_linear', float('nan')):.3f}")
    log.info("=" * 80)

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",    default="configs/config.yaml")
    parser.add_argument("--smoke_test", action="store_true")
    sys.exit(main(parser.parse_args()))
