"""
train.py
========
Training loop for CD-B2M (Phases 4–5).

Supports five experimental conditions:
  A. baseline          — task loss only
  B. real_brain        — task + alignment to real iEEG brain reps
  C. synthetic_brain   — task + alignment to synthetic matched control
  D. shuffled_brain    — task + alignment to shuffled brain reps
  E. random_brain      — task + alignment to Gaussian noise

Usage
-----
    # Run ALL conditions and seeds
    python src/train.py --config configs/config.yaml

    # Run one condition
    python src/train.py --config configs/config.yaml --condition baseline --seed 42

    # Smoke test
    python src/train.py --config configs/config.yaml --smoke_test
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ──────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        log.info("Device: CUDA (%s)", torch.cuda.get_device_name(0))
    else:
        dev = torch.device("cpu")
        log.info("Device: CPU")
    return dev


# ──────────────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────────────

def compute_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    n_classes: int,
) -> dict[str, float]:
    """Compute accuracy, balanced accuracy, and per-class counts."""
    from sklearn.metrics import balanced_accuracy_score, f1_score

    preds  = logits.argmax(dim=-1).cpu().numpy()
    y_true = labels.cpu().numpy()

    acc     = float((preds == y_true).mean())
    bal_acc = float(balanced_accuracy_score(y_true, preds))
    macro_f1 = float(f1_score(y_true, preds, average="macro", zero_division=0))

    return {
        "accuracy":           acc,
        "balanced_accuracy":  bal_acc,
        "macro_f1":           macro_f1,
    }


# ──────────────────────────────────────────────────────────────────────────────
# One epoch
# ──────────────────────────────────────────────────────────────────────────────

def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimiser: torch.optim.Optimizer | None,
    alpha: float,
    loss_type: str,
    is_brain_guided: bool,
    grad_clip: float = 1.0,
    class_weight: torch.Tensor | None = None,
) -> dict[str, float]:
    """Run one training or validation epoch. Returns metrics dict."""
    from alignment import combined_loss

    training = optimiser is not None
    model.train(training)

    total_loss = total_task = total_brain = 0.0
    all_logits, all_labels = [], []
    n_batches = 0

    with torch.set_grad_enabled(training):
        for batch in loader:
            if is_brain_guided:
                x, b, y = batch
                x, b, y = x.to(device), b.to(device), y.to(device)
            else:
                x, y = batch
                b = None
                x, y = x.to(device), y.to(device)

            logits, proj = model(x)
            # Move class_weight to the correct device each call
            cw = class_weight.to(device) if class_weight is not None else None
            loss, t_loss, b_loss = combined_loss(
                logits, y, proj, b, alpha=alpha, loss_type=loss_type,
                class_weight=cw,
            )

            if training:
                optimiser.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimiser.step()

            total_loss  += loss.item()
            total_task  += t_loss.item()
            total_brain += b_loss.item()
            all_logits.append(logits.detach().cpu())
            all_labels.append(y.detach().cpu())
            n_batches += 1

    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)
    n_cls = all_logits.shape[1]
    metrics = compute_metrics(all_logits, all_labels, n_cls)
    metrics["loss"]       = total_loss  / max(n_batches, 1)
    metrics["task_loss"]  = total_task  / max(n_batches, 1)
    metrics["brain_loss"] = total_brain / max(n_batches, 1)
    return metrics


# ──────────────────────────────────────────────────────────────────────────────
# Full training run for one (condition, alpha, seed)
# ──────────────────────────────────────────────────────────────────────────────

def train_one(
    cfg: dict,
    condition: str,
    alpha: float,
    seed: int,
    device: torch.device,
    processed_dir: Path,
    ckpt_dir: Path,
    smoke_test: bool = False,
) -> dict:
    """
    Train one model and return a results dict.

    Parameters
    ----------
    condition : 'baseline' | 'real_brain' | 'synthetic' | 'shuffled' | 'random'
    alpha     : brain alignment weight (ignored for 'baseline')
    seed      : random seed
    """
    from dataset import load_split
    from model import build_model

    set_seed(seed)

    tr_cfg = cfg["training"]
    m_cfg  = cfg["model"]
    bg_cfg = cfg["brain_guidance"]

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

    # Load datasets (stride=4 → 256 timesteps, 4× faster GRU)
    train_ds = load_split(
        processed_dir, "train",
        brain_guided=is_brain,
        brain_npz_override=get_brain_path("train"),
        stride=4,
    )
    val_ds = load_split(
        processed_dir, "val",
        brain_guided=is_brain,
        brain_npz_override=get_brain_path("val"),
        stride=4,
    )

    # Smoke-test truncation
    if smoke_test:
        mt = cfg["smoke_test"]["max_trials"]
        train_ds.X = train_ds.X[:mt]
        train_ds.y = train_ds.y[:mt]
        val_ds.X   = val_ds.X[:mt]
        val_ds.y   = val_ds.y[:mt]
        if is_brain:
            train_ds.B = train_ds.B[:mt]
            val_ds.B   = val_ds.B[:mt]

    train_loader = DataLoader(
        train_ds, batch_size=tr_cfg["batch_size"],
        shuffle=True,  num_workers=tr_cfg["num_workers"], drop_last=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=tr_cfg["batch_size"],
        shuffle=False, num_workers=tr_cfg["num_workers"]
    )

    # Build model
    n_classes  = 3
    brain_dim  = train_ds.brain_dim if is_brain else None
    cfg["model"]["input_size"] = train_ds.input_size

    model = build_model(cfg, n_classes=n_classes, brain_dim=brain_dim).to(device)

    # ── Class weights (inverse-frequency) to prevent majority-class collapse ──
    y_train_np = train_ds.y.numpy()
    n_total    = len(y_train_np)
    class_weight = torch.tensor(
        [n_total / (n_classes * float((y_train_np == c).sum())) for c in range(n_classes)],
        dtype=torch.float32,
    )
    log.info("  Class weights: %s", [f"{w:.3f}" for w in class_weight.tolist()])

    optimiser = torch.optim.Adam(
        model.parameters(),
        lr=float(tr_cfg["lr"]), weight_decay=float(tr_cfg["weight_decay"])
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="min", patience=7, factor=0.5
    )

    patience     = tr_cfg["patience"]
    max_epochs   = cfg["smoke_test"]["max_epochs"] if smoke_test else tr_cfg["max_epochs"]
    grad_clip    = tr_cfg["grad_clip"]
    loss_type    = bg_cfg["loss_type"]

    best_val_loss = float("inf")
    best_epoch    = 0
    no_improve    = 0
    history       = []

    ckpt_name = f"{condition}_alpha{alpha:.2f}_seed{seed}.pt"
    ckpt_path = ckpt_dir / ckpt_name

    t0 = time.time()

    for epoch in range(1, max_epochs + 1):
        tr_m  = run_epoch(model, train_loader, device, optimiser,
                          alpha, loss_type, is_brain, grad_clip,
                          class_weight=class_weight)
        val_m = run_epoch(model, val_loader, device, None,
                          alpha, loss_type, is_brain,
                          class_weight=class_weight)

        scheduler.step(val_m["loss"])

        history.append({
            "epoch": epoch,
            "train_loss": tr_m["loss"], "train_acc": tr_m["accuracy"],
            "val_loss":   val_m["loss"], "val_acc":   val_m["accuracy"],
            "val_bal_acc": val_m["balanced_accuracy"],
        })

        if epoch % 10 == 0 or epoch <= 3:
            log.info(
                "  [%s a=%.2f s=%d] E%03d  tr_loss=%.4f tr_acc=%.3f  "
                "val_loss=%.4f val_acc=%.3f val_bal=%.3f",
                condition, alpha, seed, epoch,
                tr_m["loss"], tr_m["accuracy"],
                val_m["loss"], val_m["accuracy"], val_m["balanced_accuracy"]
            )

        if val_m["loss"] < best_val_loss - 1e-5:
            best_val_loss = val_m["loss"]
            best_epoch    = epoch
            no_improve    = 0
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "val_metrics": val_m,
                "condition": condition,
                "alpha": alpha,
                "seed": seed,
            }, ckpt_path)
        else:
            no_improve += 1
            if no_improve >= patience:
                log.info("  Early stopping at epoch %d (best=%d)", epoch, best_epoch)
                break

    elapsed = time.time() - t0

    # Load best checkpoint for final evaluation
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state"])

    return {
        "condition": condition,
        "alpha": alpha,
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "final_val_metrics": ckpt["val_metrics"],
        "history": history,
        "elapsed_s": round(elapsed, 1),
        "checkpoint": str(ckpt_path),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

CONDITIONS = ["baseline", "real_brain", "synthetic", "shuffled", "random"]


def main(args: argparse.Namespace) -> int:
    project_root = Path(__file__).parent.parent

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = project_root / cfg_path
    with open(cfg_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    smoke_test = args.smoke_test or cfg["smoke_test"]["enabled"]
    if smoke_test:
        log.info("=== SMOKE-TEST MODE ===")

    processed_dir = Path(cfg["paths"]["processed_dir"])
    if not processed_dir.is_absolute():
        processed_dir = project_root / processed_dir

    ckpt_dir = Path(cfg["paths"]["checkpoints_dir"])
    if not ckpt_dir.is_absolute():
        ckpt_dir = project_root / ckpt_dir
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    reports_dir = Path(cfg["paths"]["reports_dir"])
    if not reports_dir.is_absolute():
        reports_dir = project_root / reports_dir
    reports_dir.mkdir(parents=True, exist_ok=True)

    metrics_dir = Path(cfg["paths"]["results_dir"]) / "metrics"
    if not metrics_dir.is_absolute():
        metrics_dir = project_root / metrics_dir
    metrics_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()

    # Which conditions / alpha values / seeds to run
    conditions = [args.condition] if args.condition else CONDITIONS
    seeds = [args.seed] if args.seed else (
        [42] if smoke_test else cfg["training"]["seeds"]
    )
    alpha_values = cfg["brain_guidance"]["alpha_values"]
    if smoke_test:
        alpha_values = [alpha_values[0]]  # only one alpha in smoke test

    all_results = []

    for condition in conditions:
        alphas = [0.0] if condition == "baseline" else alpha_values
        for alpha in alphas:
            for seed in seeds:
                log.info("=" * 55)
                log.info("Condition: %s  alpha=%.2f  seed=%d", condition, alpha, seed)
                result = train_one(
                    cfg, condition, alpha, seed, device,
                    processed_dir, ckpt_dir, smoke_test=smoke_test
                )
                all_results.append(result)
                log.info(
                    "  → val acc=%.3f  val_bal=%.3f  (epoch %d, %.1f s)",
                    result["final_val_metrics"]["accuracy"],
                    result["final_val_metrics"]["balanced_accuracy"],
                    result["best_epoch"],
                    result["elapsed_s"],
                )

    # Save training results
    results_path = metrics_dir / "training_results.json"
    with open(results_path, "w", encoding="utf-8") as fh:
        # history list can be large; store it
        json.dump(all_results, fh, indent=2)
    log.info("Training results → %s", results_path)

    # Summary table
    log.info("=" * 70)
    log.info("TRAINING SUMMARY")
    log.info("%-20s %-7s %-6s  %-8s %-10s %-10s", "Condition", "Alpha", "Seed",
             "Epochs", "Val_Acc", "Val_BalAcc")
    for r in all_results:
        log.info("%-20s %-7.2f %-6d  %-8d %-10.3f %-10.3f",
                 r["condition"], r["alpha"], r["seed"],
                 r["best_epoch"],
                 r["final_val_metrics"]["accuracy"],
                 r["final_val_metrics"]["balanced_accuracy"])
    log.info("=" * 70)

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",    default="configs/config.yaml")
    parser.add_argument("--condition", default=None, choices=CONDITIONS + [None])
    parser.add_argument("--seed",      default=None, type=int)
    parser.add_argument("--smoke_test", action="store_true")
    sys.exit(main(parser.parse_args()))
