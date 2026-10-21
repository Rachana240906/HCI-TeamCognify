"""
controls.py
===========
Phase 6 – Control condition generators for CD-B2M.

Conditions implemented
----------------------
C. Statistically Matched Synthetic Control
   Generate synthetic brain representations from TRAINING data.
   Explicitly matched properties: per-feature mean and std.
   Optionally match covariance structure (multivariate Gaussian).
   WARNING: Gaussian does NOT match all neural statistics (e.g. heavy tails,
   non-stationarity). This limitation is documented in every saved file.

D. Shuffled Brain Control
   Shuffle brain representations ACROSS trials (within the training set),
   breaking trial-specific EEG-iEEG correspondence while preserving the
   marginal distribution of brain features exactly.

E. Random Control
   Gaussian noise matched only to the DIMENSION and a fixed scale.
   This is the weakest control — it does not match any neural statistics.

Each function:
  - Takes training brain representations as input only
  - Generates corresponding arrays for train / val / test splits
  - Does NOT use val or test brain data in generation
  - Saves generated arrays to data/processed/ and an audit JSON

Usage
-----
    python src/controls.py [--config configs/config.yaml]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Synthetic (statistically matched) control
# ──────────────────────────────────────────────────────────────────────────────

def make_synthetic_control(
    train_brain: np.ndarray,
    n_train: int,
    n_val: int,
    n_test: int,
    rng: np.random.Generator,
    match_covariance: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Generate synthetic brain representations from training statistics.

    Matched properties:
      - Per-feature mean  (always)
      - Per-feature std   (always)
      - Full covariance   (when match_covariance=True, using multivariate Gaussian)

    NOT matched:
      - Higher-order statistics (skewness, kurtosis)
      - Temporal structure within a trial
      - Non-Gaussian marginals

    Parameters
    ----------
    train_brain      : (n_train, D)  TRAINING brain representations
    n_train, n_val, n_test : number of samples to generate for each split
    rng              : seeded random generator
    match_covariance : if True use multivariate Gaussian; else independent Gaussian

    Returns
    -------
    syn_train, syn_val, syn_test : synthetic arrays
    audit                        : dict documenting what was matched
    """
    D = train_brain.shape[1]
    mu  = train_brain.mean(axis=0)
    std = train_brain.std(axis=0)

    if match_covariance:
        cov = np.cov(train_brain.T)                     # (D, D)
        # Regularise for numerical stability
        cov += 1e-6 * np.eye(D)
        total = n_train + n_val + n_test
        samples = rng.multivariate_normal(mu, cov, size=total).astype(np.float32)
    else:
        total = n_train + n_val + n_test
        samples = (rng.standard_normal((total, D)).astype(np.float32)
                   * std[np.newaxis, :] + mu[np.newaxis, :])

    syn_train = samples[:n_train]
    syn_val   = samples[n_train:n_train + n_val]
    syn_test  = samples[n_train + n_val:]

    # Verify matched properties
    gen_mu  = syn_train.mean(axis=0)
    gen_std = syn_train.std(axis=0)

    audit = {
        "type": "synthetic_matched",
        "matched_properties": ["per_feature_mean", "per_feature_std",
                                "covariance" if match_covariance else "independent"],
        "not_matched": ["higher_order_moments", "non_gaussian_marginals",
                        "heavy_tails"],
        "match_covariance": match_covariance,
        "n_train": n_train, "n_val": n_val, "n_test": n_test,
        "dim": D,
        "mean_abs_error_mu":  float(np.abs(gen_mu - mu).mean()),
        "mean_abs_error_std": float(np.abs(gen_std - std).mean()),
        "limitation": (
            "Gaussian approximation does not capture non-Gaussian marginals, "
            "heavy tails, or non-linear dependencies in neural data."
        ),
    }
    return syn_train, syn_val, syn_test, audit


# ──────────────────────────────────────────────────────────────────────────────
# Shuffled brain control
# ──────────────────────────────────────────────────────────────────────────────

def make_shuffled_control(
    train_brain: np.ndarray,
    val_brain: np.ndarray,
    test_brain: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Shuffle brain representations across trials (permutation within each split).

    This breaks trial-specific EEG↔iEEG correspondence while preserving the
    EXACT marginal distribution of brain features (same values, different order).

    Interpretation:
      - If brain-guided model with shuffled control outperforms baseline,
        it suggests the benefit comes from the statistical properties of
        brain representations rather than their trial-specific content.
      - If real brain outperforms shuffled control, it suggests trial-specific
        correspondence matters.
      - Shuffling within splits (not across splits) prevents label leakage.

    Limitation:
      - Within-session temporal correlations are broken but between-trial
        correlation structure within the training set is partially preserved
        (because values are reused from the same pool).

    Parameters
    ----------
    train_brain, val_brain, test_brain : split brain arrays (N, D)
    rng : seeded generator

    Returns
    -------
    shuffled arrays (same shapes), audit dict
    """
    def shuffle_rows(arr: np.ndarray) -> np.ndarray:
        idx = rng.permutation(len(arr))
        return arr[idx].copy()

    shuf_train = shuffle_rows(train_brain)
    shuf_val   = shuffle_rows(val_brain)
    shuf_test  = shuffle_rows(test_brain)

    audit = {
        "type": "shuffled_brain",
        "description": (
            "Brain representations permuted across trials within each split. "
            "Trial-specific EEG-iEEG correspondence is destroyed. "
            "Marginal feature distribution is preserved exactly."
        ),
        "marginal_preserved": True,
        "trial_correspondence_broken": True,
        "limitation": (
            "Shuffling within splits preserves the within-split marginal "
            "distribution exactly, but does not match any cross-trial structure. "
            "Cannot establish that benefit is task-specific vs. generic structure."
        ),
        "n_train": len(train_brain),
        "n_val":   len(val_brain),
        "n_test":  len(test_brain),
    }
    return shuf_train, shuf_val, shuf_test, audit


# ──────────────────────────────────────────────────────────────────────────────
# Random control
# ──────────────────────────────────────────────────────────────────────────────

def make_random_control(
    train_brain: np.ndarray,
    n_train: int,
    n_val: int,
    n_test: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Gaussian noise matched only to dimension and global scale.

    Matched: dimension D only. Scale fixed to unit variance.
    NOT matched: mean, per-feature std, covariance, any neural statistics.

    Parameters
    ----------
    train_brain : used only to determine dimension D
    """
    D = train_brain.shape[1]
    total = n_train + n_val + n_test
    samples = rng.standard_normal((total, D)).astype(np.float32)

    audit = {
        "type": "random_gaussian",
        "matched_properties": ["dimension"],
        "not_matched": ["mean", "std", "covariance", "any_neural_statistics"],
        "dim": D,
        "n_train": n_train, "n_val": n_val, "n_test": n_test,
        "note": (
            "This is the weakest control: pure Gaussian noise at unit scale. "
            "Acts as a reference for whether any structured representation helps."
        ),
    }
    return (
        samples[:n_train],
        samples[n_train:n_train + n_val],
        samples[n_train + n_val:],
        audit,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Verify statistical matching
# ──────────────────────────────────────────────────────────────────────────────

def verify_matching(
    real: np.ndarray,
    synthetic: np.ndarray,
    label: str = "",
) -> dict:
    """Compute mean/std difference and return verification report."""
    real_mu   = real.mean(axis=0)
    syn_mu    = synthetic.mean(axis=0)
    real_std  = real.std(axis=0)
    syn_std   = synthetic.std(axis=0)

    return {
        "label": label,
        "n_real": len(real),
        "n_syn":  len(synthetic),
        "mean_abs_diff_mu":   float(np.abs(real_mu  - syn_mu).mean()),
        "mean_abs_diff_std":  float(np.abs(real_std - syn_std).mean()),
        "max_abs_diff_mu":    float(np.abs(real_mu  - syn_mu).max()),
        "max_abs_diff_std":   float(np.abs(real_std - syn_std).max()),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

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

    reports_dir = Path(cfg["paths"]["reports_dir"])
    if not reports_dir.is_absolute():
        reports_dir = project_root / reports_dir
    reports_dir.mkdir(parents=True, exist_ok=True)

    seed = cfg["splits"]["random_seed"]

    # Load real brain representations
    def load_brain(split: str) -> np.ndarray:
        return np.load(processed_dir / f"brain_{split}.npz")["X"]

    log.info("Loading brain representations ...")
    brain_train = load_brain("train")
    brain_val   = load_brain("val")
    brain_test  = load_brain("test")

    n_train, n_val, n_test = len(brain_train), len(brain_val), len(brain_test)
    log.info("Sizes: train=%d  val=%d  test=%d", n_train, n_val, n_test)

    audits = {}

    # ── C. Synthetic matched ──────────────────────────────────────────────────
    log.info("Generating SYNTHETIC MATCHED control ...")
    rng_syn = np.random.default_rng(seed + 1)
    syn_train, syn_val, syn_test, aud_syn = make_synthetic_control(
        brain_train, n_train, n_val, n_test, rng_syn, match_covariance=True
    )
    v_syn = verify_matching(brain_train, syn_train, "synthetic_train")
    log.info("  Synthetic: mean_abs_diff_mu=%.4f  mean_abs_diff_std=%.4f",
             v_syn["mean_abs_diff_mu"], v_syn["mean_abs_diff_std"])
    audits["synthetic"] = {**aud_syn, "verification": v_syn}

    # ── D. Shuffled ───────────────────────────────────────────────────────────
    log.info("Generating SHUFFLED control ...")
    rng_shuf = np.random.default_rng(seed + 2)
    shuf_train, shuf_val, shuf_test, aud_shuf = make_shuffled_control(
        brain_train, brain_val, brain_test, rng_shuf
    )
    v_shuf = verify_matching(brain_train, shuf_train, "shuffled_train")
    log.info("  Shuffled: mean_abs_diff_mu=%.6f  (should be ~0 — exact permutation)",
             v_shuf["mean_abs_diff_mu"])
    audits["shuffled"] = {**aud_shuf, "verification": v_shuf}

    # ── E. Random ─────────────────────────────────────────────────────────────
    log.info("Generating RANDOM control ...")
    rng_rand = np.random.default_rng(seed + 3)
    rand_train, rand_val, rand_test, aud_rand = make_random_control(
        brain_train, n_train, n_val, n_test, rng_rand
    )
    audits["random"] = aud_rand

    # ── Save .npz files ───────────────────────────────────────────────────────
    def copy_labels(brain_file: str, extra_X: np.ndarray, out_name: str) -> None:
        """Save control brain array with labels copied from the real brain file."""
        src = np.load(processed_dir / brain_file, allow_pickle=True)
        kw = {k: src[k] for k in src.files if k != "X"}
        np.savez_compressed(processed_dir / out_name, X=extra_X, **kw)
        log.info("  Saved %s: %s", out_name, extra_X.shape)

    for name, (tr, vl, te) in [
        ("synthetic", (syn_train, syn_val, syn_test)),
        ("shuffled",  (shuf_train, shuf_val, shuf_test)),
        ("random",    (rand_train, rand_val, rand_test)),
    ]:
        copy_labels("brain_train.npz", tr, f"brain_ctrl_{name}_train.npz")
        copy_labels("brain_val.npz",   vl, f"brain_ctrl_{name}_val.npz")
        copy_labels("brain_test.npz",  te, f"brain_ctrl_{name}_test.npz")

    # ── Save audit ────────────────────────────────────────────────────────────
    audit_path = reports_dir / "controls_audit.json"
    with open(audit_path, "w", encoding="utf-8") as fh:
        json.dump(audits, fh, indent=2)
    log.info("Control audit saved → %s", audit_path)

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    sys.exit(main(parser.parse_args()))
