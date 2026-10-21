"""
alignment.py
============
Brain-model alignment loss and representation similarity metrics.

Functions
---------
brain_alignment_loss   : MSE or cosine loss between projected EEG rep and brain rep
combined_loss          : (1 - alpha) * L_task + alpha * L_brain
rsa_correlation        : Representational Similarity Analysis (Spearman correlation)
cka_linear             : Linear Centred Kernel Alignment
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr


# ──────────────────────────────────────────────────────────────────────────────
# Losses
# ──────────────────────────────────────────────────────────────────────────────

def brain_alignment_loss(
    proj: torch.Tensor,
    brain: torch.Tensor,
    loss_type: str = "mse",
) -> torch.Tensor:
    """
    Compute alignment loss between projected EEG representation and brain rep.

    Parameters
    ----------
    proj       : (batch, D)  model projection
    brain      : (batch, D)  brain representation (iEEG band-power)
    loss_type  : 'mse' | 'cosine'

    Returns
    -------
    loss : scalar tensor
    """
    if loss_type == "mse":
        return F.mse_loss(proj, brain)
    elif loss_type == "cosine":
        # 1 - mean cosine similarity (so 0 = perfect alignment)
        cos_sim = F.cosine_similarity(proj, brain, dim=-1)
        return 1.0 - cos_sim.mean()
    else:
        raise ValueError(f"Unknown loss_type '{loss_type}'. Use 'mse' or 'cosine'.")


def combined_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    proj: torch.Tensor | None,
    brain: torch.Tensor | None,
    alpha: float,
    loss_type: str = "mse",
    class_weight: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    L_total = (1 - alpha) * L_task + alpha * L_brain

    Parameters
    ----------
    logits       : (batch, n_classes)
    labels       : (batch,)
    proj         : (batch, D)  or None (if no brain guidance)
    brain        : (batch, D)  or None
    alpha        : weight for brain alignment [0, 1]
    loss_type    : 'mse' | 'cosine'
    class_weight : (n_classes,) tensor or None; inverse-frequency weights to
                   prevent majority-class collapse on imbalanced datasets

    Returns
    -------
    total_loss, task_loss, brain_loss  — all scalar tensors
    """
    task_loss = F.cross_entropy(logits, labels, weight=class_weight)

    if proj is not None and brain is not None and alpha > 0.0:
        b_loss = brain_alignment_loss(proj, brain, loss_type)
        total  = (1.0 - alpha) * task_loss + alpha * b_loss
    else:
        b_loss = torch.zeros(1, device=logits.device)
        total  = task_loss

    return total, task_loss, b_loss


# ──────────────────────────────────────────────────────────────────────────────
# Representation similarity metrics (post-hoc, on numpy arrays)
# ──────────────────────────────────────────────────────────────────────────────

def rsa_correlation(
    rep_a: np.ndarray,
    rep_b: np.ndarray,
) -> tuple[float, float]:
    """
    Representational Similarity Analysis.

    Computes the Spearman correlation between the upper-triangular entries of
    the pairwise Euclidean distance matrices of rep_a and rep_b.

    Parameters
    ----------
    rep_a : (n_samples, d_a)
    rep_b : (n_samples, d_b)

    Returns
    -------
    rho    : Spearman correlation coefficient
    pvalue : two-tailed p-value
    """
    from scipy.spatial.distance import pdist

    n = rep_a.shape[0]
    if n < 4:
        return float("nan"), float("nan")

    rdm_a = pdist(rep_a, metric="euclidean")
    rdm_b = pdist(rep_b, metric="euclidean")

    result = spearmanr(rdm_a, rdm_b)
    return float(result.statistic), float(result.pvalue)


def _center_kernel(K: np.ndarray) -> np.ndarray:
    """Double-centre a kernel matrix."""
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    return H @ K @ H


def cka_linear(
    rep_a: np.ndarray,
    rep_b: np.ndarray,
) -> float:
    """
    Linear Centred Kernel Alignment (CKA).

    CKA = ||X^T Y||_F^2  /  (||X^T X||_F * ||Y^T Y||_F)

    where X, Y are centred.

    Parameters
    ----------
    rep_a : (n_samples, d_a)
    rep_b : (n_samples, d_b)

    Returns
    -------
    cka : float in [0, 1]
    """
    # Centre columns
    a = rep_a - rep_a.mean(axis=0, keepdims=True)
    b = rep_b - rep_b.mean(axis=0, keepdims=True)

    dot_ab  = np.linalg.norm(a.T @ b, "fro") ** 2
    dot_aa  = np.linalg.norm(a.T @ a, "fro")
    dot_bb  = np.linalg.norm(b.T @ b, "fro")

    denom = dot_aa * dot_bb
    if denom < 1e-10:
        return 0.0
    return float(dot_ab / denom)
