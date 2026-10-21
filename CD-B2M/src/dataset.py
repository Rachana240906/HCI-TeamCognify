"""
dataset.py
==========
PyTorch Dataset classes for CD-B2M.

Provides:
  - EEGDataset          : EEG epochs + labels
  - BrainGuidedDataset  : EEG epochs + iEEG brain representation + labels
  - load_split          : convenience loader from processed .npz files
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from scipy.signal import sosfiltfilt, butter
from torch.utils.data import Dataset


def _antialias_and_stride(X: np.ndarray, stride: int) -> np.ndarray:
    """
    Apply a lowpass anti-aliasing filter then decimate by `stride`.

    The filter cutoff is set to 0.8 × Nyquist of the decimated signal
    (i.e. 0.8 × sfreq / (2 × stride)) to leave a 20% guard band.

    Parameters
    ----------
    X      : (N, C, T)  float32 array at original sampling rate
    stride : decimation factor (integer ≥ 1)

    Returns
    -------
    X_dec : (N, C, T // stride)  float32
    """
    if stride <= 1:
        return X
    # Normalised Wn for butter: cutoff / Nyquist = 0.8 / stride
    # (Nyquist of the original signal = 1.0 in normalised form)
    Wn = 0.8 / stride
    sos = butter(N=4, Wn=Wn, btype="low", output="sos")
    # Apply along time axis (axis=2) with zero-phase forward-backward filter
    X_filt = sosfiltfilt(sos, X, axis=2).astype(np.float32)
    return X_filt[:, :, ::stride]


# ──────────────────────────────────────────────────────────────────────────────
# Base Dataset
# ──────────────────────────────────────────────────────────────────────────────

class EEGDataset(Dataset):
    """
    Minimal EEG dataset for baseline (no brain guidance).

    Parameters
    ----------
    eeg_npz  : path to eeg_<split>.npz
    label    : which label column to use ('setsize' | 'match' | 'correct')
    stride   : temporal stride for downsampling (1=no change, 4=4x faster)
    """

    LABEL_KEYS = {
        "setsize": "y_setsize",
        "match":   "y_match",
        "correct": "y_correct",
    }

    def __init__(self, eeg_npz: str | Path, label: str = "setsize", stride: int = 4):
        data = np.load(eeg_npz, allow_pickle=True)
        X = data["X"].astype(np.float32)
        if stride > 1:
            # Anti-aliased decimation (fixes aliasing of 32–100 Hz band)
            X = _antialias_and_stride(X, stride)
        self.X = torch.from_numpy(X)
        lkey = self.LABEL_KEYS.get(label, f"y_{label}")
        self.y = torch.from_numpy(data[lkey].astype(np.int64))
        self.subjects  = data["subject"]
        self.sessions  = data["session"]
        self.trial_idx = data["trial_idx"]
        self.stride    = stride

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]

    @property
    def n_classes(self) -> int:
        return int(self.y.max().item()) + 1

    @property
    def input_size(self) -> int:
        """Number of EEG channels (input features per timestep)."""
        return self.X.shape[1]

    @property
    def seq_len(self) -> int:
        return self.X.shape[2]


class BrainGuidedDataset(Dataset):
    """
    Dataset for brain-guided training.

    Returns (eeg, brain_rep, label) triples.

    Parameters
    ----------
    eeg_npz   : path to eeg_<split>.npz
    brain_npz : path to brain_<split>.npz  (or synthetic replacement)
    label     : 'setsize' | 'match' | 'correct'
    """

    LABEL_KEYS = {
        "setsize": "y_setsize",
        "match":   "y_match",
        "correct": "y_correct",
    }

    def __init__(
        self,
        eeg_npz: str | Path,
        brain_npz: str | Path,
        label: str = "setsize",
        stride: int = 4,
    ):
        eeg_data   = np.load(eeg_npz,   allow_pickle=True)
        brain_data = np.load(brain_npz, allow_pickle=True)

        X = eeg_data["X"].astype(np.float32)
        if stride > 1:
            # Anti-aliased decimation
            X = _antialias_and_stride(X, stride)
        self.X = torch.from_numpy(X)
        self.B = torch.from_numpy(brain_data["X"].astype(np.float32))
        self.stride = stride

        lkey = self.LABEL_KEYS.get(label, f"y_{label}")
        self.y = torch.from_numpy(eeg_data[lkey].astype(np.int64))

        assert len(self.X) == len(self.B) == len(self.y), (
            f"Size mismatch: EEG={len(self.X)}, brain={len(self.B)}, "
            f"labels={len(self.y)}"
        )

        self.subjects  = eeg_data["subject"]
        self.sessions  = eeg_data["session"]
        self.trial_idx = eeg_data["trial_idx"]

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.X[idx], self.B[idx], self.y[idx]

    @property
    def n_classes(self) -> int:
        return int(self.y.max().item()) + 1

    @property
    def input_size(self) -> int:
        return self.X.shape[1]

    @property
    def brain_dim(self) -> int:
        return self.B.shape[1]

    @property
    def seq_len(self) -> int:
        return self.X.shape[2]


# ──────────────────────────────────────────────────────────────────────────────
# Convenience loader
# ──────────────────────────────────────────────────────────────────────────────

def load_split(
    processed_dir: str | Path,
    split: str,
    label: str = "setsize",
    brain_guided: bool = False,
    brain_npz_override: str | Path | None = None,
    stride: int = 4,
) -> EEGDataset | BrainGuidedDataset:
    """
    Load a processed split from the data/processed directory.

    Parameters
    ----------
    processed_dir      : path to data/processed/
    split              : 'train' | 'val' | 'test'
    label              : classification target
    brain_guided       : if True, returns BrainGuidedDataset
    brain_npz_override : use a different brain .npz (e.g. synthetic control)
    stride             : temporal stride applied to EEG (e.g. 4 = 4× faster GRU)
    """
    d = Path(processed_dir)
    eeg_path = d / f"eeg_{split}.npz"

    if brain_guided:
        brain_path = (
            Path(brain_npz_override)
            if brain_npz_override is not None
            else d / f"brain_{split}.npz"
        )
        return BrainGuidedDataset(eeg_path, brain_path, label=label, stride=stride)
    else:
        return EEGDataset(eeg_path, label=label, stride=stride)
