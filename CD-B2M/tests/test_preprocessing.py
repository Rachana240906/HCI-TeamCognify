"""
tests/test_preprocessing.py
============================
Unit tests for the CD-B2M preprocessing pipeline.

Tests cover:
  - Event loading and filtering
  - Epoch extraction and resampling
  - Amplitude QC
  - Band-power computation
  - Normalisation (no leakage)
  - Dataset classes
  - Split integrity (no overlap)
  - Processed file shapes and label validity

Run with:
    python -m pytest tests/test_preprocessing.py -v
"""

import sys
from math import gcd
from pathlib import Path

import numpy as np
import pytest
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from preprocess import (
    COMMON_EEG_CHANNELS,
    EPOCH_DURATION_S,
    MATCH_MAP,
    SETSIZE_MAP,
    apply_eeg_norm,
    compute_bandpower,
    extract_and_resample_epochs,
    fit_eeg_norm,
    fit_brain_norm,
    load_events,
)
from dataset import BrainGuidedDataset, EEGDataset, load_split

# Path to processed outputs
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"


# ──────────────────────────────────────────────────────────────────────────────
# 1. Event loading
# ──────────────────────────────────────────────────────────────────────────────

def test_load_events_valid():
    """Valid events file returns DataFrame with only clean trials."""
    ev_path = (
        Path(__file__).parent.parent.parent
        / "dataset" / "sub-14" / "ses-04" / "ieeg"
        / "sub-14_ses-04_task-verbalWM_run-01_events.tsv"
    )
    if not ev_path.exists():
        pytest.skip("Dataset not available at expected path.")
    df = load_events(ev_path)
    assert df is not None
    assert len(df) > 0
    # All artifact flags should be 0
    assert (df["Artifact"] == 0).all(), "Artifact trials should be excluded"
    # All SetSizes should be non-zero
    assert (df["SetSize"] != 0).all(), "Invalid SetSize=0 trials should be excluded"
    # All SetSizes should be in {4, 6, 8}
    assert set(df["SetSize"].unique()).issubset({4, 6, 8})
    # All Match values should be in {IN, OUT}
    assert set(df["Match"].unique()).issubset({"IN", "OUT"})


def test_load_events_missing_file(tmp_path):
    """Missing events file returns None."""
    result = load_events(tmp_path / "nonexistent.tsv")
    assert result is None


def test_load_events_all_artifacts(tmp_path):
    """File where all trials are artifacts returns None."""
    import pandas as pd
    df = pd.DataFrame({
        "onset": [0.0, 8.0],
        "duration": [8.0, 8.0],
        "nTrial": [1, 2],
        "begSample": [1, 32001],
        "endSample": [32000, 64000],
        "SetSize": [4, 0],
        "ProbeLetter": ["n/a", "n/a"],
        "Match": ["IN", float("nan")],
        "Correct": [1, 0],
        "ResponseTime": [1.0, 0.0],
        "Artifact": [1, 1],
    })
    p = tmp_path / "events.tsv"
    df.to_csv(p, sep="\t", index=False)
    result = load_events(p)
    assert result is None


# ──────────────────────────────────────────────────────────────────────────────
# 2. Epoch extraction and resampling
# ──────────────────────────────────────────────────────────────────────────────

def make_synthetic_events(n_trials=10, native_sfreq=4000.0):
    """Create a minimal events DataFrame for testing."""
    import pandas as pd
    rows = []
    n_samples = int(native_sfreq * EPOCH_DURATION_S)
    for i in range(n_trials):
        beg = i * n_samples + 1
        end = beg + n_samples - 1
        rows.append({
            "nTrial": i + 1,
            "begSample": beg,
            "endSample": end,
            "SetSize": [4, 6, 8][i % 3],
            "Match": ["IN", "OUT"][i % 2],
            "Correct": 1,
            "ResponseTime": 1.0,
            "Artifact": 0,
        })
    return pd.DataFrame(rows)


def test_epoch_extraction_shape():
    """Extracted epochs have correct shape after resampling."""
    native_sfreq = 4000.0
    target_sfreq = 256.0
    n_trials = 5
    n_ch = 21
    n_native = int(native_sfreq * EPOCH_DURATION_S)
    n_target = int(target_sfreq * EPOCH_DURATION_S)

    rng = np.random.default_rng(42)
    data = rng.standard_normal((n_ch, n_native * n_trials)).astype(np.float64)
    events_df = make_synthetic_events(n_trials, native_sfreq)

    epochs, kept_mask = extract_and_resample_epochs(
        data, native_sfreq, target_sfreq, events_df, amp_threshold=1e9
    )
    assert epochs.shape == (n_trials, n_ch, n_target), (
        f"Expected ({n_trials}, {n_ch}, {n_target}), got {epochs.shape}"
    )
    assert kept_mask.sum() == n_trials
    assert epochs.dtype == np.float32


def test_epoch_extraction_resampled_length():
    """Resampled epoch length is exactly target_sfreq × 8 s."""
    native_sfreq = 4096.0
    target_sfreq = 256.0
    n_target = int(target_sfreq * EPOCH_DURATION_S)  # 2048
    n_native = int(native_sfreq * EPOCH_DURATION_S)  # 32768

    rng = np.random.default_rng(0)
    data = rng.standard_normal((5, n_native * 3)).astype(np.float64)
    events_df = make_synthetic_events(3, native_sfreq)

    epochs, _ = extract_and_resample_epochs(
        data, native_sfreq, target_sfreq, events_df, amp_threshold=1e9
    )
    assert epochs.shape[-1] == n_target, f"Expected {n_target}, got {epochs.shape[-1]}"


def test_amplitude_rejection():
    """Trials exceeding amplitude threshold are rejected."""
    native_sfreq = 4000.0
    target_sfreq = 256.0
    n_native = int(native_sfreq * EPOCH_DURATION_S)
    n_ch = 3
    n_trials = 4

    data = np.ones((n_ch, n_native * n_trials), dtype=np.float64) * 10.0
    # Make trial 2 have a spike exceeding threshold
    beg2 = n_native  # trial 2 starts here (0-indexed = begSample-1)
    data[0, beg2 + 100] = 2000.0  # spike

    events_df = make_synthetic_events(n_trials, native_sfreq)
    epochs, kept_mask = extract_and_resample_epochs(
        data, native_sfreq, target_sfreq, events_df, amp_threshold=1500.0
    )
    # Trial index 1 (2nd trial) should be rejected
    assert not kept_mask[1], "Spiked trial should be rejected"
    assert kept_mask[0] and kept_mask[2] and kept_mask[3], "Clean trials should be kept"
    assert epochs.shape[0] == 3


def test_no_data_beyond_recording():
    """Trials requesting samples beyond recording end are skipped gracefully."""
    native_sfreq = 4000.0
    target_sfreq = 256.0
    n_native = int(native_sfreq * EPOCH_DURATION_S)
    data = np.ones((5, n_native * 2))  # only 2 trials worth of data

    events_df = make_synthetic_events(3, native_sfreq)  # 3 trials
    epochs, kept_mask = extract_and_resample_epochs(
        data, native_sfreq, target_sfreq, events_df, amp_threshold=1e9
    )
    # Trial 3 should be skipped (no data)
    assert not kept_mask[2]
    assert epochs.shape[0] == 2


# ──────────────────────────────────────────────────────────────────────────────
# 3. Band-power computation
# ──────────────────────────────────────────────────────────────────────────────

def test_bandpower_shape():
    """Band-power output has correct shape."""
    n_ch, n_samples, sfreq = 64, 32000, 4000.0
    bands = {"theta": [4.0, 8.0], "gamma": [30.0, 80.0]}
    rng = np.random.default_rng(7)
    ep = rng.standard_normal((n_ch, n_samples)).astype(np.float32)
    bp = compute_bandpower(ep, sfreq, bands)
    assert bp.shape == (n_ch * len(bands),), (
        f"Expected ({n_ch * len(bands)},), got {bp.shape}"
    )
    assert bp.dtype == np.float32


def test_bandpower_is_log():
    """Band-power values are log10 (typically negative for low-amplitude signals)."""
    n_ch, n_samples, sfreq = 2, 4000, 4000.0
    bands = {"theta": [4.0, 8.0]}
    rng = np.random.default_rng(1)
    ep = rng.standard_normal((n_ch, n_samples)).astype(np.float32) * 0.001
    bp = compute_bandpower(ep, sfreq, bands)
    # log10 of tiny power values should be negative
    assert np.all(bp < 10.0), "Unexpectedly large band-power values"


def test_bandpower_no_nan():
    """Band-power computation does not produce NaN."""
    n_ch, n_samples, sfreq = 10, 8000, 4000.0
    bands = {"theta": [4.0, 8.0], "gamma": [30.0, 80.0]}
    ep = np.random.randn(n_ch, n_samples).astype(np.float32)
    bp = compute_bandpower(ep, sfreq, bands)
    assert not np.any(np.isnan(bp)), "NaN in band-power output"
    assert not np.any(np.isinf(bp)), "Inf in band-power output"


# ──────────────────────────────────────────────────────────────────────────────
# 4. Normalisation (leakage prevention)
# ──────────────────────────────────────────────────────────────────────────────

def test_normalisation_fit_apply():
    """z-score normalisation produces zero mean and unit std on training data."""
    rng = np.random.default_rng(42)
    n, c, t = 50, 21, 2048
    eeg_train = rng.standard_normal((n, c, t)).astype(np.float32) * 50 + 10

    mean, std = fit_eeg_norm(eeg_train)
    assert mean.shape == (c, 1)
    assert std.shape == (c, 1)

    eeg_norm = apply_eeg_norm(eeg_train, mean, std)
    # Channel-wise mean should be ~0, std ~1
    flat = eeg_norm.transpose(1, 0, 2).reshape(c, -1)
    assert np.allclose(flat.mean(axis=1), 0.0, atol=1e-4), "Mean not ~0 after normalisation"
    assert np.allclose(flat.std(axis=1), 1.0, atol=1e-3), "Std not ~1 after normalisation"


def test_normalisation_no_train_leakage():
    """Val/test normalisation uses ONLY training statistics."""
    rng = np.random.default_rng(99)
    n_train, n_val = 40, 10
    c, t = 5, 100

    train = (rng.standard_normal((n_train, c, t)) * 20 + 100).astype(np.float32)
    val   = (rng.standard_normal((n_val, c, t)) * 200 + 500).astype(np.float32)

    mean, std = fit_eeg_norm(train)
    val_norm = apply_eeg_norm(val, mean, std)

    # Val normalised with TRAIN stats should NOT have zero mean (different distributions)
    flat_val = val_norm.transpose(1, 0, 2).reshape(c, -1)
    assert not np.allclose(flat_val.mean(axis=1), 0.0, atol=0.1), (
        "Val mean is zero — possible leakage of test stats into training normaliser"
    )


def test_std_zero_handling():
    """Channels with zero std (constant signal) do not cause divide-by-zero."""
    n, c, t = 10, 3, 50
    eeg = np.zeros((n, c, t), dtype=np.float32)
    eeg[:, 1, :] = 5.0   # constant non-zero
    mean, std = fit_eeg_norm(eeg)
    assert np.all(std >= 1e-6), "Std should be clipped to prevent div-by-zero"
    normed = apply_eeg_norm(eeg, mean, std)
    assert not np.any(np.isnan(normed))
    assert not np.any(np.isinf(normed))


# ──────────────────────────────────────────────────────────────────────────────
# 5. Dataset classes
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(
    not (PROCESSED_DIR / "eeg_train.npz").exists(),
    reason="Processed data not available — run preprocess.py first",
)
class TestDatasets:

    def test_eeg_dataset_shape(self):
        ds = EEGDataset(PROCESSED_DIR / "eeg_train.npz", label="setsize")
        x, y = ds[0]
        assert x.ndim == 2, "EEG sample should be (C, T)"
        assert y.ndim == 0, "Label should be scalar"
        assert x.dtype == torch.float32
        assert y.dtype == torch.int64

    def test_eeg_dataset_labels_valid(self):
        ds = EEGDataset(PROCESSED_DIR / "eeg_train.npz", label="setsize")
        unique_y = ds.y.unique().tolist()
        assert all(y in [0, 1, 2] for y in unique_y), (
            f"SetSize labels should be in {{0,1,2}}, got {unique_y}"
        )

    def test_brain_guided_dataset_shape(self):
        ds = BrainGuidedDataset(
            PROCESSED_DIR / "eeg_train.npz",
            PROCESSED_DIR / "brain_train.npz",
            label="setsize",
        )
        x, b, y = ds[0]
        assert x.ndim == 2
        assert b.ndim == 1, "Brain rep should be 1D vector"
        assert y.ndim == 0

    def test_brain_guided_alignment(self):
        """EEG and brain arrays must have same number of trials."""
        ds = BrainGuidedDataset(
            PROCESSED_DIR / "eeg_train.npz",
            PROCESSED_DIR / "brain_train.npz",
        )
        assert len(ds.X) == len(ds.B) == len(ds.y)

    def test_brain_dim_matches_config(self):
        """Brain representation dimension = 64 channels × 2 bands = 128."""
        ds = BrainGuidedDataset(
            PROCESSED_DIR / "eeg_train.npz",
            PROCESSED_DIR / "brain_train.npz",
        )
        assert ds.brain_dim == 128, (
            f"Expected brain_dim=128, got {ds.brain_dim}"
        )

    def test_match_label_binary(self):
        ds = EEGDataset(PROCESSED_DIR / "eeg_train.npz", label="match")
        unique_y = ds.y.unique().tolist()
        assert all(y in [0, 1] for y in unique_y), (
            f"Match labels should be binary {{0,1}}, got {unique_y}"
        )

    def test_eeg_no_nan(self):
        ds = EEGDataset(PROCESSED_DIR / "eeg_train.npz")
        assert not torch.any(torch.isnan(ds.X)), "NaN found in EEG data"
        assert not torch.any(torch.isinf(ds.X)), "Inf found in EEG data"

    def test_brain_no_nan(self):
        ds = BrainGuidedDataset(
            PROCESSED_DIR / "eeg_train.npz",
            PROCESSED_DIR / "brain_train.npz",
        )
        assert not torch.any(torch.isnan(ds.B)), "NaN found in brain representations"
        assert not torch.any(torch.isinf(ds.B)), "Inf found in brain representations"


# ──────────────────────────────────────────────────────────────────────────────
# 6. Split integrity
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(
    not (PROCESSED_DIR / "eeg_train.npz").exists(),
    reason="Processed data not available",
)
class TestSplitIntegrity:

    def _load_ids(self, split: str) -> set:
        """Load (subject, session, trial_idx) tuples for a split."""
        data = np.load(PROCESSED_DIR / f"eeg_{split}.npz", allow_pickle=True)
        return set(zip(data["subject"], data["session"], data["trial_idx"]))

    def test_no_train_val_overlap(self):
        train_ids = self._load_ids("train")
        val_ids   = self._load_ids("val")
        overlap = train_ids & val_ids
        assert len(overlap) == 0, (
            f"Train/val overlap detected ({len(overlap)} trials)"
        )

    def test_no_train_test_overlap(self):
        train_ids = self._load_ids("train")
        test_ids  = self._load_ids("test")
        overlap = train_ids & test_ids
        assert len(overlap) == 0, (
            f"Train/test overlap detected ({len(overlap)} trials)"
        )

    def test_no_val_test_overlap(self):
        val_ids  = self._load_ids("val")
        test_ids = self._load_ids("test")
        overlap = val_ids & test_ids
        assert len(overlap) == 0, (
            f"Val/test overlap detected ({len(overlap)} trials)"
        )

    def test_session_aware_split(self):
        """Held-out (subject, session) pairs should ONLY appear in the test split."""
        # These are the held-out pairs from config.yaml
        held_out_pairs = {
            ("sub-14", "ses-07"),
            ("sub-14", "ses-08"),
            ("sub-15", "ses-04"),
        }
        test_data  = np.load(PROCESSED_DIR / "eeg_test.npz",  allow_pickle=True)
        train_data = np.load(PROCESSED_DIR / "eeg_train.npz", allow_pickle=True)
        val_data   = np.load(PROCESSED_DIR / "eeg_val.npz",   allow_pickle=True)

        def get_pairs(data):
            return set(zip(data["subject"], data["session"]))

        train_pairs = get_pairs(train_data)
        val_pairs   = get_pairs(val_data)
        train_val_pairs = train_pairs | val_pairs

        leaked = held_out_pairs & train_val_pairs
        assert len(leaked) == 0, (
            f"Held-out (subject, session) pairs leaked into train/val: {leaked}"
        )

    def test_sizes_positive(self):
        for split in ("train", "val", "test"):
            data = np.load(PROCESSED_DIR / f"eeg_{split}.npz", allow_pickle=True)
            assert data["X"].shape[0] > 0, f"Split '{split}' is empty"

    def test_eeg_brain_label_alignment(self):
        """EEG and brain files must have identical labels for all splits."""
        for split in ("train", "val", "test"):
            eeg   = np.load(PROCESSED_DIR / f"eeg_{split}.npz",   allow_pickle=True)
            brain = np.load(PROCESSED_DIR / f"brain_{split}.npz", allow_pickle=True)
            assert np.array_equal(eeg["y_setsize"], brain["y_setsize"]), (
                f"y_setsize mismatch in split '{split}'"
            )
            assert np.array_equal(eeg["subject"], brain["subject"]), (
                f"subject mismatch in split '{split}'"
            )


# ──────────────────────────────────────────────────────────────────────────────
# 7. Norm stats
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(
    not (PROCESSED_DIR / "norm_stats.npz").exists(),
    reason="Processed data not available",
)
def test_norm_stats_shape():
    stats = np.load(PROCESSED_DIR / "norm_stats.npz")
    n_eeg_ch = 21
    assert stats["eeg_mean"].shape == (n_eeg_ch, 1)
    assert stats["eeg_std"].shape  == (n_eeg_ch, 1)
    assert np.all(stats["eeg_std"] > 0), "EEG std should be positive"


@pytest.mark.skipif(
    not (PROCESSED_DIR / "norm_stats.npz").exists(),
    reason="Processed data not available",
)
def test_normalised_train_eeg_approx_zero_mean():
    """After normalisation, channel-wise train EEG mean should be near 0."""
    eeg = np.load(PROCESSED_DIR / "eeg_train.npz")["X"]  # (N, C, T)
    n, c, t = eeg.shape
    flat = eeg.transpose(1, 0, 2).reshape(c, -1)
    means = flat.mean(axis=1)
    assert np.allclose(means, 0.0, atol=0.1), (
        f"Train EEG channel means not near 0: {means}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
