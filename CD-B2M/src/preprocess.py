"""
preprocess.py
=============
Phase 3 – Preprocessing pipeline for CD-B2M.

Strategy (optimised for speed):
  1. Load full EDF into memory.
  2. Channel selection (common EEG set / all SEEG).
  3. Bandpass + notch filter at native rate (fast FIR on pre-loaded data).
  4. Extract individual 8-second epochs by sample index.
  5. Resample EACH epoch (polyphase resample_poly) — much faster than resampling
     the full ~400 s signal.
  6. Amplitude QC per epoch.
  7. Compute iEEG log-band-power per epoch (post-filter, pre-resample).
  8. Concatenate sessions, session-aware train/val/test split.
  9. Normalise using training-set statistics only.
 10. Save .npz files + JSON audit log.

Usage
-----
    python src/preprocess.py [--config configs/config.yaml] [--smoke_test]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from math import gcd
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import yaml
from scipy.signal import resample_poly, welch

# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)
mne.set_log_level("WARNING")

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

# EEG channels present in BOTH subjects (sub-14: 21 ch, sub-15: 23 ch)
COMMON_EEG_CHANNELS = [
    "A1", "A2", "C3", "C4", "Cz",
    "F3", "F4", "F7", "F8", "Fp1", "Fp2", "Fz",
    "O1", "O2",
    "P3", "P4", "Pz",
    "T3", "T4", "T5", "T6",
]

SETSIZE_MAP = {4: 0, 6: 1, 8: 2}
MATCH_MAP   = {"IN": 1, "OUT": 0}
EPOCH_DURATION_S = 8.0


# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

def load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# ──────────────────────────────────────────────────────────────────────────────
# Events
# ──────────────────────────────────────────────────────────────────────────────

def load_events(ev_path: Path) -> pd.DataFrame | None:
    """Load events.tsv, filter out artifacts and invalid-SetSize rows."""
    if not ev_path.exists():
        return None
    df = pd.read_csv(ev_path, sep="\t")
    df = df[(df["Artifact"] == 0) & (df["SetSize"] != 0)].copy()
    df = df.dropna(subset=["Match"]).reset_index(drop=True)
    return df if len(df) > 0 else None


# ──────────────────────────────────────────────────────────────────────────────
# EEG processing
# ──────────────────────────────────────────────────────────────────────────────

def load_and_filter_eeg(
    edf_path: Path,
    wanted_channels: list[str],
    l_freq: float,
    h_freq: float,
    notch_freqs: list[float],
) -> tuple[np.ndarray, float, list[str]]:
    """
    Load EEG EDF, pick channels, filter (bandpass + notch).

    Returns
    -------
    data_uV  : (n_channels, n_total_samples)  float64  µV
    sfreq    : native sampling frequency
    ch_names : channel list in the returned array
    """
    raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)
    available = set(raw.ch_names)
    to_pick = [ch for ch in wanted_channels if ch in available]
    missing  = [ch for ch in wanted_channels if ch not in available]
    if missing:
        log.warning("  EEG channels absent: %s", missing)
    raw.pick(to_pick)

    sfreq = raw.info["sfreq"]
    raw.filter(l_freq=l_freq, h_freq=h_freq,
               method="fir", fir_design="firwin", verbose=False)
    for nf in notch_freqs:
        if nf < sfreq / 2:
            raw.notch_filter(freqs=nf, verbose=False)

    return raw.get_data(), sfreq, list(raw.ch_names)


def load_and_filter_ieeg(
    edf_path: Path,
) -> tuple[np.ndarray, float, list[str]]:
    """
    Load iEEG EDF, filter 0.5–200 Hz + 50 Hz notch.

    Returns
    -------
    data   : (64, n_total_samples)  in EDF native physical units (µV)
    sfreq  : native sampling frequency
    ch_names : channel list
    """
    raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)
    sfreq = raw.info["sfreq"]
    raw.filter(l_freq=0.5, h_freq=min(200.0, sfreq / 2 - 1),
               method="fir", fir_design="firwin", verbose=False)
    raw.notch_filter(freqs=50.0, verbose=False)
    return raw.get_data(), sfreq, list(raw.ch_names)


# ──────────────────────────────────────────────────────────────────────────────
# Epoch extraction + per-epoch resample
# ──────────────────────────────────────────────────────────────────────────────

def extract_and_resample_epochs(
    data: np.ndarray,
    native_sfreq: float,
    target_sfreq: float,
    events_df: pd.DataFrame,
    amp_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract 8-second epochs by sample index, resample each individually.

    Parameters
    ----------
    data          : (n_channels, n_total_samples) in EDF native units
    amp_threshold : peak-to-peak rejection threshold (same units as data)

    Returns
    -------
    epochs    : float32 (n_kept, n_channels, n_target_samples)
    kept_mask : bool    (n_trials_in_df,)
    """
    n_native = int(round(native_sfreq * EPOCH_DURATION_S))
    n_target = int(round(target_sfreq * EPOCH_DURATION_S))
    n_ch = data.shape[0]

    g    = gcd(int(target_sfreq), int(native_sfreq))
    up   = int(target_sfreq) // g
    down = int(native_sfreq) // g

    kept_epochs: list[np.ndarray] = []
    kept_mask = np.zeros(len(events_df), dtype=bool)

    for i, row in events_df.iterrows():
        beg = int(row["begSample"]) - 1
        end = beg + n_native
        if end > data.shape[1]:
            log.warning("  Trial %d: samples %d–%d exceed recording length %d. Skip.",
                        int(row["nTrial"]), beg, end, data.shape[1])
            continue

        ep = data[:, beg:end]

        # Amplitude QC
        pp = ep.max(axis=-1) - ep.min(axis=-1)
        if np.any(pp >= amp_threshold):
            log.debug("  Trial %d rejected: max p-p=%.1f",
                      int(row["nTrial"]), float(pp.max()))
            continue

        # Polyphase resample
        ep_ds = resample_poly(ep, up, down, axis=-1).astype(np.float32)

        # Ensure exact length
        if ep_ds.shape[1] > n_target:
            ep_ds = ep_ds[:, :n_target]
        elif ep_ds.shape[1] < n_target:
            pad = np.zeros((n_ch, n_target - ep_ds.shape[1]), dtype=np.float32)
            ep_ds = np.concatenate([ep_ds, pad], axis=1)

        kept_epochs.append(ep_ds)
        kept_mask[i] = True

    if kept_epochs:
        return np.stack(kept_epochs, axis=0), kept_mask
    return np.empty((0, n_ch, n_target), dtype=np.float32), kept_mask


# ──────────────────────────────────────────────────────────────────────────────
# iEEG brain representation: log-band-power
# ──────────────────────────────────────────────────────────────────────────────

def compute_bandpower(
    epoch_uV: np.ndarray,
    sfreq: float,
    bands: dict[str, list[float]],
) -> np.ndarray:
    """
    Log10 band-power per electrode per band using Welch PSD.

    Parameters
    ----------
    epoch_uV : (n_channels, n_samples)
    sfreq    : Hz (of the epoch — native rate, pre-resample)
    bands    : {"theta": [4, 8], "gamma": [30, 80]}

    Returns
    -------
    bp : float32  (n_channels * n_bands,)
    """
    n_ch, n_samples = epoch_uV.shape
    n_bands = len(bands)
    bp = np.zeros((n_ch, n_bands), dtype=np.float32)
    nperseg = min(n_samples, max(64, int(sfreq * 1.0)))  # 1-second segments

    for ch in range(n_ch):
        freqs, psd = welch(epoch_uV[ch], fs=sfreq, nperseg=nperseg)
        for b_idx, (_, (flo, fhi)) in enumerate(bands.items()):
            mask = (freqs >= flo) & (freqs <= fhi)
            if mask.sum() > 1:
                power = float(np.trapezoid(psd[mask], freqs[mask]))
            else:
                power = 1e-10
            bp[ch, b_idx] = np.log10(max(power, 1e-10))

    return bp.reshape(-1)


def compute_brain_reps(
    ieeg_data: np.ndarray,
    native_sfreq: float,
    events_df: pd.DataFrame,
    kept_mask: np.ndarray,
    bands: dict[str, list[float]],
) -> np.ndarray:
    """
    Compute band-power representation for each kept trial.

    Returns
    -------
    reps : float32 (n_kept, n_channels * n_bands)
    """
    n_native = int(round(native_sfreq * EPOCH_DURATION_S))
    kept_indices = np.where(kept_mask)[0]
    reps = []

    for i in kept_indices:
        row = events_df.iloc[i]
        beg = int(row["begSample"]) - 1
        end = beg + n_native
        ep = ieeg_data[:, beg:min(end, ieeg_data.shape[1])]
        bp = compute_bandpower(ep, native_sfreq, bands)
        reps.append(bp)

    if reps:
        return np.stack(reps, axis=0)
    n_ch = ieeg_data.shape[0]
    n_bands = len(bands)
    return np.empty((0, n_ch * n_bands), dtype=np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Per-session pipeline
# ──────────────────────────────────────────────────────────────────────────────

def process_session(
    sub: str, ses: str,
    dataset_root: Path,
    cfg: dict,
    audit: list[dict],
) -> dict | None:
    pp  = cfg["preprocessing"]
    bands = {k: v for k, v in pp["ieeg_bands"].items()}
    target_sfreq = float(pp["target_sfreq"])
    amp_thresh   = float(pp["artifact_amp_uV"])

    ses_path = dataset_root / sub / ses
    prefix   = f"{sub}_{ses}_task-verbalWM_run-01"

    # Events
    events_df = load_events(ses_path / "ieeg" / f"{prefix}_events.tsv")
    if events_df is None:
        log.warning("%s/%s: no valid events, skipping.", sub, ses)
        return None

    # ── EEG ──────────────────────────────────────────────────────────────────
    eeg_edf = ses_path / "eeg" / f"{prefix}_eeg.edf"
    if not eeg_edf.exists():
        log.warning("%s/%s: EEG EDF missing, skipping.", sub, ses)
        return None

    log.info("  [%s/%s] Loading + filtering EEG ...", sub, ses)
    try:
        eeg_data, native_sfreq_eeg, ch_names = load_and_filter_eeg(
            eeg_edf,
            COMMON_EEG_CHANNELS,
            pp["bandpass_low"],
            pp["bandpass_high"],
            pp["notch_freqs"],
        )
    except Exception as exc:
        log.warning("%s/%s: EEG load/filter failed (%s). Skipping.", sub, ses, exc)
        return None
    log.info("  [%s/%s] Extracting + resampling EEG epochs ...", sub, ses)
    eeg_epochs, kept_mask = extract_and_resample_epochs(
        eeg_data, native_sfreq_eeg, target_sfreq, events_df, amp_thresh
    )
    n_kept = int(kept_mask.sum())
    log.info("  [%s/%s] EEG epochs kept: %d / %d", sub, ses, n_kept, len(events_df))

    if n_kept == 0:
        log.warning("%s/%s: 0 EEG epochs passed QC, skipping.", sub, ses)
        return None

    # ── iEEG ─────────────────────────────────────────────────────────────────
    ieeg_edf = ses_path / "ieeg" / f"{prefix}_ieeg.edf"
    if not ieeg_edf.exists():
        log.warning("%s/%s: iEEG EDF missing, skipping.", sub, ses)
        return None

    log.info("  [%s/%s] Loading + filtering iEEG ...", sub, ses)
    try:
        ieeg_data, native_sfreq_ieeg, ieeg_ch = load_and_filter_ieeg(ieeg_edf)
    except Exception as exc:
        log.warning("%s/%s: iEEG load/filter failed (%s). Skipping.", sub, ses, exc)
        return None
    log.info("  [%s/%s] Computing brain representations ...", sub, ses)
    brain_reps = compute_brain_reps(
        ieeg_data, native_sfreq_ieeg, events_df, kept_mask, bands
    )

    # ── Labels ───────────────────────────────────────────────────────────────
    kept_rows = events_df[kept_mask].reset_index(drop=True)
    y_setsize = np.array([SETSIZE_MAP[int(s)] for s in kept_rows["SetSize"]], dtype=np.int32)
    y_match   = np.array([MATCH_MAP[str(m)] for m in kept_rows["Match"]],   dtype=np.int32)
    y_correct = kept_rows["Correct"].astype(np.int32).values
    trial_idx = kept_rows["nTrial"].astype(np.int32).values

    # ── Audit ─────────────────────────────────────────────────────────────────
    audit.append({
        "subject": sub, "session": ses,
        "events_clean": len(events_df),
        "epochs_kept": n_kept,
        "native_sfreq_eeg": native_sfreq_eeg,
        "native_sfreq_ieeg": native_sfreq_ieeg,
        "n_eeg_channels": len(ch_names),
        "brain_rep_dim": brain_reps.shape[1],
        "setsize_dist": {str(k): int((y_setsize == v).sum()) for k, v in SETSIZE_MAP.items()},
        "match_dist": {"IN": int(y_match.sum()), "OUT": int((y_match == 0).sum())},
    })

    return {
        "eeg":       eeg_epochs,    # (n, C, T)
        "brain":     brain_reps,    # (n, D)
        "y_setsize": y_setsize,
        "y_match":   y_match,
        "y_correct": y_correct,
        "subject":   np.array([sub] * n_kept),
        "session":   np.array([ses] * n_kept),
        "trial_idx": trial_idx,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Aggregation helpers
# ──────────────────────────────────────────────────────────────────────────────

def concat_sessions(session_list: list[dict]) -> dict:
    keys = ["eeg", "brain", "y_setsize", "y_match", "y_correct",
            "subject", "session", "trial_idx"]
    out = {}
    for k in keys:
        arrays = [s[k] for s in session_list]
        out[k] = np.concatenate(arrays, axis=0)
    return out


def subset(data: dict, idx: np.ndarray) -> dict:
    return {k: v[idx] for k, v in data.items()}


# ──────────────────────────────────────────────────────────────────────────────
# Normalisation
# ──────────────────────────────────────────────────────────────────────────────

def fit_eeg_norm(eeg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel mean/std from (n, C, T) array."""
    n, c, t = eeg.shape
    flat = eeg.transpose(1, 0, 2).reshape(c, -1)
    mean = flat.mean(axis=1, keepdims=True).astype(np.float32)
    std  = flat.std(axis=1,  keepdims=True).astype(np.float32)
    std  = np.where(std < 1e-6, 1.0, std)
    return mean, std   # (C, 1), (C, 1)


def apply_eeg_norm(eeg: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((eeg - mean[np.newaxis, :, :]) / std[np.newaxis, :, :]).astype(np.float32)


def fit_brain_norm(brain: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Feature-wise mean/std from (n, D) array."""
    mean = brain.mean(axis=0, keepdims=True).astype(np.float32)
    std  = brain.std(axis=0,  keepdims=True).astype(np.float32)
    std  = np.where(std < 1e-6, 1.0, std)
    return mean, std


# ──────────────────────────────────────────────────────────────────────────────
# Save
# ──────────────────────────────────────────────────────────────────────────────

def save_split(out_dir: Path, name: str, data: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    kw = dict(
        y_setsize=data["y_setsize"],
        y_match=data["y_match"],
        y_correct=data["y_correct"],
        subject=data["subject"],
        session=data["session"],
        trial_idx=data["trial_idx"],
    )
    np.savez_compressed(out_dir / f"eeg_{name}.npz",   X=data["eeg"],   **kw)
    np.savez_compressed(out_dir / f"brain_{name}.npz", X=data["brain"], **kw)
    log.info("  Saved %s: %d trials  eeg%s  brain%s",
             name, len(data["eeg"]), data["eeg"].shape, data["brain"].shape)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> int:
    t0 = time.time()
    project_root = Path(__file__).parent.parent

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = project_root / cfg_path
    cfg = load_config(cfg_path)

    if args.smoke_test:
        cfg["smoke_test"]["enabled"] = True
        log.info("=== SMOKE-TEST MODE ===")

    dataset_root = Path(cfg["paths"]["dataset_root"])
    if not dataset_root.is_absolute():
        dataset_root = (project_root / dataset_root).resolve()

    out_dir = Path(cfg["paths"]["processed_dir"])
    if not out_dir.is_absolute():
        out_dir = project_root / out_dir

    reports_dir = Path(cfg["paths"]["reports_dir"])
    if not reports_dir.is_absolute():
        reports_dir = project_root / reports_dir
    reports_dir.mkdir(parents=True, exist_ok=True)

    log.info("Dataset root : %s", dataset_root)
    log.info("Output dir   : %s", out_dir)

    pp = cfg["preprocessing"]

    # ── Process sessions ──────────────────────────────────────────────────────
    held_out_cfg = cfg.get("held_out_sessions", {})
    audit: list[dict] = []
    train_pool: list[dict] = []
    test_pool:  list[dict] = []

    for sub in cfg["subjects"]:
        sub_path = dataset_root / sub
        sessions = sorted([d.name for d in sub_path.iterdir()
                            if d.is_dir() and d.name.startswith("ses-")])
        held_out = held_out_cfg.get(sub, [])

        for ses in sessions:
            log.info("Processing %s/%s ...", sub, ses)
            result = process_session(sub, ses, dataset_root, cfg, audit)
            if result is None:
                continue
            if ses in held_out:
                test_pool.append(result)
            else:
                train_pool.append(result)

    if not train_pool:
        log.error("No training data. Abort.")
        return 1

    # Smoke-test truncation
    if cfg["smoke_test"]["enabled"]:
        mt = cfg["smoke_test"]["max_trials"]
        for pool in (train_pool, test_pool):
            for s in pool:
                for k in list(s.keys()):
                    s[k] = s[k][:mt]

    # ── Concatenate ───────────────────────────────────────────────────────────
    all_train = concat_sessions(train_pool)
    all_test  = concat_sessions(test_pool) if test_pool else None

    n_total = len(all_train["eeg"])
    log.info("Train+val pool : %d trials", n_total)

    # ── Apply time window (slice maintenance epoch from full 8s trial) ────────
    # epoch_tmin / epoch_tmax from config determine which portion to use.
    # E.g. 1-5s captures the encoding+maintenance period.
    t_min = float(pp.get("epoch_tmin", 0.0))
    t_max = float(pp.get("epoch_tmax", 8.0))
    target_sfreq = float(pp["target_sfreq"])
    t_min_samp = int(round(t_min * target_sfreq))
    t_max_samp = int(round(t_max * target_sfreq))
    t_max_samp = min(t_max_samp, all_train["eeg"].shape[2])

    log.info("Applying time window %.1f–%.1f s (samples %d–%d)",
             t_min, t_max, t_min_samp, t_max_samp)

    for pool_data in [all_train] + ([all_test] if all_test else []):
        pool_data["eeg"] = pool_data["eeg"][:, :, t_min_samp:t_max_samp]

    # ── Stratified val split ──────────────────────────────────────────────────
    rng = np.random.default_rng(cfg["splits"]["random_seed"])
    val_frac = cfg["splits"]["val_fraction"]
    val_idx, train_idx = [], []
    for cls in range(3):
        cidx = np.where(all_train["y_setsize"] == cls)[0].copy()
        rng.shuffle(cidx)
        nv = max(1, int(round(len(cidx) * val_frac)))
        val_idx.extend(cidx[:nv].tolist())
        train_idx.extend(cidx[nv:].tolist())

    train_data = subset(all_train, np.array(sorted(train_idx)))
    val_data   = subset(all_train, np.array(sorted(val_idx)))

    log.info("Train : %d  Val : %d  Test : %d",
             len(train_data["eeg"]), len(val_data["eeg"]),
             len(all_test["eeg"]) if all_test else 0)

    # ── Normalise (training stats only) ───────────────────────────────────────
    eeg_mean, eeg_std     = fit_eeg_norm(train_data["eeg"])
    brain_mean, brain_std = fit_brain_norm(train_data["brain"])

    for d in (train_data, val_data) + ((all_test,) if all_test else ()):
        d["eeg"]   = apply_eeg_norm(d["eeg"], eeg_mean, eeg_std)
        d["brain"] = ((d["brain"] - brain_mean) / brain_std).astype(np.float32)

    # ── Save ──────────────────────────────────────────────────────────────────
    save_split(out_dir, "train", train_data)
    save_split(out_dir, "val",   val_data)
    if all_test:
        save_split(out_dir, "test", all_test)

    np.savez_compressed(
        out_dir / "norm_stats.npz",
        eeg_mean=eeg_mean, eeg_std=eeg_std,
        brain_mean=brain_mean, brain_std=brain_std,
    )

    # ── Audit log ─────────────────────────────────────────────────────────────
    pp_log = {
        "config": cfg["preprocessing"],
        "common_eeg_channels": COMMON_EEG_CHANNELS,
        "setsize_map": {str(k): v for k, v in SETSIZE_MAP.items()},
        "sessions": audit,
        "split_sizes": {
            "train": int(len(train_data["eeg"])),
            "val":   int(len(val_data["eeg"])),
            "test":  int(len(all_test["eeg"])) if all_test else 0,
        },
        "eeg_shape": list(train_data["eeg"].shape),
        "brain_shape": list(train_data["brain"].shape),
        "elapsed_s": round(time.time() - t0, 1),
        "smoke_test": cfg["smoke_test"]["enabled"],
    }
    log_path = reports_dir / "preprocessing_log.json"
    with open(log_path, "w", encoding="utf-8") as fh:
        json.dump(pp_log, fh, indent=2)

    elapsed = time.time() - t0
    log.info("=" * 55)
    log.info("DONE  (%.1f s)", elapsed)
    log.info("  EEG   shape (train): %s", train_data["eeg"].shape)
    log.info("  Brain shape (train): %s", train_data["brain"].shape)
    log.info("  SetSize dist (train): %s",
             {k: int((train_data["y_setsize"] == v).sum()) for k, v in SETSIZE_MAP.items()})
    log.info("  Log → %s", log_path)
    log.info("=" * 55)
    return 0


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     default="configs/config.yaml")
    parser.add_argument("--smoke_test", action="store_true")
    sys.exit(main(parser.parse_args()))
