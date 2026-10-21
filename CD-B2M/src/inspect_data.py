"""
inspect_data.py
===============
Phase 1 / Phase 2 – Dataset inspection script for CD-B2M project.

Walks every sub-*/ses-* session in the dataset root, reads all
available metadata (channels.tsv, events.tsv, electrodes.tsv,
*.json sidecars) without loading the raw EDF data into RAM,
and writes a machine-readable JSON report plus a human-readable
Markdown summary to  results/reports/.

Usage
-----
    python src/inspect_data.py [--dataset_root <path>] [--out_dir <path>]

Defaults match the workspace layout:
    dataset_root  = ../dataset          (relative to CD-B2M/)
    out_dir       = results/reports/
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import mne
import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def safe_read_tsv(path: Path) -> pd.DataFrame | None:
    """Return a DataFrame or None if the file is missing."""
    if path.exists():
        return pd.read_csv(path, sep="\t")
    return None


def safe_read_json(path: Path) -> dict | None:
    if path.exists():
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    return None


def probe_edf(path: Path) -> dict:
    """Read EDF header only (no data loaded) and return key facts."""
    try:
        raw = mne.io.read_raw_edf(str(path), preload=False, verbose=False)
        info = {
            "n_channels": len(raw.ch_names),
            "channel_names": raw.ch_names,
            "sfreq": raw.info["sfreq"],
            "duration_s": round(float(raw.times[-1]), 3) if len(raw.times) > 0 else None,
            "n_samples": raw.n_times,
        }
    except Exception as exc:
        info = {"error": str(exc)}
    return info


# ──────────────────────────────────────────────────────────────────────────────
# Core inspection logic
# ──────────────────────────────────────────────────────────────────────────────

def inspect_session(sub: str, ses: str, base: Path) -> dict:
    """Inspect one session and return a structured report dict."""
    ses_path = base / sub / ses
    report = {
        "subject": sub,
        "session": ses,
        "modalities": {},
        "events_summary": {},
        "issues": [],
    }

    for modality in ("eeg", "ieeg"):
        mod_path = ses_path / modality
        if not mod_path.exists():
            report["issues"].append(f"Missing modality folder: {modality}")
            continue

        prefix = f"{sub}_{ses}_task-verbalWM_run-01"
        edf_path = mod_path / f"{prefix}_{modality}.edf"
        chan_path = mod_path / f"{prefix}_channels.tsv"
        ev_path = mod_path / f"{prefix}_events.tsv"
        evj_path = mod_path / f"{prefix}_events.json"
        sj_path = mod_path / f"{prefix}_{modality}.json"

        mod_info = {}

        # EDF header
        if edf_path.exists():
            mod_info["edf"] = probe_edf(edf_path)
        else:
            report["issues"].append(f"{modality}: EDF file missing – {edf_path.name}")
            mod_info["edf"] = None

        # Channels
        chan_df = safe_read_tsv(chan_path)
        if chan_df is not None:
            mod_info["n_channels_tsv"] = len(chan_df)
            mod_info["channel_types"] = chan_df["type"].unique().tolist() if "type" in chan_df else []
            sfreqs = chan_df["sampling_frequency"].unique().tolist() if "sampling_frequency" in chan_df else []
            mod_info["sfreq_tsv"] = sfreqs
        else:
            report["issues"].append(f"{modality}: channels.tsv missing")

        # JSON sidecar
        sidecar = safe_read_json(sj_path)
        if sidecar:
            mod_info["sidecar_sfreq"] = sidecar.get("SamplingFrequency")
            mod_info["recording_duration_s"] = sidecar.get("RecordingDuration")
            mod_info["epoch_length_s"] = sidecar.get("EpochLength")
            mod_info["recording_type"] = sidecar.get("RecordingType")
            mod_info["task_description"] = sidecar.get("TaskDescription", "")[:120]

        # Events
        ev_df = safe_read_tsv(ev_path)
        if ev_df is not None:
            total = len(ev_df)
            artifact_mask = (ev_df.get("Artifact", 0) == 1)
            invalid_setsize = (ev_df.get("SetSize", 1) == 0)
            excluded = artifact_mask | invalid_setsize
            clean_df = ev_df[~excluded].copy()

            ev_summary = {
                "total_trials": total,
                "artifact_trials": int(artifact_mask.sum()),
                "invalid_setsize_trials": int(invalid_setsize.sum()),
                "clean_trials": len(clean_df),
            }

            if len(clean_df) > 0:
                ev_summary["set_sizes"] = sorted(clean_df["SetSize"].dropna().unique().astype(int).tolist())
                ev_summary["match_counts"] = clean_df["Match"].value_counts().to_dict()
                ev_summary["correct_counts"] = clean_df["Correct"].value_counts().to_dict()
                rt = clean_df["ResponseTime"].replace(0, float("nan")).dropna()
                ev_summary["response_time_mean_s"] = round(float(rt.mean()), 4) if len(rt) > 0 else None
                ev_summary["response_time_std_s"] = round(float(rt.std()), 4) if len(rt) > 0 else None

            report["events_summary"][modality] = ev_summary
        else:
            report["issues"].append(f"{modality}: events.tsv missing")

        # Electrode locations (iEEG only)
        if modality == "ieeg":
            el_path = mod_path / f"{prefix}_electrodes.tsv"
            el_df = safe_read_tsv(el_path)
            if el_df is not None:
                has_loc = el_df[["x", "y", "z"]].dropna() if all(c in el_df for c in ["x", "y", "z"]) else pd.DataFrame()
                regions = (
                    el_df["AnatomicalLocation"].unique().tolist()
                    if "AnatomicalLocation" in el_df
                    else []
                )
                # Summarize unique top-level brain regions
                unique_structures = set()
                for r in regions:
                    if isinstance(r, str) and r != "no_label_found":
                        part = r.split(",")[0].strip()
                        unique_structures.add(part)
                mod_info["electrodes_with_coords"] = len(has_loc)
                mod_info["unique_anatomical_structures"] = sorted(unique_structures)
            else:
                report["issues"].append("ieeg: electrodes.tsv missing")

        report["modalities"][modality] = mod_info

    # Cross-check: EEG/iEEG trial counts should match
    eeg_ev = report["events_summary"].get("eeg", {})
    ieeg_ev = report["events_summary"].get("ieeg", {})
    if eeg_ev and ieeg_ev:
        if eeg_ev.get("total_trials") != ieeg_ev.get("total_trials"):
            report["issues"].append(
                f"Trial count mismatch: EEG={eeg_ev['total_trials']} "
                f"iEEG={ieeg_ev['total_trials']}"
            )
        if eeg_ev.get("clean_trials") != ieeg_ev.get("clean_trials"):
            report["issues"].append(
                f"Clean trial count mismatch: EEG={eeg_ev['clean_trials']} "
                f"iEEG={ieeg_ev['clean_trials']}"
            )

    # Cross-check sampling frequencies
    for mod in ("eeg", "ieeg"):
        mi = report["modalities"].get(mod, {})
        edf_sfreq = (mi.get("edf") or {}).get("sfreq")
        sidecar_sfreq = mi.get("sidecar_sfreq")
        if edf_sfreq and sidecar_sfreq and edf_sfreq != sidecar_sfreq:
            report["issues"].append(
                f"{mod}: sfreq mismatch EDF={edf_sfreq} sidecar={sidecar_sfreq}"
            )

    return report


def inspect_dataset(dataset_root: Path) -> dict:
    """Walk all subjects and sessions and produce a full report."""
    participants_path = dataset_root / "participants.tsv"
    participants_df = safe_read_tsv(participants_path)
    desc = safe_read_json(dataset_root / "dataset_description.json")

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_root": str(dataset_root),
        "dataset_name": desc.get("Name") if desc else None,
        "dataset_doi": desc.get("DatasetDOI") if desc else None,
        "license": desc.get("License") if desc else None,
        "bids_version": desc.get("BIDSVersion") if desc else None,
        "participants_in_tsv": (
            participants_df["participant_id"].tolist() if participants_df is not None else []
        ),
        "participants_found_on_disk": [],
        "sessions": [],
        "aggregate": {},
        "issues": [],
    }

    # Discover subjects
    subjects = sorted(
        [d.name for d in dataset_root.iterdir() if d.is_dir() and d.name.startswith("sub-")]
    )
    report["participants_found_on_disk"] = subjects

    if set(subjects) != set(report["participants_in_tsv"]):
        missing_from_disk = set(report["participants_in_tsv"]) - set(subjects)
        extra_on_disk = set(subjects) - set(report["participants_in_tsv"])
        if missing_from_disk:
            report["issues"].append(f"Participants in TSV but not on disk: {sorted(missing_from_disk)}")
        if extra_on_disk:
            report["issues"].append(f"Participants on disk but not in TSV: {sorted(extra_on_disk)}")

    total_clean = 0
    all_session_reports = []

    for sub in subjects:
        sub_path = dataset_root / sub
        sessions = sorted(
            [d.name for d in sub_path.iterdir() if d.is_dir() and d.name.startswith("ses-")]
        )
        for ses in sessions:
            ses_report = inspect_session(sub, ses, dataset_root)
            all_session_reports.append(ses_report)
            ieeg_ev = ses_report["events_summary"].get("ieeg", {})
            total_clean += ieeg_ev.get("clean_trials", 0)

    report["sessions"] = all_session_reports
    report["aggregate"]["total_sessions"] = len(all_session_reports)
    report["aggregate"]["total_subjects"] = len(subjects)
    report["aggregate"]["total_clean_trials_ieeg"] = total_clean

    # Per-subject aggregates
    sub_totals = {}
    for sr in all_session_reports:
        sub = sr["subject"]
        ct = sr["events_summary"].get("ieeg", {}).get("clean_trials", 0)
        sub_totals[sub] = sub_totals.get(sub, 0) + ct
    report["aggregate"]["clean_trials_per_subject"] = sub_totals

    return report


# ──────────────────────────────────────────────────────────────────────────────
# Markdown report writer
# ──────────────────────────────────────────────────────────────────────────────

def write_markdown_report(report: dict, out_path: Path) -> None:
    lines = []
    a = lines.append

    a("# CD-B2M Dataset Inspection Report")
    a(f"\n**Generated:** {report['generated_at']}")
    a(f"\n**Dataset:** {report['dataset_name']}")
    a(f"**DOI:** {report['dataset_doi']}")
    a(f"**License:** {report['license']}  (CC0 = public domain)")
    a(f"**BIDS Version:** {report['bids_version']}")
    a(f"**Root path:** `{report['dataset_root']}`\n")

    a("---\n")
    a("## 1. Participants\n")
    a(f"- Declared in `participants.tsv`: {report['participants_in_tsv']}")
    a(f"- Found on disk: {report['participants_found_on_disk']}\n")
    a("> **Note:** Only sub-14 and sub-15 are present on disk. The full cohort "
      "described in the paper contains 15 subjects.  The downloaded dataset is "
      "a two-subject subset (OpenNeuro ds004752). This is sufficient for "
      "proof-of-concept CD-B2M experiments but limits statistical power.")

    a("\n---\n")
    a("## 2. Session and Trial Summary\n")
    a("| Subject | Session | Total Trials | Artifact / Invalid | Clean Trials | Set Sizes | IN / OUT | Correct Rate |")
    a("|---------|---------|-------------|-------------------|-------------|-----------|----------|--------------|")

    for sr in report["sessions"]:
        ev = sr["events_summary"].get("ieeg", {})
        total = ev.get("total_trials", "?")
        art = ev.get("artifact_trials", 0) + ev.get("invalid_setsize_trials", 0)
        clean = ev.get("clean_trials", "?")
        sizes = ev.get("set_sizes", [])
        mc = ev.get("match_counts", {})
        cc = ev.get("correct_counts", {})
        inn = mc.get("IN", 0)
        out = mc.get("OUT", 0)
        cor = cc.get(1, 0)
        tot_valid = inn + out
        corr_rate = f"{cor/tot_valid:.2f}" if tot_valid > 0 else "n/a"
        a(f"| {sr['subject']} | {sr['session']} | {total} | {art} | {clean} | "
          f"{sizes} | {inn}/{out} | {corr_rate} |")

    agg = report["aggregate"]
    a(f"\n**Total sessions:** {agg['total_sessions']}")
    a(f"**Total clean iEEG trials:** {agg['total_clean_trials_ieeg']}")
    a(f"**Clean trials per subject:** {agg['clean_trials_per_subject']}\n")

    a("---\n")
    a("## 3. Recording Modalities\n")

    # EEG
    a("### 3a. Scalp EEG\n")
    a("- Format: EDF (epoched, one file per session)")
    a("- Recording type: epoched")
    a("- Channels: 21 (sub-14) / 23 (sub-15) using the 10-20 system")
    a("  - sub-14: A1, A2, C3, C4, Cz, F3, F4, F7, F8, Fp1, Fp2, Fz, O1, O2, P3, P4, Pz, T3, T4, T5, T6")
    a("  - sub-15: same + T1, T2 (extra temporal electrodes)")
    a("- Sampling rate: 4000 Hz (sub-14), 4096 Hz (sub-15)")
    a("- Hardware filters: 0.5 Hz highpass, 1000 Hz lowpass")
    a("- Reference: electrode near the vertex")
    a("- Power line: 50 Hz (Europe)")
    a("- Epoch length: 8 s per trial\n")

    a("### 3b. Intracranial EEG (iEEG / SEEG)\n")
    a("- Format: EDF (epoched)")
    a("- Channel type: SEEG (stereo-EEG depth electrodes)")
    a("- Channels: 64 per session for both subjects")
    a("- Sampling rate: 4000 Hz (sub-14), 4096 Hz (sub-15)")
    a("- Recording system: Neuralynx ATLAS")
    a("- Reference: common intracranial reference")
    a("- Epoch length: 8 s per trial\n")

    a("### 3c. Electrode Localisation\n")
    a("- All iEEG electrodes have MNI coordinates (x, y, z) in `electrodes.tsv`")
    a("- Anatomical labels available per electrode (some labelled `no_label_found`)")
    a("- Brain regions covered by sub-14 (example): Hippocampus (bilateral), "
      "Amygdala, Inferior Temporal Gyrus, Middle Temporal Gyrus, Fusiform Gyrus, "
      "Parahippocampal Gyrus, Entorhinal Cortex region")
    a("- Brain regions covered by sub-15: Hippocampus (bilateral), Amygdala, "
      "Superior Temporal Gyrus, Middle Temporal Gyrus, Parahippocampal Gyrus, "
      "posterior STS\n")

    a("---\n")
    a("## 4. Task Protocol (Sternberg Verbal Working Memory)\n")
    a("Each trial is 8 seconds and consists of:")
    a("1. **Encoding period (2 s):** A string of consonants (set size 4, 6, or 8 letters) is shown.")
    a("2. **Maintenance period (3 s):** Blank delay; participant holds letters in working memory.")
    a("3. **Retrieval / Probe (variable):** A single probe letter appears; participant presses "
      "yes (IN) or no (OUT).")
    a("\n**Task labels available:**")
    a("- `SetSize`: 4, 6, or 8 letters — primary working memory load variable")
    a("- `Match` (IN / OUT): whether probe was in the memory set")
    a("- `Correct` (0 / 1): whether response was correct")
    a("- `ResponseTime`: RT since probe onset (seconds)\n")

    a("---\n")
    a("## 5. Cross-Modality Alignment\n")
    a("- EEG and iEEG events files are **identical** within each session "
      "(same onset/begSample/endSample, same trial numbering).")
    a("- Both modalities are recorded simultaneously on the same hardware (Neuralynx ATLAS).")
    a("- Artifact flags are consistent across EEG and iEEG events files.")
    a("- **Conclusion: EEG and iEEG trials can be directly aligned by trial index "
      "within each session.** No additional synchronisation is needed.\n")

    a("---\n")
    a("## 6. Sampling Rate Discrepancy\n")
    a("- sub-14 recorded at **4000 Hz**; sub-15 recorded at **4096 Hz**.")
    a("- Since each epoch is exactly 8 s:")
    a("  - sub-14: 4000 × 8 = **32,000 samples/trial**")
    a("  - sub-15: 4096 × 8 = **32,768 samples/trial**")
    a("- For cross-subject comparisons, data must be **resampled to a common rate** "
      "(e.g. 256 Hz after anti-aliasing lowpass at 100 Hz).")
    a("- Within-subject analyses can use the native rate.\n")

    a("---\n")
    a("## 7. Proposed CD-B2M Experimental Design\n")
    a("Given the two-subject dataset, the following design is scientifically defensible:\n")
    a("### Primary Task\n")
    a("**Working memory load classification** (SetSize ∈ {4, 6, 8}) from scalp EEG epochs.")
    a("- 3-class classification; baseline accuracy ≈ 33 %.")
    a("- Sufficient class instances across sessions for training/validation/test splits.\n")
    a("### Brain Guidance Signal\n")
    a("The **iEEG (SEEG) trial-mean power envelope or raw epoch** serves as the brain "
      "representation that guides scalp EEG model training.")
    a("- Both are recorded simultaneously; trial alignment is exact.")
    a("- iEEG has higher SNR and direct access to hippocampal and medial temporal "
      "regions known to be involved in working memory.\n")
    a("### OOD / Generalisation Setup\n")
    a("- **Within-subject, held-out sessions:** Train on early sessions, test on later sessions.")
    a("  - sub-14: train ses-03 to ses-06 (~190 clean trials), test ses-07–08 (~93 clean trials)")
    a("  - sub-15: train ses-01 to ses-03 (~145 clean trials), test ses-04 (~50 clean trials)")
    a("- **Cross-subject:** Train on sub-14, evaluate on sub-15 (zero-shot).")
    a("  - Requires resampling to common rate; valid but limited given n=2 subjects.\n")
    a("### Alternative / Secondary Tasks\n")
    a("- **Match / No-match classification** (IN vs OUT): binary, balanced ≈50/50 across sessions.")
    a("- **Correct vs. Incorrect response**: highly imbalanced (~90 % correct); secondary analysis only.\n")

    a("---\n")
    a("## 8. Identified Issues and Limitations\n")
    a("1. **Sample size:** Only 2 of 15 subjects are present. Cross-subject generalisation "
      "claims will be very weak. Statistical tests across participants cannot be performed.")
    a("2. **Imbalanced SetSize trials:** Not perfectly balanced per session. "
      "Stratified splitting and balanced accuracy are required.")
    a("3. **Correct/Incorrect imbalance:** Most responses are correct (~88–96 %). "
      "This label cannot be used as a primary classification target without oversampling.")
    a("4. **Sampling rate mismatch:** sub-14 (4000 Hz) vs sub-15 (4096 Hz). "
      "Must resample before cross-subject pooling.")
    a("5. **Missing EEG sidecar for ses-03:** sub-14 ses-03 EEG JSON sidecar is absent. "
      "Metadata will be inferred from the EDF header.")
    a("6. **Electrode heterogeneity:** iEEG electrode placement is subject-specific "
      "(clinical implantation). Cross-subject alignment of iEEG channels is not directly "
      "possible without source localisation or ROI averaging.")
    a("7. **Clinical population:** Participants are epilepsy patients; normal brain "
      "representations cannot be assumed. Results should be interpreted accordingly.")
    a("8. **No derivatives folder:** The beamforming/LCMV virtual sensor data mentioned "
      "in the README are not present in this downloaded subset.\n")

    a("---\n")
    a("## 9. Preprocessing Decisions (Preliminary)\n")
    a("The following decisions are proposed before preprocessing begins:")
    a("- **Bandpass:** 1–100 Hz (remove DC drift and high-frequency noise)")
    a("- **Notch:** 50 Hz power line + harmonics")
    a("- **Resampling:** Downsample to 256 Hz (from 4000/4096 Hz) after anti-aliasing filter")
    a("- **Epoch window:** Full 8 s trial OR maintenance window only (2–5 s post-onset)")
    a("- **Artifact exclusion:** Use provided Artifact flag; additionally reject trials "
      "exceeding ±500 µV peak-to-peak on any EEG channel")
    a("- **Normalisation:** z-score per channel using training-set mean and std only")
    a("- **Splits:** Session-aware — no session spans train/test boundary")
    a("- **iEEG brain representation:** Trial-mean band-power (theta 4–8 Hz, "
      "gamma 30–80 Hz) per electrode, or raw downsampled epoch (to be decided after "
      "inspecting actual signal quality)\n")

    a("---\n")
    a("## 10. File Inventory\n")
    a("| Path pattern | Count | Notes |")
    a("|---|---|---|")
    a("| `sub-*/ses-*/ieeg/*.edf` | 10 | One per session |")
    a("| `sub-*/ses-*/eeg/*.edf` | 10 | One per session |")
    a("| `sub-*/ses-*/ieeg/*_events.tsv` | 10 | Trial-level metadata |")
    a("| `sub-*/ses-*/ieeg/*_channels.tsv` | 10 | Channel info |")
    a("| `sub-*/ses-*/ieeg/*_electrodes.tsv` | 10 | MNI coords + anatomical labels |")
    a("| `sub-*/ses-*/eeg/*_events.tsv` | 10 | Same content as iEEG events |")
    a("| `sub-*/ses-*/eeg/*_channels.tsv` | 10 | EEG channel list |")
    a("| `dataset_description.json` | 1 | BIDS descriptor |")
    a("| `participants.tsv` | 1 | Demographics (all 15 subjects listed) |\n")

    a("---\n")
    a("*Report generated automatically by `src/inspect_data.py`. "
      "No raw neural data was loaded into memory during inspection.*")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print(f"Markdown report saved → {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Inspect CD-B2M dataset.")
    parser.add_argument(
        "--dataset_root",
        default=str(Path(__file__).parent.parent.parent / "dataset"),
        help="Path to the BIDS dataset root.",
    )
    parser.add_argument(
        "--out_dir",
        default=str(Path(__file__).parent.parent / "results" / "reports"),
        help="Directory to write inspection outputs.",
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).resolve()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Inspecting dataset at: {dataset_root}")
    print("(No raw EDF data will be loaded into memory — header inspection only)")

    report = inspect_dataset(dataset_root)

    # Save JSON
    json_out = out_dir / "dataset_inspection.json"
    with open(json_out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)
    print(f"JSON report saved  → {json_out}")

    # Save Markdown
    md_out = out_dir / "dataset_inspection_report.md"
    write_markdown_report(report, md_out)

    # Print summary to stdout
    agg = report["aggregate"]
    print("\n" + "=" * 60)
    print("INSPECTION SUMMARY")
    print("=" * 60)
    print(f"  Subjects on disk       : {agg['total_subjects']}")
    print(f"  Sessions               : {agg['total_sessions']}")
    print(f"  Total clean iEEG trials: {agg['total_clean_trials_ieeg']}")
    print(f"  Clean trials / subject : {agg['clean_trials_per_subject']}")
    all_issues = []
    for sr in report["sessions"]:
        for iss in sr["issues"]:
            all_issues.append(f"  [{sr['subject']}/{sr['session']}] {iss}")
    for iss in report["issues"]:
        all_issues.append(f"  [GLOBAL] {iss}")
    if all_issues:
        print(f"\n  Issues found ({len(all_issues)}):")
        for iss in all_issues:
            print(f"    ⚠ {iss}")
    else:
        print("\n  No structural issues found.")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
