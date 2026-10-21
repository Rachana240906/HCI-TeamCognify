# CD-B2M Dataset Inspection Report

**Generated:** 2026-10-20T15:24:19

**Dataset:** Dataset of intracranial EEG, scalp EEG and beamforming sources from epilepsy patients performing a verbal working memory task
**DOI:** doi:10.18112/openneuro.ds004752.v1.0.1
**License:** CC0  (CC0 = public domain)
**BIDS Version:** 1.4.0
**Root path:** `D:\sem5\HCI\CD-B2M-Project\dataset`

---

## 1. Participants

- Declared in `participants.tsv`: ['sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-06', 'sub-07', 'sub-08', 'sub-09', 'sub-10', 'sub-11', 'sub-12', 'sub-13', 'sub-14', 'sub-15']
- Found on disk: ['sub-14', 'sub-15']

> **Note:** Only sub-14 and sub-15 are present on disk. The full cohort described in the paper contains 15 subjects.  The downloaded dataset is a two-subject subset (OpenNeuro ds004752). This is sufficient for proof-of-concept CD-B2M experiments but limits statistical power.

---

## 2. Session and Trial Summary

| Subject | Session | Total Trials | Artifact / Invalid | Clean Trials | Set Sizes | IN / OUT | Correct Rate |
|---------|---------|-------------|-------------------|-------------|-----------|----------|--------------|
| sub-14 | ses-03 | 50 | 2 | 49 | [4, 6, 8] | 25/24 | 0.92 |
| sub-14 | ses-04 | 50 | 6 | 47 | [4, 6, 8] | 25/22 | 0.89 |
| sub-14 | ses-05 | 48 | 4 | 46 | [4, 6, 8] | 28/18 | 0.87 |
| sub-14 | ses-06 | 50 | 4 | 48 | [4, 6, 8] | 24/24 | 0.94 |
| sub-14 | ses-07 | 50 | 6 | 47 | [4, 6, 8] | 26/21 | 0.94 |
| sub-14 | ses-08 | 50 | 7 | 46 | [4, 6, 8] | 28/18 | 0.91 |
| sub-15 | ses-01 | 50 | 0 | 50 | [4, 6, 8] | 24/26 | 0.86 |
| sub-15 | ses-02 | 50 | 0 | 50 | [4, 6, 8] | 20/30 | 0.96 |
| sub-15 | ses-03 | 50 | 0 | 50 | [4, 6, 8] | 28/22 | 0.90 |
| sub-15 | ses-04 | 50 | 0 | 50 | [4, 6, 8] | 32/18 | 0.86 |

**Total sessions:** 10
**Total clean iEEG trials:** 483
**Clean trials per subject:** {'sub-14': 283, 'sub-15': 200}

---

## 3. Recording Modalities

### 3a. Scalp EEG

- Format: EDF (epoched, one file per session)
- Recording type: epoched
- Channels: 21 (sub-14) / 23 (sub-15) using the 10-20 system
  - sub-14: A1, A2, C3, C4, Cz, F3, F4, F7, F8, Fp1, Fp2, Fz, O1, O2, P3, P4, Pz, T3, T4, T5, T6
  - sub-15: same + T1, T2 (extra temporal electrodes)
- Sampling rate: 4000 Hz (sub-14), 4096 Hz (sub-15)
- Hardware filters: 0.5 Hz highpass, 1000 Hz lowpass
- Reference: electrode near the vertex
- Power line: 50 Hz (Europe)
- Epoch length: 8 s per trial

### 3b. Intracranial EEG (iEEG / SEEG)

- Format: EDF (epoched)
- Channel type: SEEG (stereo-EEG depth electrodes)
- Channels: 64 per session for both subjects
- Sampling rate: 4000 Hz (sub-14), 4096 Hz (sub-15)
- Recording system: Neuralynx ATLAS
- Reference: common intracranial reference
- Epoch length: 8 s per trial

### 3c. Electrode Localisation

- All iEEG electrodes have MNI coordinates (x, y, z) in `electrodes.tsv`
- Anatomical labels available per electrode (some labelled `no_label_found`)
- Brain regions covered by sub-14 (example): Hippocampus (bilateral), Amygdala, Inferior Temporal Gyrus, Middle Temporal Gyrus, Fusiform Gyrus, Parahippocampal Gyrus, Entorhinal Cortex region
- Brain regions covered by sub-15: Hippocampus (bilateral), Amygdala, Superior Temporal Gyrus, Middle Temporal Gyrus, Parahippocampal Gyrus, posterior STS

---

## 4. Task Protocol (Sternberg Verbal Working Memory)

Each trial is 8 seconds and consists of:
1. **Encoding period (2 s):** A string of consonants (set size 4, 6, or 8 letters) is shown.
2. **Maintenance period (3 s):** Blank delay; participant holds letters in working memory.
3. **Retrieval / Probe (variable):** A single probe letter appears; participant presses yes (IN) or no (OUT).

**Task labels available:**
- `SetSize`: 4, 6, or 8 letters — primary working memory load variable
- `Match` (IN / OUT): whether probe was in the memory set
- `Correct` (0 / 1): whether response was correct
- `ResponseTime`: RT since probe onset (seconds)

---

## 5. Cross-Modality Alignment

- EEG and iEEG events files are **identical** within each session (same onset/begSample/endSample, same trial numbering).
- Both modalities are recorded simultaneously on the same hardware (Neuralynx ATLAS).
- Artifact flags are consistent across EEG and iEEG events files.
- **Conclusion: EEG and iEEG trials can be directly aligned by trial index within each session.** No additional synchronisation is needed.

---

## 6. Sampling Rate Discrepancy

- sub-14 recorded at **4000 Hz**; sub-15 recorded at **4096 Hz**.
- Since each epoch is exactly 8 s:
  - sub-14: 4000 × 8 = **32,000 samples/trial**
  - sub-15: 4096 × 8 = **32,768 samples/trial**
- For cross-subject comparisons, data must be **resampled to a common rate** (e.g. 256 Hz after anti-aliasing lowpass at 100 Hz).
- Within-subject analyses can use the native rate.

---

## 7. Proposed CD-B2M Experimental Design

Given the two-subject dataset, the following design is scientifically defensible:

### Primary Task

**Working memory load classification** (SetSize ∈ {4, 6, 8}) from scalp EEG epochs.
- 3-class classification; baseline accuracy ≈ 33 %.
- Sufficient class instances across sessions for training/validation/test splits.

### Brain Guidance Signal

The **iEEG (SEEG) trial-mean power envelope or raw epoch** serves as the brain representation that guides scalp EEG model training.
- Both are recorded simultaneously; trial alignment is exact.
- iEEG has higher SNR and direct access to hippocampal and medial temporal regions known to be involved in working memory.

### OOD / Generalisation Setup

- **Within-subject, held-out sessions:** Train on early sessions, test on later sessions.
  - sub-14: train ses-03 to ses-06 (~190 clean trials), test ses-07–08 (~93 clean trials)
  - sub-15: train ses-01 to ses-03 (~145 clean trials), test ses-04 (~50 clean trials)
- **Cross-subject:** Train on sub-14, evaluate on sub-15 (zero-shot).
  - Requires resampling to common rate; valid but limited given n=2 subjects.

### Alternative / Secondary Tasks

- **Match / No-match classification** (IN vs OUT): binary, balanced ≈50/50 across sessions.
- **Correct vs. Incorrect response**: highly imbalanced (~90 % correct); secondary analysis only.

---

## 8. Identified Issues and Limitations

1. **Sample size:** Only 2 of 15 subjects are present. Cross-subject generalisation claims will be very weak. Statistical tests across participants cannot be performed.
2. **Imbalanced SetSize trials:** Not perfectly balanced per session. Stratified splitting and balanced accuracy are required.
3. **Correct/Incorrect imbalance:** Most responses are correct (~88–96 %). This label cannot be used as a primary classification target without oversampling.
4. **Sampling rate mismatch:** sub-14 (4000 Hz) vs sub-15 (4096 Hz). Must resample before cross-subject pooling.
5. **Missing EEG sidecar for ses-03:** sub-14 ses-03 EEG JSON sidecar is absent. Metadata will be inferred from the EDF header.
6. **Electrode heterogeneity:** iEEG electrode placement is subject-specific (clinical implantation). Cross-subject alignment of iEEG channels is not directly possible without source localisation or ROI averaging.
7. **Clinical population:** Participants are epilepsy patients; normal brain representations cannot be assumed. Results should be interpreted accordingly.
8. **No derivatives folder:** The beamforming/LCMV virtual sensor data mentioned in the README are not present in this downloaded subset.

---

## 9. Preprocessing Decisions (Preliminary)

The following decisions are proposed before preprocessing begins:
- **Bandpass:** 1–100 Hz (remove DC drift and high-frequency noise)
- **Notch:** 50 Hz power line + harmonics
- **Resampling:** Downsample to 256 Hz (from 4000/4096 Hz) after anti-aliasing filter
- **Epoch window:** Full 8 s trial OR maintenance window only (2–5 s post-onset)
- **Artifact exclusion:** Use provided Artifact flag; additionally reject trials exceeding ±500 µV peak-to-peak on any EEG channel
- **Normalisation:** z-score per channel using training-set mean and std only
- **Splits:** Session-aware — no session spans train/test boundary
- **iEEG brain representation:** Trial-mean band-power (theta 4–8 Hz, gamma 30–80 Hz) per electrode, or raw downsampled epoch (to be decided after inspecting actual signal quality)

---

## 10. File Inventory

| Path pattern | Count | Notes |
|---|---|---|
| `sub-*/ses-*/ieeg/*.edf` | 10 | One per session |
| `sub-*/ses-*/eeg/*.edf` | 10 | One per session |
| `sub-*/ses-*/ieeg/*_events.tsv` | 10 | Trial-level metadata |
| `sub-*/ses-*/ieeg/*_channels.tsv` | 10 | Channel info |
| `sub-*/ses-*/ieeg/*_electrodes.tsv` | 10 | MNI coords + anatomical labels |
| `sub-*/ses-*/eeg/*_events.tsv` | 10 | Same content as iEEG events |
| `sub-*/ses-*/eeg/*_channels.tsv` | 10 | EEG channel list |
| `dataset_description.json` | 1 | BIDS descriptor |
| `participants.tsv` | 1 | Demographics (all 15 subjects listed) |

---

*Report generated automatically by `src/inspect_data.py`. No raw neural data was loaded into memory during inspection.*