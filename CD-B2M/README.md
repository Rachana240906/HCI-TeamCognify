# CD-B2M: Causally-Disentangled Brain-to-Model Transfer

**Research Question:** Does brain-guided AI training improve generalization because
brain representations contain genuinely task-relevant information, or because they
act as a generic structured regularizer?

---

## Quick Start

```bash
# 1. Activate the virtual environment
cd d:\sem5\HCI\CD-B2M-Project
.venv\Scripts\activate

# 2. Move into the project
cd CD-B2M

# 3. Install dependencies (already satisfied in .venv)
pip install -r requirements.txt

# 4. Full pipeline (preprocessing already done — skip it)
python run_all.py --skip_preprocess

# 5. Smoke test (fast end-to-end check, ~5 min)
python run_all.py --smoke_test
```

---

## Project Structure

```
CD-B2M/
├── configs/
│   └── config.yaml           # All hyperparameters, paths, splits
├── src/
│   ├── inspect_data.py       # Phase 1–2: Dataset inspection
│   ├── preprocess.py         # Phase 3:   EEG/iEEG preprocessing
│   ├── dataset.py            # PyTorch Dataset classes
│   ├── model.py              # GRU classifier + projection head
│   ├── alignment.py          # Brain alignment loss, RSA, CKA
│   ├── controls.py           # Phase 6:   Control condition generators
│   ├── train.py              # Phase 4–5: Training loop (all conditions)
│   ├── evaluate.py           # Phase 7–8: Evaluation + robustness
│   ├── statistics.py         # Phase 9:   Statistical analysis
│   └── visualize.py          # Phase 9:   Publication-quality plots
├── tests/
│   └── test_preprocessing.py # 29 unit tests
├── data/
│   ├── raw/                  # (empty — data lives in ../dataset/)
│   └── processed/            # Preprocessed .npz files
├── results/
│   ├── metrics/              # CSV + JSON evaluation results
│   ├── plots/                # All figures (PNG + CSV data)
│   ├── checkpoints/          # Model checkpoints (.pt)
│   └── reports/              # Inspection + preprocessing logs
├── logs/                     # Training logs
├── docs/                     # Methodology + final report
├── run_all.py                # Master runner
└── requirements.txt
```

---

## Dataset

**OpenNeuro ds004752** (CC0 License)
- 2 subjects: sub-14, sub-15
- 10 sessions total (~50 trials/session)
- Simultaneous **scalp EEG** (21 ch, 10-20 system) + **iEEG/SEEG** (64 ch depth electrodes)
- Task: Sternberg Verbal Working Memory — encode letters (2 s), maintain (3 s), probe (IN/OUT)
- Labels: `SetSize` ∈ {4, 6, 8} (primary), `Match` IN/OUT (secondary), `Correct` 0/1
- **DOI:** https://doi.org/10.18112/openneuro.ds004752.v1.0.1

```
../dataset/
  sub-14/ses-03..08/eeg/*.edf    ← scalp EEG
  sub-14/ses-03..08/ieeg/*.edf   ← intracranial SEEG
  sub-15/ses-01..04/eeg/*.edf
  sub-15/ses-01..04/ieeg/*.edf
```

---

## Preprocessing

```bash
python src/preprocess.py --config configs/config.yaml
```

Steps applied:
1. Channel selection — common 21-ch 10-20 EEG set across both subjects
2. Bandpass filter 1–100 Hz (zero-phase FIR)
3. Notch filter 50 Hz
4. Epoch extraction by sample index (1–5 s maintenance window)
5. Per-epoch polyphase resample to 256 Hz
6. Amplitude QC: reject trials > 1000 units peak-to-peak
7. iEEG brain representation: log₁₀ band-power (θ 4–8 Hz, γ 30–80 Hz) → 128-dim vector
8. Session-aware split: held-out test sessions per subject
9. Z-score normalisation (training statistics only)

**Output shapes:**
- EEG: `(N, 21, 1024)` — 21 channels × 1024 time points at 256 Hz × 4 s
- Brain: `(N, 128)` — 64 iEEG channels × 2 frequency bands
- Train: 197 trials | Val: 35 | Test: 139

---

## Experimental Conditions

| Condition | Description |
|-----------|-------------|
| A. Baseline | Task loss only (cross-entropy) |
| B. Real Brain | Task + MSE alignment to iEEG band-power |
| C. Synthetic | Task + alignment to multivariate Gaussian matched to training brain stats |
| D. Shuffled | Task + alignment to permuted brain reps (marginal preserved, correspondence broken) |
| E. Random | Task + alignment to pure Gaussian noise (dimension-matched only) |

**Loss:** `L_total = (1 − α) × L_task + α × L_brain`

---

## Model

- **Architecture:** Unidirectional GRU (1 layer, hidden=64)
- **Input:** (batch, 21 channels, 1024 timesteps) — channels as features per timestep
- **Classifier:** GRU hidden → Linear(64) → ReLU → Linear(3)
- **Projection head** (brain-guided only): hidden → Linear(32) → ReLU → Linear(128)
- **Training:** Adam lr=1e-3, early stopping patience=12, max 80 epochs

---

## Running Individual Scripts

```bash
# Dataset inspection (generates report)
python src/inspect_data.py

# Controls generation (must run after preprocess)
python src/controls.py

# Train all conditions
python src/train.py

# Train one condition
python src/train.py --condition baseline --seed 42

# Evaluate
python src/evaluate.py

# Statistics
python src/statistics.py

# Visualizations
python src/visualize.py
```

---

## Tests

```bash
python -m pytest tests/test_preprocessing.py -v
# Expected: 29 passed
```

---

## Key Outputs

| File | Description |
|------|-------------|
| `results/reports/dataset_inspection_report.md` | Full dataset analysis |
| `results/reports/preprocessing_log.json` | Preprocessing audit trail |
| `results/reports/controls_audit.json` | Control condition verification |
| `results/metrics/training_results.json` | Per-epoch training history |
| `results/metrics/evaluation_results.csv` | All metrics per condition/seed |
| `results/metrics/stats_summary.csv` | Aggregated statistics |
| `results/reports/stats_interpretation.txt` | Research question interpretation |
| `results/plots/fig*.png` | All publication figures |
| `docs/methodology.md` | Full experimental methodology |
| `docs/final_report.md` | Research findings and discussion |

---

## Limitations

1. Only 2/15 subjects downloaded — cross-subject generalisation claims are very weak
2. No cross-subject statistical test possible (n < 3)
3. Synthetic control is multivariate Gaussian — does not match non-Gaussian marginals
4. Clinical epilepsy population — normal brain representations cannot be assumed
5. Electrode placement is subject-specific — iEEG channels not anatomically aligned across subjects

---

## Citation

```
Dimakopoulos V, Mégevand P, Stieglitz LH, Imbach L, Sarnthein J (2022).
Information flows from hippocampus to auditory cortex during replay of
verbal working memory items. eLife 11:e78677.
https://doi.org/10.7554/eLife.78677
```
