# CD-B2M: Experimental Methodology

## 1. Research Question

Does brain-guided AI training improve generalization because brain representations
contain **genuinely task-relevant information** (the strong hypothesis), or because
they act as a **generic structured regularizer** (the weak hypothesis)?

---

## 2. Dataset

**OpenNeuro ds004752** — Simultaneous scalp EEG and intracranial SEEG from epilepsy
patients performing a Sternberg Verbal Working Memory task.

| Property | Value |
|----------|-------|
| Subjects (downloaded) | 2 of 15 (sub-14, sub-15) |
| Sessions | 6 (sub-14), 4 (sub-15) |
| Trials per session | ~50 |
| Total clean trials | 371 (after artifact rejection) |
| EEG channels | 21 (10-20 system) |
| iEEG channels | 64 SEEG depth electrodes per subject |
| Sampling rate | 4000 Hz (sub-14), 4096 Hz (sub-15) |
| Task | Sternberg verbal working memory |
| Primary label | SetSize ∈ {4, 6, 8} |

---

## 3. Preprocessing Decisions

### 3.1 EEG Processing
- **Bandpass:** 1–100 Hz (zero-phase FIR, firwin design)
- **Notch:** 50 Hz (European power line)
- **Channel selection:** 21-channel intersection across subjects (A1, A2, C3, C4, Cz,
  F3, F4, F7, F8, Fp1, Fp2, Fz, O1, O2, P3, P4, Pz, T3, T4, T5, T6)
- **Epoch window:** 1–5 s post-trial-onset (encoding + maintenance; avoids probe period)
- **Resampling:** 4000/4096 Hz → 256 Hz via polyphase `resample_poly` (per-epoch)
- **Amplitude QC:** Reject epochs with any channel peak-to-peak ≥ 1000 units
- **Normalisation:** z-score per channel, statistics from training set only

### 3.2 iEEG Brain Representations
- **Filter:** 0.5–200 Hz bandpass + 50 Hz notch
- **Representation:** Log₁₀ Welch band-power in two bands:
  - Theta: 4–8 Hz
  - Gamma: 30–80 Hz
- **Per trial:** 64 channels × 2 bands = **128-dimensional vector**
- **Justification:** Theta and gamma are established WM-relevant oscillations;
  hippocampal theta and cortical gamma are implicated in WM encoding and replay
- **Normalisation:** feature-wise z-score, training statistics only

### 3.3 Alignment
EEG and iEEG events files are **identical per session** (same hardware, same onset
timestamps). Trial alignment is exact by index within each session. No
cross-session alignment is needed.

---

## 4. Split Strategy

**Session-aware split** — prevents temporal leakage:

| Subject | Train sessions | Val (random 15% of train pool) | Test sessions |
|---------|---------------|-------------------------------|---------------|
| sub-14  | ses-04, ses-05 | stratified from train pool | ses-07, ses-08 |
| sub-15  | ses-01, ses-02, ses-03 | stratified from train pool | ses-04 |

- ses-03 (sub-14) and ses-06 (sub-14) skipped: corrupt EDF headers
- Val split is **stratified by SetSize** to preserve class distribution
- Test set = held-out sessions only; never seen during training or normalisation

**Final sizes:** Train=197, Val=35, Test=139

---

## 5. Model Architecture

**Unidirectional GRU Classifier**

```
Input: (batch, 21, 1024)      ← EEG: channels as features, time as sequence
  ↓ permute → (batch, 1024, 21)
GRU(input=21, hidden=64, layers=1)
  ↓ final hidden state (batch, 64)
Classifier:
  Dropout(0.3) → Linear(64→64) → ReLU → Dropout(0.3) → Linear(64→3)
  ↓ logits (batch, 3)

Projection head (brain-guided only):
  Linear(64→32) → ReLU → Linear(32→128)
  ↓ projected representation (batch, 128)
```

**Design rationale:**
- GRU chosen over Transformer for small dataset (less overfit risk)
- 1 layer, hidden=64: deliberately small given n=197 training trials
- Projection head maps EEG latent → brain representation space for MSE alignment

---

## 6. Training Protocol

| Hyperparameter | Value |
|----------------|-------|
| Optimiser | Adam |
| Learning rate | 1e-3 |
| Weight decay | 1e-4 |
| Batch size | 16 |
| Max epochs | 80 |
| Early stopping patience | 12 epochs |
| Gradient clipping | 1.0 |
| Seeds | {42, 123} |
| Brain alignment α values | {0.1, 0.5} |

**Loss function:**
```
L_total = (1 − α) × CrossEntropy(logits, SetSize) + α × MSE(proj(h), brain_rep)
```

For baseline (condition A): α = 0, only cross-entropy.

---

## 7. Control Conditions

### C. Statistically Matched Synthetic Control
- Generated from **training brain data only** (no test information)
- Multivariate Gaussian with matched mean vector and full covariance matrix
- Verified: per-feature mean error < 0.05, std error < 0.05
- **What it tests:** Whether matching the Gaussian covariance structure is sufficient
  to replicate real brain benefits
- **Limitation:** Does not match higher-order statistics, non-Gaussian marginals,
  or non-stationarity of neural signals

### D. Shuffled Brain Control
- Trial indices permuted **within each split** (train shuffled separately from val/test)
- Marginal distribution preserved exactly (same values, different assignment)
- Trial-specific EEG↔iEEG correspondence destroyed
- **What it tests:** Whether trial-specific correspondence matters, vs. marginal stats

### E. Random Gaussian Control
- Standard Gaussian noise, dimension-matched (128-dim) only
- Does not match mean, std, or covariance
- **What it tests:** Whether any structured representation helps vs. pure noise

---

## 8. Evaluation Protocol

### 8.1 Primary Metric
**Balanced accuracy** on the held-out test set (SetSize 3-class classification).
Chosen because SetSize distribution is slightly imbalanced across sessions.

### 8.2 Additional Metrics
- Accuracy
- Macro-F1
- Confusion matrix (per condition, per seed)
- Per-subject balanced accuracy

### 8.3 OOD Generalization
The test set = **held-out sessions** from the same subjects but not seen during
training. This constitutes within-subject temporal OOD evaluation.
Cross-subject evaluation (train sub-14, test sub-15) is also performed.

### 8.4 Noise Robustness
Gaussian noise added to EEG inputs at σ ∈ {0.0, 0.5, 1.0, 2.0, 4.0} × data std.
Evaluates degradation of each condition under input perturbation.

### 8.5 Representation Alignment
- **RSA (ρ):** Spearman correlation between EEG projection RDM and brain RDM
- **Linear CKA:** Centred Kernel Alignment between projected EEG and brain reps

---

## 9. Primary Statistical Comparison

**Pre-defined primary comparison:** Real Brain (B) vs. Synthetic Matched Control (C)
on balanced accuracy on the test set.

- If B > C: evidence consistent with trial-specific brain content contributing
- If B ≈ C: Gaussian covariance structure may explain the benefit
- If B < C: brain alignment may be detrimental in this setting

**Statistical note:** With n=2 subjects and n=2 seeds, no test reaches conventional
significance. Results are exploratory and directional only. Cohen's d is reported
as an effect size estimate.

---

## 10. What This Study Can and Cannot Establish

**Can establish:**
- Whether brain-guided training numerically improves a specific EEG classifier
  on a specific task with this dataset
- Whether statistically matched synthetic reps produce similar or different results
- Whether trial-specific correspondence (shuffled control) matters

**Cannot establish:**
- Causal mechanisms (correlation ≠ causation)
- Generalization across subjects (n=2 is insufficient)
- Whether results hold for other architectures, tasks, or brain signals
- Whether iEEG represents "ground truth" neural computation

---

*Document generated by the CD-B2M pipeline. Last updated: October 2026.*
