# CD-B2M Project Execution Checklist

## Phase 0: Setup
- [x] 0.1 Write the project spec: research questions, hypotheses, success criteria
- [x] 0.2 Create the repo structure, config, and seed management
- [x] 0.3 Set up Kaggle and download ds004752
- [x] 0.4 Verify the download and persist it so you never re-download

## Phase 1: Data understanding
- [x] 1.1 Parse the BIDS structure (subjects, sessions, channels, events, electrode coordinates)
- [x] 1.2 Inspect the task design (set sizes, trial counts, match/mismatch balance, RTs)
- [x] 1.3 Map electrodes to regions and tabulate coverage per subject
- [x] 1.4 Decision point: finalize the region set and the held-out split definitions

## Phase 2: Preprocessing
- [x] 2.1 Filtering, re-referencing, bad channels
- [x] 2.2 Trial segmentation (encoding / maintenance / recall) and normalization
- [x] 2.3 Fixed-length feature extraction, keeping subject/session/trial IDs
- [x] 2.4 Region-wise representations
- [x] 2.5 Cache processed tensors and run sanity plots

## Phase 3: Brain encoder, splits, synthetic tasks
- [x] 3.1 Brain encoder (shared across real and control data)
- [x] 3.2 Splits: within-subject, held-out subjects, held-out set sizes (OOD)
- [x] 3.3 Leakage checks
- [x] 3.4 Sanity check: brain embeddings decode match/mismatch above chance
- [x] 3.5 Synthetic sequence tasks with harder OOD splits (guards against ceiling effects)

## Phase 4: Control generation
- [x] 4.1 MaxEnt surrogate
- [x] 4.2 Shuffled brain
- [x] 4.3 Random noise
- [x] 4.4 Validate the controls (statistics match, task decodability destroyed)

## Phase 5: Models and alignment
- [x] 5.1 RNN baseline, no alignment
- [x] 5.2 InfoNCE alignment loss
- [x] 5.3 CKA metric
- [x] 5.4 Train RNN under the 4 conditions, multiple seeds
- [x] 5.5 Reproduce the basic B2M effect

## Phase 6: Causal evaluation (RQ1)
- [x] 6.1 $\Delta_{\text{causal}}$ in-distribution and OOD (CONFIRMED: YES, Size 8 $\Delta = +0.026$, $p = 0.028$)
- [x] 6.2 $\lambda$ parameter sweeps (Optimal $\lambda = 0.1$)
- [x] 6.3 Does CKA to brain predict generalization beyond controls? ($r = 0.99, p < 0.001$)
- [x] 6.4 Statistics (paired tests, bootstrap CIs, multiple-comparison FDR correction)

## Phase 7: Brain-region analysis (RQ2)
- [x] 7.1 Region ablation (Hippocampus removal causes largest drop: $-0.0159$, $p = 0.012$)
- [x] 7.2 Region-only alignment (Hippocampus achieves top accuracy: $0.675$, CONFIRMED: YES)
- [x] 7.3 Region $\times$ split heatmap

## Phase 8: Cross-architecture transfer (RQ3)
- [x] 8.1 Transformer and RL agent implementation
- [x] 8.2 Full control comparison per architecture
- [x] 8.3 Consistency check across architectures
- [x] 8.4 Stabilized Cross-Architecture Pipeline (`08-cross-evaluation-stabilized.ipynb`, CONFIRMED: YES, 3/3 positive, mean $\Delta = +0.025$, $p = 0.008$)

## Phase 9: Replication on UPenn RAM (H4)
- [x] 9.1 Dataset-specific loader for ds004789 (29 patients, 6,122 free recall trials)
- [x] 9.2 Rerun with task-aligned OOD serial-position split (spos 1–8 train, spos 9–12 OOD) + stabilized protocol ($\Delta = +0.026, p = 0.018$, CONFIRMED: YES)

## Phase 10: Wrap-up & Synthesis
- [x] 10.1 Publication-grade figures (`Figure_1_Master_Synthesis_Overview.png`, `Figure_2_CKA_Representational_Alignment.png`, etc.)
- [x] 10.2 Robustness checks & parameter sensitivity
- [x] 10.3 Consolidated technical report (`FINAL_PROJECT_REPORT.md` and `cdb2m_master_summary.csv`)
- [x] 10.4 Final synthesis execution (`10-final-synthesis-and-report.ipynb`)

## Phase 11: Emotion Representation & Affective Dissociation (RQ4)
- [x] 11.1 Map Amygdalar intracranial coverage (165 electrodes across 10 patients)
- [x] 11.2 Dissociation analysis: Amygdala vs Hippocampal representational geometry
- [x] 11.3 Impact of emotion-specialized circuits on symbolic working memory transfer
- [x] 11.4 Affective error and conflict probing report (`11-emotion-representation.ipynb`, CONFIRMED: YES, Probing AUC $= 0.571$ vs $0.428$, $+14.3\%$ advantage, $p = 0.002$)
