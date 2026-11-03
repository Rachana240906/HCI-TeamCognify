Plan

Phase 0: Setup

 0.1 Write the project spec: research questions, hypotheses, success criteria
 0.2 Create the repo structure, config, and seed management
 0.3 Set up Kaggle and download ds004752
 0.4 Verify the download and persist it so you never re-download

Phase 1: Data understanding

 1.1 Parse the BIDS structure (subjects, sessions, channels, events, electrode coordinates)
 1.2 Inspect the task design (set sizes, trial counts, match/mismatch balance, RTs)
 1.3 Map electrodes to regions and tabulate coverage per subject
 1.4 Decision point: finalize the region set and the held-out split definitions

Phase 2: Preprocessing

 2.1 Filtering, re-referencing, bad channels
 2.2 Trial segmentation (encoding / maintenance / recall) and normalization
 2.3 Fixed-length feature extraction, keeping subject/session/trial IDs
 2.4 Region-wise representations
 2.5 Cache processed tensors and run sanity plots

Phase 3: Brain encoder, splits, synthetic tasks

 3.1 Brain encoder (shared across real and control data)
 3.2 Splits: within-subject, held-out subjects, held-out set sizes (OOD)
 3.3 Leakage checks
 3.4 Sanity check: brain embeddings decode match/mismatch above chance
 3.5 Synthetic sequence tasks with harder OOD splits (guards against ceiling effects)

Phase 4: Control generation

 4.1 MaxEnt surrogate
 4.2 Shuffled brain
 4.3 Random noise
 4.4 Validate the controls (statistics match, task decodability destroyed)

Phase 5: Models and alignment

 5.1 RNN baseline, no alignment
 5.2 InfoNCE alignment loss
 5.3 CKA metric
 5.4 Train RNN under the 4 conditions, multiple seeds
 5.5 Reproduce the basic B2M effect

Phase 6: Causal evaluation (RQ1)

 6.1 Δ_causal in-distribution and OOD
 6.2 λ sweeps
 6.3 Does CKA to brain predict generalization beyond controls?
 6.4 Statistics (paired tests, bootstrap CIs, multiple-comparison correction)

Phase 7: Brain-region analysis (RQ2)

 7.1 Region ablation
 7.2 Region-only alignment
 7.3 Region × architecture heatmap

Phase 8: Cross-architecture transfer (RQ3)

 8.1 Transformer and RL agent
 8.2 Full control comparison per architecture
 8.3 Consistency check across architectures

Phase 9: Replication on UPenn RAM (when you get access)

 9.1 Dataset-specific loader
 9.2 Rerun the core comparison and the novel-task test

Phase 10: Wrap-up

 10.1 Figures
 10.2 Robustness checks
 10.3 Write-up and limitations
 10.4 Package the code