# CD-B2M: Curated Figures for Research Paper & Publication
## High-Resolution Publication-Grade Visual Deliverables

This directory contains all standardized, publication-grade figures displaying accurate empirical results across all research questions and hypotheses.

| Figure File | File Size | Description & Paper Context |
| :--- | :---: | :--- |
| **`Figure_1_Master_Synthesis_Overview.png`** | 342.0 KB | Master 4-panel overview synthesizing Phase 6 causal transfer, Phase 7 anatomical ranking, Phase 8 cross-architecture consistency, and Phase 9 multi-cohort replication. |
| **`Figure_2_CKA_Representational_Alignment.png`** | 226.8 KB | Centered Kernel Alignment (CKA) representational similarity between neural embeddings and model representations vs. downstream generalization accuracy. |
| **`Figure_3_Spectral_Matching_and_MaxEnt_Validation.png`** | 125.6 KB | Log-log power spectral density verifying that Maximum-Entropy surrogates strictly match real intracranial EEG 1/f slope and cross-channel covariance. |
| **`Figure_4_Causal_Transfer_and_Bootstrap_CIs.png`** | 46.0 KB | Forest plot displaying Delta_causal across sequence splits with 95% bootstrap confidence intervals, demonstrating Set Size 8 strictly excludes zero. |
| **`Figure_5_Lambda_Sweep_and_Optimal_Regularization.png`** | 191.7 KB | Complete alignment weight parameter sweep (lambda = 0.0 to 5.0) showing accuracy profiles across conditions and proving high lambda causes bottlenecking. |
| **`Figure_6A_Hippocampal_Specificity_Single_Region.png`** | 83.7 KB | Single-region alignment accuracy across Hippocampus, Parahippocampal gyrus, Amygdala, and Lateral Temporal cortex, confirming Hippocampus as #1. |
| **`Figure_6B_Regional_Ablation_Impact.png`** | 56.1 KB | Performance degradation upon ablating individual anatomical regions, demonstrating that hippocampal removal causes the largest OOD drop. |
| **`Figure_6C_Region_by_Split_Heatmap.png`** | 109.2 KB | Heatmap of regional performance across out-of-distribution sequence splits. |
| **`Figure_7A_Cross_Architecture_Causal_Transfer.png`** | 45.9 KB | Causal transfer advantage (Delta_causal) across Recurrent (SeqGRU), Attention (SeqTransformer), and Reinforcement Learning (RLAgent) architectures. |
| **`Figure_7B_Cross_Architecture_Consistency_Heatmap.png`** | 58.0 KB | Consistency heatmap across architectures and evaluation sequence splits. |
| **`Figure_8_External_Multi_Cohort_Replication.png`** | 164.4 KB | Multi-cohort external replication results on the independent UPenn RAM dataset (ds004789, 29 patients, 6,122 free recall trials). |
| **`Figure_9_RQ4_Emotion_Representation_and_Dissociation.png`** | 264.4 KB | Linear readout probing of Amygdalar latent space decoding affective error/conflict (+14.3% AUC advantage) and double dissociation. |

### Figure Usage in Research Paper Sections:
1. **Main Synthesis & Overview:** `Figure_1_Master_Synthesis_Overview.png` (4-panel centerpiece).
2. **Causal Disentanglement (RQ1 / H1):** `Figure_3_Spectral_Matching_and_MaxEnt_Validation.png`, `Figure_4_Causal_Transfer_and_Bootstrap_CIs.png`, `Figure_5_Lambda_Sweep_and_Optimal_Regularization.png`.
3. **Representational Similarity:** `Figure_2_CKA_Representational_Alignment.png`.
4. **Anatomical Circuit Specificity (RQ2 / H2):** `Figure_6A_Hippocampal_Specificity_Single_Region.png`, `Figure_6B_Regional_Ablation_Impact.png`, `Figure_6C_Region_by_Split_Heatmap.png`.
5. **Cross-Architecture Invariance (RQ3 / H3):** `Figure_7A_Cross_Architecture_Causal_Transfer.png`, `Figure_7B_Cross_Architecture_Consistency_Heatmap.png`.
6. **Multi-Cohort Replication (H4):** `Figure_8_External_Multi_Cohort_Replication.png`.
7. **Emotion Representation & Dissociation (RQ4):** `Figure_9_RQ4_Emotion_Representation_and_Dissociation.png`.