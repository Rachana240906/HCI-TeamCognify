"""
organize_paper_figures.py - Curates, standardizes, and organizes all publication-grade
result figures for the CD-B2M research paper.
"""

import os
import shutil
import pathlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def generate_rq4_figure(out_path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=300)
    plt.subplots_adjust(wspace=0.35)

    # Colors
    c_amyg = "#e11d48"    # Rose/Coral
    c_hipp = "#0d9488"    # Teal
    c_ctrl = "#64748b"    # Slate

    # Panel A: Affective Feedback Probing AUC
    ax1 = axes[0]
    conditions = ["Amygdala Aligned\n(Limbic Hub)", "Hippocampus\n(Memory Hub)", "Noise Control\n(Spectral Surrogate)"]
    aucs = [0.5706, 0.5172, 0.4281]
    errs = [0.0562, 0.0584, 0.0291]
    bars = ax1.bar(conditions, aucs, yerr=errs, capsize=5, color=[c_amyg, c_hipp, c_ctrl], width=0.55, edgecolor="#1e293b", linewidth=1.2)
    ax1.axhline(0.5, color="black", linestyle="--", linewidth=1.0, alpha=0.6, label="Chance Level (0.50)")
    ax1.set_ylim(0.35, 0.70)
    ax1.set_ylabel("Affective Probing ROC-AUC", fontsize=10, fontweight="bold")
    ax1.set_title("(A) Linear Probing of Emotion/Feedback Conflict", fontsize=11, fontweight="bold", pad=10)
    ax1.annotate(f"+14.3% Advantage\np = 0.00199 **", xy=(0, 0.635), ha='center', fontsize=9, fontweight='bold', color=c_amyg)
    ax1.annotate(f"p = 0.284 (n.s.)", xy=(1, 0.585), ha='center', fontsize=8.5, color=c_hipp)
    ax1.grid(axis='y', linestyle=':', alpha=0.6)
    ax1.legend(loc="lower right", fontsize=8.5)

    # Panel B: Functional Double Dissociation
    ax2 = axes[1]
    labels = ["Memory Load Scaling\n(Theta r with SetSize)", "Working Memory Transfer\n(Delta Causal vs Ctrl)"]
    x = np.arange(len(labels))
    width = 0.32

    hipp_vals = [0.0729, +0.0200]
    amyg_vals = [0.1379, -0.0370]

    b1 = ax2.bar(x - width/2, hipp_vals, width, label="Hippocampus (Memory)", color=c_hipp, edgecolor="#1e293b", linewidth=1.2)
    b2 = ax2.bar(x + width/2, amyg_vals, width, label="Amygdala (Emotion)", color=c_amyg, edgecolor="#1e293b", linewidth=1.2)

    ax2.axhline(0, color="gray", linestyle="-", linewidth=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=9.5, fontweight="bold")
    ax2.set_ylabel("Effect Magnitude / Correlation", fontsize=10, fontweight="bold")
    ax2.set_title("(B) Functional Double Dissociation", fontsize=11, fontweight="bold", pad=10)
    ax2.grid(axis='y', linestyle=':', alpha=0.6)
    ax2.legend(loc="upper left", fontsize=8.5)

    # Annotations
    ax2.annotate("Memory Benefit\n(+2.0%)", xy=(0.84, 0.025), ha='center', fontsize=8.5, fontweight='bold', color=c_hipp)
    ax2.annotate("Interference\n(-3.7%)", xy=(1.16, -0.048), ha='center', fontsize=8.5, fontweight='bold', color=c_amyg)

    plt.suptitle("Figure 9: Representation of Emotion Learning & Double Dissociation (RQ4)", fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Generated RQ4 Figure: {out_path}")

def organize_figures():
    base_dir = pathlib.Path(__file__).resolve().parents[1]
    results_dir = base_dir / "Results"
    target_dir = base_dir / "paper_figures"
    target_dir.mkdir(parents=True, exist_ok=True)

    # 1. Generate RQ4 Figure
    fig9_path = target_dir / "Figure_9_RQ4_Emotion_Representation_and_Dissociation.png"
    generate_rq4_figure(fig9_path)

    # 2. Source Figure Mapping
    figure_map = [
        ("Results/phase10_out/fig1_cdb2m_master_synthesis.png", 
         "Figure_1_Master_Synthesis_Overview.png",
         "Master 4-panel overview synthesizing Phase 6 causal transfer, Phase 7 anatomical ranking, Phase 8 cross-architecture consistency, and Phase 9 multi-cohort replication."),
        
        ("Results/phase10_out/fig2_cka_correlation_analysis.png",
         "Figure_2_CKA_Representational_Alignment.png",
         "Centered Kernel Alignment (CKA) representational similarity between neural embeddings and model representations vs. downstream generalization accuracy."),
        
        ("Results/phase4_out/fig_power_spectrum.png",
         "Figure_3_Spectral_Matching_and_MaxEnt_Validation.png",
         "Log-log power spectral density verifying that Maximum-Entropy surrogates strictly match real intracranial EEG 1/f slope and cross-channel covariance."),
        
        ("Results/phase6_out/fig_forest_delta_causal.png",
         "Figure_4_Causal_Transfer_and_Bootstrap_CIs.png",
         "Forest plot displaying Delta_causal across sequence splits with 95% bootstrap confidence intervals, demonstrating Set Size 8 strictly excludes zero."),
        
        ("Results/phase6_out/fig_lambda_sweep_acc.png",
         "Figure_5_Lambda_Sweep_and_Optimal_Regularization.png",
         "Complete alignment weight parameter sweep (lambda = 0.0 to 5.0) showing accuracy profiles across conditions and proving high lambda causes bottlenecking."),
        
        ("Results/phase7_out/fig_single_region_acc.png",
         "Figure_6A_Hippocampal_Specificity_Single_Region.png",
         "Single-region alignment accuracy across Hippocampus, Parahippocampal gyrus, Amygdala, and Lateral Temporal cortex, confirming Hippocampus as #1."),
        
        ("Results/phase7_out/fig_ablation_drop.png",
         "Figure_6B_Regional_Ablation_Impact.png",
         "Performance degradation upon ablating individual anatomical regions, demonstrating that hippocampal removal causes the largest OOD drop."),
        
        ("Results/phase7_out/fig_region_split_heatmap.png",
         "Figure_6C_Region_by_Split_Heatmap.png",
         "Heatmap of regional performance across out-of-distribution sequence splits."),
        
        ("Results/phase8_out/fig_arch_delta_causal.png",
         "Figure_7A_Cross_Architecture_Causal_Transfer.png",
         "Causal transfer advantage (Delta_causal) across Recurrent (SeqGRU), Attention (SeqTransformer), and Reinforcement Learning (RLAgent) architectures."),
        
        ("Results/phase8_out/fig_consistency_heatmap.png",
         "Figure_7B_Cross_Architecture_Consistency_Heatmap.png",
         "Consistency heatmap across architectures and evaluation sequence splits."),
        
        ("Results/phase9b_out/replication_fig.png",
         "Figure_8_External_Multi_Cohort_Replication.png",
         "Multi-cohort external replication results on the independent UPenn RAM dataset (ds004789, 29 patients, 6,122 free recall trials).")
    ]

    copied_records = []

    for src_rel, dst_name, desc in figure_map:
        src = base_dir / src_rel
        dst = target_dir / dst_name
        if src.exists():
            shutil.copy2(src, dst)
            sz_kb = dst.stat().st_size / 1024
            copied_records.append((dst_name, f"{sz_kb:.1f} KB", desc))
            print(f"Copied: {dst_name} ({sz_kb:.1f} KB)")
        else:
            print(f"Warning: Source not found at {src}")

    copied_records.append(("Figure_9_RQ4_Emotion_Representation_and_Dissociation.png", 
                           f"{fig9_path.stat().st_size / 1024:.1f} KB", 
                           "Linear readout probing of Amygdalar latent space decoding affective error/conflict (+14.3% AUC advantage) and double dissociation."))

    # Generate README in paper_figures folder
    readme_lines = [
        "# CD-B2M: Curated Figures for Research Paper & Publication",
        "## High-Resolution Publication-Grade Visual Deliverables",
        "",
        "This directory contains all standardized, publication-grade figures displaying accurate empirical results across all research questions and hypotheses.",
        "",
        "| Figure File | File Size | Description & Paper Context |",
        "| :--- | :---: | :--- |"
    ]
    for fname, sz, desc in copied_records:
        readme_lines.append(f"| **`{fname}`** | {sz} | {desc} |")

    readme_lines.extend([
        "",
        "### Figure Usage in Research Paper Sections:",
        "1. **Main Synthesis & Overview:** `Figure_1_Master_Synthesis_Overview.png` (4-panel centerpiece).",
        "2. **Causal Disentanglement (RQ1 / H1):** `Figure_3_Spectral_Matching_and_MaxEnt_Validation.png`, `Figure_4_Causal_Transfer_and_Bootstrap_CIs.png`, `Figure_5_Lambda_Sweep_and_Optimal_Regularization.png`.",
        "3. **Representational Similarity:** `Figure_2_CKA_Representational_Alignment.png`.",
        "4. **Anatomical Circuit Specificity (RQ2 / H2):** `Figure_6A_Hippocampal_Specificity_Single_Region.png`, `Figure_6B_Regional_Ablation_Impact.png`, `Figure_6C_Region_by_Split_Heatmap.png`.",
        "5. **Cross-Architecture Invariance (RQ3 / H3):** `Figure_7A_Cross_Architecture_Causal_Transfer.png`, `Figure_7B_Cross_Architecture_Consistency_Heatmap.png`.",
        "6. **Multi-Cohort Replication (H4):** `Figure_8_External_Multi_Cohort_Replication.png`.",
        "7. **Emotion Representation & Dissociation (RQ4):** `Figure_9_RQ4_Emotion_Representation_and_Dissociation.png`."
    ])

    with open(target_dir / "FIGURES_README.md", "w", encoding="utf-8") as f:
        f.write("\n".join(readme_lines))

    # Also copy to root paper_figures
    root_target = base_dir.parent / "paper_figures"
    if root_target != target_dir:
        shutil.copytree(target_dir, root_target, dirs_exist_ok=True)

    print(f"\nAll figures successfully curated and saved to:\n  • {target_dir}\n  • {root_target}")

if __name__ == "__main__":
    organize_figures()
