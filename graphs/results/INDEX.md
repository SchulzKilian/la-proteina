# Plot Index — Masterarbeit Figures (P01–P35)

**Rendering environment:** all 71 PDFs rendered natively via the project `plot_utils.py` (PGF backend, `pdflatex` system TeX Live 2023, Computer Modern Roman, 12 pt). No Agg/usetex=False fallback was needed on this machine — pdflatex is installed at `/usr/bin/pdflatex`. Scripts must be invoked under the project conda env (e.g. `conda run -n laproteina_env python P##_v#_*.py`) because the system python lacks matplotlib.

**Headline picks for the thesis:** P02_v1 (nsteps cliff scatter), P20_v1 (predictor-real gap dual), P22_v1 (codesign Pareto), P15_v1 (variant ladder), P34_v1 (variant ablation table). Strong supporting figures: P12_v1 (wd × design), P29_v1 (alphabet collapse), P33_v2 (Jaccard collapse).

**Data sources used:**
- Real CSV: P27 (`noise_aware_ensemble_sweep/diversity_pairwise_tm.csv`), P29 (`aa_composition_nsteps400/.../aa_composition.csv`), P31 (`aromatic_burial_afdb/aromatic_frequencies.csv`).
- Real JSON: P16 (`per_t_val/*.json`).
- Mixed inline-from-experiments.md plus narrative numbers for the remainder. Each script's header notes its source.

**Caveats called out per CLAUDE.md rules:**
- All structure-derived metric plots are sourced from nsteps=400 runs (per the hard rule).
- For length-binned property metrics (P23, P25, P30) the data is binned per-L before computing Cohen's d (per `feedback_length_bin_property_sigmas.md`).
- Wandb-derived val-loss curves (P10, P14) carry a not-cross-run-comparable caveat; the recommended re-eval source is per_t_val / aggregate_val_seeds.

---

| ID | Version | Title | File | Status | Notes |
|---|---|---|---|---|---|
| P01 | v1 | Two-route thesis overview map | `P01_v1_thesis_overview_map.pdf` | rendered | Schematic with arrows; F1–F13 colored by route. |
| P01 | v2 | Two-route thesis flowchart | `P01_v2_thesis_overview_flowchart.pdf` | rendered | Linear flowchart variant. |
| P02 | v1 | nsteps=200 vs 400 cliff (scatter) | `P02_v1_nsteps_cliff_scatter.pdf` | rendered | Log-y scatter, headline cliff figure. |
| P02 | v2 | nsteps=200 vs 400 bar | `P02_v2_nsteps_cliff_bar.pdf` | rendered | Paired-bar version of the cliff. |
| P03 | v1 | Master experiment timeline (Gantt) | `P03_v1_experiment_timeline_gantt.pdf` | rendered | E001–E079 grouped by route. |
| P03 | v2 | Experiment activity density | `P03_v2_experiment_timeline_density.pdf` | rendered | Per-week activity density. |
| P04 | v1 | Predictor 5-fold R² (bars) | `P04_v1_predictor_r2_bar.pdf` | rendered | Horizontal bar sorted by mean R². |
| P04 | v2 | Predictor 5-fold R² (strip) | `P04_v2_predictor_r2_strip.pdf` | rendered | Per-fold dots over mean. |
| P05 | v1 | Probe convergence (lollipop) | `P05_v1_probe_convergence_lollipop.pdf` | rendered | Colored by regime. |
| P05 | v2 | Probe convergence (grouped bar) | `P05_v2_probe_convergence_grouped_bar.pdf` | rendered | Alternative framing. |
| P06 | v1 | Capacity probing heatmap | `P06_v1_capacity_heatmap.pdf` | rendered | 7-probe × 13-prop R² matrix. |
| P06 | v2 | Capacity probing curves | `P06_v2_capacity_curves.pdf` | rendered | Line plot of R² vs probe size. |
| P07 | v1 | Latent geometry composite | `P07_v1_latent_geometry_composite.pdf` | rendered | 4-panel composite. |
| P07 | v2 | Latent geometry summary table | `P07_v2_latent_geometry_summary_table.pdf` | rendered | Tabular summary. |
| P08 | v1 | Flow curvature dual line | `P08_v1_flow_curvature_dual_line.pdf` | rendered | Per-channel displacement vs t. |
| P08 | v2 | Flow curvature R bar | `P08_v2_flow_curvature_bar.pdf` | rendered | Just the straightness bars. |
| P09 | v1 | Sidechain perturbation box | `P09_v1_sidechain_paired_box.pdf` | rendered | Per-protein paired scRMSD. |
| P09 | v2 | Sidechain perturbation lines | `P09_v2_sidechain_paired_lines.pdf` | rendered | Connecting paired proteins. |
| P10 | v1 | Val-loss trajectory | `P10_v1_val_loss_trajectory.pdf` | rendered | Wandb caveat in script comment. |
| P10 | v2 | Val-loss per-t | `P10_v2_val_loss_per_t.pdf` | rendered | Paired re-eval framing. |
| P11 | v1 | AdaLN-Zero gate table | `P11_v1_adaln_gate_table.pdf` | rendered | matplotlib.table form. |
| P11 | v2 | AdaLN-Zero gate bar | `P11_v2_adaln_gate_bar.pdf` | rendered | Sorted bar companion. |
| P12 | v1 | wd × design grouped bar | `P12_v1_wd_design_grouped_bar.pdf` | rendered | Per-L cluster, 4-arm. |
| P12 | v2 | wd × design dotplot | `P12_v2_wd_design_dotplot.pdf` | rendered | Per-arm dots. |
| P12 | v3 | wd × design table | `P12_v3_wd_design_table.pdf` | rendered | Tabular form. |
| P13 | v1 | Per-layer norm ratio bar | `P13_v1_layer_norm_ratio_bar.pdf` | rendered | 164 layers sorted. |
| P13 | v2 | Per-layer norm ratio histogram | `P13_v2_layer_norm_ratio_hist.pdf` | rendered | Distribution with tail. |
| P14 | v1 | Val vs designability scatter | `P14_v1_val_vs_design_scatter.pdf` | rendered | One point per ckpt. |
| P14 | v2 | Val vs designability dumbbell | `P14_v2_val_vs_design_dumbbell.pdf` | rendered | Per-arm dumbbell. |
| P15 | v1 | Variant ladder (grouped bar) | `P15_v1_variant_ladder_grouped_bar.pdf` | rendered | No variant beats canon. |
| P15 | v2 | Variant ladder (heatmap) | `P15_v2_variant_ladder_heatmap.pdf` | rendered | Heatmap framing. |
| P16 | v1 | Per-t val multi-line | `P16_v1_per_t_val_multi_line.pdf` | rendered | Real per_t_val JSONs. |
| P16 | v2 | Per-t val delta | `P16_v2_per_t_val_delta.pdf` | rendered | Variant minus canonical. |
| P17 | v1 | Hybrid kink twin scatter | `P17_v1_hybrid_kink_twin_scatter.pdf` | rendered | E040/E041 handover. |
| P17 | v2 | Hybrid design bar | `P17_v2_hybrid_design_bar.pdf` | rendered | Pooled rate at t_switch. |
| P18 | v1 | Sparse vs dense crossover line | `P18_v1_sparse_dense_crossover_line.pdf` | rendered | Continuous L curves. |
| P18 | v2 | Sparse vs dense per-L bar | `P18_v2_sparse_dense_per_L_bar.pdf` | rendered | Discrete bars + OOM. |
| P19 | v1 | Dead-arm table (mpl) | `P19_v1_dead_arm_table.pdf` | rendered | matplotlib.table form. |
| P19 | v2 | Dead-arm booktabs | `P19_v2_dead_arm_booktabs.pdf` | rendered | LaTeX-styled rules. |
| P20 | v1 | Predictor:real gap dual | `P20_v1_predictor_real_gap_dual.pdf` | rendered | Headline gap-closure figure. |
| P20 | v2 | Predictor:real gap (gap only) | `P20_v2_predictor_real_gap_only.pdf` | rendered | All 5 fix variants. |
| P21 | v1 | Negative results lollipop | `P21_v1_neg_results_lollipop.pdf` | rendered | Horizontal lollipop. |
| P21 | v2 | Negative results grouped bar | `P21_v2_neg_results_grouped_bar.pdf` | rendered | Per-w bars. |
| P22 | v1 | Codesign Pareto | `P22_v1_codesign_pareto.pdf` | rendered | Production-knee annotated. |
| P22 | v2 | Codesign dual-axis | `P22_v2_pareto_dual_axis.pdf` | rendered | Trade-off as twin lines. |
| P23 | v1 | CamSol per-length strip | `P23_v1_camsol_per_length_strip.pdf` | rendered | Synthesized distribution. |
| P23 | v2 | CamSol per-length d-bar | `P23_v2_camsol_per_length_bar.pdf` | rendered | Cohen's d per L. |
| P24 | v1 | Steering anatomy 3-panel | `P24_v1_steering_anatomy.pdf` | rendered | Schedule + data-flow + trace. |
| P24 | v2 | Steering schedule only | `P24_v2_steering_schedule_only.pdf` | rendered | Clean schedule figure. |
| P25 | v1 | Per-length response heatmap | `P25_v1_per_length_steering_heatmap.pdf` | rendered | (direction,w) × L Cohen's d. |
| P25 | v2 | Per-length response lines | `P25_v2_per_length_steering_lines.pdf` | rendered | Line per recipe. |
| P26 | v1 | Cocktail grouped bar | `P26_v1_cocktail_grouped_bar.pdf` | rendered | 1/2/4-obj with codesign. |
| P26 | v2 | Cocktail radar | `P26_v2_cocktail_radar.pdf` | rendered | Per-axis delivery. |
| P27 | v1 | TM-score box per cell | `P27_v1_tm_distribution_box.pdf` | rendered | Real CSV. |
| P27 | v2 | Mean TM vs w line | `P27_v2_tm_vs_w_line.pdf` | rendered | Flat-mean message. |
| P28 | v1 | Walltime bar | `P28_v1_walltime_bar.pdf` | rendered | Per-protein wall. |
| P28 | v2 | Walltime stacked | `P28_v2_walltime_break.pdf` | rendered | ODE + predictor split. |
| P29 | v1 | AA diverging bar | `P29_v1_aa_diverging_bar.pdf` | rendered | Real AFDB+PDB CSV. |
| P29 | v2 | AA scatter (gen vs ref) | `P29_v2_aa_scatter.pdf` | rendered | Identity-line scatter. |
| P30 | v1 | Length scaling heatmap | `P30_v1_length_scaling_heatmap.pdf` | rendered | Per-50-res bins. |
| P30 | v2 | Length scaling lines | `P30_v2_length_scaling_lines.pdf` | rendered | Line per metric. |
| P31 | v1 | Aromatic burial slope | `P31_v1_aromatic_burial_slope.pdf` | rendered | Real CSV. |
| P31 | v2 | Aromatic burial grouped bar | `P31_v2_aromatic_burial_grouped_bar.pdf` | rendered | With bootstrap CIs. |
| P32 | v1 | Thermal proxy diverging | `P32_v1_thermal_diverging.pdf` | rendered | AFDB + PDB series. |
| P32 | v2 | Thermal proxy lollipop | `P32_v2_thermal_lollipop.pdf` | rendered | Sorted by magnitude. |
| P33 | v1 | Dense routing 3-panel | `P33_v1_dense_routing_audit_3panel.pdf` | rendered | Mass + Jaccard + box. |
| P33 | v2 | Dense routing Jaccard only | `P33_v2_dense_routing_jaccard_only.pdf` | rendered | Headline collapse number. |
| P34 | v1 | Variant ablation table (mpl) | `P34_v1_variant_ablation_table.pdf` | rendered | Verdict-colored cells. |
| P34 | v2 | Variant ablation booktabs | `P34_v2_variant_ablation_booktabs.pdf` | rendered | LaTeX-styled rules. |
| P35 | v1 | Finding–experiment table | `P35_v1_finding_exp_map_table.pdf` | rendered | Role-colored last column. |
| P35 | v2 | Finding–experiment dot matrix | `P35_v2_finding_exp_dot_matrix.pdf` | rendered | Primary vs supporting dots. |
| P36 | v1 | Dense vs sparse-K40 inference scaling | `P36_v1_inference_scaling_a100.pdf` | rendered | Real CSV (E120), A100-80GB. Mem + wall-clock vs L; dense OOM at L=2400. |

**Counts:**
- 36/36 P-IDs covered.
- 72 PDFs produced (P12 has 3 versions; P36 has 1; all others have 2).
- 0 fallbacks needed; all native pdflatex.
- 0 P-IDs skipped.
