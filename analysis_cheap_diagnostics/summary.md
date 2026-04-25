# Cheap diagnostics — Exp 1, steps 1+2

PDB property file: `/home/ks2218/la-proteina/laproteina_steerability/data/properties.csv` (56008 rows)
Generated property file: `/home/ks2218/la-proteina/results/generated_baseline_300_800/properties_generated.csv` (100 rows)

## Per-bin counts (50-residue bins, 300-800)

| bin | pdb_n | generated_n |
|---|---|---|
| [300,350) | 21055 | 35 |
| [350,400) | 12320 | 31 |
| [400,450) | 8182 | 12 |
| [450,500) | 5967 | 9 |
| [500,550) | 3151 | 8 |
| [550,600) | 2259 | 2 |
| [600,650) | 1215 | 1 |
| [650,700) | 868 | 1 |
| [700,750) | 800 | 0 |
| [750,800) | 191 | 1 |

**KS distance on length (300-800 only):** D = 0.0769, p = 5.690e-01
  PDB in range: 56008, Generated in range: 100

## Property correlation matrix on PDB

After NaN-drop: 56008 / 56008 PDB rows used.

**Li-Ji M_eff (Pearson):  9.000** (out of 14 properties)
**Li-Ji M_eff (Spearman): 10.000** (out of 14 properties)

Pearson  eigenvalues (sorted desc): [4.143, 2.754, 2.049, 1.455, 0.907, 0.797, 0.602, 0.384, 0.307, 0.197, 0.166, 0.131, 0.063, 0.046]

Spearman eigenvalues (sorted desc): [4.092, 2.95, 1.981, 1.298, 0.991, 0.921, 0.555, 0.33, 0.267, 0.212, 0.152, 0.142, 0.07, 0.039]

### Top 10 |Pearson| pairs

| prop A | prop B | r |
|---|---|---|
| hydrophobic_patch_total_area | hydrophobic_patch_n_large | +0.908 |
| hydrophobic_patch_total_area | sap | +0.901 |
| net_charge | pI | +0.855 |
| hydrophobic_patch_n_large | sap | +0.837 |
| iupred3 | iupred3_fraction_disordered | +0.741 |
| hydrophobic_patch_total_area | rg | +0.632 |
| tango | rg | +0.570 |
| tango_aggregation_positions | sap | +0.540 |
| net_charge | scm_negative | +0.524 |
| scm_positive | scm_negative | -0.518 |

### Top 10 |Spearman| pairs

| prop A | prop B | rho |
|---|---|---|
| net_charge | pI | +0.941 |
| hydrophobic_patch_total_area | sap | +0.878 |
| hydrophobic_patch_total_area | hydrophobic_patch_n_large | +0.843 |
| iupred3 | iupred3_fraction_disordered | +0.766 |
| hydrophobic_patch_n_large | sap | +0.752 |
| hydrophobic_patch_total_area | rg | +0.703 |
| tango | rg | +0.603 |
| swi | iupred3 | +0.600 |
| sap | rg | +0.564 |
| scm_negative | rg | -0.560 |

### Bonferroni thresholds at alpha=0.05

- naive (14 tests): 0.00357
- Li-Ji Pearson  (9.00 tests): 0.00556
- Li-Ji Spearman (10.00 tests): 0.00500

## Li-Ji M_eff per length bin (Spearman, PDB)

| bin | n | m_eff_spearman |
|---|---|---|
| [300,350) | 21055 | 9.999999999999991 |
| [350,400) | 12320 | 8.999999999999998 |
| [400,450) | 8182 | 9.999999999999998 |
| [450,500) | 5967 | 10.0 |
| [500,550) | 3151 | 9.999999999999998 |
| [550,600) | 2259 | 9.0 |
| [600,650) | 1215 | 10.000000000000002 |
| [650,700) | 868 | 9.0 |
| [700,750) | 800 | 7.999999999999998 |
| [750,800) | 191 | 10.000000000000005 |
