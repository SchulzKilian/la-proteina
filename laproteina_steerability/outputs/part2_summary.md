# Part 2: Property Probes Summary

## t-convention
All probe results use code convention: t=0 is pure noise, t=1 is clean data.
WARNING: Steering predictor convention has NOT been verified.

## Decision summary

- Properties analyzed: 13
- steerable: 0
- nonlinear_encoded: 0
- goodhart_control: 5
- drop: 8
- ambiguous: 0

## Decisions

                    property         decision  best_linear_r2  best_mlp_r2  best_knn_r2                                                                                                             note
                         swi goodhart_control       -0.603811    -1.253417    -1.093453                                                                   Flagged as Goodhart control — evaluation only.
                       tango goodhart_control       -0.949179    -2.989719    -0.915946                                                                   Flagged as Goodhart control — evaluation only.
                  net_charge             drop       -0.566239    -0.607116    -0.571171                                                                                                                 
                          pI             drop       -0.106813    -0.739587    -0.142895                                                                                                                 
                     iupred3             drop       -0.834534    -1.615833    -0.841400                                                                                                                 
 iupred3_fraction_disordered             drop       -0.098543    -0.490921    -0.127897                                                                                                                 
             shannon_entropy             drop       -0.126313    -0.289363    -0.175572                                                                                                                 
hydrophobic_patch_total_area             drop       -0.131870     0.141830     0.001460                                                                                                                 
   hydrophobic_patch_n_large             drop       -0.149905    -0.755210    -0.114321                                                                                                                 
                         sap goodhart_control       -0.138171    -0.825749    -0.388465                                                                   Flagged as Goodhart control — evaluation only.
                scm_positive goodhart_control       -0.363804    -1.149305    -0.465565                                                                   Flagged as Goodhart control — evaluation only.
                scm_negative goodhart_control       -0.126374    -0.369285    -0.090511                                                                   Flagged as Goodhart control — evaluation only.
                          rg             drop       -0.727879    -0.978387    -0.796202 Rg is backbone-determined; latent-only probes should be near-zero unless Rg correlates with an encoded property.

## Goodhart pairs

      prop_a       prop_b  correlation  diverges
scm_positive scm_negative     0.132334      True
       tango          sap     0.229854      True
         swi          sap     0.209626      True

## Top probe results (by R²)

                    property   t        input_variant probe_type   r2_mean   r2_std
hydrophobic_patch_total_area 0.8          latent_only        mlp  0.141830 0.423606
hydrophobic_patch_total_area 1.0 latent_plus_backbone        knn  0.001460 0.294348
hydrophobic_patch_total_area 1.0          latent_only        knn -0.013750 0.337523
hydrophobic_patch_total_area 0.8 latent_plus_backbone        knn -0.054490 0.319462
                scm_negative 0.5          latent_only        knn -0.090511 0.218559
 iupred3_fraction_disordered 1.0          latent_only     linear -0.098543 0.083150
                          pI 1.0          latent_only     linear -0.106813 0.123830
 iupred3_fraction_disordered 1.0 latent_plus_backbone     linear -0.110116 0.063880
   hydrophobic_patch_n_large 1.0          latent_only        knn -0.114321 0.309389
                          pI 0.5          latent_only     linear -0.124652 0.249404
             shannon_entropy 0.3          latent_only     linear -0.126313 0.115013
                scm_negative 0.5          latent_only     linear -0.126374 0.352461
hydrophobic_patch_total_area 0.5 latent_plus_backbone        knn -0.126431 0.363580
                scm_negative 0.5 latent_plus_backbone     linear -0.127373 0.266445
                          pI 1.0 latent_plus_backbone     linear -0.127438 0.155531

## Figures
- `figures/property_correlation_clustered.{png,pdf}` — clustered property heatmap
- `figures/umap_property_grid.{png,pdf}` — 3x3 UMAP grid

## Tables
- `tables/probe_results.parquet` — full probe results
- `tables/probe_results_wide.csv` — pivot for human reading
- `tables/steering_decisions.csv` — go/no-go per property
- `tables/property_clusters.csv` — correlation clusters
- `tables/goodhart_pairs.csv` — Goodhart pair analysis
