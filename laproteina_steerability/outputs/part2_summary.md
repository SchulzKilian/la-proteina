# Part 2: Property Probes Summary

## t-convention
All probe results use code convention: t=0 is pure noise, t=1 is clean data.
WARNING: Steering predictor convention has NOT been verified.

## Decision summary

- Properties analyzed: 14
- steerable: 1
- nonlinear_encoded: 0
- goodhart_control: 4
- drop: 8
- ambiguous: 1

## Decisions

                    property         decision  best_linear_r2  best_mlp_r2  best_knn_r2                                                                                                             note
                         swi goodhart_control        0.362777    -0.153170     0.409822                                                                   Flagged as Goodhart control — evaluation only.
                       tango goodhart_control        0.088862     0.206157     0.171249                                                                   Flagged as Goodhart control — evaluation only.
                  net_charge             drop        0.134751     0.166863     0.144653                                                                                                                 
                          pI             drop        0.165130     0.186044     0.202511                                                                                                                 
                     iupred3             drop        0.272821     0.162195     0.296240                                                                                                                 
 iupred3_fraction_disordered             drop        0.121319    -0.027572     0.124370                                                                                                                 
             shannon_entropy        steerable        0.459432     0.515945     0.574099                                                                                                                 
hydrophobic_patch_total_area             drop        0.019502     0.080860     0.123510                                                                                                                 
   hydrophobic_patch_n_large             drop        0.010978     0.063976     0.090212                                                                                                                 
                         sap goodhart_control        0.007432     0.079337     0.113180                                                                   Flagged as Goodhart control — evaluation only.
                scm_positive        ambiguous        0.287050     0.309777     0.303620                                                                      Does not cleanly fit any decision category.
                scm_negative             drop        0.176419     0.218341     0.215047                                                                                                                 
                          rg             drop        0.005317     0.219601     0.146431 Rg is backbone-determined; latent-only probes should be near-zero unless Rg correlates with an encoded property.
            camsol_intrinsic goodhart_control        0.259158     0.326951     0.348113                                                                   Flagged as Goodhart control — evaluation only.

## Goodhart pairs

      prop_a           prop_b  correlation  diverges
scm_positive     scm_negative    -0.538757     False
       tango              sap     0.398720      True
         swi              sap    -0.214271      True
       tango camsol_intrinsic    -0.372069      True

## Top probe results (by R²)

       property   t        input_variant probe_type  r2_mean   r2_std
shannon_entropy 1.0          latent_only        knn 0.574099 0.028576
shannon_entropy 1.0 latent_plus_backbone        knn 0.557242 0.028447
shannon_entropy 0.8          latent_only        knn 0.554840 0.027549
shannon_entropy 0.8          latent_only        mlp 0.515945 0.130237
shannon_entropy 0.8 latent_plus_backbone        knn 0.487699 0.031422
shannon_entropy 1.0          latent_only     linear 0.459432 0.048057
shannon_entropy 0.5          latent_only        knn 0.459405 0.025740
shannon_entropy 1.0 latent_plus_backbone     linear 0.458091 0.047857
shannon_entropy 0.8          latent_only     linear 0.451248 0.046666
shannon_entropy 0.8 latent_plus_backbone     linear 0.449850 0.046456
shannon_entropy 0.5          latent_only        mlp 0.423304 0.095475
shannon_entropy 0.5 latent_plus_backbone        knn 0.421055 0.025180
            swi 1.0          latent_only        knn 0.409822 0.051226
shannon_entropy 0.5 latent_plus_backbone        mlp 0.378335 0.073917
            swi 0.8          latent_only        knn 0.364958 0.056723

## Figures
- `figures/property_correlation_clustered.{png,pdf}` — clustered property heatmap
- `figures/umap_property_grid.{png,pdf}` — 3x3 UMAP grid

## Tables
- `tables/probe_results.parquet` — full probe results
- `tables/probe_results_wide.csv` — pivot for human reading
- `tables/steering_decisions.csv` — go/no-go per property
- `tables/property_clusters.csv` — correlation clusters
- `tables/goodhart_pairs.csv` — Goodhart pair analysis
