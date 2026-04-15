# Part 2: Property Probes Summary

## t-convention
All probe results use code convention: t=0 is pure noise, t=1 is clean data.
WARNING: Steering predictor convention has NOT been verified.

## Decision summary

- Properties analyzed: 8
- steerable: 6
- nonlinear_encoded: 0
- goodhart_control: 0
- drop: 2
- ambiguous: 0

## Decisions

        property  decision  best_linear_r2  best_mlp_r2  best_knn_r2                                                                                                             note
      net_charge      drop       -0.200595    -0.563253    -0.195181                                                                                                                 
         iupred3 steerable        0.720710     0.698896     0.663555                                                                                                                 
camsol_intrinsic steerable        0.726562     0.713791     0.675770                                                                                                                 
           tango steerable        0.724319     0.700016     0.660703                                                                                                                 
           canya steerable        0.720793     0.705369     0.660986                                                                                                                 
             swi steerable        0.719575     0.701719     0.653084                                                                                                                 
             sap steerable        0.731595     0.712318     0.670324                                                                                                                 
              rg      drop       -0.328253    -0.658325    -0.400139 Rg is backbone-determined; latent-only probes should be near-zero unless Rg correlates with an encoded property.

## Goodhart pairs

          prop_a prop_b  correlation  diverges
camsol_intrinsic    swi     0.847080     False
           tango  canya     0.797717     False

## Top probe results (by R²)

        property   t        input_variant probe_type  r2_mean   r2_std
             sap 1.0 latent_plus_backbone     linear 0.731595 0.024016
             sap 1.0          latent_only     linear 0.731593 0.023984
camsol_intrinsic 1.0          latent_only     linear 0.726562 0.038769
camsol_intrinsic 1.0 latent_plus_backbone     linear 0.726287 0.038870
           tango 1.0          latent_only     linear 0.724319 0.033978
           tango 1.0 latent_plus_backbone     linear 0.723361 0.034508
           canya 1.0          latent_only     linear 0.720793 0.037876
         iupred3 1.0          latent_only     linear 0.720710 0.014952
         iupred3 1.0 latent_plus_backbone     linear 0.720170 0.014806
           canya 1.0 latent_plus_backbone     linear 0.719631 0.038284
             swi 1.0          latent_only     linear 0.719575 0.019150
             swi 1.0 latent_plus_backbone     linear 0.718453 0.020407
camsol_intrinsic 1.0          latent_only        mlp 0.713791 0.035516
             sap 1.0          latent_only        mlp 0.712318 0.017740
           canya 1.0          latent_only        mlp 0.705369 0.037622

## Figures
- `figures/property_correlation_clustered.{png,pdf}` — clustered property heatmap
- `figures/umap_property_grid.{png,pdf}` — 3x3 UMAP grid

## Tables
- `tables/probe_results.parquet` — full probe results
- `tables/probe_results_wide.csv` — pivot for human reading
- `tables/steering_decisions.csv` — go/no-go per property
- `tables/property_clusters.csv` — correlation clusters
- `tables/goodhart_pairs.csv` — Goodhart pair analysis
