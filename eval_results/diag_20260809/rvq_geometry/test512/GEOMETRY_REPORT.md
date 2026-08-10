# Motion Code Geometry Diagnostics

## Inputs
- KV root: `/scratch/pf2m24/projects/Umdd/KV-Control`
- Data root: `/iridisfs/scratch/pf2m24/data/HumanML3D/HumanML3D`
- VQ checkpoint: `/iridisfs/scratch/pf2m24/projects/Umdd/momask-codes/checkpoints/t2m/rvq_nq6_dc512_nc512_noshare_qdp0.2/model/net_best_fid.tar`
- Partition file: `synthetic_full_body_per_rvq_layer`
- Split: `test`
- Motions encoded: `512`
- Latent length: `49`

## Code Usage
| Part | Active | Dead | Perplexity | Tokens |
|---|---:|---:|---:|---:|
| rvq_layer_0 | 510 | 2 | 319.780 | 17612 |
| rvq_layer_1 | 512 | 0 | 358.568 | 17612 |
| rvq_layer_2 | 512 | 0 | 374.630 | 17612 |
| rvq_layer_3 | 512 | 0 | 381.475 | 17612 |
| rvq_layer_4 | 511 | 1 | 392.265 | 17612 |
| rvq_layer_5 | 512 | 0 | 403.061 | 17612 |

## Distance Correlation
| Space | Part | Learned Spearman | Shuffled Spearman |
|---|---|---:|---:|
| feature | rvq_layer_0 | 0.8007 | 0.0641 |
| feature | rvq_layer_1 | 0.3983 | -0.0350 |
| feature | rvq_layer_2 | 0.4454 | -0.0644 |
| feature | rvq_layer_3 | 0.4591 | 0.0445 |
| feature | rvq_layer_4 | 0.5034 | -0.0216 |
| feature | rvq_layer_5 | 0.5079 | -0.0007 |

## Nearest-Neighbor Retrieval
| Space | Part | k | Learned improvement vs random | Motion-NN overlap |
|---|---|---:|---:|---:|
| feature | rvq_layer_0 | 1 | 0.6335 | 0.2609 |
| feature | rvq_layer_0 | 5 | 0.5876 | 0.4790 |
| feature | rvq_layer_0 | 10 | 0.5433 | 0.5315 |
| feature | rvq_layer_1 | 1 | 0.3776 | 0.0381 |
| feature | rvq_layer_1 | 5 | 0.3215 | 0.1175 |
| feature | rvq_layer_1 | 10 | 0.2969 | 0.1673 |
| feature | rvq_layer_2 | 1 | 0.2452 | 0.0249 |
| feature | rvq_layer_2 | 5 | 0.2051 | 0.0611 |
| feature | rvq_layer_2 | 10 | 0.1943 | 0.0826 |
| feature | rvq_layer_3 | 1 | 0.1398 | 0.0063 |
| feature | rvq_layer_3 | 5 | 0.1344 | 0.0353 |
| feature | rvq_layer_3 | 10 | 0.1429 | 0.0596 |
| feature | rvq_layer_4 | 1 | 0.0757 | 0.0151 |
| feature | rvq_layer_4 | 5 | 0.0855 | 0.0229 |
| feature | rvq_layer_4 | 10 | 0.1040 | 0.0503 |
| feature | rvq_layer_5 | 1 | 0.0561 | 0.0057 |
| feature | rvq_layer_5 | 5 | 0.1286 | 0.0280 |
| feature | rvq_layer_5 | 10 | 0.1323 | 0.0526 |

## Stratified Correlation Controls
| Space | Part | Pair control | Learned Spearman | Shuffled Spearman | Pairs |
|---|---|---|---:|---:|---:|
| feature | rvq_layer_0 | all | 0.7932 | -0.0127 | 20000 |
| feature | rvq_layer_0 | frequency_matched | 0.8343 | 0.0021 | 9384 |
| feature | rvq_layer_0 | time_matched | 0.7860 | -0.0219 | 18321 |
| feature | rvq_layer_0 | frequency_and_time_matched | 0.8206 | -0.0167 | 4498 |
| feature | rvq_layer_1 | all | 0.3851 | 0.0412 | 20000 |
| feature | rvq_layer_1 | frequency_matched | 0.5500 | 0.0908 | 12246 |
| feature | rvq_layer_1 | time_matched | 0.4223 | 0.0743 | 20000 |
| feature | rvq_layer_1 | frequency_and_time_matched | 0.5873 | 0.1552 | 6893 |
| feature | rvq_layer_2 | all | 0.4529 | 0.0016 | 20000 |
| feature | rvq_layer_2 | frequency_matched | 0.5340 | -0.0184 | 12720 |
| feature | rvq_layer_2 | time_matched | 0.4601 | -0.0236 | 20000 |
| feature | rvq_layer_2 | frequency_and_time_matched | 0.5740 | -0.0487 | 6557 |
| feature | rvq_layer_3 | all | 0.4553 | -0.0701 | 20000 |
| feature | rvq_layer_3 | frequency_matched | 0.5782 | -0.0840 | 12403 |
| feature | rvq_layer_3 | time_matched | 0.4496 | -0.0925 | 20000 |
| feature | rvq_layer_3 | frequency_and_time_matched | 0.5799 | -0.1343 | 6357 |
| feature | rvq_layer_4 | all | 0.5044 | -0.0170 | 20000 |
| feature | rvq_layer_4 | frequency_matched | 0.6101 | -0.0449 | 13612 |
| feature | rvq_layer_4 | time_matched | 0.5175 | -0.0385 | 20000 |
| feature | rvq_layer_4 | frequency_and_time_matched | 0.6433 | -0.0788 | 7274 |
| feature | rvq_layer_5 | all | 0.5132 | 0.0108 | 20000 |
| feature | rvq_layer_5 | frequency_matched | 0.6024 | 0.0283 | 15138 |
| feature | rvq_layer_5 | time_matched | 0.5286 | 0.0056 | 20000 |
| feature | rvq_layer_5 | frequency_and_time_matched | 0.6325 | 0.0435 | 7645 |

## Bootstrap CI
| Space | Part | Control | Spearman mean | 95% CI |
|---|---|---|---:|---:|
| feature | rvq_layer_0 | learned | 0.7990 | [0.7924, 0.8058] |
| feature | rvq_layer_0 | shuffled | 0.0476 | [0.0345, 0.0620] |
| feature | rvq_layer_1 | learned | 0.3900 | [0.3776, 0.4022] |
| feature | rvq_layer_1 | shuffled | -0.0327 | [-0.0469, -0.0187] |
| feature | rvq_layer_2 | learned | 0.4469 | [0.4361, 0.4579] |
| feature | rvq_layer_2 | shuffled | 0.0310 | [0.0174, 0.0450] |
| feature | rvq_layer_3 | learned | 0.4676 | [0.4568, 0.4791] |
| feature | rvq_layer_3 | shuffled | 0.0213 | [0.0072, 0.0350] |
| feature | rvq_layer_4 | learned | 0.5083 | [0.4978, 0.5184] |
| feature | rvq_layer_4 | shuffled | 0.0749 | [0.0621, 0.0884] |
| feature | rvq_layer_5 | learned | 0.5011 | [0.4905, 0.5113] |
| feature | rvq_layer_5 | shuffled | 0.0670 | [0.0528, 0.0802] |

## Trajectory Smoothness
| Part | Control | Step mean | Curvature mean |
|---|---|---:|---:|
| rvq_layer_0 | learned | 72.5528 | 120.2159 |
| rvq_layer_0 | shuffled | 142.1413 | 255.6466 |
| rvq_layer_0 | random_gaussian | 147.7949 | 264.8497 |
| rvq_layer_1 | learned | 44.9833 | 78.3466 |
| rvq_layer_1 | shuffled | 62.9269 | 111.1369 |
| rvq_layer_1 | random_gaussian | 64.8955 | 114.3397 |
| rvq_layer_2 | learned | 33.9870 | 59.3533 |
| rvq_layer_2 | shuffled | 44.5619 | 78.3581 |
| rvq_layer_2 | random_gaussian | 46.2477 | 81.2526 |
| rvq_layer_3 | learned | 28.3428 | 49.5368 |
| rvq_layer_3 | shuffled | 36.2578 | 63.7028 |
| rvq_layer_3 | random_gaussian | 37.5807 | 65.9587 |
| rvq_layer_4 | learned | 24.5143 | 42.7193 |
| rvq_layer_4 | shuffled | 31.6003 | 55.3450 |
| rvq_layer_4 | random_gaussian | 32.9170 | 57.5791 |
| rvq_layer_5 | learned | 21.7032 | 37.7493 |
| rvq_layer_5 | shuffled | 28.1796 | 49.2639 |
| rvq_layer_5 | random_gaussian | 29.5877 | 51.6076 |

## Replacement Summary
| Replacement | Whole delta | Target delta | Local target delta |
|---|---:|---:|---:|
| near | 0.01071 | 0.01071 | 0.06198 |
| mid | 0.02528 | 0.02528 | 0.16296 |
| far | 0.05159 | 0.05159 | 0.41011 |
| random | 0.02502 | 0.02502 | 0.17255 |

## Replacement Per-Part Summary
| Part | Replacement | Local target delta | 95% CI | Count |
|---|---|---:|---:|---:|
| rvq_layer_0 | far | 1.48105 | [1.24845, 1.71365] | 32 |
| rvq_layer_0 | mid | 0.55576 | [0.44919, 0.66234] | 32 |
| rvq_layer_0 | near | 0.13932 | [0.10651, 0.17213] | 32 |
| rvq_layer_0 | random | 0.60315 | [0.42705, 0.77924] | 32 |
| rvq_layer_1 | far | 0.34838 | [0.30258, 0.39417] | 32 |
| rvq_layer_1 | mid | 0.14739 | [0.11906, 0.17572] | 32 |
| rvq_layer_1 | near | 0.07150 | [0.05172, 0.09129] | 32 |
| rvq_layer_1 | random | 0.13589 | [0.10622, 0.16556] | 32 |
| rvq_layer_2 | far | 0.22137 | [0.18616, 0.25659] | 32 |
| rvq_layer_2 | mid | 0.11413 | [0.08937, 0.13888] | 32 |
| rvq_layer_2 | near | 0.06454 | [0.04427, 0.08482] | 32 |
| rvq_layer_2 | random | 0.11498 | [0.09020, 0.13975] | 32 |
| rvq_layer_3 | far | 0.13852 | [0.12421, 0.15283] | 32 |
| rvq_layer_3 | mid | 0.07278 | [0.05756, 0.08800] | 32 |
| rvq_layer_3 | near | 0.04262 | [0.03165, 0.05360] | 32 |
| rvq_layer_3 | random | 0.06509 | [0.05150, 0.07869] | 32 |
| rvq_layer_4 | far | 0.08862 | [0.06919, 0.10805] | 32 |
| rvq_layer_4 | mid | 0.04922 | [0.03970, 0.05875] | 32 |
| rvq_layer_4 | near | 0.02898 | [0.02449, 0.03347] | 32 |
| rvq_layer_4 | random | 0.05881 | [0.04203, 0.07559] | 32 |
| rvq_layer_5 | far | 0.18269 | [0.16576, 0.19962] | 32 |
| rvq_layer_5 | mid | 0.03846 | [0.03187, 0.04504] | 32 |
| rvq_layer_5 | near | 0.02493 | [0.01892, 0.03093] | 32 |
| rvq_layer_5 | random | 0.05740 | [0.03685, 0.07794] | 32 |

## Geometry-Aware Error Cost
| Space | Part | Code-distance bin | Motion damage mean | Pairs |
|---|---|---:|---:|---:|
| feature | rvq_layer_0 | 0 | 66.07519 | 4000 |
| feature | rvq_layer_0 | 1 | 99.27930 | 4000 |
| feature | rvq_layer_0 | 2 | 122.74290 | 4000 |
| feature | rvq_layer_0 | 3 | 155.57201 | 4000 |
| feature | rvq_layer_0 | 4 | 247.50115 | 4000 |
| feature | rvq_layer_1 | 0 | 49.81849 | 4000 |
| feature | rvq_layer_1 | 1 | 65.21792 | 4000 |
| feature | rvq_layer_1 | 2 | 69.11903 | 4000 |
| feature | rvq_layer_1 | 3 | 73.58047 | 4000 |
| feature | rvq_layer_1 | 4 | 93.06456 | 4000 |
| feature | rvq_layer_2 | 0 | 42.76804 | 4000 |
| feature | rvq_layer_2 | 1 | 55.13102 | 4000 |
| feature | rvq_layer_2 | 2 | 67.66075 | 4000 |
| feature | rvq_layer_2 | 3 | 78.93058 | 4000 |
| feature | rvq_layer_2 | 4 | 92.66400 | 4000 |
| feature | rvq_layer_3 | 0 | 37.65267 | 4000 |
| feature | rvq_layer_3 | 1 | 46.32745 | 4000 |
| feature | rvq_layer_3 | 2 | 56.84721 | 4000 |
| feature | rvq_layer_3 | 3 | 69.69445 | 4000 |
| feature | rvq_layer_3 | 4 | 85.36000 | 4000 |
| feature | rvq_layer_4 | 0 | 35.20236 | 4000 |
| feature | rvq_layer_4 | 1 | 40.40931 | 4000 |
| feature | rvq_layer_4 | 2 | 46.00602 | 4000 |
| feature | rvq_layer_4 | 3 | 57.61123 | 4000 |
| feature | rvq_layer_4 | 4 | 82.36066 | 4000 |
| feature | rvq_layer_5 | 0 | 34.49075 | 4000 |
| feature | rvq_layer_5 | 1 | 41.18095 | 4000 |
| feature | rvq_layer_5 | 2 | 48.23758 | 4000 |
| feature | rvq_layer_5 | 3 | 59.24903 | 4000 |
| feature | rvq_layer_5 | 4 | 79.61806 | 4000 |

## Files
- `code_usage.csv`
- `distance_correlation.csv`
- `nn_retrieval.csv`
- `distance_correlation_stratified.csv`
- `bootstrap_ci.csv`
- `geometry_error_cost.csv`
- `geometry_error_cost_summary.csv`
- `trajectory_smoothness.csv`
- `replacement_near_mid_far.csv`
- `replacement_error_cost.csv`
- `summary.json`
