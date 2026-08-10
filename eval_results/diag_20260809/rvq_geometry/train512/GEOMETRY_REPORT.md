# Motion Code Geometry Diagnostics

## Inputs
- KV root: `/scratch/pf2m24/projects/Umdd/KV-Control`
- Data root: `/iridisfs/scratch/pf2m24/data/HumanML3D/HumanML3D`
- VQ checkpoint: `/iridisfs/scratch/pf2m24/projects/Umdd/momask-codes/checkpoints/t2m/rvq_nq6_dc512_nc512_noshare_qdp0.2/model/net_best_fid.tar`
- Partition file: `synthetic_full_body_per_rvq_layer`
- Split: `train`
- Motions encoded: `512`
- Latent length: `49`

## Code Usage
| Part | Active | Dead | Perplexity | Tokens |
|---|---:|---:|---:|---:|
| rvq_layer_0 | 508 | 4 | 331.649 | 17543 |
| rvq_layer_1 | 511 | 1 | 365.641 | 17543 |
| rvq_layer_2 | 512 | 0 | 375.083 | 17543 |
| rvq_layer_3 | 511 | 1 | 375.557 | 17543 |
| rvq_layer_4 | 512 | 0 | 394.636 | 17543 |
| rvq_layer_5 | 510 | 2 | 399.264 | 17543 |

## Distance Correlation
| Space | Part | Learned Spearman | Shuffled Spearman |
|---|---|---:|---:|
| feature | rvq_layer_0 | 0.8001 | -0.0646 |
| feature | rvq_layer_1 | 0.3970 | -0.0047 |
| feature | rvq_layer_2 | 0.4785 | -0.0000 |
| feature | rvq_layer_3 | 0.5240 | 0.0028 |
| feature | rvq_layer_4 | 0.4855 | -0.0408 |
| feature | rvq_layer_5 | 0.5364 | -0.0336 |

## Nearest-Neighbor Retrieval
| Space | Part | k | Learned improvement vs random | Motion-NN overlap |
|---|---|---:|---:|---:|
| feature | rvq_layer_0 | 1 | 0.6427 | 0.2931 |
| feature | rvq_layer_0 | 5 | 0.5709 | 0.4972 |
| feature | rvq_layer_0 | 10 | 0.5417 | 0.5500 |
| feature | rvq_layer_1 | 1 | 0.3157 | 0.0494 |
| feature | rvq_layer_1 | 5 | 0.3201 | 0.1160 |
| feature | rvq_layer_1 | 10 | 0.3020 | 0.1590 |
| feature | rvq_layer_2 | 1 | 0.2180 | 0.0497 |
| feature | rvq_layer_2 | 5 | 0.2080 | 0.0602 |
| feature | rvq_layer_2 | 10 | 0.1904 | 0.0851 |
| feature | rvq_layer_3 | 1 | 0.1256 | 0.0256 |
| feature | rvq_layer_3 | 5 | 0.1180 | 0.0396 |
| feature | rvq_layer_3 | 10 | 0.1449 | 0.0732 |
| feature | rvq_layer_4 | 1 | 0.0068 | 0.0089 |
| feature | rvq_layer_4 | 5 | 0.0880 | 0.0391 |
| feature | rvq_layer_4 | 10 | 0.1008 | 0.0565 |
| feature | rvq_layer_5 | 1 | 0.0814 | 0.0058 |
| feature | rvq_layer_5 | 5 | 0.0919 | 0.0279 |
| feature | rvq_layer_5 | 10 | 0.1048 | 0.0491 |

## Stratified Correlation Controls
| Space | Part | Pair control | Learned Spearman | Shuffled Spearman | Pairs |
|---|---|---|---:|---:|---:|
| feature | rvq_layer_0 | all | 0.7996 | 0.0176 | 20000 |
| feature | rvq_layer_0 | frequency_matched | 0.8278 | 0.0340 | 10368 |
| feature | rvq_layer_0 | time_matched | 0.7949 | 0.0132 | 19586 |
| feature | rvq_layer_0 | frequency_and_time_matched | 0.8322 | 0.0525 | 4906 |
| feature | rvq_layer_1 | all | 0.3967 | -0.0059 | 20000 |
| feature | rvq_layer_1 | frequency_matched | 0.5328 | 0.0020 | 12960 |
| feature | rvq_layer_1 | time_matched | 0.3889 | -0.0227 | 20000 |
| feature | rvq_layer_1 | frequency_and_time_matched | 0.5595 | -0.0029 | 7051 |
| feature | rvq_layer_2 | all | 0.4813 | -0.0429 | 20000 |
| feature | rvq_layer_2 | frequency_matched | 0.5309 | -0.0485 | 12800 |
| feature | rvq_layer_2 | time_matched | 0.4956 | -0.0855 | 20000 |
| feature | rvq_layer_2 | frequency_and_time_matched | 0.5596 | -0.0904 | 6829 |
| feature | rvq_layer_3 | all | 0.5303 | -0.0116 | 20000 |
| feature | rvq_layer_3 | frequency_matched | 0.6219 | 0.0271 | 12090 |
| feature | rvq_layer_3 | time_matched | 0.5448 | -0.0060 | 20000 |
| feature | rvq_layer_3 | frequency_and_time_matched | 0.6322 | 0.0198 | 6370 |
| feature | rvq_layer_4 | all | 0.4964 | 0.0011 | 20000 |
| feature | rvq_layer_4 | frequency_matched | 0.6044 | -0.0113 | 14112 |
| feature | rvq_layer_4 | time_matched | 0.4974 | -0.0255 | 20000 |
| feature | rvq_layer_4 | frequency_and_time_matched | 0.6003 | -0.0345 | 8187 |
| feature | rvq_layer_5 | all | 0.5443 | 0.0196 | 20000 |
| feature | rvq_layer_5 | frequency_matched | 0.6484 | 0.0308 | 14620 |
| feature | rvq_layer_5 | time_matched | 0.5636 | 0.0141 | 20000 |
| feature | rvq_layer_5 | frequency_and_time_matched | 0.6537 | 0.0249 | 7718 |

## Bootstrap CI
| Space | Part | Control | Spearman mean | 95% CI |
|---|---|---|---:|---:|
| feature | rvq_layer_0 | learned | 0.7977 | [0.7913, 0.8045] |
| feature | rvq_layer_0 | shuffled | 0.0314 | [0.0181, 0.0444] |
| feature | rvq_layer_1 | learned | 0.3954 | [0.3840, 0.4074] |
| feature | rvq_layer_1 | shuffled | -0.0057 | [-0.0204, 0.0090] |
| feature | rvq_layer_2 | learned | 0.4786 | [0.4682, 0.4893] |
| feature | rvq_layer_2 | shuffled | 0.0116 | [-0.0043, 0.0251] |
| feature | rvq_layer_3 | learned | 0.5332 | [0.5234, 0.5433] |
| feature | rvq_layer_3 | shuffled | -0.0544 | [-0.0669, -0.0413] |
| feature | rvq_layer_4 | learned | 0.4850 | [0.4741, 0.4953] |
| feature | rvq_layer_4 | shuffled | -0.0120 | [-0.0261, 0.0021] |
| feature | rvq_layer_5 | learned | 0.5332 | [0.5228, 0.5437] |
| feature | rvq_layer_5 | shuffled | -0.0135 | [-0.0262, -0.0001] |

## Trajectory Smoothness
| Part | Control | Step mean | Curvature mean |
|---|---|---:|---:|
| rvq_layer_0 | learned | 75.7044 | 125.9685 |
| rvq_layer_0 | shuffled | 137.3716 | 246.6968 |
| rvq_layer_0 | random_gaussian | 150.4385 | 269.5145 |
| rvq_layer_1 | learned | 46.2681 | 80.5749 |
| rvq_layer_1 | shuffled | 63.2425 | 111.6453 |
| rvq_layer_1 | random_gaussian | 66.2368 | 116.6057 |
| rvq_layer_2 | learned | 34.6576 | 60.5253 |
| rvq_layer_2 | shuffled | 44.4451 | 77.9975 |
| rvq_layer_2 | random_gaussian | 46.7956 | 82.1021 |
| rvq_layer_3 | learned | 28.7730 | 50.2115 |
| rvq_layer_3 | shuffled | 36.7068 | 64.3430 |
| rvq_layer_3 | random_gaussian | 37.9984 | 66.5501 |
| rvq_layer_4 | learned | 24.7782 | 43.1942 |
| rvq_layer_4 | shuffled | 32.1868 | 56.3956 |
| rvq_layer_4 | random_gaussian | 33.1489 | 57.9526 |
| rvq_layer_5 | learned | 21.6681 | 37.7678 |
| rvq_layer_5 | shuffled | 28.3488 | 49.5282 |
| rvq_layer_5 | random_gaussian | 29.5806 | 51.6194 |

## Replacement Summary
| Replacement | Whole delta | Target delta | Local target delta |
|---|---:|---:|---:|
| near | 0.00743 | 0.00743 | 0.05310 |
| mid | 0.01915 | 0.01915 | 0.14980 |
| far | 0.04572 | 0.04572 | 0.35973 |
| random | 0.02044 | 0.02044 | 0.15972 |

## Replacement Per-Part Summary
| Part | Replacement | Local target delta | 95% CI | Count |
|---|---|---:|---:|---:|
| rvq_layer_0 | far | 1.21104 | [1.06961, 1.35247] | 32 |
| rvq_layer_0 | mid | 0.48107 | [0.37117, 0.59096] | 32 |
| rvq_layer_0 | near | 0.10131 | [0.06782, 0.13479] | 32 |
| rvq_layer_0 | random | 0.48974 | [0.39795, 0.58154] | 32 |
| rvq_layer_1 | far | 0.31580 | [0.27334, 0.35826] | 32 |
| rvq_layer_1 | mid | 0.15050 | [0.11419, 0.18680] | 32 |
| rvq_layer_1 | near | 0.06541 | [0.04060, 0.09021] | 32 |
| rvq_layer_1 | random | 0.14360 | [0.10997, 0.17723] | 32 |
| rvq_layer_2 | far | 0.19796 | [0.17181, 0.22412] | 32 |
| rvq_layer_2 | mid | 0.09775 | [0.07918, 0.11632] | 32 |
| rvq_layer_2 | near | 0.04739 | [0.03422, 0.06057] | 32 |
| rvq_layer_2 | random | 0.12364 | [0.08760, 0.15968] | 32 |
| rvq_layer_3 | far | 0.17570 | [0.12851, 0.22290] | 32 |
| rvq_layer_3 | mid | 0.07766 | [0.05952, 0.09581] | 32 |
| rvq_layer_3 | near | 0.04697 | [0.03626, 0.05768] | 32 |
| rvq_layer_3 | random | 0.09459 | [0.06523, 0.12396] | 32 |
| rvq_layer_4 | far | 0.07177 | [0.05677, 0.08677] | 32 |
| rvq_layer_4 | mid | 0.04580 | [0.03656, 0.05504] | 32 |
| rvq_layer_4 | near | 0.02857 | [0.02111, 0.03603] | 32 |
| rvq_layer_4 | random | 0.05416 | [0.04258, 0.06574] | 32 |
| rvq_layer_5 | far | 0.18610 | [0.16624, 0.20596] | 32 |
| rvq_layer_5 | mid | 0.04601 | [0.03707, 0.05495] | 32 |
| rvq_layer_5 | near | 0.02894 | [0.02232, 0.03556] | 32 |
| rvq_layer_5 | random | 0.05258 | [0.04199, 0.06317] | 32 |

## Geometry-Aware Error Cost
| Space | Part | Code-distance bin | Motion damage mean | Pairs |
|---|---|---:|---:|---:|
| feature | rvq_layer_0 | 0 | 67.93440 | 4000 |
| feature | rvq_layer_0 | 1 | 100.43530 | 4000 |
| feature | rvq_layer_0 | 2 | 123.28051 | 4000 |
| feature | rvq_layer_0 | 3 | 153.32650 | 4000 |
| feature | rvq_layer_0 | 4 | 230.96463 | 4000 |
| feature | rvq_layer_1 | 0 | 50.24366 | 4000 |
| feature | rvq_layer_1 | 1 | 67.51581 | 4000 |
| feature | rvq_layer_1 | 2 | 74.31221 | 4000 |
| feature | rvq_layer_1 | 3 | 77.83890 | 4000 |
| feature | rvq_layer_1 | 4 | 87.28592 | 4000 |
| feature | rvq_layer_2 | 0 | 43.88758 | 4000 |
| feature | rvq_layer_2 | 1 | 54.54588 | 4000 |
| feature | rvq_layer_2 | 2 | 67.79872 | 4000 |
| feature | rvq_layer_2 | 3 | 81.40921 | 4000 |
| feature | rvq_layer_2 | 4 | 99.57693 | 4000 |
| feature | rvq_layer_3 | 0 | 36.94258 | 4000 |
| feature | rvq_layer_3 | 1 | 48.91546 | 4000 |
| feature | rvq_layer_3 | 2 | 61.35893 | 4000 |
| feature | rvq_layer_3 | 3 | 75.74826 | 4000 |
| feature | rvq_layer_3 | 4 | 91.85958 | 4000 |
| feature | rvq_layer_4 | 0 | 37.53989 | 4000 |
| feature | rvq_layer_4 | 1 | 43.69845 | 4000 |
| feature | rvq_layer_4 | 2 | 50.00358 | 4000 |
| feature | rvq_layer_4 | 3 | 58.87797 | 4000 |
| feature | rvq_layer_4 | 4 | 87.33898 | 4000 |
| feature | rvq_layer_5 | 0 | 34.15382 | 4000 |
| feature | rvq_layer_5 | 1 | 39.78545 | 4000 |
| feature | rvq_layer_5 | 2 | 47.24516 | 4000 |
| feature | rvq_layer_5 | 3 | 59.10135 | 4000 |
| feature | rvq_layer_5 | 4 | 84.57545 | 4000 |

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
