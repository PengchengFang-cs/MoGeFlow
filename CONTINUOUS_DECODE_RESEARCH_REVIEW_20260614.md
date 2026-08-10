# Continuous Decode Research Review

Date: 2026-06-14

Reviewer thread: `84df679c-4d53-4d07-9299-59f17f1f1f5e`

## Context

The balanced HumanML3D PS-CF checkpoint:

`checkpoints/t2m/codeflow_part_structured_newvqtop3_w150_p192_h1152_d6s12_drop005_w0_b64_lr1e4_e600_eval10from0_seed42_test_h200_blossom_g1_20260601/model/best_top3.pt`

was re-evaluated with matched settings:

- HumanML3D `test.txt`
- `seed=42`, `repeat_times=1`
- `steps=96`, `cond_scale=6.0`
- `terminal_mode=tied_logits`
- EMA weights
- multimodality disabled

Only `decode_mode` was changed.

| Decode | FID | Top1 | Top2 | Top3 | Matching | Diversity | CodeAcc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `nearest` | `0.058115` | `0.592457` | `0.782328` | `0.872845` | `2.598813` | `9.980471` | `0.046008` |
| `continuous` | `0.047862` | `0.589440` | `0.784052` | `0.873922` | `2.599523` | `9.913954` | `0.046008` |

## User Question

The question was whether continuous decode beating nearest decode is a new insight sufficient for a separate paper, or whether it should be folded into the current PS-CF paper.

Technical interpretation under discussion:

1. The model uses a VQ/RVQ tokenizer/interface, but the generator is a continuous flow in latent space.
2. At inference, nearest-code projection can be bypassed and the continuous flow output can be decoded directly.
3. The VQ/RVQ codebook may still define or shape the representation, but the best generator output may not be code-aligned.

## Round 1 Review Summary

Reviewer verdict:

- This is not currently sufficient for a standalone paper.
- It should be folded into the current PS-CF paper as an ablation and method clarification.
- The observed result is partly expected: nearest-code projection is a lossy quantization operation, and the reported `CodeAcc=0.046` indicates the flow is not producing code-aligned latents.

Supported claim today:

> For this checkpoint and matched single-seed settings, continuous decoding gives lower FID and essentially unchanged retrieval relative to nearest-code decoding.

Unsupported claims today:

- The effect generalizes across seeds, repeats, checkpoints, CFG/step choices, or datasets.
- The VQ/RVQ codebook improves training as a scaffold.
- Top3 is meaningfully improved; current Top3 difference is too small to claim without variance.
- The finding is a standalone novel decoding paradigm.

Main reviewer objection expected:

> Quantization adds error; continuous decode winning is expected. If the model only lands on codebook entries about 4.6% of the time, why use a codebook at inference, and where is the no-VQ continuous baseline?

## Round 2 Review Summary

The follow-up asked for the minimal current-paper evidence package.

Reviewer recommendation:

1. Decode comparison under a real protocol:
   - `repeat_times=20`
   - at least 3 seeds
   - report mean and 95% CI for FID, R@1/2/3, matching distance, diversity, and multimodality.
2. Robustness across checkpoints:
   - evaluate nearest vs continuous on 3-4 checkpoints along training, not only the metric-selected balanced checkpoint.
3. Mechanistic diagnostic:
   - report CodeAcc
   - report distance from flow output to nearest code
   - correlate per-sample quantization distance with error/metric degradation.
4. Small robustness sweep:
   - CFG values such as `{3, 6, 9}`
   - steps such as `{32, 96}`
   - FID is sufficient for this sweep.

No-VQ continuous baseline:

- Mandatory if claiming the VQ/RVQ tokenizer helps the continuous generator during training.
- Not strictly mandatory if the current paper only claims that continuous decoding is the better inference path for the already-trained PS-CF model.
- Highest leverage optional training experiment: train a no-VQ continuous flow baseline. If it matches or beats VQ-scaffold + continuous decode, the scaffold narrative should be weakened or removed.

## Final Consensus

Use this finding in the current paper, not as a new paper.

Safe framing:

> PS-CF is a continuous flow in a part-structured latent space. Since the generator is not constrained to land exactly on codebook entries, nearest-code projection at inference injects quantization error. We therefore decode the continuous flow output directly. This gives a small but consistent FID improvement while keeping retrieval effectively unchanged.

Unsafe framing:

- Do not claim continuous decode meaningfully improves retrieval unless repeat/seed confidence intervals support it.
- Do not claim the VQ/RVQ codebook is a beneficial training scaffold unless a no-VQ continuous baseline supports that.
- Do not present this as a standalone discovery from current evidence.
- Do not hide the slight diversity drop.

## Claims Matrix

| Evidence outcome | Allowed claim | Not allowed |
| --- | --- | --- |
| Single seed/repeat result only | Continuous decode is promising and was better in the matched seed-42 check. | General improvement, paper-level conclusion, retrieval improvement. |
| Repeat=20 and multi-seed show lower FID with overlapping retrieval CIs | Continuous decoding improves fidelity and preserves retrieval under PS-CF inference. | VQ scaffold helps training. |
| Effect holds across multiple checkpoints and CFG/steps | Continuous decoding is the robust default inference mode for PS-CF. | Standalone paper claim without training-baseline evidence. |
| No-VQ continuous baseline is worse than VQ-scaffold + continuous decode | VQ/RVQ representation may help train or structure the continuous generator. | Broad modality-general claim unless tested beyond motion. |
| No-VQ continuous baseline matches or beats VQ-scaffold + continuous decode | The codebook should be framed as optional or incidental for this generator. | VQ scaffold as core contribution. |

## Recommended Current-Paper Table

Table: inference decoding mode, HumanML3D test, EMA, CFG=6, steps=96.

| Decode | FID down | R@1 up | R@2 up | R@3 up | MM-Dist down | Diversity | MModality | CodeAcc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Nearest/codebook | mean +- CI | mean +- CI | mean +- CI | mean +- CI | mean +- CI | mean +- CI | mean +- CI | mean +- CI |
| Continuous | mean +- CI | mean +- CI | mean +- CI | mean +- CI | mean +- CI | mean +- CI | mean +- CI | mean +- CI |

Bold only statistically supported improvements. If retrieval CIs overlap, describe retrieval as unchanged.

## Suggested Paragraph

> PS-CF's generator is a continuous flow in the part-structured latent space rather than a discrete code predictor. Empirically, its samples are not constrained to codebook entries, with low code-alignment accuracy. Projecting the continuous flow output to the nearest code at inference therefore introduces quantization error. We instead decode the continuous output directly. Across the matched evaluation protocol, continuous decoding lowers FID while keeping retrieval metrics effectively unchanged; diversity decreases slightly. We use continuous decoding by default. Whether the VQ tokenizer additionally regularizes the continuous generator during training is a separate question and requires a no-VQ continuous baseline.

## Prioritized TODO

1. Run nearest vs continuous with `repeat_times=20` and at least 3 seeds on the balanced checkpoint.
2. Repeat nearest vs continuous on 3-4 checkpoints along the training trajectory.
3. Add flow-output-to-nearest-code distance diagnostics and CodeAcc reporting.
4. Run a small CFG/steps robustness sweep.
5. If compute permits, train the no-VQ continuous baseline before making any scaffold claim.

## Follow-Up: Why Use RVQ If Continuous Decode Wins?

Additional user clarification: nearest-vs-continuous decode has reportedly been
tested roughly 2,000 times and the qualitative rule is stable: continuous decode
consistently outperforms nearest decode.

Reviewer follow-up verdict:

- This strengthens the narrow inference claim: continuous decode is robustly
  better than nearest-code projection for this model family.
- It does not by itself prove that RVQ is a useful training scaffold. It mainly
  confirms that snapping a continuous flow output to a discrete codebook adds
  quantization error.
- The important unresolved question is representation-level, not decode-level:
  whether the current RVQ latent space is better for training a continuous flow
  than an equally strong VAE or deterministic AE latent space.

Plausible reasons RVQ may still matter:

1. RVQ may regularize latent geometry into bounded, clustered, well-covered
   regions that are easier for the flow to model.
2. The decoder may be robust to near-code continuous latents because it was
   trained around residual code embeddings.
3. Residual quantization may constrain support and avoid holes or collapse modes
   that can appear in VAE-style continuous latents.
4. The RVQ interface gives comparability and reuse of strong motion tokenizers,
   though this is a practical reason rather than a scientific claim.
5. Part-structured or residual codebooks may create cleaner factorization than a
   monolithic continuous latent.

Decisive experiment:

Train the same PS-CF generator, with matched capacity and protocol, over:

- the current RVQ latent space, decoded continuously;
- a plain VAE latent space, decoded continuously;
- a deterministic AE latent space, decoded continuously.

If RVQ latent + continuous decode clearly beats VAE/AE with confidence intervals,
then the paper-level thesis can become "quantize to train, decode continuously":
discrete residual quantization shapes the representation, but inference should
not project generated samples back to the codebook. If VAE/AE matches or beats it,
then RVQ is incidental and the current paper should avoid scaffold claims.

Additional mechanism probes if RVQ wins:

- latent intrinsic dimensionality, anisotropy, nearest-neighbor distance, and
  clusterability across RVQ/VAE/AE;
- flow convergence curves and FID-vs-epoch across latent spaces;
- decoder robustness to perturbations around training latents;
- RVQ codebook size and residual-depth sweeps under continuous decode;
- part-structured codebook vs single/shared codebook.

Updated safe framing:

> Continuous decode is the right inference path for PS-CF because inference-time
> quantization adds error. RVQ may still be valuable as a representation or
> training scaffold, but that requires a direct RVQ-vs-VAE-vs-AE latent-space
> comparison before it can be claimed as a core contribution.
