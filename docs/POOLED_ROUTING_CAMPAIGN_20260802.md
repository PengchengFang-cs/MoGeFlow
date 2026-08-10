# Pooled-routing / encoder campaign — 2026-08-02 → 08-07

Six MoGeFlow-L runs on HumanML3D, plus a full CFG sweep and one repeat-20
measurement.  Everything below is read out of `full_eval.jsonl` and the sweep
JSONs; nothing is estimated or carried over from earlier campaigns.

## 1. What was varied

All six runs share: MoGeFlow-L (part_hidden_dim 192, hidden_size 1152,
depth_double 6, depth_single 12, 12 heads, mlp_ratio 4.0, dropout 0.05),
Part-VQ `new_vq_overlap_top3_20260529` (codebook target, codebook latent norm),
batch 64, lr 1e-4, 600 epochs, seed 42, unit_length 4, continuous decode,
`test` split, evaluation every 25 epochs at cfg 6.0 / 96 steps / repeat 1 / EMA.

| # | run | text encoder | pooled routing | refiner | params |
|---|-----|--------------|----------------|---------|--------|
| 1 | `hml3d_20260802_e1_nopool` | E1 | **none** (AdaLN = timestep only) | 0 | 692.68M |
| 2 | `hml3d_20260802_e1_pooltoken` | E1 | token (prepended, + type embed) | 0 | 691.06M |
| 3 | `hml3d_20260802_clipL_pooltoken` | CLIP-L | token | 0 | 692.68M |
| 4 | `hml3d_20260802_clipB_pooltoken` | CLIP-B/32 | token | 0 | 698.73M |
| 5 | `hml3d_20260802_clipL_pooltoken_r2` | CLIP-L | token | 2 | 734.90M |
| 6 | `hml3d_20260802_normavg_pooltoken` | qwen3-normavg | token | 2 | 742.56M |

**E1** = our LLM2Vec-Qwen3 cache `humanml3d_qwen3_l2v_normavg_v2_20260802`:
Qwen3-8B base + MNTP adapter + supervised-contrastive adapter (both 08-01),
token feature = parameter-free per-layer LayerNorm averaged over all transformer
layers (`layer_norm_eps` 1e-5), pooled = masked mean over the content span,
ChatML user turn, separator never enters the tokenizer.  55862 captions.

**"no pooled"** is a design choice, not an ablation baseline: the sentence vector
is absent from the model entirely (no `text_pooled_proj` in the state dict), text
reaches the backbone only through joint attention, and AdaLN carries the timestep
alone.

## 2. Per-run results (cfg 6.0, single run, seed 42)

| # | run | best-FID ep | FID | R@1 | R@3 | MMDist | best-R@3 ep | FID | R@3 |
|---|-----|---|---|---|---|---|---|---|---|
| 1 | E1 / no pooled | 400 | 0.0479 | 0.5767 | 0.8556 | 2.6463 | 100 | 0.1737 | **0.8832** |
| 2 | E1 / pool-token | 425 | 0.0430 | 0.5784 | 0.8601 | 2.6153 | 125 | 0.1805 | 0.8828 |
| 3 | CLIP-L / pool-token | 325 | 0.0472 | 0.5823 | 0.8483 | 2.6459 | 125 | 0.1468 | 0.8679 |
| 4 | CLIP-B / pool-token | 350 | 0.0506 | 0.5703 | 0.8513 | 2.6641 | 150 | 0.1258 | 0.8703 |
| 5 | CLIP-L / pool-token / r2 | 500 | **0.0361** | 0.5597 | 0.8394 | 2.7300 | 200 | 0.1549 | 0.8696 |
| 6 | normavg / pool-token / r2 | 525 | 0.0631 | 0.5787 | 0.8621 | 2.6117 | 125 | 0.2134 | **0.8853** |

### Late-training stability (best FID → ep600)

| run | best FID | ep600 | drift |
|---|---|---|---|
| E1 / pool-token | 0.0430 | 0.0448 | **+0.0018** |
| normavg / r2 | 0.0631 | 0.0660 | +0.0030 |
| CLIP-L / r2 | 0.0361 | 0.0416 | +0.0054 |
| E1 / no pooled | 0.0479 | 0.0718 | +0.0239 |
| CLIP-B | 0.0506 | 0.1268 | +0.0762 |
| CLIP-L | 0.0472 | 0.1598 | **+0.1126** |

CLIP-L's +0.1126 is 18× the single-run FID noise and monotone over nine
consecutive evaluations, so it is a real property of the trajectory, not noise.

## 3. CFG sweep — 72/72, zero failures

12 checkpoints (6 runs × {best_fid, best_top3}) × cfg ∈ {2,3,4,5,6,7}, every
other setting identical to the training-time evaluation.  Validated first by
reproducing run 2's training number exactly (FID 0.0430, R@1 0.5784; R@2/R@3
differed by 0.0002, ≈ one sample's retrieval tie-break).

Full tables live in `eval_results/cfg_sweep_20260803/`.  Key structure:

- **R@3 peaks at cfg 4–6** on every curve — the optimum is bracketed.
- **FID keeps falling through cfg 7** on 11 of 12 curves — the optimum is *not*
  bracketed.  Only `e1_pooltoken best_fid` has an interior FID minimum (cfg 6,
  0.0430, with cfg 7 rebounding to 0.0452).  Any "lowest FID" claim across runs
  is therefore range-limited; the sweep was deliberately not extended past 7.
- cfg 6.0 (the value used throughout training) is FID-optimal for exactly one
  curve and R@3-optimal for none.

### Best R@3 anywhere in the sweep

| run | ckpt | cfg | R@3 | R@1 | MMDist | FID |
|---|---|---|---|---|---|---|
| normavg / r2 | best_top3 | 7 | **0.8866** | **0.6071** | **2.5434** | 0.1591 |
| normavg / r2 | best_top3 | 6 | 0.8851 | 0.6088 | 2.5407 | 0.1645 |
| E1 / no pooled | best_top3 | 5 | 0.8828 | 0.5970 | 2.5649 | 0.2056 |
| E1 / pool-token | best_top3 | 5 | 0.8821 | 0.5974 | 2.5643 | 0.1989 |

15 of 72 combinations reach R@3 ≥ 0.87; 7 reach ≥ 0.88.  All of them sit at
FID 0.14–0.28.  **normavg's R@3 is still rising at cfg 7** (0.8823 → 0.8851 →
0.8866), so its own optimum is also unbracketed.

## 4. repeat-20 (seeds 42–61) — 1 of 8 completed

Only one configuration finished before the A100 allocations expired.

`clipL_pooltoken_r2 / best_fid / cfg 7`

| metric | single run | repeat-20 mean ± 95% CI |
|---|---|---|
| FID | 0.0338 | **0.0440 ± 0.0030** |
| R@1 | 0.5547 | 0.5552 ± 0.0023 |
| R@2 | 0.7509 | 0.7465 ± 0.0025 |
| R@3 | 0.8388 | 0.8366 ± 0.0023 |
| MMDist | 2.7357 | 2.7288 ± 0.0054 |
| Diversity | 9.8013 | 9.5751 ± 0.0552 |

**The single run understated FID by 0.0102 — 3.4× the CI half-width — while all
three retrieval metrics landed inside their CIs.**  FID is the volatile metric;
R@1/R@3/MMDist are reproducible to ±0.002.  Any ranking built on single-run FID
differences smaller than ~0.01 is unreliable.

This also dissolves the FID leaderboard: the eight candidates spanned
0.0338–0.0465, entirely inside FID's single-run jitter, and the one that was
measured properly moved from 1st (0.0338) to mid-pack (0.0440).

## 5. Balanced operating points

Neither extreme checkpoint satisfies "low FID **and** R@3 > 0.87": across all 72
sweep combinations, FID ≤ 0.10 caps R@3 at 0.8621, and R@3 > 0.87 costs FID ≥ 0.14.

The balanced points are at **intermediate epochs**, which the sweep never
touched because only `best_fid.pt` and `best_top3.pt` are saved.  Reading the
144 training evaluations directly (cfg 6.0):

| run | ep | FID | R@1 | R@3 | MMDist |
|---|---|---|---|---|---|
| **E1 / no pooled** | 250 | **0.0677** | 0.5845 | **0.8705** | 2.6011 |
| **E1 / no pooled** | 225 | 0.0787 | 0.5871 | **0.8726** | 2.6025 |
| **E1 / no pooled** | 200 | 0.0975 | 0.5905 | **0.8767** | 2.5930 |

Three points, all from run 1.  No other run enters the region.
**These weights no longer exist** — the trainer keeps only best-FID, best-R@3 and
latest, and none of ep200/225/250 is either.

## 6. no-pooled vs pool-token — the one clean A/B

Same cache, same encoder, same refiner depth, same seed; the only difference is
whether the pooled vector is absent or prepended as an attended token.

Paired across all 24 matched epochs: **no-pooled wins both metrics at 7 epochs,
pool-token at 5, split at 12.**  The win is phase-dependent and the phases are
clean:

| phase | winner | evidence |
|---|---|---|
| ep100–350 (balanced region) | **no pooled** | 6 double wins vs 0; FID lower at every one of the 11 epochs by 0.015–0.030 |
| ep400–600 (low-FID region) | **pool-token** | 4 double wins vs 0; FID lower by 0.005–0.027, and it is the only run that does not degrade late (+0.0018 vs +0.0239) |

Best balanced point each can reach (FID < 0.10, maximise R@3):

| design | ep | FID | R@3 | clears 0.87? |
|---|---|---|---|---|
| no pooled | 200 | 0.0975 | 0.8767 | **yes** |
| pool-token | 250 | 0.0935 | 0.8698 | no |

## 7. Conclusions

1. **Pool-token does not improve the no-pooled design where it matters.**  In the
   balanced region it is systematically worse on both metrics, and it produces no
   operating point with FID < 0.10 and R@3 > 0.87.  Its advantage is confined to
   late training / best-FID checkpoints, whose R@3 (~0.86) is below the target.
2. **Pool-token does buy late-training stability.**  It is the only run whose FID
   is flat from ep425 to ep600 (+0.0018).  This is a checkpoint-selection
   robustness property, not a quality gain.
3. **Encoder ranking is cfg-dependent and mostly inside noise.**  At cfg 6 the
   best-FID checkpoints order E1 < CLIP-L < CLIP-B; at cfg 7 the top three
   collapse into 0.0013.  The only encoder result that survives is on R@3:
   qwen3-normavg leads (0.8866) with the best R@1 and MMDist as well, and E1
   follows; both CLIP variants trail by 0.015–0.020 on R@3 at every cfg.
4. **The refiner trades retrieval for FID.**  CLIP-L + refiner reaches the lowest
   single-run FID in the campaign (0.0361) with the worst R@1/R@3/MMDist of all
   six runs.  Under a both-metrics criterion it is the weakest option.
5. **FID is the noisy metric; retrieval is the stable one.**  repeat-20 gives
   ±0.0030 on FID against a single-run error of 0.0102, versus ±0.0023 on R@3
   with the single run inside the CI.  Ranking by single-run FID — as this
   campaign initially did — is the wrong axis.
6. **The checkpoint-selection policy is the binding limitation.**  Saving only
   the two single-metric optima guarantees that the balanced region is measured
   but never retained.  Every balanced point this campaign found is unrecoverable.

## 8. Known limitations

- **One training seed per arm.**  Nothing here separates design effects from
  training-trajectory variance; all claims are checkpoint-level, not method-level.
- **FID optima unbracketed.**  The sweep stops at cfg 7 while 11 of 12 FID curves
  are still descending; normavg's R@3 is also still rising at cfg 7.
- **repeat-20 is 1/8 complete.**  Only `clipL_r2 best_fid cfg7` has a mean.
- **Text RoPE is degenerate.**  `text_pos = torch.zeros(...)` at
  `dit_blocks.py:365/635/707` makes the text stream permutation-invariant; word
  order survives only inside the encoder features.  Never tested.

## 9. Artifacts

| what | where |
|---|---|
| training trajectories | `checkpoints/t2m/hml3d_20260802_*/logs/full_eval.jsonl` |
| CFG sweep (72 JSONs) | `eval_results/cfg_sweep_20260803/` |
| repeat-20 (1 JSON) | `eval_results/repeat20_20260804/` |
| sweep / repeat workers | `run_logs/launch/cfg_sweep_worker_20260803.sh`, `repeat_worker_20260804.sh` |
| run management | `run_logs/launch/pt6_manage_20260802.sh` |

## 10. Operational notes

- `/run/user/$UID` is removed when the last login session ends, which kills the
  tmux server.  The training processes survive inside their slurm steps, so a
  tmux-based liveness check reports a false DOWN — on 08-04 that caused a second
  trainer to be launched against the same checkpoints.  `pt6_manage_20260802.sh`
  now asks slurm for the process instead.  `loginctl enable-linger` is set.
- slurm allocations expiring silently killed the repeat workers for ~40 h before
  it was noticed; monitors now alert when a job leaves RUNNING.
