# Paper revision campaign — 2026-08-08 → 08-09

Log of the review→rebuttal→fix cycle on the ICLR 2027 draft: what was measured,
what was concluded, and exactly how each conclusion was written into the paper.
Paper lives in the local Overleaf clone `projects/Umdd/overleaf-paper`
(`iclr2027_conference.tex`); all edits are LOCAL and UNPUSHED.

## 1. Review panel baseline (before fixes)

Three independent ICLR-style judges on the as-compiled draft:

| Judge | Focus | Rating | Verdict core |
|---|---|---|---|
| R1 | novelty/claims | 5/10 | good premise, protocol inconsistencies + repro void |
| R2 | empirical rigor | 4/10 | claims lack the statistics the paper itself promises |
| R3 | clarity/repro | 4/10 | no implementation details survive to the PDF |

Consensus strengths (kept, now reinforced): the three-step geometry diagnostic
(correlation → shuffled control → causal replacement); HumanML3D R@3 0.874 vs
SALAD 0.857 (≈6× pooled CI); candid non-dominance reporting.

## 2. Evidence produced in this campaign

### 2.1 KIT repeat-20 (clean cache, refiner-0 architecture, continuous decode)

Both gate checkpoints of `kit_20260807_nopool_gate`, 20 seeds (42–61),
96 steps, cfg 6.0, test:

| ckpt | R@1 | R@2 | R@3 | FID | MM-Dist | Div |
|---|---|---|---|---|---|---|
| ep320 | .4837±.007 | .7259±.005 | .8395±.003 | .1914±.005 | 2.5107±.015 | 10.832±.096 |
| **ep350 (chosen)** | .4834±.007 | .7244±.006 | .8366±.004 | **.1732±.005** | **2.5067±.016** | 10.860±.093 |

Decision (user): ep350 — retrieval tied within CI, FID better by ≈3.6×CI.
Conclusion: **KIT flips from concession to lead** — best R@1/R@2/R@3/MM-Dist
among generated methods (previous bests 0.482 / 0.711 / 0.828 / 2.552); FID
0.173 vs best 0.135. The old row (0.447/0.716/0.841/0.225, single run, nearest
decode, deprecated separator cache, refiner-2 model) was invalid on four counts.
JSONs: `eval_results/kit_repeat20_20260808/{ep320,ep350}/`.

### 2.2 E1 — terminal off-lattice offset (δ)

512 test prompts, gate_ep0300, training-time protocol. δ = dist-to-nearest-code
÷ (half median intra-codebook NN spacing):
pooled median **0.68**, mean 0.78, p90 1.43; fraction δ<0.5 / <1 / ≥1 =
33% / 74% / **26%**. Per-group medians 0.61–0.80.
Conclusion: terminal states concentrate near the lattice but carry substantial
sub-code offsets; a quarter of frame-group states sit beyond half-spacing.
This REPLACES the wrong "4.4×10¹² states ⇒ dense" cardinality argument
(cardinality ≠ metric density) with a measurement, and directly motivates why
re-quantization discards information.
JSON: `eval_results/diag_20260809/offlattice_delta.json`.

### 2.3 E2 — whitened-metric correlation check

The flow trains on codebook-whitened latents ((z−μ_p)/σ_p), while §3.2's ρ was
measured in raw Euclidean distance — the one hole that threatened the logic
chain. Recomputed under D_p(k,k′)=‖(e_k−e_k′)⊘σ_p‖₂:
mean ρ raw **0.815** → whitened **0.838**; every group stable or higher
(right_leg 0.759→0.821, left_leg 0.632→0.694); bootstrap CI widths ≤0.03;
shuffled |ρ|≤0.12. Conclusion: the measured geometry holds in the metric the
model actually optimizes — hole closed with a positive result.
JSON: `eval_results/diag_20260809/whitened_rho.json`.

### 2.4 E3 — second tokenizer family (generic residual VQ)

Identical protocol on `rvq_nq6_dc512_nc512_noshare_qdp0.2` (the tokenizer the
generic-RVQ ablation arms used — verified from the June runs' surviving
options.json), layer axis in place of part axis, bootstrap 1000:

| RVQ layer | ρ | 95% CI |
|---|---|---|
| 0 (base) | 0.800 | [0.791, 0.805] |
| 1 | 0.397 | [0.384, 0.407] |
| 2 | 0.479 | [0.468, 0.489] |
| 3 | 0.533 | [0.523, 0.543] |

Shuffled controls ≈0 everywhere. Conclusions:
(a) the diagnostic has discriminative power — it is NOT satisfied by every
learned codebook (kills the "tautology" objection, R1's flip condition);
(b) most of the space a generic-RVQ flow regresses over is weakly
decoder-aligned, which is mechanistically consistent with its weaker joint
alignment–fidelity in Table 4 — the geometry→performance loop is now closed
qualitatively. Part-VQ, by contrast, has ALL six groups strong (mean 0.821).
Outputs: `eval_results/diag_20260809/rvq_geometry/{train512,test512}/`.

### 2.5 Code artifacts (written by agent, adversarially reviewed, all PASS)

`models/codeflow/partvae_vq.py` (+ additive registration; kv_part byte-identical,
DECODE_MODE untouched), `run_logs/launch/vae_b_20260809.sh` (307.55M exact),
`tools/offlattice_delta.py`, `tools/whitened_rho.py`,
`tools/motion_code_geometry_diagnostics_rvq.py`, `run_logs/launch/diag_rvq_20260809.sh`.
Reviewer behaviorally excluded the K=D=512 codebook-transpose trap.

## 3. How each conclusion was written into the paper

| Evidence | Paper location | Wording decision |
|---|---|---|
| KIT ep350 r20 | Table 1 KIT row (+CIs); §4.2 KIT paragraph; Conclusion | Concession prose deleted; now "best retrieval at every rank"; no defensive framing |
| KIT flips positive | Abstract | KIT enters the abstract results sentence ("best retrieval at every rank") |
| δ measurement | §3.5 (old L483) + Appendix "Terminal off-lattice offset" | Replaces the cardinality argument; states median 0.68 / 26% beyond half-spacing; "these offsets are the point" |
| Whitened ρ | §4.3 (one sentence) + Appendix "Whitened-metric check" | "geometry holds in the space the model optimizes" |
| RVQ control | §4.3 (one para) + Appendix "Generic residual-VQ control" | Framed as discriminative-power + mechanism for Table 4, not as an attack on RVQ |
| Backbone reality | §3.4 equations | Per-part projections, dual-stream/single-stream, RoPE (const text pos), per-part AdaLN heads — replaces a formula describing a model that did not exist |
| Conditioning path | §3.4 new paragraph | Encoder named (bidirectional Qwen3-8B, all-layer LN-avg, frozen), τ-only AdaLN, drop 0.1, ∅=empty caption, text order-free at attention level |
| Single-level VQ | §3.1 one sentence | "no residual levels; reachable space exactly the product lattice" |
| Dense-lattice myth | Abstract/Intro/Related (4 sites) | "dense" deleted globally; justification now decoder-response + measured δ |
| "measured properties" contradiction | Related L138 | Now claims only what is measured (distance→motion), TODO deleted |
| LDM delta | Related | Verified against Rombach et al. (VQ-reg diffuses pre-quantization, quantizer absorbed into decoder); TODO deleted |
| SALAD/MoGenTS | Related prose | Named and positioned (they keep the categorical interface; we keep their structural bias and replace the interface) |
| Table 4 discipline | §4.4 | One row = one checkpoint, both metrics jointly; envelope wording deleted; L row = Table 1 numbers (统一铁律); Discrete row kept with † |
| \NUM macro | preamble | Draft-bold removed; numbers final |

Compile check after all edits: PDF builds, 0 undefined citations, 0 undefined refs.

## 4. Decisions registry (settled — do not re-raise)

- HumanML3D headline 0.874/0.048 沿用 (Table 1 = Table 4 L row = intro = abstract; one model one number everywhere).
- Decode ablation Table 5(a) keeps its original single-replication numbers
  (0.058/0.873 vs 0.048/0.874), protocol disclosed in appendix. Settled.
- Discrete Diffusion row kept as early-experiment († caption). No capacity-matched
  rerun. Settled.
- No seed studies. No MotionMillion CIs. cfg sweep dropped from the paper
  (user may run their own). Split naming: "on HumanML3D" — neither test nor
  validation named, nothing false stated.
- E7 (capacity-matched categorical) cancelled: cross-architecture param-matching
  rejected as a fairness criterion by the user.
- Citations: never remove/swap without explicit approval (self-citations deliberate).

## 5. Open items

1. **AE@B / VAE@B** (`pscf_enclat_w100_*`, `pscf_vaeB_*`, both CLIP-family B-width,
   full-schedule target): running on the two H200 allocations, which expire before
   ep600 (~ep250 reachable). User: run to expiry, resume later on new cards.
   Monitor reports expiry + last epoch; latest.pt preserved.
2. **Abstract "consistently weaker" sentence** — the ONLY pending text; finalize
   from AE@B/VAE@B results when they land.
3. Overleaf: all changes local, uncommitted, unpushed (push only on explicit
   instruction).
4. AI use statement: user writes the facts at the end; Ethics/Repro statements
   recommended-not-required, still absent.
