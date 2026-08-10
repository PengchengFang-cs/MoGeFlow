"""Mask-aware Residual Vector Quantizer (RVQ) for the Graph-VQVAE tokenizer.

Design source: handoff/20260608_graph_vqvae_l5_pipeline_plan.md §"Quantizer" +
handoff/20260608_0600_graph_vqvae_review_verdict_and_forks.md (forks F3/F4,
amendment-1 four leakage paths, amendment-3 DDP EMA sync).

This is a NEW module under src/models/vq_model/ — it does NOT import or modify
the Gaussian Graph-VAE. Inspired by MoGenTS residual_vq/quantizer (EMA codebook,
dead-code reset, residual stages, quantizer dropout) but adapted to our
arbitrary-topology setting: tokens are graph-pooled COARSE SLOTS with a per-token
validity mask, NOT fixed human-joint grid cells. Padded slots / padded latent
frames must never touch the codebook or contribute to the commit loss.

Hard requirements implemented here (mask-aware quantization, amendment 1):
  (1a) dead-code RESET samples replacement codes from VALID tokens only
       (never the padded zero-vectors that flood the batch).
  (1b) EMA cluster-size + cluster-sum DENOMINATORS exclude padded tokens
       (one-hot is zeroed on invalid tokens before the EMA accumulation).
  (1c) straight-through gradient is masked AFTER the STE detach:
       z_q = (x + (q - x).detach()) * valid_mask[..., None]  — so a padded
       token contributes exactly zero to z_q AND has zero gradient path.
  (F4) commit loss is returned RAW (unweighted). commit_weight (0.02) is
       applied ONCE downstream in the loss wrapper. Never squared here.

Numerics (bf16-safe): codebook distance, argmin, and ALL EMA bookkeeping run in
fp32 even under bf16 autocast. The input x is cast to fp32 inside forward; the
returned quantized tensor is cast back to x's original dtype so the decoder sees
the autocast dtype it expects (mirrors the existing GraphAttentionBlock pattern:
sensitive reductions forced fp32, surrounding flow keeps the autocast dtype).

DDP (amendment 3): the EMA update all-reduces (SUM) the per-rank cluster-size and
cluster-sum statistics so every rank applies the SAME codebook update from the
GLOBAL valid-token population. Commit-loss GLOBAL normalization is handled by the
loss wrapper (it needs the global valid-token count, which it all-reduces there).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


def _ddp_active() -> bool:
    """True iff a torchrun DDP process group is initialized."""
    return dist.is_available() and dist.is_initialized()


class _EMACodebook(nn.Module):
    """One EMA-updated codebook for a single residual stage.

    Buffers (NOT parameters — updated by EMA, not by the optimizer):
      embed         [num_codes, code_dim]  current codebook vectors
      cluster_size  [num_codes]            EMA count of VALID tokens per code
      embed_avg     [num_codes, code_dim]  EMA sum of VALID assigned tokens
      initted       []  scalar 0/1         (reserved; codebook is randn-init)

    Mask-aware: forward takes a flat [N, D] token tensor + a [N] bool valid mask.
    All padded (invalid) tokens are excluded from argmin-driven EMA stats and
    from dead-code reset sampling.
    """

    def __init__(
        self,
        num_codes: int,
        code_dim: int,
        ema_mu: float = 0.99,
        eps: float = 1e-5,
        dead_code_threshold: float = 1.0,
    ) -> None:
        super().__init__()
        if num_codes <= 0 or code_dim <= 0:
            raise ValueError(f"num_codes/code_dim must be > 0, got {num_codes}/{code_dim}")
        if not (0.0 < ema_mu < 1.0):
            raise ValueError(f"ema_mu must be in (0,1), got {ema_mu}")
        self.num_codes = num_codes
        self.code_dim = code_dim
        self.ema_mu = ema_mu
        self.eps = eps
        self.dead_code_threshold = dead_code_threshold

        embed = torch.randn(num_codes, code_dim)
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(num_codes))
        self.register_buffer("embed_avg", embed.clone())
        self.register_buffer("initted", torch.tensor(0.0))

    @torch.no_grad()
    def _laplace_normalized_embed(self) -> torch.Tensor:
        """Laplace-smoothed EMA embedding (cluster_size + eps in the denominator)."""
        n = self.cluster_size.sum()
        smoothed = (self.cluster_size + self.eps) / (n + self.num_codes * self.eps) * n
        return self.embed_avg / smoothed.unsqueeze(1).clamp(min=self.eps)

    @torch.no_grad()
    def _reset_dead_codes(self, stage_input: torch.Tensor, valid_mask: torch.Tensor,
                          allow_collectives: bool = True) -> int:
        """Reset codes whose EMA cluster_size < threshold by resampling from the
        VALID tokens of this stage (amendment 1a — never sample padded
        zero-vectors).

        stage_input : [N, D] fp32 — all tokens entering this stage (valid+padded).
        valid_mask  : [N] bool    — True = real token.

        Bug-2 fix: this is DATA-INDEPENDENT in its collective. cluster_size is
        globally synced by the preceding EMA all_reduce, so every rank agrees on
        WHICH codes are dead and HOW MANY. The replacement vectors are sampled on
        rank 0 (from rank-0's valid tokens) and BROADCAST to all ranks, so the
        broadcast fires on every rank regardless of each rank's local valid count
        — no rank can skip the collective and deadlock. (Sampling only on rank 0
        also keeps codebooks bit-identical across ranks.)

        allow_collectives=False (rank-0-only eval pass / single-GPU): no broadcast;
        sample from THIS rank's own valid tokens. Dead-code reset only runs in
        training (caller gates on self.training).
        """
        dead = self.cluster_size < self.dead_code_threshold  # [num_codes] bool (synced)
        n_dead = int(dead.sum().item())
        if n_dead == 0:
            return 0

        ddp = allow_collectives and _ddp_active()
        rank0 = (not ddp) or (dist.get_rank() == 0)
        D = stage_input.shape[1]
        new_vecs = torch.zeros(n_dead, D, dtype=stage_input.dtype, device=stage_input.device)

        # Only the sampling source (rank 0 under DDP, else this rank) populates
        # new_vecs from its valid tokens; if it has none, leave zeros and skip the
        # in-place overwrite below so a code is never replaced by a padded zero.
        has_valid_source = False
        if rank0:
            src = stage_input[valid_mask]   # [M, D] valid tokens on the source rank
            M = src.shape[0]
            if M > 0:
                idx = torch.randint(0, M, (n_dead,), device=src.device)
                new_vecs = src[idx]
                has_valid_source = True

        if ddp:
            # Always-fired collectives (every rank participates): broadcast both
            # the source-has-valid flag and the sampled vectors from rank 0.
            flag = torch.tensor([1 if has_valid_source else 0],
                                device=stage_input.device)
            dist.broadcast(flag, src=0)
            has_valid_source = bool(int(flag.item()))
            dist.broadcast(new_vecs, src=0)

        if not has_valid_source:
            # No valid tokens to sample from this step → do not overwrite any code
            # with zeros; report 0 resets.
            return 0

        self.embed[dead] = new_vecs
        self.embed_avg[dead] = new_vecs
        # Revive with cluster mass = threshold so the reset code is not counted as
        # dead by THIS step's report. NOTE: with EMA decay it falls below threshold
        # again next step unless it wins assignments — i.e. dead codes keep getting
        # fresh samples until one is actually used. That recurring churn is the
        # intended anti-collapse behaviour, not a bug.
        self.cluster_size[dead] = float(self.dead_code_threshold)
        return n_dead

    def quantize(self, x_fp32: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Nearest-code lookup in fp32. x_fp32: [N, D] -> (codes [N], q [N, D]).

        Distance = ||x - e||^2 expanded as ||x||^2 - 2 x·e + ||e||^2 (fp32).
        """
        # (x - e)^2 = x^2 - 2xe + e^2 ; drop the constant x^2 term for argmin.
        e = self.embed  # [K, D] fp32
        dist_sq = (
            x_fp32.pow(2).sum(dim=1, keepdim=True)
            - 2.0 * x_fp32 @ e.t()
            + e.pow(2).sum(dim=1).unsqueeze(0)
        )  # [N, K]
        codes = dist_sq.argmin(dim=1)  # [N]
        q = F.embedding(codes, e)      # [N, D]
        return codes, q

    @torch.no_grad()
    def ema_update(
        self,
        x_fp32: torch.Tensor,   # [N, D] all tokens (valid + padded), fp32
        codes: torch.Tensor,    # [N] argmin code ids for ALL tokens
        valid: torch.Tensor,    # [N] bool — True = real token
        allow_collectives: bool = True,
    ) -> float:
        """EMA codebook update from VALID tokens only (amendments 1a/1b + DDP sync).

        allow_collectives gates the cross-rank all_reduce (amendment 3): True in
        the all-ranks training step (statistics summed over the GLOBAL valid-token
        population); False on a rank-0-only pass so it never blocks on a collective
        the other ranks are not making.

        Returns the GLOBAL valid-token count this step. If it is zero (every token
        on every rank is padded — degenerate, but not proven impossible), the
        codebook is left UNTOUCHED: applying EMA decay with zero new mass would
        shrink cluster_size toward 0 and make _laplace_normalized_embed divide by a
        vanishing denominator → blow-up. The all_reduce still fires unconditionally
        (every rank participates) so DDP collective ordering is preserved.
        """
        onehot = F.one_hot(codes, self.num_codes).to(x_fp32.dtype)  # [N, K]
        # (1b) zero out padded tokens BEFORE accumulation so they contribute to
        # neither the per-code count nor the per-code sum.
        vmask = valid.to(x_fp32.dtype).unsqueeze(1)  # [N, 1]
        onehot = onehot * vmask                       # [N, K]
        cluster_size_batch = onehot.sum(dim=0)        # [K]
        embed_sum_batch = onehot.t() @ x_fp32         # [K, D]

        # (amendment 3) all-reduce the GLOBAL statistics across ranks so every
        # rank applies the identical update from the global valid-token set.
        # This collective fires on EVERY rank regardless of local/global validity
        # (must stay before the zero-valid early-out so ranks never diverge).
        if allow_collectives and _ddp_active():
            dist.all_reduce(cluster_size_batch, op=dist.ReduceOp.SUM)
            dist.all_reduce(embed_sum_batch, op=dist.ReduceOp.SUM)

        global_valid = float(cluster_size_batch.sum().item())
        if global_valid <= 0.0:
            # No valid tokens anywhere → no-op (do NOT decay; would corrupt embed).
            return 0.0

        mu = self.ema_mu
        self.cluster_size.mul_(mu).add_(cluster_size_batch, alpha=1 - mu)
        self.embed_avg.mul_(mu).add_(embed_sum_batch, alpha=1 - mu)
        self.embed.copy_(self._laplace_normalized_embed())
        self.initted.fill_(1.0)
        return global_valid


class MaskedResidualVQ(nn.Module):
    """Residual VQ over graph-pooled coarse-slot tokens, mask-aware end-to-end.

    forward(x [B, T_lat, C, D], mask [B, T_lat, C] bool) -> dict:
      quantized   [B, T_lat, C, D]   STE-passthrough, zeroed on padded tokens
      indices     [B, T_lat, C, Q]   long, -1 on padded tokens
      commit_loss scalar (fp32)      RAW unweighted MSE commit term (F4); the
                                     caller applies commit_weight ONCE. Summed
                                     over residual stages, averaged over valid
                                     tokens * D within THIS rank (the loss
                                     wrapper re-normalizes over GLOBAL tokens).
      commit_sq_sum scalar (fp32)    sum of per-stage squared residuals over
                                     VALID tokens only (numerator for global norm)
      n_valid_tokens scalar (fp32)   number of valid tokens this rank (denominator
                                     helper for the global commit normalization)
      perplexity  [Q] fp32           codebook perplexity per stage (valid tokens)
      active_codes[Q] long           #codes used this batch per stage (valid)
      dead_codes  [Q] long           #codes reset this step per stage

    Codebook update (EMA + dead-code reset) runs ONLY in training mode and ONLY
    when there is at least one valid token. Quantizer dropout (F3) randomly
    truncates the residual depth per FORWARD during training so later stages stay
    trainable / used (MoGenTS-style); eval always uses the full depth.
    """

    def __init__(
        self,
        code_dim: int = 512,
        num_codes: int = 512,
        num_quantizers: int = 4,
        ema_mu: float = 0.99,
        quantize_dropout_prob: float = 0.1,
        dead_code_threshold: float = 1.0,
    ) -> None:
        super().__init__()
        if num_quantizers <= 0:
            raise ValueError(f"num_quantizers must be > 0, got {num_quantizers}")
        if not (0.0 <= quantize_dropout_prob < 1.0):
            raise ValueError(
                f"quantize_dropout_prob must be in [0,1), got {quantize_dropout_prob}")
        self.code_dim = code_dim
        self.num_codes = num_codes
        self.num_quantizers = num_quantizers
        self.quantize_dropout_prob = quantize_dropout_prob
        self.codebooks = nn.ModuleList([
            _EMACodebook(num_codes, code_dim, ema_mu=ema_mu,
                        dead_code_threshold=dead_code_threshold)
            for _ in range(num_quantizers)
        ])

    def forward(self, x: torch.Tensor, mask: torch.Tensor,
                allow_collectives: bool = True) -> dict:
        """allow_collectives gates EVERY cross-rank collective (EMA all_reduce,
        dead-code broadcast, quantizer-dropout depth broadcast, perplexity
        all_reduce). Pass True on the all-ranks training step (so the SMOKE
        exercises amendment 3's DDP EMA sync); pass False on a rank-0-only eval
        pass so the quantizer never blocks on a collective the other ranks are not
        making (the deadlock that a naive `if _ddp_active()` would cause)."""
        if x.dim() != 4 or x.shape[-1] != self.code_dim:
            raise ValueError(
                f"MaskedResidualVQ: x must be [B,T_lat,C,{self.code_dim}], "
                f"got {tuple(x.shape)}")
        B, T_lat, C, D = x.shape
        if mask.shape != (B, T_lat, C) or mask.dtype != torch.bool:
            raise ValueError(
                f"MaskedResidualVQ: mask must be [B,T_lat,C]={(B, T_lat, C)} bool, "
                f"got {tuple(mask.shape)} dtype {mask.dtype}")

        orig_dtype = x.dtype
        # Flatten to [N, D]; do all VQ math in fp32 (bf16-safe).
        x_flat_fp32 = x.reshape(-1, D).float()           # [N, D]
        valid_flat = mask.reshape(-1)                    # [N] bool
        N = x_flat_fp32.shape[0]
        device = x_flat_fp32.device
        n_valid = int(valid_flat.sum().item())

        residual = x_flat_fp32                           # running residual, fp32
        q_total = torch.zeros_like(x_flat_fp32)          # accumulated quantized, fp32
        indices_stages: list[torch.Tensor] = []
        perplexities: list[torch.Tensor] = []
        active_counts: list[torch.Tensor] = []
        dead_counts: list[torch.Tensor] = []
        commit_sq_sum = x_flat_fp32.new_zeros(())        # numerator for global norm

        # Quantizer dropout (F3): in training, WITH PROBABILITY quantize_dropout_prob
        # truncate the residual depth to a random level (so later stages still
        # receive gradient/usage, MoGenTS-style); otherwise use full depth. Eval =
        # always full depth. We always RUN all stages structurally (so DDP sees an
        # identical graph + all codebook buffers participate), but stages beyond
        # `keep_depth` are detached from q_total/commit and emit -1 indices
        # (treated like "no residual code").
        keep_depth = self.num_quantizers
        if self.training and self.quantize_dropout_prob > 0.0 and self.num_quantizers > 1:
            # Bernoulli(p): with probability quantize_dropout_prob this forward
            # TRUNCATES the residual depth (so later stages still receive
            # gradient/usage, MoGenTS-style); otherwise full depth. A dropout draw
            # must ALWAYS truncate, so the kept depth is sampled from [1, Q-1]
            # (NOT [1, Q] — drawing Q would mean "no truncation" and make the
            # effective truncation rate p*(Q-1)/Q instead of p). Draw on rank 0 and
            # broadcast BOTH the do_dropout decision and the depth so every rank
            # picks the same keep_depth (identical DDP autograd graph).
            do_dropout = bool(torch.rand((), device=device).item() < self.quantize_dropout_prob)
            rand_depth = int(torch.randint(1, self.num_quantizers, (1,)).item())  # [1, Q-1]
            if allow_collectives and _ddp_active():
                payload = torch.tensor([1 if do_dropout else 0, rand_depth], device=device)
                dist.broadcast(payload, src=0)
                do_dropout = bool(int(payload[0].item()))
                rand_depth = int(payload[1].item())
            if do_dropout:
                keep_depth = rand_depth

        for qi, cb in enumerate(self.codebooks):
            codes, q = cb.quantize(residual)             # [N], [N, D] fp32
            active = qi < keep_depth
            # Per-stage indices: -1 on padded tokens always; -1 on dropped stages.
            stage_idx = torch.full((N,), -1, dtype=torch.long, device=device)
            if active:
                stage_idx[valid_flat] = codes[valid_flat]
            indices_stages.append(stage_idx.reshape(B, T_lat, C))

            if active:
                # Commit term: pull encoder output toward the chosen code. Only
                # valid tokens count (padded residuals are meaningless). Detach
                # the code (codebook is EMA-updated, not gradient-updated).
                sq = (residual - q.detach()).pow(2).sum(dim=1)   # [N] per-token
                commit_sq_sum = commit_sq_sum + (sq * valid_flat.to(sq.dtype)).sum()
                # `stage_input` is the residual the codes were assigned from —
                # the EMA / dead-code-reset target for this stage (standard RVQ:
                # each codebook clusters the residual entering it). Capture it
                # BEFORE subtracting q below.
                stage_input = residual
                # Accumulate the quantized residual (fp32). Padded tokens are
                # zeroed at the very end via the valid mask, so we leave them in
                # the running residual here (their codes are discarded anyway).
                q_total = q_total + q
                residual = residual - q

                # ---- codebook bookkeeping / EMA update (train only) ----
                # Bug-2 fix: collective participation must NOT depend on the local
                # valid-token count. EVERY rank runs the SAME collectives for an
                # active stage during training, so a rank that happens to have zero
                # local valid tokens cannot diverge from peers and deadlock. A
                # zero-valid rank simply contributes zero stats (its one-hot is
                # zeroed by the valid mask inside ema_update; its bincount is empty),
                # which is the correct global-sum behaviour.
                if self.training:
                    # ema_update returns the GLOBAL valid count and no-ops (no decay,
                    # no embed corruption) when it is 0. _reset_dead_codes is already
                    # zero-valid-safe: when rank 0 has no valid source it fires the
                    # broadcast (collective ordering preserved) but overwrites no code
                    # and reports 0 resets. So both are safe even on a degenerate
                    # all-padded global batch.
                    cb.ema_update(x_fp32=stage_input, codes=codes, valid=valid_flat,
                                  allow_collectives=allow_collectives)
                    n_dead = cb._reset_dead_codes(
                        stage_input, valid_flat,
                        allow_collectives=allow_collectives)
                else:
                    n_dead = 0

                # ---- diagnostics ----
                with torch.no_grad():
                    vcodes = codes[valid_flat]   # may be empty on a zero-valid rank
                    counts = torch.bincount(vcodes, minlength=self.num_codes).float()
                    if allow_collectives and _ddp_active():
                        # Same collective on every rank (empty-rank contributes 0s).
                        dist.all_reduce(counts, op=dist.ReduceOp.SUM)
                    total_count = counts.sum().clamp(min=1.0)
                    probs = counts / total_count
                    nz = probs > 0
                    entropy = -(probs[nz] * probs[nz].log()).sum() if nz.any() else torch.zeros((), device=device)
                    perplexities.append(entropy.exp())
                    active_counts.append((counts > 0).sum().long())
                    dead_counts.append(torch.tensor(n_dead, dtype=torch.long, device=device))
            else:
                # Dropped stage: no commit, no accumulation, no codebook update.
                perplexities.append(torch.zeros((), device=device))
                active_counts.append(torch.zeros((), dtype=torch.long, device=device))
                dead_counts.append(torch.zeros((), dtype=torch.long, device=device))

        # (1c) Straight-through estimator, masked AFTER detach. q_total is the
        # quantized fp32 sum; gradients flow to x through the (x - x.detach())
        # identity, then the valid mask zeroes BOTH the value and the grad path
        # of padded tokens.
        x_q_flat = x_flat_fp32 + (q_total - x_flat_fp32).detach()  # [N, D] fp32
        x_q_flat = x_q_flat * valid_flat.to(x_q_flat.dtype).unsqueeze(1)
        quantized = x_q_flat.reshape(B, T_lat, C, D).to(orig_dtype)

        # RAW (unweighted) commit loss for THIS rank: mean over valid tokens * D.
        # F4 — commit_weight applied ONCE in the loss wrapper, never here.
        denom = max(n_valid, 1) * D
        commit_loss = commit_sq_sum / denom

        indices = torch.stack(indices_stages, dim=-1)   # [B, T_lat, C, Q]
        return {
            "quantized": quantized,
            "indices": indices,
            "commit_loss": commit_loss,
            "commit_sq_sum": commit_sq_sum,                       # for global norm
            "n_valid_tokens": torch.tensor(float(n_valid), device=device),
            "perplexity": torch.stack(perplexities),             # [Q]
            "active_codes": torch.stack(active_counts),          # [Q]
            "dead_codes": torch.stack(dead_counts),              # [Q]
        }
