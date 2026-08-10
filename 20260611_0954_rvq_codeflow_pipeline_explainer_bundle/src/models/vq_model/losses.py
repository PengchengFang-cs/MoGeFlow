"""No-KL VQ loss wrapper for the Graph-VQVAE tokenizer (amendment 2).

The shared src/models/graph_salad/losses.py::compute_total_loss_13ch ALWAYS
computes a KL term (losses.py:567 `masked_kl_gaussian(mu, logvar, ...)`) and
sums it into `total`. The VQ tokenizer has NO Gaussian latent (VQ replaces the
KL regularization), so we MUST NOT call compute_total_loss_13ch directly — there
is no mu/logvar, and a KL term would be meaningless. Instead we reuse the SAME
underlying reconstruction terms by IMPORT (so numerics match the VAE's anytop13
recon) and assemble a no-KL total here:

    L_total = w_pos*pos + w_rot*rot + w_vel*vel + w_contact*contact
            + w_world*world + w_fk*fk + w_traj*traj
            + w_commit*commit            <-- commit_weight applied ONCE (F4)

Reused terms (identical to compute_total_loss_13ch minus KL / pool_aux):
  pos/rot/vel : _masked_group_l1 over channel slices 0:3 / 3:9 / 9:12
  contact     : masked_contact_bce on channel 12
  world/fk/traj + gt_fk_mismatch : compute_world_rot6d_fk_terms (geometry, fp32)

F4: the quantizer returns a RAW unweighted commit loss; commit_weight (0.02) is
applied exactly once HERE — never squared.

DDP (amendment 3): the commit loss is normalized over GLOBAL valid tokens across
ranks. The quantizer hands back per-rank `commit_sq_sum` (sum of squared
residuals over valid tokens, gradient-carrying) and `n_valid_tokens` (a detached
count). This wrapper all-reduces ONLY the count to get global_n_valid, then forms
the per-rank term  commit_r = W * commit_sq_sum_r / (global_n_valid * D). The
numerator's GLOBAL sum is realized by DDP itself: DDP averages each parameter's
gradient across ranks (÷ W) on backward, so pre-multiplying by world_size W makes
the post-averaging gradient equal grad( Σ_r commit_sq_sum_r / (global_n*D) ) —
i.e. identical to single-rank training on the concatenated batch, regardless of
how tokens are sharded. (The numerator is NOT all-reduced as a value — doing so
would double-count once DDP also averages its gradient.)

Geometry terms (world/fk/traj) run in fp32 internally (the recovery functions
operate on de-normalized fp32 motion). Recon L1/BCE are computed on the decoder
output dtype; under bf16 autocast the loss is assembled and `.backward()` is
called in fp32 (callers leave autocast before loss, mirroring train_graph_vae.py).
"""

from __future__ import annotations

import torch
import torch.distributed as dist

from src.models.graph_salad.losses import (
    _masked_group_l1,
    masked_contact_bce,
    compute_world_rot6d_fk_terms,
)


def _ddp_active() -> bool:
    return dist.is_available() and dist.is_initialized()


_DEFAULT_VQ_WEIGHTS = {
    "pos": 1.0, "rot": 1.0, "vel": 1.0, "contact": 0.1,
    "world": 0.25, "fk": 1.0, "traj": 0.10, "commit": 0.02,
}


def compute_vq_loss_13ch(
    *,
    pred_motion: torch.Tensor,             # [B, T, J, 13] normalized decoder output
    gt_motion: torch.Tensor,               # [B, T, J, 13] normalized anytop_x
    foot_contact_per_joint: torch.Tensor,  # [B, T, J] raw 0/1 (BCE target)
    anytop_mean: torch.Tensor,             # [B, J, 13] raw de-norm mean
    anytop_std: torch.Tensor,              # [B, J, 13] raw de-norm std
    parent_indices,                        # list[list[int]] length B
    rest_offsets: torch.Tensor,            # [B, J, 3]
    joint_mask: torch.Tensor,              # [B, J] bool
    frame_mask: torch.Tensor,              # [B, T] bool (use frame_mask_recovered)
    code_dim: int,                         # D — commit normalization denominator
    commit_sq_sum: torch.Tensor,           # scalar fp32 — per-rank Σ valid commit sq
    n_valid_tokens: torch.Tensor,          # scalar fp32 — per-rank #valid tokens
    weights: dict[str, float] | None = None,
    allow_collectives: bool = True,        # gate the commit-norm all_reduce (DDP)
) -> dict[str, torch.Tensor]:
    """No-KL VQ recon loss. Returns named scalar losses + `total`.

    Mirrors the loss-weight names/semantics of the plan; KL and pool_aux are
    intentionally ABSENT (VQ has no Gaussian latent and EdgeSegmentPool has zero
    aux losses).
    """
    if pred_motion.shape != gt_motion.shape:
        raise ValueError(
            f"compute_vq_loss_13ch: pred_motion {tuple(pred_motion.shape)} != "
            f"gt_motion {tuple(gt_motion.shape)}")
    if pred_motion.dim() != 4 or pred_motion.shape[-1] != 13:
        raise ValueError(
            f"compute_vq_loss_13ch: pred_motion must be [B,T,J,13], "
            f"got {tuple(pred_motion.shape)}")

    if weights is None:
        weights = dict(_DEFAULT_VQ_WEIGHTS)
    else:
        merged = dict(_DEFAULT_VQ_WEIGHTS)
        merged.update(weights)
        weights = merged

    losses: dict[str, torch.Tensor] = {}
    # ---- reconstruction (same terms as compute_total_loss_13ch, NO KL) ----
    losses["pos"] = _masked_group_l1(pred_motion, gt_motion, slice(0, 3),
                                     joint_mask, frame_mask)
    losses["rot"] = _masked_group_l1(pred_motion, gt_motion, slice(3, 9),
                                     joint_mask, frame_mask)
    losses["vel"] = _masked_group_l1(pred_motion, gt_motion, slice(9, 12),
                                     joint_mask, frame_mask)
    losses["contact"] = masked_contact_bce(
        pred_motion[..., 12], foot_contact_per_joint, joint_mask, frame_mask)

    # ---- world / FK / traj geometry (fp32 inside; gt_fk_mismatch diagnostic) ----
    terms = compute_world_rot6d_fk_terms(
        pred_motion=pred_motion, gt_motion=gt_motion,
        anytop_mean=anytop_mean, anytop_std=anytop_std,
        parent_indices=parent_indices, rest_offsets=rest_offsets,
        joint_mask=joint_mask, frame_mask=frame_mask,
    )
    losses["world"] = terms["world"]
    losses["fk"] = terms["fk"]
    losses["traj"] = terms["traj"]
    losses["gt_fk_mismatch"] = terms["gt_fk_mismatch"]  # diagnostic only

    # ---- commit (F4: weight applied ONCE; amendment 3: GLOBAL normalization) ----
    # commit_sq_sum is the per-rank, gradient-carrying Σ over THIS rank's valid
    # tokens of ||residual - code.detach()||². The intended objective is the mean
    # over the GLOBAL valid-token population:
    #     commit = (Σ_ranks local_sq) / (Σ_ranks local_n * D).
    # n_valid_tokens is detached (a count, not a gradient source); we all-reduce
    # it to get the global denominator. The numerator's GLOBAL sum is realized by
    # DDP itself: DDP averages each parameter's gradient across ranks (÷ W) when
    # .backward() runs, so we pre-multiply the local term by world_size W. After
    # DDP's averaging the effective gradient equals (Σ_ranks local_sq)/(global_n*D)
    # — identical to single-rank training on the concatenated batch, i.e. the
    # commit gradient is invariant to how tokens are sharded across ranks.
    # allow_collectives=False on a rank-0-only eval pass: skip the all_reduce
    # (and the world_size scaling) so the loss never blocks on a collective the
    # other ranks are not making, and the reported val commit is the rank-0-local
    # mean (a diagnostic, not a gradient — eval has no backward).
    global_n = n_valid_tokens.detach().clone()
    world_size = 1
    if allow_collectives and _ddp_active():
        dist.all_reduce(global_n, op=dist.ReduceOp.SUM)
        world_size = dist.get_world_size()
    denom = (global_n.clamp(min=1.0) * float(code_dim))
    losses["commit"] = (commit_sq_sum * float(world_size)) / denom

    # ---- total (NO kl, NO pool_aux) ----
    total = torch.zeros((), device=pred_motion.device, dtype=losses["pos"].dtype)
    for key, w in weights.items():
        if key not in losses or w == 0.0:
            continue
        total = total + losses[key] * w
    losses["total"] = total
    return losses
