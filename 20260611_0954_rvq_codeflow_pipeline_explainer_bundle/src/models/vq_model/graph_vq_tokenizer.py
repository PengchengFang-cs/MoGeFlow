"""GraphVQTokenizer — coarse-slot structured Residual-VQ tokenizer for AnyTop
arbitrary-topology motion (Graph-VQVAE v1.0).

Pipeline (handoff/20260608_graph_vqvae_l5_pipeline_plan.md):

  anytop13 [B,J,13,T] (from AnyTopDataset)
    -> permute [B,T,J,13]
    -> SkeletonEncoder(anytop13_split, graphormer)        [reused by IMPORT]
    -> SlotNorm                                           [reused by IMPORT]
    -> EdgeSegmentPool(edge_segment, max_coarse=C)        [reused by IMPORT]
       => pooled_features [B,T_lat,C,D] + assignment P [B,J,C] + pooled graph meta
    -> (optional) n_pre_vq coarse graph+temporal refine layers
    -> MaskedResidualVQ (Q stages, mask-aware, padded slots excluded)
       => z_q [B,T_lat,C,D], indices [B,T_lat,C,Q] (-1 on padded)
    -> (optional) n_post_vq coarse graph+temporal refine layers
    -> repeat_interleave(temporal_stride) -> [B,T,C,D]
    -> MaskedMotionDecoder (F1: strict padded-slot-key mask)  [vq_model fork]
    -> anytop13 root/non-root output heads
    -> pred_motion [B,T,J,13]

This module OWNS its own decoder + heads (under src/models/vq_model/) and reuses
the encoder / pool / graph-attention by IMPORT — it does NOT subclass or call
GraphMotionVAE, so the VQ checkpoint + training loop are fully independent of the
Gaussian Graph-VAE (which keeps running unchanged).

bf16-safe: the whole forward runs under the caller's autocast; the quantizer does
its codebook distance/argmin/EMA in fp32 internally and the graph attention forces
fp32 softmax. The pool requires fp32 inputs (joint_features / adjacency /
geodesic), so the encoder output + graph tensors are cast to fp32 before the pool
call (mirrors how the VAE feeds the pool — the pool is fp32-only by contract).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.encoder import SkeletonEncoder
from src.models.slot_norm import SlotNorm
from src.models.graph_salad.pool_edge_segment import EdgeSegmentPool
from src.models.graph_salad.attention import GraphAttentionBlock
from src.models.motion_decoder import TemporalSelfAttention

from .quantizer import MaskedResidualVQ
from .masked_motion_decoder import MaskedMotionDecoder


class CoarseGraphTemporalLayer(nn.Module):
    """One coarse-latent refine layer: graph self-attention over coarse slots
    (per latent frame) + temporal self-attention over latent frames (per slot).

    Uses GraphAttentionBlock (scalar adj+geo bias — pooled_adjacency is binary
    {0,1}, pooled_geodesic is its Floyd distance, exactly the block's contract)
    for the spatial sub-block, and TemporalSelfAttention for the temporal one
    (both bf16-safe / fp32-softmax). Each sub-block re-masks padded slots/frames
    so the layer output is clean by construction (mirrors the shared
    GraphTemporalDecoderLayer discipline, but at the coarse level with the pool's
    adjacency/geodesic rather than graphormer edge types).
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.spatial = GraphAttentionBlock(d_model, n_heads, d_ff, dropout)
        self.temporal = TemporalSelfAttention(d_model, n_heads, dropout)

    def forward(
        self,
        x: torch.Tensor,                 # [B, T_lat, C, D]
        pooled_adjacency: torch.Tensor,  # [B, C, C] fp32 binary
        pooled_geodesic: torch.Tensor,   # [B, C, C] fp32 Floyd dist
        coarse_mask: torch.Tensor,       # [B, C] bool
        frame_mask_lat: torch.Tensor,    # [B, T_lat] bool
    ) -> torch.Tensor:
        B, T_lat, C, D = x.shape
        cm = coarse_mask[:, None, :, None].to(x.dtype)   # [B,1,C,1]
        fm = frame_mask_lat[:, :, None, None].to(x.dtype)  # [B,T_lat,1,1]

        # --- spatial: graph attention over coarse slots, per latent frame ---
        xs = x.reshape(B * T_lat, C, D)
        adj_e = pooled_adjacency.unsqueeze(1).expand(B, T_lat, C, C).reshape(B * T_lat, C, C)
        geo_e = pooled_geodesic.unsqueeze(1).expand(B, T_lat, C, C).reshape(B * T_lat, C, C)
        cm_e = coarse_mask.unsqueeze(1).expand(B, T_lat, C).reshape(B * T_lat, C)
        # GraphAttentionBlock requires fp32 adjacency/geodesic; node_mask bool.
        xs = self.spatial(xs, adj_e.float(), geo_e.float(), cm_e, validate_inputs=False)
        x = xs.reshape(B, T_lat, C, D) * cm   # re-mask padded slots

        # --- temporal: self-attention over latent frames, per coarse slot ---
        xt = x.permute(0, 2, 1, 3).reshape(B * C, T_lat, D)
        fm_e = frame_mask_lat.unsqueeze(1).expand(B, C, T_lat).reshape(B * C, T_lat)
        # A fully-padded slot has an all-False frame row → TemporalSelfAttention
        # nan_to_num(0.0) keeps it zero; re-mask below removes any residue.
        xt = self.temporal(xt, fm_e)
        x = xt.reshape(B, C, T_lat, D).permute(0, 2, 1, 3)
        return x * cm * fm                    # re-mask padded slots + frames


class GraphVQTokenizer(nn.Module):
    """Graph-aware coarse-slot RVQ tokenizer. See module docstring for the pipeline."""

    def __init__(
        self,
        d_model: int = 512,
        n_heads: int = 8,
        d_ff: int = 1536,
        n_graph_layers: int = 4,
        n_enc_temporal_layers: int = 2,
        n_pre_vq_layers: int = 2,
        n_post_vq_layers: int = 2,
        n_cross_layers: int = 3,
        n_dec_temporal_layers: int = 2,
        max_coarse: int = 50,
        temporal_stride: int = 4,
        temporal_kernel: int = 9,
        dropout: float = 0.1,
        joint_feat_dim: int = 9,
        # quantizer
        code_dim: int = 512,
        num_codes: int = 512,
        num_quantizers: int = 4,
        ema_mu: float = 0.99,
        quantize_dropout_prob: float = 0.1,
        dead_code_threshold: float = 1.0,
    ) -> None:
        super().__init__()
        if code_dim != d_model:
            raise ValueError(
                f"GraphVQTokenizer: code_dim ({code_dim}) must equal d_model "
                f"({d_model}) — the quantizer operates on the pooled D-dim slots.")
        self.d_model = d_model
        self.temporal_stride = temporal_stride
        self.max_coarse = max_coarse
        self.num_quantizers = num_quantizers

        # ---- Encoder (anytop13_split + graphormer), reused by import ----
        self.encoder = SkeletonEncoder(
            d_model=d_model, n_heads=n_heads, d_ff=d_ff,
            n_graph_layers=n_graph_layers,
            n_temporal_layers=n_enc_temporal_layers,
            joint_feat_dim=joint_feat_dim,
            motion_feat_dim=13,
            temporal_kernel=temporal_kernel,
            dropout=dropout,
            motion_mode="anytop13_split",
            attn_mode="graphormer",
        )
        self.slot_norm = SlotNorm(d_model)

        # ---- EdgeSegmentPool (edge_segment), reused by import ----
        self.pool = EdgeSegmentPool(
            d_model=d_model, max_coarse=max_coarse,
            temporal_stride=temporal_stride,
        )

        # ---- Optional coarse graph+temporal refine layers (pre/post VQ) ----
        self.pre_vq_layers = nn.ModuleList([
            CoarseGraphTemporalLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_pre_vq_layers)
        ])
        self.post_vq_layers = nn.ModuleList([
            CoarseGraphTemporalLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_post_vq_layers)
        ])

        # ---- Mask-aware Residual VQ ----
        self.quantizer = MaskedResidualVQ(
            code_dim=code_dim, num_codes=num_codes,
            num_quantizers=num_quantizers, ema_mu=ema_mu,
            quantize_dropout_prob=quantize_dropout_prob,
            dead_code_threshold=dead_code_threshold,
        )

        # ---- Decoder (F1 masked fork) + anytop13 heads, OWNED by this module ----
        self.decoder = MaskedMotionDecoder(
            d_model=d_model, n_heads=n_heads,
            n_cross_layers=n_cross_layers,
            n_temporal_layers=n_dec_temporal_layers,
            temporal_kernel=temporal_kernel,
            dropout=dropout,
        )
        self.anytop13_head = nn.ModuleDict({
            "out_root": nn.Linear(d_model, 13),
            "out_nonroot": nn.Linear(d_model, 13),
        })

    def encode(self, batch) -> dict:
        """anytop13 batch -> pooled coarse slots + pool graph metadata.

        Returns the pre-VQ pooled features (graph-refined if n_pre_vq>0) plus all
        masks / assignment / pooled graph tensors the decoder + VQ need.
        """
        if batch.anytop_x is None:
            raise ValueError("GraphVQTokenizer requires batch.anytop_x "
                             "(use --dataset anytop_truebones)")
        if batch.anytop_graph_dist is None or batch.anytop_joint_relations is None:
            raise ValueError("GraphVQTokenizer(graphormer) requires "
                             "batch.anytop_graph_dist + batch.anytop_joint_relations")

        motion_in = batch.anytop_x.permute(0, 3, 1, 2).contiguous()  # [B,T,J,13]
        gd, jr = batch.anytop_graph_dist, batch.anytop_joint_relations

        h0 = self.encoder(
            motion_in, batch.skeleton_features,
            batch.adjacency, batch.geodesic_dist,
            batch.joint_mask, batch.frame_mask,
            name_hashes=batch.name_hashes,
            graph_dist=gd, joint_relations=jr,
        )  # [B,T,J,D]
        s_j = self.encoder.encode_skeleton(
            batch.skeleton_features, batch.adjacency, batch.geodesic_dist,
            batch.joint_mask, name_hashes=batch.name_hashes,
            graph_dist=gd, joint_relations=jr,
        )  # [B,J,D]
        h0 = self.slot_norm(h0)

        # Pool is fp32-only by contract: cast features + graph tensors to fp32.
        pool_out = self.pool(
            joint_features=h0.float(),
            skeleton_embeddings=s_j.float(),
            adjacency=batch.adjacency.float(),
            geodesic_dist=batch.geodesic_dist.float(),
            joint_mask=batch.joint_mask,
            frame_mask=batch.frame_mask,
            parent_indices=batch.parent_indices,
        )
        h_lat = pool_out["pooled_features"]            # [B,T_lat,C,D] fp32
        coarse_mask = pool_out["pooled_mask"]          # [B,C] bool
        frame_mask_lat = pool_out["frame_mask_down"]   # [B,T_lat] bool
        assignment = pool_out["assignment"]            # [B,J,C] fp32
        pooled_adjacency = pool_out["pooled_adjacency"]   # [B,C,C] fp32
        pooled_geodesic = pool_out["pooled_geodesic"]     # [B,C,C] fp32
        # gap #1 (Graph-CodeFlow): the pool computes per-segment skeleton
        # embeddings but encode() dropped them. Expose ADDITIVELY (extra dict key,
        # no existing key/value altered) for the CodeFlow backbone's skeleton
        # conditioning + offline export. fp32, [B,C,D]; zero on padded slots.
        pooled_skeleton_embeddings = pool_out["pooled_skeleton_embeddings"]  # [B,C,D] fp32

        # Bring back to the autocast dtype for the refine layers (graph attention
        # casts adjacency/geodesic to fp32 internally; features can be bf16).
        h_lat = h_lat.to(h0.dtype)
        for layer in self.pre_vq_layers:
            h_lat = layer(h_lat, pooled_adjacency, pooled_geodesic,
                          coarse_mask, frame_mask_lat)
        # latent-slot validity = valid coarse slot AND valid latent frame.
        token_mask = coarse_mask.unsqueeze(1) & frame_mask_lat.unsqueeze(-1)  # [B,T_lat,C]

        return {
            "h_lat": h_lat,
            "s_j": s_j,
            "assignment": assignment,
            "coarse_mask": coarse_mask,
            "frame_mask_lat": frame_mask_lat,
            "token_mask": token_mask,
            "pooled_adjacency": pooled_adjacency,
            "pooled_geodesic": pooled_geodesic,
            "pooled_skeleton_embeddings": pooled_skeleton_embeddings,
        }

    def decode(self, z_q, enc, batch) -> dict:
        """Quantized coarse slots z_q [B,T_lat,C,D] -> pred_motion [B,T,J,13]."""
        coarse_mask = enc["coarse_mask"]
        frame_mask_lat = enc["frame_mask_lat"]

        # Optional post-VQ refine (on the quantized slots).
        z = z_q
        for layer in self.post_vq_layers:
            z = layer(z, enc["pooled_adjacency"], enc["pooled_geodesic"],
                      coarse_mask, frame_mask_lat)

        # Temporal upsample back to full resolution.
        z_up = z.repeat_interleave(self.temporal_stride, dim=1)   # [B,T,C,D]
        frame_mask_recovered = frame_mask_lat.repeat_interleave(
            self.temporal_stride, dim=-1)                          # [B,T]

        feats = self.decoder(
            z_up,                       # slot_features [B,T,C,D]
            enc["s_j"],                 # skeleton embeddings [B,J,D]
            enc["assignment"],          # real pool assignment P [B,J,C]
            coarse_mask,                # [B,C] bool — STRICT padded-slot-key mask (F1)
            batch.joint_mask,
            frame_mask_recovered,
        )  # [B,T,J,D]

        fm_b = frame_mask_recovered[:, :, None, None].to(feats.dtype)
        jm_b = batch.joint_mask[:, None, :, None].to(feats.dtype)
        out_root = self.anytop13_head["out_root"](feats[:, :, 0:1, :])      # [B,T,1,13]
        out_nonroot = self.anytop13_head["out_nonroot"](feats[:, :, 1:, :])  # [B,T,J-1,13]
        pred_motion = torch.cat([out_root, out_nonroot], dim=2)            # [B,T,J,13]
        pred_motion = pred_motion * fm_b * jm_b
        return {
            "pred_motion": pred_motion,
            "frame_mask_recovered": frame_mask_recovered,
        }

    # ------------------------------------------------------------------ #
    # Graph-CodeFlow frozen-tokenizer utilities (ADDITIVE).               #
    #                                                                     #
    # These are READ-ONLY helpers for the post-RVQ Graph-CodeFlow         #
    # backbone (handoff/20260609_graph_codeflow_rvq_backbone_plan.md §10).#
    # They DO NOT touch encode/decode/quantizer forward behavior, never   #
    # run EMA / dead-code reset / quantizer-dropout, and do all           #
    # codebook-distance / argmin math in fp32 (bf16-safe). Padding        #
    # contract identical to the quantizer: token_mask=False -> indices=-1,#
    # z_q=0.                                                              #
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def ids_to_embeddings(self, indices: torch.Tensor,
                          token_mask: torch.Tensor) -> torch.Tensor:
        """RVQ indices [B,T_lat,C,Q] -> summed code embeddings z_q [B,T_lat,C,D].

        Mirrors the quantizer's accumulation `q_total = sum_q embedding(codes_q,
        codebook_q.embed)` (quantizer.py:365) and the final STE masking
        (quantizer.py:416): a token's z_q is the SUM over the Q residual stages
        of codebooks[q].embed[indices[...,q]], and padded tokens (token_mask
        False, encoded as indices=-1) are exactly 0.

        fp32 throughout (codebook embeds are fp32 buffers); returns fp32.
        Per-stage: an index of -1 (padded token OR a dropped stage, though
        export uses full depth) contributes 0 for that stage.
        """
        if indices.dim() != 4 or indices.shape[-1] != self.num_quantizers:
            raise ValueError(
                f"ids_to_embeddings: indices must be [B,T_lat,C,Q={self.num_quantizers}], "
                f"got {tuple(indices.shape)}")
        B, T_lat, C, Q = indices.shape
        if token_mask.shape != (B, T_lat, C) or token_mask.dtype != torch.bool:
            raise ValueError(
                f"ids_to_embeddings: token_mask must be [B,T_lat,C]={(B, T_lat, C)} bool, "
                f"got {tuple(token_mask.shape)} dtype {token_mask.dtype}")
        device = indices.device
        D = self.d_model
        z_q = torch.zeros(B, T_lat, C, D, dtype=torch.float32, device=device)
        for qi, cb in enumerate(self.quantizer.codebooks):
            idx_q = indices[..., qi]                       # [B,T_lat,C] long, -1 = none
            valid_q = idx_q >= 0                           # per-stage validity
            # Gather embed for valid entries; clamp(min=0) keeps the gather index
            # in-range for padded -1 slots (their contribution is masked to 0).
            gathered = F.embedding(idx_q.clamp(min=0), cb.embed.float())  # [B,T_lat,C,D]
            z_q = z_q + gathered * valid_q.unsqueeze(-1).to(z_q.dtype)
        # Final STE-style mask: padded tokens are exactly 0 (defensive — per-stage
        # -1 masking already zeroes a fully-padded token, but token_mask is the
        # authoritative validity used by the quantizer's z_q masking).
        z_q = z_q * token_mask.unsqueeze(-1).to(z_q.dtype)
        return z_q

    @torch.no_grad()
    def nearest_residual_ids(self, z_hat: torch.Tensor,
                             token_mask: torch.Tensor) -> dict:
        """Residual-nearest RVQ projection of a continuous latent z_hat
        [B,T_lat,C,D] back onto the frozen codebooks.

        Mirrors the quantizer's residual loop (quantizer.py:342-367) EXACTLY,
        stage by stage, but with NO EMA update / NO dead-code reset / NO
        quantizer dropout (this is inference-only projection):

            r = z_hat
            for q in range(Q):
                idx_q = argmin_k || r - codebook_q[k] ||^2   (cb.quantize)
                e_q   = codebook_q[idx_q]
                r     = r - e_q
            z_snap = sum_q e_q           (== ids_to_embeddings(indices_hat))

        Returns dict:
          indices_hat       [B,T_lat,C,Q] long  (-1 on padded tokens)
          z_snap            [B,T_lat,C,D] fp32   (0 on padded tokens)
          projection_error  scalar fp32  = mean over VALID tokens * D of
                                           (z_hat - z_snap)^2

        All math in fp32 (bf16-safe). z_hat is cast to fp32 internally.
        """
        if z_hat.dim() != 4 or z_hat.shape[-1] != self.d_model:
            raise ValueError(
                f"nearest_residual_ids: z_hat must be [B,T_lat,C,D={self.d_model}], "
                f"got {tuple(z_hat.shape)}")
        B, T_lat, C, D = z_hat.shape
        if token_mask.shape != (B, T_lat, C) or token_mask.dtype != torch.bool:
            raise ValueError(
                f"nearest_residual_ids: token_mask must be [B,T_lat,C]={(B, T_lat, C)} "
                f"bool, got {tuple(token_mask.shape)} dtype {token_mask.dtype}")
        device = z_hat.device
        Q = self.num_quantizers
        z_flat = z_hat.reshape(-1, D).float()              # [N,D] fp32
        valid_flat = token_mask.reshape(-1)                # [N] bool
        N = z_flat.shape[0]

        residual = z_flat
        z_snap_flat = torch.zeros_like(z_flat)             # accumulated quantized
        indices_stages: list[torch.Tensor] = []
        for cb in self.quantizer.codebooks:
            codes, q = cb.quantize(residual)               # [N], [N,D] (fp32 argmin)
            # Per-stage indices: -1 on padded tokens (mirrors quantizer.py:346-348).
            stage_idx = torch.full((N,), -1, dtype=torch.long, device=device)
            stage_idx[valid_flat] = codes[valid_flat]
            indices_stages.append(stage_idx.reshape(B, T_lat, C))
            # Accumulate / recurse on the SAME math as the quantizer (it leaves
            # padded residuals in place; we zero padded z_snap at the end).
            z_snap_flat = z_snap_flat + q
            residual = residual - q
        # Final masking: padded tokens -> z_snap exactly 0 (STE convention).
        z_snap_flat = z_snap_flat * valid_flat.unsqueeze(-1).to(z_snap_flat.dtype)

        indices_hat = torch.stack(indices_stages, dim=-1)  # [B,T_lat,C,Q]
        z_snap = z_snap_flat.reshape(B, T_lat, C, D)
        # projection_error = masked MSE over valid tokens * D.
        n_valid = int(valid_flat.sum().item())
        diff_sq = (z_flat - z_snap_flat).pow(2)            # [N,D] (padded rows now 0-0=0)
        proj_err = diff_sq[valid_flat].sum() / (max(n_valid, 1) * D)
        return {
            "indices_hat": indices_hat,
            "z_snap": z_snap,
            "projection_error": proj_err,
        }

    @torch.no_grad()
    def prepare_skeleton_only(self, batch, T_lat: int) -> dict:
        """Motion-INDEPENDENT coarse-graph metadata for CodeFlow inference.

        Computes everything the decoder + CodeFlow conditioning need WITHOUT
        consuming motion frames (motion is the unknown being generated). Mirrors
        GraphMotionVAE.encode_skeleton_only (vae.py:518): runs encoder.encode_
        skeleton + pool.compute_assignment_and_graph (both motion-independent),
        then synthesizes an all-True frame_mask_lat of length T_lat (every latent
        frame is generated) and the matching token_mask.

        Returns a schema-aligned subset of encode()'s keys (so decode() / the
        CodeFlow backbone can consume it directly):
          s_j, assignment, coarse_mask, frame_mask_lat, token_mask,
          pooled_adjacency, pooled_geodesic, pooled_skeleton_embeddings

        Caller owns the freeze contract (tokenizer.eval() + torch.no_grad()).
        """
        if batch.anytop_graph_dist is None or batch.anytop_joint_relations is None:
            raise ValueError("prepare_skeleton_only (graphormer) requires "
                             "batch.anytop_graph_dist + batch.anytop_joint_relations")
        if T_lat <= 0:
            raise ValueError(f"prepare_skeleton_only: T_lat must be > 0, got {T_lat}")
        gd, jr = batch.anytop_graph_dist, batch.anytop_joint_relations
        s_j = self.encoder.encode_skeleton(
            batch.skeleton_features, batch.adjacency, batch.geodesic_dist,
            batch.joint_mask, name_hashes=batch.name_hashes,
            graph_dist=gd, joint_relations=jr,
        )  # [B,J,D]
        geom = self.pool.compute_assignment_and_graph(
            skeleton_embeddings=s_j.float(),
            adjacency=batch.adjacency.float(),
            geodesic_dist=batch.geodesic_dist.float(),
            joint_mask=batch.joint_mask,
            parent_indices=batch.parent_indices,
        )
        coarse_mask = geom["pooled_mask"]                  # [B,C] bool
        B, C = coarse_mask.shape
        device = coarse_mask.device
        # Every latent frame is to be generated -> all-True frame_mask_lat.
        frame_mask_lat = torch.ones(B, T_lat, dtype=torch.bool, device=device)
        token_mask = coarse_mask.unsqueeze(1) & frame_mask_lat.unsqueeze(-1)  # [B,T_lat,C]
        return {
            "s_j": s_j,
            "assignment": geom["assignment"],
            "coarse_mask": coarse_mask,
            "frame_mask_lat": frame_mask_lat,
            "token_mask": token_mask,
            "pooled_adjacency": geom["pooled_adjacency"],
            "pooled_geodesic": geom["pooled_geodesic"],
            "pooled_skeleton_embeddings": geom["pooled_skeleton_embeddings"],
        }

    @torch.no_grad()
    def decode_from_indices(self, indices: torch.Tensor, skeleton_meta: dict,
                            batch) -> dict:
        """RVQ indices [B,T_lat,C,Q] -> pred_motion [B,T,J,13] via the frozen
        decoder.

        Convenience glue (plan §8 step 6): ids_to_embeddings(indices) -> decode.
        `skeleton_meta` must carry the decode metadata (from encode() OR
        prepare_skeleton_only()): coarse_mask, frame_mask_lat, pooled_adjacency,
        pooled_geodesic, s_j, assignment. Delegates to self.decode (post-VQ
        refine + temporal upsample + masked decoder + anytop13 heads), unchanged.
        """
        token_mask = skeleton_meta["coarse_mask"].unsqueeze(1) \
            & skeleton_meta["frame_mask_lat"].unsqueeze(-1)
        z_q = self.ids_to_embeddings(indices, token_mask)  # [B,T_lat,C,D] fp32
        # decode() reads the autocast dtype from z_q; keep fp32 (decoder casts
        # graph tensors to fp32 internally and accepts fp32 features).
        return self.decode(z_q, skeleton_meta, batch)

    def forward(self, batch, allow_collectives: bool = True) -> dict:
        """allow_collectives: True on the all-ranks DDP training step (the
        quantizer's EMA all_reduce / dead-code + dropout broadcasts fire across
        ranks — amendment 3). Pass False on a rank-0-only eval pass so the
        quantizer does not block on collectives the other ranks skip."""
        enc = self.encode(batch)
        vq = self.quantizer(enc["h_lat"], enc["token_mask"],
                            allow_collectives=allow_collectives)
        dec = self.decode(vq["quantized"], enc, batch)
        return {
            # decoder
            "pred_motion": dec["pred_motion"],
            "frame_mask_recovered": dec["frame_mask_recovered"],
            # latent / quantizer
            "z_q": vq["quantized"],
            "indices": vq["indices"],
            "commit_loss": vq["commit_loss"],
            "commit_sq_sum": vq["commit_sq_sum"],
            "n_valid_tokens": vq["n_valid_tokens"],
            "perplexity": vq["perplexity"],
            "active_codes": vq["active_codes"],
            "dead_codes": vq["dead_codes"],
            # masks / graph (for loss + diagnostics)
            "coarse_mask": enc["coarse_mask"],
            "frame_mask_lat": enc["frame_mask_lat"],
            "token_mask": enc["token_mask"],
            "assignment": enc["assignment"],
        }
