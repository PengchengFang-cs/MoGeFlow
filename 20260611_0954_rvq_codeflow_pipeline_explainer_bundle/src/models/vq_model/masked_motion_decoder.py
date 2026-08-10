"""Masked Motion Decoder for the Graph-VQVAE tokenizer (fork F1).

WHY A FORK (not a wrapper over the shared MotionDecoder):
The shared src/models/motion_decoder.py SlotToJointCrossAttention biases the
attention scores with `assignment.clamp(min=1e-8).log()` (motion_decoder.py:64).
A PADDED slot column has assignment == 0 → clamp(1e-8).log() ≈ -18.42, a large
FINITE negative — NOT -inf. So padded slots still receive a tiny but non-zero
softmax weight, and a large positive raw score (q·k) can overwhelm -18.42. The
shared decoder also has NO coarse_mask parameter. Therefore a wrapper CANNOT
achieve the STRICT -inf slot-key masking the review (forks F1 + amendment 1d)
requires. Per the user's F1 decision, we fork a masked copy HERE inside
src/models/vq_model/ with an explicit `coarse_mask`, and DO NOT touch the shared
MotionDecoder (which is used by the running Gaussian-VAE + diffusion training).

This module is otherwise a structural mirror of motion_decoder.py:20-206 (same
sub-blocks, same dims, same einsum unpool, same temporal refine, same output
masking) so its reconstruction behavior matches the validated coarse_xattn path —
the ONLY semantic change is the strict padded-slot-key mask in the slot
cross-attention (amendment 1d / F1).

bf16-safe: softmax is forced to fp32 then cast back (mirrors the shared decoder).
The -inf is injected in the fp32 score space; a fully-(-inf) query row cannot
occur because every valid joint always has ≥1 valid slot key (the root slot is
always valid), and padded-joint query rows are masked to zero at the output.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedSlotToJointCrossAttention(nn.Module):
    """Slot cross-attention with STRICT -inf masking of padded slot KEYS.

    Mirrors motion_decoder.py::SlotToJointCrossAttention (same projections, same
    learnable assign_bias_scale, same fp32 softmax) but adds a `coarse_mask`
    [B, K] bool that forces padded slot keys to exactly -inf BEFORE softmax →
    exactly zero attention weight on padded slots (amendment 1d / F1).
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)

        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.assign_bias_scale = nn.Parameter(torch.tensor(1.0))

    def forward(
        self,
        joint_queries: torch.Tensor,   # [B, J, D]
        slot_features: torch.Tensor,   # [B, K, D]
        assignment: torch.Tensor,      # [B, J, K] — attention bias (log P)
        coarse_mask: torch.Tensor,     # [B, K] bool — True = valid slot key
    ) -> torch.Tensor:
        B, J, D = joint_queries.shape
        K = slot_features.shape[1]

        q = self.q_proj(self.norm_q(joint_queries))
        kv_in = self.norm_kv(slot_features)
        k = self.k_proj(kv_in)
        v = self.v_proj(kv_in)

        q = q.view(B, J, self.n_heads, self.d_head).permute(0, 2, 1, 3)
        k = k.view(B, K, self.n_heads, self.d_head).permute(0, 2, 1, 3)
        v = v.view(B, K, self.n_heads, self.d_head).permute(0, 2, 1, 3)

        # Attention scores [B, H, J, K]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)

        # Assignment log-bias (joints attend more to their assigned slots).
        assign_bias = assignment.unsqueeze(1).clamp(min=1e-8).log()  # [B, 1, J, K]
        scores = scores + self.assign_bias_scale * assign_bias

        # STRICT padded-slot-KEY mask (F1): set padded slot columns to -inf so
        # their softmax weight is EXACTLY 0. Done in fp32 score space below.
        # [B, K] -> [B, 1, 1, K]
        key_mask = coarse_mask.bool().unsqueeze(1).unsqueeze(2)  # [B,1,1,K]
        scores = scores.float().masked_fill(~key_mask, float("-inf"))

        attn = F.softmax(scores, dim=-1)              # fp32 softmax
        # A query row with at least one valid key never produces all-(-inf); but
        # guard NaN defensively (padded-joint query rows are zeroed downstream).
        attn = attn.nan_to_num(0.0).to(v.dtype)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)  # [B, H, J, d_head]
        out = out.permute(0, 2, 1, 3).contiguous().view(B, J, D)
        return joint_queries + self.o_proj(out)


class TemporalRefineBlock(nn.Module):
    """1D temporal convolution for smoothing decoded motion.

    Structural mirror of motion_decoder.py::TemporalRefineBlock, but FRAME-MASK
    AWARE: the two convs are split (not an nn.Sequential) so a frame mask can be
    applied to (a) the normed conv input, (b) the hidden activation BETWEEN the
    two convs, and (c) the output after the residual add. Without (b), conv1's
    bias makes padded-frame positions nonzero after GELU, and conv2 (kernel>1)
    would then mix those padded positions into adjacent VALID frames — the
    intra-block leak the final output mask alone cannot stop.
    """

    def __init__(self, d_model: int, kernel_size: int = 9, dropout: float = 0.1):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv1 = nn.Conv1d(d_model, d_model, kernel_size, padding=padding)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(d_model, d_model, kernel_size, padding=padding)
        self.drop2 = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, frame_mask: torch.Tensor) -> torch.Tensor:
        """x: [N, T, D]  frame_mask: [N, T, 1] (float, 1=valid) -> [N, T, D]."""
        residual = x
        fm_c = frame_mask.transpose(1, 2)            # [N, 1, T] for conv (channel-first)
        h = self.norm(x).permute(0, 2, 1)            # [N, D, T]
        h = h * fm_c                                  # zero padded frames at conv1 input
        h = self.drop1(self.act(self.conv1(h)))      # [N, D, T]
        h = h * fm_c                                  # (b) zero padded frames BEFORE conv2
        h = self.drop2(self.conv2(h))                # [N, D, T]
        h = (h * fm_c).permute(0, 2, 1)              # [N, T, D] (zero padded frames out)
        return residual + h


class MaskedMotionDecoder(nn.Module):
    """Decode coarse-slot features -> per-joint motion features, with strict
    padded-slot-key masking in every cross-attention layer (F1).

    Structural mirror of motion_decoder.py::MotionDecoder (same unpool einsum,
    same per-frame cross-attn + FFN, same per-joint temporal refine, same output
    norm + joint/frame output masking). Returns the masked pre-output-projection
    features [B, T, J, D] (the VQ tokenizer's anytop13 head does the real output,
    mirroring the shared VAE's return_features=True usage).
    """

    def __init__(
        self,
        d_model: int = 512,
        n_heads: int = 8,
        n_cross_layers: int = 3,
        n_temporal_layers: int = 2,
        temporal_kernel: int = 9,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model

        self.cross_layers = nn.ModuleList([
            MaskedSlotToJointCrossAttention(d_model, n_heads, dropout)
            for _ in range(n_cross_layers)
        ])
        self.ffn_layers = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model * 4, d_model),
                nn.Dropout(dropout),
            )
            for _ in range(n_cross_layers)
        ])
        self.temporal_layers = nn.ModuleList([
            TemporalRefineBlock(d_model, temporal_kernel, dropout)
            for _ in range(n_temporal_layers)
        ])
        self.output_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        slot_features: torch.Tensor,        # [B, T, K, D]
        skeleton_embeddings: torch.Tensor,  # [B, J, D]
        assignment: torch.Tensor,           # [B, J, K] real pool assignment P
        coarse_mask: torch.Tensor,          # [B, K] bool — True = valid slot
        joint_mask: torch.Tensor,           # [B, J] bool
        frame_mask: torch.Tensor,           # [B, T] bool
    ) -> torch.Tensor:
        """Returns masked pre-output features [B, T, J, D]."""
        B, T, K, D = slot_features.shape
        J = skeleton_embeddings.shape[1]

        # 1. Initial unpool: [B, T, J, D]. Assignment already zeroes padded slot
        # columns (P[:, :, c]=0 for padded c), so the einsum never mixes in
        # padded slot features here. The cross-attention below adds the strict
        # -inf key mask on top.
        unpool_features = torch.einsum('bjk,btkd->btjd', assignment, slot_features)

        # 2. Topology-conditioned joint queries.
        joint_features = unpool_features + skeleton_embeddings.unsqueeze(1).expand(-1, T, -1, -1)

        # mask broadcasts reused below: [B,1,J,1] joints, [B,T,1,1] frames.
        jm_b = joint_mask[:, None, :, None].to(joint_features.dtype)
        fm_b = frame_mask[:, :, None, None].to(joint_features.dtype)

        # 3. Cross-attention refinement (per frame).
        cm_e = coarse_mask.unsqueeze(1).expand(-1, T, -1).reshape(B * T, K)  # [B*T, K]
        for cross_attn, ffn in zip(self.cross_layers, self.ffn_layers):
            jf = joint_features.reshape(B * T, J, D)
            sf = slot_features.reshape(B * T, K, D)
            assign_expand = assignment.unsqueeze(1).expand(-1, T, -1, -1).reshape(B * T, J, K)
            jf = cross_attn(jf, sf, assign_expand, cm_e)
            jf = jf + ffn(jf)
            joint_features = jf.reshape(B, T, J, D)

        # Re-mask padded joints + frames BEFORE the temporal conv. Cross-attn +
        # the added skeleton embeddings made padded (joint,frame) entries nonzero;
        # the per-joint temporal conv (kernel>1) mixes neighboring frames, so a
        # nonzero padded FRAME would leak into adjacent valid frames. Zeroing here
        # (and after each temporal layer below) makes padded frames contribute
        # nothing — clean by construction, not just at the final output mask.
        joint_features = joint_features * jm_b * fm_b

        # 4. Temporal refinement (per joint across time). The block is frame-mask
        # aware (masks the conv input, the inter-conv hidden activation, AND its
        # output), so padded frames neither produce output nor leak into valid
        # frames through the convs. The input `joint_features` is already masked
        # above; each block returns a clean (padded-frame-zero) result.
        jf = joint_features.permute(0, 2, 1, 3).reshape(B * J, T, D)  # [B*J, T, D]
        fm_t = frame_mask[:, None, :, None].expand(B, J, T, 1).reshape(B * J, T, 1).to(jf.dtype)
        for temp_layer in self.temporal_layers:
            jf = temp_layer(jf, fm_t)
        joint_features = jf.reshape(B, J, T, D).permute(0, 2, 1, 3)   # [B, T, J, D]

        # 5. Output norm + mask invalid joints / frames.
        features = self.output_norm(joint_features)
        features = features * joint_mask[:, None, :, None].to(features.dtype)
        features = features * frame_mask[:, :, None, None].to(features.dtype)
        return features
