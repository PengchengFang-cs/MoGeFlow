"""GraphPSCFFlowNet — graph-aware PSCF / FLUX-style double-stream + single-stream
rectified-flow backbone over the FROZEN Graph-VQVAE post-RVQ z_q grid.

Spec (LOCKED): handoff/20260609_1650_graph_pscf_final_executor_spec.md
Plan §4:      handoff/20260609_graph_codeflow_pscf_double_single_impl_plan.md

This is the formal backbone replacing the Level-A `GraphStructuredCodeFlow` probe.
It is PURE ADDITIVE: it only imports (never copies) the shared audited ingredients
and does NOT touch the Gaussian VAE / latent diffusion / Graph-VQVAE training /
graph_salad behavior.

Three internal streams (plan §3):
  slot stream:  h_slot  [B, T_lat, C, D]   graph-pooled RVQ latent grid
  frame stream: h_frame [B, T_lat, D]      graph-aware frame holder tokens (persistent, seeded once — Q4)
  text stream:  h_text  [B, L, D]          T5 token stream

Conditioning is two-path text (Q2):
  - pooled_text -> cond (= timestep_emb + pooled_text, CFG-gated) -> every AdaLN/FiLM
  - token_text  -> h_text stream -> double/single DiT joint attention

The three new modules:
  - GraphSlotTemporalBlock   keeps the slot stream graph-aware (GraphAttentionBlock
                              over C with pooled adjacency/geodesic) + temporal
                              (TemporalSelfAttention over T_lat), with DenseFiLM
                              modulation. Mirrors graph_codeflow.py:94-122.
  - GraphFrameSlotCoupling   NON-GRAPH (Q1) bridge: the frame holder reads slots
                              via masked cross-attention; slots receive the updated
                              frame via a FiLM-gated additive injection. Does NOT
                              build a [1+C] extended graph (Floyd hub collapse).
  - GraphPSCFFlowNet         6 double-stream + 12 single-stream stack with graph
                              slot stream + couplings, zero-init velocity output.

bf16 discipline: fp32 softmax everywhere; the ported DiT uses a bf16 -1e4 / fp32
-1e9 mask sentinel. Graph tensors (adjacency / geodesic / skeleton) stay fp32 for
GraphAttentionBlock. Strict 2D `[T_lat,C]` re-mask after every sub-block.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.graph_salad.attention import GraphAttentionBlock
from src.models.motion_decoder import TemporalSelfAttention
from src.models.graph_salad.denoiser import (
    SinusoidalTimestepEmbedding,
    DenseFiLM,
)
from src.models.CodeFlow_Model.dit_blocks import (
    DoubleStreamBlock,
    SingleStreamBlock,
)


def _film_frame(film: DenseFiLM, h_frame: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
    """Apply a 4D DenseFiLM to a [B,T,D] frame stream by treating it as a single
    coarse-slot grid [B,T,1,D]. DenseFiLM broadcasts (scale,shift) as [B,1,1,D],
    so on [B,T,1,D] this is the identical affine the slot stream gets — we reuse
    the exact audited DenseFiLM rather than introducing a parallel 3D variant."""
    return film(h_frame.unsqueeze(2), cond).squeeze(2)


class GraphSlotTemporalBlock(nn.Module):
    """Graph-spatial (over C) + temporal (over T_lat) update of the slot stream.

    Mirrors graph_codeflow.py:94-109 EXACTLY for the reshape / expand / mask
    patterns:
      1. graph-spatial: reshape [B*T,C,D], expand adj/geo/coarse_mask over T,
         GraphAttentionBlock(..., validate_inputs), reshape back, FiLM(cond),
         re-mask by coarse_mask.
      2. temporal: permute to [B*C,T,D], expand frame_mask over C,
         TemporalSelfAttention, permute back, FiLM(cond).
      3. strict re-mask by coarse_mask AND frame_mask_lat.
    """

    def __init__(self, code_dim: int, n_heads: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.spatial = GraphAttentionBlock(code_dim, n_heads, d_ff, dropout=dropout)
        self.temporal = TemporalSelfAttention(code_dim, n_heads, dropout=dropout)
        self.film_after_spatial = DenseFiLM(code_dim, code_dim)
        self.film_after_temporal = DenseFiLM(code_dim, code_dim)

    def forward(
        self,
        h_slot: torch.Tensor,         # [B, T, C, D]
        cond: torch.Tensor,           # [B, D]
        pooled_adj: torch.Tensor,     # [B, C, C] fp32
        pooled_geo: torch.Tensor,     # [B, C, C] fp32
        coarse_mask: torch.Tensor,    # [B, C] bool
        frame_mask_lat: torch.Tensor,  # [B, T] bool
        *,
        validate_inputs: bool = False,
    ) -> torch.Tensor:
        B, T, C, D = h_slot.shape

        # --- 1. Graph-spatial self-attn (per latent frame, over C slots) ---
        x_sp_in = h_slot.reshape(B * T, C, D)
        adj_exp = pooled_adj.unsqueeze(1).expand(B, T, C, C).reshape(B * T, C, C)
        geo_exp = pooled_geo.unsqueeze(1).expand(B, T, C, C).reshape(B * T, C, C)
        cm_exp = coarse_mask.unsqueeze(1).expand(B, T, C).reshape(B * T, C)
        x_sp = self.spatial(x_sp_in, adj_exp, geo_exp, cm_exp,
                            validate_inputs=validate_inputs)
        h_slot = x_sp.reshape(B, T, C, D)
        h_slot = self.film_after_spatial(h_slot, cond)
        # re-mask by coarse_mask AND frame_mask_lat after the spatial sub-block:
        # padded query slots/frames still attend valid keys -> non-zero output, and
        # I7 strict padded-zero requires clearing BOTH after every sub-block (else
        # padded frame rows leak as non-zero queries into the temporal sub-block).
        h_slot = (h_slot * coarse_mask[:, None, :, None].to(h_slot.dtype)
                  * frame_mask_lat[:, :, None, None].to(h_slot.dtype))

        # --- 2. Temporal self-attn (per slot, over T_lat frames) ---
        x_t_in = h_slot.permute(0, 2, 1, 3).contiguous().reshape(B * C, T, D)
        fm_exp = frame_mask_lat.unsqueeze(1).expand(B, C, T).reshape(B * C, T)
        x_t = self.temporal(x_t_in, fm_exp)
        h_slot = x_t.reshape(B, C, T, D).permute(0, 2, 1, 3).contiguous()
        h_slot = self.film_after_temporal(h_slot, cond)

        # --- 3. Strict padded re-mask (coarse_mask AND frame_mask_lat) ---
        cm = coarse_mask[:, None, :, None].to(h_slot.dtype)
        fm = frame_mask_lat[:, :, None, None].to(h_slot.dtype)
        return h_slot * cm * fm


class GraphFrameSlotCoupling(nn.Module):
    """NON-GRAPH (Q1) bridge between the per-frame holder and the C graph slots.

    Per Q1 the holder is NOT inserted into the GraphAttentionBlock adjacency /
    geodesic (a holder hub collapses slot-slot geodesics to <=2 hops and trips the
    Floyd strict validation). Instead:

      - holder reads slots: per latent frame, h_frame[:,t] (1 query) cross-attends
        over h_slot[:,t] (C keys), keys masked by coarse_mask (True=valid). FiLM,
        residual.
      - slots receive frame: a FiLM-gated additive injection of a Linear-projected
        h_frame broadcast over the C slots. residual.
      - strict re-mask both streams.

    fp32 softmax (bf16-safe). No graph metadata is touched here.
    """

    def __init__(self, code_dim: int, n_heads: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        if code_dim % n_heads != 0:
            raise ValueError(
                f"code_dim ({code_dim}) must be divisible by n_heads ({n_heads})")
        self.n_heads = n_heads
        self.d_head = code_dim // n_heads

        # --- holder-reads-slots cross-attention (manual, masked, fp32 softmax) ---
        self.norm_q = nn.LayerNorm(code_dim)
        self.norm_kv = nn.LayerNorm(code_dim)
        self.q_proj = nn.Linear(code_dim, code_dim)
        self.k_proj = nn.Linear(code_dim, code_dim)
        self.v_proj = nn.Linear(code_dim, code_dim)
        self.o_proj = nn.Linear(code_dim, code_dim)
        # zero-init output -> the holder-read is a no-op residual at init (stable).
        nn.init.zeros_(self.o_proj.weight)
        nn.init.zeros_(self.o_proj.bias)
        self.dropout = nn.Dropout(dropout)
        self.film_frame = DenseFiLM(code_dim, code_dim)

        # --- slots-receive-frame additive injection (FiLM-gated) ---
        self.norm_inject = nn.LayerNorm(code_dim)
        self.inject_proj = nn.Linear(code_dim, code_dim)
        # zero-init injection -> slots receive nothing at init (no-op residual).
        nn.init.zeros_(self.inject_proj.weight)
        nn.init.zeros_(self.inject_proj.bias)
        self.film_slot = DenseFiLM(code_dim, code_dim)

    def forward(
        self,
        h_frame: torch.Tensor,        # [B, T, D]
        h_slot: torch.Tensor,         # [B, T, C, D]
        cond: torch.Tensor,           # [B, D]
        coarse_mask: torch.Tensor,    # [B, C] bool, True = valid
        frame_mask_lat: torch.Tensor,  # [B, T] bool, True = valid
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, C, D = h_slot.shape

        # ---- 1. holder reads slots: per-frame [B*T,1,D] q over [B*T,C,D] kv ----
        q_in = self.norm_q(h_frame).reshape(B * T, 1, D)         # [B*T,1,D]
        kv_in = self.norm_kv(h_slot).reshape(B * T, C, D)        # [B*T,C,D]
        q = self.q_proj(q_in).view(B * T, 1, self.n_heads, self.d_head).permute(0, 2, 1, 3)
        k = self.k_proj(kv_in).view(B * T, C, self.n_heads, self.d_head).permute(0, 2, 1, 3)
        v = self.v_proj(kv_in).view(B * T, C, self.n_heads, self.d_head).permute(0, 2, 1, 3)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)  # [B*T,H,1,C]
        # key_padding_mask = ~coarse_mask expanded over T (True = ignore).
        kpm = (~coarse_mask)[:, None, :].expand(B, T, C).reshape(B * T, C)       # [B*T,C]
        scores = scores.masked_fill(kpm[:, None, None, :], -1e9)
        # fp32 softmax (bf16-safe; fp32 path is a no-op). all-padded rows cannot
        # occur (>=1 valid slot per sample guaranteed by coarse_mask), but
        # nan_to_num guards defensively.
        attn = F.softmax(scores.float(), dim=-1).to(scores.dtype)
        attn = attn.nan_to_num(0.0)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)                                # [B*T,H,1,dh]
        out = out.permute(0, 2, 1, 3).contiguous().view(B * T, 1, D)
        out = self.o_proj(out).reshape(B, T, D)
        h_frame = h_frame + self.dropout(out)
        h_frame = _film_frame(self.film_frame, h_frame, cond)
        # re-mask frame stream (padded latent frames -> 0).
        h_frame = h_frame * frame_mask_lat[:, :, None].to(h_frame.dtype)

        # ---- 2. slots receive frame: FiLM-gated additive injection ----
        inj = self.inject_proj(self.norm_inject(h_frame))          # [B,T,D]
        # broadcast the per-frame holder context to all C slots, then FiLM-gate.
        inj = inj[:, :, None, :].expand(B, T, C, D)                # [B,T,C,D]
        inj = self.film_slot(inj, cond)
        h_slot = h_slot + inj

        # ---- 3. strict re-mask both streams ----
        cm = coarse_mask[:, None, :, None].to(h_slot.dtype)
        fm4 = frame_mask_lat[:, :, None, None].to(h_slot.dtype)
        h_slot = h_slot * cm * fm4
        h_frame = h_frame * frame_mask_lat[:, :, None].to(h_frame.dtype)
        return h_frame, h_slot


class GraphPSCFFlowNet(nn.Module):
    """Graph-aware PSCF / FLUX-style rectified-flow velocity network over post-RVQ z_q.

    6 double-stream + 12 single-stream DiT blocks (depth_double / depth_single)
    with a graph-aware slot stream (GraphSlotTemporalBlock) bridged to the frame /
    text DiT by GraphFrameSlotCoupling (NON-GRAPH, Q1). Conditioning is two-path
    text (Q2): pooled_text -> cond (AdaLN/FiLM, CFG-gated); token_text -> h_text
    stream. h_frame is persistent (Q4): seeded once and updated in place by every
    double / single / coupling.

    forward signature is the EXACT 11-positional-arg contract of
    GraphStructuredCodeFlow.forward (graph_codeflow.py:194-209) so flow.py /
    predict_velocity / CFG stay drop-in.
    """

    def __init__(
        self,
        code_dim: int = 512,
        n_heads: int = 8,
        d_ff: int = 2048,
        depth_double: int = 6,
        depth_single: int = 12,
        d_text: int = 768,
        text_token_dim: int = 768,
        max_T_lat: int = 75,
        mlp_ratio: float = 4.0,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        if code_dim % n_heads != 0:
            raise ValueError(
                f"code_dim ({code_dim}) must be divisible by n_heads ({n_heads})")
        if depth_double <= 0 or depth_single <= 0:
            raise ValueError(
                f"depth_double / depth_single must be > 0, got "
                f"{depth_double} / {depth_single}")
        self.code_dim = code_dim
        self.d_model = code_dim
        self.n_heads = n_heads
        self.head_dim = code_dim // n_heads
        self.d_ff = d_ff
        self.depth_double = depth_double
        self.depth_single = depth_single
        self.d_text = d_text
        self.text_token_dim = text_token_dim
        self.max_T_lat = max_T_lat
        self.mlp_ratio = mlp_ratio
        # RoPE over the single frame axis: one axis spanning the full head_dim.
        self.rope_axes_dims = [self.head_dim]

        # ---- timestep embedding (sinusoidal -> MLP) ----
        self.t_sin = SinusoidalTimestepEmbedding(code_dim)
        self.t_mlp = nn.Sequential(
            nn.Linear(code_dim, code_dim * 4), nn.SiLU(),
            nn.Linear(code_dim * 4, code_dim))

        # ---- dual-text projections (T5-768 -> code_dim) ----
        self.text_pooled_proj = nn.Linear(d_text, code_dim)
        self.text_token_proj = nn.Linear(text_token_dim, code_dim)

        # ---- slot input projection (z_t -> slot stream; skeleton added after) ----
        self.input_proj = nn.Linear(code_dim, code_dim)

        # ---- persistent frame holder seed (Q4) + frame positional embedding ----
        self.frame_seed = nn.Parameter(torch.randn(1, max_T_lat, code_dim) * 0.02)
        # additive sinusoidal frame position over T_lat (precomputed, registered
        # as a non-persistent buffer so it travels by shape, not by save).
        self.register_buffer(
            "time_pos_embedding",
            self._build_sinusoidal_pos(max_T_lat, code_dim),
            persistent=False,
        )

        # ---- double-stream blocks ----
        self.double_blocks = nn.ModuleList([
            nn.ModuleDict({
                "slot_temporal": GraphSlotTemporalBlock(code_dim, n_heads, d_ff, dropout),
                "coupling_pre": GraphFrameSlotCoupling(code_dim, n_heads, d_ff, dropout),
                "dit": DoubleStreamBlock(code_dim, n_heads, mlp_ratio, dropout),
                "coupling_post": GraphFrameSlotCoupling(code_dim, n_heads, d_ff, dropout),
            })
            for _ in range(depth_double)
        ])

        # ---- single-stream blocks ----
        self.single_blocks = nn.ModuleList([
            nn.ModuleDict({
                "dit": SingleStreamBlock(code_dim, n_heads, mlp_ratio, dropout),
                "slot_temporal": GraphSlotTemporalBlock(code_dim, n_heads, d_ff, dropout),
                "coupling": GraphFrameSlotCoupling(code_dim, n_heads, d_ff, dropout),
            })
            for _ in range(depth_single)
        ])

        # ---- output: pre-norm + zero-init linear -> v_pred ~ 0 at init ----
        self.output_norm = nn.LayerNorm(code_dim)
        self.output_proj = nn.Linear(code_dim, code_dim)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    @staticmethod
    def _build_sinusoidal_pos(max_len: int, dim: int) -> torch.Tensor:
        """Standard [max_len, dim] sinusoidal table (additive frame position)."""
        if dim % 2 != 0:
            raise ValueError(f"dim must be even, got {dim}")
        pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)      # [L,1]
        div = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim))
        pe = torch.zeros(max_len, dim, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe

    def forward(
        self,
        z_t: torch.Tensor,                       # [B, T_lat, C, D]
        timesteps: torch.Tensor,                 # [B] (float in [0,1])
        text_global: torch.Tensor,               # [B, 768] mean caption
        text_tokens: torch.Tensor,               # [B, L, 768] token caption
        text_token_mask: torch.Tensor,           # [B, L] bool, True = valid token
        has_text: torch.Tensor,                  # [B] bool (CFG gate)
        pooled_adjacency: torch.Tensor,          # [B, C, C]
        pooled_geodesic: torch.Tensor,           # [B, C, C]
        pooled_skeleton_embeddings: torch.Tensor,  # [B, C, D]
        coarse_mask: torch.Tensor,               # [B, C] bool
        frame_mask_lat: torch.Tensor,            # [B, T_lat] bool
        *,
        validate_inputs: bool = False,
    ) -> torch.Tensor:
        if z_t.dim() != 4 or z_t.shape[-1] != self.code_dim:
            raise ValueError(
                f"z_t must be [B,T_lat,C,D={self.code_dim}], got {tuple(z_t.shape)}")
        B, T_lat, C, D = z_t.shape
        ref_device = z_t.device
        # ---- contract checks (mirror GraphStructuredCodeFlow; fail-loud) ----
        if timesteps.shape != (B,):
            raise ValueError(f"timesteps must be [B={B}], got {tuple(timesteps.shape)}")
        if coarse_mask.shape != (B, C) or coarse_mask.dtype != torch.bool:
            raise ValueError(
                f"coarse_mask must be [B={B},C={C}] bool, got "
                f"{tuple(coarse_mask.shape)} {coarse_mask.dtype}")
        if frame_mask_lat.shape != (B, T_lat) or frame_mask_lat.dtype != torch.bool:
            raise ValueError(
                f"frame_mask_lat must be [B={B},T_lat={T_lat}] bool, got "
                f"{tuple(frame_mask_lat.shape)} {frame_mask_lat.dtype}")
        if has_text.shape != (B,) or has_text.dtype != torch.bool:
            raise ValueError(
                f"has_text must be [B={B}] bool, got {tuple(has_text.shape)} "
                f"{has_text.dtype}")
        if pooled_adjacency.shape != (B, C, C) or pooled_geodesic.shape != (B, C, C):
            raise ValueError(
                f"pooled_adjacency/geodesic must be [B={B},C={C},C={C}], got "
                f"{tuple(pooled_adjacency.shape)} / {tuple(pooled_geodesic.shape)}")
        if pooled_skeleton_embeddings.shape != (B, C, D):
            raise ValueError(
                f"pooled_skeleton_embeddings must be [B={B},C={C},D={D}], got "
                f"{tuple(pooled_skeleton_embeddings.shape)}")
        if text_global.dim() != 2 or text_global.shape != (B, self.d_text):
            raise ValueError(
                f"text_global must be [B={B},{self.d_text}], got {tuple(text_global.shape)}")
        if (text_tokens.dim() != 3 or text_tokens.shape[0] != B
                or text_tokens.shape[2] != self.text_token_dim):
            raise ValueError(
                f"text_tokens must be [B={B},L,{self.text_token_dim}], got "
                f"{tuple(text_tokens.shape)}")
        L = text_tokens.shape[1]
        if text_token_mask.shape != (B, L) or text_token_mask.dtype != torch.bool:
            raise ValueError(
                f"text_token_mask must be [B={B},L={L}] bool, got "
                f"{tuple(text_token_mask.shape)} {text_token_mask.dtype}")
        if T_lat > self.max_T_lat:
            raise ValueError(
                f"T_lat={T_lat} exceeds max_T_lat={self.max_T_lat} (frame_seed / "
                f"time_pos_embedding cannot cover it)")
        for name, t in (
            ("timesteps", timesteps), ("coarse_mask", coarse_mask),
            ("frame_mask_lat", frame_mask_lat), ("has_text", has_text),
            ("pooled_adjacency", pooled_adjacency), ("pooled_geodesic", pooled_geodesic),
            ("pooled_skeleton_embeddings", pooled_skeleton_embeddings),
            ("text_global", text_global), ("text_tokens", text_tokens),
            ("text_token_mask", text_token_mask),
        ):
            if t.device != ref_device:
                raise ValueError(
                    f"GraphPSCFFlowNet: {name}.device {t.device} != "
                    f"z_t.device {ref_device}")
        # Float conditioning tensors must match z_t.dtype on the fp32 path (same
        # contract as GraphStructuredCodeFlow / GraphAttentionBlock).
        if z_t.dtype in (torch.float32, torch.float64):
            for name, t in (
                ("pooled_adjacency", pooled_adjacency),
                ("pooled_geodesic", pooled_geodesic),
                ("pooled_skeleton_embeddings", pooled_skeleton_embeddings),
                ("text_global", text_global), ("text_tokens", text_tokens),
            ):
                if t.dtype != z_t.dtype:
                    raise ValueError(
                        f"GraphPSCFFlowNet: {name}.dtype {t.dtype} != "
                        f"z_t.dtype {z_t.dtype}")

        dtype = z_t.dtype

        # ---- conditioning (Q2): cond = timestep_emb + pooled_text (CFG-gated) ----
        t_emb = self.t_mlp(self.t_sin(timesteps)).to(dtype)        # [B, D]
        text_pooled = self.text_pooled_proj(text_global)           # [B, D]
        text_pooled = text_pooled * has_text[:, None].to(dtype)    # CFG gate
        cond = t_emb + text_pooled                                 # [B, D]

        # ---- slot stream: input_proj(z_t) + pooled skeleton (I2) ----
        h_slot = self.input_proj(z_t)
        h_slot = h_slot + pooled_skeleton_embeddings.unsqueeze(1).expand(-1, T_lat, -1, -1)
        cm = coarse_mask[:, None, :, None].to(dtype)
        fm4 = frame_mask_lat[:, :, None, None].to(dtype)
        h_slot = h_slot * cm * fm4

        # ---- frame stream (Q4): persistent seed + sinusoidal frame pos ----
        h_frame = self.frame_seed[:, :T_lat, :].expand(B, T_lat, D).to(dtype)
        time_pos = self.time_pos_embedding[:T_lat].to(device=ref_device, dtype=dtype)
        h_frame = h_frame + time_pos[None, :, :]
        h_frame = h_frame * frame_mask_lat[:, :, None].to(dtype)

        # ---- text stream: token projection (I3: True = valid) ----
        h_text = self.text_token_proj(text_tokens)                 # [B, L, D]
        text_valid = text_token_mask & has_text[:, None]           # [B, L] True = valid
        h_text = h_text * text_valid[:, :, None].to(dtype)

        # ---- frame motion pos_ids over T_lat: [B,T_lat,1] (single RoPE axis) ----
        frame_pos = torch.arange(T_lat, device=ref_device, dtype=torch.long)
        motion_pos_ids = frame_pos[None, :, None].expand(B, T_lat, 1)  # [B,T_lat,1]

        # ============================ DOUBLE STAGE ============================
        # per block (plan §4.5):
        #   h_slot = slot_temporal(h_slot)
        #   (h_frame,h_slot) = coupling_pre(h_frame,h_slot)
        #   (h_frame,h_text) = DoubleStreamBlock(h_frame,h_text,cond)
        #   (h_frame,h_slot) = coupling_post(h_frame,h_slot)
        #   strict mask all
        for blk in self.double_blocks:
            h_slot = blk["slot_temporal"](
                h_slot, cond, pooled_adjacency, pooled_geodesic,
                coarse_mask, frame_mask_lat, validate_inputs=validate_inputs)
            h_frame, h_slot = blk["coupling_pre"](
                h_frame, h_slot, cond, coarse_mask, frame_mask_lat)
            # DiT boundary: ported DoubleStreamBlock takes True=valid directly.
            h_frame, h_text = blk["dit"](
                h_frame, h_text, cond,
                frame_mask_lat, text_valid,
                motion_pos_ids, self.rope_axes_dims)
            # ported DiT only key-masks; re-mask the padded frame/text rows.
            h_frame = h_frame * frame_mask_lat[:, :, None].to(dtype)
            h_text = h_text * text_valid[:, :, None].to(dtype)
            h_frame, h_slot = blk["coupling_post"](
                h_frame, h_slot, cond, coarse_mask, frame_mask_lat)
            # strict mask all
            h_slot = h_slot * cm * fm4
            h_frame = h_frame * frame_mask_lat[:, :, None].to(dtype)
            h_text = h_text * text_valid[:, :, None].to(dtype)

        # ============================ SINGLE STAGE ===========================
        # text PERSISTS across single blocks (mirrors FrameMotionTextDiT). per
        # block (plan §4.6):
        #   joint = concat(h_frame, h_text)
        #   joint = SingleStreamBlock(joint, cond, valid, joint_pos)
        #   split -> h_frame, h_text
        #   h_slot = slot_temporal(h_slot)
        #   (h_frame,h_slot) = coupling(h_frame,h_slot)
        #   strict mask
        text_pos = torch.zeros(B, L, 1, device=ref_device, dtype=torch.long)
        joint_pos = torch.cat([motion_pos_ids, text_pos], dim=1)   # [B,T_lat+L,1]
        joint_valid = torch.cat([frame_mask_lat, text_valid], dim=1)  # [B,T_lat+L]
        for blk in self.single_blocks:
            joint = torch.cat([h_frame, h_text], dim=1)            # [B,T_lat+L,D]
            joint = blk["dit"](
                joint, cond, joint_valid, joint_pos, self.rope_axes_dims)
            h_frame, h_text = joint.split([T_lat, L], dim=1)
            # ported DiT only key-masks; re-mask padded frame/text rows.
            h_frame = h_frame * frame_mask_lat[:, :, None].to(dtype)
            h_text = h_text * text_valid[:, :, None].to(dtype)
            h_slot = blk["slot_temporal"](
                h_slot, cond, pooled_adjacency, pooled_geodesic,
                coarse_mask, frame_mask_lat, validate_inputs=validate_inputs)
            h_frame, h_slot = blk["coupling"](
                h_frame, h_slot, cond, coarse_mask, frame_mask_lat)
            # strict mask
            h_slot = h_slot * cm * fm4
            h_frame = h_frame * frame_mask_lat[:, :, None].to(dtype)
            h_text = h_text * text_valid[:, :, None].to(dtype)

        # ---- output: pre-norm + zero-init linear + final strict re-mask ----
        v_pred = self.output_proj(self.output_norm(h_slot))
        return v_pred * cm * fm4
