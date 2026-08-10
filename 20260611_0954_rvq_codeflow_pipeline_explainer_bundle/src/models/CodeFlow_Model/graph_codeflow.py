"""GraphStructuredCodeFlow — Level-A graph rectified-flow backbone over the
FROZEN Graph-VQVAE post-RVQ z_q grid.

Plan: handoff/20260609_graph_codeflow_rvq_backbone_plan.md §7 (Level A) +
handoff/20260609_0530_graph_codeflow_locked_recipe_and_state.md (LOCKED recipe).

This is the SEPARATE post-RVQ generation branch (handoff §16): it does NOT touch
the Gaussian VAE / latent diffusion / graph_salad denoiser. It REUSES (by import,
never copy) the shared graph-temporal + dual-text ingredients:

  - GraphAttentionBlock          (graph_salad.attention) — graph-spatial attn
                                   over the C coarse-slot axis (pooled adjacency /
                                   geodesic bias), per latent frame.
  - TemporalSelfAttention        (motion_decoder)        — temporal attn over the
                                   T_lat latent-frame axis, per coarse slot.
  - SinusoidalTimestepEmbedding,
    DenseFiLM, TextCrossAttention (graph_salad.denoiser)  — timestep FiLM, the
                                   dual_text global-add path, and token-level text
                                   cross-attention (CFG-uncond rows zeroed).

I/O is the post-RVQ token grid `[B, T_lat, C, D]` (D = RVQ code_dim). The model
predicts the rectified-flow velocity `v` at the same shape. Strict 2D `[T_lat,C]`
masking (`token_mask = coarse_mask & frame_mask_lat`) is re-applied after every
sub-block AND at input/output — padded slots / padded latent frames never leak.

Text conditioning is the project-default DUAL text (T5-768):
  - GLOBAL: mean-pooled caption [B,768] -> Linear -> additive per-layer (gated by
    has_text for CFG).  Mirrors the denoiser's dual_text global path.
  - TOKEN:  token-level caption [B,L,768] -> Linear -> per-layer cross-attention
    (key-padding-masked; CFG-uncond rows contribute exactly 0).

NOTE on precision: like the graph_salad denoiser, the graph-spatial block requires
fp32 adjacency/geodesic and forces fp32 softmax. This model is intended to run in
fp32 for the flow math (the trainer keeps z_q / graph tensors fp32); bf16 autocast
may wrap the matmuls (GraphAttentionBlock is bf16-safe for features), but the
adjacency/geodesic/text/skeleton conditioning tensors must match z_t.dtype on the
fp32 path (enforced below, same contract as GraphSaladDenoiser).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.graph_salad.attention import GraphAttentionBlock
from src.models.motion_decoder import TemporalSelfAttention
from src.models.graph_salad.denoiser import (
    SinusoidalTimestepEmbedding,
    DenseFiLM,
    TextCrossAttention,
)


class GraphCodeFlowLayer(nn.Module):
    """One Level-A flow layer over the `[B,T_lat,C,D]` token grid.

    Ordering mirrors the proven GraphSaladDenoiserLayer (graph_salad/denoiser.py):
      graph-spatial attn -> FiLM -> temporal attn -> FiLM ->
      [token cross-attn] + [global text add] -> FiLM -> strict padded re-mask.

    The graph-spatial sub-block uses pooled_adjacency / pooled_geodesic exactly as
    CoarseGraphTemporalLayer / the denoiser do; the temporal sub-block self-attends
    over T_lat per slot. Timestep modulation is via DenseFiLM (zero-init -> identity
    at init). Token cross-attn + global add are both gated by has_text for CFG.
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, d_t: int,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.spatial = GraphAttentionBlock(d_model, n_heads, d_ff, dropout=dropout)
        self.temporal = TemporalSelfAttention(d_model, n_heads, dropout=dropout)
        self.text_cross_attn = TextCrossAttention(d_model, n_heads, dropout=dropout)
        self.film_after_spatial = DenseFiLM(d_t, d_model)
        self.film_after_temporal = DenseFiLM(d_t, d_model)
        self.film_after_text = DenseFiLM(d_t, d_model)

    def forward(
        self,
        x: torch.Tensor,                  # [B, T_lat, C, D]
        t_emb: torch.Tensor,              # [B, D_t]
        text_global: torch.Tensor,        # [B, D] projected mean caption
        has_text: torch.Tensor,           # [B] bool
        tok_emb: torch.Tensor,            # [B, L, D] projected token caption
        text_key_padding_mask: torch.Tensor,  # [B, L] bool, True = mask
        pooled_adj: torch.Tensor,         # [B, C, C] fp32
        pooled_geo: torch.Tensor,         # [B, C, C] fp32
        coarse_mask: torch.Tensor,        # [B, C] bool
        frame_mask_lat: torch.Tensor,     # [B, T_lat] bool
        *,
        validate_inputs: bool = False,
    ) -> torch.Tensor:
        B, T_lat, C, D = x.shape

        # --- 1. Graph-spatial self-attn (per latent frame, over C slots) ---
        x_sp_in = x.reshape(B * T_lat, C, D)
        adj_exp = pooled_adj.unsqueeze(1).expand(B, T_lat, C, C).reshape(B * T_lat, C, C)
        geo_exp = pooled_geo.unsqueeze(1).expand(B, T_lat, C, C).reshape(B * T_lat, C, C)
        cm_exp = coarse_mask.unsqueeze(1).expand(B, T_lat, C).reshape(B * T_lat, C)
        x_sp = self.spatial(x_sp_in, adj_exp, geo_exp, cm_exp,
                            validate_inputs=validate_inputs)
        x = x_sp.reshape(B, T_lat, C, D)
        x = self.film_after_spatial(x, t_emb)

        # --- 2. Temporal self-attn (per slot, over T_lat frames) ---
        x_t_in = x.permute(0, 2, 1, 3).contiguous().reshape(B * C, T_lat, D)
        fm_exp = frame_mask_lat.unsqueeze(1).expand(B, C, T_lat).reshape(B * C, T_lat)
        x_t = self.temporal(x_t_in, fm_exp)
        x = x_t.reshape(B, C, T_lat, D).permute(0, 2, 1, 3).contiguous()
        x = self.film_after_temporal(x, t_emb)

        # --- 3. Dual text: token cross-attn THEN global add (both has_text-gated) ---
        q = x.reshape(B, T_lat * C, D)
        ca = self.text_cross_attn(q, tok_emb, text_key_padding_mask)
        x = x + ca.reshape(B, T_lat, C, D)
        text_gated = text_global * has_text[:, None].to(text_global.dtype)  # [B, D]
        x = x + text_gated[:, None, None, :]
        x = self.film_after_text(x, t_emb)

        # --- 4. Strict padded re-mask (padded slots/frames must be 0 after layer) ---
        cm = coarse_mask[:, None, :, None].to(x.dtype)
        fm = frame_mask_lat[:, :, None, None].to(x.dtype)
        return x * cm * fm


class GraphStructuredCodeFlow(nn.Module):
    """Level-A graph rectified-flow velocity network over post-RVQ z_q.

    forward(z_t, timesteps, text_global, text_tokens, text_token_mask, has_text,
            pooled_adjacency, pooled_geodesic, pooled_skeleton_embeddings,
            coarse_mask, frame_mask_lat) -> v_pred [B, T_lat, C, D].

    Skip-transformer (SALAD-style, n_layers odd: enc + mid + dec with symmetric
    skips) of GraphCodeFlowLayer blocks, mirroring GraphSaladDenoiser's structure
    but operating on the frozen RVQ code grid (D = code_dim) instead of the
    Gaussian latent. Output zero-init so v_pred ≈ 0 at init (flow-stable).
    """

    def __init__(
        self,
        code_dim: int = 512,
        n_heads: int = 8,
        d_ff: int | None = None,
        n_layers: int = 5,
        d_text: int = 768,
        text_token_dim: int = 768,
        d_t: int | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if n_layers % 2 == 0:
            raise ValueError(
                f"n_layers must be odd for the SALAD skip-transformer, got {n_layers}")
        if code_dim % n_heads != 0:
            raise ValueError(f"code_dim ({code_dim}) must divide n_heads ({n_heads})")
        if d_ff is None:
            d_ff = 4 * code_dim
        if d_t is None:
            d_t = code_dim
        self.code_dim = code_dim
        self.d_model = code_dim  # the token grid feature dim == RVQ code_dim
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.n_layers = n_layers
        self.d_text = d_text
        self.text_token_dim = text_token_dim
        self.d_t = d_t

        # Timestep embedding (sinusoidal -> MLP), shared across all FiLMs.
        self.t_sin = SinusoidalTimestepEmbedding(d_t)
        self.t_mlp = nn.Sequential(
            nn.Linear(d_t, d_t * 4), nn.SiLU(), nn.Linear(d_t * 4, d_t))

        # Dual-text projections (T5-768 -> code_dim): global mean + token level.
        self.text_proj = nn.Linear(d_text, code_dim)
        self.text_token_proj = nn.Linear(text_token_dim, code_dim)

        # Input projection: z_t + additive skeleton-slot conditioning.
        self.input_proj = nn.Linear(code_dim, code_dim)

        self.layers = nn.ModuleList([
            GraphCodeFlowLayer(code_dim, n_heads, d_ff, d_t, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.depth = (n_layers - 1) // 2
        self.skip_mergers = nn.ModuleList(
            [nn.Linear(2 * code_dim, code_dim) for _ in range(self.depth)])

        # Output: pre-norm + zero-init linear -> initial v_pred ≈ 0.
        self.output_norm = nn.LayerNorm(code_dim)
        self.output_proj = nn.Linear(code_dim, code_dim)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

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
        # ---- contract checks (mirror GraphSaladDenoiser; fail-loud) ----
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
                    f"GraphStructuredCodeFlow: {name}.device {t.device} != "
                    f"z_t.device {ref_device}")
        # Float conditioning tensors must match z_t.dtype on the fp32 path (same
        # contract as GraphSaladDenoiser / GraphAttentionBlock).
        if z_t.dtype in (torch.float32, torch.float64):
            for name, t in (
                ("pooled_adjacency", pooled_adjacency),
                ("pooled_geodesic", pooled_geodesic),
                ("pooled_skeleton_embeddings", pooled_skeleton_embeddings),
                ("text_global", text_global), ("text_tokens", text_tokens),
            ):
                if t.dtype != z_t.dtype:
                    raise ValueError(
                        f"GraphStructuredCodeFlow: {name}.dtype {t.dtype} != "
                        f"z_t.dtype {z_t.dtype}")

        # ---- timestep embedding ----
        t_emb = self.t_mlp(self.t_sin(timesteps))         # [B, D_t]

        # ---- dual-text prep (project + build the shared key-padding mask) ----
        text_global_proj = self.text_proj(text_global)    # [B, D]
        tok_emb = self.text_token_proj(text_tokens)        # [B, L, D]
        # valid key = token present AND has_text=True; key_padding_mask is the
        # inverse (True = mask). has_text=False -> whole row masked -> cross-attn
        # output zeroed in TextCrossAttention (CFG-uncond contributes 0).
        valid_key = text_token_mask & has_text[:, None]    # [B, L]
        text_key_padding_mask = ~valid_key                 # [B, L] True = mask

        # ---- input projection + additive skeleton conditioning ----
        x = self.input_proj(z_t)                           # [B, T_lat, C, D]
        x = x + pooled_skeleton_embeddings.unsqueeze(1).expand(-1, T_lat, -1, -1)
        cm = coarse_mask[:, None, :, None].to(x.dtype)
        fm = frame_mask_lat[:, :, None, None].to(x.dtype)
        x = x * cm * fm

        # ---- skip-transformer: enc -> mid -> dec with symmetric skips ----
        def _run(layer, h):
            return layer(
                h, t_emb, text_global_proj, has_text, tok_emb, text_key_padding_mask,
                pooled_adjacency, pooled_geodesic, coarse_mask, frame_mask_lat,
                validate_inputs=validate_inputs)

        enc_outputs: list[torch.Tensor] = []
        for i in range(self.depth):
            x = _run(self.layers[i], x)
            enc_outputs.append(x)
        x = _run(self.layers[self.depth], x)
        for i in range(self.depth):
            dec_layer = self.layers[self.depth + 1 + i]
            skip = enc_outputs[self.depth - 1 - i]
            x = torch.cat([x, skip], dim=-1)               # [B, T_lat, C, 2D]
            x = self.skip_mergers[i](x)                     # [B, T_lat, C, D]
            x = _run(dec_layer, x)

        # ---- output: pre-norm + zero-init linear + final re-mask ----
        x = self.output_norm(x)
        v_pred = self.output_proj(x)
        return v_pred * cm * fm
