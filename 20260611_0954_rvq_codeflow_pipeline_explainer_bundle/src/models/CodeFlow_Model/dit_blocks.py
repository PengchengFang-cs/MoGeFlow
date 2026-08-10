"""DiT/Flux-style transformer blocks for graph_pscf — LOCAL PORT.

This is a local, self-contained port (copy/adapt) of the FLUX/CodeFlow DiT
blocks from ``outside_docs/CodeFlow/models/codeflow/dit_blocks.py``. It is NOT a
runtime import of that tree: importing ``outside_docs/CodeFlow`` pulls
``__init__ -> eval_t2m -> utils.metrics`` which raises ModuleNotFoundError, so
the blocks are copied here verbatim (with only the minimal adaptation noted
below) so the CodeFlow_Model branch is self-contained and auditable.

Status: commit-pending; audited downstream by graph_pscf.py (M1+) and the
milestone codex review for M0.

Ported classes (exact forward signatures preserved so graph_pscf.py can call
them positionally):

  RMSNorm
  SwiGLU
  MultiHeadAttention   (incl. RoPE helpers _rope_cos_sin / _apply_rope and the
                        key_valid / query_valid / query_pos / memory_pos /
                        rope_axes_dims plumbing — runs standalone)
  AdaLNModulation      (AdaLN-zero: linear weight+bias zero-init -> gates ~0)
  DoubleStreamBlock    forward(motion, text, cond, motion_valid, text_valid,
                               pos_ids, rope_axes_dims)
  SingleStreamBlock    forward(x, cond, valid, pos_ids, rope_axes_dims)

bf16 discipline (matches src/models/graph_salad/attention.py):
  - softmax is computed in fp32 then cast back (manual fallback path);
  - the additive attention mask uses a bf16-safe finite sentinel (-1.0e4 for
    fp16/bf16, -1.0e9 for fp32/fp64), never -inf, so bf16 autocast is safe.
  RMSNorm and RoPE also internally upcast to fp32 then cast back to the input
  dtype, matching the original.

Mask polarity note: these blocks consume ``*_valid`` masks where True == valid
(``key_valid`` / ``query_valid`` / ``motion_valid`` / ``text_valid`` /
``valid``). The project-wide True==valid convention therefore passes straight
through; the True==pad -> True==valid conversion at the model boundary is the
caller's (graph_pscf) responsibility (spec must-fix #2), not these blocks'.

Deviations from the original file: NONE in behavior. Only the classes/helpers
required by the spec (RMSNorm, SwiGLU, MultiHeadAttention + RoPE/_attention,
AdaLNModulation, DoubleStreamBlock, SingleStreamBlock) plus the shared
ModulationOut dataclass are ported; the original's higher-level assemblies
(TimestepEmbedder, FrameHolderCouplingBlock, FrameHolderOutput, FinalLayer,
MotionTextDiT, FrameMotionTextDiT) are intentionally omitted because graph_pscf
replaces them with graph-aware modules. The ported code is byte-for-byte
faithful to the originals it copies.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (self.weight * x).to(dtype)


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.gate = nn.Linear(dim, hidden_dim)
        self.up = nn.Linear(dim, hidden_dim)
        self.down = nn.Linear(hidden_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


@dataclass
class ModulationOut:
    shift: torch.Tensor
    scale: torch.Tensor
    gate: torch.Tensor


class AdaLNModulation(nn.Module):
    """AdaLN-Zero modulation: returns shift, scale and residual gate.

    Zero-init of both weight and bias makes shift/scale/gate start at 0, so the
    residual gate is ~0 at init (AdaLN-zero): every block is initialized as the
    identity, which is what stabilizes flow startup.
    """

    def __init__(self, hidden_size: int, num: int = 1) -> None:
        super().__init__()
        self.num = num
        self.linear = nn.Linear(hidden_size, hidden_size * 3 * num)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, cond: torch.Tensor) -> List[ModulationOut]:
        chunks = self.linear(F.silu(cond)).chunk(self.num * 3, dim=-1)
        outs = []
        for idx in range(self.num):
            shift, scale, gate = chunks[idx * 3 : (idx + 1) * 3]
            outs.append(ModulationOut(shift[:, None], scale[:, None], gate[:, None]))
        return outs


def _rope_cos_sin(
    pos_ids: torch.Tensor,
    head_dim: int,
    axes_dims: List[int],
    theta: int = 10000,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if sum(axes_dims) != head_dim:
        raise ValueError(f"axes_dims sum {sum(axes_dims)} must equal head_dim {head_dim}")
    if pos_ids.shape[-1] != len(axes_dims):
        raise ValueError(f"pos_ids has {pos_ids.shape[-1]} axes, expected {len(axes_dims)}")
    cos_parts = []
    sin_parts = []
    for axis, axis_dim in enumerate(axes_dims):
        if axis_dim % 2 != 0:
            raise ValueError(f"RoPE axis dim must be even, got {axis_dim}")
        half = axis_dim // 2
        scale = torch.arange(0, half, dtype=torch.float32, device=pos_ids.device) / max(half, 1)
        omega = 1.0 / (theta ** scale)
        angles = pos_ids[..., axis].float()[..., None] * omega
        cos_parts.append(torch.cos(angles))
        sin_parts.append(torch.sin(angles))
    return torch.cat(cos_parts, dim=-1), torch.cat(sin_parts, dim=-1)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: [B, H, N, D], cos/sin: [B, N, D/2]
    x_float = x.float()
    x_even = x_float[..., 0::2]
    x_odd = x_float[..., 1::2]
    cos = cos[:, None]
    sin = sin[:, None]
    out = torch.empty_like(x_float)
    out[..., 0::2] = x_even * cos - x_odd * sin
    out[..., 1::2] = x_even * sin + x_odd * cos
    return out.to(x.dtype)


def _attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    key_valid: Optional[torch.Tensor],
    dropout_p: float,
) -> torch.Tensor:
    # q/k/v: [B, H, N, D]; key_valid: [B, N] bool (True == valid key)
    attn_mask = None
    if key_valid is not None:
        # bf16-safe additive mask sentinel: -1e4 (finite) for fp16/bf16 so the
        # masked-softmax does not overflow under autocast; -1e9 on the fp32/fp64
        # path (matches graph_salad/attention.py). Never -inf.
        mask_value = -1.0e4 if q.dtype in (torch.float16, torch.bfloat16) else -1.0e9
        attn_mask = torch.zeros(
            key_valid.shape[0], 1, 1, key_valid.shape[1],
            device=key_valid.device,
            dtype=q.dtype,
        )
        attn_mask = attn_mask.masked_fill(~key_valid[:, None, None], mask_value)

    if hasattr(F, "scaled_dot_product_attention"):
        return F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=False,
        )

    scale = q.shape[-1] ** -0.5
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    if attn_mask is not None:
        scores = scores + attn_mask
    # softmax in fp32 for bf16-safety (sentinel + reduction precision), then cast
    # the probabilities back to the working dtype for the attn @ v matmul.
    probs = torch.softmax(scores.float(), dim=-1).to(q.dtype)
    probs = F.dropout(probs, p=dropout_p, training=dropout_p > 0)
    return torch.matmul(probs, v)


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(f"hidden_size {hidden_size} must be divisible by num_heads {num_heads}")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.q = nn.Linear(hidden_size, hidden_size)
        self.kv = nn.Linear(hidden_size, hidden_size * 2)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        self.out = nn.Linear(hidden_size, hidden_size)
        self.dropout = dropout

    def forward(
        self,
        query: torch.Tensor,
        memory: torch.Tensor,
        key_valid: Optional[torch.Tensor] = None,
        query_valid: Optional[torch.Tensor] = None,
        query_pos: Optional[torch.Tensor] = None,
        memory_pos: Optional[torch.Tensor] = None,
        rope_axes_dims: Optional[List[int]] = None,
    ) -> torch.Tensor:
        bsz, q_len, _ = query.shape
        k_len = memory.shape[1]
        q = self.q(query).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k, v = self.kv(memory).chunk(2, dim=-1)
        k = k.view(bsz, k_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, k_len, self.num_heads, self.head_dim).transpose(1, 2)
        q = self.q_norm(q)
        k = self.k_norm(k)

        if query_pos is not None and memory_pos is not None and rope_axes_dims is not None:
            q_cos, q_sin = _rope_cos_sin(query_pos, self.head_dim, rope_axes_dims)
            k_cos, k_sin = _rope_cos_sin(memory_pos, self.head_dim, rope_axes_dims)
            q = _apply_rope(q, q_cos, q_sin)
            k = _apply_rope(k, k_cos, k_sin)

        out = _attention(
            q, k, v,
            key_valid=key_valid,
            dropout_p=self.dropout if self.training else 0.0,
        )
        out = out.transpose(1, 2).contiguous().view(bsz, q_len, self.hidden_size)
        out = self.out(out)
        if query_valid is not None:
            out = out * query_valid[:, :, None].to(out.dtype)
        return out


class DoubleStreamBlock(nn.Module):
    """Joint text-motion attention with separate stream updates."""

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.motion_mod = AdaLNModulation(hidden_size, num=2)
        self.text_mod = AdaLNModulation(hidden_size, num=2)
        self.motion_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.text_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.joint_attn = MultiHeadAttention(hidden_size, num_heads, dropout)
        self.motion_ffn_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.text_ffn_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.motion_ffn = nn.Sequential(SwiGLU(hidden_size, mlp_hidden), nn.Dropout(dropout))
        self.text_ffn = nn.Sequential(SwiGLU(hidden_size, mlp_hidden), nn.Dropout(dropout))

    def forward(
        self,
        motion: torch.Tensor,
        text: torch.Tensor,
        cond: torch.Tensor,
        motion_valid: torch.Tensor,
        text_valid: torch.Tensor,
        pos_ids: torch.Tensor,
        rope_axes_dims: List[int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        m_attn, m_ff = self.motion_mod(cond)
        t_attn, t_ff = self.text_mod(cond)

        motion_in = self.motion_norm(motion)
        motion_in = (1 + m_attn.scale) * motion_in + m_attn.shift
        text_in = self.text_norm(text)
        text_in = (1 + t_attn.scale) * text_in + t_attn.shift

        joint = torch.cat([motion_in, text_in], dim=1)
        joint_valid = torch.cat([motion_valid, text_valid], dim=1)
        text_pos = torch.zeros(
            text.shape[0], text.shape[1], pos_ids.shape[-1],
            device=pos_ids.device,
            dtype=pos_ids.dtype,
        )
        joint_pos = torch.cat([pos_ids, text_pos], dim=1)
        joint_out = self.joint_attn(
            joint,
            joint,
            key_valid=joint_valid,
            query_valid=joint_valid,
            query_pos=joint_pos,
            memory_pos=joint_pos,
            rope_axes_dims=rope_axes_dims,
        )
        motion_out, text_out = joint_out.split([motion.shape[1], text.shape[1]], dim=1)
        motion = motion + m_attn.gate * motion_out
        text = text + t_attn.gate * text_out

        motion_ff = self.motion_ffn_norm(motion)
        motion_ff = (1 + m_ff.scale) * motion_ff + m_ff.shift
        motion = motion + m_ff.gate * self.motion_ffn(motion_ff)

        text_ff = self.text_ffn_norm(text)
        text_ff = (1 + t_ff.scale) * text_ff + t_ff.shift
        text = text + t_ff.gate * self.text_ffn(text_ff)
        return motion, text


class SingleStreamBlock(nn.Module):
    """Single-stream DiT block over concatenated motion and text tokens."""

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mod = AdaLNModulation(hidden_size, num=2)
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = MultiHeadAttention(hidden_size, num_heads, dropout)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.ffn = nn.Sequential(SwiGLU(hidden_size, mlp_hidden), nn.Dropout(dropout))

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        valid: torch.Tensor,
        pos_ids: torch.Tensor,
        rope_axes_dims: List[int],
    ) -> torch.Tensor:
        attn_mod, ffn_mod = self.mod(cond)
        h = self.norm1(x)
        h = (1 + attn_mod.scale) * h + attn_mod.shift
        x = x + attn_mod.gate * self.attn(
            h,
            h,
            key_valid=valid,
            query_valid=valid,
            query_pos=pos_ids,
            memory_pos=pos_ids,
            rope_axes_dims=rope_axes_dims,
        )
        h = self.norm2(x)
        h = (1 + ffn_mod.scale) * h + ffn_mod.shift
        x = x + ffn_mod.gate * self.ffn(h)
        return x
