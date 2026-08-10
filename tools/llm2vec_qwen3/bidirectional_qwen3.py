"""Bidirectional Qwen3 for the LLM2Vec recipe, written against transformers >= 4.51.

The official ``llm2vec`` package cannot be used here: it pins
``transformers<=4.44.2`` (it subclasses the per-backend attention classes that
were removed in 4.48), while Qwen3 modelling code only exists from 4.51.  This
module therefore ports the *method*, not the code.

Modern transformers funnels all mask construction through
``Qwen3Model._update_causal_mask``, so a single override is enough to make the
trunk bidirectional -- no attention subclassing required.

Two silent-failure paths must be closed, and both are handled here:

1. SDPA computes ``is_causal = q_len > 1 and mask is None`` when the caller
   passes ``is_causal=None``.  The stock ``_update_causal_mask`` may return
   ``None`` (via ``_ignore_causal_mask_sdpa``) when there is nothing to mask,
   which would silently restore causal attention.  The override therefore
   *always* returns a dense 4-D mask, never ``None``.
2. FlashAttention-2 ignores the 4-D mask and reads ``module.is_causal``.  It is
   rejected outright, and ``is_causal=False`` is additionally stamped on every
   attention module as a second line of defence.

Neither guard alone is sufficient; ``assert_bidirectional`` verifies the result
empirically rather than trusting either.
"""

from __future__ import annotations

from typing import Optional, Union

import torch
import torch.nn as nn
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3ForCausalLM,
    Qwen3Model,
    Qwen3PreTrainedModel,
)

SUPPORTED_ATTENTIONS = ("sdpa", "eager")


def _bidirectional_mask(
    attention_mask: Optional[torch.Tensor],
    input_tensor: torch.Tensor,
    past_key_values=None,
) -> torch.Tensor:
    """Build a dense additive mask that hides padding keys and nothing else.

    Returns ``[batch, 1, q_len, kv_len]`` with ``0`` on attended positions and
    ``finfo.min`` on padded keys.  Only *keys* are masked, so every query row --
    including rows at padded positions -- still attends to at least one real
    token and softmax cannot produce NaN.
    """
    dtype = input_tensor.dtype
    device = input_tensor.device
    min_dtype = torch.finfo(dtype).min
    batch, q_len = input_tensor.shape[0], input_tensor.shape[1]

    past_length = 0
    if past_key_values is not None:
        seen = past_key_values.get_seq_length()
        past_length = int(seen) if seen is not None else 0
    kv_len = past_length + q_len

    mask = torch.zeros(batch, 1, q_len, kv_len, dtype=dtype, device=device)
    if attention_mask is None:
        return mask
    if attention_mask.dim() == 4:
        # A 4-D mask was supplied explicitly; honour it untouched.
        return attention_mask.to(dtype=dtype, device=device)

    padding = attention_mask.to(device=device)
    if padding.shape[-1] < kv_len:
        pad_width = kv_len - padding.shape[-1]
        padding = torch.cat(
            [padding.new_ones(batch, pad_width), padding], dim=-1
        )
    padding = padding[:, :kv_len]
    return mask.masked_fill(padding[:, None, None, :] == 0, min_dtype)


def _assert_override_is_reachable() -> None:
    """Fail loudly if ``Qwen3Model.forward`` no longer calls our override.

    Everything here rests on one assumption: that ``forward`` builds its mask by
    calling ``self._update_causal_mask``.  Later transformers releases moved
    several architectures to a module-level ``create_causal_mask(...)`` helper
    instead.  Under that layout the override below is simply never called --
    the model loads, trains, and produces finite features, but attention is
    causal again and every cached vector is wrong with nothing to show for it.
    Checking the source at import time turns that silent regression into an
    error before a single GPU-hour is spent.
    """
    import inspect

    try:
        source = inspect.getsource(Qwen3Model.forward)
    except (OSError, TypeError):  # zipped or compiled install: cannot verify
        return
    if "_update_causal_mask" not in source:
        raise RuntimeError(
            "transformers "
            f"{__import__('transformers').__version__} builds the Qwen3 attention mask "
            "without calling Qwen3Model._update_causal_mask, so the bidirectional override "
            "in this module would be dead code and attention would stay CAUSAL. Port the "
            "override to whatever mask hook this version uses before continuing."
        )


_assert_override_is_reachable()


class Qwen3BiModel(Qwen3Model):
    """Qwen3 trunk with the causal triangle removed."""

    # Incremented on every call; ``assert_mask_override_fired`` reads it to
    # confirm the hook actually ran during a real forward pass, which the
    # import-time source check cannot prove on its own.
    bidirectional_mask_calls: int = 0

    def _update_causal_mask(
        self,
        attention_mask: Union[torch.Tensor, None],
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values=None,
        output_attentions: bool = False,
    ) -> torch.Tensor:
        implementation = getattr(self.config, "_attn_implementation", "sdpa")
        if implementation not in SUPPORTED_ATTENTIONS:
            raise ValueError(
                f"Bidirectional Qwen3 requires attn_implementation in {SUPPORTED_ATTENTIONS}, "
                f"got {implementation!r}. FlashAttention-2 bypasses the 4-D mask and would "
                "silently keep causal attention."
            )
        type(self).bidirectional_mask_calls += 1
        return _bidirectional_mask(attention_mask, input_tensor, past_key_values)


class Qwen3BiForMNTP(Qwen3ForCausalLM):
    """Bidirectional Qwen3 with the LM head, used for MNTP training."""

    def __init__(self, config):
        Qwen3PreTrainedModel.__init__(self, config)
        self.model = Qwen3BiModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    # PEFT wraps/unwraps the trunk only, leaving the LM head untouched.
    def get_model_for_peft(self):
        return self.model

    def set_model_for_peft(self, model) -> None:
        self.model = model

    def save_peft_model(self, path: str) -> None:
        self.model.save_pretrained(path)


def assert_mask_override_fired(minimum: int = 1) -> int:
    """Confirm the bidirectional mask hook ran during real forward passes.

    The import-time source check proves the hook is *reachable*; this proves it
    was actually *reached*, which also covers the case where PEFT or a wrapper
    swapped the trunk for a stock ``Qwen3Model``.
    """
    calls = int(Qwen3BiModel.bidirectional_mask_calls)
    if calls < minimum:
        raise RuntimeError(
            f"The bidirectional mask override ran {calls} times (expected >= {minimum}). "
            "Attention was causal for these forward passes; any features produced are invalid."
        )
    return calls


def disable_module_causal_flags(model: nn.Module) -> int:
    """Stamp ``is_causal=False`` on every attention module (defence in depth)."""
    touched = 0
    for module in model.modules():
        if hasattr(module, "is_causal") and module.is_causal is not False:
            module.is_causal = False
            touched += 1
    return touched


@torch.no_grad()
def assert_bidirectional(model: nn.Module, tokenizer, device=None, threshold: float = 0.01) -> float:
    """Empirically verify bidirectionality; raise if attention is still causal.

    Encodes a prefix twice, once alone and once followed by extra text.  Under
    causal attention the prefix states are identical (right context is
    invisible), so the relative change is ~0.  Returns that relative change.

    MUST be run in float32.  Measured on Qwen3-8B, the causal noise floor is
    ~3e-6 in fp32 but 1.3e-2 to 5.8e-2 in bfloat16 -- differing sequence
    lengths change matmul tiling, and that noise alone exceeds this threshold,
    so a bf16 check cannot tell causal from bidirectional.
    """
    device = device or next(model.parameters()).device
    param_dtype = next(model.parameters()).dtype
    if param_dtype != torch.float32:
        raise ValueError(
            f"assert_bidirectional requires a float32 model; got {param_dtype}. "
            "Reduced precision noise (~1e-2 relative) swamps the signal."
        )
    trunk = getattr(model, "model", model)
    short = tokenizer("a person walks forward", return_tensors="pt")
    long = tokenizer(
        "a person walks forward and then suddenly stops to sit down on the floor",
        return_tensors="pt",
    )
    h_short = trunk(**{k: v.to(device) for k, v in short.items()}).last_hidden_state
    h_long = trunk(**{k: v.to(device) for k, v in long.items()}).last_hidden_state

    n = h_short.shape[1]
    delta = (h_short[0, :n].float() - h_long[0, :n].float()).abs().mean().item()
    scale = h_short[0, :n].float().abs().mean().item()
    ratio = delta / max(scale, 1e-9)
    if ratio < threshold:
        raise RuntimeError(
            f"Bidirectionality check FAILED: prefix states moved by only {ratio:.6f} "
            f"of feature scale (threshold {threshold}). Attention is still causal."
        )
    return ratio
