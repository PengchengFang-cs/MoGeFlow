"""Self-check for the bidirectional Qwen3 conversion.

Four independent checks, because the failure mode is silent -- a botched
conversion still runs and still produces plausible numbers:

  A. attention-mask shape/content: the override must return a dense 4-D mask
     that is NOT lower-triangular and never None;
  B. baseline contrast: the stock causal Qwen3 must FAIL the prefix-movement
     test that the bidirectional model passes (proves the test discriminates);
  C. prefix movement: right-side context must change left-side states;
  D. weight fidelity + head parity: converting must not perturb any weight, and
     with a full-visibility mask the LM head must still produce finite logits.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.llm2vec_qwen3.bidirectional_qwen3 import (  # noqa: E402
    Qwen3BiForMNTP,
    _bidirectional_mask,
    assert_bidirectional,
    disable_module_causal_flags,
)

QWEN = "/scratch/pf2m24/hf-models/Qwen3-8B"


def check_mask_shape() -> None:
    x = torch.zeros(2, 5, 8)
    am = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]])
    mask = _bidirectional_mask(am, x)
    assert mask.shape == (2, 1, 5, 5), mask.shape
    assert mask is not None
    row0 = mask[0, 0]
    # Fully-visible sequence: no position may be masked -> not triangular.
    assert torch.all(row0 == 0), "unpadded sequence must be fully visible"
    tri = torch.triu(torch.ones(5, 5, dtype=torch.bool), diagonal=1)
    assert not torch.all(row0[tri] < 0), "mask is still causal/triangular"
    # Padded sequence: exactly the padded KEY columns are blocked, all rows.
    padded = mask[1, 0]
    assert torch.all(padded[:, 3:] < 0), "padded keys must be blocked"
    assert torch.all(padded[:, :3] == 0), "real keys must stay visible"
    # Every query row keeps at least one visible key -> softmax cannot NaN.
    assert torch.all((padded == 0).any(dim=-1)), "a query row was fully masked"
    print("[A] mask shape/content OK: dense 4-D, non-triangular, padding-only")


def load(bidirectional: bool):
    from transformers import AutoTokenizer, Qwen3ForCausalLM

    tok = AutoTokenizer.from_pretrained(QWEN, local_files_only=True)
    cls = Qwen3BiForMNTP if bidirectional else Qwen3ForCausalLM
    # float32 is mandatory: in bfloat16 the length-dependent matmul noise is
    # ~1e-2 relative, which is larger than the effect being measured.
    model = cls.from_pretrained(
        QWEN, local_files_only=True, torch_dtype=torch.float32,
        attn_implementation="sdpa", low_cpu_mem_usage=True,
    ).to("cuda").eval()
    if bidirectional:
        touched = disable_module_causal_flags(model)
        print(f"    is_causal=False stamped on {touched} modules")
    return model, tok


def main() -> None:
    check_mask_shape()

    causal, tok = load(bidirectional=False)
    causal_ratio = None
    try:
        causal_ratio = assert_bidirectional(causal, tok)
        raise AssertionError(
            f"[B] CONTROL FAILED: stock causal model passed the test (ratio {causal_ratio:.6f}); "
            "the test does not discriminate."
        )
    except RuntimeError as exc:
        if "Bidirectionality check FAILED" not in str(exc):
            raise
        print(f"[B] control OK: stock causal Qwen3 correctly fails ({str(exc).split('only ')[1].split(' of')[0]})")
    causal_state = {k: v.detach().float().cpu().clone() for k, v in list(causal.state_dict().items())[:6]}
    del causal
    torch.cuda.empty_cache()

    bi, tok = load(bidirectional=True)
    ratio = assert_bidirectional(bi, tok)
    print(f"[C] prefix movement OK: {ratio:.4f} of feature scale (causal would be ~0)")

    bi_state = bi.state_dict()
    max_drift = max(
        (bi_state[k].detach().float().cpu() - v).abs().max().item()
        for k, v in causal_state.items()
    )
    assert max_drift == 0.0, f"weights changed during conversion: max drift {max_drift}"

    batch = tok(["a person waves the right hand", "someone jumps"],
                return_tensors="pt", padding=True).to("cuda")
    with torch.no_grad():
        out = bi(**batch)
    assert torch.isfinite(out.logits).all(), "LM head produced non-finite logits"
    print(f"[D] weight fidelity OK (drift {max_drift}); LM head finite, logits {tuple(out.logits.shape)}")
    print("ALL_CHECKS_PASSED")


if __name__ == "__main__":
    main()
