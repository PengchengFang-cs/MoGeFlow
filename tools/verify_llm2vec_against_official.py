#!/usr/bin/env python3
"""Compare our cached pooled vector against the official ``LLM2Vec.encode``.

The round-trip check re-uses our own tokenize/pooling code on both sides, so it
cannot detect a bug *inside* that code -- a wrong text format matches itself.
This check is the external reference: it drives the upstream package end to
end (``encode`` applies ``prepare_for_tokenization``, splits on the separator,
builds ``embed_mask`` and pools over the content span) and compares the result
with what we stored.

Only the Llama caches can be checked this way; the upstream package has no
Qwen3 support, which is the whole reason our Qwen3 path exists.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))



def _assert_reference_has_adapters(model, base_model: str) -> None:
    """The reference must not silently be the raw base model.

    The builder has checked this since it was written; the reference side did
    not, so a reference with no adapters at all read as "the cache is wrong".
    A comparison whose baseline can be untrained is not a check.
    """
    import safetensors.torch as st

    target = next(
        (m for n, m in model.named_modules() if n.endswith("layers.0.self_attn.q_proj")), None
    )
    if target is None:
        raise SystemExit("could not locate layers.0.self_attn.q_proj on the reference model")
    base_dir = Path(base_model)
    index = json.loads((base_dir / "model.safetensors.index.json").read_text())
    key = next(k for k in index["weight_map"] if k.endswith("layers.0.self_attn.q_proj.weight"))
    with st.safe_open(str(base_dir / index["weight_map"][key]), framework="pt") as f:
        raw = f.get_tensor(key)
    delta = (target.weight.detach().float().cpu() - raw.float()).abs().max().item()
    if delta == 0.0:
        raise SystemExit(
            "the reference encoder is bit-identical to the raw base model: its LLM2Vec "
            "adapters were dropped, so any disagreement it reports is its own fault"
        )
    print(f"  reference adapters applied: max |delta| vs base q_proj = {delta:.6f}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("cache_dir")
    p.add_argument("--samples", type=int, default=8)
    p.add_argument("--min_cos", type=float, default=0.99)
    p.add_argument("--force_hub_name", default="meta-llama/Meta-Llama-3-8B-Instruct",
                   help="Set config._name_or_path so upstream applies its chat wrapper; "
                        "pass an empty string to reproduce the raw local-path behaviour.")
    args = p.parse_args()

    root = Path(args.cache_dir).resolve()
    cfg = json.loads((root / "manifest.json").read_text())["encoder"]
    if cfg.get("recipe") != "llm2vec":
        raise SystemExit(f"only the Llama recipe can be checked against upstream; got {cfg.get('recipe')}")

    from llm2vec import LLM2Vec
    from peft import PeftModel

    from models.codeflow.text_encoder import CachedLLM2VecTextEncoder

    cached = CachedLLM2VecTextEncoder(str(root), use_pooled=True)
    captions = [t for t in cached._caption_to_id if t][: args.samples]

    official = LLM2Vec.from_pretrained(
        cfg["base_model"],
        peft_model_name_or_path=cfg["mntp_model"],
        # Load-bearing, and its absence is why this check once failed at cos 0.15
        # against a correct cache: without it the MNTP LoRA stays wrapped, the
        # supervised adapter re-initialises the "default" adapter in place, its
        # own keys miss the nested module paths and are dropped silently, and
        # merge_and_unload folds in a zero delta -- leaving a trunk bit-identical
        # to raw Llama-3 masquerading as the reference encoder.
        merge_peft=True,
        device_map="cuda",
        torch_dtype=torch.bfloat16,
        pooling_mode="mean",
        max_length=int(cfg["max_length"]),
    )
    if cfg.get("supervised_model"):
        official.model = PeftModel.from_pretrained(official.model, cfg["supervised_model"])
        official.model = official.model.merge_and_unload()
    _assert_reference_has_adapters(official.model, cfg["base_model"])
    # Upstream gates the chat wrapper on config._name_or_path matching the HF
    # hub id.  Loading from a local path silently skips it -- but the encoder
    # was MNTP/supervised-trained with the hub id, i.e. WITH the wrapper, so the
    # local-path behaviour is off-distribution.  Restore the id so the reference
    # applies the same wrapper our cache uses.
    if args.force_hub_name:
        official.model.config._name_or_path = args.force_hub_name
    print(f"  reference prepare_for_tokenization -> "
          f"{official.prepare_for_tokenization('CAPTION')!r}\n")
    official.model.eval()

    worst = 1.0
    for caption in captions:
        # batch_size=1: left padding would shift RoPE positions.
        ref = torch.as_tensor(official.encode([caption], batch_size=1)[0]).float()
        ours = cached([caption]).pooled[0].float().cpu()
        cos = torch.nn.functional.cosine_similarity(ref[None], ours[None]).item()
        worst = min(worst, cos)
        print(f"  cos={cos:.6f}  {caption[:58]!r}")

    print(f"\n  worst cosine vs official encode(): {worst:.6f}")
    if worst < args.min_cos:
        raise RuntimeError(
            f"cache disagrees with the official encoder (cos {worst:.6f} < {args.min_cos}); "
            "the stored features are not what upstream would produce"
        )
    print("OFFICIAL_MATCH_OK")


if __name__ == "__main__":
    main()
