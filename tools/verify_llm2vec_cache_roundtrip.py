#!/usr/bin/env python3
"""Round-trip check: cached features must equal a live re-encode.

This catches, in one shot, every class of cache bug we have actually hit --
wrong text format, separator leaking into the model input, pooling over the
wrong span, row/offset misalignment.  Any of those still yields finite
features and a training run that converges, so nothing else surfaces them.

Usage:
    python tools/verify_llm2vec_cache_roundtrip.py <cache_dir> [--samples 8]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("cache_dir")
    p.add_argument("--samples", type=int, default=8)
    p.add_argument("--tol", type=float, default=0.02, help="informational relative-L2 bound")
    p.add_argument("--min_cos", type=float, default=0.999, help="decision metric: minimum cosine")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.cache_dir).resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    enc_cfg = manifest["encoder"]
    recipe = enc_cfg.get("recipe", "")
    print(f"cache: {root.name}\n  recipe={recipe} text_format={enc_cfg.get('text_format', '(未记录)')}")

    from models.codeflow.text_encoder import CachedLLM2VecTextEncoder

    cached = CachedLLM2VecTextEncoder(str(root), use_pooled=True)
    captions = [t for t in cached._caption_to_id if t][: args.samples]
    if not captions:
        raise RuntimeError("cache holds no non-empty caption")

    if recipe == "llm2vec_qwen3":
        from peft import PeftModel
        from transformers import AutoTokenizer

        from tools.llm2vec_qwen3.bidirectional_qwen3 import (
            Qwen3BiForMNTP,
            disable_module_causal_flags,
        )
        from tools.llm2vec_qwen3.supervised_encoder import (
            SEPARATOR,
            Qwen3SentenceEncoder,
            prepare_for_tokenization,
        )

        tok = AutoTokenizer.from_pretrained(enc_cfg["base_model"], local_files_only=True)
        tok.padding_side = "left"
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = Qwen3BiForMNTP.from_pretrained(
            enc_cfg["base_model"], local_files_only=True, torch_dtype=torch.bfloat16,
            attn_implementation="sdpa", low_cpu_mem_usage=True,
        )
        disable_module_causal_flags(model)
        trunk = model.model
        for path in [enc_cfg["mntp_model"]] + ([enc_cfg["supervised_model"]] if enc_cfg.get("supervised_model") else []):
            trunk = PeftModel.from_pretrained(trunk, path); trunk = trunk.merge_and_unload()
        live = Qwen3SentenceEncoder(trunk, tok, max_length=int(enc_cfg["max_length"])).cuda().eval()

        # A cache built with --token_feature normavg stores the layer average,
        # not the final hidden state.  Re-encoding only the final state would
        # make this check report cosine ~0.27 on a perfectly good cache, and its
        # failure would be indistinguishable from the row-misalignment and
        # wrong-text-format corruption it exists to catch.
        token_feature = str(enc_cfg.get("token_feature", "last"))
        if token_feature == "normavg":
            from tools.precompute_qwen_normavg_text_cache import normavg_hidden
            layer_eps = enc_cfg.get("layer_norm_eps")
            if layer_eps is None:
                raise SystemExit(
                    "cache declares token_feature=normavg but records no layer_norm_eps; "
                    "its token rows cannot be reproduced, so this check cannot run"
                )
            n_layers = int(live.model.config.num_hidden_layers)
            print(f"  reproducing token rows with the normavg recipe "
                  f"(layers={n_layers}, eps={float(layer_eps)})")

        def encode(caption):
            item = prepare_for_tokenization("" + SEPARATOR + caption)
            batch = {k: v.cuda() for k, v in live.tokenize([item]).items()}
            feat = {k: v for k, v in batch.items() if k != "embed_mask"}
            with torch.no_grad():
                want_layers = token_feature == "normavg"
                out = live.model(**feat, output_hidden_states=want_layers)
                h = (
                    normavg_hidden(out.hidden_states, n_layers, float(layer_eps))[0]
                    if want_layers else out.last_hidden_state[0]
                )
                p = live(batch)[0]
            return h.float().cpu(), p.float().cpu()
    else:
        from tools.precompute_llm2vec_text_cache import build_encoder, content_span, tokenize

        ns = argparse.Namespace(
            base_model=enc_cfg["base_model"], mntp_model=enc_cfg["mntp_model"],
            supervised_model=enc_cfg.get("supervised_model", ""),
            model_dtype="bfloat16", device="cuda", max_length=int(enc_cfg["max_length"]),
        )
        enc, _ = build_encoder(ns)
        tok = enc.tokenizer

        def encode(caption):
            batch = tokenize(tok, caption, ns.max_length)
            ids = batch["input_ids"].cuda(); mask = batch["attention_mask"].cuda()
            with torch.no_grad():
                h = enc.model(input_ids=ids, attention_mask=mask).last_hidden_state
                span = content_span(tok, caption, ns.max_length)
                p = h[:, -span:, :].mean(dim=1)
            return h[0].float().cpu(), p[0].float().cpu()

    # Relative L2 and cosine, not max-abs: the cache stores bf16, whose ~0.4%
    # relative precision makes a max-abs metric flag pure rounding as a bug.
    def compare(a, b):
        a = a.reshape(-1).double(); b = b.reshape(-1).double()
        l2 = ((a - b).norm() / b.norm().clamp_min(1e-12)).item()
        cos = torch.nn.functional.cosine_similarity(a[None], b[None]).item()
        return l2, cos

    worst_tok_l2 = worst_pool_l2 = 0.0
    best_tok_cos = best_pool_cos = 1.0
    for caption in captions:
        live_tok, live_pool = encode(caption)
        cond = cached([caption])
        cache_tok = cond.tokens[0].cpu()
        cache_pool = cond.pooled[0].cpu()
        if cache_tok.shape != live_tok.shape:
            raise RuntimeError(
                f"token count differs: cache {tuple(cache_tok.shape)} vs live {tuple(live_tok.shape)} "
                f"for {caption[:50]!r} -- the cache was built with a different text format"
            )
        l2, cos = compare(cache_tok, live_tok)
        worst_tok_l2 = max(worst_tok_l2, l2); best_tok_cos = min(best_tok_cos, cos)
        l2, cos = compare(cache_pool, live_pool)
        worst_pool_l2 = max(worst_pool_l2, l2); best_pool_cos = min(best_pool_cos, cos)

    print(f"  checked {len(captions)} captions")
    print(f"  tokens : rel-L2 {worst_tok_l2:.5f}  min cos {best_tok_cos:.6f}")
    print(f"  pooled : rel-L2 {worst_pool_l2:.5f}  min cos {best_pool_cos:.6f}")
    # Cosine is the decision metric: bf16 storage plus a bf16 forward pass move
    # the norm by a couple of percent run to run, but they do not rotate the
    # vector.  A format, span or ordering bug does -- and a token-count bug is
    # already caught by the shape check above.
    if best_tok_cos < args.min_cos or best_pool_cos < args.min_cos:
        raise RuntimeError(
            f"round-trip cosine below {args.min_cos}: the cache and the live encoder "
            "disagree on more than numerical precision"
        )
    print("ROUNDTRIP_OK")


if __name__ == "__main__":
    main()
