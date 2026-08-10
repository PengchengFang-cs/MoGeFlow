#!/usr/bin/env python3
"""Precompute a Qwen3 LLM2Vec caption cache (per-token features + pooled).

Writes the same ``momask_llm2vec_cache`` layout as the Llama builder, so the
existing ``llm2vec_cache`` / ``llm2vec_pooled_cache`` providers read it
unchanged.  Only the encoder differs: our own bidirectional Qwen3 with the
MNTP adapter and, optionally, the supervised adapter merged on top.

Text is wrapped in the Qwen3 ChatML user turn, matching how the encoder was
trained -- feeding raw text would be off-distribution.  The LLM2Vec separator
is placed to mark where the content begins and is then *stripped* by
``tokenize`` before the model sees anything, so it never becomes input tokens;
the positions it marks are stored in ``spans.npy``.  Encoding runs at
batch_size=1 because left padding shifts RoPE positions and changes the
vectors.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.llm2vec_qwen3.bidirectional_qwen3 import (  # noqa: E402
    Qwen3BiForMNTP,
    assert_mask_override_fired,
    disable_module_causal_flags,
)
from tools.llm2vec_qwen3.supervised_encoder import (  # noqa: E402
    QWEN3_TURN_PREFIX,
    QWEN3_TURN_SUFFIX,
    SEPARATOR,
    Qwen3SentenceEncoder,
    prepare_for_tokenization,
)
from tools.precompute_qwen_normavg_text_cache import normavg_hidden  # noqa: E402
from tools.precompute_llm2vec_text_cache import (  # noqa: E402
    TEXT_CACHE_FORMAT,
    TEXT_CACHE_VERSION,
    atomic_write_json,
    caption_digest,
    collect_captions,
    read_caption_index,
    write_caption_index,
)


def assert_resume_matches(manifest: dict, args: argparse.Namespace) -> None:
    """Refuse to resume a cache that a different encoder started.

    The caption digest proves *what* was encoded, not *by which encoder*.
    Resuming with different adapters or max_length appends rows from a second
    model to rows from the first; every value stays finite and nothing
    downstream reports the mixture.
    """
    cfg = manifest.get("encoder", {})
    expected = {
        "base_model": str(Path(args.base_model).resolve()),
        "mntp_model": str(Path(args.mntp_model).resolve()),
        "supervised_model": str(Path(args.supervised_model).resolve()) if args.supervised_model else "",
        "max_length": int(args.max_length),
        # Without these two, a build resumed with a different --token_feature
        # keeps the rows already written and appends rows of the other kind,
        # then rewrites the manifest to claim whichever was requested -- and the
        # semantic fingerprint follows the manifest, so the chimera is
        # indistinguishable from a clean cache at every later check.
        "token_feature": str(args.token_feature),
        # Compared only for normavg: it is the eps of the parameter-free
        # LayerNorm that recipe applies, and is meaningless for final-hidden
        # -state features.
        "layer_norm_eps": float(args.layer_norm_eps) if args.token_feature == "normavg" else None,
    }
    # Caches built before --token_feature existed stored the final hidden state,
    # which is exactly what "last" now names; absence is that value, not a
    # mismatch, and those caches carry no eps either.
    observed = dict(cfg)
    observed.setdefault("token_feature", "last")
    if observed.get("token_feature") != "normavg":
        observed["layer_norm_eps"] = None
    mismatched = {k: (observed.get(k), v) for k, v in expected.items() if observed.get(k) != v}
    if mismatched:
        detail = "; ".join(f"{k}: cache={old!r} requested={new!r}" for k, (old, new) in mismatched.items())
        raise RuntimeError(f"Cannot resume: the cache was built by a different encoder -- {detail}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data_root", default="dataset/HumanML3D")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--base_model", default="/scratch/pf2m24/hf-models/Qwen3-8B")
    p.add_argument("--mntp_model", default="/scratch/pf2m24/hf-models/Qwen3-8B-mntp")
    p.add_argument("--supervised_model",
                   default="/scratch/pf2m24/hf-models/Qwen3-8B-mntp-supervised",
                   help="Empty string caches the MNTP-only encoder.")
    p.add_argument("--max_length", type=int, default=128)
    p.add_argument("--skip_dataset_captions", action="store_true")
    p.add_argument("--caption_file", action="append", default=[])
    p.add_argument("--include_caption", action="append", default=[])
    p.add_argument("--max_captions", type=int, default=0)
    p.add_argument(
        "--token_feature",
        default="last",
        choices=["last", "normavg"],
        help="Per-token features to cache: the final hidden state (the LLM2Vec "
             "recipe as published), or the MotionCraft qwen_normavg recipe -- a "
             "parameter-free LayerNorm of every transformer layer, averaged. "
             "The pooled sentence vector is unaffected: it stays the trained "
             "LLM2Vec pooling over the final hidden state either way.",
    )
    p.add_argument(
        "--layer_norm_eps",
        type=float,
        default=1e-5,
        help="Epsilon of the parameter-free LayerNorm used by --token_feature "
             "normavg.  Default matches the reference qwen_normavg cache, which "
             "is what makes 'same recipe, different encoder' true; the trunk's "
             "own rms_norm_eps is a different normalizer's constant.",
    )
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def build_encoder(args):
    from peft import PeftModel
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, local_files_only=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = Qwen3BiForMNTP.from_pretrained(
        args.base_model, local_files_only=True, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa", low_cpu_mem_usage=True,
    )
    disable_module_causal_flags(model)
    trunk = model.model
    applied = []
    for path in [args.mntp_model] + ([args.supervised_model] if args.supervised_model else []):
        trunk = PeftModel.from_pretrained(trunk, path)
        trunk = trunk.merge_and_unload()
        applied.append(Path(path).name)
    print(f"[setup] adapters merged: {applied}", flush=True)
    encoder = Qwen3SentenceEncoder(trunk, tokenizer, max_length=int(args.max_length)).cuda().eval()
    return encoder, applied


def main() -> None:
    args = parse_args()
    captions, source_stats = collect_captions(args)
    out = Path(args.output_dir).expanduser().resolve()
    manifest_path = out / "manifest.json"
    digest = caption_digest(captions)

    progress_path = out / "progress.json"
    resuming = False
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("captions_sha256") != digest:
            raise RuntimeError("Cannot resume: caption set changed")
        if manifest.get("status") == "complete":
            # Identity first: returning success on a complete cache built by a
            # different recipe would let a build-then-train script proceed on
            # features it did not ask for.
            assert_resume_matches(manifest, args)
            print(f"CACHE_ALREADY_COMPLETE {out}", flush=True)
            return
        if not args.resume:
            raise RuntimeError(f"Incomplete cache at {out}; pass --resume")
        if read_caption_index(out / "captions.jsonl") != captions:
            raise RuntimeError("Cannot resume: caption index changed")
        assert_resume_matches(manifest, args)
        for required in ("offsets.npy", "spans.npy", "tokens.npy", "pooled.npy", "progress.json"):
            if not (out / required).is_file():
                raise RuntimeError(
                    f"Cannot resume: {out / required} is missing. This cache was written by an "
                    "older build that could not resume; rebuild it from scratch."
                )
        resuming = True
    if out.exists() and any(out.iterdir()) and not manifest_path.is_file():
        raise RuntimeError(f"Refusing to overwrite non-empty directory: {out}")
    out.mkdir(parents=True, exist_ok=True)

    encoder, applied = build_encoder(args)
    hidden = int(encoder.width) if hasattr(encoder, "width") else int(encoder.model.config.hidden_size)

    if resuming:
        offsets = np.load(out / "offsets.npy", allow_pickle=False)
        spans = np.load(out / "spans.npy", allow_pickle=False)
        total = int(offsets[-1])
        tokens_mm = np.lib.format.open_memmap(out / "tokens.npy", mode="r+")
        pooled_mm = np.lib.format.open_memmap(out / "pooled.npy", mode="r+")
        start = int(json.loads(progress_path.read_text(encoding="utf-8")).get("count", 0))
        print(f"CACHE_RESUME from caption {start}/{len(captions)}", flush=True)
    else:
        # Pass 1: lengths and content spans, so the ragged token store can be
        # laid out up front and the model can later pool over content only.
        lengths = np.zeros(len(captions), dtype=np.int64)
        spans = np.zeros(len(captions), dtype=np.int64)
        for idx, caption in enumerate(captions):
            item = prepare_for_tokenization("" + SEPARATOR + caption)
            batch = encoder.tokenize([item])
            lengths[idx] = int(batch["input_ids"].shape[1])
            # embed_mask is what forward() pools over, so it *is* the span.
            spans[idx] = int(batch["embed_mask"].sum().item())
            if idx == 0 or (idx + 1) % 8192 == 0 or idx + 1 == len(captions):
                print(f"TOKEN_LENGTHS {idx + 1}/{len(captions)}", flush=True)
        if int(spans.min()) < 1:
            raise RuntimeError("embed_mask produced an empty content span")
        over = int((spans > lengths).sum())
        if over:
            raise RuntimeError(f"{over} captions have a content span longer than their sequence")
        # Truncation clamps the span to max_length, which then equals the stored
        # length: "pool the caption span" silently becomes "pool everything,
        # wrapper included".  Count and report it -- it cannot be fixed here.
        degraded = int((spans >= lengths).sum())
        if degraded:
            print(
                f"WARNING content_span degenerates to whole-sequence pooling for {degraded}/"
                f"{len(captions)} captions truncated at max_length={int(args.max_length)}",
                flush=True,
            )
        offsets = np.zeros(len(captions) + 1, dtype=np.int64)
        offsets[1:] = np.cumsum(lengths)
        total = int(offsets[-1])

        write_caption_index(out / "captions.jsonl", captions)
        np.save(out / "offsets.npy", offsets, allow_pickle=False)
        np.save(out / "spans.npy", spans, allow_pickle=False)
        tokens_mm = np.lib.format.open_memmap(out / "tokens.npy", mode="w+",
                                              dtype=np.uint16, shape=(total, hidden))
        pooled_mm = np.lib.format.open_memmap(out / "pooled.npy", mode="w+",
                                              dtype=np.uint16, shape=(len(captions), hidden))
        start = 0
        print(f"CACHE_PLAN captions={len(captions)} total_tokens={total} "
              f"tokens_gib={total * hidden * 2 / 2**30:.2f} "
              f"content_span_mean={float(spans.mean()):.2f} "
              f"content_fraction={float((spans / np.maximum(lengths, 1)).mean()):.3f}", flush=True)

    manifest = {
        "format": TEXT_CACHE_FORMAT,
        "version": TEXT_CACHE_VERSION,
        "status": "building",
        "num_captions": len(captions),
        "captions_sha256": digest,
        "source": source_stats,
        "build": {
            "transformers_version": __import__("transformers").__version__,
            "torch_version": torch.__version__,
            "model_dtype": "bfloat16",
            "encoder_batch_size": 1,
        },
        "storage_dtype": {"tokens": "bfloat16", "numpy": "uint16"},
        "encoder": {
            "recipe": "llm2vec_qwen3",
            "base_model": str(Path(args.base_model).resolve()),
            "mntp_model": str(Path(args.mntp_model).resolve()),
            "supervised_model": str(Path(args.supervised_model).resolve()) if args.supervised_model else "",
            "adapters": applied,
            "hidden_size": hidden,
            "feature_layer": (
                "last_hidden_state_bidirectional" if args.token_feature == "last"
                else "all_transformer_layers_layernorm_mean_bidirectional"
            ),
            "pooling": "masked_mean_over_content_span",
            "instruction": "",
            # The model input is PREFIX + caption + SUFFIX.  The "!@#$%^&*()"
            # separator is a position marker that tokenize() strips via
            # "".join(parts); it is never tokenized, so naming it here (as an
            # earlier revision did with "..._with_separator") misdescribes what
            # was actually encoded.
            "text_format": "qwen3_chatml_user_turn",
            "text_format_detail": {
                "prefix": QWEN3_TURN_PREFIX,
                "suffix": QWEN3_TURN_SUFFIX,
                "separator_in_model_input": False,
                "pooling_span": "trailing embed_mask tokens (caption + suffix)",
            },
            "token_feature": args.token_feature,
            "layer_norm_eps": float(args.layer_norm_eps) if args.token_feature == "normavg" else None,
            "max_length": int(args.max_length),
            "total_tokens": total,
        },
        "files": {"captions": "captions.jsonl", "tokens": "tokens.npy",
                  "offsets": "offsets.npy", "pooled": "pooled.npy",
                  "spans": "spans.npy"},
    }
    atomic_write_json(manifest_path, manifest)
    if not resuming:
        atomic_write_json(progress_path, {"count": 0})

    if resuming and start >= len(captions):
        # Interrupted after the last progress write but before the completion
        # marker.  The loop would run zero times, so neither the bidirectionality
        # sentinel nor the pooling check would fire, and the manifest below would
        # stamp whatever --token_feature was requested onto rows produced by the
        # previous one.
        raise RuntimeError(
            f"Cannot resume: progress already covers all {len(captions)} captions but the "
            "cache was never marked complete, so its rows cannot be re-verified. "
            "Delete the directory and rebuild."
        )

    trunk_cfg = encoder.model.config
    num_layers = int(trunk_cfg.num_hidden_layers)
    layer_norm_eps = float(args.layer_norm_eps)
    print(f"[setup] token_feature={args.token_feature} layers={num_layers} eps={layer_norm_eps}", flush=True)

    with torch.no_grad():
        for idx in range(start, len(captions)):
            item = prepare_for_tokenization("" + SEPARATOR + captions[idx])
            batch = {k: v.cuda() for k, v in encoder.tokenize([item]).items()}
            feature = {k: v for k, v in batch.items() if k != "embed_mask"}
            need_layers = args.token_feature == "normavg"
            # NOT "out": that name is the output directory in this scope.
            trunk_out = encoder.model(**feature, output_hidden_states=need_layers)
            # Pooling always reads the final hidden state -- that is the tensor
            # the supervised objective shaped, and the tensor
            # Qwen3SentenceEncoder.forward pools, so the shortcut check below
            # stays exact regardless of which features are cached.
            final_states = trunk_out.last_hidden_state[0]
            token_states = (
                normavg_hidden(trunk_out.hidden_states, num_layers, layer_norm_eps)[0]
                if need_layers else final_states
            )
            span = int(batch["embed_mask"].sum().item())
            if span != int(spans[idx]):
                raise RuntimeError(
                    f"caption {idx}: embed_mask span {span} disagrees with "
                    f"stored span {int(spans[idx])}"
                )
            # Pool from the states already computed.  Calling encoder(batch)
            # re-runs the entire trunk just to average its output, doubling the
            # GPU cost of a 56k-caption build; Qwen3SentenceEncoder.forward
            # pools hidden[-embed_mask.sum():].mean(0), which is exactly this.
            pooled = final_states[-span:].mean(dim=0)
            lo, hi = int(offsets[idx]), int(offsets[idx + 1])
            if idx == start:
                # If the mask override never fired, attention was causal and
                # every feature here is worthless.
                assert_mask_override_fired(minimum=1)
                # Prove the shortcut above is exact rather than merely plausible,
                # once, against the encoder's own pooling.
                reference = encoder(batch)[0]
                delta = (pooled.float() - reference.float()).abs().max().item()
                if delta != 0.0:
                    raise RuntimeError(
                        f"pooling shortcut disagrees with Qwen3SentenceEncoder.forward "
                        f"by {delta:.3e}; refusing to build the cache"
                    )
                print("POOLING_SHORTCUT_EXACT", flush=True)
                if need_layers:
                    # Nothing else looks at token_states.  Were normavg_hidden to
                    # return the final hidden state, every other check here --
                    # pooling, finiteness, the fingerprint -- would still pass and
                    # the cache would claim a recipe it does not contain.
                    drift = (token_states.float() - final_states.float()).abs().max().item()
                    if drift == 0.0:
                        raise RuntimeError(
                            "normavg token features are bit-identical to the final "
                            "hidden state; the layer average did not take effect"
                        )
                    print(f"NORMAVG_ACTIVE max|delta_vs_last|={drift:.4f}", flush=True)
            if int(token_states.shape[0]) != hi - lo:
                raise RuntimeError(
                    f"caption {idx}: encoded {int(token_states.shape[0])} rows but the layout "
                    f"reserves {hi - lo}; a single-row mismatch would broadcast silently"
                )
            tokens_mm[lo:hi] = token_states[: hi - lo].to(torch.bfloat16).view(torch.uint16).cpu().numpy()
            pooled_mm[idx] = pooled.to(torch.bfloat16).view(torch.uint16).cpu().numpy()
            if (idx + 1) % 500 == 0 or idx + 1 == len(captions):
                tokens_mm.flush(); pooled_mm.flush()
                # Flush the data before advancing the counter: a crash between
                # the two costs a re-encode, the reverse order loses rows.
                atomic_write_json(progress_path, {"count": idx + 1})
                print(f"ENCODE {idx + 1}/{len(captions)}", flush=True)
    tokens_mm.flush(); pooled_mm.flush()
    atomic_write_json(progress_path, {"count": len(captions)})
    del tokens_mm, pooled_mm

    # Validate BEFORE advertising completion: the training workers poll for
    # status=complete, and validate_cache scans the whole token mmap, so writing
    # the marker first hands them a cache that has not passed its own checks.
    manifest["status"] = "validating"
    atomic_write_json(manifest_path, manifest)

    from models.codeflow.text_encoder import CachedLLM2VecTextEncoder

    check = CachedLLM2VecTextEncoder(str(out), use_pooled=True, require_complete=False)
    example = next((t for t in check._caption_to_id if t), None)
    cond = check(["", example] if example else [""])
    if not (torch.isfinite(cond.tokens).all() and torch.isfinite(cond.pooled).all()):
        manifest["status"] = "invalid"
        atomic_write_json(manifest_path, manifest)
        raise RuntimeError("Cache validation found non-finite features")
    manifest["status"] = "complete"
    atomic_write_json(manifest_path, manifest)
    print("CACHE_VALID " + json.dumps(check.cache_summary(), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
