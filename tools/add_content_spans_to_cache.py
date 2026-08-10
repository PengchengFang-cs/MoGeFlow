#!/usr/bin/env python3
"""Add ``spans.npy`` to an existing LLM2Vec caption cache.

``--text_refiner_pool content_span`` needs to know which positions of each
cached sequence hold the caption rather than the chat wrapper.  That is a
property of the *tokenizer*, not of the encoder, so it can be recovered for a
cache that was already built -- no GPU, no model weights, minutes not hours.

Only the two LLM2Vec recipes are supported, because both wrap text as
``PREFIX + caption + SUFFIX`` and pool the trailing tokens, which is what a
single span per caption can express.  ``momask_qwen3_normavg_cache`` renders
the caption through Qwen's chat template and pads on the right, so its content
sits between wrapper tokens; a trailing span would be silently wrong there and
this tool refuses it.

The stored sequence lengths are re-derived and compared against ``offsets.npy``
before anything is written: if the tokenizer available now disagrees with the
one that built the cache, the spans would point at the wrong tokens, and that
is exactly the class of error that produces plausible-looking garbage.

Usage:
    python tools/add_content_spans_to_cache.py <cache_dir> [--dry_run]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SUPPORTED_RECIPES = ("llm2vec", "llm2vec_qwen3")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("cache_dir")
    p.add_argument("--dry_run", action="store_true",
                   help="Compute and report spans without writing anything.")
    return p.parse_args()


def read_captions(path: Path) -> list:
    captions = []
    with path.open("r", encoding="utf-8") as handle:
        for expected_id, line in enumerate(handle):
            payload = json.loads(line)
            if int(payload["id"]) != expected_id:
                raise RuntimeError(f"Non-contiguous caption id in {path}: {payload['id']}")
            captions.append(str(payload["text"]))
    return captions


def spans_for_llama(base_model: str, captions: list, max_length: int):
    from transformers import AutoTokenizer

    from tools.precompute_llm2vec_text_cache import content_span, prepare_text

    tokenizer = AutoTokenizer.from_pretrained(base_model, local_files_only=True)
    spans = np.zeros(len(captions), dtype=np.int64)
    lengths = np.zeros(len(captions), dtype=np.int64)
    for idx, caption in enumerate(captions):
        ids = tokenizer(prepare_text(caption), add_special_tokens=True)["input_ids"]
        lengths[idx] = min(len(ids), max_length)
        spans[idx] = content_span(tokenizer, caption, max_length)
        if idx == 0 or (idx + 1) % 8192 == 0 or idx + 1 == len(captions):
            print(f"SPANS {idx + 1}/{len(captions)}", flush=True)
    return spans, lengths


def spans_for_qwen3(base_model: str, captions: list, max_length: int):
    import torch.nn as nn
    from transformers import AutoTokenizer

    from tools.llm2vec_qwen3.supervised_encoder import (
        SEPARATOR,
        Qwen3SentenceEncoder,
        prepare_for_tokenization,
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model, local_files_only=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # tokenize() never touches the trunk, so a placeholder keeps this CPU-only
    # while still going through the exact code path the builder used.
    encoder = Qwen3SentenceEncoder(nn.Identity(), tokenizer, max_length=max_length)

    spans = np.zeros(len(captions), dtype=np.int64)
    lengths = np.zeros(len(captions), dtype=np.int64)
    for idx, caption in enumerate(captions):
        batch = encoder.tokenize([prepare_for_tokenization("" + SEPARATOR + caption)])
        lengths[idx] = int(batch["input_ids"].shape[1])
        spans[idx] = int(batch["embed_mask"].sum().item())
        if idx == 0 or (idx + 1) % 8192 == 0 or idx + 1 == len(captions):
            print(f"SPANS {idx + 1}/{len(captions)}", flush=True)
    return spans, lengths


def main() -> None:
    args = parse_args()
    root = Path(args.cache_dir).expanduser().resolve()
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    status = manifest.get("status")
    # "building" is accepted deliberately: an interrupted build is precisely the
    # case where --resume refuses for want of spans.npy and points here, and the
    # spans depend only on the captions and the tokenizer -- both final from the
    # moment captions.jsonl and offsets.npy are written, before any encoding.
    if status not in ("complete", "building"):
        raise SystemExit(f"cache status is {status!r}; expected 'complete' or 'building': {root}")

    # Non-LLM2Vec caches carry no "encoder" block at all, so read it defensively:
    # they must reach the recipe check below and be refused by name, not crash on
    # a KeyError that reads like a bug in the cache.
    cfg = manifest.get("encoder", {})
    recipe = cfg.get("recipe") or manifest.get("format")
    if recipe not in SUPPORTED_RECIPES:
        raise SystemExit(
            f"recipe {recipe!r} is not supported; only {SUPPORTED_RECIPES} place the caption "
            "at the end of the sequence, which is what a single span per caption can express"
        )

    files = manifest.get("files", {})
    spans_path = root / files.get("spans", "spans.npy")
    if spans_path.is_file():
        print(f"ALREADY_PRESENT {spans_path}")
        return

    captions = read_captions(root / files.get("captions", "captions.jsonl"))
    offsets = np.load(root / files.get("offsets", "offsets.npy"), allow_pickle=False)
    stored_lengths = np.diff(offsets)
    max_length = int(cfg["max_length"])

    builder = spans_for_llama if recipe == "llm2vec" else spans_for_qwen3
    spans, lengths = builder(cfg["base_model"], captions, max_length)

    mismatched = int((lengths != stored_lengths).sum())
    if mismatched:
        first = int(np.argmax(lengths != stored_lengths))
        raise SystemExit(
            f"{mismatched} captions tokenize to a different length than the cache records "
            f"(first at id {first}: now {int(lengths[first])}, cached {int(stored_lengths[first])}). "
            "The tokenizer does not match the one that built this cache, so any spans derived "
            "here would point at the wrong tokens. Rebuild the cache instead."
        )
    if int(spans.min()) < 1:
        raise SystemExit("computed a non-positive content span")
    over = int((spans > stored_lengths).sum())
    if over:
        raise SystemExit(f"{over} spans exceed their stored sequence length")

    fraction = float((spans / np.maximum(stored_lengths, 1)).mean())
    degraded = int((spans >= stored_lengths).sum())
    print(f"SPANS_OK status={status} n={len(spans)} mean_span={float(spans.mean()):.2f} "
          f"mean_content_fraction={fraction:.3f} "
          f"(wrapper is {100 * (1 - fraction):.1f}% of the average sequence)")
    if degraded:
        print(f"WARNING {degraded} truncated captions pool over the whole sequence, wrapper included")
    if args.dry_run:
        print("DRY_RUN no files written")
        return

    np.save(spans_path, spans, allow_pickle=False)
    files["spans"] = spans_path.name
    manifest["files"] = files
    # Written only after spans.npy exists: a manifest that advertises a missing
    # file would make the cache unloadable rather than merely unmigrated.
    tmp = manifest_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    tmp.replace(manifest_path)
    print(f"WROTE {spans_path}")


if __name__ == "__main__":
    main()
