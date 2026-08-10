#!/usr/bin/env python3
"""Precompute a resumable LLM2Vec caption cache (per-token features + pooled).

The encoder is the McGill LLM2Vec recipe on Llama-3-8B: MNTP-adapted
bidirectional attention plus the supervised contrastive adapter.  Both the
per-token hidden states and the masked-mean pooled sentence vector are stored,
so one cache serves both conditioning variants:

  * token-only  -- the model reads ``tokens`` and gets ``pooled=None``;
  * token+global -- the model additionally feeds ``pooled`` into AdaLN.

Captions are encoded exactly as ``LLM2Vec.encode`` would with an empty
instruction: wrapped in the Llama-3 user turn (``prepare_for_tokenization``),
with upstream's ``!@#$%^&*()`` separator acting only as a position marker --
``tokenize`` strips it via ``"".join(parts)``, so it never reaches the model.
Pooling averages the trailing content span (caption + ``<|eot_id|>``), which is
what upstream's ``embed_mask`` selects.  Feeding raw text instead put the
encoder off-distribution and cost five runs; see the notes on ``prepare_text``.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

TEXT_CACHE_FORMAT = "momask_llm2vec_cache"
TEXT_CACHE_VERSION = 1
FEATURE_LAYER = "last_hidden_state_bidirectional"
# Part of the semantic fingerprint the model pins.  It names what encode()
# actually does -- average the caption span, not every token -- so a cache built
# before that fix and one built after cannot hash to the same value.
POOLING = "masked_mean_over_content_span"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data_root", type=str, default="dataset/HumanML3D")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--base_model",
        type=str,
        default="/scratch/pf2m24/hf-models/Meta-Llama-3-8B-Instruct",
    )
    parser.add_argument(
        "--mntp_model",
        type=str,
        default="/scratch/pf2m24/hf-models/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp",
    )
    parser.add_argument(
        "--supervised_model",
        type=str,
        default="/scratch/pf2m24/hf-models/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised",
        help="Set to an empty string to cache the MNTP-only encoder.",
    )
    parser.add_argument("--max_length", type=int, default=128)
    # encode() runs one caption at a time on purpose: left padding shifts RoPE
    # positions and changes the vectors.  The flag is kept so existing launch
    # scripts still parse, but any other value is rejected rather than silently
    # ignored and then recorded in the manifest as if it had been honoured.
    parser.add_argument("--batch_size", type=int, default=1, choices=[1])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model_dtype", type=str, default="bfloat16", choices=["bfloat16", "float16"])
    parser.add_argument(
        "--storage_dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="bfloat16 is stored losslessly as raw uint16 words in the mmap.",
    )
    parser.add_argument("--caption_file", action="append", default=[])
    parser.add_argument("--include_caption", action="append", default=[])
    parser.add_argument("--skip_dataset_captions", action="store_true")
    parser.add_argument("--max_captions", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--validate_only", action="store_true")
    return parser.parse_args()


def atomic_write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def caption_digest(captions: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for caption in captions:
        encoded = caption.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, byteorder="little", signed=False))
        digest.update(encoded)
    return digest.hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_captions(args: argparse.Namespace) -> Tuple[List[str], Dict[str, object]]:
    text_dir = Path(args.data_root).expanduser().resolve() / "texts"
    if not args.skip_dataset_captions and not text_dir.is_dir():
        raise FileNotFoundError(f"Dataset text directory not found: {text_dir}")

    captions = {""}  # Unconditional feature for standard CFG.
    source_lines = 0
    text_files = [] if args.skip_dataset_captions else sorted(text_dir.glob("*.txt"))
    for path in text_files:
        for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            source_lines += 1
            caption = raw_line.strip().split("#", 1)[0]
            if caption:
                captions.add(caption)

    extra_lines = 0
    for value in args.include_caption:
        captions.add(str(value))
        extra_lines += 1
    for filename in args.caption_file:
        path = Path(filename).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Extra caption file not found: {path}")
        for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            caption = raw_line.strip()
            if caption:
                captions.add(caption)
                extra_lines += 1

    ordered = [""] + sorted(caption for caption in captions if caption)
    if args.max_captions > 0:
        ordered = ordered[: max(1, int(args.max_captions))]

    split_contract: Dict[str, object] = {}
    data_root = Path(args.data_root).expanduser().resolve()
    for split in (() if args.skip_dataset_captions else ("train", "val", "test")):
        split_path = data_root / f"{split}.txt"
        if split_path.is_file():
            split_contract[split] = {
                "path": str(split_path),
                "records": len(split_path.read_text(encoding="utf-8").splitlines()),
                "sha256": file_sha256(split_path),
            }
    stats: Dict[str, object] = {
        "dataset_text_files": len(text_files),
        "dataset_caption_records": source_lines,
        "dataset_captions_skipped": bool(args.skip_dataset_captions),
        "extra_caption_records": extra_lines,
        "unique_captions_with_empty": len(ordered),
        "splits": split_contract,
    }
    return ordered, stats


def write_caption_index(path: Path, captions: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for idx, caption in enumerate(captions):
            handle.write(json.dumps({"id": idx, "text": caption}, ensure_ascii=False) + "\n")


def read_caption_index(path: Path) -> List[str]:
    captions: List[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for expected_id, line in enumerate(handle):
            payload = json.loads(line)
            if int(payload["id"]) != expected_id:
                raise RuntimeError(f"Invalid caption id at line {expected_id + 1} in {path}")
            captions.append(str(payload["text"]))
    return captions


def verify_adapters_applied(model, args: argparse.Namespace) -> None:
    """Fail loudly if the merged trunk still equals the raw base model.

    peft silently drops adapter weights whose module paths do not match, so a
    mis-stacked adapter produces a bit-identical base model rather than an
    error.  Comparing one projection against the untouched checkpoint is enough
    to catch it and costs a single shard read.
    """
    import safetensors.torch as st

    target = next(
        (m for n, m in model.named_modules() if n.endswith("layers.0.self_attn.q_proj")),
        None,
    )
    if target is None:
        raise RuntimeError("could not locate layers.0.self_attn.q_proj to verify adapters")

    base_dir = Path(args.base_model)
    index = json.loads((base_dir / "model.safetensors.index.json").read_text())
    key = next(k for k in index["weight_map"] if k.endswith("layers.0.self_attn.q_proj.weight"))
    shard = base_dir / index["weight_map"][key]
    with st.safe_open(str(shard), framework="pt") as f:
        raw = f.get_tensor(key)

    merged = target.weight.detach().to(raw.dtype).cpu()
    if torch.equal(merged, raw.cpu()):
        raise RuntimeError(
            "LLM2Vec adapters had no effect: the merged trunk is bit-identical to the "
            "base checkpoint. The adapter weights were silently dropped -- check that "
            "merge_peft=True and that adapters are merged one at a time."
        )
    delta = (merged.float() - raw.float().cpu()).abs().max().item()
    print(f"[verify] adapters applied: max |Δ| vs base q_proj = {delta:.6f}", flush=True)


def build_encoder(args: argparse.Namespace):
    """Load base -> MNTP LoRA (merged) -> optional supervised LoRA (merged)."""
    from llm2vec import LLM2Vec
    from peft import PeftModel

    dtype = torch.bfloat16 if args.model_dtype == "bfloat16" else torch.float16
    # merge_peft=True is load-bearing.  With the default (False) the MNTP LoRA
    # stays wrapped, and stacking the supervised adapter on that PeftModel
    # re-initialises the "default" adapter in place (destroying MNTP) while the
    # supervised checkpoint keys miss the now doubly-nested module paths and are
    # dropped into unexpected_keys, which peft does not raise on.  merge_and_unload
    # then folds in a zero delta and the trunk is bit-identical to raw Llama-3 --
    # silently, with the script still reporting the adapter as applied.
    encoder = LLM2Vec.from_pretrained(
        args.base_model,
        peft_model_name_or_path=args.mntp_model,
        merge_peft=True,
        device_map=args.device,
        torch_dtype=dtype,
        pooling_mode="mean",
        max_length=int(args.max_length),
    )
    supervised_applied = False
    if args.supervised_model:
        encoder.model = PeftModel.from_pretrained(encoder.model, args.supervised_model)
        encoder.model = encoder.model.merge_and_unload()
        supervised_applied = True

    verify_adapters_applied(encoder.model, args)
    encoder.model.eval()
    encoder.model.requires_grad_(False)
    return encoder, supervised_applied


# Kept only to document what upstream's marker is.  It is deliberately NOT used:
# LLM2Vec.tokenize splits on it and rejoins with "".join(parts), so it marks the
# content span and never reaches the tokenizer.  Feeding it verbatim costs 7 junk
# tokens on Llama-3 -- a third of a short motion caption -- which is a bug this
# builder has already had once.
UPSTREAM_SPAN_MARKER_NOT_MODEL_INPUT = "!@#$%^&*()"
LLAMA3_TURN_PREFIX = "<|start_header_id|>user<|end_header_id|>\n\n"
LLAMA3_TURN_SUFFIX = "<|eot_id|>"


def prepare_text(caption: str) -> str:
    """Wrap in the Llama-3 user turn, as ``LLM2Vec.encode`` does.

    ``llm2vec.LLM2Vec.encode`` calls ``prepare_for_tokenization`` on every
    sentence, so the encoder only ever saw chat-wrapped text during training
    and evaluation.  Feeding raw text is off-distribution.
    """
    body = caption.strip() if caption.strip() else " "
    return LLAMA3_TURN_PREFIX + body + LLAMA3_TURN_SUFFIX


def content_span(tokenizer, caption: str, max_length: int) -> int:
    """Number of trailing tokens that are the caption plus the turn suffix.

    Upstream marks the embeddable span with the ``!@#$%^&*()`` separator and
    then strips it (``"".join(parts)``) before tokenizing -- the separator is a
    position marker, never model input.  Feeding it verbatim costs 7 junk
    tokens on Llama-3, about a third of a short motion caption.  Here the span
    is computed directly instead, and pooling averages only these positions.
    """
    body = caption.strip() if caption.strip() else " "
    tail = tokenizer(body + LLAMA3_TURN_SUFFIX, add_special_tokens=False)["input_ids"]
    total = len(tokenizer(prepare_text(caption), add_special_tokens=True)["input_ids"])
    return max(1, min(len(tail), min(total, int(max_length))))


def tokenize(tokenizer, caption: str, max_length: int):
    return tokenizer(
        prepare_text(caption),
        add_special_tokens=True,
        truncation=True,
        max_length=int(max_length),
        return_attention_mask=True,
        return_tensors="pt",
    )


def compute_lengths(
    tokenizer, captions: Sequence[str], max_length: int
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    """Token counts and content spans for every caption.

    Spans are persisted rather than recomputed at encode time so that pooling
    and the ``spans.npy`` the model reads can never drift apart -- both come
    from this one call to ``content_span``.
    """
    lengths = np.zeros(len(captions), dtype=np.int64)
    spans = np.zeros(len(captions), dtype=np.int64)
    untruncated = np.zeros(len(captions), dtype=np.int64)
    for idx, caption in enumerate(captions):
        ids = tokenizer(prepare_text(caption), add_special_tokens=True)["input_ids"]
        untruncated[idx] = max(1, len(ids))
        lengths[idx] = min(len(ids), int(max_length))
        spans[idx] = content_span(tokenizer, caption, int(max_length))
        if idx == 0 or idx + 1 == len(captions) or (idx + 1) % 8192 == 0:
            print(f"TOKEN_LENGTHS {idx + 1}/{len(captions)}", flush=True)
    if int(spans.min()) < 1:
        raise RuntimeError("content_span produced a non-positive span")
    over = int((spans > lengths).sum())
    if over:
        raise RuntimeError(f"{over} captions have a content span longer than the stored sequence")
    # A caption long enough to be truncated has its span clamped to max_length,
    # which then equals the stored length -- so "pool the content span" quietly
    # becomes "pool everything, wrapper included", i.e. the bug this span
    # machinery exists to remove.  It cannot be fixed here (the prefix tokens
    # genuinely are part of the truncated sequence), so it is counted and
    # reported rather than left to be inferred from truncated_count.
    degraded = int((spans >= lengths).sum())
    if degraded:
        print(
            f"WARNING content_span degenerates to whole-sequence pooling for {degraded}/"
            f"{len(captions)} captions truncated at max_length={int(max_length)}; "
            "their pooled vectors include the chat wrapper",
            flush=True,
        )
    stats = {
        "min": int(untruncated.min()),
        "mean": float(untruncated.mean()),
        "p99": float(np.percentile(untruncated, 99)),
        "max": int(untruncated.max()),
        "truncated_count": int((untruncated > int(max_length)).sum()),
        "content_span_mean": float(spans.mean()),
        "content_span_fraction_mean": float((spans / np.maximum(lengths, 1)).mean()),
        "content_span_degenerate_count": degraded,
    }
    return lengths, spans, stats


def assert_resume_matches(manifest: Dict[str, object], args: argparse.Namespace) -> None:
    """Refuse to resume a cache that a different encoder started."""
    cfg = manifest.get("encoder", {})
    stored_storage = manifest.get("storage_dtype", {})
    expected = {
        "base_model": str(Path(args.base_model).resolve()),
        "mntp_model": str(Path(args.mntp_model).resolve()),
        "supervised_model": str(Path(args.supervised_model).resolve()) if args.supervised_model else "",
        "max_length": int(args.max_length),
    }
    mismatched = {
        key: (cfg.get(key), value)
        for key, value in expected.items()
        if cfg.get(key) != value
    }
    stored_tokens_dtype = (
        stored_storage.get("tokens") if isinstance(stored_storage, dict) else stored_storage
    )
    if stored_tokens_dtype != args.storage_dtype:
        mismatched["storage_dtype"] = (stored_tokens_dtype, args.storage_dtype)
    # model_dtype lives under "build", not "encoder", and is just as much part of
    # the encoder identity: resuming a bfloat16 build in float16 leaves the cache
    # half computed at each precision, all finite and entirely unreported.
    stored_model_dtype = (manifest.get("build") or {}).get("model_dtype")
    if stored_model_dtype != args.model_dtype:
        mismatched["model_dtype"] = (stored_model_dtype, args.model_dtype)
    if mismatched:
        detail = "; ".join(f"{k}: cache={old!r} requested={new!r}" for k, (old, new) in mismatched.items())
        raise RuntimeError(f"Cannot resume: the cache was built by a different encoder -- {detail}")


def create_or_resume_cache(args: argparse.Namespace, captions: List[str], source_stats: Dict[str, object]):
    output_dir = Path(args.output_dir).expanduser().resolve()
    manifest_path = output_dir / "manifest.json"
    progress_path = output_dir / "progress.json"
    captions_path = output_dir / "captions.jsonl"
    tokens_path = output_dir / "tokens.npy"
    offsets_path = output_dir / "offsets.npy"
    pooled_path = output_dir / "pooled.npy"
    spans_path = output_dir / "spans.npy"
    expected_digest = caption_digest(captions)

    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("captions_sha256") != expected_digest:
            raise RuntimeError("Cannot resume: caption set changed")
        if manifest.get("status") == "complete":
            return output_dir, manifest, {"count": len(captions)}
        if not args.resume:
            raise RuntimeError(f"Incomplete cache exists at {output_dir}; pass --resume")
        if read_caption_index(captions_path) != captions:
            raise RuntimeError("Cannot resume: caption index changed")
        # The caption digest proves *what* was encoded, not *by which encoder*.
        # Resuming with different adapters or a different max_length appends
        # rows from a second model to rows from the first: every value stays
        # finite, training converges, and nothing ever reports the mixture.
        assert_resume_matches(manifest, args)
        if not spans_path.is_file():
            raise RuntimeError(
                f"Cannot resume: {spans_path} is missing, so this cache predates content-span "
                "support. Rebuild it, or run tools/add_content_spans_to_cache.py first."
            )
        return output_dir, manifest, json.loads(progress_path.read_text(encoding="utf-8"))

    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"Refusing to overwrite non-empty directory without a manifest: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, local_files_only=True)
    lengths, spans, length_stats = compute_lengths(tokenizer, captions, int(args.max_length))
    if captions[0] != "":
        raise RuntimeError(f"CFG null contract expects the empty caption at id 0; got {captions[0]!r}")

    offsets = np.zeros(len(captions) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(lengths, dtype=np.int64)
    total_tokens = int(offsets[-1])
    hidden = int(json.loads((Path(args.base_model) / "config.json").read_text())["hidden_size"])

    write_caption_index(captions_path, captions)
    np.save(offsets_path, offsets, allow_pickle=False)
    np.save(spans_path, spans, allow_pickle=False)
    storage = np.dtype("uint16" if args.storage_dtype == "bfloat16" else args.storage_dtype)
    for path, shape in ((tokens_path, (total_tokens, hidden)), (pooled_path, (len(captions), hidden))):
        memmap = np.lib.format.open_memmap(path, mode="w+", dtype=storage, shape=shape)
        memmap.flush()
        del memmap

    manifest: Dict[str, object] = {
        "format": TEXT_CACHE_FORMAT,
        "version": TEXT_CACHE_VERSION,
        "status": "building",
        "num_captions": len(captions),
        "captions_sha256": expected_digest,
        "source": source_stats,
        "build": {
            "transformers_version": __import__("transformers").__version__,
            "torch_version": torch.__version__,
            "model_dtype": args.model_dtype,
            "encoder_batch_size": int(args.batch_size),
        },
        "storage_dtype": {"tokens": args.storage_dtype, "numpy": str(storage)},
        "encoder": {
            "recipe": "llm2vec",
            "base_model": str(Path(args.base_model).resolve()),
            "mntp_model": str(Path(args.mntp_model).resolve()),
            "supervised_model": str(Path(args.supervised_model).resolve()) if args.supervised_model else "",
            "hidden_size": hidden,
            "feature_layer": FEATURE_LAYER,
            "pooling": POOLING,
            "instruction": "",
            # The model input is exactly PREFIX + caption + SUFFIX.  Upstream's
            # "!@#$%^&*()" separator is a position marker that `tokenize` strips
            # via "".join(parts); it is never tokenized, so naming it here (as
            # an earlier revision did with "..._with_separator") misdescribes
            # what was encoded.
            "text_format": "llama3_user_turn",
            "text_format_detail": {
                "prefix": LLAMA3_TURN_PREFIX,
                "suffix": LLAMA3_TURN_SUFFIX,
                "separator_in_model_input": False,
                "pooling_span": "trailing content_span tokens (caption + suffix)",
            },
            "max_length": int(args.max_length),
            "total_tokens": total_tokens,
            "length_stats": length_stats,
        },
        "files": {
            "captions": captions_path.name,
            "tokens": tokens_path.name,
            "offsets": offsets_path.name,
            "pooled": pooled_path.name,
            "spans": spans_path.name,
        },
    }
    progress = {"count": 0}
    atomic_write_json(manifest_path, manifest)
    atomic_write_json(progress_path, progress)
    print(
        f"CACHE_PLAN captions={len(captions)} total_tokens={total_tokens} "
        f"tokens_gib={total_tokens * hidden * storage.itemsize / 2**30:.2f} "
        f"pooled_gib={len(captions) * hidden * storage.itemsize / 2**30:.2f}",
        flush=True,
    )
    return output_dir, manifest, progress


def encode(args: argparse.Namespace, captions: Sequence[str], output_dir: Path,
           manifest: Dict[str, object], progress: Dict[str, int]) -> None:
    start = int(progress.get("count", 0))
    if start >= len(captions):
        return
    encoder, supervised_applied = build_encoder(args)
    print(f"ENCODER supervised_adapter={'applied' if supervised_applied else 'absent'}", flush=True)
    tokenizer = encoder.tokenizer
    device = next(encoder.model.parameters()).device

    offsets = np.load(output_dir / "offsets.npy", mmap_mode="r", allow_pickle=False)
    # Pool over the same spans the model will later read from spans.npy, rather
    # than recomputing them here -- two copies of the rule can silently diverge.
    spans = np.load(output_dir / "spans.npy", mmap_mode="r", allow_pickle=False)
    tokens_cache = np.lib.format.open_memmap(output_dir / "tokens.npy", mode="r+")
    pooled_cache = np.lib.format.open_memmap(output_dir / "pooled.npy", mode="r+")
    progress_path = output_dir / "progress.json"
    store_bf16 = args.storage_dtype == "bfloat16"

    def to_storage(values: torch.Tensor) -> np.ndarray:
        if store_bf16:
            return values.to(torch.bfloat16).view(torch.uint16).numpy()
        return values.float().numpy().astype(args.storage_dtype, copy=False)

    for idx in range(start, len(captions)):
        batch = tokenize(tokenizer, captions[idx], int(args.max_length))
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        with torch.inference_mode():
            hidden = encoder.model(input_ids=ids, attention_mask=mask).last_hidden_state
            # Average only the caption span, as upstream's embed_mask does;
            # averaging the chat-turn markers as well dilutes the sentence
            # vector with template tokens.
            span = int(spans[idx])
            pooled = hidden[:, -span:, :].mean(dim=1)
        length = int(offsets[idx + 1]) - int(offsets[idx])
        if span > length:
            raise RuntimeError(f"caption {idx}: content span {span} exceeds stored length {length}")
        rows = hidden[0, :length].contiguous().cpu()
        tokens_cache[int(offsets[idx]): int(offsets[idx]) + length] = to_storage(rows).astype(
            tokens_cache.dtype, copy=False
        )
        pooled_cache[idx] = to_storage(pooled[0].contiguous().cpu()).astype(pooled_cache.dtype, copy=False)
        if (idx + 1) % 500 == 0 or idx + 1 == len(captions):
            tokens_cache.flush()
            pooled_cache.flush()
            progress["count"] = idx + 1
            atomic_write_json(progress_path, progress)
            print(f"ENCODE {idx + 1}/{len(captions)}", flush=True)
    tokens_cache.flush()
    pooled_cache.flush()
    progress["count"] = len(captions)
    atomic_write_json(progress_path, progress)
    del tokens_cache, pooled_cache, offsets, encoder
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def validate_cache(output_dir: Path) -> Dict[str, object]:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from models.codeflow.text_encoder import CachedLLM2VecTextEncoder

    encoder = CachedLLM2VecTextEncoder(str(output_dir), use_pooled=True)
    tokens = encoder._tokens
    nonfinite = zero_rows = 0
    for start in range(0, int(tokens.shape[0]), 2048):
        chunk = np.asarray(tokens[start: start + 2048])
        if chunk.dtype == np.uint16:
            zero_rows += int((~np.any((chunk & np.uint16(0x7FFF)) != 0, axis=1)).sum())
            nonfinite += int(((chunk & np.uint16(0x7F80)) == np.uint16(0x7F80)).sum())
        else:
            zero_rows += int((~np.any(chunk != 0, axis=1)).sum())
            nonfinite += int((~np.isfinite(chunk)).sum())
    if nonfinite or zero_rows:
        raise RuntimeError(f"Cache validation failed: nonfinite={nonfinite}, zero_rows={zero_rows}")

    example = next((t for t in encoder._caption_to_id if t), None)
    condition = encoder(["", example] if example else [""])
    summary = encoder.cache_summary()
    summary.update({
        "sample_tokens_shape": list(condition.tokens.shape),
        "sample_pooled_shape": list(condition.pooled.shape),
        "finite": bool(torch.isfinite(condition.tokens).all() and torch.isfinite(condition.pooled).all()),
        "zero_token_rows": zero_rows,
        "nonfinite_values": nonfinite,
    })
    if not summary["finite"]:
        raise RuntimeError("Cache validation found non-finite features")
    return summary


def main() -> None:
    args = parse_args()
    captions, source_stats = collect_captions(args)
    output_dir = Path(args.output_dir).expanduser().resolve()
    if args.validate_only:
        print("CACHE_VALID " + json.dumps(validate_cache(output_dir), sort_keys=True), flush=True)
        return

    output_dir, manifest, progress = create_or_resume_cache(args, captions, source_stats)
    if manifest.get("status") == "complete":
        print("CACHE_VALID " + json.dumps(validate_cache(output_dir), sort_keys=True), flush=True)
        return

    encode(args, captions, output_dir, manifest, progress)
    manifest["status"] = "complete"
    atomic_write_json(output_dir / "manifest.json", manifest)
    try:
        validation = validate_cache(output_dir)
    except Exception:
        manifest["status"] = "invalid"
        atomic_write_json(output_dir / "manifest.json", manifest)
        raise
    print("CACHE_VALID " + json.dumps(validation, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
