#!/usr/bin/env python3
"""Precompute a resumable token-only Qwen3-8B norm-average caption cache.

Per token, the cached feature is the parameter-free ``F.layer_norm`` of every
transformer-layer hidden state (embedding output excluded) averaged across all
layers.  Captions are rendered as a user-only chat message with no system
prompt and no template-prefix cropping (full chat), following the MotionCraft
``qwen_normavg`` recipe.  There is no CLIP branch: the resulting cache feeds
the token-only ``qwen_normavg_cache`` text provider, where the model has no
pooled/global text conditioning path.
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
import torch.nn.functional as F


TEXT_CACHE_FORMAT = "momask_qwen3_normavg_cache"
TEXT_CACHE_VERSION = 1
FEATURE_LAYER = "all_transformer_layers_layernorm_mean"
QWEN_REPO_ID = "Qwen/Qwen3-8B"
CHAT_TEMPLATE_DESC = (
    "user-only; add_generation_prompt=False; enable_thinking=False; full chat, no prefix crop"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data_root", type=str, default="dataset/HumanML3D")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/scratch/pf2m24/text-caches/humanml3d_qwen3_8b_normavg_userchat_l128_v1",
    )
    parser.add_argument("--qwen_path", type=str, default="/scratch/pf2m24/hf-models/Qwen3-8B")
    parser.add_argument("--qwen_max_length", type=int, default=128)
    parser.add_argument("--layer_norm_eps", type=float, default=1.0e-5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--tokenizer_batch_size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model_dtype", type=str, default="bfloat16", choices=["bfloat16", "float16"])
    parser.add_argument(
        "--qwen_storage_dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="bfloat16 is stored losslessly as raw uint16 words in the mmap.",
    )
    parser.add_argument("--attn_implementation", type=str, default="sdpa", choices=["eager", "sdpa", "flash_attention_2"])
    parser.add_argument(
        "--caption_file",
        action="append",
        default=[],
        help="Optional extra prompt file (one caption per line); may be supplied repeatedly.",
    )
    parser.add_argument("--include_caption", action="append", default=[])
    parser.add_argument(
        "--skip_dataset_captions",
        action="store_true",
        help="Build a small compatible prompt cache from only the empty prompt and --caption_file/--include_caption.",
    )
    parser.add_argument("--max_captions", type=int, default=0, help="Smoke-test cap; 0 uses every unique caption.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--recompute_tail",
        action="store_true",
        help="Re-encode the final partial batch of a complete cache using fixed batch padding.",
    )
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


def local_hf_revision(model_path: Path) -> str:
    metadata = model_path / ".cache" / "huggingface" / "download" / "config.json.metadata"
    if not metadata.is_file():
        return "unknown"
    lines = metadata.read_text(encoding="utf-8").splitlines()
    return lines[0].strip() if lines else "unknown"


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

    captions = {""}  # Required unconditional feature for standard CFG.
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


def format_qwen_caption(tokenizer, caption: str) -> str:
    messages = [{"role": "user", "content": caption}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )


def tokenize_qwen_batch(tokenizer, captions: Sequence[str], max_length: int):
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"
    rendered = [format_qwen_caption(tokenizer, caption) for caption in captions]
    return tokenizer(
        rendered,
        add_special_tokens=False,
        return_length=False,
        return_overflowing_tokens=False,
        truncation=True,
        return_attention_mask=True,
        max_length=int(max_length),
        padding="max_length",
        return_tensors="pt",
    )


def compute_qwen_lengths(
    tokenizer,
    captions: Sequence[str],
    max_length: int,
    batch_size: int,
) -> Tuple[np.ndarray, Dict[str, object]]:
    lengths = np.zeros(len(captions), dtype=np.int64)
    original_lengths = np.zeros(len(captions), dtype=np.int64)
    for start in range(0, len(captions), batch_size):
        end = min(start + batch_size, len(captions))
        rendered = [format_qwen_caption(tokenizer, caption) for caption in captions[start:end]]
        untruncated = tokenizer(
            rendered,
            add_special_tokens=False,
            truncation=False,
            padding=False,
            return_attention_mask=False,
        )["input_ids"]
        original_lengths[start:end] = np.asarray(
            [max(1, len(ids)) for ids in untruncated],
            dtype=np.int64,
        )
        encoded = tokenize_qwen_batch(tokenizer, captions[start:end], max_length)
        batch_lengths = encoded["attention_mask"].sum(dim=-1).long()
        batch_lengths = batch_lengths.clamp(min=1, max=int(max_length))
        lengths[start:end] = batch_lengths.cpu().numpy().astype(np.int64, copy=False)
        if start == 0 or end == len(captions) or end % 4096 == 0:
            print(f"TOKEN_LENGTHS {end}/{len(captions)}", flush=True)
    stats = {
        "original_min": int(original_lengths.min()),
        "original_mean": float(original_lengths.mean()),
        "original_p95": float(np.percentile(original_lengths, 95)),
        "original_p99": float(np.percentile(original_lengths, 99)),
        "original_max": int(original_lengths.max()),
        "truncated_count": int((original_lengths > int(max_length)).sum()),
    }
    return lengths, stats


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


def assert_manifest_matches_args(
    manifest: Dict[str, object],
    args: argparse.Namespace,
    expected_digest: str,
) -> None:
    qwen_path = Path(args.qwen_path).expanduser().resolve()
    expected_storage = {
        "qwen": args.qwen_storage_dtype,
        "qwen_numpy": str(np.dtype("uint16" if args.qwen_storage_dtype == "bfloat16" else args.qwen_storage_dtype)),
    }
    checks = {
        "captions_sha256": (manifest.get("captions_sha256"), expected_digest),
        "qwen.max_length": (manifest.get("qwen", {}).get("max_length"), int(args.qwen_max_length)),
        "qwen.revision": (manifest.get("qwen", {}).get("revision"), local_hf_revision(qwen_path)),
        "qwen.feature_layer": (manifest.get("qwen", {}).get("feature_layer"), FEATURE_LAYER),
        "qwen.layer_norm_eps": (manifest.get("qwen", {}).get("layer_norm_eps"), float(args.layer_norm_eps)),
        "qwen.chat_template": (manifest.get("qwen", {}).get("chat_template"), CHAT_TEMPLATE_DESC),
        "qwen.padding_side": (manifest.get("qwen", {}).get("padding_side", "right"), "right"),
        "qwen.truncation_side": (manifest.get("qwen", {}).get("truncation_side", "right"), "right"),
        "storage_dtype": (manifest.get("storage_dtype"), expected_storage),
        "build.model_dtype": (manifest.get("build", {}).get("model_dtype"), args.model_dtype),
        "build.attn_implementation": (
            manifest.get("build", {}).get("attn_implementation"),
            args.attn_implementation,
        ),
        "build.encoder_batch_size": (
            manifest.get("build", {}).get("encoder_batch_size"),
            int(args.batch_size),
        ),
        "build.fixed_batch_padding": (
            manifest.get("build", {}).get("fixed_batch_padding"),
            True,
        ),
        "build.transformers_version": (
            manifest.get("build", {}).get("transformers_version"),
            __import__("transformers").__version__,
        ),
        "build.torch_version": (
            manifest.get("build", {}).get("torch_version"),
            torch.__version__,
        ),
    }
    mismatches = [f"{name}: cache={actual!r} expected={expected!r}" for name, (actual, expected) in checks.items() if actual != expected]
    if mismatches:
        raise RuntimeError("Text cache contract mismatch:\n  " + "\n  ".join(mismatches))


def create_or_resume_cache(args: argparse.Namespace, captions: List[str], source_stats: Dict[str, object]):
    output_dir = Path(args.output_dir).expanduser().resolve()
    manifest_path = output_dir / "manifest.json"
    progress_path = output_dir / "progress.json"
    captions_path = output_dir / "captions.jsonl"
    tokens_path = output_dir / "qwen_tokens.npy"
    offsets_path = output_dir / "qwen_offsets.npy"
    expected_digest = caption_digest(captions)

    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert_manifest_matches_args(manifest, args, expected_digest)
        if manifest.get("status") == "complete":
            return output_dir, manifest, {"qwen_count": len(captions)}
        if not args.resume:
            raise RuntimeError(f"Incomplete cache exists at {output_dir}; pass --resume")
        cached_captions = read_caption_index(captions_path)
        if cached_captions != captions or manifest.get("captions_sha256") != expected_digest:
            raise RuntimeError("Cannot resume: caption index/digest changed")
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        return output_dir, manifest, progress

    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"Refusing to overwrite non-empty directory without a manifest: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer

    qwen_path = Path(args.qwen_path).expanduser().resolve()
    qwen_config = json.loads((qwen_path / "config.json").read_text(encoding="utf-8"))
    qwen_dim = int(qwen_config["hidden_size"])
    num_hidden_layers = int(qwen_config["num_hidden_layers"])

    tokenizer = AutoTokenizer.from_pretrained(str(qwen_path), local_files_only=True, use_fast=True)
    lengths, length_stats = compute_qwen_lengths(
        tokenizer,
        captions,
        max_length=int(args.qwen_max_length),
        batch_size=int(args.tokenizer_batch_size),
    )
    if captions[0] != "":
        raise RuntimeError(f"CFG null contract expects empty caption at id 0; got {captions[0]!r}")
    if int(lengths[0]) < 1:
        raise RuntimeError(f"Empty caption rendered to zero tokens; got length={int(lengths[0])}")
    offsets = np.zeros(len(captions) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(lengths, dtype=np.int64)
    total_tokens = int(offsets[-1])

    write_caption_index(captions_path, captions)
    np.save(offsets_path, offsets, allow_pickle=False)
    qwen_storage_dtype = np.dtype("uint16" if args.qwen_storage_dtype == "bfloat16" else args.qwen_storage_dtype)
    qwen_memmap = np.lib.format.open_memmap(
        tokens_path,
        mode="w+",
        dtype=qwen_storage_dtype,
        shape=(total_tokens, qwen_dim),
    )
    qwen_memmap.flush()
    del qwen_memmap

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
            "attn_implementation": args.attn_implementation,
            "encoder_batch_size": int(args.batch_size),
            "fixed_batch_padding": True,
        },
        "storage_dtype": {
            "qwen": args.qwen_storage_dtype,
            "qwen_numpy": str(qwen_storage_dtype),
        },
        "qwen": {
            "repo_id": QWEN_REPO_ID,
            "local_path": str(qwen_path),
            "revision": local_hf_revision(qwen_path),
            "architecture": qwen_config.get("architectures", []),
            "hidden_size": qwen_dim,
            "num_hidden_layers": num_hidden_layers,
            "feature_layer": FEATURE_LAYER,
            "layer_norm_eps": float(args.layer_norm_eps),
            "norm_dtype": "float32",
            "max_length": int(args.qwen_max_length),
            "padding_side": "right",
            "truncation_side": "right",
            "chat_template": CHAT_TEMPLATE_DESC,
            "null_prompt_length": int(lengths[0]),
            "total_tokens": total_tokens,
            "length_stats": length_stats,
        },
        "files": {
            "captions": captions_path.name,
            "qwen_tokens": tokens_path.name,
            "qwen_offsets": offsets_path.name,
        },
    }
    progress = {"qwen_count": 0}
    atomic_write_json(manifest_path, manifest)
    atomic_write_json(progress_path, progress)
    print(
        f"CACHE_PLAN captions={len(captions)} total_qwen_tokens={total_tokens} "
        f"qwen_file_gib={total_tokens * qwen_dim * qwen_storage_dtype.itemsize / 2**30:.2f}",
        flush=True,
    )
    return output_dir, manifest, progress


def model_dtype(name: str) -> torch.dtype:
    return torch.bfloat16 if name == "bfloat16" else torch.float16


def fixed_encoder_batch(captions: Sequence[str], batch_size: int) -> Tuple[List[str], int]:
    """Pad a partial encoder batch so SDPA/GEMM batch shapes stay invariant."""
    values = list(captions)
    real_count = len(values)
    if real_count == 0:
        return values, 0
    if real_count > int(batch_size):
        raise ValueError(f"Encoder batch has {real_count} rows, expected at most {int(batch_size)}")
    values.extend([values[-1]] * (int(batch_size) - real_count))
    return values, real_count


def normavg_hidden(hidden_states, num_hidden_layers: int, layer_norm_eps: float) -> torch.Tensor:
    """Parameter-free LayerNorm of every transformer layer, averaged.

    The embedding-layer output ``hidden_states[0]`` is excluded, matching the
    MotionCraft ``qwen_normavg`` recipe.  Normalization and accumulation run
    in float32 for numerical stability before the storage cast.
    """
    if hidden_states is None or len(hidden_states) != num_hidden_layers + 1:
        actual = 0 if hidden_states is None else len(hidden_states)
        raise RuntimeError(
            f"Expected {num_hidden_layers + 1} hidden states "
            f"({num_hidden_layers} transformer layers + embeddings), got {actual}"
        )
    hidden_size = int(hidden_states[-1].shape[-1])
    averaged = None
    for hidden in hidden_states[1:]:
        normalized = F.layer_norm(
            hidden.float(),
            [hidden_size],
            eps=float(layer_norm_eps),
        )
        if averaged is None:
            averaged = normalized
        else:
            averaged.add_(normalized)
    if averaged is None:
        raise RuntimeError("No transformer-layer hidden states were returned")
    averaged.div_(float(num_hidden_layers))
    return averaged


def encode_qwen(
    args: argparse.Namespace,
    captions: Sequence[str],
    output_dir: Path,
    manifest: Dict[str, object],
    progress: Dict[str, int],
) -> None:
    start_count = int(progress.get("qwen_count", 0))
    if start_count >= len(captions):
        return
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device(args.device)
    dtype = model_dtype(args.model_dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.qwen_path, local_files_only=True, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.qwen_path,
        local_files_only=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        attn_implementation=args.attn_implementation,
        use_safetensors=True,
    ).to(device)
    model.eval().requires_grad_(False)
    backbone = getattr(model, "model", model)
    num_hidden_layers = int(manifest["qwen"]["num_hidden_layers"])
    layer_norm_eps = float(manifest["qwen"]["layer_norm_eps"])
    offsets = np.load(output_dir / "qwen_offsets.npy", mmap_mode="r", allow_pickle=False)
    cache = np.lib.format.open_memmap(output_dir / "qwen_tokens.npy", mode="r+")
    progress_path = output_dir / "progress.json"

    for start in range(start_count, len(captions), int(args.batch_size)):
        end = min(start + int(args.batch_size), len(captions))
        encoder_captions, real_count = fixed_encoder_batch(captions[start:end], int(args.batch_size))
        encoded = tokenize_qwen_batch(
            tokenizer,
            encoder_captions,
            max_length=int(args.qwen_max_length),
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        with torch.inference_mode():
            outputs = backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
            hidden = normavg_hidden(outputs.hidden_states, num_hidden_layers, layer_norm_eps)
        for local_idx, cache_id in enumerate(range(start, start + real_count)):
            token_start = int(offsets[cache_id])
            token_end = int(offsets[cache_id + 1])
            length = token_end - token_start
            values = hidden[local_idx, :length].contiguous().cpu()
            if args.qwen_storage_dtype == "bfloat16":
                values = values.to(torch.bfloat16).view(torch.uint16).numpy()
            else:
                values = values.float().numpy().astype(args.qwen_storage_dtype, copy=False)
            cache[token_start:token_end] = values.astype(cache.dtype, copy=False)
        cache.flush()
        progress["qwen_count"] = end
        atomic_write_json(progress_path, progress)
        if start == start_count or end == len(captions) or end % 256 == 0:
            print(f"QWEN {end}/{len(captions)}", flush=True)
    del cache, offsets, backbone, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def validate_cache(output_dir: Path) -> Dict[str, object]:
    from models.codeflow.text_encoder import CachedQwenNormAvgTextEncoder

    encoder = CachedQwenNormAvgTextEncoder(str(output_dir))
    qwen = encoder._qwen_tokens

    nonfinite_qwen_values = 0
    zero_qwen_rows = 0
    rows_per_chunk = 2048
    for start in range(0, int(qwen.shape[0]), rows_per_chunk):
        chunk = np.asarray(qwen[start : start + rows_per_chunk])
        if chunk.dtype == np.uint16 and encoder.qwen_storage_dtype == "bfloat16":
            # BF16 has the same 8-bit exponent layout as FP32; exponent=all
            # ones denotes Inf/NaN, so this checks the raw mmap losslessly.
            zero_qwen_rows += int(
                (~np.any((chunk & np.uint16(0x7FFF)) != 0, axis=1)).sum()
            )
            nonfinite_qwen_values += int(
                ((chunk & np.uint16(0x7F80)) == np.uint16(0x7F80)).sum()
            )
        else:
            zero_qwen_rows += int((~np.any(chunk != 0, axis=1)).sum())
            nonfinite_qwen_values += int((~np.isfinite(chunk)).sum())
    if nonfinite_qwen_values or zero_qwen_rows:
        raise RuntimeError(
            "Qwen cache validation failed: "
            f"nonfinite_values={nonfinite_qwen_values}, all_zero_token_rows={zero_qwen_rows}"
        )

    nonempty_example = next((text for text in encoder._caption_to_id if text), None)
    examples = [""] if nonempty_example is None else ["", nonempty_example]
    condition = encoder(examples)
    if condition.pooled is not None:
        raise RuntimeError("Token-only cache unexpectedly produced a pooled feature")
    summary = encoder.cache_summary()
    summary.update(
        {
            "sample_tokens_shape": list(condition.tokens.shape),
            "sample_padding_shape": list(condition.padding_mask.shape),
            "finite": bool(torch.isfinite(condition.tokens).all()),
            "full_qwen_nonfinite_values": nonfinite_qwen_values,
            "full_qwen_zero_token_rows": zero_qwen_rows,
        }
    )
    if not summary["finite"]:
        raise RuntimeError("Text cache validation found non-finite features")
    return summary


def main() -> None:
    args = parse_args()
    if args.qwen_max_length <= 0 or args.batch_size <= 0 or args.tokenizer_batch_size <= 0:
        raise ValueError("Lengths and batch sizes must be positive")
    if args.layer_norm_eps <= 0:
        raise ValueError("layer_norm_eps must be positive")
    captions, source_stats = collect_captions(args)
    output_dir = Path(args.output_dir).expanduser().resolve()
    if args.validate_only:
        print("CACHE_VALID " + json.dumps(validate_cache(output_dir), sort_keys=True), flush=True)
        return

    output_dir, manifest, progress = create_or_resume_cache(args, captions, source_stats)
    if manifest.get("status") == "complete":
        if not args.recompute_tail:
            print("CACHE_VALID " + json.dumps(validate_cache(output_dir), sort_keys=True), flush=True)
            return
        tail_start = (len(captions) // int(args.batch_size)) * int(args.batch_size)
        if tail_start == len(captions):
            print("CACHE_VALID " + json.dumps(validate_cache(output_dir), sort_keys=True), flush=True)
            return
        progress = {"qwen_count": tail_start}
        manifest["status"] = "building"
        atomic_write_json(output_dir / "manifest.json", manifest)
        atomic_write_json(output_dir / "progress.json", progress)
        print(
            f"RECOMPUTE_TAIL start={tail_start} count={len(captions) - tail_start} "
            f"encoder_batch_size={int(args.batch_size)}",
            flush=True,
        )

    encode_qwen(args, captions, output_dir, manifest, progress)
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
