#!/usr/bin/env python3
"""Compare cached CLIP-L/Qwen3 features with freshly encoded features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from models.codeflow.text_encoder import CachedCLIPQwenTextEncoder
from tools.precompute_clip_qwen_text_cache import fixed_encoder_batch, tokenize_qwen_batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--cache_path", required=True)
    parser.add_argument("--qwen_path", default="/scratch/pf2m24/hf-models/Qwen3-8B")
    parser.add_argument("--clip_path", default="/scratch/pf2m24/hf-models/clip-vit-large-patch14")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--sample_count", type=int, default=4)
    parser.add_argument("--selection", choices=["spread", "first", "last"], default="spread")
    parser.add_argument("--attn_implementation", default="sdpa")
    parser.add_argument("--qwen_tolerance", type=float, default=1e-3)
    parser.add_argument("--clip_tolerance", type=float, default=1e-5)
    return parser.parse_args()


def selected_captions(encoder: CachedCLIPQwenTextEncoder, count: int, selection: str):
    ordered = list(encoder._caption_to_id)
    if len(ordered) == 1:
        return ordered
    count = min(max(2, int(count)), len(ordered))
    if selection == "first":
        return ordered[:count]
    if selection == "last":
        return ordered[-count:]
    indices = sorted({round(idx * (len(ordered) - 1) / (count - 1)) for idx in range(count)})
    if 0 not in indices:
        indices.insert(0, 0)
    return [ordered[idx] for idx in indices]


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    encoder = CachedCLIPQwenTextEncoder(args.cache_path).to(device)
    captions = selected_captions(encoder, args.sample_count, args.selection)
    cached = encoder(captions)
    manifest = encoder.manifest
    encoder_batch_size = int(manifest.get("build", {}).get("encoder_batch_size", len(captions)))
    live_captions, real_count = fixed_encoder_batch(captions, encoder_batch_size)

    from transformers import AutoModelForCausalLM, AutoTokenizer, CLIPTextModel, CLIPTokenizer

    clip_tokenizer = CLIPTokenizer.from_pretrained(args.clip_path, local_files_only=True)
    clip_tokenizer.padding_side = "right"
    clip_tokenizer.truncation_side = "right"
    clip_model = CLIPTextModel.from_pretrained(
        args.clip_path,
        local_files_only=True,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    ).to(device).eval()
    clip_batch = clip_tokenizer(
        live_captions,
        truncation=True,
        max_length=77,
        padding=True,
        return_tensors="pt",
    )
    clip_batch = {key: value.to(device) for key, value in clip_batch.items()}
    with torch.inference_mode():
        live_clip = clip_model(**clip_batch).pooler_output[:real_count].float()
    clip_max_abs = float((cached.pooled - live_clip).abs().max().item())
    del clip_model, clip_batch, live_clip
    torch.cuda.empty_cache()

    qwen_tokenizer = AutoTokenizer.from_pretrained(args.qwen_path, local_files_only=True, use_fast=True)
    qwen_tokenizer.padding_side = "right"
    qwen_tokenizer.truncation_side = "right"
    qwen_model = AutoModelForCausalLM.from_pretrained(
        args.qwen_path,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation=args.attn_implementation,
        use_safetensors=True,
    ).to(device).eval()
    crop_start = int(manifest["qwen"]["crop_start"])
    max_length = int(manifest["qwen"]["max_length"])
    qwen_batch = tokenize_qwen_batch(qwen_tokenizer, live_captions, max_length, crop_start)
    input_ids = qwen_batch["input_ids"].to(device)
    attention_mask = qwen_batch["attention_mask"].to(device)
    with torch.inference_mode():
        output = qwen_model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        ).last_hidden_state
    live_qwen = output[:real_count, crop_start : crop_start + max_length].float()
    live_lengths = (
        attention_mask[:real_count].sum(dim=-1).long() - crop_start
    ).clamp(min=1, max=max_length)
    cached_lengths = (~cached.padding_mask).sum(dim=-1).long()
    if not torch.equal(live_lengths.cpu(), cached_lengths.cpu()):
        raise RuntimeError(
            f"Cached/live Qwen length mismatch: cached={cached_lengths.tolist()} live={live_lengths.tolist()}"
        )
    qwen_max_abs = 0.0
    qwen_abs_sum = 0.0
    qwen_value_count = 0
    qwen_max_rel_l2 = 0.0
    qwen_min_cosine = 1.0
    for row, length in enumerate(cached_lengths.tolist()):
        cached_row = cached.tokens[row, :length].float()
        live_row = live_qwen[row, :length].float()
        abs_diff = (cached_row - live_row).abs()
        qwen_max_abs = max(qwen_max_abs, float(abs_diff.max().item()))
        qwen_abs_sum += float(abs_diff.double().sum().item())
        qwen_value_count += int(abs_diff.numel())
        rel_l2 = float(
            torch.linalg.vector_norm((cached_row - live_row).reshape(-1)).item()
            / torch.linalg.vector_norm(live_row.reshape(-1)).clamp_min(1e-12).item()
        )
        cosine = float(
            torch.nn.functional.cosine_similarity(
                cached_row.reshape(1, -1),
                live_row.reshape(1, -1),
            ).item()
        )
        qwen_max_rel_l2 = max(qwen_max_rel_l2, rel_l2)
        qwen_min_cosine = min(qwen_min_cosine, cosine)

    payload = {
        "cache": str(Path(args.cache_path).expanduser().resolve()),
        "sample_count": len(captions),
        "selection": args.selection,
        "encoder_batch_size": encoder_batch_size,
        "clip_max_abs": clip_max_abs,
        "qwen_max_abs": qwen_max_abs,
        "qwen_mean_abs": qwen_abs_sum / max(qwen_value_count, 1),
        "qwen_max_rel_l2": qwen_max_rel_l2,
        "qwen_min_cosine": qwen_min_cosine,
        "qwen_cached_absmax": float(cached.tokens.abs().max().item()),
        "qwen_cached_mean_abs": float(cached.tokens.abs().mean().item()),
        "lengths": cached_lengths.tolist(),
    }
    if clip_max_abs > float(args.clip_tolerance):
        raise RuntimeError(f"CLIP cache parity failed: {payload}")
    if qwen_max_abs > float(args.qwen_tolerance):
        raise RuntimeError(f"Qwen cache parity failed: {payload}")
    print("LIVE_CACHE_PARITY " + json.dumps(payload, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
