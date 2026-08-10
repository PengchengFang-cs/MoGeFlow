#!/usr/bin/env python3
"""MNTP training for bidirectional Qwen3-8B (LLM2Vec step 2).

Semantics are replicated from ``llm2vec/experiments/run_mntp.py`` rather than
ported line by line, because that script targets transformers<=4.44.2 while
Qwen3 needs >=4.51.  The parts that define the method are reproduced exactly:

* mask token is ``"_"`` (``mask_token_type=blank``) -- an existing vocabulary
  entry, so the embedding matrix is never resized;
* 20% of non-special tokens are selected as prediction targets and corrupted
  BERT-style -- 80% mask token, 10% random vocabulary token, 10% left
  unchanged -- with loss on every selected position.  Upstream's config sets
  ``"data_collator_type": "default"``, i.e. the stock HF collator, which is
  exactly this; an earlier revision of this file masked 100% of targets
  instead.  The distinction matters here more than for a plain MLM: the 10%
  random and 10% unchanged branches are what stop the model from treating
  "mask token present" as the cue to build a good representation, and we then
  extract features from text with no masks in it at all;
* the "next token" half comes from the causal-LM label shift inside
  ``Qwen3ForCausalLM.forward``: masking position i and training with the
  standard shift makes position i-1 predict token i, using bidirectional
  context on both sides;
* LoRA r=16 / alpha=2*r / dropout=0.05 on q,k,v,o,gate,up,down_proj, applied to
  the trunk only.  The LM head is deliberately left OUTSIDE the PEFT wrapper and
  therefore trains in full, exactly as upstream does -- it never touches
  ``lm_head`` or ``requires_grad``.  For Qwen3-8B that is 622M extra trainable
  parameters on top of the 43.6M of LoRA (``tie_word_embeddings=False``).

Defaults mirror ``train_configs/mntp/MetaLlama3.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.llm2vec_qwen3.bidirectional_qwen3 import (  # noqa: E402
    Qwen3BiForMNTP,
    assert_bidirectional,
    disable_module_causal_flags,
)

LORA_TARGETS = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--model_path", default="/scratch/pf2m24/hf-models/Qwen3-8B")
    p.add_argument("--output_dir", default="/scratch/pf2m24/hf-models/Qwen3-8B-mntp")
    p.add_argument("--dataset_name", default="wikitext")
    p.add_argument("--dataset_config_name", default="wikitext-103-raw-v1")
    p.add_argument("--max_seq_length", type=int, default=512)
    p.add_argument("--mlm_probability", type=float, default=0.2)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=0,
                   help="0 keeps the upstream convention lora_alpha = 2 * lora_r.")
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--per_device_train_batch_size", type=int, default=32)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--learning_rate", type=float, default=5e-5)
    p.add_argument("--max_steps", type=int, default=1000)
    # Upstream's mntp config sets neither learning_rate nor warmup, so HF's
    # TrainingArguments defaults apply: lr 5e-5 and NO warmup.  An earlier
    # revision here warmed up over 100 steps.
    p.add_argument("--warmup_steps", type=int, default=0)
    p.add_argument("--logging_steps", type=int, default=20)
    p.add_argument("--save_steps", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_proc", type=int, default=8)
    p.add_argument("--prepared_dataset",
                   default="/scratch/pf2m24/text-caches/wikitext103_qwen3_blocks512")
    p.add_argument("--force_prepare", action="store_true")
    p.add_argument("--prepare_only", action="store_true",
                   help="Tokenize and save the dataset, then exit (run on the login node).")
    p.add_argument("--smoke_test", action="store_true",
                   help="Tiny run: 3 steps on a small slice, for wiring validation only.")
    return p.parse_args()


MASK_REPLACE_PROB = 0.8
RANDOM_REPLACE_PROB = 0.1


def build_collator(tokenizer, mlm_probability: float):
    """The stock BERT-style collator, as upstream's ``"default"`` selects.

    The 80/10/10 split is passed explicitly rather than left to the library
    default: transformers only exposed ``mask_replace_prob`` /
    ``random_replace_prob`` as arguments in 4.52, so pinning them here keeps
    the recipe identical if the environment's transformers changes.
    """
    from transformers import DataCollatorForLanguageModeling

    mask_token_id = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
    if mask_token_id is None or mask_token_id == tokenizer.unk_token_id:
        raise ValueError(f"Mask token {tokenizer.mask_token!r} is not in the vocabulary")
    return DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=float(mlm_probability),
        mask_replace_prob=MASK_REPLACE_PROB,
        random_replace_prob=RANDOM_REPLACE_PROB,
    )


def build_dataset(args, tokenizer):
    """Tokenize and block wikitext, caching the result to disk.

    Compute nodes have no outbound network and ``datasets`` still contacts the
    Hub for builder metadata even with the offline flags set, so preparation
    runs once on the login node (``--prepare_only``) and training reads the
    saved arrow dataset.
    """
    from datasets import load_dataset, load_from_disk

    prepared = Path(args.prepared_dataset)
    if prepared.is_dir() and not args.force_prepare:
        dataset = load_from_disk(str(prepared))
        print(f"[data] loaded prepared dataset from {prepared}", flush=True)
        return dataset

    split = "train[:2000]" if args.smoke_test else "train"
    raw = load_dataset(args.dataset_name, args.dataset_config_name, split=split)
    num_proc = 1 if args.smoke_test else args.num_proc

    def tokenize(batch):
        return tokenizer(batch["text"], return_attention_mask=False)

    tokenized = raw.map(
        tokenize, batched=True, num_proc=num_proc,
        remove_columns=raw.column_names, desc="tokenizing",
    )
    block = int(args.max_seq_length)

    def group(batch):
        flat = [tok for row in batch["input_ids"] for tok in row]
        total = (len(flat) // block) * block
        return {"input_ids": [flat[i: i + block] for i in range(0, total, block)]}

    grouped = tokenized.map(group, batched=True, num_proc=num_proc, desc=f"grouping into {block}-token blocks")
    grouped = grouped.filter(lambda row: len(row["input_ids"]) == block, num_proc=num_proc)
    prepared.parent.mkdir(parents=True, exist_ok=True)
    grouped.save_to_disk(str(prepared))
    print(f"[data] saved prepared dataset to {prepared}", flush=True)
    return grouped


def main() -> None:
    from peft import LoraConfig, get_peft_model
    from transformers import AutoTokenizer, Trainer, TrainingArguments, set_seed

    args = parse_args()
    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    if tokenizer.mask_token is None:
        tokenizer.mask_token = "_"          # blank-style mask, reuses an existing token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.prepare_only:
        dataset = build_dataset(args, tokenizer)
        print(f"PREPARE_DONE blocks={len(dataset)} path={args.prepared_dataset}", flush=True)
        return

    model = Qwen3BiForMNTP.from_pretrained(
        args.model_path, local_files_only=True, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa", low_cpu_mem_usage=True,
    )
    stamped = disable_module_causal_flags(model)
    print(f"[setup] is_causal=False on {stamped} modules", flush=True)

    lora_alpha = args.lora_alpha if args.lora_alpha > 0 else 2 * args.lora_r
    lora = LoraConfig(
        r=args.lora_r, lora_alpha=lora_alpha, lora_dropout=args.lora_dropout,
        target_modules=LORA_TARGETS, bias="none", task_type=None,
    )
    # PEFT wraps the trunk only, so lm_head stays outside and trains in full --
    # this mirrors upstream, which never touches lm_head or requires_grad.
    model.model = get_peft_model(model.model, lora)
    model.model.print_trainable_parameters()

    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    lora_params = [n for n in trainable if "lora_" in n]
    head_params = [n for n in trainable if "lm_head" in n]
    other = [n for n in trainable if n not in set(lora_params) | set(head_params)]
    if not lora_params:
        raise RuntimeError("LoRA was not applied: no lora_* parameter requires grad")
    if not head_params:
        raise RuntimeError("lm_head should train during MNTP but is frozen")
    if other:
        raise RuntimeError(f"Unexpected trainable parameters outside LoRA/lm_head: {other[:3]}")
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"[setup] lora_alpha={lora_alpha}; trainable {n_train / 1e6:.1f}M "
        f"({len(lora_params)} LoRA tensors + lm_head)",
        flush=True,
    )

    # Gradient checkpointing silently blocks LoRA gradients unless the inputs
    # are marked as requiring grad, so enable that explicitly.
    model.enable_input_require_grads()
    model.config.use_cache = False

    dataset = build_dataset(args, tokenizer)
    print(f"[data] {len(dataset)} blocks of {args.max_seq_length} tokens", flush=True)
    collator = build_collator(tokenizer, args.mlm_probability)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        per_device_train_batch_size=(2 if args.smoke_test else args.per_device_train_batch_size),
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        max_steps=(3 if args.smoke_test else args.max_steps),
        warmup_steps=(0 if args.smoke_test else args.warmup_steps),
        logging_steps=(1 if args.smoke_test else args.logging_steps),
        save_steps=args.save_steps,
        save_total_limit=2,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to=[],
        seed=args.seed,
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
        processing_class=tokenizer,
    )
    result = trainer.train()
    losses = [h["loss"] for h in trainer.state.log_history if "loss" in h]
    if not losses or not all(l == l for l in losses):  # NaN check
        raise RuntimeError(f"Training produced no finite loss: {losses}")
    print(f"[train] first loss {losses[0]:.4f} -> last loss {losses[-1]:.4f}", flush=True)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.model.save_pretrained(str(out))     # LoRA adapter for the trunk
    # lm_head trains in full and is not part of the adapter, so persist it too;
    # without this the trained head would be silently lost on reload.
    torch.save(model.lm_head.state_dict(), out / "lm_head.pt")
    tokenizer.save_pretrained(str(out))
    (out / "mntp_run.json").write_text(json.dumps({
        "base_model": args.model_path,
        "recipe": "llm2vec_mntp",
        "mask_token": tokenizer.mask_token,
        "mlm_probability": args.mlm_probability,
        "data_collator": "DataCollatorForLanguageModeling",
        "mask_replace_prob": MASK_REPLACE_PROB,
        "random_replace_prob": RANDOM_REPLACE_PROB,
        "max_seq_length": args.max_seq_length,
        "lora": {"r": args.lora_r, "alpha": lora_alpha,
                 "dropout": args.lora_dropout, "targets": LORA_TARGETS},
        "lm_head_trained": True,
        "max_steps": training_args.max_steps,
        "dataset": f"{args.dataset_name}/{args.dataset_config_name}",
        "train_loss": result.training_loss,
        "loss_first": losses[0],
        "loss_last": losses[-1],
    }, indent=2) + "\n", encoding="utf-8")
    print(f"SAVED_ADAPTER {out}", flush=True)

    # The trained model must still be bidirectional; verify in fp32.
    model = model.float().eval()
    ratio = assert_bidirectional(model, tokenizer)
    print(f"[verify] post-training bidirectionality ratio {ratio:.4f}", flush=True)
    print("MNTP_DONE", flush=True)


if __name__ == "__main__":
    main()
