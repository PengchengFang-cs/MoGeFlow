#!/usr/bin/env python3
"""Supervised contrastive training for Qwen3 (LLM2Vec step 3).

Follows ``llm2vec/experiments/run_supervised.py`` and
``train_configs/supervised/MetaLlama3.json``.  Only the model-side plumbing is
reimplemented (the upstream package cannot be imported alongside Qwen3); the
E5 dataset, its instruction prompts, and HardNegativeNLLLoss are vendored
verbatim under ``vendored/``.

Assembly order, as upstream:
    base Qwen3-8B -> MNTP LoRA (merged) -> fresh LoRA for supervised training

Two details are load-bearing and easy to get silently wrong:

* the dataset pre-groups samples so that every consecutive block of
  ``effective_batch_size`` comes from one source dataset, which is what makes
  in-batch negatives meaningful -- so the sampler MUST stay sequential;
* the tokenizer must pad LEFT, because pooling averages the trailing
  ``seq_length`` positions.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Union

import torch
from torch import nn
from torch.utils.data import DataLoader, SequentialSampler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.llm2vec_qwen3.bidirectional_qwen3 import (  # noqa: E402
    Qwen3BiForMNTP,
    disable_module_causal_flags,
)
from tools.llm2vec_qwen3.supervised_encoder import (  # noqa: E402
    Qwen3SentenceEncoder,
    prepare_for_tokenization,
)
from tools.llm2vec_qwen3.vendored.E5Data import E5Data  # noqa: E402
from tools.llm2vec_qwen3.vendored.HardNegativeNLLLoss import (  # noqa: E402
    HardNegativeNLLLoss,
)

LORA_TARGETS = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--base_model", default="/scratch/pf2m24/hf-models/Qwen3-8B")
    p.add_argument("--mntp_model", default="/scratch/pf2m24/hf-models/Qwen3-8B-mntp")
    p.add_argument("--output_dir", default="/scratch/pf2m24/hf-models/Qwen3-8B-mntp-supervised")
    p.add_argument("--dataset_file_path", default="/scratch/pf2m24/text-caches/echo-data")
    p.add_argument("--max_seq_length", type=int, default=512)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--per_device_train_batch_size", type=int, default=64)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--max_steps", type=int, default=1000)
    p.add_argument("--warmup_steps", type=int, default=300)
    p.add_argument("--logging_steps", type=int, default=20)
    p.add_argument("--save_steps", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    # 50 comes from upstream's experiments/run_supervised.py CustomArguments
    # (`loss_scale: float = field(default=50.0)`), which train_configs/supervised/
    # MetaLlama3.json does not override.  Note this is NOT the HardNegativeNLLLoss
    # class default -- that is 20, which is what an earlier revision here used.
    p.add_argument("--loss_scale", type=float, default=50.0)
    p.add_argument("--smoke_test", action="store_true",
                   help="3 steps at batch 4 on a small slice, for wiring validation only.")
    return p.parse_args()


class TripletCollator:
    """Tokenize the (query, positive, negative) columns of a batch separately."""

    def __init__(self, encoder: Qwen3SentenceEncoder) -> None:
        self.encoder = encoder

    def __call__(self, batch: List[Any]):
        num_texts = len(batch[0].texts)
        columns: List[List[str]] = [[] for _ in range(num_texts)]
        labels = []
        for example in batch:
            for idx, text in enumerate(example.texts):
                columns[idx].append(prepare_for_tokenization(text))
            labels.append(example.label)
        features = [self.encoder.tokenize(column) for column in columns]
        return features, torch.tensor(labels)


def build_trainer_cls():
    from transformers import Trainer

    class SupervisedTrainer(Trainer):
        def __init__(self, *args, loss_function=None, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.loss_function = loss_function

        def compute_loss(
            self,
            model: nn.Module,
            inputs: Dict[str, Union[torch.Tensor, Any]],
            return_outputs: bool = False,
            **kwargs,
        ):
            features, _ = inputs
            q_reps = model(features[0])
            d_reps = model(features[1])
            d_reps_neg = model(features[2]) if len(features) > 2 else None
            loss = self.loss_function(q_reps, d_reps, d_reps_neg)
            return (loss, {"q": q_reps, "d": d_reps}) if return_outputs else loss

        def get_train_dataloader(self) -> DataLoader:
            # Sequential on purpose: E5Data already arranged the rows so each
            # consecutive block of effective_batch_size shares a source
            # dataset.  Shuffling here would destroy the in-batch negatives.
            if self.train_dataset is None:
                raise ValueError("Trainer: training requires a train_dataset.")
            return self.accelerator.prepare(
                DataLoader(
                    self.train_dataset,
                    batch_size=self._train_batch_size,
                    collate_fn=self.data_collator,
                    num_workers=self.args.dataloader_num_workers,
                    pin_memory=self.args.dataloader_pin_memory,
                    sampler=SequentialSampler(self.train_dataset),
                    drop_last=True,
                )
            )

    return SupervisedTrainer


def move_to_device(features, device):
    return [{k: v.to(device) for k, v in feature.items()} for feature in features]


def main() -> None:
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoTokenizer, Trainer, TrainingArguments, set_seed

    args = parse_args()
    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, local_files_only=True)
    tokenizer.padding_side = "left"          # required by the pooling
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    batch_size = 4 if args.smoke_test else args.per_device_train_batch_size
    dataset = E5Data(
        dataset_name="E5", split="train",
        file_path=args.dataset_file_path,
        effective_batch_size=batch_size,
    )
    if args.smoke_test:
        dataset.data = dataset.data[: batch_size * 8]
    print(f"[data] {len(dataset)} triplets, effective_batch_size={batch_size}", flush=True)

    model = Qwen3BiForMNTP.from_pretrained(
        args.base_model, local_files_only=True, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa", low_cpu_mem_usage=True,
    )
    disable_module_causal_flags(model)

    # Step 2 adapter is merged in, then a fresh adapter is trained on top.
    trunk = PeftModel.from_pretrained(model.model, args.mntp_model)
    trunk = trunk.merge_and_unload()
    print("[setup] MNTP adapter merged", flush=True)

    lora = LoraConfig(
        r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=args.lora_dropout,
        target_modules=LORA_TARGETS, bias="none", task_type=None,
    )
    trunk = get_peft_model(trunk, lora)
    trunk.print_trainable_parameters()

    encoder = Qwen3SentenceEncoder(trunk, tokenizer, max_length=args.max_seq_length).cuda()
    trainable = [n for n, p in encoder.named_parameters() if p.requires_grad]
    if not any("lora_" in n for n in trainable):
        raise RuntimeError("LoRA was not applied: no lora_* parameter requires grad")
    if any("lm_head" in n for n in trainable):
        raise RuntimeError("lm_head must not train in the supervised stage")
    print(f"[setup] {sum(p.numel() for p in encoder.parameters() if p.requires_grad)/1e6:.1f}M trainable", flush=True)

    trunk.enable_input_require_grads()
    trunk.config.use_cache = False

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        max_steps=(3 if args.smoke_test else args.max_steps),
        warmup_steps=(0 if args.smoke_test else args.warmup_steps),
        logging_steps=(1 if args.smoke_test else args.logging_steps),
        save_steps=args.save_steps,
        save_total_limit=1,
        save_only_model=True,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to=[],
        seed=args.seed,
        remove_unused_columns=False,
        dataloader_num_workers=2,
    )
    trainer = build_trainer_cls()(
        model=encoder,
        args=training_args,
        train_dataset=dataset,
        data_collator=TripletCollator(encoder),
        loss_function=HardNegativeNLLLoss(scale=args.loss_scale),
    )
    result = trainer.train()
    losses = [h["loss"] for h in trainer.state.log_history if "loss" in h]
    if not losses or any(l != l for l in losses):
        raise RuntimeError(f"Training produced no finite loss: {losses}")
    print(f"[train] first loss {losses[0]:.4f} -> last loss {losses[-1]:.4f}", flush=True)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    trunk.save_pretrained(str(out))
    tokenizer.save_pretrained(str(out))
    (out / "supervised_run.json").write_text(json.dumps({
        "base_model": args.base_model,
        "mntp_model": args.mntp_model,
        "recipe": "llm2vec_mntp_supervised",
        "dataset": args.dataset_file_path,
        "pooling": "mean", "skip_instruction": True, "padding_side": "left",
        "lora": {"r": args.lora_r, "alpha": 2 * args.lora_r,
                 "dropout": args.lora_dropout, "targets": LORA_TARGETS},
        "loss": {"name": "HardNegativeNLLLoss", "scale": args.loss_scale},
        "max_steps": training_args.max_steps,
        "per_device_train_batch_size": batch_size,
        "learning_rate": args.learning_rate,
        "train_loss": result.training_loss,
        "loss_first": losses[0], "loss_last": losses[-1],
    }, indent=2) + "\n", encoding="utf-8")
    print(f"SAVED_ADAPTER {out}", flush=True)
    print("SUPERVISED_DONE", flush=True)


if __name__ == "__main__":
    main()
