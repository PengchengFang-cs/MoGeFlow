"""Sentence encoder wrapper for supervised LLM2Vec training on Qwen3.

Reimplements the parts of ``llm2vec.LLM2Vec`` that the supervised trainer
needs.  The upstream class cannot be imported here (it pulls in
``bidirectional_llama`` etc., which require transformers<=4.44.2, while Qwen3
needs >=4.51), so the behaviour is reproduced rather than reused.

Semantics preserved exactly from upstream:

* text arrives as ``"<instruction>; " + SEPARATOR + "<content>"``.  Only the
  content contributes to the sentence vector: ``embed_mask`` marks the last N
  positions (N = number of content tokens) and replaces ``attention_mask``
  before pooling (``skip_instruction=True``);
* pooling is a mean over the last ``seq_length`` positions, which is only
  correct with LEFT padding -- upstream asserts this and so does this module;
* ``prepare_for_tokenization`` wraps text in the model's chat turn markers.
  Upstream hardcodes a per-checkpoint mapping and has no Qwen3 entry, so the
  Qwen3 ChatML markers are used here, matching the Qwen2 entry upstream.
"""

from __future__ import annotations

from typing import Dict, Iterable, List

import torch
import torch.nn as nn

SEPARATOR = "!@#$%^&*()"
QWEN3_TURN_PREFIX = "<|im_start|>user\n"
QWEN3_TURN_SUFFIX = "<|im_end|>"


def prepare_for_tokenization(text: str) -> str:
    """Wrap a single text in Qwen3 ChatML user-turn markers."""
    return QWEN3_TURN_PREFIX + text.strip() + QWEN3_TURN_SUFFIX


class Qwen3SentenceEncoder(nn.Module):
    """Mean-pooled sentence encoder over a bidirectional Qwen3 trunk."""

    def __init__(self, model: nn.Module, tokenizer, max_length: int = 512,
                 pooling_mode: str = "mean", skip_instruction: bool = True) -> None:
        super().__init__()
        if pooling_mode != "mean":
            raise ValueError(f"Only mean pooling is ported; got {pooling_mode!r}")
        if tokenizer.padding_side != "left":
            raise ValueError(
                "Pooling takes the trailing seq_length positions, so the tokenizer "
                f"must pad on the left; got padding_side={tokenizer.padding_side!r}"
            )
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        self.pooling_mode = pooling_mode
        self.skip_instruction = bool(skip_instruction)

    # HF Trainer drives gradient checkpointing and config lookups through the
    # top-level model, which here is this wrapper, so delegate to the trunk.
    @property
    def config(self):
        return self.model.config

    def gradient_checkpointing_enable(self, **kwargs):
        return self.model.gradient_checkpointing_enable(**kwargs)

    def gradient_checkpointing_disable(self):
        return self.model.gradient_checkpointing_disable()

    def enable_input_require_grads(self):
        return self.model.enable_input_require_grads()

    def tokenize(self, texts: Iterable[str]) -> Dict[str, torch.Tensor]:
        """Tokenize, and mark which trailing tokens are the embeddable content."""
        texts = list(texts)
        contents: List[str] = []
        joined: List[str] = []
        for text in texts:
            parts = text.split(SEPARATOR)
            contents.append(parts[1] if len(parts) > 1 else "")
            joined.append("".join(parts))

        batch = self.tokenizer(
            joined, return_tensors="pt", padding=True, truncation=True,
            max_length=self.max_length,
        )
        rows = []
        for idx, content in enumerate(contents):
            ids = self.tokenizer(
                [content], return_tensors="pt", padding=True, truncation=True,
                max_length=self.max_length, add_special_tokens=False,
            )["input_ids"][0]
            row = torch.zeros_like(batch["attention_mask"][idx])
            if len(ids) > 0:
                row[-len(ids):] = 1
            rows.append(row)
        batch["embed_mask"] = torch.stack(rows, dim=0)
        return batch

    def forward(self, sentence_feature: Dict[str, torch.Tensor]) -> torch.Tensor:
        feature = dict(sentence_feature)
        embed_mask = feature.pop("embed_mask", None)
        reps = self.model(**feature)
        hidden = reps.last_hidden_state

        mask = sentence_feature["attention_mask"]
        if self.skip_instruction:
            if embed_mask is None:
                raise ValueError("skip_instruction=True requires embed_mask")
            if embed_mask.shape != mask.shape:
                raise ValueError(
                    f"embed_mask shape {tuple(embed_mask.shape)} != "
                    f"attention_mask shape {tuple(mask.shape)}"
                )
            mask = embed_mask
        lengths = mask.sum(dim=-1)
        return torch.stack(
            [
                hidden[i, -int(length):, :].mean(dim=0)
                if int(length) > 0
                else hidden[i, -1:, :].mean(dim=0)
                for i, length in enumerate(lengths)
            ],
            dim=0,
        )
