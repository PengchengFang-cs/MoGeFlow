"""Tokenizer backend selection for CodeFlow."""

from __future__ import annotations

from typing import Optional

import torch

from .kv_vq import PartVQTokenizer
from .momask_vq import MoMaskRVQTokenizer


def build_codeflow_tokenizer(
    backend: str,
    kv_root: str,
    checkpoint_path: Optional[str] = None,
    partition_path: Optional[str] = None,
    opt_path: Optional[str] = None,
    device: Optional[torch.device] = None,
):
    if backend == "kv_part":
        return PartVQTokenizer(
            kv_root=kv_root,
            checkpoint_path=checkpoint_path,
            partition_path=partition_path,
            device=device,
        )
    if backend == "momask_rvq":
        if not checkpoint_path:
            raise ValueError("momask_rvq backend requires --vq_checkpoint")
        return MoMaskRVQTokenizer(
            checkpoint_path=checkpoint_path,
            opt_path=opt_path,
            device=device,
        )
    raise ValueError(f"Unsupported VQ backend: {backend}")
