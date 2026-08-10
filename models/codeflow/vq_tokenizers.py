"""Tokenizer backend selection for CodeFlow."""

from __future__ import annotations

from typing import Optional

import torch

from .kv_vq import PartVQTokenizer
from .momask_vq import MoMaskRVQTokenizer
from .partvae_vq import PartVAEContinuousTokenizer


def build_codeflow_tokenizer(
    backend: str,
    kv_root: str,
    checkpoint_path: Optional[str] = None,
    partition_path: Optional[str] = None,
    opt_path: Optional[str] = None,
    num_codes: Optional[int] = None,
    code_dim: Optional[int] = None,
    kv_part_target_mode: str = "codebook",
    rvq_target_mode: str = "stage",
    device: Optional[torch.device] = None,
):
    if backend == "kv_part":
        return PartVQTokenizer(
            kv_root=kv_root,
            checkpoint_path=checkpoint_path,
            partition_path=partition_path,
            num_codes=num_codes,
            code_dim=code_dim,
            target_mode=kv_part_target_mode,
            device=device,
        )
    if backend == "momask_rvq":
        if not checkpoint_path:
            raise ValueError("momask_rvq backend requires --vq_checkpoint")
        return MoMaskRVQTokenizer(
            checkpoint_path=checkpoint_path,
            opt_path=opt_path,
            target_mode=rvq_target_mode,
            device=device,
        )
    if backend == "partvae_continuous":
        if not checkpoint_path:
            raise ValueError("partvae_continuous backend requires --vq_checkpoint")
        return PartVAEContinuousTokenizer(
            checkpoint_path=checkpoint_path,
            config_path=opt_path,
            partition_path=partition_path,
            code_dim=code_dim,
            device=device,
        )
    raise ValueError(f"Unsupported VQ backend: {backend}")
