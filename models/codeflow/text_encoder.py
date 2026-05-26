"""Frozen CLIP text encoder used by the code-flow prior."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import torch
import torch.nn as nn
import clip


CLIP_MODEL_PATH_ENV = "MASKCONTROL_CLIP_MODEL_PATH"
CLIP_CACHE_DIR_ENV = "MASKCONTROL_CLIP_CACHE_DIR"
CLIP_MODEL_FILENAMES = {
    "RN50": "RN50.pt",
    "RN101": "RN101.pt",
    "RN50x4": "RN50x4.pt",
    "RN50x16": "RN50x16.pt",
    "RN50x64": "RN50x64.pt",
    "ViT-B/32": "ViT-B-32.pt",
    "ViT-B/16": "ViT-B-16.pt",
    "ViT-L/14": "ViT-L-14.pt",
    "ViT-L/14@336px": "ViT-L-14-336px.pt",
}


@dataclass
class TextCondition:
    pooled: torch.Tensor
    tokens: torch.Tensor
    padding_mask: torch.Tensor


def _as_text_list(raw_text: Iterable[str]) -> List[str]:
    if isinstance(raw_text, str):
        return [raw_text]
    return [str(item) for item in raw_text]


def resolve_clip_checkpoint(
    clip_version: str,
    explicit_path: Optional[str] = None,
    kv_root: Optional[str] = None,
) -> str:
    if explicit_path:
        path = Path(explicit_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"CLIP checkpoint not found: {path}")
        return str(path)

    env_path = os.environ.get(CLIP_MODEL_PATH_ENV)
    if env_path:
        path = Path(env_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"{CLIP_MODEL_PATH_ENV} points to missing file: {path}")
        return str(path)

    version_path = Path(clip_version).expanduser()
    if version_path.is_file():
        return str(version_path.resolve())

    if kv_root is not None and clip_version in CLIP_MODEL_FILENAMES:
        candidate = (
            Path(kv_root).expanduser().resolve()
            / "checkpoints"
            / "clip"
            / CLIP_MODEL_FILENAMES[clip_version]
        )
        if candidate.is_file():
            return str(candidate)

    cache_dir = os.environ.get(CLIP_CACHE_DIR_ENV)
    if cache_dir and clip_version in CLIP_MODEL_FILENAMES:
        candidate = Path(cache_dir).expanduser().resolve() / CLIP_MODEL_FILENAMES[clip_version]
        if candidate.is_file():
            return str(candidate)

    return clip_version


class FrozenCLIPTextEncoder(nn.Module):
    """OpenAI CLIP text tower with pooled and per-token outputs."""

    def __init__(
        self,
        clip_version: str = "ViT-B/32",
        clip_path: Optional[str] = None,
        kv_root: Optional[str] = None,
    ) -> None:
        super().__init__()
        resolved = resolve_clip_checkpoint(clip_version, explicit_path=clip_path, kv_root=kv_root)
        self.clip_version = clip_version
        self.clip_path = resolved
        model, _ = clip.load(resolved, device="cpu", jit=False)
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)
        self.clip_model = model
        self.width = int(model.ln_final.weight.shape[0])
        self.output_dim = int(model.text_projection.shape[1])

    @property
    def device(self) -> torch.device:
        return next(self.clip_model.parameters()).device

    @torch.no_grad()
    def encode(
        self,
        raw_text: Iterable[str],
        drop_prob: float = 0.0,
        force_drop: bool = False,
    ) -> TextCondition:
        texts = _as_text_list(raw_text)
        if force_drop:
            texts = [""] * len(texts)
        elif drop_prob > 0.0:
            keep = torch.rand(len(texts), device=self.device) >= float(drop_prob)
            texts = [text if bool(keep[i].item()) else "" for i, text in enumerate(texts)]

        text_tokens = clip.tokenize(texts, truncate=True).to(self.device)
        cm = self.clip_model
        x = cm.token_embedding(text_tokens).type(cm.dtype)
        x = x + cm.positional_embedding.type(cm.dtype)
        x = x.permute(1, 0, 2)
        x = cm.transformer(x)
        x = x.permute(1, 0, 2)
        x = cm.ln_final(x).type(cm.dtype)

        pooled = x[torch.arange(x.shape[0], device=x.device), text_tokens.argmax(dim=-1)] @ cm.text_projection
        padding_mask = text_tokens == 0
        return TextCondition(
            pooled=pooled.float(),
            tokens=x.float(),
            padding_mask=padding_mask,
        )

    def forward(
        self,
        raw_text: Iterable[str],
        drop_prob: float = 0.0,
        force_drop: bool = False,
    ) -> TextCondition:
        return self.encode(raw_text, drop_prob=drop_prob, force_drop=force_drop)
