"""Text feature providers used by the code-flow prior.

The legacy provider evaluates one frozen OpenAI CLIP tower online.  The
``clip_qwen_cache`` provider reads precomputed CLIP-L pooled features and
Qwen3 token features from a versioned, memory-mapped cache so the 8B language
model is never loaded during CodeFlow training.
"""

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
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
    # ``pooled`` is None for token-only providers (``qwen_normavg_cache``),
    # where text conditions the model exclusively through the token stream.
    pooled: Optional[torch.Tensor]
    tokens: torch.Tensor
    padding_mask: torch.Tensor
    # True on the positions holding the caption itself.  Chat-wrapped providers
    # keep the ``<|im_start|>user`` scaffolding in the token stream, so a plain
    # mean over valid positions blends wrapper tokens into the summary; the
    # refiner pools over this mask instead.  None when the provider cannot say
    # which positions are content -- callers that need it must refuse rather
    # than fall back, since a wrong span is silent.
    content_mask: Optional[torch.Tensor] = None


def _load_content_spans(
    root: Path, files: Dict[str, str], count: int
) -> Optional[np.ndarray]:
    """Load per-caption content-span lengths, or None for caches without them.

    ``spans[i]`` counts the *trailing* tokens of caption ``i`` that are the
    caption itself: the wrapper prefix comes first, so the content occupies
    ``[length - span, length)``.  This matches how the LLM2Vec builders pool
    (``hidden[:, -span:, :].mean(dim=1)``) and is only valid for recipes that
    put the content last -- see CachedQwenNormAvgTextEncoder for one that does
    not.
    """
    path = root / files.get("spans", "spans.npy")
    if not path.is_file():
        return None
    spans = np.load(path, allow_pickle=False)
    if spans.shape != (count,):
        raise RuntimeError(
            f"{path} holds spans of shape {tuple(spans.shape)}, expected ({count},)"
        )
    if int(spans.min()) < 1:
        raise RuntimeError(f"{path} holds a non-positive content span: min={int(spans.min())}")
    return spans


def _content_mask_from_spans(
    spans: Optional[np.ndarray],
    ids: Iterable[int],
    lengths: Iterable[int],
    batch: int,
    max_len: int,
) -> Optional[torch.Tensor]:
    """Mark the caption span of each row, or None when the cache has no spans."""
    if spans is None:
        return None
    mask = torch.zeros(batch, max_len, dtype=torch.bool)
    for row, (cache_id, length) in enumerate(zip(ids, lengths)):
        length = int(length)
        # A truncated row can be shorter than the span recorded at build time.
        span = min(int(spans[int(cache_id)]), length)
        mask[row, length - span: length] = True
    return mask


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


class _ScaffoldFreeProvider:
    """Mixin marker: every valid token of this provider is caption text.

    CLIP-family towers wrap the caption in SOT/EOT only, so "content span" and
    "all valid tokens" are the same set and the refiner's two pooling modes are
    numerically identical.  Providers that render an instruction/chat template
    must NOT set this.
    """

    provides_content_mask = True


class FrozenCLIPTextEncoder(_ScaffoldFreeProvider, nn.Module):
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
            # CLIP wraps the caption in SOT/EOT only, with no instruction
            # scaffolding, so every valid position is content.
            content_mask=~padding_mask,
        )

    def forward(
        self,
        raw_text: Iterable[str],
        drop_prob: float = 0.0,
        force_drop: bool = False,
    ) -> TextCondition:
        return self.encode(raw_text, drop_prob=drop_prob, force_drop=force_drop)


class DualCLIPTextEncoder(_ScaffoldFreeProvider, nn.Module):
    """Two frozen CLIP text towers: OpenAI ViT-B/32 plus HF CLIP-L/14.

    Both towers are parameter-free from the model's perspective, so the two
    feature spaces are kept separable without adding trainable weights here:
    token sequences are concatenated along time, and each tower writes into
    its own slice of the feature axis (ViT-B/32 into ``[0:512]``, CLIP-L into
    ``[512:1280]``), zero elsewhere.  A single downstream
    ``Linear(1280, hidden)`` therefore acts as two independent per-tower
    projections that share only a bias.  Pooled vectors are concatenated so
    the downstream pooled MLP sees both sentence embeddings jointly.
    """

    def __init__(
        self,
        clip_version: str = "ViT-B/32",
        clip_path: Optional[str] = None,
        clip_hf_path: Optional[str] = None,
        kv_root: Optional[str] = None,
    ) -> None:
        super().__init__()
        from transformers import CLIPTextModel, CLIPTokenizer

        resolved = resolve_clip_checkpoint(clip_version, explicit_path=clip_path, kv_root=kv_root)
        self.clip_version = clip_version
        self.clip_path = resolved
        model, _ = clip.load(resolved, device="cpu", jit=False)
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)
        self.clip_model = model
        self.openai_width = int(model.ln_final.weight.shape[0])
        self.openai_output_dim = int(model.text_projection.shape[1])

        if not clip_hf_path:
            raise ValueError("text_encoder_type='dual_clip' requires --clip_hf_path")
        hf_path = Path(clip_hf_path).expanduser().resolve()
        if not hf_path.is_dir():
            raise FileNotFoundError(f"HF CLIP directory not found: {hf_path}")
        self.clip_hf_path = str(hf_path)
        self.hf_tokenizer = CLIPTokenizer.from_pretrained(str(hf_path), local_files_only=True)
        self.hf_tokenizer.padding_side = "right"
        self.hf_tokenizer.truncation_side = "right"
        hf_model = CLIPTextModel.from_pretrained(
            str(hf_path),
            local_files_only=True,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        )
        hf_model.eval()
        for param in hf_model.parameters():
            param.requires_grad_(False)
        self.clip_hf_model = hf_model
        self.hf_width = int(hf_model.config.hidden_size)
        self.hf_max_length = int(getattr(hf_model.config, "max_position_embeddings", 77))

        self.width = self.openai_width + self.hf_width
        self.output_dim = self.openai_output_dim + self.hf_width

    @property
    def device(self) -> torch.device:
        return next(self.clip_model.parameters()).device

    def encoder_summary(self) -> Dict[str, object]:
        return {
            "type": "dual_clip",
            "openai_clip_version": self.clip_version,
            "openai_clip_path": self.clip_path,
            "openai_width": self.openai_width,
            "openai_output_dim": self.openai_output_dim,
            "hf_clip_path": self.clip_hf_path,
            "hf_width": self.hf_width,
            "hf_max_length": self.hf_max_length,
            "token_width": self.width,
            "pooled_width": self.output_dim,
            "token_layout": "sequence concat; block-diagonal feature slices",
        }

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

        device = self.device
        cm = self.clip_model
        openai_tokens = clip.tokenize(texts, truncate=True).to(device)
        x = cm.token_embedding(openai_tokens).type(cm.dtype)
        x = x + cm.positional_embedding.type(cm.dtype)
        x = x.permute(1, 0, 2)
        x = cm.transformer(x)
        x = x.permute(1, 0, 2)
        x = cm.ln_final(x).type(cm.dtype)
        openai_seq = x.float()
        openai_pooled = (
            x[torch.arange(x.shape[0], device=device), openai_tokens.argmax(dim=-1)] @ cm.text_projection
        ).float()
        openai_pad = openai_tokens == 0

        batch = self.hf_tokenizer(
            texts,
            truncation=True,
            max_length=self.hf_max_length,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt",
        )
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        hf_out = self.clip_hf_model(input_ids=input_ids, attention_mask=attention_mask)
        hf_seq = hf_out.last_hidden_state.float()
        hf_pooled = hf_out.pooler_output.float()
        hf_pad = attention_mask == 0

        bsz = len(texts)
        seq_len = openai_seq.shape[1] + hf_seq.shape[1]
        tokens = torch.zeros(bsz, seq_len, self.width, device=device, dtype=torch.float32)
        tokens[:, : openai_seq.shape[1], : self.openai_width] = openai_seq
        tokens[:, openai_seq.shape[1] :, self.openai_width :] = hf_seq
        padding_mask = torch.cat([openai_pad, hf_pad], dim=1)
        pooled = torch.cat([openai_pooled, hf_pooled], dim=1)
        return TextCondition(
            pooled=pooled,
            tokens=tokens,
            padding_mask=padding_mask,
            content_mask=~padding_mask,  # both towers are scaffolding-free
        )

    def forward(
        self,
        raw_text: Iterable[str],
        drop_prob: float = 0.0,
        force_drop: bool = False,
    ) -> TextCondition:
        return self.encode(raw_text, drop_prob=drop_prob, force_drop=force_drop)


class HFCLIPTextEncoder(_ScaffoldFreeProvider, nn.Module):
    """A single frozen HF CLIP text tower (used for CLIP-L/14).

    Structurally identical to the legacy ``clip`` provider -- all 77 token
    states feed the joint attention and the pooled sentence vector drives
    AdaLN -- so a run against it isolates the effect of the CLIP size alone.
    ``pooler_output`` is used unprojected, matching how CLIP-L is already
    consumed by the ``clip_qwen_cache`` provider.
    """

    def __init__(self, clip_hf_path: Optional[str] = None, use_pooled: bool = True) -> None:
        super().__init__()
        from transformers import CLIPTextModel, CLIPTokenizer

        if not clip_hf_path:
            raise ValueError("text_encoder_type='clip_large' requires --clip_hf_path")
        # Same switch the cached LLM2Vec provider uses: output_dim=None is what
        # tells the model not to build the pooled/global AdaLN branch at all.
        # The tower still computes pooler_output; it is simply not consumed.
        self.use_pooled = bool(use_pooled)
        path = Path(clip_hf_path).expanduser().resolve()
        if not path.is_dir():
            raise FileNotFoundError(f"HF CLIP directory not found: {path}")
        self.clip_hf_path = str(path)
        self.tokenizer = CLIPTokenizer.from_pretrained(str(path), local_files_only=True)
        self.tokenizer.padding_side = "right"
        self.tokenizer.truncation_side = "right"
        model = CLIPTextModel.from_pretrained(
            str(path), local_files_only=True, torch_dtype=torch.float32,
            low_cpu_mem_usage=True, use_safetensors=True,
        )
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)
        self.clip_model = model
        self.width = int(model.config.hidden_size)
        self.output_dim = int(model.config.hidden_size) if self.use_pooled else None
        self.max_length = int(getattr(model.config, "max_position_embeddings", 77))

    @property
    def device(self) -> torch.device:
        return next(self.clip_model.parameters()).device

    def encoder_summary(self) -> Dict[str, object]:
        return {
            "type": "clip_large" if self.use_pooled else "clip_large_tokenonly",
            "path": self.clip_hf_path,
            "token_width": self.width,
            "pooled_width": self.output_dim,
            "max_length": self.max_length,
            "pooling": "pooler_output" if self.use_pooled else None,
        }

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

        batch = self.tokenizer(
            texts, truncation=True, max_length=self.max_length,
            padding="max_length", return_attention_mask=True, return_tensors="pt",
        )
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        out = self.clip_model(input_ids=input_ids, attention_mask=attention_mask)
        padding_mask = attention_mask == 0
        return TextCondition(
            pooled=out.pooler_output.float() if self.use_pooled else None,
            tokens=out.last_hidden_state.float(),
            padding_mask=padding_mask,
            content_mask=~padding_mask,  # no instruction scaffolding
        )

    def forward(
        self,
        raw_text: Iterable[str],
        drop_prob: float = 0.0,
        force_drop: bool = False,
    ) -> TextCondition:
        return self.encode(raw_text, drop_prob=drop_prob, force_drop=force_drop)


TEXT_CACHE_FORMAT = "momask_clip_l_qwen3_cache"
TEXT_CACHE_VERSION = 1

QWEN_NORMAVG_CACHE_FORMAT = "momask_qwen3_normavg_cache"
QWEN_NORMAVG_CACHE_VERSION = 1
QWEN_NORMAVG_FEATURE_LAYER = "all_transformer_layers_layernorm_mean"


class CachedCLIPQwenTextEncoder(nn.Module):
    """Read frozen CLIP-L + Qwen3 features from a memory-mapped cache.

    The cache stores one CLIP pooled vector per unique caption and a packed,
    variable-length array of Qwen token features.  Only the caption index is
    resident in each training process; the large feature arrays remain mmap'd
    and therefore share the operating-system page cache across DDP ranks.
    """

    def __init__(
        self,
        cache_path: str,
        expected_qwen_max_length: int = 0,
        expected_fingerprint: str = "",
        require_empty_prompt: bool = True,
    ) -> None:
        super().__init__()
        root = Path(cache_path).expanduser().resolve()
        manifest_path = root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Text cache manifest not found: {manifest_path}")

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("format") != TEXT_CACHE_FORMAT:
            raise RuntimeError(
                f"Unsupported text cache format {manifest.get('format')!r}; "
                f"expected {TEXT_CACHE_FORMAT!r}"
            )
        if int(manifest.get("version", -1)) != TEXT_CACHE_VERSION:
            raise RuntimeError(
                f"Unsupported text cache version {manifest.get('version')}; "
                f"expected {TEXT_CACHE_VERSION}"
            )
        if manifest.get("status") != "complete":
            raise RuntimeError(f"Text cache is not complete: {manifest_path}")

        qwen_cfg = manifest.get("qwen", {})
        clip_cfg = manifest.get("clip", {})
        qwen_max_length = int(qwen_cfg.get("max_length", 0))
        if expected_qwen_max_length > 0 and qwen_max_length != int(expected_qwen_max_length):
            raise RuntimeError(
                f"Text cache Qwen max_length={qwen_max_length} does not match "
                f"configured value {int(expected_qwen_max_length)}"
            )

        files = manifest.get("files", {})
        captions_path = root / files.get("captions", "captions.jsonl")
        pooled_path = root / files.get("clip_pooled", "clip_pooled.npy")
        tokens_path = root / files.get("qwen_tokens", "qwen_tokens.npy")
        offsets_path = root / files.get("qwen_offsets", "qwen_offsets.npy")
        for path in (captions_path, pooled_path, tokens_path, offsets_path):
            if not path.is_file():
                raise FileNotFoundError(f"Text cache file not found: {path}")

        captions: List[str] = []
        with captions_path.open("r", encoding="utf-8") as handle:
            for expected_id, line in enumerate(handle):
                payload = json.loads(line)
                cache_id = int(payload["id"])
                if cache_id != expected_id:
                    raise RuntimeError(
                        f"Non-contiguous caption id in {captions_path}: "
                        f"expected {expected_id}, got {cache_id}"
                    )
                captions.append(str(payload["text"]))

        declared_count = int(manifest.get("num_captions", -1))
        if declared_count != len(captions):
            raise RuntimeError(
                f"Text cache caption count mismatch: manifest={declared_count}, "
                f"index={len(captions)}"
            )
        caption_to_id: Dict[str, int] = {}
        for idx, caption in enumerate(captions):
            if caption in caption_to_id:
                raise RuntimeError(f"Duplicate caption in text cache: {caption!r}")
            caption_to_id[caption] = idx
        if require_empty_prompt and "" not in caption_to_id:
            raise RuntimeError("Text cache must contain the empty prompt for standard CFG")

        pooled = np.load(pooled_path, mmap_mode="r", allow_pickle=False)
        tokens = np.load(tokens_path, mmap_mode="r", allow_pickle=False)
        offsets = np.load(offsets_path, mmap_mode="r", allow_pickle=False)
        clip_dim = int(clip_cfg.get("hidden_size", 0))
        qwen_dim = int(qwen_cfg.get("hidden_size", 0))
        if pooled.shape != (len(captions), clip_dim):
            raise RuntimeError(
                f"CLIP cache shape mismatch: got {pooled.shape}, "
                f"expected {(len(captions), clip_dim)}"
            )
        if tokens.ndim != 2 or tokens.shape[1] != qwen_dim:
            raise RuntimeError(
                f"Qwen cache shape mismatch: got {tokens.shape}, expected [N,{qwen_dim}]"
            )
        if offsets.shape != (len(captions) + 1,):
            raise RuntimeError(
                f"Qwen offsets shape mismatch: got {offsets.shape}, "
                f"expected {(len(captions) + 1,)}"
            )
        if int(offsets[0]) != 0 or int(offsets[-1]) != int(tokens.shape[0]):
            raise RuntimeError(
                f"Qwen offsets do not span token storage: first={int(offsets[0])}, "
                f"last={int(offsets[-1])}, tokens={int(tokens.shape[0])}"
            )
        lengths = np.diff(offsets)
        if np.any(lengths <= 0) or np.any(lengths > qwen_max_length):
            raise RuntimeError(
                f"Invalid cached Qwen lengths: min={int(lengths.min())}, "
                f"max={int(lengths.max())}, configured max={qwen_max_length}"
            )

        self.cache_path = str(root)
        self.manifest = manifest
        # Bind checkpoints to the feature-extraction contract, not to one
        # particular caption index.  This lets evaluation/generation use a
        # separately built cache that is a semantic-compatible superset (for
        # example, a cache extended with user prompts) without accepting a
        # different model, template, crop, layer, or numerical extraction
        # setup.
        clip_semantic = {
            key: manifest.get("clip", {}).get(key)
            for key in (
                "repo_id",
                "revision",
                "hidden_size",
                "pooling",
                "max_length",
                "padding_side",
                "truncation_side",
            )
        }
        clip_semantic["padding_side"] = clip_semantic.get("padding_side") or "right"
        clip_semantic["truncation_side"] = clip_semantic.get("truncation_side") or "right"
        qwen_semantic = {
            key: manifest.get("qwen", {}).get(key)
            for key in (
                "repo_id",
                "revision",
                "architecture",
                "hidden_size",
                "feature_layer",
                "max_length",
                "crop_start",
                "padding_side",
                "truncation_side",
                "system_prompt",
                "chat_template",
            )
        }
        qwen_semantic["padding_side"] = qwen_semantic.get("padding_side") or "right"
        qwen_semantic["truncation_side"] = qwen_semantic.get("truncation_side") or "right"
        build_semantic = {
            key: manifest.get("build", {}).get(key)
            for key in (
                "transformers_version",
                "torch_version",
                "model_dtype",
                "attn_implementation",
                "encoder_batch_size",
                "fixed_batch_padding",
            )
        }
        semantic_manifest = {
            "format": manifest.get("format"),
            "version": manifest.get("version"),
            "clip": clip_semantic,
            "qwen": qwen_semantic,
            "build": build_semantic,
            "storage_dtype": manifest.get("storage_dtype"),
        }
        self.semantic_fingerprint = hashlib.sha256(
            json.dumps(semantic_manifest, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        content_manifest = {
            "format": manifest.get("format"),
            "version": manifest.get("version"),
            "captions_sha256": manifest.get("captions_sha256"),
            "num_captions": manifest.get("num_captions"),
            "total_qwen_tokens": int(tokens.shape[0]),
        }
        self.content_fingerprint = hashlib.sha256(
            json.dumps(content_manifest, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        if expected_fingerprint and self.semantic_fingerprint != str(expected_fingerprint):
            raise RuntimeError(
                "Text cache semantic fingerprint mismatch: "
                f"cache={self.semantic_fingerprint}, expected={expected_fingerprint}"
            )
        self.width = qwen_dim
        self.output_dim = clip_dim
        self.max_length = qwen_max_length
        storage_cfg = manifest.get("storage_dtype", {})
        self.qwen_storage_dtype = (
            str(storage_cfg.get("qwen", "float16"))
            if isinstance(storage_cfg, dict)
            else str(storage_cfg)
        )
        self._caption_to_id = caption_to_id
        self._clip_pooled = pooled
        self._qwen_tokens = tokens
        self._qwen_offsets = offsets
        # No content spans: this recipe renders the caption with Qwen's chat
        # template and pads on the RIGHT, so the caption sits *between* the
        # wrapper tokens rather than at the end.  The trailing-span form the
        # LLM2Vec caches use cannot express that, and a wrong span is silent --
        # so this provider reports None and the model refuses
        # --text_refiner_pool content_span outright.
        self._spans = None
        # Gives a parameter-free feature provider normal nn.Module device
        # semantics without serializing any cached content in checkpoints.
        self.register_buffer("_device_anchor", torch.empty(0), persistent=False)

    @property
    def device(self) -> torch.device:
        return self._device_anchor.device

    def cache_summary(self) -> Dict[str, object]:
        return {
            "path": self.cache_path,
            "format": self.manifest["format"],
            "version": int(self.manifest["version"]),
            "num_captions": int(self.manifest["num_captions"]),
            "clip_hidden_size": int(self.output_dim),
            "qwen_hidden_size": int(self.width),
            "qwen_max_length": int(self.max_length),
            "qwen_storage_dtype": self.qwen_storage_dtype,
            "total_qwen_tokens": int(self._qwen_tokens.shape[0]),
            "semantic_fingerprint": self.semantic_fingerprint,
            "content_fingerprint": self.content_fingerprint,
            "captions_sha256": str(self.manifest.get("captions_sha256", "")),
            "clip_revision": str(self.manifest.get("clip", {}).get("revision", "")),
            "qwen_revision": str(self.manifest.get("qwen", {}).get("revision", "")),
        }

    def validate_coverage(self, captions: Iterable[str]) -> Dict[str, object]:
        requested = set(_as_text_list(captions))
        missing = sorted(text for text in requested if text not in self._caption_to_id)
        if missing:
            examples = ", ".join(repr(text) for text in missing[:5])
            raise RuntimeError(
                f"Text cache coverage failure: {len(missing)} of {len(requested)} unique captions "
                f"are missing from {self.cache_path}; examples: {examples}"
            )
        return {
            "requested_unique_captions": len(requested),
            "cached_captions": len(self._caption_to_id),
            "missing": 0,
        }

    def _resolve_texts(
        self,
        raw_text: Iterable[str],
        drop_prob: float,
        force_drop: bool,
    ) -> List[str]:
        texts = _as_text_list(raw_text)
        if force_drop:
            return [""] * len(texts)
        if drop_prob > 0.0:
            keep = torch.rand(len(texts), device=self.device) >= float(drop_prob)
            return [text if bool(keep[idx].item()) else "" for idx, text in enumerate(texts)]
        return texts

    @torch.no_grad()
    def encode(
        self,
        raw_text: Iterable[str],
        drop_prob: float = 0.0,
        force_drop: bool = False,
    ) -> TextCondition:
        texts = self._resolve_texts(raw_text, drop_prob=drop_prob, force_drop=force_drop)
        missing = [text for text in texts if text not in self._caption_to_id]
        if missing:
            examples = ", ".join(repr(text) for text in missing[:3])
            raise KeyError(
                f"{len(missing)} caption(s) are missing from text cache {self.cache_path}; "
                f"examples: {examples}. Rebuild/extend the cache before training or evaluation."
            )

        ids = np.asarray([self._caption_to_id[text] for text in texts], dtype=np.int64)
        pooled_np = np.asarray(self._clip_pooled[ids]).copy()
        pooled = torch.from_numpy(pooled_np).to(device=self.device, dtype=torch.float32)

        lengths = [
            int(self._qwen_offsets[idx + 1]) - int(self._qwen_offsets[idx])
            for idx in ids.tolist()
        ]
        max_length = max(lengths)
        tokens_cpu = torch.zeros(
            len(texts), max_length, self.width,
            dtype=torch.float32,
        )
        padding_mask_cpu = torch.ones(
            len(texts), max_length,
            dtype=torch.bool,
        )
        for row, (cache_id, length) in enumerate(zip(ids.tolist(), lengths)):
            start = int(self._qwen_offsets[cache_id])
            end = start + length
            token_np = np.asarray(self._qwen_tokens[start:end]).copy()
            token_tensor = torch.from_numpy(token_np)
            if self.qwen_storage_dtype == "bfloat16":
                if token_tensor.dtype != torch.uint16:
                    raise RuntimeError(
                        f"Expected uint16 storage for cached bfloat16 Qwen features, got {token_tensor.dtype}"
                    )
                token_tensor = token_tensor.view(torch.bfloat16)
            token_tensor = token_tensor.to(dtype=torch.float32)
            tokens_cpu[row, :length].copy_(token_tensor)
            padding_mask_cpu[row, :length] = False
        tokens = tokens_cpu.to(device=self.device, non_blocking=False)
        padding_mask = padding_mask_cpu.to(device=self.device, non_blocking=False)
        content_mask = _content_mask_from_spans(
            self._spans, ids.tolist(), lengths, len(texts), max_length
        )
        if content_mask is not None:
            content_mask = content_mask.to(device=self.device, non_blocking=False)
        return TextCondition(
            pooled=pooled,
            tokens=tokens,
            padding_mask=padding_mask,
            content_mask=content_mask,
        )

    def forward(
        self,
        raw_text: Iterable[str],
        drop_prob: float = 0.0,
        force_drop: bool = False,
    ) -> TextCondition:
        return self.encode(raw_text, drop_prob=drop_prob, force_drop=force_drop)


class CachedQwenNormAvgTextEncoder(nn.Module):
    """Read all-layer norm-averaged Qwen3 token features from a mmap cache.

    Token-only provider: the cached feature per token is the parameter-free
    ``F.layer_norm`` of every transformer-layer hidden state (embedding output
    excluded) averaged across layers, computed offline over the full chat
    template without prefix cropping.  There is no CLIP array and no pooled
    vector of its own; with ``use_pooled=True`` the final token's feature is
    exposed as the sentence vector (see __init__), otherwise ``encode``
    returns ``TextCondition(pooled=None, ...)`` and the model conditions on
    text through the token stream only.
    """

    def __init__(
        self,
        cache_path: str,
        expected_qwen_max_length: int = 0,
        expected_fingerprint: str = "",
        require_empty_prompt: bool = True,
        use_pooled: bool = False,
    ) -> None:
        super().__init__()
        root = Path(cache_path).expanduser().resolve()
        manifest_path = root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Text cache manifest not found: {manifest_path}")

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("format") != QWEN_NORMAVG_CACHE_FORMAT:
            raise RuntimeError(
                f"Unsupported text cache format {manifest.get('format')!r}; "
                f"expected {QWEN_NORMAVG_CACHE_FORMAT!r}"
            )
        if int(manifest.get("version", -1)) != QWEN_NORMAVG_CACHE_VERSION:
            raise RuntimeError(
                f"Unsupported text cache version {manifest.get('version')}; "
                f"expected {QWEN_NORMAVG_CACHE_VERSION}"
            )
        if manifest.get("status") != "complete":
            raise RuntimeError(f"Text cache is not complete: {manifest_path}")

        qwen_cfg = manifest.get("qwen", {})
        if qwen_cfg.get("feature_layer") != QWEN_NORMAVG_FEATURE_LAYER:
            raise RuntimeError(
                f"Unexpected feature_layer {qwen_cfg.get('feature_layer')!r}; "
                f"expected {QWEN_NORMAVG_FEATURE_LAYER!r}"
            )
        qwen_max_length = int(qwen_cfg.get("max_length", 0))
        if expected_qwen_max_length > 0 and qwen_max_length != int(expected_qwen_max_length):
            raise RuntimeError(
                f"Text cache Qwen max_length={qwen_max_length} does not match "
                f"configured value {int(expected_qwen_max_length)}"
            )

        files = manifest.get("files", {})
        captions_path = root / files.get("captions", "captions.jsonl")
        tokens_path = root / files.get("qwen_tokens", "qwen_tokens.npy")
        offsets_path = root / files.get("qwen_offsets", "qwen_offsets.npy")
        for path in (captions_path, tokens_path, offsets_path):
            if not path.is_file():
                raise FileNotFoundError(f"Text cache file not found: {path}")

        captions: List[str] = []
        with captions_path.open("r", encoding="utf-8") as handle:
            for expected_id, line in enumerate(handle):
                payload = json.loads(line)
                cache_id = int(payload["id"])
                if cache_id != expected_id:
                    raise RuntimeError(
                        f"Non-contiguous caption id in {captions_path}: "
                        f"expected {expected_id}, got {cache_id}"
                    )
                captions.append(str(payload["text"]))

        declared_count = int(manifest.get("num_captions", -1))
        if declared_count != len(captions):
            raise RuntimeError(
                f"Text cache caption count mismatch: manifest={declared_count}, "
                f"index={len(captions)}"
            )
        caption_to_id: Dict[str, int] = {}
        for idx, caption in enumerate(captions):
            if caption in caption_to_id:
                raise RuntimeError(f"Duplicate caption in text cache: {caption!r}")
            caption_to_id[caption] = idx
        if require_empty_prompt and "" not in caption_to_id:
            raise RuntimeError("Text cache must contain the empty prompt for standard CFG")

        tokens = np.load(tokens_path, mmap_mode="r", allow_pickle=False)
        offsets = np.load(offsets_path, mmap_mode="r", allow_pickle=False)
        qwen_dim = int(qwen_cfg.get("hidden_size", 0))
        if tokens.ndim != 2 or tokens.shape[1] != qwen_dim:
            raise RuntimeError(
                f"Qwen cache shape mismatch: got {tokens.shape}, expected [N,{qwen_dim}]"
            )
        if offsets.shape != (len(captions) + 1,):
            raise RuntimeError(
                f"Qwen offsets shape mismatch: got {offsets.shape}, "
                f"expected {(len(captions) + 1,)}"
            )
        if int(offsets[0]) != 0 or int(offsets[-1]) != int(tokens.shape[0]):
            raise RuntimeError(
                f"Qwen offsets do not span token storage: first={int(offsets[0])}, "
                f"last={int(offsets[-1])}, tokens={int(tokens.shape[0])}"
            )
        lengths = np.diff(offsets)
        if np.any(lengths <= 0) or np.any(lengths > qwen_max_length):
            raise RuntimeError(
                f"Invalid cached Qwen lengths: min={int(lengths.min())}, "
                f"max={int(lengths.max())}, configured max={qwen_max_length}"
            )

        self.cache_path = str(root)
        self.manifest = manifest
        qwen_semantic = {
            key: manifest.get("qwen", {}).get(key)
            for key in (
                "repo_id",
                "revision",
                "architecture",
                "hidden_size",
                "num_hidden_layers",
                "feature_layer",
                "layer_norm_eps",
                "max_length",
                "padding_side",
                "truncation_side",
                "chat_template",
            )
        }
        qwen_semantic["padding_side"] = qwen_semantic.get("padding_side") or "right"
        qwen_semantic["truncation_side"] = qwen_semantic.get("truncation_side") or "right"
        build_semantic = {
            key: manifest.get("build", {}).get(key)
            for key in (
                "transformers_version",
                "torch_version",
                "model_dtype",
                "attn_implementation",
                "encoder_batch_size",
                "fixed_batch_padding",
            )
        }
        semantic_manifest = {
            "format": manifest.get("format"),
            "version": manifest.get("version"),
            "qwen": qwen_semantic,
            "build": build_semantic,
            "storage_dtype": manifest.get("storage_dtype"),
        }
        self.semantic_fingerprint = hashlib.sha256(
            json.dumps(semantic_manifest, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        content_manifest = {
            "format": manifest.get("format"),
            "version": manifest.get("version"),
            "captions_sha256": manifest.get("captions_sha256"),
            "num_captions": manifest.get("num_captions"),
            "total_qwen_tokens": int(tokens.shape[0]),
        }
        self.content_fingerprint = hashlib.sha256(
            json.dumps(content_manifest, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        if expected_fingerprint and self.semantic_fingerprint != str(expected_fingerprint):
            raise RuntimeError(
                "Text cache semantic fingerprint mismatch: "
                f"cache={self.semantic_fingerprint}, expected={expected_fingerprint}"
            )
        self.width = qwen_dim
        self.max_length = qwen_max_length
        storage_cfg = manifest.get("storage_dtype", {})
        self.qwen_storage_dtype = (
            str(storage_cfg.get("qwen", "float16"))
            if isinstance(storage_cfg, dict)
            else str(storage_cfg)
        )
        self._caption_to_id = caption_to_id
        self._qwen_tokens = tokens
        self._qwen_offsets = offsets
        # Qwen3 is causal, so only the final position has attended over the whole
        # caption; its feature is the conventional sentence vector for a decoder
        # LM, and is the same construction CLIP uses (its pooler_output is the
        # EOT position's hidden state).  It is already in the cached stream, so
        # exposing it costs nothing and needs no rebuild.  Unlike the LLM2Vec
        # pooled vector, nothing trained it to be a sentence embedding -- that
        # difference is a property of the encoder being compared, not a defect.
        self.use_pooled = bool(use_pooled)
        self.output_dim = qwen_dim if self.use_pooled else None
        # No content spans: this recipe renders the caption with Qwen's chat
        # template and pads on the RIGHT, so the caption sits *between* the
        # wrapper tokens rather than at the end.  The trailing-span form the
        # LLM2Vec caches use cannot express that, and a wrong span is silent --
        # so this provider reports None and the model refuses
        # --text_refiner_pool content_span outright.
        self._spans = None
        self.register_buffer("_device_anchor", torch.empty(0), persistent=False)

    @property
    def device(self) -> torch.device:
        return self._device_anchor.device

    def cache_summary(self) -> Dict[str, object]:
        # ``clip_*`` keys are kept for trainer-log compatibility; this provider
        # has no CLIP branch so they are recorded as empty/zero.
        return {
            "path": self.cache_path,
            "format": self.manifest["format"],
            "version": int(self.manifest["version"]),
            "num_captions": int(self.manifest["num_captions"]),
            "clip_hidden_size": 0,
            "qwen_hidden_size": int(self.width),
            "qwen_max_length": int(self.max_length),
            "qwen_feature_layer": str(self.manifest.get("qwen", {}).get("feature_layer", "")),
            "qwen_num_hidden_layers": int(self.manifest.get("qwen", {}).get("num_hidden_layers", 0)),
            "qwen_storage_dtype": self.qwen_storage_dtype,
            "total_qwen_tokens": int(self._qwen_tokens.shape[0]),
            "semantic_fingerprint": self.semantic_fingerprint,
            "content_fingerprint": self.content_fingerprint,
            "captions_sha256": str(self.manifest.get("captions_sha256", "")),
            "clip_revision": "",
            "qwen_revision": str(self.manifest.get("qwen", {}).get("revision", "")),
        }

    def validate_coverage(self, captions: Iterable[str]) -> Dict[str, object]:
        requested = set(_as_text_list(captions))
        missing = sorted(text for text in requested if text not in self._caption_to_id)
        if missing:
            examples = ", ".join(repr(text) for text in missing[:5])
            raise RuntimeError(
                f"Text cache coverage failure: {len(missing)} of {len(requested)} unique captions "
                f"are missing from {self.cache_path}; examples: {examples}"
            )
        return {
            "requested_unique_captions": len(requested),
            "cached_captions": len(self._caption_to_id),
            "missing": 0,
        }

    def _resolve_texts(
        self,
        raw_text: Iterable[str],
        drop_prob: float,
        force_drop: bool,
    ) -> List[str]:
        texts = _as_text_list(raw_text)
        if force_drop:
            return [""] * len(texts)
        if drop_prob > 0.0:
            keep = torch.rand(len(texts), device=self.device) >= float(drop_prob)
            return [text if bool(keep[idx].item()) else "" for idx, text in enumerate(texts)]
        return texts

    @torch.no_grad()
    def encode(
        self,
        raw_text: Iterable[str],
        drop_prob: float = 0.0,
        force_drop: bool = False,
    ) -> TextCondition:
        texts = self._resolve_texts(raw_text, drop_prob=drop_prob, force_drop=force_drop)
        missing = [text for text in texts if text not in self._caption_to_id]
        if missing:
            examples = ", ".join(repr(text) for text in missing[:3])
            raise KeyError(
                f"{len(missing)} caption(s) are missing from text cache {self.cache_path}; "
                f"examples: {examples}. Rebuild/extend the cache before training or evaluation."
            )

        ids = np.asarray([self._caption_to_id[text] for text in texts], dtype=np.int64)
        lengths = [
            int(self._qwen_offsets[idx + 1]) - int(self._qwen_offsets[idx])
            for idx in ids.tolist()
        ]
        max_length = max(lengths)
        tokens_cpu = torch.zeros(
            len(texts), max_length, self.width,
            dtype=torch.float32,
        )
        padding_mask_cpu = torch.ones(
            len(texts), max_length,
            dtype=torch.bool,
        )
        for row, (cache_id, length) in enumerate(zip(ids.tolist(), lengths)):
            start = int(self._qwen_offsets[cache_id])
            end = start + length
            token_np = np.asarray(self._qwen_tokens[start:end]).copy()
            token_tensor = torch.from_numpy(token_np)
            if self.qwen_storage_dtype == "bfloat16":
                if token_tensor.dtype != torch.uint16:
                    raise RuntimeError(
                        f"Expected uint16 storage for cached bfloat16 Qwen features, got {token_tensor.dtype}"
                    )
                token_tensor = token_tensor.view(torch.bfloat16)
            token_tensor = token_tensor.to(dtype=torch.float32)
            tokens_cpu[row, :length].copy_(token_tensor)
            padding_mask_cpu[row, :length] = False
        tokens = tokens_cpu.to(device=self.device, non_blocking=False)
        padding_mask = padding_mask_cpu.to(device=self.device, non_blocking=False)
        content_mask = _content_mask_from_spans(
            self._spans, ids.tolist(), lengths, len(texts), max_length
        )
        if content_mask is not None:
            content_mask = content_mask.to(device=self.device, non_blocking=False)
        pooled = None
        if self.use_pooled:
            # Rows are left-aligned, so position length-1 is the final real
            # token of that caption -- the only one that has attended over the
            # whole thing under Qwen3's causal mask.
            last = torch.as_tensor(
                [int(n) - 1 for n in lengths], device=tokens.device, dtype=torch.long
            )
            pooled = tokens[torch.arange(tokens.shape[0], device=tokens.device), last]
        return TextCondition(
            pooled=pooled,
            tokens=tokens,
            padding_mask=padding_mask,
            content_mask=content_mask,
        )

    def forward(
        self,
        raw_text: Iterable[str],
        drop_prob: float = 0.0,
        force_drop: bool = False,
    ) -> TextCondition:
        return self.encode(raw_text, drop_prob=drop_prob, force_drop=force_drop)


LLM2VEC_CACHE_FORMAT = "momask_llm2vec_cache"
LLM2VEC_CACHE_VERSION = 1


class CachedLLM2VecTextEncoder(nn.Module):
    """Read LLM2Vec (bidirectional Llama-3) features from a memory-mapped cache.

    The cache stores both the per-token bidirectional hidden states and the
    masked-mean pooled sentence vector, so one cache serves both conditioning
    variants.  ``use_pooled`` selects between them: when False the encoder
    reports ``output_dim = None`` and returns ``pooled=None``, so the model
    builds no global text branch and AdaLN sees only the timestep.
    """

    def __init__(
        self,
        cache_path: str,
        use_pooled: bool = False,
        expected_max_length: int = 0,
        expected_fingerprint: str = "",
        require_empty_prompt: bool = True,
        require_complete: bool = True,
    ) -> None:
        super().__init__()
        root = Path(cache_path).expanduser().resolve()
        manifest_path = root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Text cache manifest not found: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("format") != LLM2VEC_CACHE_FORMAT:
            raise RuntimeError(
                f"Unsupported text cache format {manifest.get('format')!r}; "
                f"expected {LLM2VEC_CACHE_FORMAT!r}"
            )
        if int(manifest.get("version", -1)) != LLM2VEC_CACHE_VERSION:
            raise RuntimeError(f"Unsupported text cache version {manifest.get('version')}")
        # "validating" is the builder's own pass: it writes that status, verifies
        # the bytes it just wrote through this very class, and only then marks
        # the cache complete -- so a trainer polling for "complete" can never be
        # handed a cache that has not passed validation.  Only that pass may
        # set require_complete=False.
        allowed = ("complete",) if require_complete else ("complete", "validating")
        if manifest.get("status") not in allowed:
            raise RuntimeError(
                f"Text cache is not complete: {manifest_path} (status={manifest.get('status')!r})"
            )

        cfg = manifest.get("encoder", {})
        max_length = int(cfg.get("max_length", 0))
        if expected_max_length > 0 and max_length != int(expected_max_length):
            raise RuntimeError(
                f"Text cache max_length={max_length} does not match "
                f"configured value {int(expected_max_length)}"
            )

        files = manifest.get("files", {})
        captions_path = root / files.get("captions", "captions.jsonl")
        tokens_path = root / files.get("tokens", "tokens.npy")
        offsets_path = root / files.get("offsets", "offsets.npy")
        pooled_path = root / files.get("pooled", "pooled.npy")
        for path in (captions_path, tokens_path, offsets_path, pooled_path):
            if not path.is_file():
                raise FileNotFoundError(f"Text cache file not found: {path}")

        captions: List[str] = []
        with captions_path.open("r", encoding="utf-8") as handle:
            for expected_id, line in enumerate(handle):
                payload = json.loads(line)
                if int(payload["id"]) != expected_id:
                    raise RuntimeError(
                        f"Non-contiguous caption id in {captions_path}: "
                        f"expected {expected_id}, got {payload['id']}"
                    )
                captions.append(str(payload["text"]))
        if int(manifest.get("num_captions", -1)) != len(captions):
            raise RuntimeError("Text cache caption count mismatch")
        caption_to_id: Dict[str, int] = {}
        for idx, caption in enumerate(captions):
            if caption in caption_to_id:
                raise RuntimeError(f"Duplicate caption in text cache: {caption!r}")
            caption_to_id[caption] = idx
        if require_empty_prompt and "" not in caption_to_id:
            raise RuntimeError("Text cache must contain the empty prompt for standard CFG")

        tokens = np.load(tokens_path, mmap_mode="r", allow_pickle=False)
        offsets = np.load(offsets_path, mmap_mode="r", allow_pickle=False)
        pooled = np.load(pooled_path, mmap_mode="r", allow_pickle=False)
        hidden = int(cfg.get("hidden_size", 0))
        if tokens.ndim != 2 or tokens.shape[1] != hidden:
            raise RuntimeError(f"Token cache shape mismatch: {tokens.shape}, expected [N,{hidden}]")
        if pooled.shape != (len(captions), hidden):
            raise RuntimeError(
                f"Pooled cache shape mismatch: {pooled.shape}, expected {(len(captions), hidden)}"
            )
        if offsets.shape != (len(captions) + 1,):
            raise RuntimeError(f"Offsets shape mismatch: {offsets.shape}")
        if int(offsets[0]) != 0 or int(offsets[-1]) != int(tokens.shape[0]):
            raise RuntimeError("Offsets do not span token storage")
        lengths = np.diff(offsets)
        if np.any(lengths <= 0) or np.any(lengths > max_length):
            raise RuntimeError(
                f"Invalid cached lengths: min={int(lengths.min())}, max={int(lengths.max())}"
            )

        self.cache_path = str(root)
        self.manifest = manifest
        self.use_pooled = bool(use_pooled)
        semantic = {
            "format": manifest.get("format"),
            "version": manifest.get("version"),
            "encoder": {
                key: cfg.get(key)
                for key in (
                    "recipe", "base_model", "mntp_model", "supervised_model",
                    "hidden_size", "feature_layer", "pooling", "instruction", "max_length",
                )
            },
            "build": manifest.get("build"),
            "storage_dtype": manifest.get("storage_dtype"),
        }
        self.semantic_fingerprint = hashlib.sha256(
            json.dumps(semantic, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        self.content_fingerprint = hashlib.sha256(
            json.dumps(
                {
                    "captions_sha256": manifest.get("captions_sha256"),
                    "num_captions": manifest.get("num_captions"),
                    "total_tokens": int(tokens.shape[0]),
                },
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        if expected_fingerprint and self.semantic_fingerprint != str(expected_fingerprint):
            raise RuntimeError(
                "Text cache semantic fingerprint mismatch: "
                f"cache={self.semantic_fingerprint}, expected={expected_fingerprint}"
            )

        self.width = hidden
        self.output_dim = hidden if self.use_pooled else None
        self.max_length = max_length
        storage = manifest.get("storage_dtype", {})
        self.storage_dtype = str(storage.get("tokens", "float16")) if isinstance(storage, dict) else str(storage)
        self._caption_to_id = caption_to_id
        self._tokens = tokens
        self._offsets = offsets
        self._pooled = pooled
        self._spans = _load_content_spans(root, files, len(captions))
        if self._spans is not None:
            # The loader only knows the array's shape; cross-check it against
            # this cache's own sequence lengths so a spans.npy copied from a
            # different cache (same caption list -> same shape) is rejected
            # instead of mismarking every row.
            row_lengths = np.diff(np.asarray(offsets))
            if np.any(self._spans > row_lengths):
                bad = int(np.argmax(self._spans > row_lengths))
                raise RuntimeError(
                    f"{root / 'spans.npy'} disagrees with this cache: caption {bad} has span "
                    f"{int(self._spans[bad])} but only {int(row_lengths[bad])} cached tokens. "
                    "The spans file does not belong to this cache."
                )
        self.provides_content_mask = self._spans is not None
        self.register_buffer("_device_anchor", torch.empty(0), persistent=False)

    @property
    def device(self) -> torch.device:
        return self._device_anchor.device

    def cache_summary(self) -> Dict[str, object]:
        cfg = self.manifest.get("encoder", {})
        return {
            "path": self.cache_path,
            "format": self.manifest["format"],
            "version": int(self.manifest["version"]),
            "num_captions": int(self.manifest["num_captions"]),
            "hidden_size": int(self.width),
            "use_pooled": self.use_pooled,
            "max_length": int(self.max_length),
            "feature_layer": str(cfg.get("feature_layer", "")),
            "pooling": str(cfg.get("pooling", "")),
            "supervised_model": str(cfg.get("supervised_model", "")),
            "total_tokens": int(self._tokens.shape[0]),
            "storage_dtype": self.storage_dtype,
            "semantic_fingerprint": self.semantic_fingerprint,
            "content_fingerprint": self.content_fingerprint,
            "captions_sha256": str(self.manifest.get("captions_sha256", "")),
            "clip_revision": "",
            "qwen_revision": "",
        }

    def validate_coverage(self, captions: Iterable[str]) -> Dict[str, object]:
        requested = set(_as_text_list(captions))
        missing = sorted(text for text in requested if text not in self._caption_to_id)
        if missing:
            examples = ", ".join(repr(text) for text in missing[:5])
            raise RuntimeError(
                f"Text cache coverage failure: {len(missing)} of {len(requested)} unique captions "
                f"are missing from {self.cache_path}; examples: {examples}"
            )
        return {
            "requested_unique_captions": len(requested),
            "cached_captions": len(self._caption_to_id),
            "missing": 0,
        }

    def _decode(self, raw: np.ndarray) -> torch.Tensor:
        tensor = torch.from_numpy(np.asarray(raw).copy())
        if self.storage_dtype == "bfloat16":
            if tensor.dtype != torch.uint16:
                raise RuntimeError(f"Expected uint16 storage for bfloat16, got {tensor.dtype}")
            tensor = tensor.view(torch.bfloat16)
        return tensor.to(dtype=torch.float32)

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
        missing = [text for text in texts if text not in self._caption_to_id]
        if missing:
            examples = ", ".join(repr(text) for text in missing[:3])
            raise KeyError(
                f"{len(missing)} caption(s) are missing from text cache {self.cache_path}; "
                f"examples: {examples}. Rebuild/extend the cache before training or evaluation."
            )

        ids = [self._caption_to_id[text] for text in texts]
        lengths = [int(self._offsets[i + 1]) - int(self._offsets[i]) for i in ids]
        max_len = max(lengths)
        tokens_cpu = torch.zeros(len(texts), max_len, self.width, dtype=torch.float32)
        padding_cpu = torch.ones(len(texts), max_len, dtype=torch.bool)
        for row, (cache_id, length) in enumerate(zip(ids, lengths)):
            start = int(self._offsets[cache_id])
            tokens_cpu[row, :length].copy_(self._decode(self._tokens[start: start + length]))
            padding_cpu[row, :length] = False
        tokens = tokens_cpu.to(device=self.device)
        padding_mask = padding_cpu.to(device=self.device)
        pooled = None
        if self.use_pooled:
            pooled = self._decode(self._pooled[np.asarray(ids, dtype=np.int64)]).to(device=self.device)
        content_mask = _content_mask_from_spans(
            self._spans, ids, lengths, len(texts), max_len
        )
        if content_mask is not None:
            content_mask = content_mask.to(device=self.device)
        return TextCondition(
            pooled=pooled,
            tokens=tokens,
            padding_mask=padding_mask,
            content_mask=content_mask,
        )

    def forward(
        self,
        raw_text: Iterable[str],
        drop_prob: float = 0.0,
        force_drop: bool = False,
    ) -> TextCondition:
        return self.encode(raw_text, drop_prob=drop_prob, force_drop=force_drop)


def build_text_encoder(
    encoder_type: str,
    *,
    clip_version: str,
    clip_path: Optional[str],
    kv_root: Optional[str],
    text_cache_path: Optional[str],
    qwen_max_length: int,
    text_cache_fingerprint: str = "",
    clip_hf_path: Optional[str] = None,
) -> nn.Module:
    if encoder_type == "clip":
        return FrozenCLIPTextEncoder(
            clip_version=clip_version,
            clip_path=clip_path,
            kv_root=kv_root,
        )
    if encoder_type == "clip_large":
        return HFCLIPTextEncoder(clip_hf_path=clip_hf_path, use_pooled=True)
    if encoder_type == "clip_large_tokenonly":
        return HFCLIPTextEncoder(clip_hf_path=clip_hf_path, use_pooled=False)
    if encoder_type == "dual_clip":
        return DualCLIPTextEncoder(
            clip_version=clip_version,
            clip_path=clip_path,
            clip_hf_path=clip_hf_path,
            kv_root=kv_root,
        )
    if encoder_type == "clip_qwen_cache":
        if not text_cache_path:
            raise ValueError("text_encoder_type='clip_qwen_cache' requires --text_cache_path")
        return CachedCLIPQwenTextEncoder(
            cache_path=text_cache_path,
            expected_qwen_max_length=qwen_max_length,
            expected_fingerprint=text_cache_fingerprint,
            require_empty_prompt=True,
        )
    if encoder_type in ("llm2vec_cache", "llm2vec_pooled_cache"):
        if not text_cache_path:
            raise ValueError(f"text_encoder_type={encoder_type!r} requires --text_cache_path")
        return CachedLLM2VecTextEncoder(
            cache_path=text_cache_path,
            use_pooled=(encoder_type == "llm2vec_pooled_cache"),
            expected_max_length=qwen_max_length,
            expected_fingerprint=text_cache_fingerprint,
            require_empty_prompt=True,
        )
    if encoder_type in ("qwen_normavg_cache", "qwen_normavg_pooled_cache"):
        if not text_cache_path:
            raise ValueError("text_encoder_type='qwen_normavg_cache' requires --text_cache_path")
        return CachedQwenNormAvgTextEncoder(
            cache_path=text_cache_path,
            expected_qwen_max_length=qwen_max_length,
            expected_fingerprint=text_cache_fingerprint,
            require_empty_prompt=True,
            use_pooled=encoder_type == "qwen_normavg_pooled_cache",
        )
    raise ValueError(f"Unknown text encoder type: {encoder_type}")
