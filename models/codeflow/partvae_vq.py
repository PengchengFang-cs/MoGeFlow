"""Frozen part-aware continuous VAE tokenizer/decoder wrapper.

This backend has no codebooks.  It exposes the PartVAE posterior mean as the
CodeFlow target latent and the frozen PartVAE decoder as the only decode path.

Contract (MEMORY.md, "PartVAE Continuous CodeFlow Baseline"):
  - frozen PartVAE encoder/decoder
  - ``encode(motion)`` returns ``(None, latent)``
  - latent shape: ``[B, T/4, P, D]`` (HumanML3D: ``[B, T/4, 6, 128]``)
  - target latent: posterior mean, ``sample=False``
  - decode path: ``decode_embeddings(latent)`` only
  - no ids, no codebook, no nearest projection, no code token metrics

The PartVAE code lives in an external repository whose top-level package names
(``models``, ``options``, ``utils``) collide with this repository, so its
modules are loaded from file with the colliding names only masked in
``sys.modules`` for the duration of each module exec.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import types
from argparse import Namespace
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


PARTVAE_CODE_ROOT_ENV = "PARTVAE_CODE_ROOT"
# a4cde8a clone of https://github.com/CHDTevior/part-aware-partvae; the HF
# artifact repo (partvae-overlap-128-20260614) ships weights/configs but not
# models/encdec.py, so the code repo is required for the architecture.
DEFAULT_PARTVAE_CODE_ROOTS = (
    "/scratch/pf2m24/projects/Umdd/external_repos/part-aware-partvae",
    "/iridisfs/scratch/pf2m24/projects/Umdd/external_repos/part-aware-partvae",
)

_MISSING = object()
_PARTVAE_MODULE_CACHE: Dict[str, types.ModuleType] = {}


def resolve_partvae_code_root(code_root: Optional[str] = None) -> Path:
    candidates = []
    if code_root:
        candidates.append(code_root)
    env_root = os.environ.get(PARTVAE_CODE_ROOT_ENV, "")
    if env_root:
        candidates.append(env_root)
    candidates.extend(DEFAULT_PARTVAE_CODE_ROOTS)
    for candidate in candidates:
        path = Path(candidate).expanduser()
        if (path / "models" / "PartVAE" / "part_vae.py").is_file():
            return path.resolve()
    raise FileNotFoundError(
        "PartVAE code repository not found. Checked: "
        + ", ".join(str(Path(c).expanduser()) for c in candidates)
        + f". Set ${PARTVAE_CODE_ROOT_ENV} or pass code_root explicitly."
    )


def _exec_module_from_file(name: str, path: Path, masked: Dict[str, types.ModuleType]) -> types.ModuleType:
    """Execute ``path`` as module ``name`` with ``masked`` entries in sys.modules.

    Every masked name (including ``name`` itself) is restored to its previous
    binding afterwards, so the repository's own ``models`` package is never
    left shadowed.
    """
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot build import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    overrides = dict(masked)
    overrides[name] = module
    saved = {}
    try:
        for mod_name, mod in overrides.items():
            saved[mod_name] = sys.modules.get(mod_name, _MISSING)
            sys.modules[mod_name] = mod
        spec.loader.exec_module(module)
    finally:
        for mod_name, original in saved.items():
            if original is _MISSING:
                sys.modules.pop(mod_name, None)
            else:
                sys.modules[mod_name] = original
    return module


def load_partvae_module(code_root: Optional[str] = None) -> types.ModuleType:
    """Load the external ``models/PartVAE/part_vae.py`` under masked imports."""
    root = resolve_partvae_code_root(code_root)
    cache_key = str(root)
    if cache_key in _PARTVAE_MODULE_CACHE:
        return _PARTVAE_MODULE_CACHE[cache_key]

    models_dir = root / "models"
    resnet_path = models_dir / "resnet.py"
    encdec_path = models_dir / "encdec.py"
    part_vae_path = models_dir / "PartVAE" / "part_vae.py"
    for path in (resnet_path, encdec_path, part_vae_path):
        if not path.is_file():
            raise FileNotFoundError(f"PartVAE source file not found: {path}")

    fake_models_pkg = types.ModuleType("models")
    fake_models_pkg.__path__ = [str(models_dir)]
    masked: Dict[str, types.ModuleType] = {"models": fake_models_pkg}
    resnet_mod = _exec_module_from_file("models.resnet", resnet_path, masked)
    fake_models_pkg.resnet = resnet_mod
    masked["models.resnet"] = resnet_mod
    encdec_mod = _exec_module_from_file("models.encdec", encdec_path, masked)
    fake_models_pkg.encdec = encdec_mod
    masked["models.encdec"] = encdec_mod
    part_vae_mod = _exec_module_from_file("models.PartVAE.part_vae", part_vae_path, masked)

    _PARTVAE_MODULE_CACHE[cache_key] = part_vae_mod
    return part_vae_mod


def _lengths_to_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    return torch.arange(max_len, device=lengths.device)[None, :] < lengths[:, None]


class PartVAEContinuousTokenizer(nn.Module):
    """Frozen wrapper around the external part-aware continuous VAE.

    Public shapes:
      - motion: [B, F, 263], normalized with the tokenizer mean/std
      - latent: [B, T, P, D] posterior means (T = F // downsample_factor)

    There are no code ids anywhere on this boundary: ``encode`` returns
    ``(None, latent)`` and ``supports_code_metrics`` is False.
    """

    supports_code_metrics = False

    def __init__(
        self,
        checkpoint_path: str,
        config_path: Optional[str] = None,
        partition_path: Optional[str] = None,
        code_dim: Optional[int] = None,
        code_root: Optional[str] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        ckpt_path = Path(checkpoint_path).expanduser().resolve()
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"PartVAE checkpoint not found: {ckpt_path}")
        cfg_path = Path(config_path).expanduser().resolve() if config_path else (ckpt_path.parent / "config.json")
        if not cfg_path.is_file():
            raise FileNotFoundError(f"PartVAE config not found: {cfg_path}")
        part_path = (
            Path(partition_path).expanduser().resolve()
            if partition_path
            else (ckpt_path.parent / "skeleton_partition.json")
        )
        if not part_path.is_file():
            raise FileNotFoundError(f"PartVAE partition not found: {part_path}")

        with cfg_path.open("r", encoding="utf-8") as f:
            config = json.load(f)
        with part_path.open("r", encoding="utf-8") as f:
            self.partition = json.load(f)

        dataname = str(config.get("dataname", "t2m"))
        latent_dim = int(config.get("latent_dim", 128))
        if code_dim is not None and int(code_dim) != latent_dim:
            raise ValueError(
                f"Config code_dim={int(code_dim)} does not match PartVAE latent_dim={latent_dim}"
            )

        part_vae_mod = load_partvae_module(code_root)
        vae_args = Namespace(dataname=dataname, partition_file=str(part_path))
        model = part_vae_mod.PartVAE_263(
            vae_args,
            latent_dim=latent_dim,
            encoder_emb_width=int(config.get("encoder_emb_width", 128)),
            down_t=int(config.get("down_t", 2)),
            stride_t=int(config.get("stride_t", 2)),
            width=int(config.get("width", 512)),
            depth=int(config.get("depth", 3)),
            dilation_growth_rate=int(config.get("dilation_growth_rate", 3)),
            activation=str(config.get("vae_act", "relu")),
            norm=config.get("vae_norm"),
            kl_reduction=str(config.get("kl_reduction", "sum")),
        )

        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        state_dict = ckpt["net"] if isinstance(ckpt, dict) and "net" in ckpt else ckpt
        # The training checkpoint wraps PartVAE_263 as HumanPartVAE.partvae.
        prefix = "partvae."
        if any(key.startswith(prefix) for key in state_dict):
            state_dict = {
                key[len(prefix):]: value for key, value in state_dict.items() if key.startswith(prefix)
            }
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                "PartVAE load mismatch: "
                f"missing={list(missing)[:8]} unexpected={list(unexpected)[:8]}"
            )
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)

        self.vq_model = model
        self.checkpoint_path = ckpt_path
        self.config_path = cfg_path
        self.partition_path = part_path
        self.dataname = dataname
        self.input_dim = int(model.output_dim)
        self.num_parts = int(model.num_parts)
        self.num_codes = 0
        self.code_dim = latent_dim
        self.downsample_factor = int(config.get("stride_t", 2)) ** int(config.get("down_t", 2))
        self.target_mode = "posterior_mean"
        self.latent_source = "posterior_mean"

        if device is not None:
            self.to(device)
        self._verify_static_contract()

    def _verify_static_contract(self) -> None:
        for attr in ("encode", "decode", "partSeg", "limb_encoders", "mu_heads", "decoder"):
            if not hasattr(self.vq_model, attr):
                raise RuntimeError(f"PartVAE model is missing .{attr}")
        if len(self.vq_model.partSeg) != self.num_parts:
            raise RuntimeError(
                f"Expected {self.num_parts} parts, model has {len(self.vq_model.partSeg)}"
            )
        if int(self.vq_model.latent_dim) != self.code_dim:
            raise RuntimeError(
                f"PartVAE latent_dim={int(self.vq_model.latent_dim)} != code_dim={self.code_dim}"
            )

    @property
    def device(self) -> torch.device:
        return next(self.vq_model.parameters()).device

    def _check_motion(self, motion: torch.Tensor) -> None:
        if motion.ndim != 3:
            raise ValueError(f"Expected motion [B, F, {self.input_dim}], got shape {tuple(motion.shape)}")
        if motion.shape[-1] != self.input_dim:
            raise ValueError(f"Expected {self.input_dim}-dim motion features, got {motion.shape[-1]}")

    def _check_latent(self, z: torch.Tensor) -> None:
        if z.ndim != 4:
            raise ValueError(f"Expected latent [B, T, P, D], got shape {tuple(z.shape)}")
        if z.shape[2] != self.num_parts or z.shape[3] != self.code_dim:
            raise ValueError(
                f"Expected latent part/code dims ({self.num_parts}, {self.code_dim}), "
                f"got ({z.shape[2]}, {z.shape[3]})"
            )

    @torch.no_grad()
    def encode(self, motion: torch.Tensor) -> Tuple[None, torch.Tensor]:
        """Return ``(None, posterior_mean)``; sampling is deliberately unavailable."""
        self._check_motion(motion)
        latent = self.vq_model.encode(motion, sample=False)
        return None, latent

    def decode_embeddings(self, z: torch.Tensor) -> torch.Tensor:
        """Decode continuous part latents [B, T, P, D] through the frozen decoder.

        Gradients are allowed to flow to ``z``; the decoder parameters remain
        frozen because they have ``requires_grad=False``.
        """
        self._check_latent(z)
        return self.vq_model.decode(z)

    # --- codebook API stubs -------------------------------------------------
    # These exist so a misconfigured run fails with a clear message instead of
    # an AttributeError deep inside the flow model.  The trainer enforces
    # terminal_mode='none' and latent_norm_mode='empirical' for this backend,
    # so none of them is reachable on the supported path.

    def _no_codebook(self, method: str) -> RuntimeError:
        return RuntimeError(
            f"{method} is unavailable: the partvae_continuous backend has no codebooks. "
            "Use terminal_mode='none' and latent_norm_mode='empirical'."
        )

    def ids_to_embeddings(self, ids: torch.Tensor) -> torch.Tensor:
        raise self._no_codebook("ids_to_embeddings")

    def codebook_tied_logits(self, z: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
        raise self._no_codebook("codebook_tied_logits")

    def nearest_ids(self, z: torch.Tensor) -> torch.Tensor:
        raise self._no_codebook("nearest_ids")

    def decode_ids(self, ids: torch.Tensor) -> torch.Tensor:
        raise self._no_codebook("decode_ids")

    def code_id_distances(self, target_ids: torch.Tensor, pred_ids: torch.Tensor):
        raise self._no_codebook("code_id_distances")

    @torch.no_grad()
    def verify_contract(
        self,
        motion: Optional[torch.Tensor] = None,
        lengths: Optional[torch.Tensor] = None,
        max_samples: int = 2,
    ) -> Dict[str, object]:
        """Run structural and optional real-motion checks for the frozen VAE boundary."""
        self._verify_static_contract()
        summary: Dict[str, object] = {
            "backend": "partvae_continuous",
            "checkpoint_path": str(self.checkpoint_path),
            "config_path": str(self.config_path),
            "partition_path": str(self.partition_path),
            "target_mode": self.target_mode,
            "latent_source": self.latent_source,
            "num_parts": self.num_parts,
            "num_codes": self.num_codes,
            "code_dim": self.code_dim,
            "downsample_factor": self.downsample_factor,
            "supports_code_metrics": bool(self.supports_code_metrics),
        }

        if motion is None:
            return summary

        self._check_motion(motion)
        motion = motion[:max_samples].to(self.device)
        if lengths is not None:
            lengths = lengths[: motion.shape[0]].to(self.device).long()

        ids, latent = self.encode(motion)
        if ids is not None:
            raise RuntimeError("partvae_continuous encode must return ids=None")
        self._check_latent(latent)
        if not torch.isfinite(latent).all():
            raise RuntimeError("PartVAE posterior mean contains non-finite values")
        expected_latent_len = motion.shape[1] // self.downsample_factor
        if latent.shape[1] != expected_latent_len:
            raise RuntimeError(
                f"Latent length {latent.shape[1]} does not match frames//{self.downsample_factor}"
                f"={expected_latent_len}"
            )
        _z, mu, _logvar = self.vq_model.encode(motion, sample=False, return_stats=True)
        if not torch.equal(latent, mu):
            raise RuntimeError("encode(sample=False) is not the posterior mean")

        decoded = self.decode_embeddings(latent)
        if decoded.ndim != 3 or decoded.shape[0] != motion.shape[0] or decoded.shape[-1] != motion.shape[-1]:
            raise RuntimeError(
                f"Decoded motion shape {tuple(decoded.shape)} is incompatible with input {tuple(motion.shape)}"
            )
        common_len = min(decoded.shape[1], motion.shape[1])
        recon_abs = (decoded[:, :common_len] - motion[:, :common_len]).abs()
        recon_sq = (decoded[:, :common_len] - motion[:, :common_len]).square()
        if lengths is not None:
            frame_mask = _lengths_to_mask(lengths.clamp(min=1, max=common_len), common_len)
            weight = frame_mask[:, :, None].to(recon_abs.dtype)
            denom = weight.sum().mul(motion.shape[-1]).clamp_min(1.0)
            recon_l1 = (recon_abs * weight).sum() / denom
            recon_mse = (recon_sq * weight).sum() / denom
        else:
            recon_l1 = recon_abs.mean()
            recon_mse = recon_sq.mean()
        if not torch.isfinite(recon_l1) or not torch.isfinite(recon_mse):
            raise RuntimeError("PartVAE round-trip reconstruction produced non-finite error")

        summary.update({
            "input_motion_shape": list(motion.shape),
            "latent_shape": list(latent.shape),
            "decoded_motion_shape": list(decoded.shape),
            "latent_abs_mean": float(latent.abs().mean().detach().cpu()),
            "latent_std": float(latent.float().std().detach().cpu()),
            "roundtrip_l1": float(recon_l1.detach().cpu()),
            "roundtrip_mse": float(recon_mse.detach().cpu()),
        })
        return summary


def load_partvae_continuous_tokenizer(
    checkpoint_path: str,
    config_path: Optional[str] = None,
    partition_path: Optional[str] = None,
    code_root: Optional[str] = None,
    device: Optional[torch.device] = None,
) -> PartVAEContinuousTokenizer:
    return PartVAEContinuousTokenizer(
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        partition_path=partition_path,
        code_root=code_root,
        device=device,
    )
