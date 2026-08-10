"""Text-to-motion generation by continuous flow over part-specific codebooks."""

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dit_blocks import BidirectionalTextRefiner, MotionTextDiT, TimestepEmbedder
from .text_encoder import TextCondition, build_text_encoder
from .vq_tokenizers import build_codeflow_tokenizer


TEXT_REFINER_POOL_MODES = ("all_tokens", "content_span")

# Where the text encoder's pooled sentence vector goes.
#   "modulation" -- added to the timestep embedding, so it drives AdaLN: one
#                   global (scale, shift) per block, applied to every position.
#   "token"      -- projected and prepended to the text token stream, so motion
#                   positions attend to it selectively and AdaLN keeps the
#                   timestep alone.
# Default "modulation" reproduces every checkpoint trained before the flag.
TEXT_POOLED_MODES = ("modulation", "token")

# The method decodes the flow's continuous output through the tokenizer decoder
# directly.  This is not a configurable choice and there is no CLI flag for it
# anywhere: projecting to the nearest code first ("nearest"/"ids") is the
# terminal-projection baseline the method exists to replace, and every place it
# was reachable -- a training flag, an eval fallback, a generation default --
# produced a silent nearest result that was then compared against continuous
# ones.  Reintroducing a switch reintroduces that failure.  The decode ablation
# for the paper is produced by a standalone script that calls terminal_ids() and
# tokenizer.decode_ids() directly, not by flipping anything here.
DECODE_MODE = "continuous"
# PERMANENT (2026-08-09): the backbone head predicts the clean endpoint x0.
# Velocity never leaves the sampler -- it is derived there as (x0 - z_t)/(1 - t)
# for ODE integration only.  There is no velocity-prediction mode.
PREDICTION_TYPE = "x0"


@dataclass
class MotionCodeFlowConfig:
    kv_root: str = "."
    vq_backend: str = "kv_part"
    vq_checkpoint: Optional[str] = None
    vq_partition: Optional[str] = None
    vq_opt_path: Optional[str] = None
    kv_part_target_mode: str = "codebook"
    rvq_target_mode: str = "stage"
    clip_version: str = "ViT-B/32"
    clip_path: Optional[str] = None
    clip_hf_path: Optional[str] = None
    text_encoder_type: str = "clip"
    text_cache_path: Optional[str] = None
    text_cache_fingerprint: str = ""
    qwen_max_length: int = 128
    text_refiner_depth: int = 0
    text_refiner_mlp_ratio: float = 4.0
    # Which positions the refiner's pooled summary averages over.  Both this
    # default and the argparse one are "all_tokens", so a config restored from a
    # checkpoint trained before the content-span fix -- which has no such key --
    # reproduces how it was trained.  Excluding chat-wrapper scaffolding is
    # therefore opt-in: a launch script that wants it must pass
    # --text_refiner_pool content_span explicitly.
    text_refiner_pool: str = "all_tokens"
    text_pooled_mode: str = "modulation"

    representation: str = "t_concat"
    code_dim: int = 128
    num_parts: int = 6
    num_codes: int = 128
    part_hidden_dim: int = 0
    max_motion_tokens: int = 49
    time_patch: int = 1
    coupling_mode: str = "holder_query"
    holder_depth: int = 2
    holder_mlp_ratio: float = 4.0

    hidden_size: int = 768
    num_heads: int = 12
    depth_double: int = 6
    depth_single: int = 12
    mlp_ratio: float = 4.0
    dropout: float = 0.1

    cond_drop_prob: float = 0.1
    self_cond_prob: float = 0.5
    use_self_condition: bool = True
    time_schedule: str = "logit_normal"
    denoiser_p_mean: float = -1.5
    denoiser_p_std: float = 0.8
    noise_scale: float = 1.0
    t_eps: float = 1e-4
    latent_norm_mode: str = "none"  # "none", "codebook", "empirical"
    latent_offset: float = 0.0
    latent_norm_eps: float = 1e-6
    sampling_schedule: str = "uniform"  # "uniform", "logit_normal"
    sampling_method: str = "ode"  # "ode", "sde"
    sde_gamma: float = 0.0
    # Fixed, not a knob.  The field survives only so stored options and the
    # checkpoint-selection config keep the key; anything other than DECODE_MODE
    # is rejected at construction.  See DECODE_MODE.
    decode_mode: str = DECODE_MODE

    terminal_mode: str = "tied_logits"  # "none", "nearest", "residual_nearest", "tied_logits", "learned_head"
    terminal_tau: float = 1.0
    terminal_tau_mode: str = "fixed"  # "fixed", "codebook_nn"
    terminal_tau_floor: float = 1e-6
    flow_loss_weight: float = 1.0


def lengths_to_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    return torch.arange(max_len, device=lengths.device)[None, :] < lengths[:, None]


def sample_timesteps(
    batch_size: int,
    device: torch.device,
    schedule: str,
    p_mean: float,
    p_std: float,
) -> torch.Tensor:
    if schedule == "uniform":
        return torch.rand(batch_size, device=device).clamp(1e-4, 1.0 - 1e-4)
    if schedule == "logit_normal":
        return torch.sigmoid(torch.randn(batch_size, device=device) * p_std + p_mean).clamp(1e-4, 1.0 - 1e-4)
    raise ValueError(f"Unknown time schedule: {schedule}")


def _split_rope_dims(hidden_size: int, num_heads: int) -> List[int]:
    head_dim = hidden_size // num_heads
    time_dim = int(round(head_dim * 0.75))
    time_dim = max(2, time_dim - (time_dim % 2))
    part_dim = head_dim - time_dim
    if part_dim < 2:
        part_dim = 2
        time_dim = head_dim - part_dim
    if time_dim % 2:
        time_dim -= 1
        part_dim += 1
    if part_dim % 2:
        part_dim -= 1
        time_dim += 1
    return [time_dim, part_dim]


class MotionCodeFlow(nn.Module):
    """Full DiT flow prior over frozen part-specific VQ code embeddings."""

    def __init__(self, config: MotionCodeFlowConfig) -> None:
        super().__init__()
        if config.representation != "t_concat":
            raise ValueError("holder-query CodeFlow only supports representation='t_concat'")
        if config.coupling_mode != "holder_query":
            raise ValueError("CodeFlow now uses coupling_mode='holder_query' only")
        if config.time_patch != 1:
            raise ValueError("holder-query CodeFlow uses time_patch=1")
        if config.terminal_mode not in {"none", "nearest", "residual_nearest", "tied_logits", "learned_head"}:
            raise ValueError(f"Unsupported terminal_mode: {config.terminal_mode}")
        if config.terminal_tau_mode not in {"fixed", "codebook_nn"}:
            raise ValueError(f"Unsupported terminal_tau_mode: {config.terminal_tau_mode}")
        if config.latent_norm_mode not in {"none", "codebook", "empirical"}:
            raise ValueError(f"Unsupported latent_norm_mode: {config.latent_norm_mode}")
        if config.sampling_schedule not in {"uniform", "logit_normal"}:
            raise ValueError(f"Unsupported sampling_schedule: {config.sampling_schedule}")
        if config.sampling_method not in {"ode", "sde"}:
            raise ValueError(f"Unsupported sampling_method: {config.sampling_method}")
        if config.decode_mode != DECODE_MODE:
            raise ValueError(
                f"decode_mode is fixed to {DECODE_MODE!r} and cannot be configured; "
                f"got {config.decode_mode!r}"
            )
        if config.holder_depth <= 0:
            raise ValueError(f"holder_depth must be positive, got {config.holder_depth}")
        if config.text_refiner_depth < 0:
            raise ValueError(f"text_refiner_depth must be non-negative, got {config.text_refiner_depth}")
        if config.text_refiner_pool not in TEXT_REFINER_POOL_MODES:
            raise ValueError(
                f"text_refiner_pool must be one of {sorted(TEXT_REFINER_POOL_MODES)}, "
                f"got {config.text_refiner_pool!r}"
            )
        if config.text_pooled_mode not in TEXT_POOLED_MODES:
            raise ValueError(
                f"text_pooled_mode must be one of {sorted(TEXT_POOLED_MODES)}, "
                f"got {config.text_pooled_mode!r}"
            )
        self.config = config

        self.tokenizer = build_codeflow_tokenizer(
            backend=config.vq_backend,
            kv_root=config.kv_root,
            checkpoint_path=config.vq_checkpoint,
            partition_path=config.vq_partition,
            opt_path=config.vq_opt_path,
            num_codes=config.num_codes,
            code_dim=config.code_dim,
            kv_part_target_mode=config.kv_part_target_mode,
            rvq_target_mode=config.rvq_target_mode,
        )
        if self.tokenizer.num_parts != config.num_parts:
            raise ValueError(f"Config num_parts={config.num_parts}, tokenizer has {self.tokenizer.num_parts}")
        if self.tokenizer.num_codes != config.num_codes:
            raise ValueError(f"Config num_codes={config.num_codes}, tokenizer has {self.tokenizer.num_codes}")
        if self.tokenizer.code_dim != config.code_dim:
            raise ValueError(f"Config code_dim={config.code_dim}, tokenizer has {self.tokenizer.code_dim}")
        self._init_latent_stats()
        self._init_terminal_tau()

        self.text_encoder = build_text_encoder(
            config.text_encoder_type,
            clip_version=config.clip_version,
            clip_path=config.clip_path,
            kv_root=config.kv_root,
            text_cache_path=config.text_cache_path,
            qwen_max_length=config.qwen_max_length,
            text_cache_fingerprint=config.text_cache_fingerprint,
            clip_hf_path=config.clip_hf_path,
        )

        self.input_code_dim = config.code_dim * (2 if config.use_self_condition else 1)
        conv_in = self.input_code_dim
        output_size = config.time_patch * config.code_dim

        self.input_conv = nn.Sequential(
            nn.Conv1d(conv_in, config.hidden_size, kernel_size=config.time_patch, stride=config.time_patch),
            nn.GELU(),
            nn.Conv1d(config.hidden_size, config.hidden_size, kernel_size=1),
        )
        self.part_embed = nn.Parameter(torch.zeros(config.num_parts, config.hidden_size))
        nn.init.normal_(self.part_embed, std=0.02)

        self.timestep_embed = TimestepEmbedder(config.hidden_size)
        self.text_token_proj = nn.Linear(self.text_encoder.width, config.hidden_size)
        # Token-only providers expose output_dim=None: no pooled/global text
        # branch is built and AdaLN conditioning reduces to the timestep alone.
        text_pooled_dim = getattr(self.text_encoder, "output_dim", None)
        self.text_pooled_proj = (
            nn.Sequential(
                nn.Linear(text_pooled_dim, config.hidden_size),
                nn.SiLU(),
                nn.Linear(config.hidden_size, config.hidden_size),
            )
            if text_pooled_dim
            else None
        )
        self.text_token_refiner = (
            BidirectionalTextRefiner(
                hidden_size=config.hidden_size,
                num_heads=config.num_heads,
                depth=config.text_refiner_depth,
                mlp_ratio=config.text_refiner_mlp_ratio,
                dropout=config.dropout,
            )
            if config.text_refiner_depth > 0
            else None
        )
        self._assert_refiner_pool_supported()
        # The two routings build byte-identical parameter sets -- same names,
        # same shapes -- so a strict state-dict load cannot tell them apart and
        # a checkpoint whose stored options were trimmed or rebuilt elsewhere
        # would load a token-trained model in modulation mode with every weight
        # matching and no error.  This buffer travels in the state dict, so the
        # mismatch surfaces as a value check instead of as a worse FID.
        self.register_buffer(
            "text_pooled_mode_code",
            torch.tensor(TEXT_POOLED_MODES.index(config.text_pooled_mode), dtype=torch.long),
            persistent=True,
        )
        # A marker for the prepended sentence token.  Both backbones give every
        # text position the same RoPE id, so the text stream is an unordered bag
        # and position cannot say which token is the sentence vector -- the only
        # other signal is that it came out of a different projection.  Three of
        # the four encoders in this campaign derive their pooled vector from the
        # token stream itself (CLIP-L's is literally one of the tokens), so
        # without a marker the token arm risks degenerating into a no-op.
        # Zero-initialised, so training starts exactly where it would without it.
        # Built only in token mode, which keeps modulation checkpoints loadable.
        self.pooled_type_embed = (
            nn.Parameter(torch.zeros(config.hidden_size))
            if config.text_pooled_mode == "token" and self.text_pooled_proj is not None
            else None
        )
        if config.text_pooled_mode == "token" and self.text_pooled_proj is None:
            raise ValueError(
                "--text_pooled_mode token needs a text provider that returns a pooled "
                f"sentence vector, but {type(self.text_encoder).__name__} "
                f"(text_encoder_type={config.text_encoder_type!r}) reports output_dim=None. "
                "Use a pooled provider, or --text_pooled_mode modulation."
            )

        self.backbone = MotionTextDiT(
            hidden_size=config.hidden_size,
            num_heads=config.num_heads,
            depth_double=config.depth_double,
            depth_single=config.depth_single,
            output_size=output_size,
            num_parts=config.num_parts,
            holder_depth=config.holder_depth,
            holder_mlp_ratio=config.holder_mlp_ratio,
            mlp_ratio=config.mlp_ratio,
            dropout=config.dropout,
            rope_axes_dims=_split_rope_dims(config.hidden_size, config.num_heads),
        )

        if config.terminal_mode == "learned_head":
            self.learned_heads = nn.ModuleList([
                nn.Linear(config.code_dim, config.num_codes)
                for _ in range(config.num_parts)
            ])
        else:
            self.learned_heads = None

    def _init_latent_stats(self) -> None:
        cfg = self.config
        if cfg.latent_norm_mode == "codebook":
            if not hasattr(self.tokenizer, "codebooks"):
                raise ValueError("latent_norm_mode='codebook' requires a tokenizer with codebooks")
            codebooks = self.tokenizer.codebooks.detach().float()
            if codebooks.shape[0] != cfg.num_parts:
                raise ValueError(
                    "latent_norm_mode='codebook' requires one codebook per model part; "
                    f"got codebook parts={codebooks.shape[0]} and num_parts={cfg.num_parts}. "
                    "Use latent_norm_mode='empirical' or 'none' for summed RVQ targets."
                )
            mean = codebooks.mean(dim=1)
            std = codebooks.std(dim=1, unbiased=False).clamp_min(float(cfg.latent_norm_eps))
        else:
            mean = torch.zeros(cfg.num_parts, cfg.code_dim, dtype=torch.float32)
            std = torch.ones(cfg.num_parts, cfg.code_dim, dtype=torch.float32)
        self.register_buffer("latent_mean", mean.view(1, 1, cfg.num_parts, cfg.code_dim), persistent=True)
        self.register_buffer("latent_std", std.view(1, 1, cfg.num_parts, cfg.code_dim), persistent=True)

    @torch.no_grad()
    def set_latent_stats(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        cfg = self.config
        mean = mean.detach().float().view(1, 1, cfg.num_parts, cfg.code_dim).to(self.latent_mean.device)
        std = (
            std.detach()
            .float()
            .view(1, 1, cfg.num_parts, cfg.code_dim)
            .clamp_min(float(cfg.latent_norm_eps))
            .to(self.latent_std.device)
        )
        self.latent_mean.copy_(mean)
        self.latent_std.copy_(std)

    def _init_terminal_tau(self) -> None:
        cfg = self.config
        if cfg.terminal_tau_mode == "codebook_nn":
            if not hasattr(self.tokenizer, "codebooks"):
                raise ValueError("terminal_tau_mode='codebook_nn' requires a tokenizer with codebooks")
            if self.tokenizer.codebooks.shape[0] != cfg.num_parts:
                raise ValueError(
                    "terminal_tau_mode='codebook_nn' requires one codebook per model part; "
                    f"got codebook parts={self.tokenizer.codebooks.shape[0]} and num_parts={cfg.num_parts}."
                )
            values = []
            for part_idx in range(cfg.num_parts):
                codebook = self.tokenizer.codebooks[part_idx].float()
                dist_sq = torch.cdist(codebook, codebook, p=2.0).square()
                dist_sq.fill_diagonal_(float("inf"))
                nearest = dist_sq.min(dim=1).values
                values.append(torch.median(nearest[torch.isfinite(nearest)]).clamp_min(float(cfg.terminal_tau_floor)))
            tau_parts = torch.stack(values).float()
        else:
            tau_parts = torch.full(
                (cfg.num_parts,),
                max(float(cfg.terminal_tau), float(cfg.terminal_tau_floor)),
                dtype=torch.float32,
            )
        self.register_buffer("terminal_tau_parts", tau_parts, persistent=False)

    @property
    def core_output_weight(self) -> torch.Tensor:
        return self.holder_output.linear.weight

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def holder_output(self):
        return self.backbone.holder_output.linear

    def train(self, mode: bool = True):
        super().train(mode)
        if hasattr(self.tokenizer, "vq_model"):
            self.tokenizer.vq_model.eval()
        self.text_encoder.eval()
        return self

    def trainable_named_parameters(self):
        for name, param in self.named_parameters():
            if name.startswith("tokenizer.") or name.startswith("text_encoder."):
                continue
            if param.requires_grad:
                yield name, param

    def trainable_parameters(self):
        for _name, param in self.trainable_named_parameters():
            yield param

    def state_dict_for_save(self) -> Dict[str, torch.Tensor]:
        state = self.state_dict()
        return {
            key: value
            for key, value in state.items()
            if not key.startswith("tokenizer.vq_model.")
            and not key.startswith("text_encoder.")
        }

    def load_trainable_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> Tuple[List[str], List[str]]:
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        allowed_missing = [
            key for key in missing
            if key.startswith("tokenizer.vq_model.")
            or key.startswith("text_encoder.")
            or key in {"latent_mean", "latent_std"}
            # Checkpoints predating the routing buffer simply do not carry it;
            # their mode is recovered from the stored options instead, which
            # assert_pooled_mode_matches_checkpoint treats as "cannot check".
            or key == "text_pooled_mode_code"
        ]
        bad_missing = [key for key in missing if key not in allowed_missing]
        return bad_missing, list(unexpected)

    def assert_pooled_mode_matches_checkpoint(self, state: Dict[str, torch.Tensor]) -> None:
        """Refuse a checkpoint whose pooled routing differs from this config.

        Called by the eval and generation loaders, which rebuild options from
        the checkpoint and would otherwise fall back to the argparse default
        when the stored options are incomplete.
        """
        saved = state.get("text_pooled_mode_code")
        if saved is None:
            return  # predates the buffer; its options carry the mode instead
        saved_mode = TEXT_POOLED_MODES[int(saved.item())]
        if saved_mode != self.config.text_pooled_mode:
            raise RuntimeError(
                "Checkpoint was trained with text_pooled_mode="
                f"{saved_mode!r} but this model is configured for "
                f"{self.config.text_pooled_mode!r}; the parameter shapes are "
                "identical in both modes, so this would otherwise load cleanly "
                "and only show up as degraded metrics."
            )


    def raw_to_model_latent(self, z: torch.Tensor) -> torch.Tensor:
        """Map frozen RVQ codebook embeddings into the DiT training space."""
        mean = self.latent_mean.to(device=z.device, dtype=z.dtype)
        std = self.latent_std.to(device=z.device, dtype=z.dtype)
        return (z - mean) / std + float(self.config.latent_offset)

    def model_to_raw_latent(self, z: torch.Tensor) -> torch.Tensor:
        """Map DiT-space latents back to the frozen KV decoder/codebook space."""
        mean = self.latent_mean.to(device=z.device, dtype=z.dtype)
        std = self.latent_std.to(device=z.device, dtype=z.dtype)
        return (z - float(self.config.latent_offset)) * std + mean

    def _sampling_grid(self, steps: int, device: torch.device) -> torch.Tensor:
        if steps <= 0:
            raise ValueError(f"Sampling steps must be positive, got {steps}")
        cfg = self.config
        if cfg.sampling_schedule == "uniform":
            return torch.linspace(0.0, 1.0, steps + 1, device=device)
        if cfg.sampling_schedule == "logit_normal":
            if steps == 1:
                return torch.tensor([0.0, 1.0], device=device)
            inner = sample_timesteps(
                steps - 1,
                device,
                "logit_normal",
                cfg.denoiser_p_mean,
                cfg.denoiser_p_std,
            ).sort().values
            return torch.cat([inner.new_zeros(1), inner, inner.new_ones(1)], dim=0)
        raise ValueError(f"Unknown sampling schedule: {cfg.sampling_schedule}")

    def _assert_refiner_pool_supported(self) -> None:
        """Fail at construction, not at the first training step.

        Without this the mismatch surfaces from ``_refiner_content_mask`` inside
        the forward pass -- minutes into an allocated GPU job, after dataset and
        VQ setup, which on a queued cluster is an expensive way to learn that a
        flag and a cache disagree.
        """
        if self.text_token_refiner is None or self.config.text_refiner_pool != "content_span":
            return
        if not getattr(self.text_encoder, "provides_content_mask", False):
            raise ValueError(
                f"--text_refiner_pool content_span requires a text provider that reports the "
                f"caption span, but {type(self.text_encoder).__name__} "
                f"(text_encoder_type={self.config.text_encoder_type!r}) does not. "
                "For an LLM2Vec cache, add spans with tools/add_content_spans_to_cache.py; "
                "otherwise pass --text_refiner_pool all_tokens."
            )

    def _refiner_content_mask(self, cond: TextCondition) -> Optional[torch.Tensor]:
        """Positions the refiner's pooled summary averages over, or None.

        Refuses rather than falling back: a provider with no span information
        would silently average chat-wrapper tokens into the summary, which is
        precisely the defect this mode removes, and nothing downstream would
        look wrong.
        """
        if self.config.text_refiner_pool == "all_tokens":
            return None
        if cond.content_mask is None:
            raise ValueError(
                "text_refiner_pool='content_span' needs the text provider to report which "
                f"positions hold the caption, but {type(self.text_encoder).__name__} returned "
                "content_mask=None. Add spans to the cache with "
                "tools/add_content_spans_to_cache.py, or pass --text_refiner_pool all_tokens "
                "to reproduce a run trained before the fix."
            )
        return cond.content_mask

    def _text_condition(
        self,
        texts: Iterable[str],
        drop_prob: float = 0.0,
        force_drop: bool = False,
        timesteps: Optional[torch.Tensor] = None,
        raw_condition: Optional[TextCondition] = None,
    ) -> TextCondition:
        if raw_condition is not None:
            if drop_prob > 0.0 or force_drop:
                raise ValueError("A supplied raw text condition cannot also request text dropout")
            cond = raw_condition
        else:
            cond = self.text_encoder(texts, drop_prob=drop_prob, force_drop=force_drop)
        tokens = self.text_token_proj(cond.tokens)
        if self.text_token_refiner is not None:
            if timesteps is None:
                raise ValueError("timesteps are required when text_refiner_depth > 0")
            tokens = self.text_token_refiner(
                tokens, timesteps, cond.padding_mask,
                self._refiner_content_mask(cond),
            )
        padding_mask = cond.padding_mask
        content_mask = cond.content_mask
        pooled = None
        if self.text_pooled_proj is not None:
            if cond.pooled is None:
                raise ValueError("Text encoder returned no pooled feature but text_pooled_proj exists")
            projected = self.text_pooled_proj(cond.pooled)
            if self.config.text_pooled_mode == "token":
                # The sentence vector becomes an ordinary text token instead of an
                # AdaLN signal, so motion positions attend to it selectively rather
                # than receiving one global affine per block.  AdaLN is left with
                # the timestep alone.
                #
                # Prepended, and prepending is exact rather than a convention we
                # are stuck with: both backbones give every text position the same
                # RoPE id (text_pos = zeros in DoubleStreamBlock.forward and in
                # FrameMotionTextDiT.forward), so where in the text stream it sits
                # cannot change the attention.  Front is simply the only placement
                # that needs no ragged per-row insert.
                #
                # Concatenated *after* the refiner on purpose: the refiner derives
                # its own AdaLN signal from a mean over the tokens it is given, so
                # feeding it this vector would let the sentence vector help produce
                # the modulation that then gates it.
                marked = projected + self.pooled_type_embed
                tokens = torch.cat([marked.unsqueeze(1), tokens], dim=1)
                lead = torch.zeros(
                    padding_mask.shape[0], 1, dtype=torch.bool, device=padding_mask.device
                )
                padding_mask = torch.cat([lead, padding_mask], dim=1)
                if content_mask is not None:
                    # False, not True: content_mask marks *caption* positions, and
                    # its only consumer averages over them to build the refiner's
                    # modulation.  Marking the sentence vector as caption text
                    # would let it help produce the signal that gates it -- the
                    # circularity concat-after-refiner exists to avoid -- the
                    # moment any consumer appears downstream of this concat.
                    content_mask = torch.cat([lead, content_mask], dim=1)
            else:
                pooled = projected
        return TextCondition(
            pooled=pooled,
            tokens=tokens,
            padding_mask=padding_mask,
            content_mask=content_mask,
        )

    def _pad_time(self, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        patch = self.config.time_patch
        pad_len = (patch - x.shape[1] % patch) % patch
        if pad_len:
            pad_shape = list(x.shape)
            pad_shape[1] = pad_len
            x = torch.cat([x, x.new_zeros(pad_shape)], dim=1)
        return x, pad_len

    def _patch_mask(self, token_lengths: torch.Tensor, latent_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        valid = lengths_to_mask(token_lengths, latent_len)
        patch = self.config.time_patch
        pad_len = (patch - latent_len % patch) % patch
        if pad_len:
            valid = torch.cat([valid, torch.zeros(valid.shape[0], pad_len, device=valid.device, dtype=torch.bool)], dim=1)
        patched = valid.view(valid.shape[0], -1, patch).any(dim=-1)
        return valid[:, :latent_len], patched

    def _pack_motion(
        self,
        x: torch.Tensor,
        token_lengths: torch.Tensor,
        x_self_cond: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        cfg = self.config
        if cfg.use_self_condition:
            if x_self_cond is None:
                x_self_cond = torch.zeros_like(x)
            x = torch.cat([x, x_self_cond], dim=-1)

        latent_len = x.shape[1]
        _, patch_valid = self._patch_mask(token_lengths, latent_len)
        x, pad_len = self._pad_time(x)
        bsz, padded_len, num_parts, dim = x.shape
        x = x.permute(0, 2, 3, 1).reshape(bsz * num_parts, dim, padded_len)
        h = self.input_conv(x)
        patch_len = h.shape[-1]
        h = h.view(bsz, num_parts, cfg.hidden_size, patch_len).permute(0, 3, 1, 2)
        h = h + self.part_embed[None, None]
        tokens = h.reshape(bsz, patch_len * num_parts, cfg.hidden_size)
        motion_valid = patch_valid[:, :, None].expand(bsz, patch_len, num_parts).reshape(bsz, patch_len * num_parts)
        time_ids = torch.arange(patch_len, device=x.device, dtype=torch.float32)
        part_ids = torch.arange(num_parts, device=x.device, dtype=torch.float32)
        tt, pp = torch.meshgrid(time_ids, part_ids, indexing="ij")
        pos = torch.stack([tt, pp], dim=-1).reshape(1, patch_len * num_parts, 2).expand(bsz, -1, -1)
        return tokens, motion_valid, pos, pad_len

    def _unpack_motion(self, y: torch.Tensor, latent_len: int) -> torch.Tensor:
        cfg = self.config
        bsz = y.shape[0]
        patch = cfg.time_patch
        patch_len = y.shape[1] // cfg.num_parts
        y = y.view(bsz, patch_len, cfg.num_parts, patch, cfg.code_dim)
        y = y.permute(0, 1, 3, 2, 4).reshape(bsz, patch_len * patch, cfg.num_parts, cfg.code_dim)
        return y[:, :latent_len]

    def forward(
        self,
        z: torch.Tensor,
        timesteps: torch.Tensor,
        texts: Iterable[str],
        token_lengths: torch.Tensor,
        x_self_cond: Optional[torch.Tensor] = None,
        text_drop_prob: float = 0.0,
        force_text_drop: bool = False,
        raw_text_condition: Optional[TextCondition] = None,
    ) -> torch.Tensor:
        cfg = self.config
        if timesteps.ndim == 0:
            timesteps = timesteps.expand(z.shape[0])
        timesteps = timesteps.to(device=z.device, dtype=z.dtype)
        token_lengths = token_lengths.to(z.device).long().clamp(min=1, max=z.shape[1])

        timestep_cond = self.timestep_embed(timesteps.float())
        text_cond = self._text_condition(
            texts,
            drop_prob=text_drop_prob,
            force_drop=force_text_drop,
            timesteps=timesteps,
            raw_condition=raw_text_condition,
        )
        motion_tokens, motion_valid, motion_pos, _ = self._pack_motion(z, token_lengths, x_self_cond)
        cond = timestep_cond
        if text_cond.pooled is not None:
            cond = cond + text_cond.pooled
        pred = self.backbone(
            motion=motion_tokens,
            text=text_cond.tokens,
            cond=cond,
            motion_valid=motion_valid,
            text_padding_mask=text_cond.padding_mask,
            motion_pos_ids=motion_pos,
            return_hidden=False,
        )
        pred = pred[:, : z.shape[1]]
        valid = lengths_to_mask(token_lengths, z.shape[1]).to(pred.dtype)
        return pred * valid[:, :, None, None]

    def velocity_from_clean(
        self,
        z_t: torch.Tensor,
        timesteps: torch.Tensor,
        clean: torch.Tensor,
    ) -> torch.Tensor:
        while timesteps.ndim < z_t.ndim:
            timesteps = timesteps[..., None]
        return (clean - z_t) / (1.0 - timesteps).clamp_min(self.config.t_eps)

    def terminal_logits(self, clean_pred: torch.Tensor, mode: Optional[str] = None) -> torch.Tensor:
        mode = mode or self.config.terminal_mode
        if mode == "none":
            raise ValueError("terminal_logits is unavailable when terminal_mode='none'")
        if mode in {"nearest", "tied_logits"}:
            tau = self.terminal_tau_parts if hasattr(self, "terminal_tau_parts") else self.config.terminal_tau
            return self.tokenizer.codebook_tied_logits(clean_pred, tau=tau)
        if mode == "learned_head":
            if self.learned_heads is None:
                raise RuntimeError("learned_head terminal mode was not initialized")
            logits = []
            for part_idx, head in enumerate(self.learned_heads):
                logits.append(head(clean_pred[:, :, part_idx]))
            return torch.stack(logits, dim=2)
        raise ValueError(f"Unknown terminal mode: {mode}")

    @torch.no_grad()
    def terminal_ids(self, clean_pred: torch.Tensor, mode: Optional[str] = None) -> torch.Tensor:
        mode = mode or self.config.terminal_mode
        if mode == "none":
            return None
        if mode == "residual_nearest":
            if not hasattr(self.tokenizer, "residual_nearest_ids"):
                raise ValueError("terminal_mode='residual_nearest' requires an RVQ tokenizer backend")
            return self.tokenizer.residual_nearest_ids(clean_pred)
        if mode == "nearest":
            return self.tokenizer.nearest_ids(clean_pred)
        return self.terminal_logits(clean_pred, mode=mode).argmax(dim=-1).long()

    def compute_losses(
        self,
        target_embeddings: torch.Tensor,
        target_ids: Optional[torch.Tensor],
        texts: Iterable[str],
        token_lengths: torch.Tensor,
        include_geometry_metrics: bool = False,
        geometry_severe_quantile: float = 0.75,
        noise: Optional[torch.Tensor] = None,
        timesteps: Optional[torch.Tensor] = None,
        x_self_cond: Optional[torch.Tensor] = None,
        allow_internal_self_condition: bool = True,
        source_embeddings: Optional[torch.Tensor] = None,
        op_ids: Optional[torch.Tensor] = None,
        task_ids: Optional[torch.Tensor] = None,
        preserve_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        del source_embeddings, op_ids, task_ids, preserve_mask
        cfg = self.config
        bsz, latent_len, num_parts, _ = target_embeddings.shape
        has_code_targets = target_ids is not None and bool(getattr(self.tokenizer, "supports_code_metrics", True))
        if getattr(self.tokenizer, "target_mode", "stage") == "sum":
            if target_ids is None:
                raise ValueError("MoMask RVQ sum target requires target_ids")
            expected_quantizers = int(getattr(self.tokenizer, "num_quantizers", target_ids.shape[-1]))
            if num_parts != 1 or target_ids.ndim != 3 or target_ids.shape[-1] != expected_quantizers:
                raise ValueError(
                    "MoMask RVQ sum target expects target_embeddings [B,T,1,D] and "
                    f"target_ids [B,T,{expected_quantizers}], got "
                    f"target_embeddings={tuple(target_embeddings.shape)} target_ids={tuple(target_ids.shape)}"
                )
        token_lengths = token_lengths.to(target_embeddings.device).long().clamp(min=1, max=latent_len)
        valid = lengths_to_mask(token_lengths, latent_len)
        valid_parts = valid[:, :, None].expand(bsz, latent_len, num_parts)
        valid_ids = valid[:, :, None].expand_as(target_ids).bool() if target_ids is not None else None
        valid_float = valid_parts.to(target_embeddings.dtype)
        target_model = self.raw_to_model_latent(target_embeddings)

        if noise is None:
            noise = torch.randn_like(target_model) * cfg.noise_scale
        else:
            noise = noise.to(device=target_model.device, dtype=target_model.dtype)
        if timesteps is None:
            t = sample_timesteps(
                bsz,
                target_embeddings.device,
                cfg.time_schedule,
                cfg.denoiser_p_mean,
                cfg.denoiser_p_std,
            ).to(target_embeddings.dtype)
        else:
            t = timesteps.to(device=target_embeddings.device, dtype=target_embeddings.dtype)
            if t.ndim == 0:
                t = t.expand(bsz)
        t_view = t[:, None, None, None]
        z_t = t_view * target_model + (1.0 - t_view) * noise
        z_t = z_t * valid_float[:, :, :, None]

        if x_self_cond is not None:
            x_self_cond = x_self_cond.to(device=target_embeddings.device, dtype=target_embeddings.dtype)
        if x_self_cond is None and cfg.use_self_condition and cfg.self_cond_prob > 0.0 and allow_internal_self_condition:
            with torch.no_grad():
                v_init = self.forward(
                    z_t,
                    t,
                    texts,
                    token_lengths,
                    x_self_cond=None,
                    text_drop_prob=0.0,
                )
                clean_init = v_init.detach()  # head output IS the clean endpoint
            keep = (torch.rand(bsz, device=target_embeddings.device) < cfg.self_cond_prob).to(target_embeddings.dtype)
            x_self_cond = clean_init * keep[:, None, None, None]

        x0_pred = self.forward(
            z_t,
            t,
            texts,
            token_lengths,
            x_self_cond=x_self_cond,
            text_drop_prob=cfg.cond_drop_prob,
        )
        x0_pred_f = x0_pred.float()
        target_model_f = target_model.float()
        valid_float_f = valid_float.float()
        z_t_f = z_t.float()
        t_f = t.float()

        v_pred_f = self.velocity_from_clean(z_t_f, t_f, x0_pred_f)
        v_target_f = (target_model - noise).float()
        per_part_flow = (v_pred_f - v_target_f).square().mean(dim=-1)
        flow_loss = (per_part_flow * valid_float_f).sum() / valid_float_f.sum().clamp_min(1.0)

        # PERMANENT (2026-08-09): the sole training loss is the flow-matching
        # velocity-space MSE computed from the x0 head: MSE((x0-z_t)/(1-t), Y1-Y0).
        # Terminal/codebook CE and auxiliary clean losses are deleted outright
        # and must never be reintroduced.  clean_pred feeds no-grad diagnostics.
        clean_pred = x0_pred_f
        clean_pred_raw = self.model_to_raw_latent(clean_pred)

        code_metrics: Dict[str, torch.Tensor] = {}
        if has_code_targets:
            with torch.no_grad():
                pred_ids = self.terminal_ids(clean_pred_raw)
                if pred_ids is None:
                    raise RuntimeError("Code metrics requested but terminal_ids returned None")
                acc = ((pred_ids == target_ids.long()) & valid_ids).sum().float() / valid_ids.sum().float().clamp_min(1.0)
                nn_ids = self.tokenizer.nearest_ids(clean_pred_raw)
                nn_acc = ((nn_ids == target_ids.long()) & valid_ids).sum().float() / valid_ids.sum().float().clamp_min(1.0)
            code_metrics.update({
                "token_acc": acc,
                "nearest_acc": nn_acc,
            })

        total = cfg.flow_loss_weight * flow_loss
        out = {
            "loss": total,
            "flow_loss": flow_loss,
        }
        out.update(code_metrics)
        if include_geometry_metrics and has_code_targets:
            with torch.no_grad():
                pred_ids = self.terminal_ids(clean_pred_raw)
                if pred_ids is None:
                    raise RuntimeError("Geometry metrics requested but terminal_ids returned None")
                code_dist, rank_pct = self.tokenizer.code_id_distances(target_ids.long(), pred_ids.long())
                valid_bool = valid_ids.bool()
                wrong = (pred_ids != target_ids.long()) & valid_bool
                severe = wrong & (rank_pct >= float(geometry_severe_quantile))
                valid_count = valid_bool.sum().float().clamp_min(1.0)
                wrong_count = wrong.sum().float()
                wrong_denom = wrong_count.clamp_min(1.0)
                out.update({
                    "geom_code_dist": (code_dist * valid_bool.to(code_dist.dtype)).sum() / valid_count,
                    "geom_rank_pct": (rank_pct * valid_bool.to(rank_pct.dtype)).sum() / valid_count,
                    "geom_wrong_code_dist": (code_dist * wrong.to(code_dist.dtype)).sum() / wrong_denom,
                    "geom_wrong_rank_pct": (rank_pct * wrong.to(rank_pct.dtype)).sum() / wrong_denom,
                    "geom_wrong_rate": wrong_count / valid_count,
                    "geom_severe_rate": severe.sum().float() / valid_count,
                    "geom_wrong_severe_frac": severe.sum().float() / wrong_denom,
                })
        return out

    @torch.no_grad()
    def sample_embeddings(
        self,
        texts: Iterable[str],
        token_lengths: torch.Tensor,
        steps: int = 32,
        cond_scale: float = 3.0,
        use_self_condition: bool = True,
    ) -> torch.Tensor:
        cfg = self.config
        text_list = list(texts)
        bsz = len(text_list)
        token_lengths = token_lengths.to(self.device).long()
        latent_len = int(token_lengths.max().item())
        z = torch.randn(
            bsz,
            latent_len,
            cfg.num_parts,
            cfg.code_dim,
            device=self.device,
        ) * cfg.noise_scale
        valid = lengths_to_mask(token_lengths, latent_len).to(z.dtype)
        z = z * valid[:, :, None, None]
        x_self_cond = None
        grid = self._sampling_grid(int(steps), self.device).to(z.dtype)
        if cond_scale == 1.0:
            raw_text_condition = self.text_encoder(
                text_list,
                drop_prob=0.0,
                force_drop=False,
            )
        else:
            # Encode/cache-read both branches once.  The timestep-aware token
            # refiner remains inside forward, but frozen features are reused at
            # every ODE/SDE step.
            raw_text_condition = self.text_encoder(
                [""] * bsz + text_list,
                drop_prob=0.0,
                force_drop=False,
            )

        def forward_guided(
            z_in: torch.Tensor,
            t_in: torch.Tensor,
            x_sc: Optional[torch.Tensor],
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            if cond_scale == 1.0:
                v_out = self.forward(
                    z_in,
                    t_in,
                    text_list,
                    token_lengths,
                    x_self_cond=x_sc if use_self_condition else None,
                    text_drop_prob=0.0,
                    raw_text_condition=raw_text_condition,
                )
            else:
                z_cat = torch.cat([z_in, z_in], dim=0)
                lengths_cat = torch.cat([token_lengths, token_lengths], dim=0)
                texts_cat = [""] * bsz + text_list
                sc_cat = torch.cat([x_sc, x_sc], dim=0) if x_sc is not None and use_self_condition else None
                v_all = self.forward(
                    z_cat,
                    torch.cat([t_in, t_in], dim=0),
                    texts_cat,
                    lengths_cat,
                    x_self_cond=sc_cat,
                    text_drop_prob=0.0,
                    raw_text_condition=raw_text_condition,
                )
                x0_uncond, x0_cond = v_all.chunk(2, dim=0)
                v_out = x0_uncond + float(cond_scale) * (x0_cond - x0_uncond)
            clean_out = v_out  # head output is x0; CFG combined in x0 space
            v_out = self.velocity_from_clean(z_in, t_in, clean_out)
            return v_out, clean_out

        for idx in range(steps):
            t_cur_scalar = grid[idx]
            t_next_scalar = grid[idx + 1]
            dt = t_next_scalar - t_cur_scalar
            z_eval = z
            t_eval_scalar = t_cur_scalar
            if cfg.sampling_method == "sde" and float(cfg.sde_gamma) > 0.0:
                alpha_value = max(0.0, min(1.0, 1.0 - float(cfg.sde_gamma) * float(dt.item())))
                eps = torch.randn_like(z) * cfg.noise_scale
                z_eval = alpha_value * z + (1.0 - alpha_value) * eps
                t_eval_scalar = t_cur_scalar * alpha_value
                dt = t_next_scalar - t_eval_scalar

            t_eval = t_eval_scalar.expand(bsz)
            v, clean = forward_guided(z_eval, t_eval, x_self_cond)
            z = z_eval + dt * v
            z = z * valid[:, :, None, None]
            if cfg.use_self_condition and use_self_condition:
                x_self_cond = clean.detach() * valid[:, :, None, None]

        raw = self.model_to_raw_latent(z)
        return raw * valid[:, :, None, None]

    @torch.no_grad()
    def generate_ids(
        self,
        texts: Iterable[str],
        token_lengths: torch.Tensor,
        steps: int = 32,
        cond_scale: float = 3.0,
        terminal_mode: Optional[str] = None,
    ) -> torch.Tensor:
        clean = self.sample_embeddings(
            texts,
            token_lengths=token_lengths,
            steps=steps,
            cond_scale=cond_scale,
        )
        if (terminal_mode or self.config.terminal_mode) == "none":
            raise RuntimeError("generate_ids is unavailable when terminal_mode='none'")
        return self.terminal_ids(clean, mode=terminal_mode)

    @torch.no_grad()
    def generate_motion(
        self,
        texts: Iterable[str],
        token_lengths: torch.Tensor,
        steps: int = 32,
        cond_scale: float = 3.0,
        terminal_mode: Optional[str] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample, then decode the continuous output directly -- always.

        ``ids`` are still returned when a terminal head is configured, because
        callers report code accuracy from them; they are never the decode path.
        """
        clean = self.sample_embeddings(
            texts,
            token_lengths=token_lengths,
            steps=steps,
            cond_scale=cond_scale,
        )
        terminal = terminal_mode or self.config.terminal_mode
        ids = self.terminal_ids(clean, mode=terminal_mode) if terminal != "none" else None
        motion = self.tokenizer.decode_embeddings(clean)
        return motion, ids
