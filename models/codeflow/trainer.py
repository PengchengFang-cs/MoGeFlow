import json
import math
import os
import random
from argparse import Namespace
from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Type

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from data.t2m_dataset import (
    Text2MotionDataset,
    Text2MotionDatasetEval,
    collate_fn,
)
from models.codeflow import (
    CodeFlowEvalConfig,
    MotionCodeFlow,
    MotionCodeFlowConfig,
    evaluate_codeflow_t2m,
)
from models.codeflow.motion_code_flow import DECODE_MODE, lengths_to_mask, sample_timesteps
from models.t2m_eval_wrapper import EvaluatorModelWrapper
from options.codeflow_options import TrainCodeFlowOptions
from utils.fixseed import fixseed
from utils.get_opt import get_opt
from utils.lr_schedule import resolve_multistep_milestones
from utils.word_vectorizer import WordVectorizer


DEFAULT_BEST_CHECKPOINT_LIMIT = 1
# The split the periodic full evaluation runs on, and therefore the split that
# best-checkpoint selection is based on.  Fixed, with no CLI flag: every metric
# this project has recorded was measured here, so a run that silently used a
# different one would produce numbers that look fine and compare against
# nothing.  The resume guard in load_checkpoint still compares, which is what
# catches a checkpoint that recorded something else.
FULL_EVAL_SPLIT = "test"


def _eval_split(opt) -> str:
    return FULL_EVAL_SPLIT



def normalized_dataset_name(dataset_name: str) -> str:
    name = str(dataset_name).lower()
    if name in {"humanml", "humanml3d"}:
        return "t2m"
    return name


def expected_motion_dim(dataset_name: str) -> int:
    name = normalized_dataset_name(dataset_name)
    if name == "kit":
        return 251
    if name == "t2m":
        return 263
    raise ValueError(f"Unsupported dataset_name: {dataset_name}")


def dataset_opt_path(opt) -> Path:
    override = getattr(opt, "dataset_opt_path", "")
    if override:
        return Path(override).expanduser()
    name = normalized_dataset_name(getattr(opt, "dataset_name", "t2m"))
    return Path("./checkpoints") / name / "Comp_v6_KLD005" / "opt.txt"


class CodeFlowLossModule(nn.Module):
    def __init__(self, model: MotionCodeFlow) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        embeddings,
        ids,
        texts,
        token_lengths,
        noise=None,
        timesteps=None,
        x_self_cond=None,
        allow_internal_self_condition=True,
    ):
        return self.model.compute_losses(
            embeddings,
            ids,
            texts,
            token_lengths,
            noise=noise,
            timesteps=timesteps,
            x_self_cond=x_self_cond,
            allow_internal_self_condition=allow_internal_self_condition,
        )


def is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def setup_distributed(opt):
    local_rank_env = os.environ.get("LOCAL_RANK")
    if local_rank_env is None:
        if opt.gpu_id >= 0 and torch.cuda.is_available():
            torch.cuda.set_device(opt.gpu_id)
            device = torch.device("cuda", opt.gpu_id)
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return False, 0, 1, 0, device

    local_rank = int(local_rank_env)
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    timeout_minutes = max(1, int(getattr(opt, "ddp_timeout_minutes", 10)))
    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(minutes=timeout_minutes),
    )
    return True, rank, world_size, local_rank, torch.device("cuda", local_rank)


def cleanup_distributed():
    if is_dist():
        dist.barrier()
        dist.destroy_process_group()


def master_print(rank: int, msg: str) -> None:
    if rank == 0:
        print(msg, flush=True)


def build_train_dataset(opt):
    data_root = Path(opt.data_root).expanduser().resolve()
    mean = np.load(opt.mean_path).astype(np.float32)
    std = np.load(opt.std_path).astype(np.float32)
    ds_opt = Namespace(
        dataset_name=opt.dataset_name,
        max_motion_length=opt.motion_length,
        unit_length=opt.unit_length,
        motion_dir=str(data_root / "new_joint_vecs"),
        text_dir=str(data_root / "texts"),
    )
    split_file = data_root / "train.txt"
    return Text2MotionDataset(ds_opt, mean, std, str(split_file))


def collect_text_cache_preflight_captions(data_root: str) -> List[str]:
    text_dir = Path(data_root).expanduser().resolve() / "texts"
    if not text_dir.is_dir():
        raise FileNotFoundError(f"Dataset text directory not found for cache preflight: {text_dir}")
    captions = {""}
    for path in text_dir.glob("*.txt"):
        for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            caption = raw_line.strip().split("#", 1)[0]
            if caption:
                captions.add(caption)
    return sorted(captions)


def validate_stats_contract(opt) -> Dict[str, object]:
    mean_path = Path(opt.mean_path).expanduser().resolve()
    std_path = Path(opt.std_path).expanduser().resolve()
    if not mean_path.is_file():
        raise FileNotFoundError(f"mean_path not found: {mean_path}")
    if not std_path.is_file():
        raise FileNotFoundError(f"std_path not found: {std_path}")
    mean = np.load(mean_path)
    std = np.load(std_path)
    dataset_name = normalized_dataset_name(getattr(opt, "dataset_name", "t2m"))
    expected_dim = expected_motion_dim(dataset_name)
    expected_shape = (expected_dim,)
    if mean.shape != expected_shape or std.shape != expected_shape:
        raise RuntimeError(f"Expected {dataset_name} stats shape {expected_shape}, got mean={mean.shape} std={std.shape}")
    if not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise RuntimeError("Normalization stats contain non-finite values")
    if np.min(std) <= 0:
        raise RuntimeError(f"Normalization std must be positive, got min={float(np.min(std))}")

    kv_root = Path(opt.kv_root).expanduser().resolve()
    ref_mean_path = kv_root / "checkpoints" / "stats" / "mean.npy"
    ref_std_path = kv_root / "checkpoints" / "stats" / "std.npy"
    matched_released_vq_stats = False
    if dataset_name == "t2m" and ref_mean_path.is_file() and ref_std_path.is_file():
        ref_mean = np.load(ref_mean_path)
        ref_std = np.load(ref_std_path)
        matched_released_vq_stats = bool(
            ref_mean.shape == mean.shape
            and ref_std.shape == std.shape
            and np.allclose(mean, ref_mean, rtol=0.0, atol=1e-6)
            and np.allclose(std, ref_std, rtol=0.0, atol=1e-6)
        )
    if not opt.allow_non_vq_stats and not matched_released_vq_stats:
        raise RuntimeError(
            f"{dataset_name} mean/std do not match the released KV-Control VQ stats. "
            "Pass --allow_non_vq_stats only if this is an intentional tokenizer/stat swap."
        )
    return {
        "mean_path": str(mean_path),
        "std_path": str(std_path),
        "dataset_name": dataset_name,
        "mean_shape": list(mean.shape),
        "std_shape": list(std.shape),
        "std_min": float(np.min(std)),
        "std_max": float(np.max(std)),
        "matched_released_vq_stats": matched_released_vq_stats,
    }


def make_config(opt) -> MotionCodeFlowConfig:
    return MotionCodeFlowConfig(
        kv_root=opt.kv_root,
        vq_backend=opt.vq_backend,
        vq_checkpoint=opt.vq_checkpoint,
        vq_partition=opt.vq_partition,
        vq_opt_path=opt.vq_opt_path,
        kv_part_target_mode=getattr(opt, "kv_part_target_mode", "codebook"),
        rvq_target_mode=getattr(opt, "rvq_target_mode", "stage"),
        clip_version=opt.clip_version,
        clip_path=opt.clip_path,
        clip_hf_path=getattr(opt, "clip_hf_path", None),
        text_encoder_type=getattr(opt, "text_encoder_type", "clip"),
        text_cache_path=getattr(opt, "text_cache_path", "") or None,
        text_cache_fingerprint=str(getattr(opt, "text_cache_fingerprint", "")),
        qwen_max_length=int(getattr(opt, "qwen_max_length", 128)),
        text_refiner_depth=int(getattr(opt, "text_refiner_depth", 0)),
        text_refiner_mlp_ratio=float(getattr(opt, "text_refiner_mlp_ratio", 4.0)),
        # Default matches MotionCodeFlowConfig: an opt restored from a pre-fix
        # run has no such attribute and must keep pooling as it was trained.
        text_refiner_pool=str(getattr(opt, "text_refiner_pool", "all_tokens")),
        text_pooled_mode=str(getattr(opt, "text_pooled_mode", "modulation")),
        representation=opt.representation,
        code_dim=opt.code_dim,
        num_parts=opt.num_parts,
        num_codes=opt.num_codes,
        part_hidden_dim=opt.part_hidden_dim,
        coupling_mode=opt.coupling_mode,
        holder_depth=opt.holder_depth,
        holder_mlp_ratio=opt.holder_mlp_ratio,
        hidden_size=opt.hidden_size,
        num_heads=opt.num_heads,
        depth_double=opt.depth_double,
        depth_single=opt.depth_single,
        mlp_ratio=opt.mlp_ratio,
        dropout=opt.dropout,
        time_patch=opt.time_patch,
        cond_drop_prob=opt.cond_drop_prob,
        self_cond_prob=opt.self_cond_prob,
        use_self_condition=not opt.disable_self_condition,
        time_schedule=opt.time_schedule,
        t_eps=float(getattr(opt, "t_eps", 1e-4)),
        denoiser_p_mean=opt.denoiser_p_mean,
        denoiser_p_std=opt.denoiser_p_std,
        noise_scale=opt.noise_scale,
        latent_norm_mode=opt.latent_norm_mode,
        latent_offset=opt.latent_offset,
        latent_norm_eps=opt.latent_norm_eps,
        sampling_schedule=opt.sampling_schedule,
        sampling_method=opt.sampling_method,
        sde_gamma=opt.sde_gamma,
        decode_mode=DECODE_MODE,
        terminal_mode=opt.terminal_mode,
        terminal_tau=opt.terminal_tau,
        terminal_tau_mode=opt.terminal_tau_mode,
        terminal_tau_floor=opt.terminal_tau_floor,
        flow_loss_weight=opt.flow_loss_weight,
    )


def count_trainable(model: MotionCodeFlow) -> int:
    return sum(param.numel() for param in model.trainable_parameters())


def check_core_grad_health(model: MotionCodeFlow, loss: torch.Tensor, step: int, rank: int, min_abs: float) -> None:
    if not torch.isfinite(loss.detach()).all():
        raise RuntimeError(f"Non-finite loss before optimizer step at step={step} rank={rank}")
    weight = getattr(model, "core_output_weight", model.holder_output.linear.weight)
    grad = weight.grad
    if grad is None:
        raise RuntimeError(f"Missing core output grad at step={step} rank={rank}")
    grad_f = grad.detach().float()
    if not torch.isfinite(grad_f).all():
        raise RuntimeError(f"Non-finite core output grad at step={step} rank={rank}")
    grad_absmax = float(grad_f.abs().max().item())
    if grad_absmax <= float(min_abs):
        raise RuntimeError(
            f"core output grad is too small at step={step} rank={rank}: "
            f"absmax={grad_absmax:.6e} min_abs={float(min_abs):.6e}"
        )


def amp_dtype_from_options(opt) -> torch.dtype:
    if getattr(opt, "amp_dtype", "fp16") == "bf16":
        return torch.bfloat16
    return torch.float16


def amp_enabled(opt, device: torch.device) -> bool:
    return bool(opt.amp and device.type == "cuda")


@torch.no_grad()
def prepare_flow_training_state(
    model: MotionCodeFlow,
    target_embeddings: torch.Tensor,
    texts: List[str],
    token_lengths: torch.Tensor,
    opt,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    cfg = model.config
    bsz, latent_len, num_parts, _ = target_embeddings.shape
    token_lengths = token_lengths.to(target_embeddings.device).long().clamp(min=1, max=latent_len)
    valid = lengths_to_mask(token_lengths, latent_len)
    valid_float = valid[:, :, None].expand(bsz, latent_len, num_parts).to(target_embeddings.dtype)
    target_model = model.raw_to_model_latent(target_embeddings)
    noise = torch.randn_like(target_model) * cfg.noise_scale
    timesteps = sample_timesteps(
        bsz,
        target_embeddings.device,
        cfg.time_schedule,
        cfg.denoiser_p_mean,
        cfg.denoiser_p_std,
    ).to(target_embeddings.dtype)

    x_self_cond = None
    if cfg.use_self_condition and cfg.self_cond_prob > 0.0:
        t_view = timesteps[:, None, None, None]
        z_t = t_view * target_model + (1.0 - t_view) * noise
        z_t = z_t * valid_float[:, :, :, None]
        with torch.cuda.amp.autocast(enabled=amp_enabled(opt, device), dtype=amp_dtype_from_options(opt)):
            v_init = model.forward(
                z_t,
                timesteps,
                texts,
                token_lengths,
                x_self_cond=None,
                text_drop_prob=0.0,
            )
            clean_init = v_init.detach()  # head output IS the clean endpoint
        keep = (torch.rand(bsz, device=target_embeddings.device) < cfg.self_cond_prob).to(target_embeddings.dtype)
        x_self_cond = clean_init * keep[:, None, None, None]

    return noise, timesteps, x_self_cond


def build_lr_schedule(opt, iters_per_epoch: int) -> Dict[str, object]:
    if opt.max_epoch <= 0:
        raise ValueError(f"max_epoch must be > 0, got {opt.max_epoch}")
    if iters_per_epoch <= 0:
        raise ValueError(f"iters_per_epoch must be > 0, got {iters_per_epoch}")

    total_steps = int(opt.max_steps) if int(opt.max_steps) > 0 else int(opt.max_epoch) * int(iters_per_epoch)
    if total_steps <= 0:
        raise ValueError(f"total_steps must be > 0, got {total_steps}")

    scheduler_name = getattr(opt, "lr_scheduler", "multistep")
    summary: Dict[str, object] = {
        "lr_scheduler": scheduler_name,
        "base_lr": float(opt.lr),
        "iters_per_epoch": int(iters_per_epoch),
        "total_steps": int(total_steps),
        "max_epoch": int(opt.max_epoch),
        "max_steps": int(opt.max_steps),
        "warmup_steps": int(opt.warmup_steps),
        "warmup_epochs": float(opt.warmup_steps) / float(iters_per_epoch),
    }

    if scheduler_name == "multistep":
        resolved_milestones = resolve_multistep_milestones(
            opt.milestones,
            getattr(opt, "milestone_unit", "auto"),
            total_iters=total_steps,
            iters_per_epoch=iters_per_epoch,
            max_epoch=opt.max_epoch,
        )
        summary.update(
            {
                "gamma": float(opt.gamma),
                "milestones": [float(value) for value in opt.milestones],
                "milestone_unit": getattr(opt, "milestone_unit", "auto"),
                "resolved_milestones": resolved_milestones,
                "resolved_milestone_epochs": [float(step) / float(iters_per_epoch) for step in resolved_milestones],
                "reachable_milestones": [step for step in resolved_milestones if step <= total_steps],
            }
        )
        return summary

    if scheduler_name == "half_cosine":
        eta_min = float(opt.lr) * float(opt.eta_min_ratio)
        summary.update(
            {
                "eta_min_ratio": float(opt.eta_min_ratio),
                "eta_min": eta_min,
                "cosine_steps": max(1, total_steps - max(0, int(opt.warmup_steps))),
                "resolved_milestones": [],
                "resolved_milestone_epochs": [],
                "reachable_milestones": [],
            }
        )
        return summary

    raise ValueError(f"Unsupported lr_scheduler: {scheduler_name}")


def attach_lr_schedule_to_options(opt, schedule: Dict[str, object]) -> None:
    opt.iters_per_epoch = int(schedule["iters_per_epoch"])
    opt.total_train_steps = int(schedule["total_steps"])
    opt.resolved_milestones = list(schedule.get("resolved_milestones", []))
    opt.resolved_milestone_epochs = list(schedule.get("resolved_milestone_epochs", []))
    opt.reachable_milestones = list(schedule.get("reachable_milestones", []))
    opt.warmup_epochs = float(schedule["warmup_epochs"])


def lr_at_step(step: int, opt, schedule: Dict[str, object]) -> float:
    if opt.warmup_steps > 0 and step < opt.warmup_steps:
        return opt.lr * float(step + 1) / float(opt.warmup_steps)

    scheduler_name = schedule["lr_scheduler"]
    if scheduler_name == "multistep":
        lr = float(opt.lr)
        for milestone in schedule.get("resolved_milestones", []):
            if step >= int(milestone):
                lr *= float(opt.gamma)
        return lr

    if scheduler_name == "half_cosine":
        total_steps = int(schedule["total_steps"])
        denom = max(total_steps - int(opt.warmup_steps), 1)
        progress = min(max((step - int(opt.warmup_steps)) / denom, 0.0), 1.0)
        eta_min = float(schedule["eta_min"])
        return eta_min + (float(opt.lr) - eta_min) * 0.5 * (1.0 + math.cos(math.pi * progress))

    raise ValueError(f"Unsupported lr_scheduler: {scheduler_name}")


def update_optimizer_lr(optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        lr_mult = float(group.get("lr_mult", 1.0))
        group["lr"] = lr * lr_mult


def build_optimizer_param_groups(model: MotionCodeFlow, opt) -> Tuple[List[Dict[str, object]], Dict[str, int]]:
    named_params = list(model.trainable_named_parameters())
    if not named_params:
        raise RuntimeError("Model has no trainable parameters")
    params = [param for _name, param in named_params]
    return [{"params": params, "lr_mult": 1.0}], {"base": sum(p.numel() for p in params)}


def flatten_optimizer_params(param_groups: List[Dict[str, object]]) -> List[torch.nn.Parameter]:
    params: List[torch.nn.Parameter] = []
    for group in param_groups:
        params.extend(group["params"])
    return params


def load_initial_trainable_state(path: Path, model: MotionCodeFlow, use_ema: bool = False) -> Tuple[List[str], List[str], str]:
    ckpt = torch.load(str(path), map_location="cpu")
    if use_ema and ckpt.get("ema") is not None:
        state = ckpt["ema"]
        source = "ema"
    else:
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        source = "model"
    missing, unexpected = model.load_state_dict(state, strict=False)
    allowed_prefixes = (
        "tokenizer.vq_model.",
        "text_encoder.",
    )
    allowed_exact = {"latent_mean", "latent_std"}
    bad_missing = [key for key in missing if not key.startswith(allowed_prefixes) and key not in allowed_exact]
    return bad_missing, list(unexpected), source


def make_ema(model: MotionCodeFlow) -> Dict[str, torch.Tensor]:
    return {
        key: value.detach().clone()
        for key, value in model.state_dict_for_save().items()
        if torch.is_floating_point(value)
    }


def update_ema(ema: Dict[str, torch.Tensor], model: MotionCodeFlow, decay: float = 0.9999) -> None:
    state = model.state_dict_for_save()
    for key, value in state.items():
        if key in ema and torch.is_floating_point(value):
            ema[key].mul_(decay).add_(value.detach(), alpha=1.0 - decay)


def save_checkpoint(path: Path, model: MotionCodeFlow, optimizer, scaler, ema, epoch: int, step: int, opt) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict_for_save(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "ema": ema,
            "epoch": epoch,
            "step": step,
            "options": vars(opt),
        },
        str(path),
    )


def load_checkpoint(path: Path, model: MotionCodeFlow, optimizer, scaler, opt=None):
    ckpt = torch.load(str(path), map_location="cpu")
    if opt is not None:
        saved_options = ckpt.get("options", {}) if isinstance(ckpt, dict) else {}
        saved_fingerprint = str(saved_options.get("text_cache_fingerprint", ""))
        current_fingerprint = str(getattr(opt, "text_cache_fingerprint", ""))
        if saved_fingerprint or current_fingerprint:
            if saved_fingerprint != current_fingerprint:
                raise RuntimeError(
                    "Refusing to resume with a different text-encoder contract: "
                    f"checkpoint={saved_fingerprint or '<missing>'}, "
                    f"current={current_fingerprint or '<missing>'}"
                )
        # Training resume is intentionally stricter than eval/generation:
        # those paths only bind the semantic encoder contract and may use a
        # prompt-cache superset, whereas optimizer continuation should see the
        # exact caption cache recorded by the checkpoint.
        saved_content = str(saved_options.get("text_cache_content_fingerprint", ""))
        current_content = str(getattr(opt, "text_cache_content_fingerprint", ""))
        if saved_content and saved_content != current_content:
            raise RuntimeError(
                "Refusing to resume with different text-cache content: "
                f"checkpoint={saved_content}, current={current_content or '<missing>'}"
            )
        # Changing how the refiner pools swaps one operator for another midway
        # through optimisation: the weights keep loading and the loss keeps
        # falling, so nothing else would ever surface it.  Checkpoints predating
        # the flag were trained with all-token pooling.
        saved_pool = str(saved_options.get("text_refiner_pool", "all_tokens"))
        current_pool = str(getattr(opt, "text_refiner_pool", "all_tokens"))
        if int(getattr(opt, "text_refiner_depth", 0)) > 0 and saved_pool != current_pool:
            raise RuntimeError(
                "Refusing to resume with a different refiner pooling mode: "
                f"checkpoint={saved_pool}, current={current_pool}. "
                f"Pass --text_refiner_pool {saved_pool} to continue this run."
            )
        # Same class of silent failure: "modulation" and "token" route the same
        # projected vector to different places, so the weights still load and
        # the loss still falls while the model is no longer the one that was
        # trained.  Checkpoints predating the flag used modulation.
        # The evaluation split is not restored from the checkpoint -- it comes
        # from the CLI -- and most launch scripts predate the flag and pass
        # nothing, so a resume would silently adopt the current default.  That
        # is not merely a different number: selection_config stops matching, the
        # best-metric history is dropped, and the next evaluation unlinks the
        # retained best_fid.pt/best_top3.pt and replaces them with checkpoints
        # chosen on the other split.  Irreversible, with one log line.
        saved_split = str(saved_options.get("full_eval_split", "test"))
        current_split = _eval_split(opt)
        if saved_split != current_split:
            raise RuntimeError(
                "Refusing to resume with a different full-evaluation split: "
                f"checkpoint={saved_split}, current={current_split}. "
                "The split is fixed in the code; a checkpoint recording a different "
                "one was produced by another build and its retained best checkpoints "
                "were selected on that split, so its history cannot be continued here."
            )
        saved_pmode = str(saved_options.get("text_pooled_mode", "modulation"))
        current_pmode = str(getattr(opt, "text_pooled_mode", "modulation"))
        if saved_pmode != current_pmode:
            raise RuntimeError(
                "Refusing to resume with a different pooled-vector routing: "
                f"checkpoint={saved_pmode}, current={current_pmode}. "
                f"Pass --text_pooled_mode {saved_pmode} to continue this run."
            )
    missing, unexpected = model.load_trainable_state_dict(ckpt["model"])
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint load mismatch: missing={missing[:8]} unexpected={unexpected[:8]}")
    optimizer.load_state_dict(ckpt["optimizer"])
    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])
    return int(ckpt.get("epoch", 0)), int(ckpt.get("step", 0)), ckpt.get("ema")


def move_optimizer_state_to_device(optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device)


@torch.no_grad()
def run_vq_contract_check(model: MotionCodeFlow, loader, device, opt, rank: int) -> Dict[str, object]:
    if opt.disable_vq_contract_check:
        return {"disabled": True}
    try:
        captions, motions, lengths = next(iter(loader))
    except StopIteration as exc:
        raise RuntimeError("Cannot run VQ contract check on an empty training loader") from exc
    del captions
    sample_count = min(int(opt.vq_contract_samples), motions.shape[0])
    if sample_count <= 0:
        return {"disabled": True, "reason": "vq_contract_samples<=0"}
    motions = motions[:sample_count].float().to(device, non_blocking=True)
    lengths = lengths[:sample_count].to(device, non_blocking=True)
    summary = model.tokenizer.verify_contract(
        motion=motions,
        lengths=lengths,
        max_samples=sample_count,
    )
    master_print(rank, "VQ_CONTRACT " + json.dumps(summary, sort_keys=True))
    return summary


@torch.no_grad()
def encode_batch(model: MotionCodeFlow, motions: torch.Tensor, lengths: torch.Tensor, unit_length: int):
    ids, embeddings = model.tokenizer.encode(motions)
    token_lengths = (lengths.to(motions.device).long() // unit_length).clamp(min=1, max=embeddings.shape[1])
    return ids, embeddings, token_lengths


@torch.no_grad()
def compute_and_set_empirical_latent_stats(
    model: MotionCodeFlow,
    loader,
    device: torch.device,
    unit_length: int,
    out_dir: Path,
    rank: int,
) -> Dict[str, object]:
    cfg = model.config
    was_training = model.training
    model.eval()
    sums = torch.zeros(cfg.num_parts, cfg.code_dim, device=device, dtype=torch.float64)
    sumsq = torch.zeros_like(sums)
    counts = torch.zeros(cfg.num_parts, device=device, dtype=torch.float64)

    for _captions, motions, lengths in loader:
        motions = motions.float().to(device, non_blocking=True)
        lengths = lengths.to(device, non_blocking=True)
        _ids, embeddings, token_lengths = encode_batch(model, motions, lengths, unit_length)
        bsz, latent_len, num_parts, _dim = embeddings.shape
        if num_parts != cfg.num_parts:
            raise RuntimeError(f"Tokenizer emitted {num_parts} parts, model expects {cfg.num_parts}")
        token_lengths = token_lengths.to(device).long().clamp(min=1, max=latent_len)
        valid = lengths_to_mask(token_lengths, latent_len)
        valid_parts = valid[:, :, None].expand(bsz, latent_len, num_parts)
        weights = valid_parts[:, :, :, None].to(torch.float64)
        values = embeddings.to(device=device, dtype=torch.float64)
        sums += (values * weights).sum(dim=(0, 1))
        sumsq += (values.square() * weights).sum(dim=(0, 1))
        counts += valid_parts.sum(dim=(0, 1)).to(torch.float64)

    if is_dist():
        dist.all_reduce(sums, op=dist.ReduceOp.SUM)
        dist.all_reduce(sumsq, op=dist.ReduceOp.SUM)
        dist.all_reduce(counts, op=dist.ReduceOp.SUM)

    if torch.any(counts <= 0):
        raise RuntimeError(f"Cannot compute empirical latent stats; empty part counts={counts.detach().cpu().tolist()}")
    mean = sums / counts[:, None].clamp_min(1.0)
    var = (sumsq / counts[:, None].clamp_min(1.0) - mean.square()).clamp_min(0.0)
    std = var.sqrt().clamp_min(float(cfg.latent_norm_eps))
    model.set_latent_stats(mean.float(), std.float())
    model.train(was_training)

    summary = {
        "latent_norm_mode": str(cfg.latent_norm_mode),
        "vq_backend": str(cfg.vq_backend),
        "kv_part_target_mode": str(cfg.kv_part_target_mode),
        "rvq_target_mode": str(cfg.rvq_target_mode),
        "num_parts": int(cfg.num_parts),
        "code_dim": int(cfg.code_dim),
        "counts_min": float(counts.min().detach().cpu()),
        "counts_max": float(counts.max().detach().cpu()),
        "mean_abs_avg": float(mean.abs().mean().detach().cpu()),
        "std_avg": float(std.mean().detach().cpu()),
        "std_min": float(std.min().detach().cpu()),
        "std_max": float(std.max().detach().cpu()),
    }
    master_print(rank, "LATENT_STATS " + json.dumps(summary, sort_keys=True))
    if rank == 0:
        write_json(out_dir / "logs" / "latent_stats.json", summary)
    return summary


def append_jsonl(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=True) + "\n")


def write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_eval_components(opt, device, batch_size: int, num_workers: int):
    eval_opt = get_opt(str(dataset_opt_path(opt)), device)
    eval_opt.checkpoints_dir = "./checkpoints"
    eval_opt.save_root = str(Path(eval_opt.checkpoints_dir) / eval_opt.dataset_name / eval_opt.name)
    eval_opt.model_dir = str(Path(eval_opt.save_root) / "model")
    eval_opt.meta_dir = str(Path(eval_opt.save_root) / "meta")
    eval_opt.data_root = opt.data_root
    eval_opt.motion_dir = str(Path(opt.data_root) / "new_joint_vecs")
    eval_opt.text_dir = str(Path(opt.data_root) / "texts")
    eval_opt.unit_length = int(opt.unit_length)

    mean = np.load(str(Path(eval_opt.meta_dir) / "mean.npy")).astype(np.float32)
    std = np.load(str(Path(eval_opt.meta_dir) / "std.npy")).astype(np.float32)
    split_file = str(Path(opt.data_root) / f"{_eval_split(opt)}.txt")
    w_vectorizer = WordVectorizer("./glove", "our_vab")
    dataset = Text2MotionDatasetEval(eval_opt, mean, std, split_file, w_vectorizer)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=True,
        collate_fn=collate_fn,
        shuffle=True,
    )
    eval_wrapper = EvaluatorModelWrapper(eval_opt)
    vq_mean = np.load(opt.mean_path).astype(np.float32)
    vq_std = np.load(opt.std_path).astype(np.float32)
    return loader, dataset, eval_wrapper, vq_mean, vq_std


def build_full_eval_components(opt, device):
    return build_eval_components(
        opt,
        device,
        batch_size=int(opt.full_eval_batch_size),
        num_workers=int(opt.full_eval_num_workers),
    )


def capture_rng_state(device: torch.device) -> Dict[str, object]:
    state: Dict[str, object] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if device.type == "cuda" and torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Dict[str, object]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def clone_trainable_state(model: MotionCodeFlow) -> Dict[str, torch.Tensor]:
    return {
        key: value.detach().clone()
        for key, value in model.state_dict_for_save().items()
    }


def aggregate_full_eval_metrics(repeat_metrics: List[Dict[str, object]]) -> Dict[str, float]:
    aggregate: Dict[str, float] = {}
    if not repeat_metrics:
        return aggregate
    keys = sorted(set().union(*(metrics.keys() for metrics in repeat_metrics)))
    for key in keys:
        values = []
        for metrics in repeat_metrics:
            value = metrics.get(key)
            if isinstance(value, (int, float, np.integer, np.floating)):
                values.append(float(value))
        if values:
            aggregate[key] = float(np.mean(values))
    return aggregate


@torch.no_grad()
def run_full_eval(
    model: MotionCodeFlow,
    opt,
    components,
    ema: Dict[str, torch.Tensor],
    completed_epoch: int,
    global_step: int,
    device: torch.device,
) -> Dict[str, object]:
    loader, dataset, eval_wrapper, vq_mean, vq_std = components
    repeat_times = int(opt.full_eval_repeat_times)
    if repeat_times <= 0:
        raise ValueError(f"full_eval_repeat_times must be positive, got {repeat_times}")

    rng_state = capture_rng_state(device)
    was_training = model.training
    raw_state = None
    weight_source = "model"
    try:
        if not opt.disable_full_eval_ema and ema:
            raw_state = clone_trainable_state(model)
            missing, unexpected = model.load_trainable_state_dict(ema)
            if missing or unexpected:
                raise RuntimeError(f"EMA eval load mismatch: missing={missing[:8]} unexpected={unexpected[:8]}")
            weight_source = "ema"

        eval_cfg = CodeFlowEvalConfig(
            steps=int(opt.full_eval_steps),
            cond_scale=float(opt.full_eval_cond_scale),
            terminal_mode=None,
            unit_length=int(opt.unit_length),
            max_batches=0,
            cal_mm=False,
            include_code_metrics=bool(getattr(model.tokenizer, "supports_code_metrics", True)),
            geometry_severe_quantile=float(opt.geometry_severe_quantile),
        )
        full_eval_seed = int(getattr(opt, "full_eval_seed", opt.seed))
        eval_seeds: List[int] = []
        repeat_metrics: List[Dict[str, object]] = []
        for repeat_id in range(repeat_times):
            eval_seed = full_eval_seed + repeat_id
            eval_seeds.append(eval_seed)
            fixseed(eval_seed)
            metrics = evaluate_codeflow_t2m(
                loader=loader,
                model=model,
                eval_wrapper=eval_wrapper,
                vq_mean=vq_mean,
                vq_std=vq_std,
                eval_mean=dataset.mean,
                eval_std=dataset.std,
                cfg=eval_cfg,
                repeat_id=repeat_id,
            )
            repeat_metrics.append(metrics)
        aggregate = aggregate_full_eval_metrics(repeat_metrics)
        payload: Dict[str, object] = {
            "epoch": int(completed_epoch),
            "step": int(global_step),
            "split": _eval_split(opt),
            "weight_source": weight_source,
            "repeat_times": repeat_times,
            "eval_steps": int(opt.full_eval_steps),
            "eval_start_epoch": int(getattr(opt, "full_eval_start_epoch", 0)),
            "cond_scale": float(opt.full_eval_cond_scale),
            "full_eval_seed": full_eval_seed,
            "eval_seeds": eval_seeds,
            "sampling_schedule": str(opt.sampling_schedule),
            "sampling_method": str(opt.sampling_method),
            "sde_gamma": float(opt.sde_gamma),
            "decode_mode": DECODE_MODE,
            "vq_backend": str(getattr(opt, "vq_backend", "kv_part")),
            "latent_norm_mode": str(opt.latent_norm_mode),
            "kv_part_target_mode": str(getattr(opt, "kv_part_target_mode", "codebook")),
            "rvq_target_mode": str(getattr(opt, "rvq_target_mode", "stage")),
            "latent_offset": float(opt.latent_offset),
            "aggregate": aggregate,
            "repeat_metrics": repeat_metrics,
        }
        payload.update({f"full_eval_{key}": value for key, value in aggregate.items()})
        return payload
    finally:
        if raw_state is not None:
            missing, unexpected = model.load_trainable_state_dict(raw_state)
            if missing or unexpected:
                raise RuntimeError(f"Raw state restore mismatch: missing={missing[:8]} unexpected={unexpected[:8]}")
        restore_rng_state(rng_state)
        model.train(was_training)


def make_best_selection_config(opt) -> Dict[str, object]:
    return {
        "dataset_name": normalized_dataset_name(getattr(opt, "dataset_name", "t2m")),
        "split": _eval_split(opt),
        "weight_source": "model" if opt.disable_full_eval_ema else "ema",
        "repeat_times": int(opt.full_eval_repeat_times),
        "eval_steps": int(opt.full_eval_steps),
        "cond_scale": float(opt.full_eval_cond_scale),
        "full_eval_seed": int(getattr(opt, "full_eval_seed", opt.seed)),
        "sampling_schedule": str(opt.sampling_schedule),
        "sampling_method": str(opt.sampling_method),
        "sde_gamma": float(opt.sde_gamma),
        "decode_mode": DECODE_MODE,
        "vq_backend": str(getattr(opt, "vq_backend", "kv_part")),
        "partvae_checkpoint": "",
        "partvae_config": "",
        "partvae_partition": "",
        "partvae_sample_latent": False,
        "latent_norm_mode": str(opt.latent_norm_mode),
        "kv_part_target_mode": str(getattr(opt, "kv_part_target_mode", "codebook")),
        "rvq_target_mode": str(getattr(opt, "rvq_target_mode", "stage")),
        "latent_offset": float(opt.latent_offset),
        "terminal_mode": str(opt.terminal_mode),
        "text_encoder_type": str(getattr(opt, "text_encoder_type", "clip")),
        "text_cache_fingerprint": str(getattr(opt, "text_cache_fingerprint", "")),
        "qwen_max_length": int(getattr(opt, "qwen_max_length", 0)),
        "text_refiner_depth": int(getattr(opt, "text_refiner_depth", 0)),
        "text_refiner_mlp_ratio": float(getattr(opt, "text_refiner_mlp_ratio", 0.0)),
        "text_refiner_pool": str(getattr(opt, "text_refiner_pool", "all_tokens")),
        "text_pooled_mode": str(getattr(opt, "text_pooled_mode", "modulation")),
        "unit_length": int(opt.unit_length),
        "best_checkpoint_limit": int(getattr(opt, "best_checkpoint_limit", DEFAULT_BEST_CHECKPOINT_LIMIT)),
    }


def selection_configs_match(saved: Optional[Dict[str, object]], expected: Dict[str, object]) -> bool:
    if not isinstance(saved, dict):
        return False
    saved_normalized = dict(saved)
    expected_normalized = dict(expected)
    # The eval start epoch controls scheduling only; it must not reset comparable
    # checkpoint selection when a run changes from delayed eval to eval-from-start.
    saved_normalized.pop("eval_start_epoch", None)
    expected_normalized.pop("eval_start_epoch", None)
    if (
        "best_checkpoint_limit" not in saved_normalized
        and expected_normalized.get("best_checkpoint_limit") == DEFAULT_BEST_CHECKPOINT_LIMIT
    ):
        saved_normalized["best_checkpoint_limit"] = DEFAULT_BEST_CHECKPOINT_LIMIT
    if "dataset_name" not in saved_normalized and expected_normalized.get("dataset_name") == "t2m":
        saved_normalized["dataset_name"] = "t2m"
    # A run started before the refiner-pool flag existed pooled over all tokens.
    # Without this shim its best-FID/best-Top3 history would be discarded on the
    # first resume even when the resume correctly asks for "all_tokens".
    if (
        "text_refiner_pool" not in saved_normalized
        and expected_normalized.get("text_refiner_pool") == "all_tokens"
    ):
        saved_normalized["text_refiner_pool"] = "all_tokens"
    if (
        "text_pooled_mode" not in saved_normalized
        and expected_normalized.get("text_pooled_mode") == "modulation"
    ):
        saved_normalized["text_pooled_mode"] = "modulation"
    # With no refiner there is nothing to pool, so the flag provably does not
    # affect the run.  load_checkpoint already exempts it at depth 0; leaving it
    # in this comparison would discard a resumed run's best-metric history over
    # a setting that changed nothing.
    if int(expected_normalized.get("text_refiner_depth", 0)) == 0:
        saved_normalized.pop("text_refiner_pool", None)
        expected_normalized.pop("text_refiner_pool", None)
    return saved_normalized == expected_normalized


def load_best_metrics(path: Path, expected_selection_config: Optional[Dict[str, object]] = None) -> Dict[str, object]:
    if expected_selection_config is None:
        expected_selection_config = {}
    if not path.is_file():
        return {"selection_config": expected_selection_config} if expected_selection_config else {}
    best_metrics = json.loads(path.read_text(encoding="utf-8"))
    if expected_selection_config:
        saved_selection_config = best_metrics.get("selection_config")
        if not selection_configs_match(saved_selection_config, expected_selection_config):
            return {
                "selection_config": expected_selection_config,
                "ignored_previous_best_metrics": {
                    "path": str(path),
                    "reason": "selection_config_mismatch_or_missing",
                    "saved_selection_config": saved_selection_config,
                    "expected_selection_config": expected_selection_config,
                },
            }
    return best_metrics


def checkpoint_rank_path(out_dir: Path, metric_key: str, rank_index: int) -> Path:
    if rank_index == 0:
        return out_dir / "model" / f"{metric_key}.pt"
    return out_dir / "model" / f"{metric_key}_rank{rank_index + 1}.pt"


def get_best_checkpoint_limit(opt) -> int:
    limit = int(getattr(opt, "best_checkpoint_limit", DEFAULT_BEST_CHECKPOINT_LIMIT))
    if limit <= 0:
        raise ValueError(f"best_checkpoint_limit must be positive, got {limit}")
    return limit


def metric_topk_key(metric_key: str, limit: int) -> str:
    return f"{metric_key}_top{limit}"


def metric_record_id(record: Dict[str, object]) -> Tuple[int, int]:
    return int(record["epoch"]), int(record["step"])


def metric_sort_key(metric_key: str, record: Dict[str, object]) -> Tuple[float, float, int, int]:
    fid = float(record["fid"])
    top3 = float(record["top3"])
    epoch, step = metric_record_id(record)
    if metric_key == "best_fid":
        return fid, -top3, epoch, step
    if metric_key == "best_top3":
        return -top3, fid, epoch, step
    raise ValueError(f"Unknown metric key: {metric_key}")


def normalize_best_record(record: Dict[str, object], selection_config: Dict[str, object]) -> Dict[str, object]:
    normalized = {
        "fid": float(record["fid"]),
        "top3": float(record["top3"]),
        "epoch": int(record["epoch"]),
        "step": int(record["step"]),
        "checkpoint": str(record["checkpoint"]),
        "selection_config": selection_config,
    }
    return normalized


def get_metric_records(
    best_metrics: Dict[str, object],
    metric_key: str,
    selection_config: Dict[str, object],
    limit: int,
) -> List[Dict[str, object]]:
    raw_records = best_metrics.get(metric_topk_key(metric_key, limit))
    if not isinstance(raw_records, list) or not raw_records:
        single_record = best_metrics.get(metric_key)
        raw_records = [single_record] if isinstance(single_record, dict) else []

    records_by_id: Dict[Tuple[int, int], Dict[str, object]] = {}
    for raw in raw_records:
        if not isinstance(raw, dict):
            continue
        if not {"fid", "top3", "epoch", "step", "checkpoint"}.issubset(raw):
            continue
        normalized = normalize_best_record(raw, selection_config)
        records_by_id[metric_record_id(normalized)] = normalized
    records = sorted(records_by_id.values(), key=lambda item: metric_sort_key(metric_key, item))
    return records[:limit]


def build_best_record(
    fid: float,
    top3: float,
    completed_epoch: int,
    global_step: int,
    checkpoint: str,
    selection_config: Dict[str, object],
) -> Dict[str, object]:
    return {
        "fid": float(fid),
        "top3": float(top3),
        "epoch": int(completed_epoch),
        "step": int(global_step),
        "checkpoint": checkpoint,
        "selection_config": selection_config,
    }


def commit_metric_topk_files(
    metric_key: str,
    new_topk: List[Dict[str, object]],
    previous_records: List[Dict[str, object]],
    new_record_id: Tuple[int, int],
    out_dir: Path,
    model: MotionCodeFlow,
    optimizer,
    scaler,
    ema: Dict[str, torch.Tensor],
    completed_epoch: int,
    global_step: int,
    opt,
    limit: int,
) -> List[Dict[str, object]]:
    model_dir = out_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)

    previous_by_id = {metric_record_id(record): record for record in previous_records}
    new_record_included = new_record_id in {metric_record_id(record) for record in new_topk}
    if not new_record_included:
        return new_topk

    save_tmp = model_dir / f".{metric_key}_epoch{completed_epoch:04d}_step{global_step}_new.pt"
    if save_tmp.exists() or save_tmp.is_symlink():
        save_tmp.unlink()
    save_checkpoint(save_tmp, model, optimizer, scaler, ema, completed_epoch, global_step, opt)

    temp_sources: Dict[Tuple[int, int], Path] = {new_record_id: save_tmp}
    retained_previous_ids = [metric_record_id(record) for record in new_topk if metric_record_id(record) in previous_by_id]
    for index, record_id in enumerate(retained_previous_ids):
        source = Path(str(previous_by_id[record_id]["checkpoint"]))
        if not source.is_file():
            raise FileNotFoundError(f"Cannot retain missing {metric_key} checkpoint: {source}")
        move_tmp = model_dir / f".{metric_key}_move_{index}_{os.getpid()}.pt"
        if move_tmp.exists() or move_tmp.is_symlink():
            move_tmp.unlink()
        source.replace(move_tmp)
        temp_sources[record_id] = move_tmp

    target_paths = [checkpoint_rank_path(out_dir, metric_key, rank_index) for rank_index in range(len(new_topk))]
    for target in target_paths:
        if target.exists() or target.is_symlink():
            target.unlink()

    committed: List[Dict[str, object]] = []
    for rank_index, record in enumerate(new_topk):
        record_id = metric_record_id(record)
        source = temp_sources[record_id]
        target = checkpoint_rank_path(out_dir, metric_key, rank_index)
        source.replace(target)
        committed_record = dict(record)
        committed_record["checkpoint"] = str(target)
        committed.append(committed_record)

    retained_ids = {metric_record_id(record) for record in new_topk}
    target_path_set = {str(path) for path in target_paths}
    for old_record in previous_records:
        if metric_record_id(old_record) in retained_ids:
            continue
        old_path = Path(str(old_record["checkpoint"]))
        if str(old_path) not in target_path_set and (old_path.exists() or old_path.is_symlink()):
            old_path.unlink()

    for rank_index in range(len(new_topk), limit):
        stale_path = checkpoint_rank_path(out_dir, metric_key, rank_index)
        if stale_path.exists() or stale_path.is_symlink():
            stale_path.unlink()

    return committed


def update_metric_topk(
    best_metrics: Dict[str, object],
    metric_key: str,
    fid: float,
    top3: float,
    out_dir: Path,
    model: MotionCodeFlow,
    optimizer,
    scaler,
    ema: Dict[str, torch.Tensor],
    completed_epoch: int,
    global_step: int,
    opt,
) -> Tuple[Dict[str, object], Optional[str]]:
    selection_config = best_metrics["selection_config"]
    limit = get_best_checkpoint_limit(opt)
    previous_records = get_metric_records(best_metrics, metric_key, selection_config, limit)
    previous_rank1_id = metric_record_id(previous_records[0]) if previous_records else None
    new_record_id = (int(completed_epoch), int(global_step))

    records_by_id = {metric_record_id(record): record for record in previous_records}
    new_record_is_new = new_record_id not in records_by_id
    if new_record_is_new:
        records_by_id[new_record_id] = build_best_record(
            fid=fid,
            top3=top3,
            completed_epoch=completed_epoch,
            global_step=global_step,
            checkpoint="",
            selection_config=selection_config,
        )
    candidates = sorted(records_by_id.values(), key=lambda item: metric_sort_key(metric_key, item))
    new_topk = candidates[:limit]
    new_topk_ids = {metric_record_id(record) for record in new_topk}
    if not new_record_is_new or new_record_id not in new_topk_ids:
        best_metrics[metric_topk_key(metric_key, limit)] = previous_records
        if previous_records:
            best_metrics[metric_key] = previous_records[0]
        return best_metrics, None

    committed_topk = commit_metric_topk_files(
        metric_key=metric_key,
        new_topk=new_topk,
        previous_records=previous_records,
        new_record_id=new_record_id,
        out_dir=out_dir,
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        ema=ema,
        completed_epoch=completed_epoch,
        global_step=global_step,
        opt=opt,
        limit=limit,
    )
    best_metrics[metric_topk_key(metric_key, limit)] = committed_topk
    best_metrics[metric_key] = committed_topk[0]
    if metric_record_id(committed_topk[0]) != previous_rank1_id:
        return best_metrics, metric_key
    return best_metrics, metric_topk_key(metric_key, limit)


def save_gated_checkpoint(
    eval_payload: Dict[str, object],
    out_dir: Path,
    model: MotionCodeFlow,
    optimizer,
    scaler,
    ema: Dict[str, torch.Tensor],
    completed_epoch: int,
    global_step: int,
    opt,
) -> Optional[str]:
    """Keep every checkpoint that clears BOTH thresholds at once.

    best_fid.pt and best_top3.pt each optimise one metric and therefore sit at
    opposite ends of the trajectory; the region where both metrics are
    simultaneously good is in between and was never retained.  On the 08-02
    campaign three such points were measured (e1_nopool ep200/225/250) and all
    three were lost, because none of them was either single-metric optimum.
    This gate exists so that region survives.

    Disabled unless both thresholds are positive.  No cap: an unrequested cap
    would silently drop exactly the checkpoints the gate was asked to keep, so
    the count and cumulative size are logged instead.
    """
    gate_fid = float(getattr(opt, "gate_save_fid", 0.0) or 0.0)
    gate_top3 = float(getattr(opt, "gate_save_top3", 0.0) or 0.0)
    if gate_fid <= 0.0 or gate_top3 <= 0.0:
        return None

    aggregate = eval_payload["aggregate"]
    fid = float(aggregate["fid"])
    top3 = float(aggregate["top3"])
    if not (fid < gate_fid and top3 > gate_top3):
        return None

    gate_dir = out_dir / "model" / "gate"
    path = gate_dir / f"gate_ep{completed_epoch:04d}_fid{fid:.4f}_top3{top3:.4f}.pt"
    save_checkpoint(path, model, optimizer, scaler, ema, completed_epoch, global_step, opt)

    kept = sorted(gate_dir.glob("gate_ep*.pt"))
    total_gib = sum(p.stat().st_size for p in kept) / (1024 ** 3)
    append_jsonl(
        out_dir / "logs" / "gate_saves.jsonl",
        {
            "epoch": completed_epoch,
            "step": global_step,
            "fid": fid,
            "top3": top3,
            "top1": float(aggregate["top1"]),
            "matching_score": float(aggregate["matching_score"]),
            "gate_save_fid": gate_fid,
            "gate_save_top3": gate_top3,
            "path": str(path),
            "kept_count": len(kept),
            "kept_gib": round(total_gib, 2),
        },
    )
    return f"gate({len(kept)} kept, {total_gib:.0f}GiB)"


def update_best_checkpoints(
    best_metrics: Dict[str, object],
    eval_payload: Dict[str, object],
    out_dir: Path,
    model: MotionCodeFlow,
    optimizer,
    scaler,
    ema: Dict[str, torch.Tensor],
    completed_epoch: int,
    global_step: int,
    opt,
) -> Tuple[Dict[str, object], List[str]]:
    aggregate = eval_payload["aggregate"]
    fid = float(aggregate["fid"])
    top3 = float(aggregate["top3"])
    updated: List[str] = []
    best_metrics["selection_config"] = make_best_selection_config(opt)

    for metric_key in ("best_fid", "best_top3"):
        best_metrics, update_label = update_metric_topk(
            best_metrics=best_metrics,
            metric_key=metric_key,
            fid=fid,
            top3=top3,
            out_dir=out_dir,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            ema=ema,
            completed_epoch=completed_epoch,
            global_step=global_step,
            opt=opt,
        )
        if update_label is not None:
            updated.append(update_label)

    gate_label = save_gated_checkpoint(
        eval_payload=eval_payload,
        out_dir=out_dir,
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        ema=ema,
        completed_epoch=completed_epoch,
        global_step=global_step,
        opt=opt,
    )
    if gate_label is not None:
        updated.append(gate_label)

    return best_metrics, updated


def main(
    model_cls: Type[MotionCodeFlow] = MotionCodeFlow,
    option_defaults: Dict[str, object] = None,
):
    option_parser = TrainCodeFlowOptions()
    if option_defaults:
        option_parser.parser.set_defaults(**option_defaults)
    opt = option_parser.parse()
    if str(getattr(opt, "rvq_target_mode", "stage")) != "stage" and str(opt.vq_backend) != "momask_rvq":
        raise ValueError("--rvq_target_mode sum is only valid with --vq_backend momask_rvq")
    if str(getattr(opt, "kv_part_target_mode", "codebook")) != "codebook" and str(opt.vq_backend) != "kv_part":
        raise ValueError("--kv_part_target_mode encoder is only valid with --vq_backend kv_part")
    if str(getattr(opt, "rvq_target_mode", "stage")) == "sum":
        if int(opt.num_parts) != 1:
            raise ValueError("--rvq_target_mode sum requires --num_parts 1")
        if str(opt.terminal_mode) != "residual_nearest":
            raise ValueError("--rvq_target_mode sum requires --terminal_mode residual_nearest")
        if str(opt.terminal_tau_mode) != "fixed":
            raise ValueError("--rvq_target_mode sum requires --terminal_tau_mode fixed")
        if str(opt.latent_norm_mode) not in {"none", "empirical"}:
            raise ValueError("--rvq_target_mode sum requires --latent_norm_mode none or empirical")
    if str(opt.vq_backend) == "partvae_continuous":
        # MEMORY.md "PartVAE Continuous CodeFlow Baseline" contract: posterior-mean
        # targets, no ids, no codebooks, decode_embeddings only.
        if str(opt.terminal_mode) != "none":
            raise ValueError("--vq_backend partvae_continuous requires --terminal_mode none")
        if str(opt.terminal_tau_mode) != "fixed":
            raise ValueError("--vq_backend partvae_continuous requires --terminal_tau_mode fixed")
        if str(opt.latent_norm_mode) != "empirical":
            raise ValueError(
                "--vq_backend partvae_continuous requires --latent_norm_mode empirical "
                "(there are no codebooks to derive codebook stats from)"
            )
        if int(opt.num_codes) != 0:
            raise ValueError("--vq_backend partvae_continuous requires --num_codes 0")
    use_ddp, rank, world_size, local_rank, device = setup_distributed(opt)
    fixseed(opt.seed + rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    out_dir = Path(opt.output_dir).expanduser().resolve()
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        with (out_dir / "options.json").open("w", encoding="utf-8") as f:
            json.dump(vars(opt), f, indent=2, sort_keys=True)

    stats_summary = validate_stats_contract(opt)
    master_print(rank, "STATS_CONTRACT " + json.dumps(stats_summary, sort_keys=True))

    train_dataset = build_train_dataset(opt)
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True) if use_ddp else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    lr_schedule = build_lr_schedule(opt, len(train_loader))
    attach_lr_schedule_to_options(opt, lr_schedule)
    master_print(rank, "LR_SCHEDULE " + json.dumps(lr_schedule, sort_keys=True))
    if rank == 0:
        with (out_dir / "options.json").open("w", encoding="utf-8") as f:
            json.dump(vars(opt), f, indent=2, sort_keys=True)
        write_json(out_dir / "logs" / "scheduler.json", lr_schedule)

    if getattr(opt, "resume", "") and getattr(opt, "init_checkpoint", ""):
        raise ValueError("--resume and --init_checkpoint are mutually exclusive")

    model = model_cls(make_config(opt)).to(device)
    if hasattr(model.text_encoder, "cache_summary"):
        text_cache_summary = model.text_encoder.cache_summary()
        opt.text_cache_fingerprint = str(text_cache_summary["semantic_fingerprint"])
        opt.text_cache_content_fingerprint = str(text_cache_summary["content_fingerprint"])
        opt.text_cache_captions_sha256 = str(text_cache_summary["captions_sha256"])
        opt.text_cache_clip_revision = str(text_cache_summary["clip_revision"])
        opt.text_cache_qwen_revision = str(text_cache_summary["qwen_revision"])
        master_print(rank, "TEXT_CACHE " + json.dumps(text_cache_summary, sort_keys=True))
        cache_coverage = model.text_encoder.validate_coverage(
            collect_text_cache_preflight_captions(opt.data_root)
        )
        master_print(rank, "TEXT_CACHE_COVERAGE " + json.dumps(cache_coverage, sort_keys=True))
        if rank == 0:
            with (out_dir / "options.json").open("w", encoding="utf-8") as f:
                json.dump(vars(opt), f, indent=2, sort_keys=True)
    tokenizer_downsample = getattr(model.tokenizer, "downsample_factor", None)
    if tokenizer_downsample is not None and int(opt.unit_length) != int(tokenizer_downsample):
        raise ValueError(
            f"--unit_length must match tokenizer downsample_factor={int(tokenizer_downsample)} "
            f"for {getattr(model.tokenizer, 'target_mode', 'tokenizer')} targets; got {int(opt.unit_length)}"
        )
    if str(getattr(opt, "latent_norm_mode", "none")) == "empirical":
        compute_and_set_empirical_latent_stats(
            model=model,
            loader=train_loader,
            device=device,
            unit_length=int(opt.unit_length),
            out_dir=out_dir,
            rank=rank,
        )
    if getattr(opt, "init_checkpoint", ""):
        init_path = Path(opt.init_checkpoint).expanduser().resolve()
        if not init_path.is_file():
            raise FileNotFoundError(f"init_checkpoint not found: {init_path}")
        missing, unexpected, init_source = load_initial_trainable_state(
            init_path,
            model,
            use_ema=bool(getattr(opt, "init_checkpoint_use_ema", False)),
        )
        if missing or unexpected:
            raise RuntimeError(f"Init checkpoint load mismatch: missing={missing[:8]} unexpected={unexpected[:8]}")
        master_print(rank, f"Initialized trainable weights from {init_path} source={init_source}")
    run_vq_contract_check(model, train_loader, device, opt, rank)
    master_print(rank, f"Trainable code-flow parameters: {count_trainable(model) / 1e6:.2f}M")
    param_groups, param_group_sizes = build_optimizer_param_groups(model, opt)
    trainable_params = flatten_optimizer_params(param_groups)
    master_print(rank, "OPTIMIZER_GROUPS " + json.dumps(param_group_sizes, sort_keys=True))
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=opt.lr,
        betas=tuple(opt.betas),
        weight_decay=opt.weight_decay,
    )
    use_grad_scaler = bool(opt.amp and device.type == "cuda" and getattr(opt, "amp_dtype", "fp16") == "fp16")
    scaler = torch.cuda.amp.GradScaler(enabled=use_grad_scaler)
    start_epoch, global_step = 0, 0
    ema = make_ema(model)
    if opt.resume:
        start_epoch, global_step, loaded_ema = load_checkpoint(Path(opt.resume), model, optimizer, scaler, opt=opt)
        move_optimizer_state_to_device(optimizer, device)
        if loaded_ema is not None:
            ema = {key: value.to(device) for key, value in loaded_ema.items()}
        master_print(rank, f"Resumed from {opt.resume} at epoch={start_epoch} step={global_step}")

    loss_module = CodeFlowLossModule(model).to(device)
    if use_ddp:
        loss_module = DDP(
            loss_module,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
        )

    full_eval_components = None
    best_metrics: Dict[str, object] = {}
    if rank == 0 and int(opt.full_eval_every_epoch) > 0:
        full_eval_components = build_full_eval_components(opt, device)
        best_metrics = load_best_metrics(out_dir / "logs" / "best_metrics.json", make_best_selection_config(opt))
        ignored_previous = best_metrics.get("ignored_previous_best_metrics")
        if ignored_previous:
            master_print(rank, "BEST_METRICS_RESET " + json.dumps(ignored_previous, sort_keys=True))
    if is_dist():
        dist.barrier()

    log_path = out_dir / "logs" / "train.jsonl"
    model.train()
    stop = False
    for epoch in range(start_epoch, opt.max_epoch):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        for batch in train_loader:
            captions, motions, lengths = batch
            motions = motions.float().to(device, non_blocking=True)
            lengths = lengths.to(device, non_blocking=True)
            ids, embeddings, token_lengths = encode_batch(model, motions, lengths, opt.unit_length)
            text_list = list(captions)
            lr = lr_at_step(global_step, opt, lr_schedule)
            update_optimizer_lr(optimizer, lr)
            noise, timesteps, x_self_cond = prepare_flow_training_state(
                model,
                embeddings,
                text_list,
                token_lengths,
                opt,
                device,
            )

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp_enabled(opt, device), dtype=amp_dtype_from_options(opt)):
                losses = loss_module(
                    embeddings,
                    ids,
                    text_list,
                    token_lengths,
                    noise=noise,
                    timesteps=timesteps,
                    x_self_cond=x_self_cond,
                    allow_internal_self_condition=False,
                )
                loss = losses["loss"]
            scaler.scale(loss).backward()
            do_grad_health_check = (
                not bool(opt.disable_grad_health_check)
                and int(opt.grad_health_check_steps) > 0
                and global_step < int(opt.grad_health_check_steps)
            )
            if opt.grad_clip > 0 or do_grad_health_check:
                scaler.unscale_(optimizer)
            if do_grad_health_check:
                check_core_grad_health(model, loss, global_step, rank, opt.grad_health_min_abs)
                core_weight = getattr(model, "core_output_weight", model.holder_output.linear.weight)
                core_weight_before = core_weight.detach().float().clone()
            else:
                core_weight = None
                core_weight_before = None
            if opt.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, opt.grad_clip)
            scale_before = float(scaler.get_scale())
            scaler.step(optimizer)
            scaler.update()
            scale_after = float(scaler.get_scale())
            if do_grad_health_check:
                if bool(opt.amp and device.type == "cuda") and scale_after < scale_before:
                    raise RuntimeError(
                        f"AMP GradScaler skipped optimizer step at step={global_step} rank={rank}: "
                        f"scale {scale_before:.6e}->{scale_after:.6e}"
                    )
                core_delta = float((core_weight.detach().float() - core_weight_before).abs().max().item())
                if core_delta <= 0.0:
                    raise RuntimeError(
                        f"core output weight did not change after optimizer step at "
                        f"step={global_step} rank={rank}"
                    )
            update_ema(ema, model)

            if global_step % opt.log_every == 0:
                metrics = {key: float(value.detach().cpu()) for key, value in losses.items()}
                metrics.update({"epoch": epoch, "step": global_step, "lr": lr, "batch_kind": "base"})
                master_print(rank, " ".join([f"{key}={value:.5f}" if isinstance(value, float) else f"{key}={value}" for key, value in metrics.items()]))
                if rank == 0:
                    append_jsonl(log_path, metrics)

            global_step += 1
            if opt.max_steps > 0 and global_step >= opt.max_steps:
                stop = True
                break
        completed_epoch = epoch + 1
        if rank == 0:
            save_checkpoint(out_dir / "model" / "latest.pt", model, optimizer, scaler, ema, completed_epoch, global_step, opt)

        full_eval_every_epoch = int(opt.full_eval_every_epoch)
        full_eval_start_epoch = max(0, int(getattr(opt, "full_eval_start_epoch", 0)))
        run_full = (
            full_eval_every_epoch > 0
            and completed_epoch >= full_eval_start_epoch
            and (completed_epoch - full_eval_start_epoch) % full_eval_every_epoch == 0
        )
        if run_full:
            if is_dist():
                dist.barrier()
            if rank == 0:
                if full_eval_components is None:
                    raise RuntimeError("full_eval_components were not initialized on rank 0")
                eval_payload = run_full_eval(
                    model=model,
                    opt=opt,
                    components=full_eval_components,
                    ema=ema,
                    completed_epoch=completed_epoch,
                    global_step=global_step,
                    device=device,
                )
                append_jsonl(out_dir / "logs" / "full_eval.jsonl", eval_payload)
                best_metrics, updated = update_best_checkpoints(
                    best_metrics=best_metrics,
                    eval_payload=eval_payload,
                    out_dir=out_dir,
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    ema=ema,
                    completed_epoch=completed_epoch,
                    global_step=global_step,
                    opt=opt,
                )
                write_json(out_dir / "logs" / "best_metrics.json", best_metrics)
                aggregate = eval_payload["aggregate"]
                updated_msg = ",".join(updated) if updated else "none"
                master_print(
                    rank,
                    "FULL_EVAL "
                    f"epoch={completed_epoch} step={global_step} "
                    f"fid={float(aggregate['fid']):.5f} top3={float(aggregate['top3']):.5f} "
                    f"top1={float(aggregate['top1']):.5f} matching={float(aggregate['matching_score']):.5f} "
                    f"updated={updated_msg}",
                )
            if is_dist():
                dist.barrier()
        if stop:
            break

    cleanup_distributed()
