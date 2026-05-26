import json
import math
import time
from dataclasses import asdict, dataclass
from os.path import join as pjoin
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from data.t2m_dataset import Text2MotionDataset
from models.mask_transformer.transformer import ResidualTransformer
from models.vq.model import RVQVAE
from options.train_option import TrainT2MOptions
from utils.fixseed import fixseed
from utils.get_opt import get_opt


@dataclass
class ScheduleSummary:
    train_size: int
    iters_per_epoch: int
    total_iters: int
    warmup_iters: int
    warmup_epochs: float
    milestone_hits: List[int]
    warnings: List[str]


def build_parser():
    builder = TrainT2MOptions()
    builder.initialize()
    parser = builder.parser
    parser.description = "Short-run sweep for residual transformer hyperparameters."
    parser.add_argument(
        "--trial_sample_budget",
        type=int,
        default=32768,
        help="Total number of training samples consumed by the short trial.",
    )
    parser.add_argument(
        "--trial_steps",
        type=int,
        default=0,
        help="Override steps directly. If 0, derive from trial_sample_budget / batch_size.",
    )
    parser.add_argument(
        "--trial_warmup_steps",
        type=int,
        default=0,
        help="Warmup steps for the short trial. If 0, derive from trial_steps.",
    )
    parser.add_argument(
        "--trial_eval_batches",
        type=int,
        default=4,
        help="Number of validation batches used for the mini-val metric.",
    )
    parser.add_argument(
        "--train_subset_size",
        type=int,
        default=8192,
        help="Number of train examples to draw into the short-trial subset.",
    )
    parser.add_argument(
        "--val_subset_size",
        type=int,
        default=512,
        help="Number of val examples to draw into the mini-val subset.",
    )
    parser.add_argument(
        "--val_batch_size",
        type=int,
        default=128,
        help="Batch size used for the mini-val loader.",
    )
    parser.add_argument(
        "--subset_seed",
        type=int,
        default=1234,
        help="Seed for drawing fixed train/val subsets.",
    )
    parser.add_argument(
        "--trial_seed_offset",
        type=int,
        default=1000,
        help="Offset added to --seed before each short trial to keep subset seed separate.",
    )
    parser.add_argument(
        "--strict_schedule",
        action="store_true",
        help="Raise an error if the full-run schedule has obvious mismatches.",
    )
    parser.add_argument(
        "--save_json",
        type=str,
        default="",
        help="Optional path to save the final short-trial summary as JSON.",
    )
    return parser


def configure_dataset(opt) -> Tuple[int, str]:
    if opt.dataset_name == "t2m":
        default_data_root = "./dataset/HumanML3D"
        opt.data_root = opt.data_root or default_data_root
        opt.motion_dir = pjoin(opt.data_root, "new_joint_vecs")
        opt.joints_num = 22
        opt.max_motion_len = 55
        dim_pose = 263
    elif opt.dataset_name == "kit":
        default_data_root = "./dataset/KIT-ML"
        opt.data_root = opt.data_root or default_data_root
        opt.motion_dir = pjoin(opt.data_root, "new_joint_vecs")
        opt.joints_num = 21
        opt.max_motion_len = 55
        dim_pose = 251
    else:
        raise KeyError(f"Unsupported dataset_name: {opt.dataset_name}")
    opt.text_dir = pjoin(opt.data_root, "texts")
    return dim_pose, pjoin(opt.data_root, "train.txt")


def load_vq_model(opt, dim_pose):
    opt_path = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.vq_name, "opt.txt")
    vq_opt = get_opt(opt_path, opt.device)
    vq_model = RVQVAE(
        vq_opt,
        dim_pose,
        vq_opt.nb_code,
        vq_opt.code_dim,
        vq_opt.output_emb_width,
        vq_opt.down_t,
        vq_opt.stride_t,
        vq_opt.width,
        vq_opt.depth,
        vq_opt.dilation_growth_rate,
        vq_opt.vq_act,
        vq_opt.vq_norm,
    )
    ckpt = torch.load(
        pjoin(vq_opt.checkpoints_dir, vq_opt.dataset_name, vq_opt.name, "model", "net_best_fid.tar"),
        map_location=opt.device,
    )
    model_key = "vq_model" if "vq_model" in ckpt else "net"
    vq_model.load_state_dict(ckpt[model_key])
    vq_model.to(opt.device)
    vq_model.eval()
    return vq_model, vq_opt


def build_res_transformer(opt, vq_opt):
    opt.num_tokens = vq_opt.nb_code
    opt.num_quantizers = vq_opt.num_quantizers
    model = ResidualTransformer(
        code_dim=vq_opt.code_dim,
        cond_mode="text",
        latent_dim=opt.latent_dim,
        ff_size=opt.ff_size,
        num_layers=opt.n_layers,
        num_heads=opt.n_heads,
        dropout=opt.dropout,
        clip_dim=512,
        shared_codebook=vq_opt.shared_codebook,
        cond_drop_prob=opt.cond_drop_prob,
        share_weight=opt.share_weight,
        clip_version="ViT-B/32",
        opt=opt,
    )
    model.to(opt.device)
    return model


def compute_schedule_summary(train_size: int, opt) -> ScheduleSummary:
    if opt.batch_size <= 0:
        raise ValueError(f"batch_size must be > 0, got {opt.batch_size}")
    if opt.max_epoch <= 0:
        raise ValueError(f"max_epoch must be > 0, got {opt.max_epoch}")

    iters_per_epoch = train_size // opt.batch_size
    if iters_per_epoch <= 0:
        raise ValueError(
            f"batch_size={opt.batch_size} is larger than train_size={train_size}; full run would have zero updates."
        )
    total_iters = iters_per_epoch * opt.max_epoch
    milestone_hits = [m for m in opt.milestones if m <= total_iters]
    warmup_epochs = opt.warm_up_iter / iters_per_epoch
    warnings = []

    if not milestone_hits:
        warnings.append(
            f"No LR milestone is reachable: milestones={opt.milestones}, total_iters={total_iters}."
        )
    if opt.warm_up_iter >= total_iters:
        warnings.append(
            f"Warmup consumes the entire run: warm_up_iter={opt.warm_up_iter}, total_iters={total_iters}."
        )
    elif opt.warm_up_iter / total_iters > 0.1:
        warnings.append(
            f"Warmup is long: warm_up_iter={opt.warm_up_iter} ({opt.warm_up_iter / total_iters:.1%} of total steps)."
        )
    if getattr(opt, "res_use_uni_mask", False) and getattr(opt, "res_uni_solver_steps", 16) != 16:
        warnings.append(
            "res_uni_solver_steps affects generation/eval only, not the training forward pass."
        )

    return ScheduleSummary(
        train_size=train_size,
        iters_per_epoch=iters_per_epoch,
        total_iters=total_iters,
        warmup_iters=opt.warm_up_iter,
        warmup_epochs=warmup_epochs,
        milestone_hits=milestone_hits,
        warnings=warnings,
    )


def make_subset_indices(size: int, limit: int, seed: int) -> List[int]:
    if limit <= 0:
        raise ValueError(f"Subset size must be > 0, got {limit}")
    limit = min(size, limit)
    rng = np.random.default_rng(seed)
    indices = rng.choice(size, size=limit, replace=False)
    return indices.tolist()


def build_loader(dataset, indices: List[int], batch_size: int, shuffle: bool, seed: int, num_workers: int):
    if len(indices) < batch_size:
        raise ValueError(
            f"Subset has {len(indices)} samples but batch_size={batch_size}. Increase subset size or lower batch_size."
        )
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        Subset(dataset, indices),
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        drop_last=True,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )


def cycling(loader: DataLoader) -> Iterable:
    while True:
        for batch in loader:
            yield batch


def forward_batch(model, vq_model, batch_data, device):
    conds, motion, m_lens = batch_data
    motion = motion.detach().float().to(device)
    m_lens = m_lens.detach().long().to(device)
    with torch.no_grad():
        code_idx, _ = vq_model.encode(motion)
    m_lens = m_lens // 4
    conds = conds.to(device).float() if torch.is_tensor(conds) else conds
    ce_loss, _, acc = model(code_idx, conds, m_lens)
    return ce_loss, float(acc)


def evaluate(model, vq_model, val_loader, device, max_batches: int, eval_seed: int):
    if max_batches <= 0:
        raise ValueError(f"trial_eval_batches must be > 0, got {max_batches}")
    fixseed(eval_seed)
    model.eval()
    losses = []
    accs = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            if batch_idx >= max_batches:
                break
            loss, acc = forward_batch(model, vq_model, batch, device)
            losses.append(loss.item())
            accs.append(acc)
    if not losses:
        raise RuntimeError("Validation loader produced zero batches during short sweep.")
    return float(np.mean(losses)), float(np.mean(accs))


def pretty_schedule(summary: ScheduleSummary):
    print("Full-run schedule check:")
    print(f"  train_size={summary.train_size}")
    print(f"  iters_per_epoch={summary.iters_per_epoch}")
    print(f"  total_iters={summary.total_iters}")
    print(
        f"  warmup_iters={summary.warmup_iters} ({summary.warmup_epochs:.2f} epochs)"
    )
    print(f"  milestone_hits={summary.milestone_hits if summary.milestone_hits else 'none'}")
    if summary.warnings:
        print("  warnings:")
        for warning in summary.warnings:
            print(f"    - {warning}")


def run_short_trial(opt, train_loader, val_loader, full_train_size: int):
    schedule = compute_schedule_summary(full_train_size, opt)
    pretty_schedule(schedule)
    if opt.strict_schedule and schedule.warnings:
        raise RuntimeError("Schedule sanity check failed in strict mode.")

    trial_steps = opt.trial_steps or math.ceil(opt.trial_sample_budget / opt.batch_size)
    if trial_steps <= 0:
        raise ValueError(f"Derived trial_steps must be > 0, got {trial_steps}")
    trial_warmup_steps = opt.trial_warmup_steps or max(1, min(10, trial_steps // 10))

    trial_seed = opt.seed + opt.trial_seed_offset
    fixseed(trial_seed)

    dim_pose, _ = configure_dataset(opt)
    vq_model, vq_opt = load_vq_model(opt, dim_pose)
    model = build_res_transformer(opt, vq_opt)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        betas=(0.9, 0.99),
        lr=opt.lr,
        weight_decay=1e-5,
    )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(opt.device)

    train_iter = cycling(train_loader)
    model.train()
    losses = []
    accs = []
    step_times = []
    start_time = time.perf_counter()

    for step_idx in range(trial_steps):
        batch = next(train_iter)
        if step_idx < trial_warmup_steps:
            warmup_lr = opt.lr * float(step_idx + 1) / float(trial_warmup_steps + 1)
            for group in optimizer.param_groups:
                group["lr"] = warmup_lr
        else:
            for group in optimizer.param_groups:
                group["lr"] = opt.lr

        if torch.cuda.is_available():
            torch.cuda.synchronize(opt.device)
        step_start = time.perf_counter()

        loss, acc = forward_batch(model, vq_model, batch, opt.device)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Encountered non-finite loss at step {step_idx}: {loss.item()}")

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if torch.cuda.is_available():
            torch.cuda.synchronize(opt.device)
        step_times.append(time.perf_counter() - step_start)
        losses.append(float(loss.item()))
        accs.append(float(acc))

    elapsed = time.perf_counter() - start_time
    val_loss, val_acc = evaluate(
        model=model,
        vq_model=vq_model,
        val_loader=val_loader,
        device=opt.device,
        max_batches=opt.trial_eval_batches,
        eval_seed=trial_seed + 1,
    )

    max_memory_gb = 0.0
    if torch.cuda.is_available():
        max_memory_gb = torch.cuda.max_memory_reserved(opt.device) / (1024 ** 3)

    results = {
        "config": {
            "batch_size": opt.batch_size,
            "lr": opt.lr,
            "dropout": opt.dropout,
            "cond_drop_prob": opt.cond_drop_prob,
            "res_use_uni_mask": bool(getattr(opt, "res_use_uni_mask", False)),
            "res_uni_path_a": float(getattr(opt, "res_uni_path_a", 0.0)),
            "res_uni_path_c": float(getattr(opt, "res_uni_path_c", 0.0)),
            "res_uni_solver_steps": int(getattr(opt, "res_uni_solver_steps", 0)),
        },
        "full_run_schedule": asdict(schedule),
        "trial": {
            "steps": trial_steps,
            "sample_budget": trial_steps * opt.batch_size,
            "warmup_steps": trial_warmup_steps,
            "effective_full_epochs": (trial_steps * opt.batch_size) / float(full_train_size),
        },
        "metrics": {
            "train_loss_initial": losses[0],
            "train_loss_final": losses[-1],
            "train_loss_min": min(losses),
            "train_loss_last5": float(np.mean(losses[-5:])),
            "train_loss_delta": losses[0] - losses[-1],
            "train_acc_initial": accs[0],
            "train_acc_final": accs[-1],
            "train_acc_last5": float(np.mean(accs[-5:])),
            "mini_val_loss": val_loss,
            "mini_val_acc": val_acc,
            "avg_step_time_sec": float(np.mean(step_times)),
            "samples_per_sec": float((trial_steps * opt.batch_size) / elapsed),
            "max_memory_gb": max_memory_gb,
        },
    }
    return results


def main():
    parser = build_parser()
    opt = parser.parse_args()
    opt.is_train = False
    opt.device = torch.device("cpu" if opt.gpu_id == -1 else f"cuda:{opt.gpu_id}")
    if opt.gpu_id != -1:
        torch.cuda.set_device(opt.gpu_id)

    fixseed(opt.seed)
    dim_pose, train_split = configure_dataset(opt)
    val_split = pjoin(opt.data_root, "val.txt")

    mean = np.load(pjoin(opt.checkpoints_dir, opt.dataset_name, opt.vq_name, "meta", "mean.npy"))
    std = np.load(pjoin(opt.checkpoints_dir, opt.dataset_name, opt.vq_name, "meta", "std.npy"))

    print("Loading datasets for short sweep...")
    train_dataset = Text2MotionDataset(opt, mean.copy(), std.copy(), train_split)
    val_dataset = Text2MotionDataset(opt, mean.copy(), std.copy(), val_split)

    train_indices = make_subset_indices(len(train_dataset), opt.train_subset_size, opt.subset_seed)
    val_indices = make_subset_indices(len(val_dataset), opt.val_subset_size, opt.subset_seed + 1)
    train_loader = build_loader(
        dataset=train_dataset,
        indices=train_indices,
        batch_size=opt.batch_size,
        shuffle=True,
        seed=opt.seed,
        num_workers=opt.num_workers,
    )
    val_loader = build_loader(
        dataset=val_dataset,
        indices=val_indices,
        batch_size=opt.val_batch_size,
        shuffle=False,
        seed=opt.seed + 1,
        num_workers=opt.num_workers,
    )

    _ = dim_pose  # Explicitly keep dataset config aligned with training entrypoint.
    results = run_short_trial(
        opt=opt,
        train_loader=train_loader,
        val_loader=val_loader,
        full_train_size=len(train_dataset),
    )

    if opt.save_json:
        save_path = Path(opt.save_json)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(json.dumps(results, indent=2))

    print("SHORT_SWEEP_RESULT " + json.dumps(results, sort_keys=True))


if __name__ == "__main__":
    main()
