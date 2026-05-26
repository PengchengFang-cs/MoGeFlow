"""Smoke checks for the text-only motion code-flow implementation."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.codeflow import MotionCodeFlow, MotionCodeFlowConfig


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kv-root", type=str, default=".")
    parser.add_argument("--data-root", type=str, default="dataset/HumanML3D")
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--mean-path", type=str, default="")
    parser.add_argument("--std-path", type=str, default="")
    parser.add_argument("--contract-check", action="store_true")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--representation", type=str, default="t_concat", choices=["t_concat"])
    parser.add_argument("--coupling-mode", type=str, default="holder_query", choices=["holder_query"])
    parser.add_argument("--holder-depth", type=int, default=2)
    parser.add_argument("--holder-mlp-ratio", type=float, default=4.0)
    parser.add_argument("--terminal-mode", type=str, default="tied_logits", choices=["nearest", "tied_logits", "learned_head"])
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--latent-len", type=int, default=49)
    parser.add_argument("--hidden-size", type=int, default=384)
    parser.add_argument("--num-heads", type=int, default=6)
    parser.add_argument("--depth-double", type=int, default=2)
    parser.add_argument("--depth-single", type=int, default=2)
    parser.add_argument("--steps", type=int, default=2)
    return parser.parse_args()


def load_one_real_motion(args, kv_root: Path):
    data_root = Path(args.data_root).expanduser().resolve()
    mean_path = Path(args.mean_path).expanduser().resolve() if args.mean_path else (
        kv_root / "checkpoints" / "stats" / "mean.npy"
    )
    std_path = Path(args.std_path).expanduser().resolve() if args.std_path else (
        kv_root / "checkpoints" / "stats" / "std.npy"
    )
    mean = np.load(mean_path).astype(np.float32)
    std = np.load(std_path).astype(np.float32)
    if mean.shape != (263,) or std.shape != (263,):
        raise RuntimeError(f"Expected stats shape (263,), got mean={mean.shape} std={std.shape}")
    split_path = data_root / f"{args.split}.txt"
    motion_dir = data_root / "new_joint_vecs"
    with split_path.open("r", encoding="utf-8") as f:
        names = [line.strip() for line in f if line.strip()]
    for name in names:
        path = motion_dir / f"{name}.npy"
        if not path.is_file():
            continue
        motion = np.load(path)
        if motion.ndim != 2 or motion.shape[1] != 263 or len(motion) < 4:
            continue
        length = min(len(motion), 196)
        motion = motion[:196].astype(np.float32, copy=False)
        if len(motion) < 196:
            pad = np.zeros((196 - len(motion), 263), dtype=np.float32)
            motion = np.concatenate([motion, pad], axis=0)
        motion = (motion - mean) / std
        return torch.from_numpy(motion[None]), torch.tensor([length], dtype=torch.long), name
    raise RuntimeError(f"No usable motion found in {split_path}")


def main():
    args = parse_args()
    kv_root = Path(args.kv_root).expanduser().resolve()
    config = MotionCodeFlowConfig(
        kv_root=str(kv_root),
        vq_checkpoint=str(kv_root / "checkpoints" / "vqvae" / "net_best_fid.pth"),
        vq_partition=str(kv_root / "checkpoints" / "vqvae" / "skeleton_partition.json"),
        clip_path=str(kv_root / "checkpoints" / "clip" / "ViT-B-32.pt"),
        representation=args.representation,
        coupling_mode=args.coupling_mode,
        holder_depth=args.holder_depth,
        holder_mlp_ratio=args.holder_mlp_ratio,
        terminal_mode=args.terminal_mode,
        hidden_size=args.hidden_size,
        num_heads=args.num_heads,
        depth_double=args.depth_double,
        depth_single=args.depth_single,
        time_patch=1,
    )
    device = torch.device(args.device)
    model = MotionCodeFlow(config).to(device)
    model.train()

    if args.contract_check:
        real_motion, real_lengths, motion_name = load_one_real_motion(args, kv_root)
        summary = model.tokenizer.verify_contract(
            motion=real_motion.to(device),
            lengths=real_lengths.to(device),
            max_samples=1,
        )
        summary["motion_id"] = motion_name
        print("contract", json.dumps(summary, sort_keys=True))

    ids = torch.randint(
        low=0,
        high=config.num_codes,
        size=(args.batch_size, args.latent_len, config.num_parts),
        device=device,
    )
    embeddings = model.tokenizer.ids_to_embeddings(ids)
    texts = [
        "a person walks forward and turns around",
        "a person jumps and then stands still",
    ][: args.batch_size]
    while len(texts) < args.batch_size:
        texts.append(texts[-1])
    token_lengths = torch.full((args.batch_size,), args.latent_len, device=device, dtype=torch.long)

    losses = model.compute_losses(embeddings, ids, texts, token_lengths)
    loss_value = float(losses["loss"].detach().cpu())
    if not torch.isfinite(losses["loss"]):
        raise RuntimeError(f"Non-finite smoke loss: {loss_value}")
    print("loss", loss_value)

    model.eval()
    gen_ids = model.generate_ids(texts, token_lengths, steps=args.steps, cond_scale=1.0)
    if gen_ids.shape != ids.shape:
        raise RuntimeError(f"Generated ids shape mismatch: {tuple(gen_ids.shape)} vs {tuple(ids.shape)}")
    motion = model.tokenizer.decode_ids(gen_ids)
    print("generated_ids", tuple(gen_ids.shape))
    print("decoded_motion", tuple(motion.shape))


if __name__ == "__main__":
    main()
