"""Smoke checks for the continuous-decode CodeFlow variant."""

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.codeflow import ContinuousMotionCodeFlow, MotionCodeFlowConfig


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kv-root", type=str, default=".")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--latent-len", type=int, default=49)
    parser.add_argument("--hidden-size", type=int, default=384)
    parser.add_argument("--num-heads", type=int, default=6)
    parser.add_argument("--depth-double", type=int, default=2)
    parser.add_argument("--depth-single", type=int, default=2)
    parser.add_argument("--holder-depth", type=int, default=2)
    parser.add_argument("--steps", type=int, default=2)
    return parser.parse_args()


def main():
    args = parse_args()
    kv_root = Path(args.kv_root).expanduser().resolve()
    config = MotionCodeFlowConfig(
        kv_root=str(kv_root),
        vq_checkpoint=str(kv_root / "checkpoints" / "vqvae" / "net_best_fid.pth"),
        vq_partition=str(kv_root / "checkpoints" / "vqvae" / "skeleton_partition.json"),
        clip_path=str(kv_root / "checkpoints" / "clip" / "ViT-B-32.pt"),
        hidden_size=args.hidden_size,
        num_heads=args.num_heads,
        depth_double=args.depth_double,
        depth_single=args.depth_single,
        holder_depth=args.holder_depth,
        time_patch=1,
        coupling_mode="holder_query",
        flow_loss_weight=1.0,
        terminal_loss_weight=0.0,
        clean_loss_weight=1.0,
    )
    device = torch.device(args.device)
    model = ContinuousMotionCodeFlow(config).to(device)
    model.train()

    ids = torch.randint(
        low=0,
        high=config.num_codes,
        size=(args.batch_size, args.latent_len, config.num_parts),
        device=device,
    )
    embeddings = model.tokenizer.ids_to_embeddings(ids)
    decoded_gt = model.tokenizer.decode_embeddings(embeddings)
    if decoded_gt.shape[0] != args.batch_size or decoded_gt.shape[-1] != 263:
        raise RuntimeError(f"Continuous decoder shape mismatch: {tuple(decoded_gt.shape)}")

    texts = [
        "a person walks forward and turns around",
        "a person jumps and then stands still",
    ][: args.batch_size]
    while len(texts) < args.batch_size:
        texts.append(texts[-1])
    token_lengths = torch.full((args.batch_size,), args.latent_len, device=device, dtype=torch.long)

    losses = model.compute_losses(embeddings, ids, texts, token_lengths)
    loss = losses["loss"]
    if not torch.isfinite(loss):
        raise RuntimeError(f"Non-finite smoke loss: {float(loss.detach().cpu())}")
    loss.backward()
    grad_ok = any(param.grad is not None and torch.isfinite(param.grad).all() for param in model.trainable_parameters())
    if not grad_ok:
        raise RuntimeError("Continuous smoke backward produced no finite trainable gradients")
    print("loss", float(loss.detach().cpu()))

    model.eval()
    motion, nearest_ids = model.generate_motion(texts, token_lengths, steps=args.steps, cond_scale=1.0)
    if nearest_ids.shape != ids.shape:
        raise RuntimeError(f"Generated nearest ids shape mismatch: {tuple(nearest_ids.shape)} vs {tuple(ids.shape)}")
    print("nearest_ids", tuple(nearest_ids.shape))
    print("decoded_motion", tuple(motion.shape))


if __name__ == "__main__":
    main()
