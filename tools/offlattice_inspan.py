#!/usr/bin/env python3
"""J6: is the terminal off-lattice offset a blend of adjacent prototypes?

The paper reads the terminal state's distance from its nearest code as
"sub-code precision", and explains direct decoding by saying a state between
lattice points decodes as a blend of ADJACENT PROTOTYPES.  That explanation
requires the offset to lie in the subspace spanned by the local code
differences.  If instead the offset points in a generic direction, then by the
paper's own norm-matched control the decoder is relatively insensitive to it,
and the offset is better read as regression residual.

For every valid frame-group of a sampled terminal state this tool computes

    k*    = argmin_k || y - e_k ||
    delta = y - e_{k*}
    Q     = orthonormal basis of span{ e_{k_j} - e_{k*} },  k_j the m nearest
    f     = || Q^T delta ||^2 / || delta ||^2        (in-span energy fraction)

and compares f against two baselines on the SAME subspaces: the analytic
expectation for an isotropic direction (rank/dim) and an empirical
norm-matched random control.  f >> baseline supports the blend reading;
f ~ baseline does not.

Sampling reuses the loop in offlattice_delta.py; no model files are modified.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = Path(__file__).resolve().parent
for path in (str(REPO_ROOT), str(TOOLS_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from models.codeflow import PartStructuredMotionCodeFlow  # noqa: E402
from models.codeflow.eval_t2m_cli import build_eval_loader, load_codeflow_model, make_device  # noqa: E402
from motion_code_geometry_diagnostics_formal import FORMAL_PART_NAMES  # noqa: E402
from utils.fixseed import fixseed  # noqa: E402

DEFAULT_CHECKPOINT = str(
    REPO_ROOT
    / "checkpoints/t2m/hml3d_20260807_nopool_gate/model/gate/gate_ep0300_fid0.0750_top30.8707.pt"
)
DEFAULT_OUT = str(REPO_ROOT / "eval_results/diag_20260831/offlattice_inspan.json")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)
    p.add_argument("--out", type=str, default=DEFAULT_OUT)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--max_prompts", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--steps", type=int, default=96)
    p.add_argument("--cond_scale", type=float, default=6.0)
    p.add_argument("--no_ema", action="store_true")
    p.add_argument("--device", type=str, default="")
    p.add_argument("--gpu_id", type=int, default=-1)
    p.add_argument("--neighbors", type=int, default=8,
                   help="m: local code differences spanning the comparison subspace.")
    p.add_argument("--dataset_opt_path", type=str, default="./checkpoints/t2m/Comp_v6_KLD005/opt.txt")
    p.add_argument("--checkpoints_dir", type=str, default="./checkpoints")
    p.add_argument("--glove_dir", type=str, default="./glove")
    p.add_argument("--data_root", type=str, default="")
    p.add_argument("--kv_root", type=str, default="")
    p.add_argument("--vq_checkpoint", type=str, default="")
    p.add_argument("--vq_partition", type=str, default="")
    p.add_argument("--mean_path", type=str, default="")
    p.add_argument("--std_path", type=str, default="")
    p.add_argument("--clip_path", type=str, default="")
    p.add_argument("--text_cache_path", type=str, default="")
    p.add_argument("--unit_length", type=int, default=0)
    return p.parse_args()


def local_bases(book: torch.Tensor, m: int):
    """For each code, an orthonormal basis of the span of its m nearest code differences.

    Returns Q [K, d, m] (zero-padded columns where the local rank is short) and
    the integer rank per code.
    """
    K, d = book.shape
    dist = torch.cdist(book, book, p=2.0)
    dist.fill_diagonal_(float("inf"))
    nn = dist.topk(m, dim=1, largest=False).indices          # [K, m]
    diffs = book[nn] - book[:, None, :]                       # [K, m, d]
    Q, ranks = torch.zeros(K, d, m, device=book.device, dtype=book.dtype), []
    for k in range(K):
        u, s, _ = torch.linalg.svd(diffs[k].T, full_matrices=False)   # [d, m]
        r = int((s > s.max() * 1e-6).sum().item()) if s.numel() else 0
        if r:
            Q[k, :, :r] = u[:, :r]
        ranks.append(r)
    return Q, torch.tensor(ranks, device=book.device)


def part_group_names(tokenizer, num_parts: int) -> List[str]:
    partition = getattr(tokenizer, "partition", None) or {}
    names = partition.get("part_names")
    if isinstance(names, list) and len(names) == num_parts:
        return [str(n) for n in names]
    if num_parts == len(FORMAL_PART_NAMES):
        return list(FORMAL_PART_NAMES)
    return [f"part_{i}" for i in range(num_parts)]


def main() -> None:
    args = parse_args()
    device = make_device(args)
    ckpt = Path(args.checkpoint).expanduser().resolve()
    if not ckpt.is_file():
        raise FileNotFoundError(f"checkpoint not found: {ckpt}")

    model, opt, _c, weight_source = load_codeflow_model(
        ckpt, args, device, model_cls=PartStructuredMotionCodeFlow
    )
    if str(getattr(opt, "vq_backend", "kv_part")) != "kv_part":
        raise ValueError(f"expects a kv_part checkpoint, got vq_backend={opt.vq_backend!r}")

    loader, dataset, _eo = build_eval_loader(args, opt, device)
    unit_length = int(getattr(opt, "unit_length", 4))
    num_parts = int(model.config.num_parts)
    names = part_group_names(model.tokenizer, num_parts)
    codebooks = model.tokenizer.codebooks.detach().float().to(device)
    dim = codebooks.shape[-1]

    bases, ranks = [], []
    for p in range(num_parts):
        Q, r = local_bases(codebooks[p], args.neighbors)
        bases.append(Q)
        ranks.append(r)
    print(f"[bases] built {num_parts} x {codebooks.shape[1]} local bases, m={args.neighbors}", flush=True)

    fixseed(args.seed)
    model.eval()
    gen = torch.Generator(device=device).manual_seed(args.seed)
    acc: List[Dict[str, List[np.ndarray]]] = [
        {"frac": [], "rand": [], "rank": []} for _ in range(num_parts)
    ]
    num_prompts = 0
    with torch.no_grad():
        for batch in loader:
            _w, _pp, captions, _sl, _pose, m_length, _tok = batch
            m_length = m_length.to(device).long()
            token_lengths = (m_length // unit_length).clamp(min=1)
            terminal = model.sample_embeddings(
                list(captions), token_lengths=token_lengths,
                steps=int(args.steps), cond_scale=float(args.cond_scale),
            )
            latent_len = terminal.shape[1]
            valid = (torch.arange(latent_len, device=device)[None, :]
                     < token_lengths.clamp(max=latent_len)[:, None])
            for p in range(num_parts):
                y = terminal[:, :, p, :].float()[valid]
                if y.numel() == 0:
                    continue
                book = codebooks[p]
                kstar = torch.cdist(y, book, p=2.0).argmin(dim=1)
                delta = y - book[kstar]
                nrm2 = (delta * delta).sum(-1).clamp_min(1e-12)
                rnd = torch.randn(delta.shape, device=device, generator=gen)
                rnd = rnd / rnd.norm(dim=-1, keepdim=True).clamp_min(1e-12) * delta.norm(dim=-1, keepdim=True)
                Qg = bases[p][kstar]                                  # [N, d, m]
                f = torch.einsum("nd,ndr->nr", delta, Qg).pow(2).sum(-1) / nrm2
                fr = torch.einsum("nd,ndr->nr", rnd, Qg).pow(2).sum(-1) / nrm2
                acc[p]["frac"].append(f.cpu().numpy().astype(np.float32))
                acc[p]["rand"].append(fr.cpu().numpy().astype(np.float32))
                acc[p]["rank"].append(ranks[p][kstar].cpu().numpy().astype(np.float32))
            num_prompts += len(captions)
            print(f"[sample] {num_prompts} prompts", flush=True)
            if args.max_prompts > 0 and num_prompts >= args.max_prompts:
                break

    groups, pooled_f, pooled_r = {}, [], []
    for p in range(num_parts):
        if not acc[p]["frac"]:
            continue
        f = np.concatenate(acc[p]["frac"])
        fr = np.concatenate(acc[p]["rand"])
        rk = np.concatenate(acc[p]["rank"])
        groups[names[p]] = {
            "part": p, "count": int(f.size),
            "mean_local_rank": float(rk.mean()),
            "analytic_baseline": float(rk.mean() / dim),
            "inspan_fraction_mean": float(f.mean()),
            "inspan_fraction_median": float(np.median(f)),
            "random_control_mean": float(fr.mean()),
            "ratio_to_random": float(f.mean() / max(fr.mean(), 1e-12)),
        }
        pooled_f.append(f)
        pooled_r.append(fr)

    f_all = np.concatenate(pooled_f)
    r_all = np.concatenate(pooled_r)
    payload = {
        "checkpoint": str(ckpt), "weight_source": weight_source, "split": "test",
        "num_prompts": int(num_prompts), "steps": int(args.steps),
        "cond_scale": float(args.cond_scale), "neighbors": int(args.neighbors),
        "code_dim": int(dim),
        "overall": {
            "count": int(f_all.size),
            "inspan_fraction_mean": float(f_all.mean()),
            "inspan_fraction_median": float(np.median(f_all)),
            "random_control_mean": float(r_all.mean()),
            "ratio_to_random": float(f_all.mean() / max(r_all.mean(), 1e-12)),
        },
        "groups": groups,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\n{:20s} {:>8s} {:>10s} {:>10s} {:>8s}".format("group", "rank", "in-span", "random", "ratio"))
    print("-" * 62)
    for name, g in groups.items():
        print("{:20s} {:8.1f} {:10.4f} {:10.4f} {:8.2f}".format(
            name, g["mean_local_rank"], g["inspan_fraction_mean"],
            g["random_control_mean"], g["ratio_to_random"]))
    print("-" * 62)
    o = payload["overall"]
    print("{:20s} {:8s} {:10.4f} {:10.4f} {:8.2f}".format(
        "OVERALL", "", o["inspan_fraction_mean"], o["random_control_mean"], o["ratio_to_random"]))
    print(f"\n[done] wrote {out}")


if __name__ == "__main__":
    main()
