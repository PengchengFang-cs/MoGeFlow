"""TokenCacheDataset — reads the offline RVQ token cache produced by
scripts/export_graph_vq_tokens.py for Graph-CodeFlow training.

Each item is one exported clip: the post-RVQ z_q target + graph metadata + dual
text caption tensors. The CodeFlow trainer reads these instead of running the
frozen tokenizer encoder online every step (handoff §5.1).

Padding is along the C (coarse-slot) and T_lat axes and is ALREADY baked into the
export (token_mask/coarse_mask/frame_mask_lat). All exported clips share the same
[T_lat, C_max, D, Q] padded shape (from the frozen tokenizer's max_coarse /
temporal_stride), so the default collate stacks them directly — no ragged collate.
The pooled_geodesic sentinel (export GEO_INF_SENTINEL for +inf) is mapped back to
+inf here so GraphAttentionBlock sees its real unreachable-pair contract.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

GEO_INF_SENTINEL = 30000.0


class TokenCacheDataset(Dataset):
    def __init__(self, cache_dir: str, split: str,
                 geo_inf_sentinel: float = GEO_INF_SENTINEL) -> None:
        self.split_dir = Path(cache_dir) / split
        idx_path = self.split_dir / "index.jsonl"
        if not idx_path.exists():
            raise FileNotFoundError(
                f"TokenCacheDataset: {idx_path} not found (run "
                f"scripts/export_graph_vq_tokens.py first)")
        self.rows = [json.loads(l) for l in idx_path.read_text().splitlines() if l.strip()]
        if not self.rows:
            raise RuntimeError(f"TokenCacheDataset: empty index {idx_path}")
        self.geo_inf_sentinel = float(geo_inf_sentinel)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict:
        row = self.rows[i]
        d = np.load(self.split_dir / row["file"], allow_pickle=False)
        geo = d["pooled_geodesic"].astype(np.float32)
        geo[geo >= self.geo_inf_sentinel] = np.inf
        return {
            "z_q": torch.from_numpy(d["z_q"].astype(np.float32)),           # [T_lat,C,D]
            "indices": torch.from_numpy(d["indices"].astype(np.int64)),     # [T_lat,C,Q]
            "token_mask": torch.from_numpy(d["token_mask"].astype(np.bool_)),
            "coarse_mask": torch.from_numpy(d["coarse_mask"].astype(np.bool_)),
            "frame_mask_lat": torch.from_numpy(d["frame_mask_lat"].astype(np.bool_)),
            "pooled_adjacency": torch.from_numpy(d["pooled_adjacency"].astype(np.float32)),
            "pooled_geodesic": torch.from_numpy(geo),
            "pooled_skeleton_embeddings": torch.from_numpy(
                d["pooled_skeleton_embeddings"].astype(np.float32)),
            "assignment": torch.from_numpy(d["assignment"].astype(np.float32)),  # [J,C]
            "s_j": torch.from_numpy(d["s_j"].astype(np.float32)),               # [J,D]
            "joint_mask": torch.from_numpy(d["joint_mask"].astype(np.bool_)),
            "rest_offsets": torch.from_numpy(d["rest_offsets"].astype(np.float32)),
            "anytop_mean": torch.from_numpy(d["anytop_mean"].astype(np.float32)),
            "anytop_std": torch.from_numpy(d["anytop_std"].astype(np.float32)),
            "parent_indices": [int(p) for p in d["parent_indices"].tolist()],
            "num_joints": int(d["num_joints"]),
            "caption_emb": torch.from_numpy(d["caption_emb"].astype(np.float32)),  # [768]
            "caption_token_emb": torch.from_numpy(
                d["caption_token_emb"].astype(np.float32)),                  # [L,768]
            "caption_token_mask": torch.from_numpy(d["caption_token_mask"].astype(np.bool_)),
            "has_text": bool(d["has_text"]),
            "object_type": row["object_type"],
            "text": row.get("text", ""),
        }


def token_collate(batch: list[dict]) -> dict:
    out: dict = {}
    keys = batch[0].keys()
    for k in keys:
        v0 = batch[0][k]
        if isinstance(v0, torch.Tensor):
            out[k] = torch.stack([b[k] for b in batch])
        elif isinstance(v0, bool):
            out[k] = torch.tensor([b[k] for b in batch], dtype=torch.bool)
        elif isinstance(v0, int):
            out[k] = torch.tensor([b[k] for b in batch], dtype=torch.int64)
        else:
            out[k] = [b[k] for b in batch]
    return out
