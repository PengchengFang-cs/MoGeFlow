"""HumanML3D full-evaluation loop for CodeFlow text-to-motion generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch

from utils.metrics import (
    calculate_R_precision,
    calculate_activation_statistics,
    calculate_diversity,
    calculate_frechet_distance,
    calculate_matching_score,
    calculate_multimodality,
)


@dataclass
class CodeFlowEvalConfig:
    steps: int = 32
    cond_scale: float = 3.0
    cfg_t_lo: float = 0.0
    cfg_t_hi: float = 1.0
    terminal_mode: Optional[str] = None
    unit_length: int = 4
    max_batches: int = 0
    cal_mm: bool = True
    mm_num_batches: int = 3
    mm_num_samples: int = 30
    multimodality_times: int = 10
    allow_small_eval: bool = False
    include_code_metrics: bool = True
    geometry_severe_quantile: float = 0.75


def _norm_to_tensor(value, device: torch.device, dtype: torch.dtype) -> Optional[torch.Tensor]:
    if value is None:
        return None
    if not torch.is_tensor(value):
        value = torch.as_tensor(value)
    value = value.to(device=device, dtype=dtype)
    if value.dim() == 1:
        value = value.view(1, 1, -1)
    return value


def _zero_pad_motion(motion: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    lengths = torch.as_tensor(lengths, device=motion.device, dtype=torch.long)
    frame_ids = torch.arange(motion.shape[1], device=motion.device).view(1, -1, 1)
    valid = frame_ids < lengths.view(-1, 1, 1)
    return torch.where(valid, motion, torch.zeros_like(motion))


def prepare_codeflow_motion_for_eval(
    decoded_motion: torch.Tensor,
    reference_motion: torch.Tensor,
    lengths: torch.Tensor,
    vq_mean: torch.Tensor,
    vq_std: torch.Tensor,
    eval_mean: torch.Tensor,
    eval_std: torch.Tensor,
) -> torch.Tensor:
    """Convert CodeFlow decoded motion into the evaluator-normalized feature space."""
    bsz, seq_len, feat_dim = reference_motion.shape
    if decoded_motion.shape[0] != bsz or decoded_motion.shape[-1] != feat_dim:
        raise ValueError(
            f"Decoded motion shape {tuple(decoded_motion.shape)} is incompatible with "
            f"reference shape {tuple(reference_motion.shape)}"
        )
    padded = reference_motion.new_zeros((bsz, seq_len, feat_dim))
    copy_len = min(seq_len, decoded_motion.shape[1])
    padded[:, :copy_len] = decoded_motion[:, :copy_len].to(reference_motion.dtype)

    vq_mean = _norm_to_tensor(vq_mean, padded.device, padded.dtype)
    vq_std = _norm_to_tensor(vq_std, padded.device, padded.dtype)
    eval_mean = _norm_to_tensor(eval_mean, padded.device, padded.dtype)
    eval_std = _norm_to_tensor(eval_std, padded.device, padded.dtype)
    if vq_mean is None or vq_std is None or eval_mean is None or eval_std is None:
        raise RuntimeError("Missing VQ/evaluator normalization stats for full evaluation")

    raw_motion = padded * vq_std + vq_mean
    eval_motion = (raw_motion - eval_mean) / eval_std
    return _zero_pad_motion(eval_motion, lengths)


@torch.no_grad()
def eval_motion_to_vq_space(
    eval_motion: torch.Tensor,
    lengths: torch.Tensor,
    vq_mean: torch.Tensor,
    vq_std: torch.Tensor,
    eval_mean: torch.Tensor,
    eval_std: torch.Tensor,
) -> torch.Tensor:
    """Convert evaluator-normalized motion features back to tokenizer-normalized features."""
    vq_mean = _norm_to_tensor(vq_mean, eval_motion.device, eval_motion.dtype)
    vq_std = _norm_to_tensor(vq_std, eval_motion.device, eval_motion.dtype)
    eval_mean = _norm_to_tensor(eval_mean, eval_motion.device, eval_motion.dtype)
    eval_std = _norm_to_tensor(eval_std, eval_motion.device, eval_motion.dtype)
    raw_motion = eval_motion * eval_std + eval_mean
    vq_motion = (raw_motion - vq_mean) / vq_std
    return _zero_pad_motion(vq_motion, lengths)


@torch.no_grad()
def generate_eval_motion_and_ids(
    model,
    captions: Iterable[str],
    frame_lengths: torch.Tensor,
    reference_motion: torch.Tensor,
    vq_mean: torch.Tensor,
    vq_std: torch.Tensor,
    eval_mean: torch.Tensor,
    eval_std: torch.Tensor,
    cfg: CodeFlowEvalConfig,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    token_lengths = (frame_lengths.to(model.device).long() // int(cfg.unit_length)).clamp(min=1)
    decoded_motion, _ids = model.generate_motion(
        list(captions),
        token_lengths=token_lengths,
        steps=int(cfg.steps),
        cond_scale=float(cfg.cond_scale),
        terminal_mode=cfg.terminal_mode,
        cfg_t_lo=float(cfg.cfg_t_lo),
        cfg_t_hi=float(cfg.cfg_t_hi),
    )
    eval_motion = prepare_codeflow_motion_for_eval(
        decoded_motion=decoded_motion,
        reference_motion=reference_motion,
        lengths=frame_lengths,
        vq_mean=vq_mean,
        vq_std=vq_std,
        eval_mean=eval_mean,
        eval_std=eval_std,
    )
    return eval_motion, _ids, token_lengths


@torch.no_grad()
def generate_eval_motion(
    model,
    captions: Iterable[str],
    frame_lengths: torch.Tensor,
    reference_motion: torch.Tensor,
    vq_mean: torch.Tensor,
    vq_std: torch.Tensor,
    eval_mean: torch.Tensor,
    eval_std: torch.Tensor,
    cfg: CodeFlowEvalConfig,
) -> torch.Tensor:
    eval_motion, _ids, _token_lengths = generate_eval_motion_and_ids(
        model,
        captions,
        frame_lengths,
        reference_motion,
        vq_mean,
        vq_std,
        eval_mean,
        eval_std,
        cfg,
    )
    return eval_motion


@torch.no_grad()
def _accumulate_code_metrics(
    sums: Dict[str, float],
    tokenizer,
    target_ids: torch.Tensor,
    pred_ids: torch.Tensor,
    token_lengths: torch.Tensor,
    cfg: CodeFlowEvalConfig,
) -> None:
    common_len = min(target_ids.shape[1], pred_ids.shape[1])
    target_ids = target_ids[:, :common_len].long()
    pred_ids = pred_ids[:, :common_len].long()
    token_lengths = token_lengths.to(target_ids.device).long().clamp(min=1, max=common_len)
    valid = torch.arange(common_len, device=target_ids.device)[None, :] < token_lengths[:, None]
    valid_parts = valid[:, :, None].expand_as(target_ids)
    valid_count = valid_parts.sum().float().clamp_min(1.0)

    correct = ((pred_ids == target_ids) & valid_parts).sum().float()
    wrong = (pred_ids != target_ids) & valid_parts
    wrong_count = wrong.sum().float()
    code_dist, rank_pct = tokenizer.code_id_distances(target_ids, pred_ids)
    severe = wrong & (rank_pct >= float(cfg.geometry_severe_quantile))

    sums["code_valid_parts"] += float(valid_count.detach().cpu())
    sums["code_correct_parts"] += float(correct.detach().cpu())
    sums["code_wrong_parts"] += float(wrong_count.detach().cpu())
    sums["geom_code_dist_sum"] += float((code_dist * valid_parts.to(code_dist.dtype)).sum().detach().cpu())
    sums["geom_rank_pct_sum"] += float((rank_pct * valid_parts.to(rank_pct.dtype)).sum().detach().cpu())
    sums["geom_wrong_code_dist_sum"] += float((code_dist * wrong.to(code_dist.dtype)).sum().detach().cpu())
    sums["geom_wrong_rank_pct_sum"] += float((rank_pct * wrong.to(rank_pct.dtype)).sum().detach().cpu())
    sums["geom_severe_parts"] += float(severe.sum().detach().cpu())


def _finalize_code_metrics(code_sums: Dict[str, float]) -> Dict[str, float]:
    valid_count = max(code_sums["code_valid_parts"], 1.0)
    wrong_count = max(code_sums["code_wrong_parts"], 1.0)
    return {
        "code_token_acc": code_sums["code_correct_parts"] / valid_count,
        "code_wrong_rate": code_sums["code_wrong_parts"] / valid_count,
        "geom_code_dist": code_sums["geom_code_dist_sum"] / valid_count,
        "geom_rank_pct": code_sums["geom_rank_pct_sum"] / valid_count,
        "geom_wrong_code_dist": code_sums["geom_wrong_code_dist_sum"] / wrong_count,
        "geom_wrong_rank_pct": code_sums["geom_wrong_rank_pct_sum"] / wrong_count,
        "geom_severe_rate": code_sums["geom_severe_parts"] / valid_count,
        "geom_wrong_severe_frac": code_sums["geom_severe_parts"] / wrong_count,
    }


def _finalize_metrics(
    text_embeddings: List[torch.Tensor],
    real_motion_embeddings: List[torch.Tensor],
    pred_motion_embeddings: List[torch.Tensor],
    multimodality_embeddings: List[torch.Tensor],
    r_precision_real_sum: np.ndarray,
    r_precision_sum: np.ndarray,
    matching_real_sum: float,
    matching_pred_sum: float,
    nb_sample: int,
    cfg: CodeFlowEvalConfig,
) -> Dict[str, object]:
    motion_annotation_np = torch.cat(real_motion_embeddings, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(pred_motion_embeddings, dim=0).cpu().numpy()
    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)

    if nb_sample > 300:
        diversity_times = 300
    elif nb_sample > 100:
        diversity_times = 100
    elif cfg.allow_small_eval and nb_sample > 1:
        diversity_times = nb_sample - 1
    else:
        raise RuntimeError(
            f"Full eval needs more than 100 samples for diversity; got nb_sample={nb_sample}. "
            "Use a larger split/batch budget, or pass allow_small_eval only for smoke tests."
        )
    diversity_real = calculate_diversity(motion_annotation_np, diversity_times)
    diversity = calculate_diversity(motion_pred_np, diversity_times)
    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    multimodality = 0.0
    if cfg.cal_mm and multimodality_embeddings:
        mm_np = torch.cat(multimodality_embeddings, dim=0).cpu().numpy()
        if mm_np.shape[1] <= int(cfg.multimodality_times):
            raise RuntimeError(
                f"multimodality needs more samples per caption than multimodality_times; "
                f"got mm_num_samples={mm_np.shape[1]} and multimodality_times={cfg.multimodality_times}"
            )
        multimodality = calculate_multimodality(mm_np, int(cfg.multimodality_times))

    text_np = torch.cat(text_embeddings, dim=0).cpu().numpy()
    real_np = motion_annotation_np
    pred_np = motion_pred_np
    return {
        "fid": float(fid),
        "diversity_real": float(diversity_real),
        "diversity": float(diversity),
        "r_precision_real": (r_precision_real_sum / float(nb_sample)).astype(float).tolist(),
        "r_precision": (r_precision_sum / float(nb_sample)).astype(float).tolist(),
        "top1": float(r_precision_sum[0] / float(nb_sample)),
        "top2": float(r_precision_sum[1] / float(nb_sample)),
        "top3": float(r_precision_sum[2] / float(nb_sample)),
        "matching_score_real": float(matching_real_sum / float(nb_sample)),
        "matching_score": float(matching_pred_sum / float(nb_sample)),
        "multimodality": float(multimodality),
        "nb_sample": int(nb_sample),
        "embedding_shapes": {
            "text": list(text_np.shape),
            "motion_real": list(real_np.shape),
            "motion_pred": list(pred_np.shape),
        },
    }


@torch.no_grad()
def evaluate_codeflow_t2m(
    loader,
    model,
    eval_wrapper,
    vq_mean: torch.Tensor,
    vq_std: torch.Tensor,
    eval_mean: torch.Tensor,
    eval_std: torch.Tensor,
    cfg: CodeFlowEvalConfig,
    repeat_id: int = 0,
) -> Dict[str, object]:
    model.eval()
    text_embedding_list: List[torch.Tensor] = []
    motion_annotation_list: List[torch.Tensor] = []
    motion_pred_list: List[torch.Tensor] = []
    motion_multimodality: List[torch.Tensor] = []

    r_precision_real = np.zeros(3, dtype=np.float64)
    r_precision = np.zeros(3, dtype=np.float64)
    matching_score_real = 0.0
    matching_score_pred = 0.0
    nb_sample = 0
    code_sums = {
        "code_valid_parts": 0.0,
        "code_correct_parts": 0.0,
        "code_wrong_parts": 0.0,
        "geom_code_dist_sum": 0.0,
        "geom_rank_pct_sum": 0.0,
        "geom_wrong_code_dist_sum": 0.0,
        "geom_wrong_rank_pct_sum": 0.0,
        "geom_severe_parts": 0.0,
    }

    for batch_id, batch in enumerate(loader):
        if cfg.max_batches > 0 and batch_id >= cfg.max_batches:
            break
        word_embeddings, pos_one_hots, captions, sent_len, pose, m_length, _tokens = batch
        pose = pose.to(model.device).float()
        m_length = m_length.to(model.device).long()
        bsz = pose.shape[0]

        if cfg.cal_mm and batch_id < cfg.mm_num_batches:
            mm_batch = []
            for _ in range(int(cfg.mm_num_samples)):
                pred_motion_mm, pred_ids_mm, token_lengths_mm = generate_eval_motion_and_ids(
                    model,
                    captions,
                    m_length,
                    pose,
                    vq_mean,
                    vq_std,
                    eval_mean,
                    eval_std,
                    cfg,
                )
                _et_mm, em_mm = eval_wrapper.get_co_embeddings(
                    word_embeddings,
                    pos_one_hots,
                    sent_len,
                    pred_motion_mm.clone(),
                    m_length,
                )
                mm_batch.append(em_mm.unsqueeze(1))
            motion_multimodality.append(torch.cat(mm_batch, dim=1))
            pred_motion = pred_motion_mm
            pred_ids = pred_ids_mm
            token_lengths = token_lengths_mm
            et_pred, em_pred = _et_mm, em_mm
        else:
            pred_motion, pred_ids, token_lengths = generate_eval_motion_and_ids(
                model,
                captions,
                m_length,
                pose,
                vq_mean,
                vq_std,
                eval_mean,
                eval_std,
                cfg,
            )
            et_pred, em_pred = eval_wrapper.get_co_embeddings(
                word_embeddings,
                pos_one_hots,
                sent_len,
                pred_motion.clone(),
                m_length,
            )

        et, em = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pose, m_length)
        text_embedding_list.append(et)
        motion_annotation_list.append(em)
        motion_pred_list.append(em_pred)

        et_cpu = et.cpu().numpy()
        em_cpu = em.cpu().numpy()
        et_pred_cpu = et_pred.cpu().numpy()
        em_pred_cpu = em_pred.cpu().numpy()
        r_precision_real += calculate_R_precision(et_cpu, em_cpu, top_k=3, sum_all=True)
        r_precision += calculate_R_precision(et_pred_cpu, em_pred_cpu, top_k=3, sum_all=True)
        matching_score_real += float(calculate_matching_score(et_cpu, em_cpu, sum_all=True))
        matching_score_pred += float(calculate_matching_score(et_pred_cpu, em_pred_cpu, sum_all=True))
        nb_sample += bsz

        if cfg.include_code_metrics:
            gt_vq_motion = eval_motion_to_vq_space(pose, m_length, vq_mean, vq_std, eval_mean, eval_std)
            target_ids, _target_embeddings = model.tokenizer.encode(gt_vq_motion)
            _accumulate_code_metrics(code_sums, model.tokenizer, target_ids, pred_ids, token_lengths, cfg)

    if nb_sample == 0:
        raise RuntimeError("Eval loader produced zero samples. Check split path, batch_size, and drop_last settings.")

    metrics = _finalize_metrics(
        text_embedding_list,
        motion_annotation_list,
        motion_pred_list,
        motion_multimodality,
        r_precision_real,
        r_precision,
        matching_score_real,
        matching_score_pred,
        nb_sample,
        cfg,
    )
    if cfg.include_code_metrics:
        metrics.update(_finalize_code_metrics(code_sums))
    print(
        f"--> \t CodeFlow Eval Repeat {repeat_id}: "
        f"FID. {metrics['fid']:.4f}, Diversity Real. {metrics['diversity_real']:.4f}, "
        f"Diversity. {metrics['diversity']:.4f}, R_precision_real. {metrics['r_precision_real']}, "
        f"R_precision. {metrics['r_precision']}, matching_score_real. {metrics['matching_score_real']:.4f}, "
        f"matching_score_pred. {metrics['matching_score']:.4f}, "
        f"multimodality. {metrics['multimodality']:.4f}"
    )
    return metrics
