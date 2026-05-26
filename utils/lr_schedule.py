from typing import Iterable, List


def _normalize_milestones(milestones: Iterable) -> List[float]:
    values = []
    for milestone in milestones:
        if isinstance(milestone, str):
            milestone = milestone.strip()
            if not milestone:
                continue
        values.append(float(milestone))
    return values


def _resolve_single_milestone(
    milestone: float,
    milestone_unit: str,
    *,
    total_iters: int,
    iters_per_epoch: int,
    max_epoch: int,
) -> int:
    if milestone_unit == "iter":
        return int(round(milestone))

    if milestone_unit == "epoch":
        return int(round(milestone)) * iters_per_epoch

    if milestone_unit == "epoch_ratio":
        if not (0.0 < milestone <= 1.0):
            raise ValueError(f"epoch_ratio milestones must be in (0, 1], got {milestone}")
        return int(round(max_epoch * milestone)) * iters_per_epoch

    if milestone_unit != "auto":
        raise ValueError(f"Unsupported milestone_unit: {milestone_unit}")

    if 0.0 < milestone <= 1.0:
        return int(round(max_epoch * milestone)) * iters_per_epoch

    if float(milestone).is_integer() and 0 < int(milestone) <= max_epoch:
        return int(milestone) * iters_per_epoch

    return int(round(milestone))


def resolve_multistep_milestones(
    milestones: Iterable,
    milestone_unit: str,
    *,
    total_iters: int,
    iters_per_epoch: int,
    max_epoch: int,
) -> List[int]:
    if total_iters <= 0:
        raise ValueError(f"total_iters must be > 0, got {total_iters}")
    if iters_per_epoch <= 0:
        raise ValueError(f"iters_per_epoch must be > 0, got {iters_per_epoch}")
    if max_epoch <= 0:
        raise ValueError(f"max_epoch must be > 0, got {max_epoch}")

    normalized = _normalize_milestones(milestones)
    resolved = [
        max(
            1,
            _resolve_single_milestone(
                milestone,
                milestone_unit,
                total_iters=total_iters,
                iters_per_epoch=iters_per_epoch,
                max_epoch=max_epoch,
            ),
        )
        for milestone in normalized
    ]
    return sorted(set(resolved))
