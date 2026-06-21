"""Online token promotion for masked-token checkpoint sampling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from sampling.config import OnlineTokenPromotionConfig


@dataclass
class TokenPromotionStats:
    promotion_fraction: float
    promoted_counts: list[int]


def _top_positions(values: torch.Tensor, count: int) -> torch.Tensor:
    if count <= 0:
        return torch.empty((0,), device=values.device, dtype=torch.long)
    return torch.argsort(values, descending=True)[:count]


def _split_promote_count(promote_now: int, eos_count: int, non_eos_count: int) -> tuple[int, int]:
    total = max(1, int(eos_count) + int(non_eos_count))
    eos_promote = min(int(round(float(promote_now) * float(eos_count) / float(total))), int(eos_count))
    non_eos_promote = min(max(0, int(promote_now) - eos_promote), int(non_eos_count))
    leftover = max(0, int(promote_now) - eos_promote - non_eos_promote)
    if leftover > 0:
        add_non_eos = min(leftover, int(non_eos_count) - non_eos_promote)
        non_eos_promote += add_non_eos
        leftover -= add_non_eos
    if leftover > 0:
        eos_promote += min(leftover, int(eos_count) - eos_promote)
    return eos_promote, non_eos_promote


def select_online_token_promotion(
    remaining_mask: torch.Tensor,
    confidence: torch.Tensor,
    candidate_ids: torch.Tensor,
    config: OnlineTokenPromotionConfig,
    generator: Optional[torch.Generator] = None,
    eos_token_id: Optional[int] = None,
) -> tuple[torch.Tensor, TokenPromotionStats]:
    remaining_mask = remaining_mask.bool()
    promotion_mask = torch.zeros_like(remaining_mask, dtype=torch.bool)
    if not bool(remaining_mask.any().item()):
        return promotion_mask, TokenPromotionStats(0.0, [0 for _ in range(remaining_mask.shape[0])])

    eos_policy = str(config.eos_policy or "rank_together").lower()
    if eos_policy in {"rank_separatly", "rank_seperately", "rank_separate"}:
        eos_policy = "rank_separately"
    threshold = float(config.confidence_threshold)
    min_promote = max(0, int(config.min_promote_tokens or 0))
    scores = confidence.float()
    if eos_policy == "zero_confidence" and eos_token_id is not None:
        scores = torch.where(candidate_ids == int(eos_token_id), torch.zeros_like(scores), scores)

    promoted_counts = []
    for row_idx in range(remaining_mask.shape[0]):
        positions = torch.nonzero(remaining_mask[row_idx], as_tuple=False).flatten()
        remaining_count = int(positions.numel())
        if remaining_count <= 0:
            promoted_counts.append(0)
            continue

        row_scores = scores[row_idx, positions]
        chosen = positions[row_scores >= row_scores.new_tensor(threshold)]
        required = min(min_promote, remaining_count)
        if int(chosen.numel()) == 0:
            required = max(1, required)

        if int(chosen.numel()) < required:
            row_candidates = candidate_ids[row_idx, positions]
            if eos_policy in {"promote_all", "rank_separately"} and eos_token_id is not None:
                eos_local = row_candidates == int(eos_token_id)
                eos_positions = positions[eos_local]
                non_eos_positions = positions[~eos_local]
                eos_promote, non_eos_promote = _split_promote_count(
                    required,
                    int(eos_positions.numel()),
                    int(non_eos_positions.numel()),
                )
                extra = []
                if non_eos_promote > 0:
                    order = _top_positions(scores[row_idx, non_eos_positions], non_eos_promote)
                    extra.append(non_eos_positions[order])
                if eos_promote > 0:
                    if eos_policy == "rank_separately":
                        order = _top_positions(scores[row_idx, eos_positions], eos_promote)
                    else:
                        order = torch.randperm(
                            int(eos_positions.numel()),
                            device=positions.device,
                            generator=generator,
                        )[:eos_promote]
                    extra.append(eos_positions[order])
                if extra:
                    chosen = torch.unique(torch.cat([chosen, *extra]))
            else:
                order = _top_positions(row_scores, required)
                chosen = torch.unique(torch.cat([chosen, positions[order]]))

        promotion_mask[row_idx, chosen] = True
        promoted_counts.append(int(promotion_mask[row_idx].sum().item()))

    promoted_total = int(sum(promoted_counts))
    remaining_total = int(remaining_mask.sum().item())
    fraction = float(promoted_total) / float(max(1, remaining_total))
    return promotion_mask, TokenPromotionStats(fraction, promoted_counts)
