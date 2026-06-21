"""Token-promotion loop for masked-token sampling."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch

from sampling.model import SamplingModel
from sampling.online_token_promotion import select_online_token_promotion
from sampling.proposals import build_proposal


@dataclass
class SamplingPromotionStep:
    promotion_step: int
    remaining_before: list[int]
    promoted: list[int]
    promotion_fraction: float
    token_ids: Optional[torch.Tensor] = None
    confidence: Optional[torch.Tensor] = None
    promotion_mask: Optional[torch.Tensor] = None


@dataclass
class SamplingResult:
    input_ids: torch.Tensor
    generated_ids: torch.Tensor
    corrupt_mask: torch.Tensor
    texts: list[str]
    history: list[SamplingPromotionStep] = field(default_factory=list)


class OnlineTokenPromotionSampler:
    """Run SDE proposals with online token promotion until all masks are filled."""

    def __init__(self, model: SamplingModel):
        self.model = model
        self.config = model.sampling_config
        self.proposal = build_proposal(self.config)

    def _promote_tokens(
        self,
        current_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        valid_mask: torch.Tensor,
        original_corrupt_mask: torch.Tensor,
        prompt_condition_mask: torch.Tensor,
        promoted_generated_mask: torch.Tensor,
        mask_token: torch.Tensor,
        generator: torch.Generator,
        history: list[SamplingPromotionStep],
        progress_factory=None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        remaining_mask = original_corrupt_mask & ~promoted_generated_mask
        max_promotion_steps = max(1, int(original_corrupt_mask.sum(dim=1).max().item()))
        progress_iter = range(max_promotion_steps)
        if progress_factory is not None:
            progress_iter = progress_factory(progress_iter, desc="token promotion", leave=False)

        for step_idx in progress_iter:
            if not bool(remaining_mask.any().item()):
                break
            proposal_remaining_mask = original_corrupt_mask & ~promoted_generated_mask
            current_ids = torch.where(proposal_remaining_mask, mask_token, current_ids)
            proposal = self.proposal.propose(
                self.model,
                current_ids,
                attention_mask,
                valid_mask,
                proposal_remaining_mask,
                generator,
                prompt_condition_mask=prompt_condition_mask,
                promoted_generated_mask=promoted_generated_mask,
                progress_factory=progress_factory,
            )
            promotion_mask, stats = select_online_token_promotion(
                remaining_mask,
                proposal.confidence,
                proposal.candidate_ids,
                self.config.online_token_promotion,
                generator=generator,
                eos_token_id=self.model.eos_token_id,
            )
            remaining_before = [int(v) for v in remaining_mask.sum(dim=1).tolist()]
            current_ids = torch.where(promotion_mask, proposal.candidate_ids, current_ids)
            promoted_generated_mask = promoted_generated_mask | promotion_mask
            remaining_mask = original_corrupt_mask & ~promoted_generated_mask
            if self.config.save_history:
                history.append(
                    SamplingPromotionStep(
                        promotion_step=step_idx + 1,
                        remaining_before=remaining_before,
                        promoted=stats.promoted_counts,
                        promotion_fraction=float(stats.promotion_fraction),
                        token_ids=proposal.candidate_ids.detach().cpu(),
                        confidence=proposal.confidence.detach().float().cpu(),
                        promotion_mask=promotion_mask.detach().cpu(),
                    )
                )

        if bool(remaining_mask.any().item()):
            proposal_remaining_mask = original_corrupt_mask & ~promoted_generated_mask
            current_ids = torch.where(proposal_remaining_mask, mask_token, current_ids)
            proposal = self.proposal.propose(
                self.model,
                current_ids,
                attention_mask,
                valid_mask,
                proposal_remaining_mask,
                generator,
                prompt_condition_mask=prompt_condition_mask,
                promoted_generated_mask=promoted_generated_mask,
                progress_factory=progress_factory,
            )
            current_ids = torch.where(remaining_mask, proposal.candidate_ids, current_ids)
            promoted_generated_mask = promoted_generated_mask | remaining_mask
        return current_ids, promoted_generated_mask

    @torch.no_grad()
    def sample_batch(
        self,
        batch: dict[str, torch.Tensor],
        corrupt_mask: torch.Tensor,
        generator: torch.Generator,
        valid_mask: Optional[torch.Tensor] = None,
        progress_factory=None,
    ) -> SamplingResult:
        input_ids = batch["input_ids"].to(device=self.model.device).long()
        attention_mask = batch.get("attention_mask", torch.ones_like(input_ids)).to(device=self.model.device)
        corrupt_mask = corrupt_mask.to(device=self.model.device).bool() & attention_mask.bool()
        if valid_mask is None:
            valid_mask = attention_mask.bool()
        else:
            valid_mask = valid_mask.to(device=self.model.device).bool() & attention_mask.bool()
        original_corrupt_mask = corrupt_mask.clone()
        prompt_condition_mask = valid_mask & ~original_corrupt_mask
        promoted_generated_mask = torch.zeros_like(original_corrupt_mask, dtype=torch.bool)
        mask_token = torch.tensor(self.model.mask_token_id, device=self.model.device, dtype=torch.long)
        current_ids = torch.where(original_corrupt_mask, mask_token, input_ids)
        history = []

        current_ids, promoted_generated_mask = self._promote_tokens(
            current_ids,
            attention_mask,
            valid_mask,
            original_corrupt_mask,
            prompt_condition_mask,
            promoted_generated_mask,
            mask_token,
            generator,
            history,
            progress_factory=progress_factory,
        )
        generated = torch.where(original_corrupt_mask, current_ids, input_ids)
        return SamplingResult(
            input_ids=input_ids.detach().cpu(),
            generated_ids=generated.detach().cpu(),
            corrupt_mask=original_corrupt_mask.detach().cpu(),
            texts=self.model.decode(generated),
            history=history,
        )

    @torch.no_grad()
    def sample_prompts(self, prompts: list[str], generator: torch.Generator, progress_factory=None) -> SamplingResult:
        batch, corrupt_mask = self.model.encode_prompts(prompts, max_length=self.config.max_length)
        return self.sample_batch(batch, corrupt_mask, generator, progress_factory=progress_factory)
