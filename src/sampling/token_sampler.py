"""Token sampling and confidence helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from sampling.config import TokenSamplerConfig


@dataclass
class TokenSample:
    token_ids: torch.Tensor
    sample_prob: torch.Tensor
    max_prob: torch.Tensor
    entropy: torch.Tensor
    margin: torch.Tensor
    filtered_logits: torch.Tensor


def _apply_top_p(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    top_p = float(top_p)
    if top_p >= 1.0:
        return logits
    if top_p <= 0.0:
        raise ValueError("top_p must be positive.")
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    sorted_probs = torch.softmax(sorted_logits.float(), dim=-1)
    cumulative = sorted_probs.cumsum(dim=-1)
    remove = cumulative > top_p
    remove[..., 1:] = remove[..., :-1].clone()
    remove[..., 0] = False
    sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
    filtered = torch.full_like(logits, float("-inf"))
    return filtered.scatter(dim=-1, index=sorted_indices, src=sorted_logits)


def filter_logits(logits: torch.Tensor, config: TokenSamplerConfig) -> torch.Tensor:
    filtered = logits.float()
    suppress = list(config.suppress_token_ids or [])
    if suppress:
        valid = [int(token_id) for token_id in suppress if 0 <= int(token_id) < filtered.shape[-1]]
        if valid:
            filtered = filtered.clone()
            filtered[..., valid] = float("-inf")
    filtered = _apply_top_p(filtered, float(config.top_p))
    if not torch.isfinite(filtered).any(dim=-1).all():
        raise ValueError("Token filtering removed all candidates for at least one position.")
    return filtered


def _gumbel_noise(shape, device, generator: Optional[torch.Generator], dtype: torch.dtype):
    uniform = torch.rand(shape, device=device, generator=generator, dtype=dtype)
    uniform = uniform.clamp_(torch.finfo(dtype).tiny, 1.0 - torch.finfo(dtype).eps)
    return -torch.log(-torch.log(uniform))


def sample_tokens(
    logits: torch.Tensor,
    config: TokenSamplerConfig,
    generator: Optional[torch.Generator] = None,
) -> TokenSample:
    strategy = str(config.strategy or "gumbel").lower()
    temperature = float(config.temperature)
    if temperature < 0.0:
        raise ValueError("temperature must be non-negative.")
    filtered = filter_logits(logits, config)
    sampling_logits = filtered
    if strategy == "argmax":
        token_ids = filtered.argmax(dim=-1)
    elif strategy == "gumbel":
        if temperature == 0.0:
            token_ids = filtered.argmax(dim=-1)
        else:
            dtype = torch.float64 if str(config.gumbel_dtype).lower() == "float64" else torch.float32
            sampling_logits = (filtered.to(dtype=dtype) / temperature).to(dtype)
            token_ids = (sampling_logits + _gumbel_noise(sampling_logits.shape, sampling_logits.device, generator, dtype)).argmax(dim=-1)
    else:
        raise ValueError(f"Unknown token sampling strategy: {strategy}")

    probs = torch.softmax(sampling_logits.float(), dim=-1)
    sample_prob = probs.gather(-1, token_ids.unsqueeze(-1)).squeeze(-1)
    top2 = torch.topk(probs, k=min(2, probs.shape[-1]), dim=-1).values
    max_prob = top2[..., 0]
    margin = top2[..., 0] - (top2[..., 1] if top2.shape[-1] > 1 else torch.zeros_like(top2[..., 0]))
    log_probs = torch.log_softmax(sampling_logits.float(), dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1)
    return TokenSample(
        token_ids=token_ids.long(),
        sample_prob=sample_prob,
        max_prob=max_prob,
        entropy=entropy,
        margin=margin,
        filtered_logits=filtered,
    )


def confidence_from_sample(sample: TokenSample, score: str) -> torch.Tensor:
    score = str(score or "sample_prob").lower()
    if score == "sample_prob":
        return sample.sample_prob.float()
    if score == "max_prob":
        return sample.max_prob.float()
    if score in {"neg_entropy", "negative_entropy"}:
        return -sample.entropy.float()
    if score == "margin":
        return sample.margin.float()
    if score == "random":
        return torch.rand_like(sample.sample_prob.float())
    raise ValueError(f"Unknown confidence score: {score}")
