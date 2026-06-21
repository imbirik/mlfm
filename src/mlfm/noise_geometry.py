"""Noise geometry helpers for MLFM bridge corruption."""

from __future__ import annotations

import math
from typing import Dict, Iterable, Optional

import torch
import torch.distributed as dist

from mlfm.corruption import build_valid_token_mask
from utils.train_utils import distributed_available, get_world_size


class BridgeNoiseSampler:
    """Sample Brownian-bridge embedding noise with configurable covariance."""

    def __init__(
        self,
        mode: str,
        hidden_dim: int,
        device: torch.device,
        diag_max_tokens: int = 262144,
        diag_min_tokens: int = 8192,
        diag_shrinkage: float = 0.05,
        diag_eps: float = 1e-8,
        rank: int = 0,
    ):
        self.mode = str(mode or "isotropic").lower()
        if self.mode not in {"isotropic", "empirical_diag"}:
            raise ValueError(f"Unknown bridge_noise_covariance: {self.mode}")
        self.hidden_dim = int(hidden_dim)
        self.device = device
        self.diag_max_tokens = max(0, int(diag_max_tokens))
        self.diag_min_tokens = max(0, int(diag_min_tokens))
        self.diag_shrinkage = float(diag_shrinkage)
        self.diag_eps = float(diag_eps)
        self.rank = int(rank)
        if not 0.0 <= self.diag_shrinkage <= 1.0:
            raise ValueError("bridge_noise_diag_shrinkage must be in [0, 1].")
        if self.diag_eps <= 0.0:
            raise ValueError("bridge_noise_diag_eps must be positive.")

        self.count = torch.zeros((), device=device, dtype=torch.float64)
        self.sum = torch.zeros(self.hidden_dim, device=device, dtype=torch.float64)
        self.sum_sq = torch.zeros(self.hidden_dim, device=device, dtype=torch.float64)
        self.frozen = self.mode == "isotropic" or self.diag_max_tokens <= 0
        self._scale_cache: Optional[torch.Tensor] = None

    @classmethod
    def from_config(cls, config, hidden_dim: int, device: torch.device) -> "BridgeNoiseSampler":
        return cls(
            mode=getattr(config, "bridge_noise_covariance", "isotropic"),
            hidden_dim=hidden_dim,
            device=device,
            diag_max_tokens=int(getattr(config, "bridge_noise_diag_max_tokens", 262144) or 0),
            diag_min_tokens=int(getattr(config, "bridge_noise_diag_min_tokens", 8192) or 0),
            diag_shrinkage=float(getattr(config, "bridge_noise_diag_shrinkage", 0.05)),
            diag_eps=float(getattr(config, "bridge_noise_diag_eps", 1e-8)),
            rank=int(getattr(config, "bridge_noise_rank", 0) or 0),
        )

    def state_dict(self) -> Dict[str, torch.Tensor | int | float | str | bool]:
        return {
            "mode": self.mode,
            "hidden_dim": self.hidden_dim,
            "diag_max_tokens": self.diag_max_tokens,
            "diag_min_tokens": self.diag_min_tokens,
            "diag_shrinkage": self.diag_shrinkage,
            "diag_eps": self.diag_eps,
            "rank": self.rank,
            "count": self.count.detach().cpu(),
            "sum": self.sum.detach().cpu(),
            "sum_sq": self.sum_sq.detach().cpu(),
            "frozen": bool(self.frozen),
        }

    def load_state_dict(self, state: Optional[Dict]) -> None:
        if not state:
            return
        if str(state.get("mode", self.mode)).lower() != self.mode:
            return
        if int(state.get("hidden_dim", self.hidden_dim)) != self.hidden_dim:
            return
        self.count.copy_(state["count"].to(device=self.device, dtype=torch.float64))
        self.sum.copy_(state["sum"].to(device=self.device, dtype=torch.float64))
        self.sum_sq.copy_(state["sum_sq"].to(device=self.device, dtype=torch.float64))
        self.frozen = bool(state.get("frozen", False)) or self.mode == "isotropic"
        if self.mode == "empirical_diag" and self.diag_max_tokens > 0 and int(self.count.item()) >= self.diag_max_tokens:
            self.frozen = True
        self._scale_cache = None

    @property
    def ready(self) -> bool:
        return self.mode == "empirical_diag" and int(self.count.item()) >= max(1, self.diag_min_tokens)

    @torch.no_grad()
    def update_from_batch(
        self,
        backbone_module,
        batch: Dict[str, torch.Tensor],
        special_token_ids: Optional[Iterable[int]] = None,
    ) -> None:
        if self.mode != "empirical_diag" or self.frozen:
            return
        input_ids = batch.get("input_ids")
        if input_ids is None:
            return
        attention_mask = batch.get("attention_mask")

        remaining = self.diag_max_tokens - int(self.count.item()) if self.diag_max_tokens > 0 else 0
        if remaining <= 0:
            self.frozen = True
            return
        world_size = get_world_size()
        local_budget = max(1, int(math.ceil(float(remaining) / float(max(world_size, 1)))))

        ids = input_ids.to(device=self.device, non_blocking=True).long()
        attn = attention_mask.to(device=self.device, non_blocking=True) if torch.is_tensor(attention_mask) else None
        specials = set(special_token_ids or [])
        specials.update(int(item) for item in getattr(backbone_module, "special_token_ids", []) or [])
        mask_token_id = getattr(backbone_module, "mask_token_id", None)
        if mask_token_id is not None:
            specials.add(int(mask_token_id))
        valid = build_valid_token_mask(ids, attention_mask=attn, special_token_ids=specials)

        input_embeddings = getattr(backbone_module, "input_embeddings", None)
        weight = getattr(input_embeddings, "weight", None)
        vocab_size = int(weight.shape[0]) if weight is not None else None
        if vocab_size is not None:
            valid = valid & (ids >= 0) & (ids < vocab_size)
        token_ids = ids[valid]
        if token_ids.numel() > local_budget:
            token_ids = token_ids[:local_budget]

        local_count = torch.tensor(float(token_ids.numel()), device=self.device, dtype=torch.float64)
        local_sum = torch.zeros_like(self.sum)
        local_sum_sq = torch.zeros_like(self.sum_sq)
        if token_ids.numel() > 0:
            x = backbone_module.embed(token_ids).detach().float()
            if x.ndim == 2 and x.shape[-1] == self.hidden_dim:
                x64 = x.to(dtype=torch.float64)
                local_sum += x64.sum(dim=0)
                local_sum_sq += x64.square().sum(dim=0)
            else:
                local_count.zero_()

        if distributed_available():
            dist.all_reduce(local_count, op=dist.ReduceOp.SUM)
            dist.all_reduce(local_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(local_sum_sq, op=dist.ReduceOp.SUM)

        if float(local_count.item()) <= 0.0:
            return
        self.count += local_count
        self.sum += local_sum
        self.sum_sq += local_sum_sq
        self._scale_cache = None
        if self.diag_max_tokens > 0 and int(self.count.item()) >= self.diag_max_tokens:
            self.frozen = True

    def _variance(self) -> Optional[torch.Tensor]:
        count = float(self.count.item())
        if self.mode != "empirical_diag" or count <= 1.0:
            return None
        mean = self.sum / count
        return (self.sum_sq / count - mean.square()).clamp_min(0.0)

    def scale(self) -> torch.Tensor:
        if not self.ready:
            return torch.ones(self.hidden_dim, device=self.device, dtype=torch.float32)
        if self._scale_cache is not None:
            return self._scale_cache
        var = self._variance()
        if var is None:
            self._scale_cache = torch.ones(self.hidden_dim, device=self.device, dtype=torch.float32)
            return self._scale_cache
        var_mean = var.mean().clamp_min(self.diag_eps)
        shrunk = (1.0 - self.diag_shrinkage) * var + self.diag_shrinkage * var_mean
        denom = shrunk.mean().clamp_min(self.diag_eps)
        self._scale_cache = torch.sqrt((shrunk + self.diag_eps) / (denom + self.diag_eps)).to(dtype=torch.float32)
        return self._scale_cache

    def sample_like(self, reference: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
        noise = torch.randn(reference.shape, device=reference.device, dtype=torch.float32, generator=generator)
        if self.mode != "empirical_diag" or not self.ready:
            return noise
        if int(reference.shape[-1]) != self.hidden_dim:
            raise ValueError(f"Bridge noise hidden dim mismatch: expected {self.hidden_dim}, got {reference.shape[-1]}")
        scale = self.scale().to(device=reference.device, dtype=torch.float32)
        return noise * scale.reshape(*([1] * (reference.ndim - 1)), self.hidden_dim)

    def metrics(self, prefix: str = "bridge_noise") -> Dict[str, float]:
        result = {
            f"{prefix}/mode_is_empirical_diag": 1.0 if self.mode == "empirical_diag" else 0.0,
            f"{prefix}/tokens_used": float(self.count.item()) if self.mode == "empirical_diag" else 0.0,
            f"{prefix}/ready": 1.0 if self.ready else 0.0,
            f"{prefix}/frozen": 1.0 if self.frozen else 0.0,
            f"{prefix}/rank": float(self.rank),
        }
        scale = self.scale()
        result.update(
            {
                f"{prefix}/scale_mean": float(scale.mean().item()),
                f"{prefix}/scale_std": float(scale.std(unbiased=False).item()),
                f"{prefix}/scale_min": float(scale.min().item()),
                f"{prefix}/scale_max": float(scale.max().item()),
            }
        )
        var = self._variance()
        if var is not None:
            var_mean = var.mean().clamp_min(self.diag_eps)
            result[f"{prefix}/variance_cv"] = float((var.std(unbiased=False) / var_mean).item())
        return result


def hidden_dim_from_backbone(backbone_module) -> int:
    input_embeddings = getattr(backbone_module, "input_embeddings", None)
    weight = getattr(input_embeddings, "weight", None)
    if weight is not None and weight.ndim == 2:
        return int(weight.shape[1])
    raise ValueError("Cannot infer bridge noise hidden dim from backbone input embeddings.")
