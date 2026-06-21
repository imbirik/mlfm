"""Gamma/time schedules for standalone sampling."""

from __future__ import annotations

from copy import copy
from typing import Any

import torch

from sampling.config import SamplingExperimentConfig
from mlfm.corruption import sample_log_nsr_gamma


def _resolve(value: Any, fallback: Any):
    return fallback if value == "from_config" or value is None else value


def sampling_gamma_config(training_config, sampling_config: SamplingExperimentConfig):
    """Return a lightweight config compatible with mlfm.corruption gamma helpers."""
    cfg = copy(training_config)
    time_cfg = sampling_config.time_distribution
    source = str(time_cfg.source or "from_config").lower()
    if source in {"from_config", "training_config", "training"}:
        pass
    elif source in {"active_mixture", "gamma_active_mixture", "curve_mixture", "gamma_curve_mixture"}:
        setattr(cfg, "gamma_schedule", "active_mixture")
    elif source in {"active_piecewise", "gamma_active_piecewise", "active_empirical", "active_cdf"}:
        setattr(cfg, "gamma_schedule", "active_piecewise")
    elif source in {"uniform", "normal", "gumbel"}:
        setattr(cfg, "gamma_schedule", source)
    else:
        raise ValueError(f"Unknown sampling time distribution source: {source}")

    setattr(cfg, "gamma_min", float(_resolve(time_cfg.gamma_min, getattr(training_config, "gamma_min", -10.0))))
    setattr(cfg, "gamma_max", float(_resolve(time_cfg.gamma_max, getattr(training_config, "gamma_max", 6.0))))
    return cfg


def gamma_quantile(training_config, sampling_config: SamplingExperimentConfig, q: torch.Tensor) -> torch.Tensor:
    cfg = sampling_gamma_config(training_config, sampling_config)
    device = q.device
    generator = torch.Generator(device=device if device.type == "cuda" else torch.device("cpu"))
    generator.manual_seed(0)
    return sample_log_nsr_gamma(cfg, generator, int(q.numel()), device, dtype=q.dtype, quantiles=q.reshape(-1)).reshape_as(q)


def gamma_trajectory(
    training_config,
    sampling_config: SamplingExperimentConfig,
    steps: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    steps = int(steps)
    if steps <= 0:
        raise ValueError("steps must be positive.")
    q_start = float(sampling_config.time_distribution.init_quantile)
    q_start = max(1e-6, min(q_start, 1.0 - 1e-6))
    quantiles = torch.linspace(q_start, 1e-6, steps + 1, device=device, dtype=dtype)
    gamma = gamma_quantile(training_config, sampling_config, quantiles)
    gamma_min = float(getattr(sampling_gamma_config(training_config, sampling_config), "gamma_min", -10.0))
    gamma[-1] = gamma_min
    return gamma


def init_gamma(training_config, sampling_config: SamplingExperimentConfig, batch_size: int, device: torch.device) -> torch.Tensor:
    q = torch.full(
        (int(batch_size),),
        max(1e-6, min(float(sampling_config.time_distribution.init_quantile), 1.0 - 1e-6)),
        device=device,
        dtype=torch.float32,
    )
    return gamma_quantile(training_config, sampling_config, q)
