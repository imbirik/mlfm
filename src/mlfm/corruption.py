"""Mask sampling and continuous embedding corruption for MLFM."""

from __future__ import annotations

import math
from typing import Dict, Iterable, Optional, Sequence

import torch


def build_valid_token_mask(
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    special_token_ids: Optional[Iterable[int]] = None,
) -> torch.Tensor:
    """Return positions that may be corrupted and trained on."""
    if attention_mask is None:
        valid = torch.ones_like(input_ids, dtype=torch.bool)
    else:
        valid = attention_mask.bool()
    if special_token_ids:
        for token_id in special_token_ids:
            if token_id is not None and token_id >= 0:
                valid = valid & (input_ids != int(token_id))
    return valid


def sample_mask_ratio(
    generator: torch.Generator,
    batch_size: int,
    mode: str,
    p_min: float,
    p_max: float,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    maskgit_cosine_power: float = 1.0,
    quantiles: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Sample per-example corruption ratios."""
    if not 0.0 <= p_min <= p_max <= 1.0:
        raise ValueError(f"Expected 0 <= p_min <= p_max <= 1, got {p_min}, {p_max}")
    maskgit_cosine_power = float(maskgit_cosine_power)
    if maskgit_cosine_power <= 0:
        raise ValueError("`maskgit_cosine_power` must be positive.")
    if quantiles is None:
        u = torch.rand((batch_size,), device=device, generator=generator, dtype=dtype)
    else:
        u = quantiles.to(device=device, dtype=dtype).reshape(batch_size).clamp(0.0, 1.0)
    if mode == "uniform":
        raw = u
    elif mode == "maskgit_cosine":
        raw = torch.cos(u * math.pi / 2.0).pow(maskgit_cosine_power)
    else:
        raise ValueError(f"Unknown mask ratio sampler: {mode}")
    return p_min + (p_max - p_min) * raw


def sample_low_discrepancy_quantiles(
    generator: torch.Generator,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return one jittered quantile from each batch stratum, in random order."""
    batch_size = int(batch_size)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    strata = torch.arange(batch_size, device=device, dtype=dtype)
    jitter = torch.rand((batch_size,), device=device, generator=generator, dtype=dtype)
    quantiles = (strata + jitter) / float(batch_size)
    order = torch.randperm(batch_size, device=device, generator=generator)
    return quantiles[order]


def sample_corruption_mask(
    valid_mask: torch.Tensor,
    mask_ratio: torch.Tensor,
    generator: torch.Generator,
    guarantee_nonempty: bool = True,
) -> torch.Tensor:
    """Sample a boolean corruption mask, optionally forcing one valid token per row."""
    if valid_mask.ndim != 2:
        raise ValueError(f"valid_mask must have shape [batch, seq], got {tuple(valid_mask.shape)}")
    batch_size, seq_len = valid_mask.shape
    p = mask_ratio.to(device=valid_mask.device, dtype=torch.float32).reshape(batch_size, 1)
    draws = torch.rand((batch_size, seq_len), device=valid_mask.device, generator=generator)
    corrupt_mask = (draws < p) & valid_mask.bool()
    if not guarantee_nonempty:
        return corrupt_mask

    has_valid = valid_mask.any(dim=1)
    empty = (~corrupt_mask.any(dim=1)) & has_valid
    if empty.any():
        weights = valid_mask[empty].float()
        chosen = torch.multinomial(weights, num_samples=1, replacement=True, generator=generator).squeeze(1)
        rows = torch.arange(batch_size, device=valid_mask.device)
        corrupt_mask[rows[empty], chosen] = True
    return corrupt_mask


def _as_float_list(value, default: Sequence[float]):
    if value is None:
        return list(default)
    if isinstance(value, str):
        return [float(item.strip()) for item in value.split(",") if item.strip()]
    return [float(item) for item in value]


def _sample_normal_gamma(
    loc: float,
    scale: float,
    gamma_min: float,
    gamma_max: float,
    count: int,
    generator: torch.Generator,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if scale <= 0:
        raise ValueError("Normal gamma scales must be positive.")
    gamma = torch.randn((count,), device=device, generator=generator, dtype=dtype) * float(scale) + float(loc)
    return gamma.clamp(min=gamma_min, max=gamma_max)


def _normal_cdf(gamma: torch.Tensor, loc: float, scale: float) -> torch.Tensor:
    if scale <= 0:
        raise ValueError("Normal gamma scales must be positive.")
    z = (gamma - float(loc)) / (float(scale) * math.sqrt(2.0))
    return 0.5 * (1.0 + torch.erf(z))


def _normal_pdf(gamma: torch.Tensor, loc: float, scale: float) -> torch.Tensor:
    if scale <= 0:
        raise ValueError("Normal gamma scales must be positive.")
    z = (gamma - float(loc)) / float(scale)
    return torch.exp(-0.5 * z * z) / (float(scale) * math.sqrt(2.0 * math.pi))


def _is_active_piecewise_gamma_schedule(schedule: str) -> bool:
    return schedule in {"active_piecewise", "gamma_active_piecewise", "active_empirical", "active_cdf"}


def _is_active_mixture_gamma_schedule(schedule: str) -> bool:
    return schedule in {"active_mixture", "gamma_active_mixture", "curve_mixture", "gamma_curve_mixture"}


def _active_piecewise_knots(config, device=None, dtype=torch.float32):
    return _piecewise_knots(
        config,
        gamma_attr="gamma_active_piecewise_gamma",
        cdf_attr="gamma_active_piecewise_cdf",
        device=device,
        dtype=dtype,
    )


def _piecewise_knots(config, gamma_attr: str, cdf_attr: str, device=None, dtype=torch.float32):
    gamma_knots = getattr(config, gamma_attr, None)
    cdf_knots = getattr(config, cdf_attr, None)
    if gamma_knots is None or cdf_knots is None:
        return None
    if len(gamma_knots) != len(cdf_knots) or len(gamma_knots) < 2:
        return None
    gamma = torch.tensor([float(value) for value in gamma_knots], device=device, dtype=dtype)
    cdf = torch.tensor([float(value) for value in cdf_knots], device=device, dtype=dtype)
    valid = torch.isfinite(gamma) & torch.isfinite(cdf)
    gamma = gamma[valid]
    cdf = cdf[valid]
    if gamma.numel() < 2:
        return None
    order = torch.argsort(gamma)
    gamma = gamma[order]
    cdf = cdf[order].clamp(0.0, 1.0)
    cdf = torch.cummax(cdf, dim=0).values
    keep = torch.ones_like(cdf, dtype=torch.bool)
    keep[1:] = cdf[1:] > cdf[:-1] + 1e-8
    gamma = gamma[keep]
    cdf = cdf[keep]
    if gamma.numel() < 2 or float((cdf[-1] - cdf[0]).item()) <= 1e-8:
        return None
    cdf = (cdf - cdf[0]) / (cdf[-1] - cdf[0])
    cdf[0] = 0.0
    cdf[-1] = 1.0
    return gamma, cdf


def _active_piecewise_available(config) -> bool:
    return _active_piecewise_knots(config, device=torch.device("cpu"), dtype=torch.float64) is not None


def _interp_monotone(x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor) -> torch.Tensor:
    x = x.to(device=xp.device, dtype=xp.dtype)
    idx = torch.searchsorted(xp.contiguous(), x.contiguous(), right=True) - 1
    idx = idx.clamp(0, xp.numel() - 2)
    x0 = xp[idx]
    x1 = xp[idx + 1]
    y0 = fp[idx]
    y1 = fp[idx + 1]
    frac = (x - x0) / (x1 - x0).clamp_min(torch.finfo(xp.dtype).eps)
    return y0 + frac * (y1 - y0)


def _active_piecewise_cdf(gamma: torch.Tensor, config) -> torch.Tensor:
    return _piecewise_cdf(gamma, config, _active_piecewise_knots)


def _piecewise_cdf(gamma: torch.Tensor, config, knots_fn) -> torch.Tensor:
    knots = knots_fn(config, device=gamma.device, dtype=gamma.dtype)
    if knots is None:
        return torch.zeros_like(gamma)
    gamma_knots, cdf_knots = knots
    return _interp_monotone(gamma.clamp(gamma_knots[0], gamma_knots[-1]), gamma_knots, cdf_knots).clamp(0.0, 1.0)


def _active_piecewise_pdf(gamma: torch.Tensor, config) -> torch.Tensor:
    return _piecewise_pdf(gamma, config, _active_piecewise_knots)


def _piecewise_pdf(gamma: torch.Tensor, config, knots_fn) -> torch.Tensor:
    knots = knots_fn(config, device=gamma.device, dtype=gamma.dtype)
    if knots is None:
        return torch.zeros_like(gamma)
    gamma_knots, cdf_knots = knots
    idx = torch.searchsorted(gamma_knots.contiguous(), gamma.contiguous(), right=True) - 1
    idx = idx.clamp(0, gamma_knots.numel() - 2)
    slopes = (cdf_knots[1:] - cdf_knots[:-1]) / (gamma_knots[1:] - gamma_knots[:-1]).clamp_min(
        torch.finfo(gamma.dtype).eps
    )
    pdf = slopes[idx]
    inside = (gamma >= gamma_knots[0]) & (gamma <= gamma_knots[-1])
    return torch.where(inside, pdf, torch.zeros_like(pdf))


def _active_piecewise_quantile(quantiles: torch.Tensor, config) -> torch.Tensor:
    return _piecewise_quantile(quantiles, config, _active_piecewise_knots)


def _piecewise_quantile(quantiles: torch.Tensor, config, knots_fn) -> torch.Tensor:
    knots = knots_fn(config, device=quantiles.device, dtype=quantiles.dtype)
    if knots is None:
        raise ValueError("Empirical piecewise gamma CDF is not available.")
    gamma_knots, cdf_knots = knots
    q = quantiles.clamp(0.0, 1.0)
    return _interp_monotone(q, cdf_knots, gamma_knots)


def _gumbel_pdf(gamma: torch.Tensor, loc: float, scale: float) -> torch.Tensor:
    if scale <= 0:
        raise ValueError("`gamma_scale` must be positive.")
    z = (gamma - float(loc)) / float(scale)
    return torch.exp(-(z + torch.exp(-z))) / float(scale)


def _gumbel_cdf(gamma: torch.Tensor, loc: float, scale: float) -> torch.Tensor:
    if scale <= 0:
        raise ValueError("`gamma_scale` must be positive.")
    z = (gamma - float(loc)) / float(scale)
    return torch.exp(-torch.exp(-z))


def _uniform_pdf(gamma: torch.Tensor, gamma_min: float, gamma_max: float) -> torch.Tensor:
    width = float(gamma_max) - float(gamma_min)
    if width <= 0.0:
        raise ValueError(f"Expected gamma_max > gamma_min, got {gamma_min}, {gamma_max}.")
    inside = (gamma >= float(gamma_min)) & (gamma <= float(gamma_max))
    return torch.where(inside, torch.full_like(gamma, 1.0 / width), torch.zeros_like(gamma))


def _active_mixture_weights(config, device=None, dtype=torch.float32) -> torch.Tensor:
    weights = _as_float_list(getattr(config, "gamma_active_mixture_weights", None), (0.1, 0.2, 0.7))
    if len(weights) != 3:
        raise ValueError("`gamma_active_mixture_weights` must have three entries: uniform, normal, active_curve.")
    if weights[2] > 0.0 and not _active_piecewise_available(config):
        weights[1] += weights[2]
        weights[2] = 0.0
    weights_tensor = torch.tensor(weights, device=device, dtype=dtype)
    if torch.any(weights_tensor < 0):
        raise ValueError("`gamma_active_mixture_weights` entries must be non-negative.")
    total = weights_tensor.sum()
    if not bool(torch.isfinite(total).item()) or float(total.item()) <= 0.0:
        raise ValueError("`gamma_active_mixture_weights` must have positive finite sum.")
    return weights_tensor / total


def _active_mixture_cdf(gamma: torch.Tensor, config) -> torch.Tensor:
    gamma_min = float(getattr(config, "gamma_min", -6.0))
    gamma_max = float(getattr(config, "gamma_max", 6.0))
    weights = _active_mixture_weights(config, device=gamma.device, dtype=gamma.dtype)
    uniform_cdf = ((gamma - gamma_min) / (gamma_max - gamma_min)).clamp(0.0, 1.0)
    normal_cdf = _normal_cdf(
        gamma,
        loc=float(getattr(config, "gamma_loc", 0.0)),
        scale=float(getattr(config, "gamma_scale", 2.0)),
    )
    active_cdf = _active_piecewise_cdf(gamma, config)
    return weights[0] * uniform_cdf + weights[1] * normal_cdf + weights[2] * active_cdf


def _active_mixture_pdf(gamma: torch.Tensor, config) -> torch.Tensor:
    gamma_min = float(getattr(config, "gamma_min", -6.0))
    gamma_max = float(getattr(config, "gamma_max", 6.0))
    weights = _active_mixture_weights(config, device=gamma.device, dtype=gamma.dtype)
    normal_pdf = _normal_pdf(
        gamma,
        loc=float(getattr(config, "gamma_loc", 0.0)),
        scale=float(getattr(config, "gamma_scale", 2.0)),
    )
    return (
        weights[0] * _uniform_pdf(gamma, gamma_min, gamma_max)
        + weights[1] * normal_pdf
        + weights[2] * _active_piecewise_pdf(gamma, config)
    )


def _active_mixture_quantiles(config, quantiles: torch.Tensor) -> torch.Tensor:
    gamma_min = float(getattr(config, "gamma_min", -6.0))
    gamma_max = float(getattr(config, "gamma_max", 6.0))
    lo = torch.full_like(quantiles, gamma_min)
    hi = torch.full_like(quantiles, gamma_max)
    for _ in range(64):
        mid = (lo + hi) * 0.5
        cdf_mid = _active_mixture_cdf(mid, config)
        lo = torch.where(cdf_mid < quantiles, mid, lo)
        hi = torch.where(cdf_mid >= quantiles, mid, hi)
    return (lo + hi) * 0.5


def _non_fitted_gamma_cdf(gamma: torch.Tensor, config) -> torch.Tensor:
    gamma_min = float(getattr(config, "gamma_min", -6.0))
    gamma_max = float(getattr(config, "gamma_max", 6.0))
    schedule = str(getattr(config, "gamma_schedule", "gumbel") or "gumbel").lower()
    loc = float(getattr(config, "gamma_loc", 0.0))
    scale = float(getattr(config, "gamma_scale", 2.0))
    uniform_cdf = ((gamma - gamma_min) / (gamma_max - gamma_min)).clamp(0.0, 1.0)

    if schedule == "uniform" or _is_active_piecewise_gamma_schedule(schedule) or _is_active_mixture_gamma_schedule(schedule):
        return uniform_cdf
    if schedule == "normal":
        return _normal_cdf(gamma, loc=loc, scale=scale)
    if schedule == "gumbel":
        return _gumbel_cdf(gamma, loc=loc, scale=scale)
    raise ValueError(f"Unknown gamma_schedule: {schedule}")


def gamma_non_fitted_quantile_edges(config, bin_count: int) -> list:
    """Gamma bin edges from the fixed part of the configured schedule."""
    gamma_min = float(getattr(config, "gamma_min", -6.0))
    gamma_max = float(getattr(config, "gamma_max", 6.0))
    bin_count = int(bin_count)
    if gamma_max <= gamma_min:
        raise ValueError(f"Expected gamma_max > gamma_min, got {gamma_min}, {gamma_max}.")
    if bin_count <= 0:
        raise ValueError("bin_count must be positive.")

    dtype = torch.float64
    device = torch.device("cpu")
    min_tensor = torch.tensor(gamma_min, device=device, dtype=dtype)
    max_tensor = torch.tensor(gamma_max, device=device, dtype=dtype)
    cdf_min = _non_fitted_gamma_cdf(min_tensor, config)
    cdf_max = _non_fitted_gamma_cdf(max_tensor, config)
    quantiles = torch.linspace(float(cdf_min.item()), float(cdf_max.item()), bin_count + 1, device=device, dtype=dtype)

    lo = torch.full_like(quantiles, gamma_min)
    hi = torch.full_like(quantiles, gamma_max)
    for _ in range(64):
        mid = (lo + hi) * 0.5
        cdf_mid = _non_fitted_gamma_cdf(mid, config)
        lo = torch.where(cdf_mid < quantiles, mid, lo)
        hi = torch.where(cdf_mid >= quantiles, mid, hi)
    edges = ((lo + hi) * 0.5).tolist()
    edges[0] = gamma_min
    edges[-1] = gamma_max
    return [round(float(edge), 12) for edge in edges]


def gamma_distribution_component_pdfs(config, gamma: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Return continuous PDF components for the configured gamma sampler.

    Clamped normal/gumbel samplers also create point mass at the gamma bounds. This helper
    intentionally returns only the continuous density inside the plotted range.
    """
    gamma_min = float(getattr(config, "gamma_min", -6.0))
    gamma_max = float(getattr(config, "gamma_max", 6.0))
    schedule = str(getattr(config, "gamma_schedule", "gumbel") or "gumbel").lower()
    loc = float(getattr(config, "gamma_loc", 0.0))
    scale = float(getattr(config, "gamma_scale", 2.0))
    uniform = _uniform_pdf(gamma, gamma_min, gamma_max)
    normal = _normal_pdf(gamma, loc, scale)
    active = _active_piecewise_pdf(gamma, config)

    if schedule == "uniform":
        return {"uniform": uniform}
    if schedule == "normal":
        return {"normal": normal}
    if schedule == "gumbel":
        return {"gumbel": _gumbel_pdf(gamma, loc, scale)}
    if _is_active_piecewise_gamma_schedule(schedule):
        return {"active_piecewise": active}
    if _is_active_mixture_gamma_schedule(schedule):
        weights = _active_mixture_weights(config, device=gamma.device, dtype=gamma.dtype)
        components = {
            "uniform": weights[0] * uniform,
            "normal": weights[1] * normal,
            "active_piecewise": weights[2] * active,
        }
        components["mixture_total"] = components["uniform"] + components["normal"] + components["active_piecewise"]
        return components
    raise ValueError(f"Unknown gamma_schedule: {schedule}")


def gamma_distribution_component_cdfs(config, gamma: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Return CDF components for the configured gamma sampler."""
    gamma_min = float(getattr(config, "gamma_min", -6.0))
    gamma_max = float(getattr(config, "gamma_max", 6.0))
    schedule = str(getattr(config, "gamma_schedule", "gumbel") or "gumbel").lower()
    loc = float(getattr(config, "gamma_loc", 0.0))
    scale = float(getattr(config, "gamma_scale", 2.0))
    uniform = ((gamma - gamma_min) / (gamma_max - gamma_min)).clamp(0.0, 1.0)
    normal = _normal_cdf(gamma, loc, scale)
    active = _active_piecewise_cdf(gamma, config)

    if schedule == "uniform":
        return {"uniform": uniform}
    if schedule == "normal":
        return {"normal": normal}
    if schedule == "gumbel":
        return {"gumbel": _gumbel_cdf(gamma, loc, scale)}
    if _is_active_piecewise_gamma_schedule(schedule):
        return {"active_piecewise": active}
    if _is_active_mixture_gamma_schedule(schedule):
        weights = _active_mixture_weights(config, device=gamma.device, dtype=gamma.dtype)
        components = {
            "uniform": weights[0] * uniform,
            "normal": weights[1] * normal,
            "active_piecewise": weights[2] * active,
        }
        components["mixture_total"] = components["uniform"] + components["normal"] + components["active_piecewise"]
        return components
    raise ValueError(f"Unknown gamma_schedule: {schedule}")


def gamma_distribution_pdf(config, gamma: torch.Tensor) -> torch.Tensor:
    components = gamma_distribution_component_pdfs(config, gamma)
    if "mixture_total" in components:
        return components["mixture_total"]
    return next(iter(components.values()))


def gamma_distribution_cdf(config, gamma: torch.Tensor) -> torch.Tensor:
    components = gamma_distribution_component_cdfs(config, gamma)
    if "mixture_total" in components:
        return components["mixture_total"]
    return next(iter(components.values()))


def _normal_quantile(quantiles: torch.Tensor, loc: float, scale: float) -> torch.Tensor:
    if scale <= 0:
        raise ValueError("Normal gamma scales must be positive.")
    q = quantiles.clamp(1e-6, 1.0 - 1e-6)
    return float(loc) + float(scale) * math.sqrt(2.0) * torch.erfinv(2.0 * q - 1.0)


def sample_log_nsr_gamma(
    config,
    generator: torch.Generator,
    batch_size: int,
    device: torch.device,
    dtype=torch.float32,
    quantiles: Optional[torch.Tensor] = None,
):
    """Sample log noise-to-signal ratio gamma values."""
    gamma_min = float(getattr(config, "gamma_min", -6.0))
    gamma_max = float(getattr(config, "gamma_max", 6.0))
    if gamma_max <= gamma_min:
        raise ValueError(f"Expected gamma_max > gamma_min, got {gamma_min}, {gamma_max}.")
    schedule = str(getattr(config, "gamma_schedule", "gumbel") or "gumbel").lower()
    loc = float(getattr(config, "gamma_loc", 0.0))
    scale = float(getattr(config, "gamma_scale", 2.0))
    if scale <= 0:
        raise ValueError("`gamma_scale` must be positive.")
    q = None if quantiles is None else quantiles.to(device=device, dtype=dtype).reshape(batch_size).clamp(0.0, 1.0)

    if schedule == "uniform":
        u = torch.rand((batch_size,), device=device, generator=generator, dtype=dtype) if q is None else q
        gamma = gamma_min + (gamma_max - gamma_min) * u
    elif schedule == "normal":
        if q is None:
            gamma = _sample_normal_gamma(loc, scale, gamma_min, gamma_max, batch_size, generator, device, dtype)
        else:
            gamma = _normal_quantile(q, loc, scale).clamp(min=gamma_min, max=gamma_max)
    elif _is_active_piecewise_gamma_schedule(schedule):
        u = torch.rand((batch_size,), device=device, generator=generator, dtype=dtype) if q is None else q
        gamma = _active_piecewise_quantile(u, config).clamp(min=gamma_min, max=gamma_max)
    elif _is_active_mixture_gamma_schedule(schedule):
        u = torch.rand((batch_size,), device=device, generator=generator, dtype=dtype) if q is None else q
        gamma = _active_mixture_quantiles(config, u).clamp(min=gamma_min, max=gamma_max)
    elif schedule == "gumbel":
        u = (
            torch.rand((batch_size,), device=device, generator=generator, dtype=dtype)
            if q is None
            else q
        ).clamp_(1e-6, 1.0 - 1e-6)
        gamma = loc - scale * torch.log(-torch.log(u))
    else:
        raise ValueError(f"Unknown gamma_schedule: {schedule}")
    return gamma.clamp(min=gamma_min, max=gamma_max)


def normalize_forward_process(process: str) -> str:
    """Validate the camera-ready forward-process surface."""
    process = str(process or "brownian_bridge").lower()
    if process != "brownian_bridge":
        raise ValueError("Camera-ready MLFM supports only forward_process=`brownian_bridge`.")
    return process


def gamma_to_bridge_t(gamma: torch.Tensor, bridge_sigma: float = 1.0) -> torch.Tensor:
    """Map log NSR gamma to Brownian bridge time."""
    bridge_sigma = float(bridge_sigma)
    if bridge_sigma <= 0:
        raise ValueError("`bridge_sigma` must be positive for log NSR Brownian bridge conversion.")
    gamma = gamma.float()
    log_sigma_sq = math.log(bridge_sigma * bridge_sigma)
    return torch.sigmoid(torch.full_like(gamma, log_sigma_sq) - gamma)


def gamma_to_process_coefficients(
    gamma: torch.Tensor,
    process: str,
    bridge_sigma: float = 1.0,
):
    """Return process time, signal coefficient, and noise coefficient for log NSR gamma."""
    normalize_forward_process(process)
    t = gamma_to_bridge_t(gamma, bridge_sigma=bridge_sigma)
    noise = float(bridge_sigma) * torch.sqrt(torch.clamp(t * (1.0 - t), min=0.0))
    return t, t, noise


def gamma_sampling_steps(config, n_steps: int, device=None, dtype=torch.float32) -> torch.Tensor:
    """Return monotone high-to-low gamma values for validation generation."""
    device = device or torch.device("cpu")
    n_steps = int(n_steps)
    if n_steps <= 0:
        raise ValueError("n_steps must be positive.")
    gamma_min = float(getattr(config, "gamma_min", -6.0))
    gamma_max = float(getattr(config, "gamma_max", 6.0))
    schedule = str(getattr(config, "gamma_schedule", "gumbel") or "gumbel").lower()
    loc = float(getattr(config, "gamma_loc", 0.0))
    scale = float(getattr(config, "gamma_scale", 2.0))
    if scale <= 0:
        raise ValueError("`gamma_scale` must be positive.")

    if schedule == "uniform":
        return torch.linspace(gamma_max, gamma_min, n_steps + 1, device=device, dtype=dtype)

    if schedule == "normal":
        min_tensor = torch.tensor(gamma_min, device=device, dtype=dtype)
        max_tensor = torch.tensor(gamma_max, device=device, dtype=dtype)
        cdf_min = _normal_cdf(min_tensor, loc=loc, scale=scale)
        cdf_max = _normal_cdf(max_tensor, loc=loc, scale=scale)
        q = torch.linspace(cdf_max.item(), cdf_min.item(), n_steps + 1, device=device, dtype=dtype)
        gamma = _normal_quantile(q, loc=loc, scale=scale)
        gamma[0] = gamma_max
        gamma[-1] = gamma_min
        return gamma.clamp(min=gamma_min, max=gamma_max)

    if _is_active_piecewise_gamma_schedule(schedule):
        q = torch.linspace(1.0, 0.0, n_steps + 1, device=device, dtype=dtype)
        gamma = _active_piecewise_quantile(q, config)
        gamma[0] = gamma_max
        gamma[-1] = gamma_min
        return gamma.clamp(min=gamma_min, max=gamma_max)

    if _is_active_mixture_gamma_schedule(schedule):
        cdf_min = _active_mixture_cdf(torch.tensor(gamma_min, device=device, dtype=dtype), config)
        cdf_max = _active_mixture_cdf(torch.tensor(gamma_max, device=device, dtype=dtype), config)
        q = torch.linspace(cdf_max.item(), cdf_min.item(), n_steps + 1, device=device, dtype=dtype)
        gamma = _active_mixture_quantiles(config, q)
        gamma[0] = gamma_max
        gamma[-1] = gamma_min
        return gamma.clamp(min=gamma_min, max=gamma_max)

    if schedule == "gumbel":
        def cdf(value: float) -> float:
            return math.exp(-math.exp(-(float(value) - loc) / scale))

        p_min = max(min(cdf(gamma_min), 1.0 - 1e-6), 1e-6)
        p_max = max(min(cdf(gamma_max), 1.0 - 1e-6), 1e-6)
        u = torch.linspace(p_max, p_min, n_steps + 1, device=device, dtype=dtype)
        gamma = loc - scale * torch.log(-torch.log(u))
        gamma[0] = gamma_max
        gamma[-1] = gamma_min
        return gamma.clamp(min=gamma_min, max=gamma_max)
    raise ValueError(f"Unknown gamma_schedule: {schedule}")


def _expand_t(t: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    t = t.to(device=target.device, dtype=target.dtype)
    if t.ndim == 0:
        t = t.reshape(1)
    if t.ndim == 1:
        t = t.reshape(-1, 1, 1)
    while t.ndim < target.ndim:
        t = t.unsqueeze(-1)
    return t


def _expand_mask_embedding(mask_embedding: torch.Tensor, clean_embeddings: torch.Tensor) -> torch.Tensor:
    mask_embedding = mask_embedding.to(device=clean_embeddings.device, dtype=clean_embeddings.dtype)
    if mask_embedding.ndim == 1:
        return mask_embedding.reshape(1, 1, -1).expand_as(clean_embeddings)
    if mask_embedding.ndim == 2 and mask_embedding.shape[0] == clean_embeddings.shape[0]:
        return mask_embedding[:, None, :].expand_as(clean_embeddings)
    return mask_embedding.expand_as(clean_embeddings)


def apply_forward_process(
    clean_embeddings: torch.Tensor,
    mask_embedding: torch.Tensor,
    corrupt_mask: torch.Tensor,
    t: torch.Tensor,
    generator: torch.Generator,
    process: str = "brownian_bridge",
    bridge_sigma: float = 1.0,
    bridge_noise_sampler=None,
) -> torch.Tensor:
    """Corrupt selected embedding positions and restore all observed positions exactly."""
    normalize_forward_process(process)
    mask_base = _expand_mask_embedding(mask_embedding, clean_embeddings)
    clean_values = clean_embeddings.float()
    mask_base_values = mask_base.float()

    if bridge_noise_sampler is None:
        noise = torch.randn(
            clean_embeddings.shape,
            device=clean_embeddings.device,
            dtype=torch.float32,
            generator=generator,
        )
    else:
        noise = bridge_noise_sampler.sample_like(clean_embeddings, generator)
    t_expanded = _expand_t(t, clean_values)
    bridge_mean = (1.0 - t_expanded) * mask_base_values + t_expanded * clean_values
    variance = torch.clamp(t_expanded * (1.0 - t_expanded), min=0.0)
    corrupted_values = bridge_mean + float(bridge_sigma) * torch.sqrt(variance) * noise

    corrupted_values = corrupted_values.to(dtype=clean_embeddings.dtype)
    corrupt = corrupt_mask.to(device=clean_embeddings.device).bool().unsqueeze(-1)
    return torch.where(corrupt, corrupted_values, clean_embeddings)
