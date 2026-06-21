"""Nested configuration for camera-ready checkpoint sampling."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional

import yaml


@dataclass
class TokenSamplerConfig:
    strategy: str = "gumbel"
    temperature: float = 1.0
    top_p: float = 1.0
    gumbel_dtype: str = "float64"
    suppress_token_ids: list[int] = field(default_factory=list)


@dataclass
class TimeDistributionConfig:
    source: str = "from_config"
    gamma_min: Any = "from_config"
    gamma_max: Any = "from_config"
    init_quantile: float = 0.95


@dataclass
class ConfidenceConfig:
    score: str = "sample_prob"


@dataclass
class OnlineTokenPromotionConfig:
    confidence_threshold: float = 0.8
    min_promote_tokens: int = 1
    eos_policy: str = "rank_together"


@dataclass
class CFGConfig:
    enabled: bool = False
    scale: float = 0.0
    mode: str = "drop"
    condition_scope: str = "prompt_only"
    target: str = "logits"
    logit_mode: str = "prob_ratio"
    rescale: str = "none"


@dataclass
class SamplingExperimentConfig:
    adapter_mode: str = "finetuned"
    checkpoint_weight_source: str = "ema"
    precision: str = "bf16"
    max_length: int = 1024
    save_history: bool = True
    stepper: str = "sde"
    steps: int = 341
    bridge_sigma: Any = "from_config"
    confidence_source: str = "quantile"
    confidence_quantile: float = 0.5
    token_sampler: TokenSamplerConfig = field(default_factory=TokenSamplerConfig)
    time_distribution: TimeDistributionConfig = field(default_factory=TimeDistributionConfig)
    confidence: ConfidenceConfig = field(default_factory=ConfidenceConfig)
    online_token_promotion: OnlineTokenPromotionConfig = field(default_factory=OnlineTokenPromotionConfig)
    cfg: CFGConfig = field(default_factory=CFGConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _merge_dict(base: dict[str, Any], update: Mapping[str, Any]) -> dict[str, Any]:
    for key, value in update.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            base[key] = _merge_dict(dict(base[key]), value)
        else:
            base[key] = value
    return base


def _known(cls, data: Mapping[str, Any]) -> dict[str, Any]:
    fields = getattr(cls, "__dataclass_fields__", {})
    return {key: data[key] for key in fields if key in data}


def normalize_cfg_mode(mode: Any) -> str:
    value = str(mode or "drop").lower()
    aliases = {"drop": "drop", "mask": "mask", "corrupt": "corrupt"}
    if value not in aliases:
        raise ValueError(f"Unknown CFG mode: {mode}")
    return aliases[value]


def _from_dict(data: Mapping[str, Any]) -> SamplingExperimentConfig:
    defaults = SamplingExperimentConfig().to_dict()
    merged = _merge_dict(defaults, dict(data))
    time_data = dict(merged.get("time_distribution") or {})
    cfg_data = dict(merged.get("cfg") or {})
    cfg_data["mode"] = normalize_cfg_mode(cfg_data.get("mode", "drop"))
    config = SamplingExperimentConfig(
        adapter_mode=str(merged.get("adapter_mode", "finetuned")),
        checkpoint_weight_source=str(merged.get("checkpoint_weight_source", "ema")),
        precision=str(merged.get("precision", "bf16")),
        max_length=int(merged.get("max_length", 1024)),
        save_history=bool(merged.get("save_history", True)),
        stepper=str(merged.get("stepper", "sde")),
        steps=int(merged.get("steps", 341)),
        bridge_sigma=merged.get("bridge_sigma", "from_config"),
        confidence_source=str(merged.get("confidence_source", "quantile")),
        confidence_quantile=float(merged.get("confidence_quantile", 0.5)),
        token_sampler=TokenSamplerConfig(**_known(TokenSamplerConfig, dict(merged.get("token_sampler") or {}))),
        time_distribution=TimeDistributionConfig(
            source=time_data.get("source", "from_config"),
            gamma_min=time_data.get("gamma_min", "from_config"),
            gamma_max=time_data.get("gamma_max", "from_config"),
            init_quantile=float(time_data.get("init_quantile", 0.95)),
        ),
        confidence=ConfidenceConfig(**_known(ConfidenceConfig, dict(merged.get("confidence") or {}))),
        online_token_promotion=OnlineTokenPromotionConfig(
            **_known(OnlineTokenPromotionConfig, dict(merged.get("online_token_promotion") or {}))
        ),
        cfg=CFGConfig(**_known(CFGConfig, cfg_data)),
    )
    _validate_sampling_experiment_config(config)
    return config


def _validate_sampling_experiment_config(config: SamplingExperimentConfig) -> None:
    stepper = str(config.stepper or "sde").lower()
    precision = str(config.precision or "bf16").lower()
    confidence_source = str(config.confidence_source or "quantile").lower()
    time_source = str(config.time_distribution.source or "from_config").lower()
    scope = str(config.cfg.condition_scope or "prompt_only").lower()
    target = str(config.cfg.target or "logits").lower()
    logit_mode = str(getattr(config.cfg, "logit_mode", "prob_ratio") or "prob_ratio").lower()
    rescale = str(config.cfg.rescale or "none").lower()
    eos_policy = str(config.online_token_promotion.eos_policy or "rank_together").lower()

    if stepper != "sde":
        raise ValueError("The camera-ready sampler supports only sampling.stepper=`sde`.")
    if precision not in {"bf16", "float32", "fp32"}:
        raise ValueError(f"Unknown sampling precision: {config.precision}")
    if int(config.steps) < 1:
        raise ValueError(f"sampling.steps must be >= 1, got {config.steps}.")
    if confidence_source not in {"quantile", "final"}:
        raise ValueError(f"Unknown sampling confidence_source: {config.confidence_source}")
    if time_source not in {
        "from_config",
        "training_config",
        "training",
        "uniform",
        "normal",
        "gumbel",
        "active_piecewise",
        "gamma_active_piecewise",
        "active_empirical",
        "active_cdf",
        "active_mixture",
        "gamma_active_mixture",
        "curve_mixture",
        "gamma_curve_mixture",
    }:
        raise ValueError(f"Unknown sampling time_distribution.source: {config.time_distribution.source}")
    if float(config.online_token_promotion.confidence_threshold) < 0.0:
        raise ValueError("online_token_promotion.confidence_threshold must be non-negative.")
    if int(config.online_token_promotion.min_promote_tokens) < 0:
        raise ValueError("online_token_promotion.min_promote_tokens must be non-negative.")
    if eos_policy not in {"rank_together", "rank_separately", "zero_confidence", "promote_all"}:
        raise ValueError(f"Unknown online_token_promotion.eos_policy: {config.online_token_promotion.eos_policy}")
    if scope not in {"prompt_only", "prompt_and_promoted"}:
        raise ValueError(f"Unknown CFG condition_scope: {config.cfg.condition_scope}")
    if target not in {"logits", "velocity"}:
        raise ValueError(f"Unknown CFG target: {config.cfg.target}")
    if logit_mode not in {"prob_ratio", "probability_ratio", "log_prob_ratio", "raw", "raw_logits", "smdm"}:
        raise ValueError(f"Unknown CFG logit_mode: {config.cfg.logit_mode}")
    if rescale not in {"none", "match_cond_std", "match_cond_norm"}:
        raise ValueError(f"Unknown CFG rescale mode: {config.cfg.rescale}")
    if normalize_cfg_mode(config.cfg.mode) == "drop" and scope != "prompt_only":
        raise ValueError("CFG mode `drop` only supports condition_scope=`prompt_only`.")
    if target == "logits" and rescale != "none":
        raise ValueError("CFG target=`logits` requires cfg.rescale=`none`.")


def _set_nested(data: dict[str, Any], path: str, value: Any) -> None:
    cursor = data
    parts = [part for part in path.split(".") if part]
    if not parts:
        raise ValueError("Empty sampling override key.")
    for part in parts[:-1]:
        next_value = cursor.setdefault(part, {})
        if not isinstance(next_value, dict):
            raise ValueError(f"Cannot set nested override through non-dict key: {path}")
        cursor = next_value
    cursor[parts[-1]] = value


def load_sampling_experiment_config(
    config_path: Optional[str] = None,
    sampling_config_path: Optional[str] = None,
    overrides: Optional[list[str]] = None,
) -> SamplingExperimentConfig:
    data: dict[str, Any] = {}
    for path in (config_path, sampling_config_path):
        if not path:
            continue
        resolved = Path(path)
        if not resolved.exists():
            continue
        with resolved.open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
        sampling_data = loaded.get("sampling", loaded)
        if isinstance(sampling_data, Mapping):
            data = _merge_dict(data, sampling_data)
    for override in overrides or []:
        if "=" not in override:
            raise ValueError(f"Invalid sampling override {override!r}; expected key=value.")
        key, raw_value = override.split("=", 1)
        _set_nested(data, key.strip(), yaml.safe_load(raw_value.strip()))
    return _from_dict(data)
