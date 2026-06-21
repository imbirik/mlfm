"""Self-contained checkpoint sampling utilities."""

from sampling.config import SamplingExperimentConfig, load_sampling_experiment_config

__all__ = [
    "OnlineTokenPromotionSampler",
    "SamplingExperimentConfig",
    "SamplingModel",
    "SamplingResult",
    "load_sampling_experiment_config",
]


def __getattr__(name):
    if name in {"OnlineTokenPromotionSampler", "SamplingResult"}:
        from sampling.sampler import OnlineTokenPromotionSampler, SamplingResult

        return {"OnlineTokenPromotionSampler": OnlineTokenPromotionSampler, "SamplingResult": SamplingResult}[name]
    if name == "SamplingModel":
        from sampling.model import SamplingModel

        return SamplingModel
    raise AttributeError(name)
