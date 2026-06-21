"""SDE proposal backend for online token-promotion checkpoint sampling."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch

from sampling.config import SamplingExperimentConfig, normalize_cfg_mode
from sampling.model import SamplingModel
from sampling.time import gamma_trajectory
from sampling.token_sampler import confidence_from_sample, sample_tokens
from mlfm.corruption import apply_forward_process, gamma_to_process_coefficients, normalize_forward_process
from mlfm.train_step import _posterior_mean_embedding


@dataclass
class ProposalOutput:
    candidate_ids: torch.Tensor
    confidence: torch.Tensor
    logits: Optional[torch.Tensor] = None
    aux: dict = field(default_factory=dict)


@dataclass
class _FieldOutput:
    logits: torch.Tensor
    conditional_logits: torch.Tensor
    unconditional_logits: Optional[torch.Tensor]
    x0_pred: torch.Tensor
    velocity: torch.Tensor


def _observed_mask(valid_mask: torch.Tensor, remaining_mask: torch.Tensor) -> torch.Tensor:
    return valid_mask.bool() & ~remaining_mask.bool()


def _time_condition(config, gamma: torch.Tensor, process_t: torch.Tensor) -> torch.Tensor:
    return gamma if str(getattr(config, "time_conditioning", "t") or "t").lower() == "gamma" else process_t


def _cfg_enabled(config: SamplingExperimentConfig) -> bool:
    cfg = getattr(config, "cfg", None)
    return bool(cfg is not None and cfg.enabled and float(cfg.scale) != 0.0)


def _cfg_target(config: SamplingExperimentConfig) -> str:
    return str(getattr(config.cfg, "target", "logits") or "logits").lower()


def _sampling_precision(config: SamplingExperimentConfig) -> str:
    precision = str(getattr(config, "precision", "bf16") or "bf16").lower()
    if precision == "fp32":
        return "float32"
    if precision not in {"bf16", "float32"}:
        raise ValueError(f"Unknown sampling precision: {getattr(config, 'precision', precision)}")
    return precision


def _sampler_dtype(model: SamplingModel) -> Optional[torch.dtype]:
    return torch.float32 if _sampling_precision(model.sampling_config) == "float32" else None


def _to_sampler_dtype(model: SamplingModel, tensor: torch.Tensor) -> torch.Tensor:
    dtype = _sampler_dtype(model)
    return tensor if dtype is None else tensor.to(dtype=dtype)


def _posterior_mean_embedding_sampling(model: SamplingModel, logits: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    if _sampling_precision(model.sampling_config) != "float32":
        return _posterior_mean_embedding(model.module, logits, dtype=dtype)

    input_embeddings = getattr(model.module, "input_embeddings", None)
    if input_embeddings is not None and hasattr(input_embeddings, "weight"):
        embedding_weight = input_embeddings.weight
    else:
        token_ids = torch.arange(logits.shape[-1], device=logits.device)
        embedding_weight = model.module.embed(token_ids)
    if embedding_weight.shape[0] != logits.shape[-1]:
        if embedding_weight.shape[0] < logits.shape[-1]:
            raise ValueError(
                "Input embedding table is smaller than the predicted logit vocabulary: "
                f"{embedding_weight.shape[0]} embeddings for {logits.shape[-1]} logits."
            )
        embedding_weight = embedding_weight[: logits.shape[-1]]
    probs = torch.softmax(logits.float(), dim=-1)
    expected = torch.matmul(probs, embedding_weight.to(device=logits.device, dtype=torch.float32))
    return expected.to(dtype=dtype)


def _cfg_selected_condition_mask(
    config: SamplingExperimentConfig,
    prompt_condition_mask: Optional[torch.Tensor],
    promoted_generated_mask: Optional[torch.Tensor],
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    if prompt_condition_mask is None:
        selected = torch.zeros_like(attention_mask, dtype=torch.bool)
    else:
        selected = prompt_condition_mask.to(device=attention_mask.device).bool()
    scope = str(getattr(config.cfg, "condition_scope", "prompt_only") or "prompt_only").lower()
    if scope == "prompt_and_promoted":
        if promoted_generated_mask is not None:
            selected = selected | promoted_generated_mask.to(device=attention_mask.device).bool()
    elif scope != "prompt_only":
        raise ValueError(f"Unknown CFG condition_scope: {config.cfg.condition_scope}")
    return selected & attention_mask.bool()


def _apply_cfg_rescale(conditional: torch.Tensor, delta: torch.Tensor, rescale: str) -> torch.Tensor:
    rescale = str(rescale or "none").lower()
    if rescale == "none":
        return delta
    delta32 = delta.float()
    cond32 = conditional.float()
    if rescale == "match_cond_std":
        cond_scale = cond32.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-6)
        delta_scale = delta32.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-6)
        return (delta32 * (cond_scale / delta_scale)).to(dtype=delta.dtype)
    if rescale == "match_cond_norm":
        cond_scale = cond32.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        delta_scale = delta32.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        return (delta32 * (cond_scale / delta_scale)).to(dtype=delta.dtype)
    raise ValueError(f"Unknown CFG rescale mode: {rescale}")


def _apply_cfg(conditional: torch.Tensor, unconditional: torch.Tensor, config: SamplingExperimentConfig) -> torch.Tensor:
    if _sampling_precision(config) == "float32":
        conditional = conditional.float()
        unconditional = unconditional.float()
    delta = conditional - unconditional.to(dtype=conditional.dtype)
    delta = _apply_cfg_rescale(conditional, delta, getattr(config.cfg, "rescale", "none"))
    return conditional + float(config.cfg.scale) * delta


def _apply_logit_cfg(conditional: torch.Tensor, unconditional: torch.Tensor, config: SamplingExperimentConfig) -> torch.Tensor:
    rescale = str(getattr(config.cfg, "rescale", "none") or "none").lower()
    if rescale != "none":
        raise ValueError("CFG target=`logits` requires cfg.rescale=`none`.")
    mode = str(getattr(config.cfg, "logit_mode", "prob_ratio") or "prob_ratio").lower()
    if mode in {"raw", "raw_logits", "smdm"}:
        return _apply_cfg(conditional, unconditional, config)
    if mode not in {"prob_ratio", "probability_ratio", "log_prob_ratio"}:
        raise ValueError(f"Unknown CFG logit_mode: {getattr(config.cfg, 'logit_mode', mode)}")
    cond_logp = torch.log_softmax(conditional.float(), dim=-1)
    uncond_logp = torch.log_softmax(unconditional.float(), dim=-1)
    guided = cond_logp + float(config.cfg.scale) * (cond_logp - uncond_logp)
    return guided.to(dtype=conditional.dtype) if _sampling_precision(config) != "float32" else guided


def _repacked_unconditional_logits(
    model: SamplingModel,
    embeddings: torch.Tensor,
    attention_mask: torch.Tensor,
    time_t: torch.Tensor,
    observed_mask: torch.Tensor,
    drop_mask: torch.Tensor,
    fallback_logits: torch.Tensor,
) -> torch.Tensor:
    keep_mask = attention_mask.bool() & ~drop_mask.bool()
    lengths = keep_mask.long().sum(dim=1)
    if bool((lengths <= 0).any().item()):
        raise ValueError("CFG drop produced an empty unconditional row.")
    batch_size = embeddings.shape[0]
    max_len = int(lengths.max().item())
    packed_embeddings = embeddings.new_zeros((batch_size, max_len, embeddings.shape[-1]))
    packed_attention = attention_mask.new_zeros((batch_size, max_len))
    packed_observed = torch.zeros((batch_size, max_len), device=embeddings.device, dtype=torch.bool)
    kept_positions = []
    for row_idx in range(batch_size):
        positions = torch.nonzero(keep_mask[row_idx], as_tuple=False).flatten()
        length = int(positions.numel())
        kept_positions.append(positions)
        packed_embeddings[row_idx, :length] = embeddings[row_idx, positions]
        packed_attention[row_idx, :length] = 1
        packed_observed[row_idx, :length] = observed_mask[row_idx, positions]
    packed_logits = model.forward_embeddings(packed_embeddings, packed_attention, time_t, packed_observed)
    full_logits = fallback_logits.clone()
    for row_idx, positions in enumerate(kept_positions):
        full_logits[row_idx, positions] = packed_logits[row_idx, : int(positions.numel())]
    return full_logits


def _unconditional_logits(
    model: SamplingModel,
    embeddings: torch.Tensor,
    attention_mask: torch.Tensor,
    time_t: torch.Tensor,
    observed_mask: torch.Tensor,
    condition_mask: torch.Tensor,
    generator: torch.Generator,
    fallback_logits: torch.Tensor,
    process: Optional[str] = None,
    process_t: Optional[torch.Tensor] = None,
    bridge_sigma: Optional[float] = None,
) -> torch.Tensor:
    mode = normalize_cfg_mode(model.sampling_config.cfg.mode)
    uncond_observed = observed_mask.bool() & ~condition_mask.bool()
    if mode == "drop":
        return _repacked_unconditional_logits(
            model,
            embeddings,
            attention_mask,
            time_t,
            uncond_observed,
            condition_mask,
            fallback_logits,
        )
    if mode == "mask":
        mask_embedding = model.mask_embedding().to(device=embeddings.device, dtype=embeddings.dtype)
        uncond_embeddings = torch.where(condition_mask.unsqueeze(-1), mask_embedding.reshape(1, 1, -1), embeddings)
    elif mode == "corrupt":
        if process is None or process_t is None:
            raise ValueError("CFG mode `corrupt` requires the current forward process time.")
        uncond_embeddings = apply_forward_process(
            embeddings,
            model.mask_embedding(),
            condition_mask,
            process_t,
            generator,
            process=process,
            bridge_sigma=float(1.0 if bridge_sigma is None else bridge_sigma),
            bridge_noise_sampler=model.bridge_noise_sampler,
        )
    else:
        raise ValueError(f"Unknown CFG mode: {mode}")
    return model.forward_embeddings(uncond_embeddings, attention_mask, time_t, uncond_observed)


def _cfg_logits_pair(
    model: SamplingModel,
    embeddings: torch.Tensor,
    attention_mask: torch.Tensor,
    time_t: torch.Tensor,
    observed_mask: torch.Tensor,
    generator: torch.Generator,
    prompt_condition_mask: Optional[torch.Tensor],
    promoted_generated_mask: Optional[torch.Tensor],
    process: Optional[str] = None,
    process_t: Optional[torch.Tensor] = None,
    bridge_sigma: Optional[float] = None,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    conditional = model.forward_embeddings(embeddings, attention_mask, time_t, observed_mask)
    if not _cfg_enabled(model.sampling_config):
        return conditional, None
    condition_mask = _cfg_selected_condition_mask(
        model.sampling_config,
        prompt_condition_mask,
        promoted_generated_mask,
        attention_mask,
    )
    if not bool(condition_mask.any().item()):
        return conditional, None
    unconditional = _unconditional_logits(
        model,
        embeddings,
        attention_mask,
        time_t,
        observed_mask,
        condition_mask,
        generator,
        fallback_logits=conditional,
        process=process,
        process_t=process_t,
        bridge_sigma=bridge_sigma,
    )
    return conditional, unconditional


def _guided_logits(
    model: SamplingModel,
    embeddings: torch.Tensor,
    attention_mask: torch.Tensor,
    time_t: torch.Tensor,
    observed_mask: torch.Tensor,
    generator: torch.Generator,
    prompt_condition_mask: Optional[torch.Tensor],
    promoted_generated_mask: Optional[torch.Tensor],
    process: Optional[str] = None,
    process_t: Optional[torch.Tensor] = None,
    bridge_sigma: Optional[float] = None,
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    conditional, unconditional = _cfg_logits_pair(
        model,
        embeddings,
        attention_mask,
        time_t,
        observed_mask,
        generator,
        prompt_condition_mask,
        promoted_generated_mask,
        process=process,
        process_t=process_t,
        bridge_sigma=bridge_sigma,
    )
    if unconditional is None:
        return conditional, conditional, None
    return _apply_logit_cfg(conditional, unconditional, model.sampling_config), conditional, unconditional


def _field(
    model: SamplingModel,
    z: torch.Tensor,
    clean_embeddings: torch.Tensor,
    attention_mask: torch.Tensor,
    observed_mask: torch.Tensor,
    remaining_mask: torch.Tensor,
    gamma: torch.Tensor,
    process_t: torch.Tensor,
    generator: torch.Generator,
    prompt_condition_mask: Optional[torch.Tensor],
    promoted_generated_mask: Optional[torch.Tensor],
    process: str,
    bridge_sigma: float,
    t_eps: float = 1e-4,
) -> _FieldOutput:
    cfg = model.sampling_config
    time_t = _time_condition(model.config, gamma, process_t)
    target = _cfg_target(cfg) if _cfg_enabled(cfg) else "logits"
    if target == "logits":
        logits, cond_logits, uncond_logits = _guided_logits(
            model,
            z,
            attention_mask,
            time_t,
            observed_mask,
            generator,
            prompt_condition_mask,
            promoted_generated_mask,
            process=process,
            process_t=process_t,
            bridge_sigma=bridge_sigma,
        )
    else:
        cond_logits, uncond_logits = _cfg_logits_pair(
            model,
            z,
            attention_mask,
            time_t,
            observed_mask,
            generator,
            prompt_condition_mask,
            promoted_generated_mask,
            process=process,
            process_t=process_t,
            bridge_sigma=bridge_sigma,
        )
        logits = cond_logits

    x0_pred = clean_embeddings.clone()
    velocity = torch.zeros_like(z)
    if bool(remaining_mask.any().item()):
        denom = (1.0 - process_t.float()).clamp_min(float(t_eps)).reshape(-1, 1, 1).expand_as(z)
        if target == "velocity" and uncond_logits is not None:
            x0_cond = _posterior_mean_embedding_sampling(model, cond_logits[remaining_mask], dtype=z.dtype)
            x0_uncond = _posterior_mean_embedding_sampling(model, uncond_logits[remaining_mask], dtype=z.dtype)
            v_cond = (x0_cond.float() - z[remaining_mask].float()) / denom[remaining_mask]
            v_uncond = (x0_uncond.float() - z[remaining_mask].float()) / denom[remaining_mask]
            v_cfg = _apply_cfg(v_cond.to(dtype=z.dtype), v_uncond.to(dtype=z.dtype), cfg)
            velocity[remaining_mask] = v_cfg
            x0_pred[remaining_mask] = (z[remaining_mask].float() + denom[remaining_mask] * v_cfg.float()).to(dtype=z.dtype)
        else:
            x0_remaining = _posterior_mean_embedding_sampling(model, logits[remaining_mask], dtype=z.dtype)
            x0_pred[remaining_mask] = x0_remaining
            velocity[remaining_mask] = ((x0_remaining.float() - z[remaining_mask].float()) / denom[remaining_mask]).to(
                dtype=z.dtype
            )
    return _FieldOutput(
        logits=logits,
        conditional_logits=cond_logits,
        unconditional_logits=uncond_logits,
        x0_pred=x0_pred,
        velocity=velocity,
    )


def _bridge_noise_like(z, generator, bridge_noise_sampler=None):
    if bridge_noise_sampler is None:
        return torch.randn(z.shape, device=z.device, dtype=torch.float32, generator=generator)
    return bridge_noise_sampler.sample_like(z, generator).float()


def _sde_bridge_transition(z, x0_pred, t, t_next, noise, bridge_sigma):
    z32 = z.float()
    x032 = x0_pred.float()
    t = t.float().reshape(-1, 1, 1)
    t_next = t_next.float().reshape(-1, 1, 1)
    denom = (1.0 - t).clamp_min(1e-4)
    dt = (t_next - t).clamp_min(0.0)
    mean = z32 + dt * (x032 - z32) / denom
    variance = torch.clamp(dt * (1.0 - t_next) / denom, min=0.0)
    return mean + float(bridge_sigma) * torch.sqrt(variance) * noise.float()


def _sde_step(z, x0_pred, t, t_next, corrupt_mask, clean_embeddings, generator, bridge_sigma, bridge_noise_sampler=None):
    noise = _bridge_noise_like(z, generator, bridge_noise_sampler)
    updated = _sde_bridge_transition(z, x0_pred, t, t_next, noise, bridge_sigma)
    return torch.where(corrupt_mask.unsqueeze(-1), updated.to(dtype=z.dtype), clean_embeddings)


class BaseProposal:
    def propose(
        self,
        model: SamplingModel,
        current_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        valid_mask: torch.Tensor,
        remaining_mask: torch.Tensor,
        generator: torch.Generator,
        prompt_condition_mask: Optional[torch.Tensor] = None,
        promoted_generated_mask: Optional[torch.Tensor] = None,
        progress_factory=None,
    ) -> ProposalOutput:
        raise NotImplementedError


class SamplingProposal(BaseProposal):
    def propose(
        self,
        model,
        current_ids,
        attention_mask,
        valid_mask,
        remaining_mask,
        generator,
        prompt_condition_mask=None,
        promoted_generated_mask=None,
        progress_factory=None,
    ):
        cfg = model.sampling_config
        if str(cfg.stepper or "sde").lower() != "sde":
            raise ValueError("The camera-ready sampler supports only sampling.stepper=`sde`.")

        steps = max(1, int(cfg.steps))
        gamma_steps = gamma_trajectory(model.config, cfg, steps, model.device)
        clean_embeddings = _to_sampler_dtype(model, model.embed(current_ids))
        observed = _observed_mask(valid_mask, remaining_mask)
        process = normalize_forward_process(getattr(model.config, "forward_process", "brownian_bridge"))
        if process != "brownian_bridge":
            raise ValueError("Camera-ready sampling is implemented only for the brownian_bridge forward process.")
        bridge_sigma = float(
            getattr(model.config, "brownian_bridge_sigma", 1.0)
            if cfg.bridge_sigma == "from_config"
            else cfg.bridge_sigma
        )
        t0, _, _ = gamma_to_process_coefficients(
            gamma_steps[0].expand(current_ids.shape[0]), process, bridge_sigma=bridge_sigma
        )
        z = apply_forward_process(
            clean_embeddings,
            model.mask_embedding(),
            remaining_mask,
            t0,
            generator,
            process=process,
            bridge_sigma=bridge_sigma,
            bridge_noise_sampler=model.bridge_noise_sampler,
        )
        saved_sample = None
        confidence_step = max(0, min(steps - 1, int(float(cfg.confidence_quantile) * steps)))
        step_iter = range(steps)
        if progress_factory is not None:
            step_iter = progress_factory(step_iter, desc="SDE steps", leave=False)
        for idx in step_iter:
            gamma = gamma_steps[idx].expand(current_ids.shape[0])
            gamma_next = gamma_steps[idx + 1].expand(current_ids.shape[0])
            t, _, _ = gamma_to_process_coefficients(gamma, process, bridge_sigma=bridge_sigma)
            t_next, _, _ = gamma_to_process_coefficients(gamma_next, process, bridge_sigma=bridge_sigma)
            field = _field(
                model,
                z,
                clean_embeddings,
                attention_mask,
                observed,
                remaining_mask,
                gamma,
                t,
                generator,
                prompt_condition_mask,
                promoted_generated_mask,
                process,
                bridge_sigma,
            )
            if str(cfg.confidence_source).lower() == "quantile" and idx == confidence_step:
                saved_sample = sample_tokens(field.logits, cfg.token_sampler, generator)
            z = _sde_step(
                z,
                field.x0_pred,
                t,
                t_next,
                remaining_mask,
                clean_embeddings,
                generator,
                bridge_sigma,
                model.bridge_noise_sampler,
            )

        final_gamma = gamma_steps[-1].expand(current_ids.shape[0])
        final_t, _, _ = gamma_to_process_coefficients(final_gamma, process, bridge_sigma=bridge_sigma)
        final_time_t = _time_condition(model.config, final_gamma, final_t)
        if _cfg_enabled(cfg) and _cfg_target(cfg) == "logits":
            final_logits, _, _ = _guided_logits(
                model,
                z,
                attention_mask,
                final_time_t,
                observed,
                generator,
                prompt_condition_mask,
                promoted_generated_mask,
                process=process,
                process_t=final_t,
                bridge_sigma=bridge_sigma,
            )
        else:
            final_logits = model.forward_embeddings(z, attention_mask, final_time_t, observed)
        token_sample = sample_tokens(final_logits, cfg.token_sampler, generator)
        confidence_sample = saved_sample if str(cfg.confidence_source).lower() == "quantile" and saved_sample is not None else token_sample
        confidence = confidence_from_sample(confidence_sample, cfg.confidence.score)
        return ProposalOutput(
            token_sample.token_ids,
            confidence,
            logits=final_logits,
            aux={"sample": token_sample, "confidence_sample": confidence_sample},
        )


def build_proposal(config: SamplingExperimentConfig) -> BaseProposal:
    if str(config.stepper or "sde").lower() != "sde":
        raise ValueError("The camera-ready sampler supports only sampling.stepper=`sde`.")
    return SamplingProposal()
