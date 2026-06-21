"""MLFM CE + embedding MSE objective."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from mlfm.corruption import (
    apply_forward_process,
    build_valid_token_mask,
    gamma_to_process_coefficients,
    normalize_forward_process,
    sample_corruption_mask,
    sample_low_discrepancy_quantiles,
    sample_log_nsr_gamma,
    sample_mask_ratio,
)


def unwrap_backbone_helpers(backbone):
    """Return the original backbone object for custom helper methods."""
    backbone = backbone.module if hasattr(backbone, "module") else backbone
    return backbone._orig_mod if hasattr(backbone, "_orig_mod") else backbone


def _as_special_ids(backbone, config):
    backbone = unwrap_backbone_helpers(backbone)
    ids = set(getattr(backbone, "special_token_ids", []) or [])
    extra = getattr(config, "special_token_ids", None)
    if extra:
        if isinstance(extra, str):
            extra = [int(item.strip()) for item in extra.split(",") if item.strip()]
        ids.update(int(item) for item in extra)
    return sorted(ids)


def _eos_token_id(backbone, config) -> Optional[int]:
    backbone = unwrap_backbone_helpers(backbone)
    eos_id = getattr(config, "eos_token_id", None)
    if eos_id is not None:
        return int(eos_id)
    tokenizer = getattr(backbone, "tokenizer", None)
    if tokenizer is not None:
        eos_id = getattr(tokenizer, "eos_token_id", None)
        if eos_id is not None:
            return int(eos_id)
        pad_id = getattr(tokenizer, "pad_token_id", None)
        if pad_id is not None:
            return int(pad_id)
    return None


def _reduce_corrupted_ce(
    ce: torch.Tensor,
    corrupt_mask: torch.Tensor,
    valid_mask: torch.Tensor,
    mask_ratio: torch.Tensor,
    weighting: str,
) -> torch.Tensor:
    corrupt = corrupt_mask.to(ce.dtype)
    if weighting == "inverse_count":
        per_sample_sum = (ce * corrupt).sum(dim=1)
        counts = corrupt.sum(dim=1).clamp_min(1.0)
        has_corrupt = (corrupt.sum(dim=1) > 0).to(ce.dtype)
        return (per_sample_sum / counts * has_corrupt).sum() / has_corrupt.sum().clamp_min(1.0)
    if weighting == "inverse_p":
        p = mask_ratio.to(device=ce.device, dtype=ce.dtype).clamp_min(1e-6).reshape(-1, 1)
        weights = corrupt / p
        denom = valid_mask.to(ce.dtype).sum().clamp_min(1.0)
        return (ce * weights).sum() / denom
    if weighting == "none":
        return (ce * corrupt).sum() / corrupt.sum().clamp_min(1.0)
    raise ValueError(f"Unknown MLFM loss weighting: {weighting}")


def _reduce_sft_response_loss(
    values: torch.Tensor,
    corrupt_mask: torch.Tensor,
    response_mask: torch.Tensor,
    mask_ratio: torch.Tensor,
) -> torch.Tensor:
    """LLaDA-style SFT reduction over masked response tokens."""
    corrupt = corrupt_mask.to(values.dtype)
    response_lengths = response_mask.to(values.dtype).sum(dim=1).clamp_min(1.0).reshape(-1, 1)
    p = mask_ratio.to(device=values.device, dtype=values.dtype).clamp_min(1e-6).reshape(-1, 1)
    per_token = values * corrupt / p / response_lengths
    return per_token.sum() / max(int(values.shape[0]), 1)


def _posterior_mean_embedding(backbone_module, logits: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Return E[z_0 | z_t] by pushing predicted token probabilities through E."""
    input_embeddings = getattr(backbone_module, "input_embeddings", None)
    if input_embeddings is not None and hasattr(input_embeddings, "weight"):
        embedding_weight = input_embeddings.weight
    else:
        token_ids = torch.arange(logits.shape[-1], device=logits.device)
        embedding_weight = backbone_module.embed(token_ids)
    if embedding_weight.shape[0] != logits.shape[-1]:
        if embedding_weight.shape[0] < logits.shape[-1]:
            raise ValueError(
                "Input embedding table is smaller than the predicted logit vocabulary: "
                f"{embedding_weight.shape[0]} embeddings for {logits.shape[-1]} logits."
            )
        embedding_weight = embedding_weight[: logits.shape[-1]]
    probs = torch.softmax(logits.float(), dim=-1).to(dtype=embedding_weight.dtype)
    expected = torch.matmul(probs, embedding_weight.to(device=logits.device))
    return expected.to(dtype=dtype)


def _mean_corrupted(values: torch.Tensor, corrupt_mask: torch.Tensor) -> torch.Tensor:
    corrupt = corrupt_mask.to(values.dtype)
    return (values * corrupt).sum() / corrupt.sum().clamp_min(1.0)


def _is_sft_batch(batch: Dict[str, torch.Tensor]) -> bool:
    return "sft_response_mask" in batch or "prompt_lengths" in batch


def _sft_response_mask(batch: Dict[str, torch.Tensor], input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    if "sft_response_mask" in batch:
        return batch["sft_response_mask"].to(device=input_ids.device).bool() & attention_mask.bool()
    prompt_lengths = batch["prompt_lengths"].to(device=input_ids.device).long().clamp(min=0, max=input_ids.shape[1] - 1)
    positions = torch.arange(input_ids.shape[1], device=input_ids.device).reshape(1, -1)
    return (positions >= prompt_lengths.reshape(-1, 1)) & attention_mask.bool()


def compute_mlfm_loss(
    backbone,
    batch: Dict[str, torch.Tensor],
    config,
    generator: torch.Generator,
    bridge_noise_sampler=None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Compute CE plus optional posterior-mean embedding MSE on corrupted positions."""
    input_ids = batch["input_ids"].long()
    attention_mask = batch.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    device = input_ids.device
    is_sft = _is_sft_batch(batch)

    backbone_module = unwrap_backbone_helpers(backbone)
    clean_embeddings = backbone_module.embed(input_ids)
    batch_size = input_ids.shape[0]
    dtype = clean_embeddings.dtype
    noise_parameterization = str(getattr(config, "noise_parameterization", "log_nsr") or "log_nsr").lower()
    process = normalize_forward_process(getattr(config, "forward_process", "brownian_bridge"))
    if noise_parameterization != "log_nsr":
        raise ValueError("Camera-ready MLFM training requires noise_parameterization=`log_nsr`.")

    gamma = None
    signal_coeff = None
    noise_coeff = None
    use_low_discrepancy = bool(getattr(config, "use_low_discrepancy", False))
    gamma_quantiles = (
        sample_low_discrepancy_quantiles(generator, batch_size, device=device, dtype=torch.float32)
        if use_low_discrepancy
        else None
    )
    gamma = sample_log_nsr_gamma(
        config,
        generator,
        batch_size,
        device=device,
        dtype=torch.float32,
        quantiles=gamma_quantiles,
    )
    t, signal_coeff, noise_coeff = gamma_to_process_coefficients(
        gamma,
        process,
        bridge_sigma=float(getattr(config, "brownian_bridge_sigma", 1.0)),
    )
    t = t.to(device=device, dtype=torch.float32)
    signal_coeff = signal_coeff.to(device=device, dtype=torch.float32)
    noise_coeff = noise_coeff.to(device=device, dtype=torch.float32)

    mask_quantiles = (
        sample_low_discrepancy_quantiles(generator, batch_size, device=device, dtype=torch.float32)
        if use_low_discrepancy
        else None
    )
    mask_ratio = sample_mask_ratio(
        generator,
        batch_size,
        mode=getattr(config, "mask_ratio_sampler", "maskgit_cosine"),
        p_min=float(getattr(config, "mask_p_min", 0.05)),
        p_max=float(getattr(config, "mask_p_max", 1.0)),
        device=device,
        dtype=torch.float32,
        maskgit_cosine_power=float(getattr(config, "maskgit_cosine_power", 1.0)),
        quantiles=mask_quantiles,
    )

    if is_sft:
        valid_mask = _sft_response_mask(batch, input_ids, attention_mask)
        corrupt_mask = sample_corruption_mask(
            valid_mask,
            mask_ratio,
            generator=generator,
            guarantee_nonempty=bool(getattr(config, "mask_guarantee_nonempty", True)),
        )
        full_mask_prob = float(getattr(config, "sft_full_response_mask_prob", 0.0) or 0.0)
        sft_full_rows = torch.zeros((batch_size,), device=device, dtype=torch.bool)
        if full_mask_prob > 0.0:
            sft_full_rows = torch.rand((batch_size,), device=device, generator=generator) < full_mask_prob
            corrupt_mask = torch.where(sft_full_rows.reshape(-1, 1), valid_mask, corrupt_mask)
            mask_ratio = torch.where(sft_full_rows, torch.ones_like(mask_ratio), mask_ratio)
        observed_mask = attention_mask.bool() & ~corrupt_mask
    else:
        valid_mask = build_valid_token_mask(
            input_ids=input_ids,
            attention_mask=attention_mask,
            special_token_ids=_as_special_ids(backbone, config),
        )
        corrupt_mask = sample_corruption_mask(
            valid_mask,
            mask_ratio,
            generator=generator,
            guarantee_nonempty=bool(getattr(config, "mask_guarantee_nonempty", True)),
        )
        observed_mask = valid_mask & ~corrupt_mask

    corrupted_embeddings = apply_forward_process(
        clean_embeddings=clean_embeddings,
        mask_embedding=backbone_module.mask_embedding(),
        corrupt_mask=corrupt_mask,
        t=t,
        generator=generator,
        process=process,
        bridge_sigma=float(getattr(config, "brownian_bridge_sigma", 1.0)),
        bridge_noise_sampler=bridge_noise_sampler,
    )

    time_conditioning = str(getattr(config, "time_conditioning", "t") or "t").lower()
    conditioning = gamma if (time_conditioning == "gamma" and gamma is not None) else t
    if hasattr(backbone, "module") or hasattr(backbone, "_orig_mod"):
        logits = backbone(
            inputs_embeds=corrupted_embeddings,
            attention_mask=attention_mask,
            t=conditioning,
            observed_mask=observed_mask,
        )
    else:
        logits = backbone.forward_from_embeddings(
            inputs_embeds=corrupted_embeddings,
            attention_mask=attention_mask,
            t=conditioning,
            observed_mask=observed_mask,
        )
    ce = F.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]),
        input_ids.reshape(-1),
        reduction="none",
    ).reshape_as(input_ids)
    if is_sft:
        ce_loss = _reduce_sft_response_loss(
            ce,
            corrupt_mask=corrupt_mask,
            response_mask=valid_mask,
            mask_ratio=mask_ratio,
        )
    else:
        ce_loss = _reduce_corrupted_ce(
            ce,
            corrupt_mask=corrupt_mask,
            valid_mask=valid_mask,
            mask_ratio=mask_ratio,
            weighting=getattr(config, "mlfm_loss_weighting", "inverse_count"),
        )
    lambda_mse = float(getattr(config, "lambda_mse", 0.0) or 0.0)
    if lambda_mse != 0.0 and bool(corrupt_mask.any().item()):
        flat_corrupt = corrupt_mask.reshape(-1)
        selected_logits = logits.reshape(-1, logits.shape[-1])[flat_corrupt]
        selected_clean = clean_embeddings.detach().reshape(-1, clean_embeddings.shape[-1])[flat_corrupt]
        selected_pred = _posterior_mean_embedding(backbone_module, selected_logits, dtype=clean_embeddings.dtype)
        selected_mse = (selected_pred.float() - selected_clean.float()).pow(2).mean(dim=-1)
        mse_per_token = logits.new_zeros(input_ids.numel(), dtype=torch.float32)
        mse_per_token[flat_corrupt] = selected_mse
        mse_per_token = mse_per_token.reshape_as(input_ids)
        if is_sft:
            mse_loss = _reduce_sft_response_loss(
                mse_per_token,
                corrupt_mask=corrupt_mask,
                response_mask=valid_mask,
                mask_ratio=mask_ratio,
            )
        else:
            mse_loss = _reduce_corrupted_ce(
                mse_per_token,
                corrupt_mask=corrupt_mask,
                valid_mask=valid_mask,
                mask_ratio=mask_ratio,
                weighting=getattr(config, "mlfm_loss_weighting", "inverse_count"),
            )
    else:
        mse_per_token = logits.new_zeros(input_ids.shape, dtype=torch.float32)
        mse_loss = ce_loss.new_zeros(())
    loss = ce_loss + lambda_mse * mse_loss
    total_tokens = torch.as_tensor(input_ids.numel(), device=device, dtype=loss.dtype)
    valid_tokens = valid_mask.sum().to(loss.dtype)
    corrupt_tokens = corrupt_mask.sum().to(loss.dtype)
    probs = torch.softmax(logits.detach().float(), dim=-1)
    pred_ids = probs.argmax(dim=-1)
    correct = pred_ids.eq(input_ids)
    target_confidence = probs.gather(-1, input_ids.unsqueeze(-1)).squeeze(-1)
    entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=-1)
    corrupt_float = corrupt_mask.float()
    per_sample_corrupt = corrupt_float.sum(dim=1).clamp_min(1.0)
    per_sample_ce = (ce.detach().float() * corrupt_float).sum(dim=1) / per_sample_corrupt
    eos_token_id = _eos_token_id(backbone_module, config)
    no_eos_corrupt_mask = corrupt_mask if eos_token_id is None else corrupt_mask & input_ids.ne(int(eos_token_id))
    no_eos_corrupt_float = no_eos_corrupt_mask.float()
    per_sample_non_eos_corrupt = no_eos_corrupt_float.sum(dim=1)
    per_sample_ce_no_eos = (
        (ce.detach().float() * no_eos_corrupt_float).sum(dim=1)
        / per_sample_non_eos_corrupt.clamp_min(1.0)
    )
    per_sample_ce_no_eos = torch.where(
        per_sample_non_eos_corrupt > 0,
        per_sample_ce_no_eos,
        torch.full_like(per_sample_ce_no_eos, float("nan")),
    )
    per_sample_mse = (mse_per_token.detach().float() * corrupt_float).sum(dim=1) / per_sample_corrupt
    per_sample_acc = (correct.detach().float() * corrupt_float).sum(dim=1) / per_sample_corrupt
    per_sample_confidence = (target_confidence.detach().float() * corrupt_float).sum(dim=1) / per_sample_corrupt
    per_sample_entropy = (entropy.detach().float() * corrupt_float).sum(dim=1) / per_sample_corrupt
    metrics = {
        "loss": loss.detach(),
        "ce_loss": ce_loss.detach(),
        "mse_loss": mse_loss.detach(),
        "mse_weighted_loss": (mse_loss * lambda_mse).detach(),
        "lambda_mse": torch.as_tensor(lambda_mse, device=device, dtype=loss.dtype),
        "corrupt_fraction": (corrupt_tokens / valid_tokens.clamp_min(1.0)).detach(),
        "total_tokens": total_tokens.detach(),
        "corrupt_tokens": corrupt_tokens.detach(),
        "valid_tokens": valid_tokens.detach(),
        "mean_t": t.float().mean().detach(),
        "mean_mask_ratio": mask_ratio.float().mean().detach(),
        "token_acc": (correct.float() * corrupt_mask.float()).sum().detach() / corrupt_tokens.clamp_min(1.0),
        "confidence": _mean_corrupted(target_confidence, corrupt_mask).detach(),
        "entropy": _mean_corrupted(entropy, corrupt_mask).detach(),
        "sample_mask_ratio": mask_ratio.detach().float(),
        "sample_ce": per_sample_ce.detach(),
        "sample_ce_no_eos": per_sample_ce_no_eos.detach(),
        "sample_mse": per_sample_mse.detach(),
        "sample_token_acc": per_sample_acc.detach(),
        "sample_error": (1.0 - per_sample_acc).detach(),
        "sample_confidence": per_sample_confidence.detach(),
        "sample_entropy": per_sample_entropy.detach(),
        "sample_corrupt_tokens": corrupt_float.sum(dim=1).detach(),
        "sample_corrupt_non_eos_tokens": per_sample_non_eos_corrupt.detach(),
        "sample_valid_tokens": valid_mask.float().sum(dim=1).detach(),
        "is_sft_batch": torch.as_tensor(1.0 if is_sft else 0.0, device=device, dtype=loss.dtype),
        "packed_batch_fraction": torch.as_tensor(0.0 if is_sft else 1.0, device=device, dtype=loss.dtype),
    }
    if is_sft:
        source_type = batch.get("sft_source_type")
        if source_type is not None:
            source_type = source_type.to(device=device).long()
            metrics.update(
                {
                    "sft_general_fraction": (source_type == 0).float().mean().detach(),
                    "sft_math_fraction": (source_type == 1).float().mean().detach(),
                    "sft_code_fraction": (source_type == 2).float().mean().detach(),
                }
            )
        else:
            metrics.update(
                {
                    "sft_general_fraction": torch.as_tensor(0.0, device=device, dtype=loss.dtype),
                    "sft_math_fraction": torch.as_tensor(0.0, device=device, dtype=loss.dtype),
                    "sft_code_fraction": torch.as_tensor(0.0, device=device, dtype=loss.dtype),
                }
            )
        metrics.update(
            {
                "sft_batch_fraction": torch.as_tensor(1.0, device=device, dtype=loss.dtype),
                "sft_full_response_fraction": sft_full_rows.float().mean().detach(),
                "sft_response_tokens": valid_mask.sum().to(loss.dtype).detach(),
                "sft_ce_loss": ce_loss.detach(),
            }
        )
    else:
        metrics.update(
            {
                "sft_batch_fraction": torch.as_tensor(0.0, device=device, dtype=loss.dtype),
                "sft_full_response_fraction": torch.as_tensor(0.0, device=device, dtype=loss.dtype),
                "sft_response_tokens": torch.as_tensor(0.0, device=device, dtype=loss.dtype),
                "sft_general_fraction": torch.as_tensor(0.0, device=device, dtype=loss.dtype),
                "sft_math_fraction": torch.as_tensor(0.0, device=device, dtype=loss.dtype),
                "sft_code_fraction": torch.as_tensor(0.0, device=device, dtype=loss.dtype),
                "packed_ce_loss": ce_loss.detach(),
            }
        )
    if "source_id" in batch:
        metrics["sample_source_id"] = batch["source_id"].detach().float()
    if gamma is not None:
        metrics.update(
            {
                "mean_gamma": gamma.float().mean().detach(),
                "mean_alpha": signal_coeff.float().mean().detach(),
                "mean_sigma": noise_coeff.float().mean().detach(),
                "sample_gamma": gamma.detach().float(),
            }
        )
    return loss, metrics
