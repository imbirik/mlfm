"""Validation helpers for MLFM."""

from __future__ import annotations

import json
import math
import os
import re
from typing import Dict, Optional

import torch

from mlfm.corruption import (
    apply_forward_process,
    build_valid_token_mask,
    gamma_sampling_steps,
    gamma_to_process_coefficients,
    normalize_forward_process,
    sample_corruption_mask,
    sample_mask_ratio,
)
from mlfm.train_step import (
    _as_special_ids,
    _posterior_mean_embedding,
    compute_mlfm_loss,
    unwrap_backbone_helpers,
)
from utils.train_utils import autocast_context


SFT_SOURCE_TYPE_NAMES = {
    0: "general",
    1: "math",
    2: "code",
}

_GSM8K_INTEGER_RE = re.compile(r"[-+]?\d[\d,]*")
_GSM8K_STRICT_RE = re.compile(r"#{3,4}\s*([-+]?\d[\d,]*)")


def _chat_prompt(prompt: str) -> str:
    return f"USER:\n{str(prompt).strip()}\nASSISTANT:\n"


def move_batch_to_device(batch, device: torch.device):
    result = {}
    for key, value in batch.items():
        result[key] = value.to(device, non_blocking=True) if torch.is_tensor(value) else value
    return result


def _truncate_text(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."


def _append_generation_ppl_text(texts: Optional[list], text: str, limit: int):
    if texts is None:
        return
    if limit > 0 and len(texts) >= limit:
        return
    if isinstance(text, str) and text.strip():
        texts.append(text)


def _decode_row(tokenizer, token_ids: torch.Tensor, attention_mask: Optional[torch.Tensor], max_chars: int) -> str:
    ids = token_ids.detach().cpu()
    if attention_mask is not None:
        length = int(attention_mask.detach().long().sum().cpu().item())
        ids = ids[:length]
    id_list = ids.tolist()
    try:
        text = tokenizer.decode(id_list, skip_special_tokens=True)
    except Exception:
        text = " ".join(str(token_id) for token_id in id_list)
    return _truncate_text(text, max_chars)


def _decode_masked_row(
    tokenizer,
    token_ids: torch.Tensor,
    corrupt_mask: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    max_chars: int,
) -> str:
    ids = token_ids.detach().cpu()
    mask = corrupt_mask.detach().bool().cpu()
    if attention_mask is not None:
        length = int(attention_mask.detach().long().sum().cpu().item())
        ids = ids[:length]
        mask = mask[:length]

    parts = []
    buffer = []

    def flush_buffer():
        if not buffer:
            return
        try:
            text = tokenizer.decode(buffer, skip_special_tokens=True)
        except Exception:
            text = " ".join(str(token_id) for token_id in buffer)
        if text:
            parts.append(text)
        buffer.clear()

    for token_id, is_masked in zip(ids.tolist(), mask.tolist()):
        if is_masked:
            flush_buffer()
            parts.append("[MASK]")
        else:
            buffer.append(int(token_id))
    flush_buffer()
    return _truncate_text(" ".join(part for part in parts if part), max_chars)


def _tokenizer_ids(tokenizer, text: str) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=False)
    if isinstance(encoded, dict):
        return list(encoded.get("input_ids", []))
    return list(getattr(encoded, "input_ids", []))


def _decode_token_ids(tokenizer, token_ids, max_chars: int) -> str:
    try:
        text = tokenizer.decode([int(token_id) for token_id in token_ids], skip_special_tokens=True)
    except Exception:
        text = " ".join(str(int(token_id)) for token_id in token_ids)
    return _truncate_text(text, max_chars)


def _mask_token_id(backbone_module, config) -> int:
    mask_id = getattr(backbone_module, "mask_token_id", None)
    if mask_id is None:
        mask_id = getattr(config, "mask_token_id", None)
    if mask_id is None:
        raise ValueError("Generation validation requires `mask_token_id` on the backbone or config.")
    return int(mask_id)


def _eos_token_id(backbone_module, config) -> Optional[int]:
    eos_id = getattr(config, "eos_token_id", None)
    if eos_id is not None:
        return int(eos_id)
    tokenizer = getattr(backbone_module, "tokenizer", None)
    if tokenizer is not None:
        eos_id = getattr(tokenizer, "eos_token_id", None)
        if eos_id is not None:
            return int(eos_id)
        pad_id = getattr(tokenizer, "pad_token_id", None)
        if pad_id is not None:
            return int(pad_id)
    return None


def _resolve_path(path: str) -> str:
    if os.path.isabs(path) and os.path.exists(path):
        return path
    candidates = [
        path,
        os.path.join(os.getcwd(), path),
        os.path.join(os.path.dirname(__file__), "..", "..", path),
    ]
    for candidate in candidates:
        candidate = os.path.abspath(candidate)
        if os.path.exists(candidate):
            return candidate
    return os.path.abspath(path)


def _load_gsm8k_records(path: str) -> list[dict]:
    records = []
    resolved = _resolve_path(path)
    if not os.path.exists(resolved):
        return records
    with open(resolved, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            question = str(data.get("question", "")).strip()
            answer = str(data.get("answer", "")).strip()
            if question:
                records.append({"question": question, "answer": answer})
    return records


def _clean_prompt_text(text: str) -> str:
    return str(text).replace("\r\n", "\n").replace("\r", "\n").strip()


def _canonical_integer(text: Optional[str]) -> Optional[str]:
    if text is None:
        return None
    value = str(text).replace(",", "").strip()
    try:
        return str(int(value))
    except ValueError:
        return None


def _extract_gsm8k_strict_answer(text: str) -> Optional[str]:
    match = _GSM8K_STRICT_RE.search(str(text))
    return _canonical_integer(match.group(1)) if match else None


def _extract_gsm8k_flexible_answer(text: str) -> Optional[str]:
    matches = _GSM8K_INTEGER_RE.findall(str(text))
    return _canonical_integer(matches[-1]) if matches else None


def _extract_gsm8k_gold_answer(answer: str) -> Optional[str]:
    return _extract_gsm8k_strict_answer(answer) or _extract_gsm8k_flexible_answer(answer)


@torch.no_grad()
def evaluate_corrupted_ce(
    backbone,
    dataloader,
    config,
    generator: torch.Generator,
    device: torch.device,
    max_batches: Optional[int] = None,
    amp_dtype=None,
    bridge_noise_sampler=None,
) -> Dict[str, float]:
    """Average corrupted-token CE/PPL over a validation loader."""
    was_training = backbone.training
    backbone.eval()
    losses = []
    ce_losses = []
    mse_losses = []
    corrupt_fracs = []
    source_losses = {}

    for batch_idx, batch in enumerate(dataloader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        batch = move_batch_to_device(batch, device)
        with autocast_context(device, amp_dtype):
            loss, metrics = compute_mlfm_loss(backbone, batch, config, generator, bridge_noise_sampler=bridge_noise_sampler)
        losses.append(loss.detach().float())
        ce_losses.append(metrics.get("ce_loss", loss).detach().float())
        if "mse_loss" in metrics:
            mse_losses.append(metrics["mse_loss"].detach().float())
        corrupt_fracs.append(metrics["corrupt_fraction"].detach().float())

        if "source_id" in batch:
            for source_id in torch.unique(batch["source_id"]).tolist():
                keep = batch["source_id"] == int(source_id)
                if not keep.any():
                    continue
                sub_batch = {key: (value[keep] if torch.is_tensor(value) and value.shape[:1] == keep.shape else value) for key, value in batch.items()}
                with autocast_context(device, amp_dtype):
                    sub_loss, sub_metrics = compute_mlfm_loss(
                        backbone,
                        sub_batch,
                        config,
                        generator,
                        bridge_noise_sampler=bridge_noise_sampler,
                    )
                source_losses.setdefault(int(source_id), []).append(sub_metrics.get("ce_loss", sub_loss).detach().float())

    if was_training:
        backbone.train()
    if not losses:
        return {"ce": float("nan"), "ppl": float("nan")}
    loss_value = torch.stack(losses).mean().item()
    ce = torch.stack(ce_losses).mean().item()
    result = {
        "loss": loss_value,
        "ce": ce,
        "ppl": math.exp(min(ce, 20.0)),
        "corrupt_fraction": torch.stack(corrupt_fracs).mean().item(),
    }
    if mse_losses:
        result["mse_loss"] = torch.stack(mse_losses).mean().item()
    for source_id, values in source_losses.items():
        source_ce = torch.stack(values).mean().item()
        result[f"source_{source_id}_ce"] = source_ce
        result[f"source_{source_id}_ppl"] = math.exp(min(source_ce, 20.0))
    return result


def _generation_steps(config, seq_len: int) -> int:
    configured = int(getattr(config, "val_generation_steps", 0) or 0)
    if configured > 0:
        return configured
    cap = int(getattr(config, "val_generation_steps_cap", 128) or 128)
    return max(1, min(int(seq_len), cap))


def _generation_target_length(config) -> int:
    length = int(getattr(config, "max_length", 0) or 0)
    if length <= 0:
        raise ValueError("Generation validation requires config.max_length > 0.")
    return length


def _prompt_prefix_and_mask_span(prefix_ids: list[int], target_length: int) -> tuple[list[int], int]:
    target_length = max(1, int(target_length))
    if len(prefix_ids) >= target_length:
        prefix_ids = prefix_ids[-max(1, target_length - 1):]
    span_len = max(1, target_length - len(prefix_ids))
    return prefix_ids, span_len


def _embedding_sde_step(
    z: torch.Tensor,
    x0_pred: torch.Tensor,
    t: torch.Tensor,
    t_next: torch.Tensor,
    corrupt_mask: torch.Tensor,
    clean_embeddings: torch.Tensor,
    generator: torch.Generator,
    bridge_sigma: float = 1.0,
    bridge_noise_sampler=None,
    t_eps: float = 1e-4,
) -> torch.Tensor:
    """Stochastic Brownian-bridge step conditioned on the predicted clean embedding."""
    compute_dtype = torch.float32
    z_values = z.to(dtype=compute_dtype)
    x0_values = x0_pred.to(dtype=compute_dtype)
    t = t.to(device=z.device, dtype=compute_dtype).reshape(-1, 1, 1)
    t_next = t_next.to(device=z.device, dtype=compute_dtype).reshape(-1, 1, 1)
    denom = (1.0 - t).clamp_min(float(t_eps))
    dt = (t_next - t).clamp_min(0.0)
    mean = z_values + dt * (x0_values - z_values) / denom
    variance = torch.clamp(dt * (1.0 - t_next) / denom, min=0.0)
    if bridge_noise_sampler is None:
        noise = torch.randn(z.shape, device=z.device, dtype=compute_dtype, generator=generator)
    else:
        noise = bridge_noise_sampler.sample_like(z, generator)
    updated = mean + float(bridge_sigma) * torch.sqrt(variance) * noise
    updated = updated.to(dtype=z.dtype)
    return torch.where(corrupt_mask.to(device=z.device).bool().unsqueeze(-1), updated, clean_embeddings)


def _model_logits_from_embeddings(backbone, current_embeddings, attention_mask, t, observed_mask, amp_dtype):
    with autocast_context(current_embeddings.device, amp_dtype):
        if hasattr(backbone, "module") or hasattr(backbone, "_orig_mod"):
            return backbone(
                inputs_embeds=current_embeddings,
                attention_mask=attention_mask,
                t=t,
                observed_mask=observed_mask,
            )
        return backbone.forward_from_embeddings(current_embeddings, attention_mask, t=t, observed_mask=observed_mask)


def _validate_base_generation_sampler(config) -> str:
    sampler = str(getattr(config, "generation_sampler", "sde") or "sde").lower()
    if sampler != "sde":
        raise ValueError("The camera-ready validation sampler supports only generation_sampler=`sde`.")
    return sampler


def _run_base_generation_pass(
    backbone,
    backbone_module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    valid_mask: torch.Tensor,
    corrupt_mask: torch.Tensor,
    config,
    generator,
    device,
    steps: int,
    amp_dtype=None,
    bridge_noise_sampler=None,
    progress_callback=None,
):
    """Run one full SDE generation trajectory over `corrupt_mask`."""
    batch_size = int(input_ids.shape[0])
    clean_embeddings = backbone_module.embed(input_ids)
    observed_mask = valid_mask & ~corrupt_mask
    noise_parameterization = str(getattr(config, "noise_parameterization", "log_nsr") or "log_nsr").lower()
    if noise_parameterization != "log_nsr":
        raise ValueError("Camera-ready validation generation requires noise_parameterization=`log_nsr`.")
    process = normalize_forward_process(getattr(config, "forward_process", "brownian_bridge"))
    bridge_sigma = float(getattr(config, "brownian_bridge_sigma", 1.0))
    _validate_base_generation_sampler(config)
    gamma_steps = gamma_sampling_steps(config, steps, device=device, dtype=torch.float32)
    gamma0 = torch.full((batch_size,), float(gamma_steps[0].item()), device=device, dtype=torch.float32)
    t0 = gamma_to_process_coefficients(gamma0, process, bridge_sigma=bridge_sigma)[0]
    t0 = t0.to(dtype=torch.float32, device=device)

    current_embeddings = apply_forward_process(
        clean_embeddings,
        backbone_module.mask_embedding(),
        corrupt_mask,
        t0,
        generator,
        process=process,
        bridge_sigma=bridge_sigma,
        bridge_noise_sampler=bridge_noise_sampler,
    )

    for step in range(steps):
        gamma_step = torch.full((batch_size,), float(gamma_steps[step].item()), device=device, dtype=torch.float32)
        gamma_next = torch.full((batch_size,), float(gamma_steps[step + 1].item()), device=device, dtype=torch.float32)
        process_t = gamma_to_process_coefficients(gamma_step, process, bridge_sigma=bridge_sigma)[0]
        process_t_next = gamma_to_process_coefficients(gamma_next, process, bridge_sigma=bridge_sigma)[0]
        process_t = process_t.to(dtype=torch.float32, device=device)
        process_t_next = process_t_next.to(dtype=torch.float32, device=device)
        time_conditioning = str(getattr(config, "time_conditioning", "gamma") or "gamma").lower()
        t = gamma_step if time_conditioning == "gamma" else process_t
        ode_t = process_t
        ode_t_next = process_t_next

        logits = _model_logits_from_embeddings(backbone, current_embeddings, attention_mask, t, observed_mask, amp_dtype)
        x0_pred = clean_embeddings.clone()
        if corrupt_mask.any():
            x0_pred[corrupt_mask] = _posterior_mean_embedding(
                backbone_module,
                logits[corrupt_mask],
                dtype=current_embeddings.dtype,
            )
        current_embeddings = _embedding_sde_step(
            current_embeddings,
            x0_pred,
            ode_t,
            ode_t_next,
            corrupt_mask,
            clean_embeddings,
            generator=generator,
            bridge_sigma=bridge_sigma,
            bridge_noise_sampler=bridge_noise_sampler,
        )
        if progress_callback is not None:
            progress_callback(step + 1, steps)

    final_gamma = torch.full((batch_size,), float(gamma_steps[-1].item()), device=device, dtype=torch.float32)
    final_process_t = gamma_to_process_coefficients(final_gamma, process, bridge_sigma=bridge_sigma)[0]
    final_process_t = final_process_t.to(dtype=torch.float32, device=device)
    time_conditioning = str(getattr(config, "time_conditioning", "gamma") or "gamma").lower()
    final_t = final_gamma if time_conditioning == "gamma" else final_process_t
    final_logits = _model_logits_from_embeddings(backbone, current_embeddings, attention_mask, final_t, observed_mask, amp_dtype)
    pred_ids = backbone_module.decode_logits(final_logits)
    generated = torch.where(corrupt_mask, pred_ids, input_ids)
    return generated, final_logits


@torch.no_grad()
def reverse_denoise_generate(
    backbone,
    batch,
    config,
    generator,
    device,
    num_steps: Optional[int] = None,
    amp_dtype=None,
    return_corrupt_mask: bool = False,
    forced_corrupt_mask: Optional[torch.Tensor] = None,
    forced_valid_mask: Optional[torch.Tensor] = None,
    bridge_noise_sampler=None,
):
    """Tiny iterative validation generation loop for smoke testing."""
    batch = move_batch_to_device(batch, device)
    backbone_module = unwrap_backbone_helpers(backbone)
    input_ids = batch["input_ids"].long()
    attention_mask = batch.get("attention_mask", torch.ones_like(input_ids))
    if forced_valid_mask is None:
        valid_mask = build_valid_token_mask(input_ids, attention_mask, _as_special_ids(backbone, config))
    else:
        valid_mask = forced_valid_mask.to(device=device).bool() & attention_mask.to(device=device).bool()
    batch_size = input_ids.shape[0]
    if forced_corrupt_mask is None:
        mask_ratio = sample_mask_ratio(
            generator,
            batch_size,
            mode=getattr(config, "mask_ratio_sampler", "maskgit_cosine"),
            p_min=float(getattr(config, "mask_p_min", 0.05)),
            p_max=float(getattr(config, "mask_p_max", 1.0)),
            device=device,
            maskgit_cosine_power=float(getattr(config, "maskgit_cosine_power", 1.0)),
        )
        corrupt_mask = sample_corruption_mask(valid_mask, mask_ratio, generator)
    else:
        corrupt_mask = forced_corrupt_mask.to(device=device).bool() & valid_mask
    steps = max(1, int(num_steps or _generation_steps(config, input_ids.shape[1])))
    generated, _ = _run_base_generation_pass(
        backbone,
        backbone_module,
        input_ids,
        attention_mask,
        valid_mask,
        corrupt_mask,
        config,
        generator,
        device,
        steps,
        amp_dtype=amp_dtype,
        bridge_noise_sampler=bridge_noise_sampler,
    )
    if return_corrupt_mask:
        return generated, corrupt_mask
    return generated


@torch.no_grad()
def evaluate_generation_smoke(
    backbone,
    dataloader,
    config,
    generator,
    device,
    max_samples: int = 64,
    amp_dtype=None,
    sample_rows: Optional[list] = None,
    sample_limit: int = 8,
    ppl_texts: Optional[list] = None,
    ppl_limit: int = 64,
    bridge_noise_sampler=None,
) -> Dict[str, float]:
    """Run a bounded reverse-denoising smoke validation and report token accuracy."""
    was_training = backbone.training
    backbone.eval()
    total_matches = torch.zeros((), device=device, dtype=torch.float32)
    total_tokens = torch.zeros((), device=device, dtype=torch.float32)
    total_samples = 0

    for batch in dataloader:
        if total_samples >= max_samples:
            break
        batch = move_batch_to_device(batch, device)
        remaining = max_samples - total_samples
        if batch["input_ids"].shape[0] > remaining:
            batch = {
                key: (value[:remaining] if torch.is_tensor(value) and value.shape[:1] == batch["input_ids"].shape[:1] else value)
                for key, value in batch.items()
            }
        generated, corrupt_mask = reverse_denoise_generate(
            backbone,
            batch,
            config,
            generator,
            device,
            num_steps=_generation_steps(config, batch["input_ids"].shape[1]),
            amp_dtype=amp_dtype,
            return_corrupt_mask=True,
            bridge_noise_sampler=bridge_noise_sampler,
        )
        attention_mask = batch.get("attention_mask", torch.ones_like(batch["input_ids"]))
        valid_mask = build_valid_token_mask(batch["input_ids"].long(), attention_mask, _as_special_ids(backbone, config))
        score_mask = corrupt_mask & valid_mask
        matches = (generated == batch["input_ids"].long()) & score_mask
        needs_sample_rows = sample_rows is not None and len(sample_rows) < sample_limit
        needs_ppl_texts = ppl_texts is not None and (ppl_limit <= 0 or len(ppl_texts) < ppl_limit)
        if needs_sample_rows or needs_ppl_texts:
            backbone_module = unwrap_backbone_helpers(backbone)
            tokenizer = backbone_module.tokenizer
            max_chars = int(getattr(config, "wandb_generation_max_chars", 2000) or 2000)
            batch_size = int(batch["input_ids"].shape[0])
            attention_mask_for_decode = batch.get("attention_mask")
            for row_idx in range(batch_size):
                if not needs_sample_rows and not needs_ppl_texts:
                    break
                if needs_sample_rows and len(sample_rows) < sample_limit:
                    row_valid = score_mask[row_idx].float().sum().clamp_min(1.0)
                    row_acc = (matches[row_idx].float().sum() / row_valid).detach().float().cpu().item()
                    sample_rows.append(
                        {
                            "masked_token_acc": float(row_acc),
                            "masked_tokens": float(score_mask[row_idx].float().sum().detach().cpu().item()),
                            "target": _decode_row(
                                tokenizer,
                                batch["input_ids"][row_idx].long(),
                                attention_mask_for_decode[row_idx] if attention_mask_for_decode is not None else None,
                                max_chars=max_chars,
                            ),
                            "masked_target": _decode_masked_row(
                                tokenizer,
                                batch["input_ids"][row_idx].long(),
                                corrupt_mask[row_idx],
                                attention_mask_for_decode[row_idx] if attention_mask_for_decode is not None else None,
                                max_chars=max_chars,
                            ),
                            "sampled": _decode_row(
                                tokenizer,
                                generated[row_idx].long(),
                                attention_mask_for_decode[row_idx] if attention_mask_for_decode is not None else None,
                                max_chars=max_chars,
                            ),
                        }
                    )
                if needs_ppl_texts:
                    _append_generation_ppl_text(
                        ppl_texts,
                        _decode_row(
                            tokenizer,
                            generated[row_idx].long(),
                            attention_mask_for_decode[row_idx] if attention_mask_for_decode is not None else None,
                            max_chars=0,
                        ),
                        ppl_limit,
                    )
                needs_sample_rows = sample_rows is not None and len(sample_rows) < sample_limit
                needs_ppl_texts = ppl_texts is not None and (ppl_limit <= 0 or len(ppl_texts) < ppl_limit)
        total_matches = total_matches + matches.float().sum()
        total_tokens = total_tokens + score_mask.float().sum()
        total_samples += int(batch["input_ids"].shape[0])

    if was_training:
        backbone.train()
    if total_samples == 0:
        return {
            "generation_token_accuracy": float("nan"),
            "generation_masked_token_accuracy": float("nan"),
            "generation_samples": 0.0,
        }
    masked_accuracy = (total_matches / total_tokens.clamp_min(1.0)).item()
    return {
        "generation_token_accuracy": masked_accuracy,
        "generation_masked_token_accuracy": masked_accuracy,
        "generation_samples": float(total_samples),
    }


@torch.no_grad()
def evaluate_sft_prompt_conditional_generations(
    backbone,
    dataloader,
    config,
    generator,
    device,
    amp_dtype=None,
    sample_rows: Optional[list] = None,
    sample_limit_per_type: int = 2,
    max_batches: int = 16,
    ppl_texts: Optional[list] = None,
    ppl_limit: int = 64,
    bridge_noise_sampler=None,
) -> Dict[str, float]:
    """Generate response continuations from SFT prompt/response batches."""
    per_type_limit = max(0, int(sample_limit_per_type or 0))
    max_batches = max(1, int(max_batches or 1))
    if per_type_limit <= 0:
        return {"sft_prompt_generation_samples": 0.0}

    was_training = backbone.training
    backbone.eval()
    backbone_module = unwrap_backbone_helpers(backbone)
    tokenizer = backbone_module.tokenizer
    mask_id = _mask_token_id(backbone_module, config)
    max_chars = int(getattr(config, "wandb_generation_max_chars", 2000) or 2000)
    counts = {name: 0 for name in SFT_SOURCE_TYPE_NAMES.values()}
    total_samples = 0

    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= max_batches:
            break
        if "prompt_lengths" not in batch or "sft_source_type" not in batch:
            continue
        if all(count >= per_type_limit for count in counts.values()):
            break
        batch = move_batch_to_device(batch, device)
        input_ids = batch["input_ids"].long()
        attention_mask = batch.get("attention_mask", torch.ones_like(input_ids)).long()
        prompt_lengths = batch["prompt_lengths"].long().clamp(min=0, max=input_ids.shape[1])
        true_lengths = batch.get("true_lengths", attention_mask.long().sum(dim=1)).long().clamp(min=1, max=input_ids.shape[1])
        source_types = batch["sft_source_type"].long()
        selected_indices = []
        selected_meta = []
        for row_idx in range(int(input_ids.shape[0])):
            source_name = SFT_SOURCE_TYPE_NAMES.get(int(source_types[row_idx].item()), f"source_{int(source_types[row_idx].item())}")
            if source_name in counts and counts[source_name] >= per_type_limit:
                continue
            prompt_len = int(prompt_lengths[row_idx].item())
            if prompt_len >= int(input_ids.shape[1]):
                continue
            selected_indices.append(row_idx)
            selected_meta.append(
                {
                    "source_type": source_name,
                    "prompt_len": prompt_len,
                    "true_len": int(true_lengths[row_idx].item()),
                }
            )
            counts.setdefault(source_name, 0)
            counts[source_name] += 1
            if all(count >= per_type_limit for count in counts.values()):
                break
        if not selected_indices:
            continue

        index = torch.tensor(selected_indices, device=device, dtype=torch.long)
        original_ids = input_ids.index_select(0, index)
        # SFT qualitative validation is prompt-conditioned generation to max_length:
        # keep the prompt observed and start every response/EOS-padding position masked.
        selected_attention = torch.ones_like(original_ids, dtype=torch.long)
        selected_prompt_lengths = prompt_lengths.index_select(0, index)
        positions = torch.arange(original_ids.shape[1], device=device).unsqueeze(0)
        response_mask = positions >= selected_prompt_lengths.unsqueeze(1)
        generation_ids = torch.where(response_mask, torch.full_like(original_ids, mask_id), original_ids)
        generation_batch = {
            "input_ids": generation_ids,
            "attention_mask": selected_attention,
        }

        generated = reverse_denoise_generate(
            backbone,
            generation_batch,
            config,
            generator,
            device,
            num_steps=_generation_steps(config, original_ids.shape[1]),
            amp_dtype=amp_dtype,
            forced_corrupt_mask=response_mask,
            forced_valid_mask=selected_attention.bool(),
            bridge_noise_sampler=bridge_noise_sampler,
        )

        for local_idx, meta in enumerate(selected_meta):
            prompt_len = int(meta["prompt_len"])
            true_len = max(prompt_len, int(meta["true_len"]))
            prompt_ids = original_ids[local_idx, :prompt_len].detach().cpu().tolist()
            target_ids = original_ids[local_idx, prompt_len:true_len].detach().cpu().tolist()
            generated_response_ids = generated[local_idx, prompt_len:].detach().cpu().tolist()
            sampled_response_full = _decode_token_ids(tokenizer, generated_response_ids, max_chars=0)
            _append_generation_ppl_text(ppl_texts, sampled_response_full, ppl_limit)
            if sample_rows is not None:
                sample_rows.append(
                    {
                        "source_type": meta["source_type"],
                        "prompt_tokens": prompt_len,
                        "target_tokens": max(0, true_len - prompt_len),
                        "masked_tokens": int(response_mask[local_idx].sum().detach().cpu().item()),
                        "sampled_tokens": int(generated.shape[1] - prompt_len),
                        "prompt": _decode_token_ids(tokenizer, prompt_ids, max_chars=max_chars),
                        "target": _decode_token_ids(tokenizer, target_ids, max_chars=max_chars),
                        "masked_target": _decode_masked_row(
                            tokenizer,
                            generation_ids[local_idx].long(),
                            response_mask[local_idx],
                            selected_attention[local_idx],
                            max_chars=max_chars,
                        ),
                        "sampled_response": _truncate_text(sampled_response_full, max_chars),
                        "sampled": _decode_row(
                            tokenizer,
                            generated[local_idx].long(),
                            selected_attention[local_idx],
                            max_chars=max_chars,
                        ),
                    }
                )
            total_samples += 1

    if was_training:
        backbone.train()
    result = {"sft_prompt_generation_samples": float(total_samples)}
    for source_name, count in counts.items():
        result[f"sft_prompt_generation_{source_name}_samples"] = float(count)
    return result


@torch.no_grad()
def evaluate_unconditional_generations(
    backbone,
    config,
    generator,
    device,
    amp_dtype=None,
    sample_rows: Optional[list] = None,
    sample_limit: int = 8,
    ppl_texts: Optional[list] = None,
    ppl_limit: int = 64,
    bridge_noise_sampler=None,
) -> Dict[str, float]:
    """Generate fully masked sequences for a small qualitative WandB table."""
    count = max(0, int(sample_limit or 0))
    if count <= 0:
        return {"unconditional_generation_samples": 0.0}

    was_training = backbone.training
    backbone.eval()
    backbone_module = unwrap_backbone_helpers(backbone)
    tokenizer = backbone_module.tokenizer
    mask_id = _mask_token_id(backbone_module, config)
    length = _generation_target_length(config)
    input_ids = torch.full((count, length), mask_id, device=device, dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    corrupt_mask = torch.ones_like(input_ids, dtype=torch.bool)
    batch = {"input_ids": input_ids, "attention_mask": attention_mask}

    generated = reverse_denoise_generate(
        backbone,
        batch,
        config,
        generator,
        device,
        num_steps=_generation_steps(config, length),
        amp_dtype=amp_dtype,
        forced_corrupt_mask=corrupt_mask,
        forced_valid_mask=attention_mask.bool(),
        bridge_noise_sampler=bridge_noise_sampler,
    )

    if sample_rows is not None:
        max_chars = int(getattr(config, "wandb_generation_max_chars", 2000) or 2000)
        for row_idx in range(count):
            full_sample = _decode_row(
                tokenizer,
                generated[row_idx].long(),
                attention_mask[row_idx],
                max_chars=0,
            )
            _append_generation_ppl_text(ppl_texts, full_sample, ppl_limit)
            sample_rows.append(
                {
                    "length": length,
                    "sampled_tokens": int(generated.shape[1]),
                    "sampled": _truncate_text(full_sample, max_chars),
                }
            )
    elif ppl_texts is not None:
        for row_idx in range(count):
            _append_generation_ppl_text(
                ppl_texts,
                _decode_row(tokenizer, generated[row_idx].long(), attention_mask[row_idx], max_chars=0),
                ppl_limit,
            )

    if was_training:
        backbone.train()
    return {"unconditional_generation_samples": float(count)}


@torch.no_grad()
def evaluate_gsm8k_conditional_generations(
    backbone,
    config,
    generator,
    device,
    amp_dtype=None,
    sample_rows: Optional[list] = None,
    sample_limit: int = 8,
    ppl_texts: Optional[list] = None,
    ppl_limit: int = 64,
    bridge_noise_sampler=None,
) -> Dict[str, float]:
    """Generate answer continuations from random GSM8K eval questions."""
    count = max(0, int(sample_limit or 0))
    if count <= 0:
        return {"gsm8k_conditional_generation_samples": 0.0}

    gsm8k_path = str(getattr(config, "gsm8k_eval_path", "data/gsm8k/test.jsonl") or "")
    records = _load_gsm8k_records(gsm8k_path)
    if not records:
        return {"gsm8k_conditional_generation_samples": 0.0}

    was_training = backbone.training
    backbone.eval()
    backbone_module = unwrap_backbone_helpers(backbone)
    tokenizer = backbone_module.tokenizer
    mask_id = _mask_token_id(backbone_module, config)
    target_length = _generation_target_length(config)
    max_chars = int(getattr(config, "wandb_generation_max_chars", 2000) or 2000)

    sampled_indices = torch.randint(
        len(records),
        (count,),
        generator=generator,
        device=device,
    ).detach().cpu().tolist()
    selected = [records[int(idx)] for idx in sampled_indices]
    encoded_rows = []
    for record in selected:
        prompt = _chat_prompt(record["question"])
        prefix_ids = _tokenizer_ids(tokenizer, prompt)
        prefix_ids, span_len = _prompt_prefix_and_mask_span(prefix_ids, target_length)
        answer_ids = _tokenizer_ids(tokenizer, record["answer"])
        target_ids = prefix_ids + answer_ids[:span_len]
        input_ids = prefix_ids + [mask_id] * span_len
        corrupt = [False] * len(prefix_ids) + [True] * span_len
        encoded_rows.append(
            {
                "record": record,
                "prefix_len": len(prefix_ids),
                "span_len": span_len,
                "target_ids": target_ids,
                "input_ids": input_ids,
                "corrupt": corrupt,
            }
        )

    seq_len = max(len(row["input_ids"]) for row in encoded_rows)
    input_ids = torch.full((len(encoded_rows), seq_len), mask_id, device=device, dtype=torch.long)
    attention_mask = torch.zeros((len(encoded_rows), seq_len), device=device, dtype=torch.long)
    corrupt_mask = torch.zeros((len(encoded_rows), seq_len), device=device, dtype=torch.bool)
    for row_idx, row in enumerate(encoded_rows):
        ids = torch.tensor(row["input_ids"], device=device, dtype=torch.long)
        mask = torch.tensor(row["corrupt"], device=device, dtype=torch.bool)
        input_ids[row_idx, : len(row["input_ids"])] = ids
        attention_mask[row_idx, : len(row["input_ids"])] = 1
        corrupt_mask[row_idx, : len(row["input_ids"])] = mask

    batch = {"input_ids": input_ids, "attention_mask": attention_mask}
    generation_steps = _generation_steps(config, seq_len)
    metrics: Dict[str, float] = {
        "gsm8k_conditional_generation_samples": float(len(encoded_rows)),
        "gsm8k_eval_examples_per_rank": float(len(encoded_rows)),
    }
    generated = reverse_denoise_generate(
        backbone,
        batch,
        config,
        generator,
        device,
        num_steps=generation_steps,
        amp_dtype=amp_dtype,
        forced_corrupt_mask=corrupt_mask,
        forced_valid_mask=attention_mask.bool(),
        bridge_noise_sampler=bridge_noise_sampler,
    )

    strict_matches = 0
    flexible_matches = 0
    strict_no_answer = 0
    flexible_no_answer = 0
    gold_no_answer = 0
    for row_idx, row in enumerate(encoded_rows):
        answer_start = int(row["prefix_len"])
        answer_end = answer_start + int(row["span_len"])
        sampled_answer_full = _decode_token_ids(
            tokenizer,
            generated[row_idx, answer_start:answer_end].detach().cpu().tolist(),
            max_chars=0,
        )
        gold = _extract_gsm8k_gold_answer(row["record"]["answer"])
        strict = _extract_gsm8k_strict_answer(sampled_answer_full)
        flexible = _extract_gsm8k_flexible_answer(sampled_answer_full)
        strict_match = bool(strict is not None and gold is not None and strict == gold)
        flexible_match = bool(flexible is not None and gold is not None and flexible == gold)
        strict_matches += int(strict_match)
        flexible_matches += int(flexible_match)
        strict_no_answer += int(strict is None)
        flexible_no_answer += int(flexible is None)
        gold_no_answer += int(gold is None)

        _append_generation_ppl_text(ppl_texts, sampled_answer_full, ppl_limit)
        if sample_rows is not None:
            sample_rows.append(
                {
                    "generation_steps": int(generation_steps),
                    "target_length": int(target_length),
                    "prompt_tokens": int(row["prefix_len"]),
                    "target_tokens": int(len(row["target_ids"])),
                    "sampled_tokens": int(generated.shape[1]),
                    "question": _truncate_text(row["record"]["question"], max_chars),
                    "gold_answer": gold,
                    "strict_prediction": strict,
                    "flexible_prediction": flexible,
                    "strict_match": strict_match,
                    "flexible_match": flexible_match,
                    "target": _decode_token_ids(
                        tokenizer,
                        row["target_ids"],
                        max_chars=max_chars,
                    ),
                    "masked_target": _decode_masked_row(
                        tokenizer,
                        input_ids[row_idx].long(),
                        corrupt_mask[row_idx],
                        attention_mask[row_idx],
                        max_chars,
                    ),
                    "sampled_answer": _truncate_text(sampled_answer_full, max_chars),
                    "sampled": _decode_row(
                        tokenizer,
                        generated[row_idx].long(),
                        attention_mask[row_idx],
                        max_chars=max_chars,
                    ),
                    "masked_tokens": float(row["span_len"]),
                }
            )

    denom = max(len(encoded_rows), 1)
    metrics["gsm8k/generation_steps"] = float(generation_steps)
    metrics["gsm8k/strict_accuracy"] = float(strict_matches) / float(denom)
    metrics["gsm8k/flexible_accuracy"] = float(flexible_matches) / float(denom)
    metrics["gsm8k/strict_no_answer_rate"] = float(strict_no_answer) / float(denom)
    metrics["gsm8k/flexible_no_answer_rate"] = float(flexible_no_answer) / float(denom)
    metrics["gsm8k/gold_no_answer_rate"] = float(gold_no_answer) / float(denom)

    if was_training:
        backbone.train()
    return metrics
