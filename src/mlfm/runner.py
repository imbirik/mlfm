"""Training loop for MLFM."""

from __future__ import annotations

import json
import logging
import math
import os
import subprocess
import time
from bisect import bisect_right
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from typing import Dict, Optional, Sequence

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from mlfm.adapters import AdaLNWrapper, DiTBlockWrapper, LoRALinear, TiedOutputLoRA, iter_trainable_named_parameters
from mlfm.corruption import (
    build_valid_token_mask,
    gamma_distribution_component_cdfs,
    gamma_distribution_component_pdfs,
    gamma_non_fitted_quantile_edges,
)
from mlfm.loaders import load_mlfm_backbone, load_mlfm_dataloader
from mlfm.noise_geometry import BridgeNoiseSampler, hidden_dim_from_backbone
from mlfm.train_step import _as_special_ids, compute_mlfm_loss
from mlfm.validation import (
    evaluate_corrupted_ce,
    evaluate_gsm8k_conditional_generations,
    evaluate_generation_smoke,
    evaluate_sft_prompt_conditional_generations,
    evaluate_unconditional_generations,
    move_batch_to_device,
)
from utils.logging_utils import log_for_0
from utils.train_utils import (
    autocast_context,
    barrier,
    distributed_available,
    get_rank,
    get_world_size,
    is_main_process,
    reduce_metrics,
    resolve_precision,
    setup_distributed_and_device,
)


logger = logging.getLogger(__name__)

GPU_RUNTIME_METRIC_KEYS = (
    "gpu/memory_allocated_gb",
    "gpu/memory_reserved_gb",
    "gpu/max_memory_allocated_gb",
    "gpu/max_memory_reserved_gb",
    "gpu/memory_total_gb",
    "gpu/memory_free_gb",
    "gpu/memory_used_gb",
    "gpu/memory_used_pct",
    "gpu/utilization_pct",
    "gpu/memory_utilization_pct",
    "gpu/power_w",
    "gpu/temperature_c",
)


def _make_generator(seed: int, device: torch.device):
    gen_device = device if device.type == "cuda" else torch.device("cpu")
    generator = torch.Generator(device=gen_device)
    generator.manual_seed(seed)
    return generator


def _rank_resume_seed(base_seed: int, rank: int, step: int) -> int:
    """Deterministic nonzero-rank seed used after loading rank-0 checkpoint RNG."""
    modulus = (1 << 63) - 1
    seed = int(base_seed) + int(rank) * 1_000_003 + int(step) * 9_176 + 0x9E3779B97F4A7C15
    return int(seed % modulus)


def _diversify_nonzero_rank_resume_rng(config, rank: int, step: int, device: torch.device, generator: torch.Generator):
    """Avoid synchronizing all rank RNG streams when a rank-0 checkpoint is restored."""
    if int(rank) <= 0:
        return None
    seed = _rank_resume_seed(int(getattr(config, "seed", 0)), int(rank), int(step))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if generator is not None:
        generator.manual_seed(seed)
    return seed


def _set_loader_epoch(loader, epoch: int):
    if hasattr(loader, "set_epoch"):
        loader.set_epoch(epoch)
        return
    sampler = getattr(loader, "sampler", None)
    if hasattr(sampler, "set_epoch"):
        sampler.set_epoch(epoch)


def _restore_train_iterator(loader, epoch: int, microbatches_consumed: int):
    """Return an iterator positioned at the saved microbatch within an epoch."""
    if len(loader) <= 0:
        raise ValueError("Training dataloader is empty. Check data paths, batch size, and drop_last.")
    epoch = int(epoch)
    remaining_skip = int(microbatches_consumed)
    _set_loader_epoch(loader, epoch)
    iterator = iter(loader)
    while remaining_skip > 0:
        try:
            next(iterator)
            remaining_skip -= 1
        except StopIteration:
            epoch += 1
            _set_loader_epoch(loader, epoch)
            iterator = iter(loader)
    return iterator, epoch


def _resume_train_iterator_plan(
    loader_len: int,
    checkpoint_epoch: int,
    start_step: int,
    grad_accum_steps: int,
    restore_exact: bool,
    max_skip_batches: int,
) -> Tuple[int, int, int, str]:
    """Choose the dataloader epoch/offset used after checkpoint resume.

    Exact replay requires consuming the saved offset from the dataloader. On
    very large packed datasets this can mean tens of thousands of batches per
    rank before the first backward pass, which is slow and can desynchronize
    ranks enough to trigger NCCL timeouts. The default is therefore to resume
    from the next sampler epoch.
    """
    loader_len = int(loader_len)
    if loader_len <= 0:
        raise ValueError("Training dataloader is empty. Check data paths, batch size, and drop_last.")
    checkpoint_epoch = int(checkpoint_epoch)
    start_step = int(start_step)
    grad_accum_steps = max(1, int(grad_accum_steps))
    requested_skip = (start_step * grad_accum_steps) % loader_len
    if start_step <= 0:
        return checkpoint_epoch, 0, requested_skip, "fresh"
    if restore_exact:
        max_skip_batches = int(max_skip_batches)
        if max_skip_batches < 0 or requested_skip <= max_skip_batches:
            return checkpoint_epoch, requested_skip, requested_skip, "exact"
        return checkpoint_epoch + 1, 0, requested_skip, "skip_limit"
    return checkpoint_epoch + 1, 0, requested_skip, "next_epoch"


def _unwrap(model):
    model = model.module if hasattr(model, "module") else model
    return model._orig_mod if hasattr(model, "_orig_mod") else model


def _paths(value, fallback=None):
    if value is None:
        value = fallback
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return list(value)


def _as_config_dict(config) -> Dict:
    def convert(value):
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        if isinstance(value, (list, tuple)):
            return [convert(item) for item in value]
        if isinstance(value, dict):
            return {str(key): convert(item) for key, item in value.items()}
        if hasattr(value, "__dict__"):
            return {key: convert(item) for key, item in vars(value).items() if not key.startswith("_")}
        return str(value)

    return {key: convert(value) for key, value in vars(config).items() if not key.startswith("_")}


def _resolve_wandb_dir(config) -> str:
    project_tmp = None
    project_dir = os.environ.get("PROJECTDIR")
    if project_dir:
        project_tmp = os.path.join(project_dir, "tmp", "wandb")
    path = (
        getattr(config, "wandb_dir", None)
        or os.environ.get("WANDB_DIR")
        or project_tmp
        or os.path.join(str(getattr(config, "output_dir", ".")), "wandb")
    )
    os.makedirs(path, exist_ok=True)
    return path


def _lr_multiplier(step: int, max_steps: int, warmup_steps: int, min_lr_ratio: float) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return float(step + 1) / float(warmup_steps)
    denom = max(max_steps - warmup_steps, 1)
    progress = min(max(float(step - warmup_steps) / float(denom), 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return float(min_lr_ratio) + (1.0 - float(min_lr_ratio)) * cosine


def _no_weight_decay(name: str) -> bool:
    lowered = name.lower()
    return (
        name.endswith(".bias")
        or "norm" in lowered
        or "time_mlp" in lowered
        or "adaln" in lowered
        or "lora_scale" in lowered
    )


def _trainable_param_ids_for_modules(model, module_types: tuple) -> set:
    if model is None:
        return set()
    param_ids = set()
    for module in _unwrap(model).modules():
        if isinstance(module, module_types):
            for param in module.parameters(recurse=True):
                if param.requires_grad:
                    param_ids.add(id(param))
    return param_ids


def _count_trainable_param_ids(model, param_ids: set) -> int:
    return sum(param.numel() for param in _unwrap(model).parameters() if id(param) in param_ids and param.requires_grad)


def _adapter_param_counts(model) -> Dict[str, int]:
    model = _unwrap(model)
    lora_ids = _trainable_param_ids_for_modules(model, (LoRALinear, TiedOutputLoRA))
    adaln_ids = _trainable_param_ids_for_modules(model, (AdaLNWrapper, DiTBlockWrapper))
    output_lora_ids = _trainable_param_ids_for_modules(getattr(model, "output_lora", None), (TiedOutputLoRA,))

    named_trainable = list(iter_trainable_named_parameters(model))
    for name, param in named_trainable:
        lowered = name.lower()
        if id(param) in lora_ids or id(param) in adaln_ids:
            continue
        if "lora" in lowered:
            lora_ids.add(id(param))
        elif "time_mlp" in lowered or "adaln" in lowered:
            adaln_ids.add(id(param))

    # DiT block wrappers contain the wrapped block, so recurse=True also sees
    # LoRA modules inside the block. Keep the reported categories disjoint.
    adaln_ids = adaln_ids - lora_ids
    lora_total = _count_trainable_param_ids(model, lora_ids)
    output_lora = _count_trainable_param_ids(model, output_lora_ids)
    adaln_total = _count_trainable_param_ids(model, adaln_ids)
    accounted = lora_ids | adaln_ids
    other = sum(param.numel() for _, param in named_trainable if id(param) not in accounted)
    return {
        "lora": lora_total,
        "lora_output": output_lora,
        "lora_backbone": max(0, lora_total - output_lora),
        "adaln": adaln_total,
        "other": other,
    }


@torch.no_grad()
def _embedding_geometry_stats(backbone) -> Dict[str, float]:
    module = _unwrap(backbone)
    input_embeddings = getattr(module, "input_embeddings", None)
    weight = getattr(input_embeddings, "weight", None)
    if weight is None:
        return {}

    embedding = weight.detach().float()
    if embedding.ndim != 2 or embedding.shape[0] <= 1 or embedding.shape[1] <= 1:
        return {}

    special_ids = set(getattr(module, "special_token_ids", []) or [])
    mask_token_id = getattr(module, "mask_token_id", None)
    if mask_token_id is not None:
        special_ids.add(int(mask_token_id))
    keep = torch.ones(embedding.shape[0], device=embedding.device, dtype=torch.bool)
    for token_id in special_ids:
        if 0 <= int(token_id) < int(embedding.shape[0]):
            keep[int(token_id)] = False
    token_embedding = embedding[keep]
    if token_embedding.shape[0] <= 1:
        token_embedding = embedding

    x = token_embedding.cpu()
    mean = x.mean(dim=0)
    centered = x - mean
    vocab_size, dim = int(x.shape[0]), int(x.shape[1])
    cov = centered.t().matmul(centered) / max(vocab_size - 1, 1)
    eigvals = torch.linalg.eigvalsh(cov).clamp_min(0.0)
    trace = eigvals.sum().clamp_min(1e-30)
    probs = eigvals / trace
    positive = eigvals[eigvals > trace * 1e-12]
    effective_rank = float(torch.exp(-(probs * probs.clamp_min(1e-30).log()).sum()).item())
    participation_rank = float((trace * trace / eigvals.square().sum().clamp_min(1e-30)).item())
    top_eig = float(eigvals[-1].item())
    top10 = float(eigvals[-min(10, dim) :].sum().item())
    condition = float((positive[-1] / positive[0]).item()) if positive.numel() > 0 else float("nan")
    per_dim_var = centered.var(dim=0, unbiased=True)
    row_norms = x.norm(dim=1)
    radius = math.sqrt(float(trace.item()))
    mean_norm = float(mean.norm().item())
    pair_distance_rms = math.sqrt(max(2.0 * float(trace.item()), 0.0))

    stats = {
        "embedding/vocab_rows_used": float(vocab_size),
        "embedding/vocab_rows_total": float(embedding.shape[0]),
        "embedding/special_rows_excluded": float(max(0, int(embedding.shape[0]) - vocab_size)),
        "embedding/dim": float(dim),
        "embedding/mean_norm": mean_norm,
        "embedding/rms_centered_norm": radius,
        "embedding/mean_norm_over_rms_centered_norm": mean_norm / max(radius, 1e-30),
        "embedding/row_norm_mean": float(row_norms.mean().item()),
        "embedding/row_norm_std": float(row_norms.std(unbiased=True).item()),
        "embedding/row_norm_min": float(row_norms.min().item()),
        "embedding/row_norm_max": float(row_norms.max().item()),
        "embedding/per_dim_variance_mean": float(per_dim_var.mean().item()),
        "embedding/per_dim_variance_std": float(per_dim_var.std(unbiased=True).item()),
        "embedding/per_dim_variance_cv": float((per_dim_var.std(unbiased=True) / per_dim_var.mean().clamp_min(1e-30)).item()),
        "embedding/per_dim_variance_min": float(per_dim_var.min().item()),
        "embedding/per_dim_variance_max": float(per_dim_var.max().item()),
        "embedding/total_variance_trace": float(trace.item()),
        "embedding/rms_pairwise_distance": pair_distance_rms,
        "embedding/effective_rank_entropy": effective_rank,
        "embedding/effective_rank_entropy_frac": effective_rank / float(dim),
        "embedding/participation_rank": participation_rank,
        "embedding/participation_rank_frac": participation_rank / float(dim),
        "embedding/top_eigenvalue": top_eig,
        "embedding/top_eigenvalue_fraction": top_eig / float(trace.item()),
        "embedding/top10_eigenvalue_fraction": top10 / float(trace.item()),
        "embedding/cov_condition_number": condition,
    }

    if mask_token_id is not None and 0 <= int(mask_token_id) < int(embedding.shape[0]):
        mask = embedding[int(mask_token_id)].detach().float().cpu()
        mask_centered_norm = float((mask - mean).norm().item())
        stats.update(
            {
                "embedding/mask_norm": float(mask.norm().item()),
                "embedding/mask_to_mean_norm": mask_centered_norm,
                "embedding/mask_to_mean_over_rms_centered_norm": mask_centered_norm / max(radius, 1e-30),
            }
        )
    return stats


def _log_embedding_geometry(backbone, wandb_run=None):
    if not is_main_process():
        return
    try:
        stats = _embedding_geometry_stats(backbone)
    except Exception as exc:
        log_for_0(f"Embedding geometry diagnostics failed: {exc}", level=logging.WARNING)
        return
    if not stats:
        return

    log_for_0(
        "Input embedding geometry: "
        f"rows={int(stats['embedding/vocab_rows_used']):,}/"
        f"{int(stats['embedding/vocab_rows_total']):,}, "
        f"dim={int(stats['embedding/dim'])}, "
        f"mean_norm={stats['embedding/mean_norm']:.4g}, "
        f"rms_centered_norm={stats['embedding/rms_centered_norm']:.4g}, "
        f"row_norm_mean={stats['embedding/row_norm_mean']:.4g}, "
        f"row_norm_std={stats['embedding/row_norm_std']:.4g}"
    )
    log_for_0(
        "Input embedding covariance: "
        f"var_mean={stats['embedding/per_dim_variance_mean']:.4g}, "
        f"var_cv={stats['embedding/per_dim_variance_cv']:.4g}, "
        f"effective_rank={stats['embedding/effective_rank_entropy']:.1f}/"
        f"{int(stats['embedding/dim'])} "
        f"({100.0 * stats['embedding/effective_rank_entropy_frac']:.2f}%), "
        f"participation_rank={stats['embedding/participation_rank']:.1f}/"
        f"{int(stats['embedding/dim'])} "
        f"({100.0 * stats['embedding/participation_rank_frac']:.2f}%), "
        f"top1_var={100.0 * stats['embedding/top_eigenvalue_fraction']:.2f}%, "
        f"top10_var={100.0 * stats['embedding/top10_eigenvalue_fraction']:.2f}%"
    )
    if "embedding/mask_to_mean_norm" in stats:
        log_for_0(
            "Mask embedding geometry: "
            f"mask_norm={stats['embedding/mask_norm']:.4g}, "
            f"mask_to_mean_norm={stats['embedding/mask_to_mean_norm']:.4g}, "
            f"mask_to_mean/rms_centered={stats['embedding/mask_to_mean_over_rms_centered_norm']:.4g}"
        )
    if wandb_run is not None:
        wandb_run.summary.update(stats)


def _geometry_stats_from_moments(
    *,
    prefix: str,
    count: int,
    dim: int,
    sum_vec: torch.Tensor,
    sum_outer: torch.Tensor,
    row_norm_sum: float,
    row_norm_sum2: float,
    row_norm_min: float,
    row_norm_max: float,
    rows_total: Optional[int] = None,
) -> Dict[str, float]:
    if count <= 1 or dim <= 1:
        return {}
    n = float(count)
    sum_vec = sum_vec.detach().cpu().double()
    sum_outer = sum_outer.detach().cpu().double()
    mean = sum_vec / n
    cov = (sum_outer - n * torch.outer(mean, mean)) / max(n - 1.0, 1.0)
    cov = 0.5 * (cov + cov.t())
    eigvals = torch.linalg.eigvalsh(cov).clamp_min(0.0)
    trace = eigvals.sum().clamp_min(1e-30)
    probs = eigvals / trace
    positive = eigvals[eigvals > trace * 1e-12]
    effective_rank = float(torch.exp(-(probs * probs.clamp_min(1e-30).log()).sum()).item())
    participation_rank = float((trace * trace / eigvals.square().sum().clamp_min(1e-30)).item())
    per_dim_var = cov.diag().clamp_min(0.0)
    radius = math.sqrt(float(trace.item()))
    row_norm_mean = float(row_norm_sum) / n
    row_norm_var = max((float(row_norm_sum2) - n * row_norm_mean * row_norm_mean) / max(n - 1.0, 1.0), 0.0)
    top_eig = float(eigvals[-1].item())
    top10 = float(eigvals[-min(10, dim) :].sum().item())
    condition = float((positive[-1] / positive[0]).item()) if positive.numel() > 0 else float("nan")

    stats = {
        f"{prefix}/tokens_used": float(count),
        f"{prefix}/dim": float(dim),
        f"{prefix}/mean_norm": float(mean.norm().item()),
        f"{prefix}/rms_centered_norm": radius,
        f"{prefix}/mean_norm_over_rms_centered_norm": float(mean.norm().item()) / max(radius, 1e-30),
        f"{prefix}/row_norm_mean": row_norm_mean,
        f"{prefix}/row_norm_std": math.sqrt(row_norm_var),
        f"{prefix}/row_norm_min": float(row_norm_min),
        f"{prefix}/row_norm_max": float(row_norm_max),
        f"{prefix}/per_dim_variance_mean": float(per_dim_var.mean().item()),
        f"{prefix}/per_dim_variance_std": float(per_dim_var.std(unbiased=True).item()),
        f"{prefix}/per_dim_variance_cv": float((per_dim_var.std(unbiased=True) / per_dim_var.mean().clamp_min(1e-30)).item()),
        f"{prefix}/per_dim_variance_min": float(per_dim_var.min().item()),
        f"{prefix}/per_dim_variance_max": float(per_dim_var.max().item()),
        f"{prefix}/total_variance_trace": float(trace.item()),
        f"{prefix}/rms_pairwise_distance": math.sqrt(max(2.0 * float(trace.item()), 0.0)),
        f"{prefix}/effective_rank_entropy": effective_rank,
        f"{prefix}/effective_rank_entropy_frac": effective_rank / float(dim),
        f"{prefix}/participation_rank": participation_rank,
        f"{prefix}/participation_rank_frac": participation_rank / float(dim),
        f"{prefix}/top_eigenvalue": top_eig,
        f"{prefix}/top_eigenvalue_fraction": top_eig / float(trace.item()),
        f"{prefix}/top10_eigenvalue_fraction": top10 / float(trace.item()),
        f"{prefix}/cov_condition_number": condition,
    }
    if rows_total is not None:
        stats[f"{prefix}/vocab_rows_total"] = float(rows_total)
    return stats


@torch.no_grad()
def _empirical_embedding_geometry_stats(backbone, train_loader, device: torch.device, config) -> Dict[str, float]:
    module = _unwrap(backbone)
    input_embeddings = getattr(module, "input_embeddings", None)
    weight = getattr(input_embeddings, "weight", None)
    if weight is None:
        return {}
    dim = int(weight.shape[1])
    if dim <= 1:
        return {}

    max_tokens = int(getattr(config, "embedding_geometry_empirical_tokens", 65536) or 0)
    max_batches = int(getattr(config, "embedding_geometry_empirical_batches", 64) or 0)
    if max_tokens <= 1 or max_batches <= 0:
        return {}

    special_ids = set(getattr(module, "special_token_ids", []) or [])
    mask_token_id = getattr(module, "mask_token_id", None)
    if mask_token_id is not None:
        special_ids.add(int(mask_token_id))

    sum_vec = torch.zeros(dim, device=device, dtype=torch.float64)
    sum_outer = torch.zeros((dim, dim), device=device, dtype=torch.float64)
    count = torch.zeros((), device=device, dtype=torch.float64)
    row_norm_sum = torch.zeros((), device=device, dtype=torch.float64)
    row_norm_sum2 = torch.zeros((), device=device, dtype=torch.float64)
    row_norm_min = torch.full((), float("inf"), device=device, dtype=torch.float64)
    row_norm_max = torch.full((), float("-inf"), device=device, dtype=torch.float64)

    _set_loader_epoch(train_loader, 0)
    for batch_idx, batch in enumerate(train_loader):
        if batch_idx >= max_batches or int(count.item()) >= max_tokens:
            break
        input_ids = batch["input_ids"].to(device=device, non_blocking=True)
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device=device, non_blocking=True)
        valid_mask = build_valid_token_mask(input_ids, attention_mask=attention_mask, special_token_ids=special_ids)
        valid_mask = valid_mask & (input_ids >= 0) & (input_ids < int(weight.shape[0]))
        token_ids = input_ids[valid_mask]
        if token_ids.numel() == 0:
            continue
        remaining = max_tokens - int(count.item())
        if token_ids.numel() > remaining:
            token_ids = token_ids[:remaining]
        x = module.embed(token_ids).detach().float()
        if x.ndim != 2 or x.shape[0] == 0:
            continue
        norms = x.norm(dim=1).double()
        count += float(x.shape[0])
        sum_vec += x.sum(dim=0).double()
        sum_outer += x.t().matmul(x).double()
        row_norm_sum += norms.sum()
        row_norm_sum2 += norms.square().sum()
        row_norm_min = torch.minimum(row_norm_min, norms.min())
        row_norm_max = torch.maximum(row_norm_max, norms.max())

    if distributed_available():
        dist.all_reduce(count, op=dist.ReduceOp.SUM)
        dist.all_reduce(sum_vec, op=dist.ReduceOp.SUM)
        dist.all_reduce(sum_outer, op=dist.ReduceOp.SUM)
        dist.all_reduce(row_norm_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(row_norm_sum2, op=dist.ReduceOp.SUM)
        dist.all_reduce(row_norm_min, op=dist.ReduceOp.MIN)
        dist.all_reduce(row_norm_max, op=dist.ReduceOp.MAX)

    if int(count.item()) <= 1:
        return {}
    stats = _geometry_stats_from_moments(
        prefix="embedding_empirical",
        count=int(count.item()),
        dim=dim,
        sum_vec=sum_vec,
        sum_outer=sum_outer,
        row_norm_sum=float(row_norm_sum.item()),
        row_norm_sum2=float(row_norm_sum2.item()),
        row_norm_min=float(row_norm_min.item()),
        row_norm_max=float(row_norm_max.item()),
        rows_total=int(weight.shape[0]),
    )
    stats["embedding_empirical/max_tokens_per_rank"] = float(max_tokens)
    stats["embedding_empirical/max_batches_per_rank"] = float(max_batches)
    return stats


def _log_empirical_embedding_geometry(backbone, train_loader, device: torch.device, config, wandb_run=None):
    try:
        stats = _empirical_embedding_geometry_stats(backbone, train_loader, device, config)
    except Exception as exc:
        log_for_0(f"Empirical embedding geometry diagnostics failed: {exc}", level=logging.WARNING)
        return
    if not is_main_process() or not stats:
        return

    log_for_0(
        "Empirical training-token embedding geometry: "
        f"tokens={int(stats['embedding_empirical/tokens_used']):,}, "
        f"dim={int(stats['embedding_empirical/dim'])}, "
        f"mean_norm={stats['embedding_empirical/mean_norm']:.4g}, "
        f"rms_centered_norm={stats['embedding_empirical/rms_centered_norm']:.4g}, "
        f"row_norm_mean={stats['embedding_empirical/row_norm_mean']:.4g}, "
        f"row_norm_std={stats['embedding_empirical/row_norm_std']:.4g}"
    )
    log_for_0(
        "Empirical training-token covariance: "
        f"var_mean={stats['embedding_empirical/per_dim_variance_mean']:.4g}, "
        f"var_cv={stats['embedding_empirical/per_dim_variance_cv']:.4g}, "
        f"effective_rank={stats['embedding_empirical/effective_rank_entropy']:.1f}/"
        f"{int(stats['embedding_empirical/dim'])} "
        f"({100.0 * stats['embedding_empirical/effective_rank_entropy_frac']:.2f}%), "
        f"participation_rank={stats['embedding_empirical/participation_rank']:.1f}/"
        f"{int(stats['embedding_empirical/dim'])} "
        f"({100.0 * stats['embedding_empirical/participation_rank_frac']:.2f}%), "
        f"top1_var={100.0 * stats['embedding_empirical/top_eigenvalue_fraction']:.2f}%, "
        f"top10_var={100.0 * stats['embedding_empirical/top10_eigenvalue_fraction']:.2f}%"
    )
    if wandb_run is not None:
        wandb_run.summary.update(stats)


def build_mlfm_optimizer(config, model):
    groups = {}
    base_lr = float(getattr(config, "base_lr", None) or getattr(config, "lr", None) or getattr(config, "lora_lr", 1e-4))
    adapter_weight_decay = float(getattr(config, "weight_decay", 0.01))
    base_weight_decay_value = getattr(config, "base_weight_decay", None)
    base_weight_decay = float(adapter_weight_decay if base_weight_decay_value is None else base_weight_decay_value)
    for name, param in iter_trainable_named_parameters(_unwrap(model)):
        if "output_lora" in name:
            lr = float(getattr(config, "lora_output_lr", getattr(config, "lora_lr", 1e-4)))
            weight_decay = adapter_weight_decay
        elif "time_mlp" in name or "adaln" in name:
            lr = float(getattr(config, "adaln_lr", 3e-4))
            weight_decay = adapter_weight_decay
        elif "lora" in name.lower():
            lr = float(getattr(config, "lora_lr", 1e-4))
            weight_decay = adapter_weight_decay
        else:
            lr = base_lr
            weight_decay = base_weight_decay
        wd = 0.0 if _no_weight_decay(name) else weight_decay
        key = (lr, wd)
        groups.setdefault(key, {"params": [], "lr": lr, "base_lr": lr, "weight_decay": wd})
        groups[key]["params"].append(param)
    if not groups:
        raise ValueError("No trainable parameters found for mlfm.")
    return torch.optim.AdamW(
        list(groups.values()),
        betas=(float(getattr(config, "adam_b1", 0.9)), float(getattr(config, "adam_b2", 0.95))),
    )


def set_group_lrs(optimizer, multiplier: float):
    for group in optimizer.param_groups:
        group["lr"] = group["base_lr"] * multiplier


def save_mlfm_checkpoint(
    model,
    optimizer,
    scaler,
    config,
    output_dir: str,
    step: int,
    epoch: int,
    generator=None,
    token_counters=None,
    ema_state=None,
    bridge_noise_sampler=None,
    loss_diagnostic_state=None,
):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"checkpoint_{step:08d}.pt")
    module = _unwrap(model)
    payload = {
        "model": module.state_dict() if bool(getattr(config, "save_full_model", False)) else module.adapter_state_dict(),
        "save_full_model": bool(getattr(config, "save_full_model", False)),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "step": int(step),
        "epoch": int(epoch),
        "config": _as_config_dict(config),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "generator_state": generator.get_state() if generator is not None else None,
        "token_counters": dict(token_counters or {}),
        "ema_adapter": ema_state,
        "bridge_noise_sampler": bridge_noise_sampler.state_dict() if bridge_noise_sampler is not None else None,
    }
    if loss_diagnostic_state is not None:
        payload["loss_diagnostic_state"] = _serialize_loss_diagnostic_state(loss_diagnostic_state, config)
    torch.save(payload, path)
    latest_path = os.path.join(output_dir, "checkpoint_latest.pt")
    torch.save(payload, latest_path)
    return path


def _normalize_adapter_weight_source(source: str) -> str:
    source = str(source or "model").strip().lower()
    if source in {"model", "live", "adapter", "adapters", "raw"}:
        return "model"
    if source in {"ema", "ema_adapter", "adapter_ema"}:
        return "ema"
    raise ValueError("adapter weight source must be one of: model, live, ema, ema_adapter.")


def _checkpoint_adapter_state(payload: Dict, source: str):
    source = _normalize_adapter_weight_source(source)
    model_state = payload.get("model")
    if source == "model":
        if model_state is None:
            raise KeyError("Checkpoint payload does not contain `model` adapter weights.")
        return model_state, "model"
    ema_state = payload.get("ema_adapter")
    if ema_state is None:
        if model_state is None:
            raise KeyError(
                "Checkpoint was requested with adapter weight source `ema`, but it contains neither "
                "`ema_adapter` nor `model` adapter weights."
            )
        log_for_0(
            "Checkpoint was requested with adapter weight source `ema`, but it does not contain "
            "`ema_adapter`; falling back to `model` adapter weights for backward compatibility.",
            level=logging.WARNING,
        )
        return model_state, "model"
    if model_state is None:
        return ema_state, "ema"
    merged_state = dict(model_state)
    merged_state.update(ema_state)
    if len(ema_state) < len(model_state):
        log_for_0(
            "Checkpoint `ema_adapter` contains fewer keys than `model`; loading `model` weights and "
            "overriding available keys with EMA weights. This is expected for older adapter-only EMA checkpoints.",
            level=logging.WARNING,
        )
    return merged_state, "ema"


def load_mlfm_checkpoint(
    path: str,
    model,
    optimizer=None,
    scaler=None,
    device: Optional[torch.device] = None,
    config=None,
    generator=None,
    token_counters=None,
    ema_state=None,
    bridge_noise_sampler=None,
    loss_diagnostic_state=None,
    restore_rng: bool = True,
    adapter_weight_source: str = "model",
):
    if os.path.isdir(path):
        path = os.path.join(path, "checkpoint_latest.pt")
    payload = torch.load(path, map_location=device or "cpu")
    module = _unwrap(model)
    adapter_state, _ = _checkpoint_adapter_state(payload, adapter_weight_source)
    module.load_adapter_state_dict(adapter_state, strict=False)
    if optimizer is not None and payload.get("optimizer") is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if scaler is not None and payload.get("scaler") is not None:
        scaler.load_state_dict(payload["scaler"])
    if restore_rng and payload.get("torch_rng_state") is not None:
        torch.set_rng_state(payload["torch_rng_state"].cpu())
    if restore_rng and torch.cuda.is_available() and payload.get("cuda_rng_state_all") is not None:
        torch.cuda.set_rng_state_all([state.cpu() for state in payload["cuda_rng_state_all"]])
    if generator is not None and payload.get("generator_state") is not None:
        generator.set_state(payload["generator_state"].cpu())
    if token_counters is not None and payload.get("token_counters") is not None:
        token_counters.update({str(key): float(value) for key, value in payload["token_counters"].items()})
    if ema_state is not None:
        saved_ema = payload.get("ema_adapter")
        if saved_ema is not None:
            ema_state.clear()
            ema_state.update({key: value.detach().clone() for key, value in saved_ema.items()})
    if bridge_noise_sampler is not None and payload.get("bridge_noise_sampler") is not None:
        bridge_noise_sampler.load_state_dict(payload["bridge_noise_sampler"])
    if config is not None and bool(getattr(config, "restore_adaptive_gamma_state", True)):
        saved_config = payload.get("config") or {}
        for key in (
            "gamma_active_piecewise_gamma",
            "gamma_active_piecewise_cdf",
        ):
            if key in saved_config and saved_config[key] is not None:
                setattr(config, key, [float(value) for value in saved_config[key]])
        for key in (
            "gamma_curve_updates",
            "gamma_curve_last_update_step",
        ):
            if key in saved_config:
                setattr(config, key, int(saved_config[key]))
    if (
        loss_diagnostic_state is not None
        and config is not None
        and bool(getattr(config, "restore_loss_diagnostic_state", True))
    ):
        restored_state = _deserialize_loss_diagnostic_state(
            payload.get("loss_diagnostic_state"),
            config,
            saved_config=payload.get("config"),
        )
        if restored_state is not None:
            loss_diagnostic_state.clear()
            loss_diagnostic_state.update(restored_state)
    return int(payload.get("step", 0)), int(payload.get("epoch", 0))


def _init_ema_adapter_state(model) -> Dict[str, torch.Tensor]:
    module = _unwrap(model)
    return {key: value.detach().clone() for key, value in module.adapter_state_dict().items()}


@torch.no_grad()
def _update_ema_adapter_state(ema_state: Dict[str, torch.Tensor], model, decay: float):
    module = _unwrap(model)
    current = module.adapter_state_dict()
    for key, value in current.items():
        value = value.detach()
        if key not in ema_state:
            ema_state[key] = value.clone()
            continue
        target = ema_state[key]
        if torch.is_floating_point(target):
            target.mul_(float(decay)).add_(value.to(device=target.device, dtype=target.dtype), alpha=1.0 - float(decay))
        else:
            target.copy_(value.to(device=target.device, dtype=target.dtype))


@contextmanager
def _using_ema_adapter_weights(model, ema_state: Optional[Dict[str, torch.Tensor]]):
    if not ema_state:
        yield
        return
    module = _unwrap(model)
    live_state = {key: value.detach().clone() for key, value in module.adapter_state_dict().items()}
    module.load_adapter_state_dict(ema_state, strict=False)
    try:
        yield
    finally:
        module.load_adapter_state_dict(live_state, strict=False)


def _log_jsonl(output_dir: str, record: Dict):
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "metrics.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def _reduce_float_metrics(metrics: Dict[str, float], device: torch.device) -> Dict[str, float]:
    reduced = reduce_metrics({key: torch.tensor(value, device=device) for key, value in metrics.items()})
    result = {key: value.item() for key, value in reduced.items()}
    if "ce" in result:
        result["ppl"] = math.exp(min(result["ce"], 20.0))
    for key, value in list(result.items()):
        if key.endswith("_ce"):
            result[key[:-3] + "_ppl"] = math.exp(min(value, 20.0))
    return result


_GENERATION_PPL_CACHE = {}


def _compute_generative_ppl_metrics(config, device: torch.device, text_groups: Dict[str, Sequence[str]]) -> Dict[str, float]:
    if not bool(getattr(config, "online_eval", True)):
        return {}
    model_name = str(getattr(config, "eval_ppl_model", "gpt2-large") or "")
    if not model_name:
        return {}

    groups = {
        name: [text for text in texts if isinstance(text, str) and text.strip()]
        for name, texts in text_groups.items()
    }
    if not any(groups.values()):
        return {}

    max_length = int(getattr(config, "eval_ppl_max_length", 1024) or 1024)
    batch_size = int(getattr(config, "eval_ppl_batch_size", 64) or 64)
    cache_key = (model_name, batch_size, max_length, str(device))
    try:
        from utils.metrics_utils import Metrics as PPLMetrics

        ppl_metrics = _GENERATION_PPL_CACHE.get(cache_key)
        if ppl_metrics is None:
            ppl_metrics = PPLMetrics(
                gen_ppl_eval_model_name_or_path=model_name,
                eval_ppl_batch_size=batch_size,
                eval_context_size=max_length,
                device=device,
            )
            _GENERATION_PPL_CACHE[cache_key] = ppl_metrics

        result = {}
        for name, texts in groups.items():
            if not texts:
                continue
            ppl_metrics.reset()
            values = ppl_metrics.record_generative_perplexity(texts, max_length=max_length, retokenize=True)
            result[f"eval/generative_ppl/{name}"] = float(values["ppl"])
            result[f"eval/generative_ppl/{name}_mean_entropy"] = float(values["mean_entropy"])
            result[f"eval/generative_ppl/{name}_samples"] = float(len(texts))
        return result
    except Exception as exc:
        log_for_0(f"Generative PPL evaluation failed: {exc}", level=logging.WARNING)
        return {"eval/generative_ppl/error": 1.0}


def _query_nvidia_smi(device_index: int) -> Dict[str, float]:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                f"--id={int(device_index)}",
                "--query-gpu=utilization.gpu,utilization.memory,power.draw,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=2.0,
        )
    except Exception:
        return {}
    line = output.strip().splitlines()[0] if output.strip() else ""
    parts = [part.strip() for part in line.split(",")]
    if len(parts) < 4:
        return {}

    def parse_float(value: str) -> Optional[float]:
        try:
            return float(value)
        except ValueError:
            return None

    values = [parse_float(part) for part in parts[:4]]
    keys = [
        "gpu/utilization_pct",
        "gpu/memory_utilization_pct",
        "gpu/power_w",
        "gpu/temperature_c",
    ]
    return {key: value for key, value in zip(keys, values) if value is not None}


def _local_gpu_runtime_metrics(device: torch.device) -> Dict[str, float]:
    if device.type != "cuda" or not torch.cuda.is_available():
        return {}
    index = device.index if device.index is not None else torch.cuda.current_device()
    allocated = float(torch.cuda.memory_allocated(device)) / 1e9
    reserved = float(torch.cuda.memory_reserved(device)) / 1e9
    max_allocated = float(torch.cuda.max_memory_allocated(device)) / 1e9
    max_reserved = float(torch.cuda.max_memory_reserved(device)) / 1e9
    metrics = {
        "gpu/memory_allocated_gb": allocated,
        "gpu/memory_reserved_gb": reserved,
        "gpu/max_memory_allocated_gb": max_allocated,
        "gpu/max_memory_reserved_gb": max_reserved,
    }
    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        total_gb = float(total_bytes) / 1e9
        free_gb = float(free_bytes) / 1e9
        used_gb = total_gb - free_gb
        metrics.update(
            {
                "gpu/memory_total_gb": total_gb,
                "gpu/memory_free_gb": free_gb,
                "gpu/memory_used_gb": used_gb,
                "gpu/memory_used_pct": 100.0 * used_gb / max(total_gb, 1e-12),
            }
        )
    except Exception:
        pass
    try:
        utilization = getattr(torch.cuda, "utilization")(device)
        metrics["gpu/utilization_pct"] = float(utilization)
    except Exception:
        pass
    try:
        memory_utilization = getattr(torch.cuda, "memory_usage")(device)
        metrics["gpu/memory_utilization_pct"] = float(memory_utilization)
    except Exception:
        pass
    for key, value in _query_nvidia_smi(index).items():
        metrics.setdefault(key, value)
    return metrics


def _reduce_gpu_runtime_metrics(local_metrics: Dict[str, float], device: torch.device) -> Dict[str, float]:
    if not local_metrics:
        return {}
    reduced = {}
    if not distributed_available():
        return {key: float(value) for key, value in local_metrics.items()}

    for key in GPU_RUNTIME_METRIC_KEYS:
        has_value = key in local_metrics
        value = float(local_metrics[key]) if has_value else 0.0
        sum_tensor = torch.tensor(value, device=device)
        count_tensor = torch.tensor(1.0 if has_value else 0.0, device=device)
        max_tensor = torch.tensor(value if has_value else -float("inf"), device=device)
        dist.all_reduce(sum_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(max_tensor, op=dist.ReduceOp.MAX)
        count = float(count_tensor.detach().cpu().item())
        if count <= 0.0:
            continue
        reduced[f"{key}_mean"] = float(sum_tensor.detach().cpu().item()) / count
        reduced[f"{key}_max"] = float(max_tensor.detach().cpu().item())
    return reduced


def _scalar_from_metric(metrics: Dict, key: str, default: float = 0.0) -> float:
    value = metrics.get(key)
    if value is None:
        return float(default)
    if isinstance(value, torch.Tensor):
        return float(value.detach().float().mean().cpu().item())
    return float(value)


def _without_sample_metrics(metrics: Dict) -> Dict:
    return {key: value for key, value in metrics.items() if not key.startswith("sample_")}


def _is_sft_like_batch(batch: Dict) -> bool:
    return "sft_response_mask" in batch or "prompt_lengths" in batch


def _random_length_expected_length(config, current_length: int) -> float:
    prob = max(0.0, min(1.0, float(getattr(config, "random_length_prob", 0.0) or 0.0)))
    if prob <= 0.0:
        return float(current_length)
    min_length = max(1, int(getattr(config, "random_length_min", 1) or 1))
    max_length = int(getattr(config, "random_length_max", 0) or 0)
    if max_length <= 0:
        max_length = int(current_length)
    max_length = max(1, min(int(current_length), max_length))
    min_length = min(min_length, max_length)
    random_mean = 0.5 * float(min_length + max_length)
    return (1.0 - prob) * float(current_length) + prob * random_mean


def _maybe_apply_random_length_training(batch: Dict, config, generator: torch.Generator):
    """SMDM-style stochastic sequence length for packed pretraining batches."""
    input_ids = batch.get("input_ids")
    if input_ids is None or input_ids.ndim < 2:
        return batch, {}
    device = input_ids.device
    batch_size, current_length = int(input_ids.shape[0]), int(input_ids.shape[1])
    stats = {
        "sequence_length": torch.as_tensor(float(current_length), device=device),
        "random_length_applied": torch.as_tensor(0.0, device=device),
    }
    if str(getattr(config, "training_stage", "pretrain") or "pretrain").lower() != "pretrain":
        return batch, stats
    if _is_sft_like_batch(batch):
        return batch, stats
    prob = max(0.0, min(1.0, float(getattr(config, "random_length_prob", 0.0) or 0.0)))
    if prob <= 0.0 or current_length <= 1:
        return batch, stats
    should_crop = bool((torch.rand((), device=device, generator=generator) < prob).item())
    if not should_crop:
        return batch, stats
    min_length = max(1, int(getattr(config, "random_length_min", 1) or 1))
    max_length = int(getattr(config, "random_length_max", 0) or 0)
    if max_length <= 0:
        max_length = current_length
    max_length = max(1, min(current_length, max_length))
    min_length = min(min_length, max_length)
    if min_length == max_length:
        length = max_length
    else:
        length = int(torch.randint(min_length, max_length + 1, (1,), device=device, generator=generator).item())
    if length >= current_length:
        return batch, stats
    cropped = {}
    for key, value in batch.items():
        if torch.is_tensor(value) and value.ndim >= 2 and int(value.shape[0]) == batch_size and int(value.shape[1]) == current_length:
            cropped[key] = value[:, :length, ...].contiguous()
        else:
            cropped[key] = value
    stats = {
        "sequence_length": torch.as_tensor(float(length), device=device),
        "random_length_applied": torch.as_tensor(1.0, device=device),
    }
    return cropped, stats


def _maybe_apply_sft_dynamic_crop(batch: Dict, config):
    """Crop SFT microbatches to the DDP-global max true length."""
    input_ids = batch.get("input_ids")
    true_lengths = batch.get("true_lengths")
    if input_ids is None or true_lengths is None or input_ids.ndim < 2:
        return batch, {}
    if not _is_sft_like_batch(batch):
        return batch, {}
    device = input_ids.device
    batch_size, current_length = int(input_ids.shape[0]), int(input_ids.shape[1])
    true_lengths = true_lengths.to(device=device).long().clamp(min=1, max=current_length)
    local_max = true_lengths.max() if true_lengths.numel() else torch.as_tensor(current_length, device=device)
    global_max = local_max.clone()
    if distributed_available():
        dist.all_reduce(global_max, op=dist.ReduceOp.MAX)
    crop_multiple = int(getattr(config, "sft_dynamic_crop_multiple", 64) or 1)
    crop_multiple = max(1, crop_multiple)
    crop_len = int(global_max.detach().cpu().item())
    crop_len = int(math.ceil(float(crop_len) / float(crop_multiple)) * crop_multiple)
    crop_len = max(1, min(current_length, crop_len))
    enabled = bool(getattr(config, "sft_dynamic_crop", False))
    target_length = crop_len if enabled else current_length

    if target_length < current_length:
        cropped = {}
        for key, value in batch.items():
            if torch.is_tensor(value) and value.ndim >= 2 and int(value.shape[0]) == batch_size and int(value.shape[1]) == current_length:
                cropped[key] = value[:, :target_length, ...].contiguous()
            else:
                cropped[key] = value
        batch = cropped

    prompt_lengths = batch.get("prompt_lengths")
    if prompt_lengths is not None:
        prompt_lengths = prompt_lengths.to(device=device).long().clamp(min=0, max=target_length)
        response_true = (true_lengths.clamp(max=target_length) - prompt_lengths.clamp(max=target_length)).clamp_min(0)
    else:
        response_true = true_lengths.clamp(max=target_length)
    true_tokens = true_lengths.clamp(max=target_length).sum().to(dtype=torch.float32)
    total_slots = torch.as_tensor(float(batch_size * target_length), device=device)
    padded_tokens = (total_slots - true_tokens).clamp_min(0.0)
    response_padded = (target_length - true_lengths.clamp(max=target_length)).clamp_min(0).sum().to(dtype=torch.float32)

    stats = {
        "sft/batch_seq_len": torch.as_tensor(float(target_length), device=device),
        "sft/dynamic_crop_applied": torch.as_tensor(1.0 if target_length < current_length else 0.0, device=device),
        "sft/true_tokens_per_batch": true_tokens,
        "sft/padded_tokens_per_batch": padded_tokens,
        "sft/padding_fraction": padded_tokens / total_slots.clamp_min(1.0),
        "sft/response_true_tokens_per_batch": response_true.sum().to(dtype=torch.float32),
        "sft/response_padded_tokens_per_batch": response_padded,
    }
    source_id = batch.get("source_id")
    if torch.is_tensor(source_id) and source_id.numel() > 0:
        num_sources_value = batch.get("sft_num_sources")
        if torch.is_tensor(num_sources_value) and num_sources_value.numel() > 0:
            source_range = range(max(1, int(num_sources_value.detach().flatten()[0].cpu().item())))
        else:
            source_range = range(int(source_id.detach().long().max().cpu().item()) + 1)
        for source in source_range:
            mask = source_id.long() == int(source)
            source_true_tokens = true_lengths.clamp(max=target_length)[mask].sum().to(dtype=torch.float32)
            stats[f"sft/source_{int(source)}_step_fraction"] = mask.float().mean()
            stats[f"sft/source_{int(source)}_true_tokens_per_batch"] = source_true_tokens
            stats[f"sft/source_{int(source)}_true_token_fraction"] = source_true_tokens / true_tokens.clamp_min(1.0)
    return batch, stats


def _aggregate_train_metrics(pending_metrics, config) -> Dict[str, float]:
    reduced = [reduce_metrics(_without_sample_metrics(metrics)) for metrics in pending_metrics]
    avg_loss = sum(_scalar_from_metric(metrics, "loss") for metrics in reduced) / max(len(reduced), 1)
    avg_ce = sum(_scalar_from_metric(metrics, "ce_loss", default=_scalar_from_metric(metrics, "loss")) for metrics in reduced) / max(len(reduced), 1)
    avg_corrupt = sum(_scalar_from_metric(metrics, "corrupt_fraction") for metrics in reduced) / max(len(reduced), 1)
    record = {
        "loss": avg_loss,
        "ce": avg_ce,
        "ce_loss": avg_ce,
        "corrupt_fraction": avg_corrupt,
    }
    for key in (
        "mse_loss",
        "mse_weighted_loss",
        "lambda_mse",
        "mean_t",
        "mean_gamma",
        "mean_alpha",
        "mean_sigma",
        "mean_mask_ratio",
        "token_acc",
        "entropy",
        "confidence",
        "is_sft_batch",
        "packed_batch_fraction",
        "sft_batch_fraction",
        "sft_full_response_fraction",
        "sft_response_tokens",
        "sft_general_fraction",
        "sft_math_fraction",
        "sft_code_fraction",
        "sft_ce_loss",
        "packed_ce_loss",
        "sequence_length",
        "random_length_applied",
    ):
        values = [_scalar_from_metric(metrics, key, default=float("nan")) for metrics in reduced if key in metrics]
        if values:
            record[key] = sum(values) / len(values)
    dynamic_prefixes = ("sft/",)
    dynamic_keys = sorted(
        {
            key
            for metrics in reduced
            for key in metrics.keys()
            if key.startswith(dynamic_prefixes) and key not in record
        }
    )
    for key in dynamic_keys:
        values = [_scalar_from_metric(metrics, key, default=float("nan")) for metrics in reduced if key in metrics]
        values = [value for value in values if math.isfinite(value)]
        if values:
            record[key] = sum(values) / len(values)

    world_size = get_world_size()
    for metric_key, record_key in (
        ("total_tokens", "train/window_total_tokens"),
        ("valid_tokens", "train/window_valid_tokens"),
        ("corrupt_tokens", "train/window_corrupt_tokens"),
        ("sft_response_tokens", "train/window_sft_response_tokens"),
    ):
        if reduced and metric_key in reduced[0]:
            record[record_key] = sum(_scalar_from_metric(metrics, metric_key) for metrics in reduced) * world_size

    return record


def _initial_token_counters() -> Dict[str, float]:
    return {
        "train/tokens_seen": 0.0,
        "train/valid_tokens_seen": 0.0,
        "train/corrupt_tokens_seen": 0.0,
        "train/sft_response_tokens_seen": 0.0,
    }


def _update_token_counters(token_counters: Dict[str, float], aggregate: Dict[str, float], elapsed_seconds: float):
    total_window = float(aggregate.get("train/window_total_tokens", 0.0))
    valid_window = float(aggregate.get("train/window_valid_tokens", 0.0))
    corrupt_window = float(aggregate.get("train/window_corrupt_tokens", 0.0))
    sft_response_window = float(aggregate.get("train/window_sft_response_tokens", 0.0))
    token_counters["train/tokens_seen"] = float(token_counters.get("train/tokens_seen", 0.0)) + total_window
    token_counters["train/valid_tokens_seen"] = float(token_counters.get("train/valid_tokens_seen", 0.0)) + valid_window
    token_counters["train/corrupt_tokens_seen"] = float(token_counters.get("train/corrupt_tokens_seen", 0.0)) + corrupt_window
    token_counters["train/sft_response_tokens_seen"] = (
        float(token_counters.get("train/sft_response_tokens_seen", 0.0)) + sft_response_window
    )
    aggregate.update(token_counters)
    aggregate["train/billions_tokens_seen"] = token_counters["train/tokens_seen"] / 1e9
    aggregate["train/billions_valid_tokens_seen"] = token_counters["train/valid_tokens_seen"] / 1e9
    aggregate["train/billions_corrupt_tokens_seen"] = token_counters["train/corrupt_tokens_seen"] / 1e9
    aggregate["train/billions_sft_response_tokens_seen"] = token_counters["train/sft_response_tokens_seen"] / 1e9
    aggregate["train/tokens_per_sec"] = total_window / max(float(elapsed_seconds), 1e-8)


def _metric_vector(metrics: Dict, key: str):
    value = metrics.get(key)
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.detach().float().flatten().cpu().tolist()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [float(item) for item in value]
    return [float(value)]


def _loss_diagnostic_mask_edges(config) -> list:
    p_min = float(getattr(config, "mask_p_min", 0.05))
    p_max = float(getattr(config, "mask_p_max", 1.0))
    width = float(getattr(config, "loss_diagnostics_mask_bin_width", 0.05))
    if not (0.0 <= p_min < p_max <= 1.0):
        raise ValueError(f"Expected 0 <= mask_p_min < mask_p_max <= 1, got {p_min}, {p_max}")
    if width <= 0.0:
        raise ValueError(f"loss_diagnostics_mask_bin_width must be positive, got {width}")
    bin_count = max(1, int(math.ceil((p_max - p_min) / width - 1e-12)))
    edges = [round(p_min + idx * width, 12) for idx in range(bin_count)]
    edges.append(p_max)
    return edges


def _loss_diagnostic_target_samples(config) -> int:
    return max(1, int(getattr(config, "loss_diagnostics_target_samples_per_cell", 200) or 200))


def _loss_diagnostic_estimator(config) -> str:
    estimator = str(getattr(config, "loss_diagnostic_estimator", "window") or "window").lower()
    if estimator in {"window", "rolling", "rolling_window"}:
        return "window"
    if estimator in {"ema", "exp", "exponential", "exponential_moving_average"}:
        return "ema"
    raise ValueError(f"Unknown loss_diagnostic_estimator: {estimator}. Expected 'window' or 'ema'.")


def _count_value(value: float):
    value = float(value)
    rounded = round(value)
    if abs(value - rounded) <= 1e-9:
        return int(rounded)
    return value


def _loss_diagnostic_plot_min_samples(config) -> int:
    return max(2, _loss_diagnostic_target_samples(config) // 10)


def _loss_diagnostic_batch_size(config) -> int:
    batch_size = getattr(config, "global_batch_size", None)
    if batch_size is None:
        batch_size = getattr(config, "batch_size", 1)
    return max(1, int(batch_size or 1))


def _loss_diagnostic_effective_batches(config) -> float:
    estimator = _loss_diagnostic_estimator(config)
    if estimator == "ema":
        decay = float(getattr(config, "loss_diagnostic_ema_decay", 0.98) or 0.98)
        decay = min(max(decay, 0.0), 0.999999)
        log_freq = max(1, int(getattr(config, "log_freq", 20) or 20))
        grad_accum = max(1, int(getattr(config, "grad_accum_steps", 1) or 1))
        return float(log_freq * grad_accum) / max(1.0 - decay, 1e-12)
    window_batches = int(getattr(config, "sample_diagnostics_window_batches", 100) or 100)
    return float(max(1, window_batches))


def _loss_diagnostic_effective_examples(config) -> int:
    return max(1, int(round(float(_loss_diagnostic_batch_size(config)) * _loss_diagnostic_effective_batches(config))))


def _loss_diagnostic_window_examples(config) -> int:
    return _loss_diagnostic_effective_examples(config)


def _loss_diagnostic_gamma_bin_count(config, mask_bin_count: Optional[int] = None) -> int:
    configured = int(getattr(config, "loss_diagnostics_gamma_bins", 0) or 0)
    if configured > 0:
        return max(2, configured)
    mask_bin_count = int(mask_bin_count or len(_loss_diagnostic_mask_edges(config)) - 1)
    raw = math.floor(
        _loss_diagnostic_effective_examples(config)
        / max(mask_bin_count * _loss_diagnostic_target_samples(config), 1)
    )
    return max(2, int(raw))


def _loss_diagnostic_gamma_edges(config, mask_bin_count: Optional[int] = None) -> list:
    gamma_min = float(getattr(config, "gamma_min", -6.0))
    gamma_max = float(getattr(config, "gamma_max", 6.0))
    if not gamma_min < gamma_max:
        raise ValueError(f"Expected gamma_min < gamma_max, got {gamma_min}, {gamma_max}")
    bin_count = _loss_diagnostic_gamma_bin_count(config, mask_bin_count=mask_bin_count)
    return gamma_non_fitted_quantile_edges(config, bin_count)


def _loss_diagnostic_grid(config) -> Dict:
    mask_edges = _loss_diagnostic_mask_edges(config)
    gamma_edges = _loss_diagnostic_gamma_edges(config, mask_bin_count=len(mask_edges) - 1)
    return {
        "mask_edges": mask_edges,
        "gamma_edges": gamma_edges,
        "mask_bins": len(mask_edges) - 1,
        "gamma_bins": len(gamma_edges) - 1,
        "target_samples_per_cell": _loss_diagnostic_target_samples(config),
        "min_plot_samples": _loss_diagnostic_plot_min_samples(config),
        "effective_batches": _loss_diagnostic_effective_batches(config),
        "effective_examples": _loss_diagnostic_effective_examples(config),
        "window_examples": _loss_diagnostic_window_examples(config),
        "configured_gamma_bins": int(getattr(config, "loss_diagnostics_gamma_bins", 0) or 0),
    }


def _config_like(config):
    if isinstance(config, dict):
        return SimpleNamespace(**config)
    return config


def _rounded_edge_list(edges: Sequence[float]) -> list:
    return [round(float(edge), 12) for edge in edges]


def _loss_diagnostic_grid_signature(config) -> Dict:
    config = _config_like(config)
    grid = _loss_diagnostic_grid(config)
    return {
        "version": 1,
        "mask_edges": _rounded_edge_list(grid["mask_edges"]),
        "gamma_edges": _rounded_edge_list(grid["gamma_edges"]),
    }


def _loss_diagnostic_grid_signatures_match(left: Optional[Dict], right: Optional[Dict]) -> bool:
    if not left or not right:
        return False
    for key in ("mask_edges", "gamma_edges"):
        left_edges = left.get(key) or []
        right_edges = right.get(key) or []
        if len(left_edges) != len(right_edges):
            return False
        for left_edge, right_edge in zip(left_edges, right_edges):
            if not math.isclose(float(left_edge), float(right_edge), rel_tol=0.0, abs_tol=1e-10):
                return False
    return True


def _remap_loss_diagnostic_summary_grid(summary: Optional[Dict], source_signature: Dict, target_signature: Dict) -> Optional[Dict]:
    normalized = _normalize_loss_diagnostic_batch_summary(summary)
    if normalized is None:
        return None
    if _loss_diagnostic_grid_signatures_match(source_signature, target_signature):
        return normalized

    source_mask_edges = [float(edge) for edge in source_signature.get("mask_edges") or []]
    source_gamma_edges = [float(edge) for edge in source_signature.get("gamma_edges") or []]
    target_mask_edges = [float(edge) for edge in target_signature.get("mask_edges") or []]
    target_gamma_edges = [float(edge) for edge in target_signature.get("gamma_edges") or []]
    if len(source_mask_edges) < 2 or len(source_gamma_edges) < 2 or len(target_mask_edges) < 2 or len(target_gamma_edges) < 2:
        return None

    cells = {}
    for raw_key, value in normalized["cells"].items():
        mask_idx, gamma_idx = int(raw_key[0]), int(raw_key[1])
        if mask_idx < 0 or mask_idx + 1 >= len(source_mask_edges):
            continue
        if gamma_idx < 0 or gamma_idx + 1 >= len(source_gamma_edges):
            continue
        mask_mid = 0.5 * (source_mask_edges[mask_idx] + source_mask_edges[mask_idx + 1])
        gamma_mid = 0.5 * (source_gamma_edges[gamma_idx] + source_gamma_edges[gamma_idx + 1])
        new_mask_idx = _loss_diagnostic_bin_index(mask_mid, target_mask_edges)
        new_gamma_idx = _loss_diagnostic_bin_index(gamma_mid, target_gamma_edges)
        if new_mask_idx is None or new_gamma_idx is None:
            continue
        target = cells.setdefault((new_mask_idx, new_gamma_idx), [0.0, 0.0, 0.0])
        target[0] += float(value[0])
        target[1] += float(value[1])
        target[2] += float(value[2])

    if not cells:
        return None
    return _make_loss_diagnostic_batch_summary(
        cells,
        examples=float(normalized.get("examples", 0) or 0),
        total_examples=float(normalized.get("total_examples", 0) or 0),
    )


def _loss_diagnostic_bin_index(value: float, edges: Sequence[float]) -> Optional[int]:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    if value < edges[0] or value > edges[-1]:
        return None
    if math.isclose(value, edges[-1], rel_tol=0.0, abs_tol=1e-12):
        return len(edges) - 2
    idx = bisect_right(edges, value) - 1
    if idx < 0 or idx >= len(edges) - 1:
        return None
    return idx


def _make_loss_diagnostic_batch_summary(cells: Dict, examples: int, total_examples: int) -> Dict:
    return {
        "cells": cells,
        "examples": _count_value(examples),
        "total_examples": _count_value(total_examples),
    }


def _normalize_loss_diagnostic_batch_summary(summary: Optional[Dict]) -> Optional[Dict]:
    if not summary:
        return None
    cells = {}
    for raw_key, raw_value in (summary.get("cells") or {}).items():
        if isinstance(raw_key, str):
            pieces = [piece.strip() for piece in raw_key.replace("(", "").replace(")", "").split(",") if piece.strip()]
            if len(pieces) != 2:
                continue
            key = (int(pieces[0]), int(pieces[1]))
        elif isinstance(raw_key, Sequence) and not isinstance(raw_key, (bytes, bytearray)):
            if len(raw_key) != 2:
                continue
            key = (int(raw_key[0]), int(raw_key[1]))
        else:
            continue
        if not isinstance(raw_value, Sequence) or isinstance(raw_value, (str, bytes, bytearray)) or len(raw_value) < 3:
            continue
        n = float(raw_value[0])
        sum_ce = float(raw_value[1])
        sum_ce2 = float(raw_value[2])
        if n <= 0 or not math.isfinite(sum_ce) or not math.isfinite(sum_ce2):
            continue
        cells[key] = [_count_value(n), sum_ce, sum_ce2]
    if not cells:
        return None
    return _make_loss_diagnostic_batch_summary(
        cells,
        examples=float(summary.get("examples", 0) or 0),
        total_examples=float(summary.get("total_examples", 0) or 0),
    )


def _empty_loss_diagnostic_core_state(config) -> Dict:
    return {
        "batch_window": [],
        "ema_summary": None,
        "ema_updates": 0,
        "estimator": _loss_diagnostic_estimator(config),
        "batches_since_log": 0,
    }


def _serialize_loss_diagnostic_core_state(state: Optional[Dict], config=None) -> Optional[Dict]:
    if not state:
        return None
    window = []
    for summary in state.get("batch_window") or []:
        normalized = _normalize_loss_diagnostic_batch_summary(summary)
        if normalized is not None:
            window.append(normalized)
    ema_summary = _normalize_loss_diagnostic_batch_summary(state.get("ema_summary"))
    payload = {
        "version": 2,
        "estimator": str(state.get("estimator", "window")),
        "batch_window": window,
        "ema_summary": ema_summary,
        "ema_updates": int(state.get("ema_updates", 0) or 0),
        "batches_since_log": int(state.get("batches_since_log", 0) or 0),
    }
    if config is not None:
        payload["grid_signature"] = _loss_diagnostic_grid_signature(config)
    return payload


def _empty_loss_diagnostic_state(config) -> Dict:
    state = _empty_loss_diagnostic_core_state(config)
    state["no_eos"] = _empty_loss_diagnostic_core_state(config)
    return state


def _serialize_loss_diagnostic_state(state: Optional[Dict], config=None) -> Optional[Dict]:
    payload = _serialize_loss_diagnostic_core_state(state, config=config)
    if payload is None:
        return None
    no_eos_payload = _serialize_loss_diagnostic_core_state(state.get("no_eos"), config=config)
    if no_eos_payload is not None:
        payload["no_eos"] = no_eos_payload
    return payload


def _deserialize_loss_diagnostic_core_state(state: Optional[Dict], config, saved_config=None) -> Optional[Dict]:
    if not state:
        return None
    estimator = _loss_diagnostic_estimator(config)
    current_signature = _loss_diagnostic_grid_signature(config)
    saved_signature = state.get("grid_signature")
    if saved_signature is None and saved_config:
        try:
            saved_signature = _loss_diagnostic_grid_signature(saved_config)
        except Exception as exc:
            log_for_0(f"Could not infer saved loss diagnostic grid; dropping restored diagnostic state: {exc}", level=logging.WARNING)
            return _empty_loss_diagnostic_core_state(config)
    if saved_signature is None:
        log_for_0(
            "Checkpoint loss diagnostic state has no grid metadata; dropping it to avoid reusing stale bin indices.",
            level=logging.WARNING,
        )
        return _empty_loss_diagnostic_core_state(config)
    grid_changed = not _loss_diagnostic_grid_signatures_match(saved_signature, current_signature)
    if grid_changed:
        log_for_0(
            "Loss diagnostic grid changed across checkpoint resume; remapping restored diagnostic bins "
            f"from gamma_bins={len(saved_signature.get('gamma_edges') or []) - 1} "
            f"to gamma_bins={len(current_signature.get('gamma_edges') or []) - 1}.",
            level=logging.WARNING,
        )
    window = []
    for summary in state.get("batch_window") or []:
        normalized = _remap_loss_diagnostic_summary_grid(summary, saved_signature, current_signature)
        if normalized is not None:
            window.append(normalized)
    if estimator != "ema":
        max_batches = int(getattr(config, "sample_diagnostics_window_batches", 100) or 100)
    else:
        max_batches = 0
    if max_batches > 0 and len(window) > max_batches:
        window = window[-max_batches:]
    ema_summary = _remap_loss_diagnostic_summary_grid(state.get("ema_summary"), saved_signature, current_signature)
    if estimator == "ema" and ema_summary is None and window:
        ema_summary = _aggregate_loss_diagnostic_window(window)
        window = []
    return {
        "batch_window": window,
        "ema_summary": ema_summary,
        "ema_updates": int(state.get("ema_updates", 0) or 0),
        "estimator": estimator,
        "batches_since_log": int(state.get("batches_since_log", 0) or 0),
    }


def _deserialize_loss_diagnostic_state(state: Optional[Dict], config, saved_config=None) -> Optional[Dict]:
    restored = _deserialize_loss_diagnostic_core_state(state, config, saved_config=saved_config)
    if restored is None:
        return None
    no_eos_state = _deserialize_loss_diagnostic_core_state(
        (state or {}).get("no_eos"),
        config,
        saved_config=saved_config,
    )
    restored["no_eos"] = no_eos_state or _empty_loss_diagnostic_core_state(config)
    return restored


def _merge_loss_diagnostic_batch_summaries(summaries: Sequence[Optional[Dict]]) -> Optional[Dict]:
    cells = {}
    examples = 0
    total_examples = 0
    for summary in summaries:
        if not summary:
            continue
        examples += float(summary.get("examples", 0))
        total_examples += float(summary.get("total_examples", 0))
        for key, value in summary.get("cells", {}).items():
            n, sum_ce, sum_ce2 = value
            target = cells.setdefault(tuple(key), [0, 0.0, 0.0])
            target[0] += float(n)
            target[1] += float(sum_ce)
            target[2] += float(sum_ce2)
    if not cells:
        return None
    return _make_loss_diagnostic_batch_summary(cells, examples, total_examples)


def _scale_loss_diagnostic_summary(summary: Optional[Dict], scale: float) -> Optional[Dict]:
    normalized = _normalize_loss_diagnostic_batch_summary(summary)
    if normalized is None:
        return None
    scale = float(scale)
    if scale <= 0.0:
        return None
    cells = {}
    for key, value in normalized["cells"].items():
        cells[tuple(key)] = [
            float(value[0]) * scale,
            float(value[1]) * scale,
            float(value[2]) * scale,
        ]
    return _make_loss_diagnostic_batch_summary(
        cells,
        examples=float(normalized.get("examples", 0) or 0) * scale,
        total_examples=float(normalized.get("total_examples", 0) or 0) * scale,
    )


def _update_loss_diagnostic_ema(state: Dict, new_summary: Optional[Dict], config):
    if new_summary is None:
        return
    decay = float(getattr(config, "loss_diagnostic_ema_decay", 0.98) or 0.98)
    decay = min(max(decay, 0.0), 0.999999)
    existing = _normalize_loss_diagnostic_batch_summary(state.get("ema_summary"))
    if existing is None and state.get("batch_window"):
        existing = _aggregate_loss_diagnostic_window(state.get("batch_window") or [])
    if existing is None:
        updated = _normalize_loss_diagnostic_batch_summary(new_summary)
    else:
        updated = _merge_loss_diagnostic_batch_summaries(
            [
                _scale_loss_diagnostic_summary(existing, decay),
                new_summary,
            ]
        )
    state["ema_summary"] = updated
    state["ema_updates"] = int(state.get("ema_updates", 0) or 0) + 1
    state["batch_window"] = []


def _update_loss_diagnostic_state(state: Dict, batch_summaries: list, config) -> int:
    if not batch_summaries:
        state["estimator"] = _loss_diagnostic_estimator(config)
        return 0
    estimator = _loss_diagnostic_estimator(config)
    state["estimator"] = estimator
    added = len(batch_summaries)
    if estimator == "ema":
        new_summary = _merge_loss_diagnostic_batch_summaries(batch_summaries)
        _update_loss_diagnostic_ema(state, new_summary, config)
    else:
        _extend_loss_diagnostic_batch_window(
            state.setdefault("batch_window", []),
            batch_summaries,
            max_batches=int(getattr(config, "sample_diagnostics_window_batches", 100) or 100),
        )
    state["batches_since_log"] = int(state.get("batches_since_log", 0) or 0) + added
    return added


def _loss_diagnostic_batches_for_fit(state: Optional[Dict], config) -> list:
    if not state:
        return []
    if _loss_diagnostic_estimator(config) == "ema":
        ema_summary = _normalize_loss_diagnostic_batch_summary(state.get("ema_summary"))
        if ema_summary is not None:
            return [ema_summary]
        if state.get("batch_window"):
            return [_aggregate_loss_diagnostic_window(state.get("batch_window") or [])]
        return []
    return list(state.get("batch_window") or [])


def _loss_diagnostic_no_eos_state(state: Dict, config) -> Dict:
    no_eos = state.get("no_eos")
    if not isinstance(no_eos, dict):
        no_eos = _empty_loss_diagnostic_core_state(config)
        state["no_eos"] = no_eos
    return no_eos


def _sample_loss_diagnostic_batches_from_pending(pending_metrics, config, ce_metric_key: str = "sample_ce") -> list:
    grid = _loss_diagnostic_grid(config)
    batches = []
    for metrics in pending_metrics:
        gamma = _metric_vector(metrics, "sample_gamma")
        ce = _metric_vector(metrics, ce_metric_key)
        mask_ratio = _metric_vector(metrics, "sample_mask_ratio")
        if gamma is None or ce is None or mask_ratio is None:
            continue
        total_examples = min(len(gamma), len(ce), len(mask_ratio))
        cells = {}
        examples = 0
        for idx in range(total_examples):
            mask_idx = _loss_diagnostic_bin_index(mask_ratio[idx], grid["mask_edges"])
            gamma_idx = _loss_diagnostic_bin_index(gamma[idx], grid["gamma_edges"])
            try:
                ce_value = float(ce[idx])
            except (TypeError, ValueError):
                continue
            if mask_idx is None or gamma_idx is None or not math.isfinite(ce_value):
                continue
            key = (mask_idx, gamma_idx)
            stat = cells.setdefault(key, [0, 0.0, 0.0])
            stat[0] += 1
            stat[1] += ce_value
            stat[2] += ce_value * ce_value
            examples += 1
        if cells:
            batches.append(_make_loss_diagnostic_batch_summary(cells, examples, total_examples))
    return batches


def _gather_loss_diagnostic_batches(local_batches: list) -> list:
    if not distributed_available():
        return local_batches if is_main_process() else []
    gathered = [None for _ in range(get_world_size())]
    dist.all_gather_object(gathered, local_batches)
    if not is_main_process():
        return []
    batches = []
    max_len = max((len(shard or []) for shard in gathered), default=0)
    for batch_idx in range(max_len):
        merged = _merge_loss_diagnostic_batch_summaries(
            shard[batch_idx] if shard and batch_idx < len(shard) else None for shard in gathered
        )
        if merged:
            batches.append(merged)
    return batches


def _extend_loss_diagnostic_batch_window(window: list, batch_summaries: list, max_batches: int) -> int:
    if not batch_summaries:
        return 0
    window.extend(batch_summaries)
    if len(window) > max_batches:
        del window[: len(window) - max_batches]
    return len(batch_summaries)


def _aggregate_loss_diagnostic_window(batch_window: list) -> Dict:
    merged = _merge_loss_diagnostic_batch_summaries(batch_window)
    return merged or _make_loss_diagnostic_batch_summary({}, 0, 0)


def _loss_cell_stats(n: float, sum_ce: float, sum_ce2: float) -> tuple[float, float, float, float]:
    n = float(n)
    if n <= 0:
        return (float("nan"), float("nan"), float("nan"), float("nan"))
    mean = sum_ce / float(n)
    if n > 1:
        variance = max((sum_ce2 - (sum_ce * sum_ce) / float(n)) / float(n - 1), 0.0)
    else:
        variance = 0.0
    se = math.sqrt(variance / float(n))
    delta = 1.96 * se
    return mean, mean - delta, mean + delta, se


def _loss_diagnostic_summary_rows(batch_window: list, config, step: Optional[int] = None) -> tuple[list, Dict]:
    grid = _loss_diagnostic_grid(config)
    aggregate = _aggregate_loss_diagnostic_window(batch_window)
    cells = aggregate["cells"]
    rows = []
    rows_by_key = {}

    for mask_idx in range(grid["mask_bins"]):
        mask_left = float(grid["mask_edges"][mask_idx])
        mask_right = float(grid["mask_edges"][mask_idx + 1])
        mask_label = f"{mask_left:.2f}-{mask_right:.2f}"
        for gamma_idx in range(grid["gamma_bins"]):
            n, sum_ce, sum_ce2 = cells.get((mask_idx, gamma_idx), [0, 0.0, 0.0])
            if n <= 0:
                continue
            gamma_left = float(grid["gamma_edges"][gamma_idx])
            gamma_right = float(grid["gamma_edges"][gamma_idx + 1])
            ce_mean, ce_low, ce_high, ce_se = _loss_cell_stats(float(n), float(sum_ce), float(sum_ce2))
            row = {
                "step": step,
                "mask_idx": mask_idx,
                "gamma_idx": gamma_idx,
                "mask_left": mask_left,
                "mask_right": mask_right,
                "mask_label": mask_label,
                "gamma_left": gamma_left,
                "gamma_right": gamma_right,
                "gamma_mid": 0.5 * (gamma_left + gamma_right),
                "n": _count_value(n),
                "sum_ce": float(sum_ce),
                "sum_ce2": float(sum_ce2),
                "ce_mean": ce_mean,
                "ce_ci_low": ce_low,
                "ce_ci_high": ce_high,
                "ce_se": ce_se,
                "dce_dgamma": float("nan"),
                "dce_ci_low": float("nan"),
                "dce_ci_high": float("nan"),
            }
            rows.append(row)
            rows_by_key[(mask_idx, gamma_idx)] = row

    min_samples = int(grid["min_plot_samples"])
    for mask_idx in range(grid["mask_bins"]):
        for gamma_idx in range(grid["gamma_bins"]):
            row = rows_by_key.get((mask_idx, gamma_idx))
            if row is None or row["n"] < min_samples:
                continue
            prev_row = rows_by_key.get((mask_idx, gamma_idx - 1))
            next_row = rows_by_key.get((mask_idx, gamma_idx + 1))
            prev_valid = prev_row is not None and prev_row["n"] >= min_samples
            next_valid = next_row is not None and next_row["n"] >= min_samples
            if prev_valid and next_valid:
                left_row, right_row = prev_row, next_row
            elif next_valid:
                left_row, right_row = row, next_row
            elif prev_valid:
                left_row, right_row = prev_row, row
            else:
                continue
            dx = float(right_row["gamma_mid"]) - float(left_row["gamma_mid"])
            if dx <= 0.0:
                continue
            derivative = (float(right_row["ce_mean"]) - float(left_row["ce_mean"])) / dx
            derivative_se = math.sqrt(float(left_row["ce_se"]) ** 2 + float(right_row["ce_se"]) ** 2) / dx
            derivative_delta = 1.96 * derivative_se
            row["dce_dgamma"] = derivative
            row["dce_ci_low"] = derivative - derivative_delta
            row["dce_ci_high"] = derivative + derivative_delta

    rows.sort(key=lambda item: (item["mask_idx"], item["gamma_idx"]))
    meta = {
        "window_rows": _count_value(float(aggregate.get("examples", 0) or 0)),
        "window_total_examples": _count_value(float(aggregate.get("total_examples", 0) or 0)),
        "window_batches": len(batch_window),
        "window_target_batches": int(getattr(config, "sample_diagnostics_window_batches", 100) or 100),
        "window_examples_target": int(grid["window_examples"]),
        "mask_bins": int(grid["mask_bins"]),
        "gamma_bins": int(grid["gamma_bins"]),
        "target_samples_per_cell": int(grid["target_samples_per_cell"]),
        "min_plot_samples": int(grid["min_plot_samples"]),
        "effective_batches": _count_value(float(grid["effective_batches"])),
        "effective_examples": int(grid["effective_examples"]),
        "configured_gamma_bins": int(grid["configured_gamma_bins"]),
    }
    return rows, meta


def _finite_float(value) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _table_value(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _loss_diagnostic_table(wandb, rows: list, step: int):
    columns = [
        "step",
        "mask_left",
        "mask_right",
        "mask_label",
        "gamma_left",
        "gamma_right",
        "gamma_mid",
        "n",
        "ce_mean",
        "ce_ci_low",
        "ce_ci_high",
        "dce_dgamma",
        "dce_ci_low",
        "dce_ci_high",
    ]
    data = [
        [
            step,
            row["mask_left"],
            row["mask_right"],
            row["mask_label"],
            row["gamma_left"],
            row["gamma_right"],
            row["gamma_mid"],
            row["n"],
            _table_value(row["ce_mean"]),
            _table_value(row["ce_ci_low"]),
            _table_value(row["ce_ci_high"]),
            _table_value(row["dce_dgamma"]),
            _table_value(row["dce_ci_low"]),
            _table_value(row["dce_ci_high"]),
        ]
        for row in rows
    ]
    return wandb.Table(columns=columns, data=data)


def _rgba(hex_color: str, alpha: float) -> str:
    hex_color = hex_color.lstrip("#")
    red = int(hex_color[0:2], 16)
    green = int(hex_color[2:4], 16)
    blue = int(hex_color[4:6], 16)
    return f"rgba({red},{green},{blue},{float(alpha):.3f})"


def _loss_diagnostic_plotly_figure(rows: list, y_key: str, low_key: str, high_key: str, title: str, y_title: str):
    import plotly.graph_objects as go

    palette = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]
    fig = go.Figure()
    labels = sorted({row["mask_label"] for row in rows})
    for label_idx, label in enumerate(labels):
        group = [
            row
            for row in rows
            if row["mask_label"] == label
            and _finite_float(row.get(y_key))
            and _finite_float(row.get(low_key))
            and _finite_float(row.get(high_key))
        ]
        if not group:
            continue
        group = sorted(group, key=lambda row: row["gamma_mid"])
        x = [float(row["gamma_mid"]) for row in group]
        y = [float(row[y_key]) for row in group]
        low = [float(row[low_key]) for row in group]
        high = [float(row[high_key]) for row in group]
        color = palette[label_idx % len(palette)]
        fig.add_trace(
            go.Scatter(
                x=x + x[::-1],
                y=high + low[::-1],
                fill="toself",
                fillcolor=_rgba(color, 0.14),
                line={"color": "rgba(0,0,0,0)"},
                hoverinfo="skip",
                showlegend=False,
                name=f"{label} CI",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines+markers",
                line={"color": color, "width": 2},
                marker={"size": 4},
                name=label,
            )
        )
    fig.update_layout(
        title=title,
        xaxis_title="gamma",
        yaxis_title=y_title,
        hovermode="x unified",
        legend_title="mask_p interval",
        template="plotly_white",
    )
    return fig


def _aggregate_loss_diagnostic_ce_by_gamma(rows: list, min_samples: int) -> list:
    by_gamma = {}
    for row in rows:
        key = int(row["gamma_idx"])
        target = by_gamma.setdefault(
            key,
            {
                "gamma_idx": key,
                "gamma_mid": float(row["gamma_mid"]),
                "n": 0,
                "sum_ce": 0.0,
                "sum_ce2": 0.0,
            },
        )
        target["n"] += float(row["n"])
        target["sum_ce"] += float(row["sum_ce"])
        target["sum_ce2"] += float(row["sum_ce2"])

    aggregate_rows = []
    for row in by_gamma.values():
        if float(row["n"]) < float(min_samples):
            continue
        ce_mean, ce_low, ce_high, ce_se = _loss_cell_stats(
            float(row["n"]),
            float(row["sum_ce"]),
            float(row["sum_ce2"]),
        )
        aggregate_rows.append(
            {
                **row,
                "ce_mean": ce_mean,
                "ce_ci_low": ce_low,
                "ce_ci_high": ce_high,
                "ce_se": ce_se,
            }
        )
    return sorted(aggregate_rows, key=lambda item: item["gamma_mid"])


def _loss_diagnostic_original_bin_average_metrics(batch_window: list, config, prefix: str = "train") -> Dict[str, float]:
    """Equal-weight CE summaries over the fixed diagnostic bins.

    These are progress metrics, not adaptive-fit metrics. They intentionally
    average each populated original bin equally so changes in adaptive gamma
    sampling density do not dominate the plotted training curve.
    """
    if not batch_window:
        return {}
    rows, meta = _loss_diagnostic_summary_rows(batch_window, config)
    min_samples = int(meta["min_plot_samples"])
    valid_rows = [
        row
        for row in rows
        if float(row.get("n", 0) or 0) >= float(min_samples)
        and _finite_float(row.get("ce_mean"))
    ]
    if not valid_rows:
        return {}

    cell_avg = sum(float(row["ce_mean"]) for row in valid_rows) / float(len(valid_rows))
    cell_weight = sum(float(row["n"]) for row in valid_rows)
    sample_weighted = (
        sum(float(row["sum_ce"]) for row in valid_rows) / cell_weight
        if cell_weight > 0.0
        else float("nan")
    )

    gamma_rows = _aggregate_loss_diagnostic_ce_by_gamma(valid_rows, min_samples=min_samples)
    metrics = {
        f"{prefix}/loss_ce_original_bin_avg": float(cell_avg),
        f"{prefix}/loss_ce_original_bin_sample_weighted": float(sample_weighted),
        f"{prefix}/loss_ce_original_bin_count": float(len(valid_rows)),
    }
    if gamma_rows:
        gamma_avg = sum(float(row["ce_mean"]) for row in gamma_rows) / float(len(gamma_rows))
        metrics.update(
            {
                f"{prefix}/loss_ce_original_gamma_bin_avg": float(gamma_avg),
                f"{prefix}/loss_ce_original_gamma_bin_count": float(len(gamma_rows)),
            }
        )
    return metrics


def _normal_cdf_tensor(x: torch.Tensor, loc: float, scale: float) -> torch.Tensor:
    return 0.5 * (1.0 + torch.erf((x - float(loc)) / (float(scale) * math.sqrt(2.0))))


def _normal_pdf_tensor(x: torch.Tensor, loc: float, scale: float) -> torch.Tensor:
    z = (x - float(loc)) / float(scale)
    return torch.exp(-0.5 * z * z) / (float(scale) * math.sqrt(2.0 * math.pi))


def _gamma_curve_fit_family(config) -> str:
    family = str(getattr(config, "gamma_curve_estimator", "normal") or "normal").lower()
    if family in {"normal", "gaussian"}:
        return "normal"
    if family in {"generalized_logistic", "logistic", "genlogistic", "glogistic"}:
        return "generalized_logistic"
    if family in {"empirical", "piecewise", "piecewise_empirical", "empirical_piecewise"}:
        return "empirical"
    raise ValueError(f"Unknown gamma_curve_estimator: {family}")


def _generalized_logistic_cdf_tensor(x: torch.Tensor, loc: float, scale: float, shape: float) -> torch.Tensor:
    z = (x - float(loc)) / float(scale)
    return torch.sigmoid(z).pow(float(shape))


def _generalized_logistic_pdf_tensor(x: torch.Tensor, loc: float, scale: float, shape: float) -> torch.Tensor:
    z = (x - float(loc)) / float(scale)
    logistic = torch.sigmoid(z)
    return float(shape) * logistic.pow(float(shape)) * (1.0 - logistic) / float(scale)


def _normal_quantile_scalar(probability: float, loc: float, scale: float) -> float:
    probability = min(max(float(probability), 1e-6), 1.0 - 1e-6)
    q = torch.tensor(probability, dtype=torch.float64)
    z = math.sqrt(2.0) * torch.erfinv(2.0 * q - 1.0)
    return float(loc) + float(scale) * float(z.item())


def _generalized_logistic_quantile_scalar(probability: float, loc: float, scale: float, shape: float) -> float:
    probability = min(max(float(probability), 1e-6), 1.0 - 1e-6)
    return float(loc) - float(scale) * math.log(probability ** (-1.0 / float(shape)) - 1.0)


def _fit_quantile_scalar(fit: Dict[str, float], probability: float) -> float:
    family = str(fit.get("family", "normal"))
    if family == "generalized_logistic":
        return _generalized_logistic_quantile_scalar(
            probability,
            loc=float(fit["loc"]),
            scale=float(fit["scale"]),
            shape=float(fit.get("shape", 1.0)),
        )
    return _normal_quantile_scalar(probability, loc=float(fit["loc"]), scale=float(fit["scale"]))


def _curve_fit_dict(config_or_fit) -> Dict[str, float]:
    if isinstance(config_or_fit, dict):
        family = str(config_or_fit.get("family", "normal"))
        loc = float(config_or_fit.get("loc", config_or_fit.get("gamma_loc", 0.0)))
        scale = float(config_or_fit.get("scale", config_or_fit.get("gamma_scale", 1.0)))
        shape = float(config_or_fit.get("shape", 1.0))
        return {"family": family, "loc": loc, "scale": scale, "shape": shape}
    return {
        "family": _gamma_curve_fit_family(config_or_fit),
        "loc": float(getattr(config_or_fit, "gamma_loc", 0.0)),
        "scale": float(getattr(config_or_fit, "gamma_scale", 1.0)),
        "shape": 1.0,
    }


def _curve_fit_quantile_metrics(config_or_fit, prefix: str = "train/gamma_curve_fit") -> Dict[str, float]:
    fit = _curve_fit_dict(config_or_fit)
    return {
        f"{prefix}_q05": _fit_quantile_scalar(fit, 0.05),
        f"{prefix}_q10": _fit_quantile_scalar(fit, 0.10),
        f"{prefix}_q25": _fit_quantile_scalar(fit, 0.25),
        f"{prefix}_q50": _fit_quantile_scalar(fit, 0.50),
        f"{prefix}_q75": _fit_quantile_scalar(fit, 0.75),
        f"{prefix}_q90": _fit_quantile_scalar(fit, 0.90),
        f"{prefix}_q95": _fit_quantile_scalar(fit, 0.95),
    }


def _fit_cdf_tensor(x: torch.Tensor, fit: Dict[str, float]) -> torch.Tensor:
    family = str(fit.get("family", "normal"))
    if family == "generalized_logistic":
        return _generalized_logistic_cdf_tensor(
            x,
            loc=float(fit["loc"]),
            scale=float(fit["scale"]),
            shape=float(fit.get("shape", 1.0)),
        )
    return _normal_cdf_tensor(x, loc=float(fit["loc"]), scale=float(fit["scale"]))


def _fit_pdf_tensor(x: torch.Tensor, fit: Dict[str, float]) -> torch.Tensor:
    family = str(fit.get("family", "normal"))
    if family == "generalized_logistic":
        return _generalized_logistic_pdf_tensor(
            x,
            loc=float(fit["loc"]),
            scale=float(fit["scale"]),
            shape=float(fit.get("shape", 1.0)),
        )
    return _normal_pdf_tensor(x, loc=float(fit["loc"]), scale=float(fit["scale"]))


def _fit_label(fit: Dict[str, float]) -> str:
    return "generalized logistic" if str(fit.get("family", "normal")) == "generalized_logistic" else "normal"


def _fit_normal_cdf_to_ce_rows(ce_rows: list) -> Optional[Dict[str, float]]:
    valid = [
        row
        for row in ce_rows
        if _finite_float(row.get("gamma_mid"))
        and _finite_float(row.get("ce_mean"))
        and float(row.get("n", 0) or 0) > 0
    ]
    if len(valid) < 4:
        return None

    gamma = torch.tensor([float(row["gamma_mid"]) for row in valid], dtype=torch.float64)
    ce = torch.tensor([float(row["ce_mean"]) for row in valid], dtype=torch.float64)
    weight = torch.tensor([float(max(float(row["n"]), 1.0)) for row in valid], dtype=torch.float64)
    order = torch.argsort(gamma)
    gamma = gamma[order]
    ce = ce[order]
    weight = weight[order]

    ce_min = float(ce.min().item())
    ce_max = float(ce.max().item())
    ce_range = ce_max - ce_min
    if ce_range <= 1e-8:
        return None

    eps = 1e-4
    probability = ((ce - ce_min) / ce_range).clamp(eps, 1.0 - eps)
    probability = torch.cummax(probability, dim=0).values.clamp(eps, 1.0 - eps)
    probit = math.sqrt(2.0) * torch.erfinv(2.0 * probability - 1.0)

    weight = weight / weight.sum().clamp_min(1e-12)
    probit_mean = (weight * probit).sum()
    gamma_mean = (weight * gamma).sum()
    probit_var = (weight * (probit - probit_mean).square()).sum()
    if float(probit_var.item()) <= 1e-12:
        return None
    scale = (weight * (probit - probit_mean) * (gamma - gamma_mean)).sum() / probit_var
    if not torch.isfinite(scale) or float(scale.item()) <= 1e-8:
        return None
    loc = gamma_mean - scale * probit_mean

    fitted_probability = _normal_cdf_tensor(gamma, loc=float(loc.item()), scale=float(scale.item()))
    fitted_ce = ce_min + ce_range * fitted_probability
    ce_mean = (weight * ce).sum()
    sse = (weight * (ce - fitted_ce).square()).sum()
    sst = (weight * (ce - ce_mean).square()).sum()
    r2 = 1.0 - float((sse / sst.clamp_min(1e-12)).item())

    return {
        "family": "normal",
        "loc": float(loc.item()),
        "scale": float(scale.item()),
        "shape": 1.0,
        "ce_min": ce_min,
        "ce_max": ce_max,
        "r2": r2,
    }


def _fit_generalized_logistic_cdf_to_ce_rows(
    ce_rows: list,
    shape_min: float = 0.05,
    shape_max: float = 20.0,
) -> Optional[Dict[str, float]]:
    valid = [
        row
        for row in ce_rows
        if _finite_float(row.get("gamma_mid"))
        and _finite_float(row.get("ce_mean"))
        and float(row.get("n", 0) or 0) > 0
    ]
    if len(valid) < 4:
        return None

    gamma = torch.tensor([float(row["gamma_mid"]) for row in valid], dtype=torch.float64)
    ce = torch.tensor([float(row["ce_mean"]) for row in valid], dtype=torch.float64)
    weight = torch.tensor([float(max(float(row["n"]), 1.0)) for row in valid], dtype=torch.float64)
    order = torch.argsort(gamma)
    gamma = gamma[order]
    ce = ce[order]
    weight = weight[order]

    ce_min = float(ce.min().item())
    ce_max = float(ce.max().item())
    ce_range = ce_max - ce_min
    if ce_range <= 1e-8:
        return None

    eps = 1e-4
    probability = ((ce - ce_min) / ce_range).clamp(eps, 1.0 - eps)
    probability = torch.cummax(probability, dim=0).values.clamp(eps, 1.0 - eps)
    weight = weight / weight.sum().clamp_min(1e-12)
    ce_mean = (weight * ce).sum()
    sst = (weight * (ce - ce_mean).square()).sum().clamp_min(1e-12)

    shape_min = max(float(shape_min), 1e-4)
    shape_max = max(float(shape_max), shape_min)
    shape_grid = torch.exp(
        torch.linspace(math.log(shape_min), math.log(shape_max), 121, dtype=torch.float64)
    )
    best = None
    for shape in shape_grid:
        shape_value = float(shape.item())
        transformed = -torch.log((probability.pow(-1.0 / shape_value) - 1.0).clamp_min(1e-12))
        transformed_mean = (weight * transformed).sum()
        gamma_mean = (weight * gamma).sum()
        transformed_var = (weight * (transformed - transformed_mean).square()).sum()
        if float(transformed_var.item()) <= 1e-12:
            continue
        scale = (weight * (transformed - transformed_mean) * (gamma - gamma_mean)).sum() / transformed_var
        if not torch.isfinite(scale) or float(scale.item()) <= 1e-8:
            continue
        loc = gamma_mean - scale * transformed_mean
        fitted_probability = _generalized_logistic_cdf_tensor(
            gamma,
            loc=float(loc.item()),
            scale=float(scale.item()),
            shape=shape_value,
        )
        fitted_ce = ce_min + ce_range * fitted_probability
        sse = (weight * (ce - fitted_ce).square()).sum()
        r2 = 1.0 - float((sse / sst).item())
        candidate = {
            "family": "generalized_logistic",
            "loc": float(loc.item()),
            "scale": float(scale.item()),
            "shape": shape_value,
            "ce_min": ce_min,
            "ce_max": ce_max,
            "r2": r2,
            "sse": float(sse.item()),
        }
        if best is None or candidate["sse"] < best["sse"]:
            best = candidate

    if best is None:
        return None
    best.pop("sse", None)
    return best


def _fit_gamma_cdf_to_ce_rows(ce_rows: list, config) -> Optional[Dict[str, float]]:
    family = _gamma_curve_fit_family(config)
    if family == "empirical":
        return None
    if family == "generalized_logistic":
        return _fit_generalized_logistic_cdf_to_ce_rows(
            ce_rows,
            shape_min=float(getattr(config, "gamma_curve_shape_min", 0.05) or 0.05),
            shape_max=float(getattr(config, "gamma_curve_shape_max", 20.0) or 20.0),
        )
    return _fit_normal_cdf_to_ce_rows(ce_rows)


def _gamma_ce_rows_from_diagnostic_window(batch_window: list, config) -> list:
    if not batch_window:
        return []
    rows, meta = _loss_diagnostic_summary_rows(batch_window, config)
    plot_rows = [row for row in rows if row["n"] >= meta["min_plot_samples"]]
    if not plot_rows:
        return []
    return _aggregate_loss_diagnostic_ce_by_gamma(plot_rows, min_samples=meta["min_plot_samples"])


def _weighted_isotonic_non_decreasing(values: Sequence[float], weights: Sequence[float]) -> list[float]:
    blocks = []
    for idx, (value, weight) in enumerate(zip(values, weights)):
        weight = max(float(weight), 1e-12)
        block = {"start": idx, "end": idx, "weight": weight, "mean": float(value)}
        blocks.append(block)
        while len(blocks) >= 2 and blocks[-2]["mean"] > blocks[-1]["mean"]:
            right = blocks.pop()
            left = blocks.pop()
            merged_weight = float(left["weight"]) + float(right["weight"])
            merged_mean = (
                float(left["mean"]) * float(left["weight"])
                + float(right["mean"]) * float(right["weight"])
            ) / max(merged_weight, 1e-12)
            blocks.append(
                {
                    "start": int(left["start"]),
                    "end": int(right["end"]),
                    "weight": merged_weight,
                    "mean": merged_mean,
                }
            )
    smoothed = [0.0 for _ in values]
    for block in blocks:
        for idx in range(int(block["start"]), int(block["end"]) + 1):
            smoothed[idx] = float(block["mean"])
    return smoothed


def _empirical_piecewise_gamma_cdf_from_ce_rows(
    ce_rows: list,
    config,
    *,
    smoothing: str = "cummax",
) -> Optional[Dict[str, object]]:
    valid = [
        row
        for row in ce_rows
        if _finite_float(row.get("gamma_mid"))
        and _finite_float(row.get("ce_mean"))
        and float(row.get("n", 0) or 0) > 0
    ]
    if len(valid) < 4:
        return None
    valid = sorted(valid, key=lambda row: float(row["gamma_mid"]))
    gamma_min = float(getattr(config, "gamma_min", -6.0))
    gamma_max = float(getattr(config, "gamma_max", 6.0))
    ce_values = torch.tensor([float(row["ce_mean"]) for row in valid], dtype=torch.float64)
    if str(smoothing or "cummax").lower() == "isotonic":
        ce_values = torch.tensor(
            _weighted_isotonic_non_decreasing(
                [float(row["ce_mean"]) for row in valid],
                [float(max(float(row.get("n", 0) or 0), 1.0)) for row in valid],
            ),
            dtype=torch.float64,
        )
    ce_min = float(ce_values.min().item())
    ce_max = float(ce_values.max().item())
    ce_range = ce_max - ce_min
    if ce_range <= 1e-8:
        return None

    probabilities = ((ce_values - ce_min) / ce_range).clamp(0.0, 1.0)
    probabilities = torch.cummax(probabilities, dim=0).values
    gamma_knots = [gamma_min]
    cdf_knots = [0.0]
    eps = 1e-5
    for row, probability in zip(valid, probabilities.tolist()):
        gamma = float(row["gamma_mid"])
        probability = float(probability)
        if gamma <= gamma_min or gamma >= gamma_max:
            continue
        if gamma <= gamma_knots[-1] + eps or probability <= cdf_knots[-1] + eps:
            continue
        gamma_knots.append(gamma)
        cdf_knots.append(probability)
    if cdf_knots[-1] < 1.0 - eps:
        gamma_knots.append(gamma_max)
        cdf_knots.append(1.0)
    if len(gamma_knots) < 2:
        return None
    return {
        "gamma": [float(value) for value in gamma_knots],
        "cdf": [float(value) for value in cdf_knots],
        "count": len(gamma_knots),
        "ce_min": ce_min,
        "ce_max": ce_max,
    }


def _apply_active_piecewise_gamma_cdf_to_config(config, curve: Optional[Dict[str, object]]) -> None:
    if not curve:
        return
    gamma = curve.get("gamma")
    cdf = curve.get("cdf")
    if not isinstance(gamma, Sequence) or not isinstance(cdf, Sequence) or len(gamma) != len(cdf) or len(gamma) < 2:
        return
    setattr(config, "gamma_active_piecewise_gamma", [float(value) for value in gamma])
    setattr(config, "gamma_active_piecewise_cdf", [float(value) for value in cdf])


def _gamma_curve_quantile_grid(config) -> list[float]:
    points = max(3, int(getattr(config, "gamma_curve_quantile_points", 101) or 101))
    return [float(idx) / float(points - 1) for idx in range(points)]


def _piecewise_inverse_values(gamma: Sequence[float], cdf: Sequence[float], quantiles: Sequence[float]) -> list[float]:
    if len(gamma) != len(cdf) or len(gamma) < 2:
        raise ValueError("Piecewise inverse CDF needs matching gamma/cdf knots.")
    pairs = sorted((float(c), float(g)) for g, c in zip(gamma, cdf) if math.isfinite(float(g)) and math.isfinite(float(c)))
    if len(pairs) < 2:
        raise ValueError("Piecewise inverse CDF needs at least two finite knots.")
    cdf_t = torch.tensor([pair[0] for pair in pairs], dtype=torch.float64).clamp(0.0, 1.0)
    gamma_t = torch.tensor([pair[1] for pair in pairs], dtype=torch.float64)
    cdf_t = torch.cummax(cdf_t, dim=0).values
    keep = torch.ones_like(cdf_t, dtype=torch.bool)
    keep[1:] = cdf_t[1:] > cdf_t[:-1] + 1e-8
    cdf_t = cdf_t[keep]
    gamma_t = gamma_t[keep]
    if cdf_t.numel() < 2:
        raise ValueError("Piecewise inverse CDF collapsed to fewer than two knots.")
    cdf_t = (cdf_t - cdf_t[0]) / (cdf_t[-1] - cdf_t[0]).clamp_min(1e-12)
    cdf_t[0] = 0.0
    cdf_t[-1] = 1.0
    values = []
    for probability in quantiles:
        q = torch.tensor([min(max(float(probability), 0.0), 1.0)], dtype=torch.float64)
        idx = torch.searchsorted(cdf_t.contiguous(), q.contiguous(), right=True) - 1
        idx = idx.clamp(0, cdf_t.numel() - 2)
        c0 = cdf_t[idx]
        c1 = cdf_t[idx + 1]
        g0 = gamma_t[idx]
        g1 = gamma_t[idx + 1]
        frac = (q - c0) / (c1 - c0).clamp_min(1e-12)
        values.append(float((g0 + frac * (g1 - g0)).item()))
    return values


def _monotone_gamma_curve_from_quantiles(config, quantiles: Sequence[float], gamma_values: Sequence[float]) -> Dict[str, object]:
    gamma_min = float(getattr(config, "gamma_min", -6.0))
    gamma_max = float(getattr(config, "gamma_max", 6.0))
    values = [min(max(float(value), gamma_min), gamma_max) for value in gamma_values]
    if values:
        values[0] = gamma_min
        values[-1] = gamma_max
    eps = 1e-6
    for idx in range(1, len(values)):
        values[idx] = max(values[idx], values[idx - 1] + eps)
    if values:
        values[-1] = gamma_max
    for idx in range(len(values) - 2, -1, -1):
        values[idx] = min(values[idx], values[idx + 1] - eps)
    if values:
        values[0] = gamma_min
        values[-1] = gamma_max
    return {"gamma": [float(value) for value in values], "cdf": [float(value) for value in quantiles]}


def _candidate_curve_from_fit(config, fit: Dict[str, float], quantiles: Sequence[float]) -> Dict[str, object]:
    gamma_min = float(getattr(config, "gamma_min", -6.0))
    gamma_max = float(getattr(config, "gamma_max", 6.0))
    values = []
    for probability in quantiles:
        if probability <= 0.0:
            values.append(gamma_min)
        elif probability >= 1.0:
            values.append(gamma_max)
        else:
            values.append(min(max(_fit_quantile_scalar(fit, probability), gamma_min), gamma_max))
    curve = _monotone_gamma_curve_from_quantiles(config, quantiles, values)
    curve.update({"params": dict(fit), "r2": float(fit.get("r2", float("nan")))})
    return curve


def _gamma_curve_ce_rows_from_diagnostic_window(batch_window: list, config) -> tuple[list, Dict]:
    if not batch_window:
        return [], {}
    rows, meta = _loss_diagnostic_summary_rows(batch_window, config)
    min_samples = int(meta.get("min_plot_samples", 1) or 1)
    mask_min = float(getattr(config, "gamma_curve_mask_p_min", 0.05) or 0.05)
    mask_max = float(getattr(config, "gamma_curve_mask_p_max", 1.0) or 1.0)
    filtered_rows = []
    for row in rows:
        mask_mid = 0.5 * (float(row.get("mask_left", 0.0)) + float(row.get("mask_right", 0.0)))
        if float(row.get("n", 0) or 0) < float(min_samples):
            continue
        if mask_mid + 1e-12 < mask_min or mask_mid - 1e-12 > mask_max:
            continue
        filtered_rows.append(row)
    if not filtered_rows:
        return [], meta
    return _aggregate_loss_diagnostic_ce_by_gamma(filtered_rows, min_samples=min_samples), meta


def _estimate_candidate_gamma_curve(batch_window: list, config) -> Optional[Dict[str, object]]:
    ce_rows, _ = _gamma_curve_ce_rows_from_diagnostic_window(batch_window, config)
    min_bins = int(getattr(config, "gamma_curve_min_bins", 8) or 8)
    min_examples = float(getattr(config, "gamma_curve_min_examples", 4096) or 4096)
    if len(ce_rows) < min_bins:
        return None
    total_examples = sum(float(row.get("n", 0) or 0) for row in ce_rows)
    if total_examples < min_examples:
        return None

    estimator = str(getattr(config, "gamma_curve_estimator", "generalized_logistic") or "generalized_logistic").lower()
    estimator = {
        "glogistic": "generalized_logistic",
        "logistic": "generalized_logistic",
        "piecewise": "empirical",
        "piecewise_empirical": "empirical",
        "empirical_piecewise": "empirical",
    }.get(estimator, estimator)
    quantiles = _gamma_curve_quantile_grid(config)
    if estimator == "empirical":
        empirical = _empirical_piecewise_gamma_cdf_from_ce_rows(
            ce_rows,
            config,
            smoothing=str(getattr(config, "gamma_curve_smoothing", "isotonic") or "isotonic"),
        )
        if empirical is None:
            return None
        values = _piecewise_inverse_values(empirical["gamma"], empirical["cdf"], quantiles)
        curve = _monotone_gamma_curve_from_quantiles(config, quantiles, values)
        curve.update({"estimator": "empirical", "r2": float("nan")})
    elif estimator == "normal":
        fit = _fit_normal_cdf_to_ce_rows(ce_rows)
        if fit is None:
            return None
        min_r2 = float(getattr(config, "gamma_curve_min_r2", 0.95) or 0.0)
        if math.isfinite(float(fit.get("r2", float("nan")))) and float(fit["r2"]) < min_r2:
            return None
        curve = _candidate_curve_from_fit(config, fit, quantiles)
        curve["estimator"] = "normal"
    elif estimator in {"generalized_logistic", "fitted_generalized_logistic"}:
        fit = _fit_generalized_logistic_cdf_to_ce_rows(
            ce_rows,
            shape_min=float(getattr(config, "gamma_curve_shape_min", 0.05) or 0.05),
            shape_max=float(getattr(config, "gamma_curve_shape_max", 20.0) or 20.0),
        )
        if fit is None:
            return None
        min_r2 = float(getattr(config, "gamma_curve_min_r2", 0.95) or 0.0)
        if math.isfinite(float(fit.get("r2", float("nan")))) and float(fit["r2"]) < min_r2:
            return None
        curve = _candidate_curve_from_fit(config, fit, quantiles)
        curve["estimator"] = "generalized_logistic"
    else:
        raise ValueError(f"Unknown gamma_curve_estimator: {estimator}")

    curve["source_bins"] = float(len(ce_rows))
    curve["source_examples"] = float(total_examples)
    return curve


def _ema_update_active_gamma_curve(config, candidate: Dict[str, object]) -> Dict[str, object]:
    quantiles = _gamma_curve_quantile_grid(config)
    candidate_values = _piecewise_inverse_values(candidate["gamma"], candidate["cdf"], quantiles)
    old_gamma = getattr(config, "gamma_active_piecewise_gamma", None)
    old_cdf = getattr(config, "gamma_active_piecewise_cdf", None)
    rate = min(max(float(getattr(config, "gamma_curve_update_rate", 0.02) or 0.0), 0.0), 1.0)
    if old_gamma is not None and old_cdf is not None:
        old_values = _piecewise_inverse_values(old_gamma, old_cdf, quantiles)
        active_values = [(1.0 - rate) * old + rate * new for old, new in zip(old_values, candidate_values)]
    else:
        old_values = None
        active_values = candidate_values
    active = _monotone_gamma_curve_from_quantiles(config, quantiles, active_values)
    _apply_active_piecewise_gamma_cdf_to_config(config, active)
    setattr(config, "gamma_curve_updates", int(getattr(config, "gamma_curve_updates", 0) or 0) + 1)
    return {"active": active, "candidate_values": candidate_values, "old_values": old_values, "update_rate": rate}


def _gamma_curve_adaptation_enabled(config) -> bool:
    if not bool(getattr(config, "gamma_curve_adapt_enabled", False)):
        return False
    noise_parameterization = str(getattr(config, "noise_parameterization", "log_nsr") or "log_nsr").lower()
    if noise_parameterization != "log_nsr":
        return False
    schedule = str(getattr(config, "gamma_schedule", "gumbel") or "gumbel").lower()
    return schedule in {
        "active_mixture",
        "gamma_active_mixture",
        "curve_mixture",
        "gamma_curve_mixture",
        "active_piecewise",
        "gamma_active_piecewise",
        "active_empirical",
        "active_cdf",
    }


def _gamma_curve_uses_no_eos_ce(config) -> bool:
    """Use response-token CE excluding EOS for the SFT gamma curve.

    SFT contains many easy EOS/pad-after-answer targets at high mask ratios.
    The active SFT gamma curve should track the difficulty of non-EOS answer
    tokens, while the mask-p slice is still controlled by
    gamma_curve_mask_p_min/max.
    """
    return str(getattr(config, "training_stage", "pretrain") or "pretrain").lower() == "sft"


def _gamma_curve_batches_for_adaptation(config, regular_batches: list, no_eos_batches: list) -> tuple[list, str]:
    if _gamma_curve_uses_no_eos_ce(config):
        return no_eos_batches, "no_eos_ce"
    return regular_batches, "ce"


def _adapt_active_gamma_curve(config, batch_window: list, step: int, device: torch.device) -> Dict[str, float]:
    enabled = _gamma_curve_adaptation_enabled(config)
    payload = [None]
    if is_main_process() and enabled:
        candidate = _estimate_candidate_gamma_curve(batch_window, config)
        if candidate is not None:
            update = _ema_update_active_gamma_curve(config, candidate)
            setattr(config, "gamma_curve_last_update_step", int(step))
            payload[0] = {
                "applied": True,
                "candidate": candidate,
                "active": update["active"],
                "update_rate": update["update_rate"],
                "updates": int(getattr(config, "gamma_curve_updates", 0) or 0),
                "last_update_step": int(getattr(config, "gamma_curve_last_update_step", 0) or 0),
            }
        else:
            payload[0] = {
                "applied": False,
                "updates": int(getattr(config, "gamma_curve_updates", 0) or 0),
                "last_update_step": int(getattr(config, "gamma_curve_last_update_step", 0) or 0),
            }
    if distributed_available():
        dist.broadcast_object_list(payload, src=0)
    if payload[0] and payload[0].get("active") is not None:
        _apply_active_piecewise_gamma_cdf_to_config(config, payload[0]["active"])
        setattr(config, "gamma_curve_updates", int(payload[0].get("updates", 0) or 0))
        setattr(config, "gamma_curve_last_update_step", int(payload[0].get("last_update_step", 0) or 0))

    candidate = payload[0].get("candidate") if payload[0] else None
    metrics = {
        "train/gamma_curve/enabled": float(enabled),
        "train/gamma_curve/applied": float(bool(payload[0] and payload[0].get("applied", False))),
        "train/gamma_curve/update_rate": float(getattr(config, "gamma_curve_update_rate", 0.02) or 0.0),
        "train/gamma_curve/updates": float(int(getattr(config, "gamma_curve_updates", 0) or 0)),
        "train/gamma_curve/last_update_step": float(int(getattr(config, "gamma_curve_last_update_step", 0) or 0)),
        "train/gamma_curve/mask_p_min": float(getattr(config, "gamma_curve_mask_p_min", 0.05) or 0.05),
        "train/gamma_curve/mask_p_max": float(getattr(config, "gamma_curve_mask_p_max", 1.0) or 1.0),
        "train/gamma_curve/uses_no_eos_ce": float(_gamma_curve_uses_no_eos_ce(config)),
    }
    if candidate is not None:
        metrics.update(
            {
                "train/gamma_curve/source_bins": float(candidate.get("source_bins", 0.0) or 0.0),
                "train/gamma_curve/source_examples": float(candidate.get("source_examples", 0.0) or 0.0),
                "train/gamma_curve/r2": float(candidate.get("r2", float("nan"))),
            }
        )
        metrics.update(_piecewise_quantile_metrics(candidate.get("gamma"), candidate.get("cdf"), "train/gamma_curve_candidate"))
    metrics.update(_piecewise_quantile_metrics(
        getattr(config, "gamma_active_piecewise_gamma", None),
        getattr(config, "gamma_active_piecewise_cdf", None),
        "train/gamma_curve_active",
    ))
    return metrics


def _piecewise_quantile_metrics(gamma, cdf, prefix: str) -> Dict[str, float]:
    if not gamma or not cdf or len(gamma) != len(cdf) or len(gamma) < 2:
        return {}
    gamma_t = torch.tensor([float(value) for value in gamma], dtype=torch.float64)
    cdf_t = torch.tensor([float(value) for value in cdf], dtype=torch.float64)
    order = torch.argsort(cdf_t)
    cdf_t = cdf_t[order].clamp(0.0, 1.0)
    gamma_t = gamma_t[order]
    cdf_t = torch.cummax(cdf_t, dim=0).values
    keep = torch.ones_like(cdf_t, dtype=torch.bool)
    keep[1:] = cdf_t[1:] > cdf_t[:-1] + 1e-8
    cdf_t = cdf_t[keep]
    gamma_t = gamma_t[keep]
    if cdf_t.numel() < 2:
        return {}

    metrics = {f"{prefix}_knots": float(cdf_t.numel())}
    for probability in (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95):
        q = torch.tensor([probability], dtype=torch.float64)
        idx = torch.searchsorted(cdf_t.contiguous(), q.contiguous(), right=True) - 1
        idx = idx.clamp(0, cdf_t.numel() - 2)
        c0 = cdf_t[idx]
        c1 = cdf_t[idx + 1]
        g0 = gamma_t[idx]
        g1 = gamma_t[idx + 1]
        frac = (q - c0) / (c1 - c0).clamp_min(1e-12)
        value = g0 + frac * (g1 - g0)
        metrics[f"{prefix}_q{int(probability * 100):02d}"] = float(value.item())
    return metrics


def _aggregate_loss_diagnostic_derivative_by_gamma(ce_rows: list) -> list:
    if len(ce_rows) < 2:
        return []
    rows = sorted(ce_rows, key=lambda item: item["gamma_mid"])
    derivative_rows = []
    for idx, row in enumerate(rows):
        if idx == 0:
            left_row, right_row = rows[idx], rows[idx + 1]
        elif idx == len(rows) - 1:
            left_row, right_row = rows[idx - 1], rows[idx]
        else:
            left_row, right_row = rows[idx - 1], rows[idx + 1]
        dx = float(right_row["gamma_mid"]) - float(left_row["gamma_mid"])
        if dx <= 0.0:
            continue
        derivative = (float(right_row["ce_mean"]) - float(left_row["ce_mean"])) / dx
        derivative_se = math.sqrt(float(left_row["ce_se"]) ** 2 + float(right_row["ce_se"]) ** 2) / dx
        derivative_delta = 1.96 * derivative_se
        derivative_rows.append(
            {
                "gamma_mid": float(row["gamma_mid"]),
                "dce_dgamma": derivative,
                "dce_ci_low": derivative - derivative_delta,
                "dce_ci_high": derivative + derivative_delta,
            }
        )
    return derivative_rows


def _loss_diagnostic_ce_cdf_figure(rows: list, config, min_samples: int):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    ce_rows = _aggregate_loss_diagnostic_ce_by_gamma(rows, min_samples=min_samples)
    if not ce_rows:
        return None, None

    x = [float(row["gamma_mid"]) for row in ce_rows]
    y = [float(row["ce_mean"]) for row in ce_rows]
    low = [float(row["ce_ci_low"]) for row in ce_rows]
    high = [float(row["ce_ci_high"]) for row in ce_rows]
    cdf_fit = _fit_gamma_cdf_to_ce_rows(ce_rows, config)

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(
            x=x + x[::-1],
            y=high + low[::-1],
            fill="toself",
            fillcolor=_rgba("#1f77b4", 0.16),
            line={"color": "rgba(0,0,0,0)"},
            hoverinfo="skip",
            showlegend=False,
            name="CE 95% CI",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="lines+markers",
            line={"color": "#1f77b4", "width": 3},
            marker={"size": 5},
            name="empirical CE",
            hovertemplate="gamma=%{x:.3f}<br>CE=%{y:.4f}<extra></extra>",
        ),
        secondary_y=False,
    )

    gamma_min = float(getattr(config, "gamma_min", -6.0))
    gamma_max = float(getattr(config, "gamma_max", 6.0))
    gamma_grid = torch.linspace(gamma_min, gamma_max, 512, dtype=torch.float32)
    if cdf_fit is not None:
        fitted_cdf = _fit_cdf_tensor(gamma_grid, cdf_fit)
        fitted_ce = cdf_fit["ce_min"] + (cdf_fit["ce_max"] - cdf_fit["ce_min"]) * fitted_cdf
        fig.add_trace(
            go.Scatter(
                x=gamma_grid.tolist(),
                y=fitted_ce.detach().float().cpu().tolist(),
                mode="lines",
                line={"color": "#2ca02c", "width": 3, "dash": "dot"},
                name=f"{_fit_label(cdf_fit)} CDF fit to CE",
                hovertemplate="gamma=%{x:.3f}<br>fit CE=%{y:.4f}<extra></extra>",
            ),
            secondary_y=False,
        )
    cdfs = gamma_distribution_component_cdfs(config, gamma_grid)
    component_order = [
        "mixture_total",
        "gumbel",
        "normal",
        "active_piecewise",
        "uniform",
    ]
    cdf_colors = {
        "mixture_total": "#111111",
        "gumbel": "#111111",
        "normal": "#ff7f0e",
        "active_piecewise": "#9467bd",
        "uniform": "#7f7f7f",
    }
    for name in component_order:
        cdf = cdfs.get(name)
        if cdf is None:
            continue
        fig.add_trace(
            go.Scatter(
                x=gamma_grid.tolist(),
                y=cdf.detach().float().cpu().tolist(),
                mode="lines",
                line={
                    "color": cdf_colors.get(name, "#111111"),
                    "width": 3 if name in {"mixture_total", "gumbel"} else 2,
                    "dash": "solid" if name in {"mixture_total", "gumbel"} else "dash",
                },
                name=f"{name} CDF",
                hovertemplate="gamma=%{x:.3f}<br>CDF=%{y:.4f}<extra></extra>",
            ),
            secondary_y=True,
        )

    fig.update_layout(
        title="CE by gamma with configured gamma CDF",
        xaxis_title="gamma",
        hovermode="x unified",
        template="plotly_white",
        legend_title="curve",
    )
    fig.update_yaxes(title_text="empirical corrupted-token CE", secondary_y=False)
    fig.update_yaxes(title_text="configured gamma CDF", secondary_y=True)
    return fig, cdf_fit


def _loss_diagnostic_derivative_pdf_figure(rows: list, config, min_samples: int, cdf_fit: Optional[Dict[str, float]] = None):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    ce_rows = _aggregate_loss_diagnostic_ce_by_gamma(rows, min_samples=min_samples)
    derivative_rows = _aggregate_loss_diagnostic_derivative_by_gamma(ce_rows)
    if not derivative_rows:
        return None

    x = [float(row["gamma_mid"]) for row in derivative_rows]
    y = [float(row["dce_dgamma"]) for row in derivative_rows]
    low = [float(row["dce_ci_low"]) for row in derivative_rows]
    high = [float(row["dce_ci_high"]) for row in derivative_rows]

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(
            x=x + x[::-1],
            y=high + low[::-1],
            fill="toself",
            fillcolor=_rgba("#1f77b4", 0.16),
            line={"color": "rgba(0,0,0,0)"},
            hoverinfo="skip",
            showlegend=False,
            name="dCE/dgamma 95% CI",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="lines+markers",
            line={"color": "#1f77b4", "width": 3},
            marker={"size": 5},
            name="empirical dCE/dgamma",
            hovertemplate="gamma=%{x:.3f}<br>dCE/dgamma=%{y:.4f}<extra></extra>",
        ),
        secondary_y=False,
    )

    gamma_min = float(getattr(config, "gamma_min", -6.0))
    gamma_max = float(getattr(config, "gamma_max", 6.0))
    gamma_grid = torch.linspace(gamma_min, gamma_max, 512, dtype=torch.float32)
    if cdf_fit is not None:
        fitted_pdf = _fit_pdf_tensor(gamma_grid, cdf_fit)
        fitted_derivative = (cdf_fit["ce_max"] - cdf_fit["ce_min"]) * fitted_pdf
        fig.add_trace(
            go.Scatter(
                x=gamma_grid.tolist(),
                y=fitted_derivative.detach().float().cpu().tolist(),
                mode="lines",
                line={"color": "#2ca02c", "width": 3, "dash": "dot"},
                name=f"{_fit_label(cdf_fit)} CDF-fit derivative",
                hovertemplate="gamma=%{x:.3f}<br>fit dCE/dgamma=%{y:.4f}<extra></extra>",
            ),
            secondary_y=False,
        )

    pdfs = gamma_distribution_component_pdfs(config, gamma_grid)
    component_order = [
        "mixture_total",
        "gumbel",
        "normal",
        "active_piecewise",
        "uniform",
    ]
    pdf_colors = {
        "mixture_total": "#111111",
        "gumbel": "#111111",
        "normal": "#ff7f0e",
        "active_piecewise": "#9467bd",
        "uniform": "#7f7f7f",
    }
    for name in component_order:
        pdf = pdfs.get(name)
        if pdf is None:
            continue
        fig.add_trace(
            go.Scatter(
                x=gamma_grid.tolist(),
                y=pdf.detach().float().cpu().tolist(),
                mode="lines",
                line={
                    "color": pdf_colors.get(name, "#111111"),
                    "width": 3 if name in {"mixture_total", "gumbel"} else 2,
                    "dash": "solid" if name in {"mixture_total", "gumbel"} else "dash",
                },
                name=f"{name} PDF",
                hovertemplate="gamma=%{x:.3f}<br>PDF=%{y:.4f}<extra></extra>",
            ),
            secondary_y=True,
        )

    fig.update_layout(
        title="dCE/dgamma by gamma with configured gamma PDF",
        xaxis_title="gamma",
        hovermode="x unified",
        template="plotly_white",
        legend_title="curve",
    )
    fig.update_yaxes(title_text="empirical dCE/dgamma", secondary_y=False)
    fig.update_yaxes(title_text="configured gamma PDF", secondary_y=True)
    return fig


def _wandb_plotly(wandb, fig):
    return wandb.Plotly(fig) if hasattr(wandb, "Plotly") else fig


def _log_wandb_loss_diagnostics(
    wandb_run,
    step: int,
    loss_diagnostic_batch_window: list,
    config,
    no_eos_batch_window: Optional[list] = None,
):
    if wandb_run is None or not loss_diagnostic_batch_window:
        return
    try:
        import wandb
    except Exception:
        return

    rows, meta = _loss_diagnostic_summary_rows(loss_diagnostic_batch_window, config, step=step)
    payload = {
        "train/sample_diagnostics/window_rows": meta["window_rows"],
        "train/sample_diagnostics/window_total_examples": meta["window_total_examples"],
        "train/sample_diagnostics/window_batches": meta["window_batches"],
        "train/sample_diagnostics/window_target_batches": meta["window_target_batches"],
        "train/sample_diagnostics/window_examples_target": meta["window_examples_target"],
        "train/sample_diagnostics/mask_bins": meta["mask_bins"],
        "train/sample_diagnostics/gamma_bins": meta["gamma_bins"],
        "train/sample_diagnostics/effective_batches": meta["effective_batches"],
        "train/sample_diagnostics/effective_examples": meta["effective_examples"],
        "train/sample_diagnostics/configured_gamma_bins": meta["configured_gamma_bins"],
        "train/sample_diagnostics/target_samples_per_cell": meta["target_samples_per_cell"],
        "train/sample_diagnostics/min_plot_samples": meta["min_plot_samples"],
        "train/sample_diagnostics/plot_error": 0.0,
    }
    payload.update(_loss_diagnostic_original_bin_average_metrics(loss_diagnostic_batch_window, config, prefix="train"))
    no_eos_rows = []
    no_eos_meta = None
    if no_eos_batch_window:
        no_eos_rows, no_eos_meta = _loss_diagnostic_summary_rows(no_eos_batch_window, config, step=step)
    if not rows:
        wandb_run.log(payload, step=step)
        return

    try:
        plot_rows = [row for row in rows if row["n"] >= meta["min_plot_samples"]]
        payload["train/loss_gamma_mask_p_table"] = _loss_diagnostic_table(wandb, rows, step)
        if plot_rows:
            ce_fig = _loss_diagnostic_plotly_figure(
                plot_rows,
                "ce_mean",
                "ce_ci_low",
                "ce_ci_high",
                "CE by gamma and mask_p",
                "corrupted-token CE",
            )
            payload["train/loss_ce_by_gamma_mask_p"] = _wandb_plotly(wandb, ce_fig)
            if no_eos_rows and no_eos_meta is not None:
                no_eos_plot_rows = [row for row in no_eos_rows if row["n"] >= no_eos_meta["min_plot_samples"]]
                if no_eos_plot_rows:
                    no_eos_ce_fig = _loss_diagnostic_plotly_figure(
                        no_eos_plot_rows,
                        "ce_mean",
                        "ce_ci_low",
                        "ce_ci_high",
                        "CE without EOS by gamma and mask_p",
                        "corrupted non-EOS-token CE",
                    )
                    payload["train/no_eos_ce_by_gamma_mask_p"] = _wandb_plotly(wandb, no_eos_ce_fig)
                    no_eos_gamma_rows = _aggregate_loss_diagnostic_ce_by_gamma(
                        no_eos_plot_rows,
                        min_samples=no_eos_meta["min_plot_samples"],
                    )
                    if no_eos_gamma_rows:
                        no_eos_gamma_fig = _loss_diagnostic_plotly_figure(
                            [
                                {
                                    **row,
                                    "mask_label": "all mask_p",
                                }
                                for row in no_eos_gamma_rows
                            ],
                            "ce_mean",
                            "ce_ci_low",
                            "ce_ci_high",
                            "CE without EOS by gamma",
                            "corrupted non-EOS-token CE",
                        )
                        payload["train/no_eos_ce_by_gamma"] = _wandb_plotly(wandb, no_eos_gamma_fig)
            ce_cdf_fig, ce_cdf_fit = _loss_diagnostic_ce_cdf_figure(
                plot_rows,
                config,
                min_samples=meta["min_plot_samples"],
            )
            if ce_cdf_fig is not None:
                payload["train/loss_ce_gamma_cdf"] = _wandb_plotly(wandb, ce_cdf_fig)
            if ce_cdf_fit is not None:
                payload.update(
                    {
                        "train/loss_ce_gamma_cdf_fit/loc": ce_cdf_fit["loc"],
                        "train/loss_ce_gamma_cdf_fit/scale": ce_cdf_fit["scale"],
                        "train/loss_ce_gamma_cdf_fit/shape": ce_cdf_fit.get("shape", 1.0),
                        "train/loss_ce_gamma_cdf_fit/ce_min": ce_cdf_fit["ce_min"],
                        "train/loss_ce_gamma_cdf_fit/ce_max": ce_cdf_fit["ce_max"],
                        "train/loss_ce_gamma_cdf_fit/r2": ce_cdf_fit["r2"],
                    }
                )
                payload.update(_curve_fit_quantile_metrics(ce_cdf_fit, prefix="train/loss_ce_gamma_cdf_fit"))
            derivative_pdf_fig = _loss_diagnostic_derivative_pdf_figure(
                plot_rows,
                config,
                min_samples=meta["min_plot_samples"],
                cdf_fit=ce_cdf_fit,
            )
            if derivative_pdf_fig is not None:
                payload["train/dloss_dgamma_gamma_pdf"] = _wandb_plotly(wandb, derivative_pdf_fig)
            derivative_rows = [row for row in plot_rows if _finite_float(row.get("dce_dgamma"))]
            if derivative_rows:
                derivative_fig = _loss_diagnostic_plotly_figure(
                    derivative_rows,
                    "dce_dgamma",
                    "dce_ci_low",
                    "dce_ci_high",
                    "dCE/dgamma by gamma and mask_p",
                    "dCE/dgamma",
                )
                payload["train/dloss_dgamma_by_gamma_mask_p"] = _wandb_plotly(wandb, derivative_fig)
    except Exception as exc:
        log_for_0(f"WandB loss diagnostic rendering failed: {exc}", level=logging.WARNING)
        payload["train/sample_diagnostics/plot_error"] = 1.0
        wandb_run.log(payload, step=step)
        return

    wandb_run.log(payload, step=step)


def _log_wandb_generation_samples(wandb_run, step: int, sample_rows=None):
    if wandb_run is None or not sample_rows:
        return
    try:
        import wandb
    except Exception:
        return
    columns = ["step", "masked_token_acc", "masked_tokens", "target", "masked_target", "sampled"]
    data = [
        [
            step,
            row.get("masked_token_acc"),
            row.get("masked_tokens"),
            row.get("target", ""),
            row.get("masked_target", ""),
            row.get("sampled", ""),
        ]
        for row in sample_rows
    ]
    wandb_run.log({"eval_generated_samples": wandb.Table(columns=columns, data=data)}, step=step)


def _log_wandb_unconditional_generations(wandb_run, step: int, sample_rows=None):
    if wandb_run is None or not sample_rows:
        return
    try:
        import wandb
    except Exception:
        return
    columns = ["step", "length", "sampled_tokens", "sampled"]
    data = [[step, row.get("length"), row.get("sampled_tokens"), row.get("sampled", "")] for row in sample_rows]
    wandb_run.log({"eval_unconditional_generations": wandb.Table(columns=columns, data=data)}, step=step)


def _log_wandb_gsm8k_generations(wandb_run, step: int, sample_rows=None, table_key: str = "eval_gsm8k_conditional_generations"):
    if wandb_run is None or not sample_rows:
        return
    try:
        import wandb
    except Exception:
        return
    columns = [
        "step",
        "generation_steps",
        "target_length",
        "prompt_tokens",
        "target_tokens",
        "masked_tokens",
        "sampled_tokens",
        "question",
        "gold_answer",
        "strict_prediction",
        "flexible_prediction",
        "strict_match",
        "flexible_match",
        "target",
        "masked_target",
        "sampled_answer",
        "sampled",
    ]
    data = []
    for row in sample_rows:
        data.append(
            [
                step,
                row.get("generation_steps"),
                row.get("target_length"),
                row.get("prompt_tokens"),
                row.get("target_tokens"),
                row.get("masked_tokens"),
                row.get("sampled_tokens"),
                row.get("question", ""),
                row.get("gold_answer"),
                row.get("strict_prediction"),
                row.get("flexible_prediction"),
                row.get("strict_match"),
                row.get("flexible_match"),
                row.get("target", ""),
                row.get("masked_target", ""),
                row.get("sampled_answer", ""),
                row.get("sampled", ""),
            ]
        )
    wandb_run.log({table_key: wandb.Table(columns=columns, data=data)}, step=step)


def _log_wandb_sft_prompt_generations(wandb_run, step: int, sample_rows=None):
    if wandb_run is None or not sample_rows:
        return
    try:
        import wandb
    except Exception:
        return
    columns = [
        "step",
        "prompt_tokens",
        "target_tokens",
        "masked_tokens",
        "sampled_tokens",
        "prompt",
        "target",
        "masked_target",
        "sampled_response",
        "sampled",
    ]
    table_keys = {
        "general": "eval_sft_general_generations",
        "math": "eval_sft_math_generations",
        "code": "eval_sft_code_generations",
    }
    grouped = {source_type: [] for source_type in table_keys}
    for row in sample_rows:
        source_type = str(row.get("source_type", ""))
        if source_type in grouped:
            grouped[source_type].append(row)
    payload = {}
    for source_type, rows in grouped.items():
        if not rows:
            continue
        data = []
        for row in rows:
            data.append(
                [
                    step,
                    row.get("prompt_tokens"),
                    row.get("target_tokens"),
                    row.get("masked_tokens"),
                    row.get("sampled_tokens"),
                    row.get("prompt", ""),
                    row.get("target", ""),
                    row.get("masked_target", ""),
                    row.get("sampled_response", ""),
                    row.get("sampled", ""),
                ]
            )
        payload[table_keys[source_type]] = wandb.Table(columns=columns, data=data)
    if payload:
        wandb_run.log(payload, step=step)


def run_mlfm_training(config):
    device = setup_distributed_and_device(config)
    rank = get_rank()
    world_size = get_world_size()
    torch.manual_seed(int(config.seed) + rank)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(config.seed) + rank)
    generator = _make_generator(int(config.seed) + rank * 100003, device)
    eval_seed_base = int(config.seed) + rank * 100003 + 10_000_019
    precision, amp_dtype = resolve_precision(config, device)

    log_for_0("=" * 60)
    log_for_0("MLFM Training")
    log_for_0("=" * 60)
    log_for_0(f"Backbone: {config.backbone_model_name_or_path}")
    log_for_0(f"Backbone type: {getattr(config, 'backbone_type', 'auto')}")
    log_for_0(f"Training stage: {getattr(config, 'training_stage', 'pretrain')}")
    log_for_0(f"Forward process: {getattr(config, 'forward_process', 'brownian_bridge')}")
    log_for_0(f"World size: {world_size}")
    log_for_0(f"Device: {device}")
    log_for_0(f"Precision: {precision}")
    if world_size > 1:
        log_for_0("DDP enabled. For 8B-class runs, prefer FSDP or DeepSpeed ZeRO-3.")

    wandb_run = None
    if bool(getattr(config, "use_wandb", False)) and is_main_process():
        try:
            import wandb

            wandb_run = wandb.init(
                project=getattr(config, "wandb_project", "MLFM"),
                entity=getattr(config, "wandb_entity", None),
                name=getattr(config, "wandb_run_name", None),
                group=getattr(config, "wandb_group", None),
                job_type=getattr(config, "wandb_job_type", None),
                tags=(getattr(config, "wandb_tag", None) or "").split(",") if getattr(config, "wandb_tag", None) else None,
                config=_as_config_dict(config),
                dir=_resolve_wandb_dir(config),
            )
        except Exception as exc:
            log_for_0(f"WandB init failed, continuing without WandB: {exc}", level=logging.WARNING)

    torch_dtype = amp_dtype if amp_dtype in (torch.bfloat16, torch.float16) else None
    backbone = load_mlfm_backbone(config, torch_dtype=torch_dtype).to(device)
    tokenizer = backbone.tokenizer
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        raise ValueError("Tokenizer must expose a pad token for packed MLFM batches.")
    _log_embedding_geometry(backbone, wandb_run=wandb_run)
    bridge_noise_sampler = BridgeNoiseSampler.from_config(
        config,
        hidden_dim=hidden_dim_from_backbone(backbone),
        device=device,
    )
    log_for_0(
        "Bridge noise covariance: "
        f"mode={bridge_noise_sampler.mode}, "
        f"diag_min_tokens={bridge_noise_sampler.diag_min_tokens:,}, "
        f"diag_max_tokens={bridge_noise_sampler.diag_max_tokens:,}, "
        f"diag_shrinkage={bridge_noise_sampler.diag_shrinkage:g}, "
        f"rank={bridge_noise_sampler.rank}"
    )
    if bridge_noise_sampler.mode == "empirical_diag" and bridge_noise_sampler.rank:
        log_for_0("bridge_noise_rank is ignored for empirical_diag covariance.", level=logging.WARNING)
    if wandb_run is not None:
        wandb_run.summary["bridge_noise/mode"] = bridge_noise_sampler.mode
        wandb_run.summary["bridge_noise/rank"] = bridge_noise_sampler.rank

    trainable_count = sum(param.numel() for _, param in iter_trainable_named_parameters(backbone))
    total_count = sum(param.numel() for param in backbone.parameters())
    frozen_count = total_count - trainable_count
    adapter_counts = _adapter_param_counts(backbone)
    trainable_pct = 100.0 * trainable_count / max(total_count, 1)
    frozen_pct = 100.0 * frozen_count / max(total_count, 1)
    log_for_0(f"Total parameters: {total_count:,}")
    log_for_0(f"Trainable parameters: {trainable_count:,} ({trainable_pct:.4f}%)")
    log_for_0(f"Frozen parameters: {frozen_count:,} ({frozen_pct:.4f}%)")
    log_for_0(
        "Trainable adapter parameters: "
        f"LoRA={adapter_counts['lora']:,} "
        f"(backbone={adapter_counts['lora_backbone']:,}, output_head={adapter_counts['lora_output']:,}), "
        f"AdaLN/DiT={adapter_counts['adaln']:,}, other={adapter_counts['other']:,}"
    )
    if wandb_run is not None:
        wandb_run.summary["params/total"] = total_count
        wandb_run.summary["params/trainable"] = trainable_count
        wandb_run.summary["params/frozen"] = frozen_count
        wandb_run.summary["params/lora_trainable"] = adapter_counts["lora"]
        wandb_run.summary["params/lora_backbone_trainable"] = adapter_counts["lora_backbone"]
        wandb_run.summary["params/lora_output_trainable"] = adapter_counts["lora_output"]
        wandb_run.summary["params/adaln_trainable"] = adapter_counts["adaln"]
        wandb_run.summary["params/other_trainable"] = adapter_counts["other"]
    log_for_0(
        f"AdaLN/DiT adapters attached: {len(getattr(backbone, 'adaln_wrapped', []))} "
        f"(mode={getattr(config, 'adaln_mode', 'vanilla')}, "
        f"time_embed_dim={getattr(config, 'adaln_time_embed_dim', 256)}, "
        f"adaln_hidden_dim={getattr(config, 'adaln_hidden_dim', 0)}, "
        f"backbone_hidden_dim={getattr(config, 'backbone_hidden_dim', 0)})"
    )
    if getattr(backbone, "output_lora", None) is not None:
        log_for_0(f"Output-head LoRA enabled; tie status: {backbone.tie_info.tied} ({backbone.tie_info.reason})")

    optimizer = build_mlfm_optimizer(config, backbone)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and precision == "fp16"))
    start_step, epoch = 0, 0
    token_counters = _initial_token_counters()
    loss_diagnostic_state = _empty_loss_diagnostic_state(config)
    use_ema = bool(getattr(config, "use_ema", True))
    eval_use_ema = bool(getattr(config, "eval_use_ema", True))
    ema_decay = float(getattr(config, "ema_decay1", 0.9999))
    ema_state = {} if use_ema else None
    reset_resume_training_state = bool(getattr(config, "reset_resume_training_state", False))
    resume_adapter_weight_source = _normalize_adapter_weight_source(
        getattr(config, "resume_adapter_weight_source", "model")
    )
    if getattr(config, "resume", None):
        if reset_resume_training_state:
            checkpoint_step, checkpoint_epoch = load_mlfm_checkpoint(
                config.resume,
                backbone,
                optimizer=None,
                scaler=None,
                device=device,
                config=config,
                generator=None,
                token_counters=None,
                ema_state=None,
                bridge_noise_sampler=bridge_noise_sampler,
                restore_rng=False,
                adapter_weight_source=resume_adapter_weight_source,
            )
            start_step, epoch = 0, 0
            log_for_0(
                "Initialized MLFM weights from checkpoint "
                f"{config.resume} (checkpoint_step={checkpoint_step}, checkpoint_epoch={checkpoint_epoch}) "
                f"using adapter_weight_source={resume_adapter_weight_source}; "
                "reset training state: step=0, epoch=0, optimizer/scaler/RNG/token counters/EMA not restored."
            )
        else:
            if resume_adapter_weight_source != "model":
                raise ValueError(
                    "`resume_adapter_weight_source` must be `model` when reset_resume_training_state is false, "
                    "because optimizer/scaler state is restored for the live checkpoint weights."
                )
            start_step, epoch = load_mlfm_checkpoint(
                config.resume,
                backbone,
                optimizer,
                scaler,
                device=device,
                config=config,
                generator=generator,
                token_counters=token_counters,
                ema_state=ema_state,
                bridge_noise_sampler=bridge_noise_sampler,
                loss_diagnostic_state=loss_diagnostic_state,
                adapter_weight_source="model",
            )
            _diversify_nonzero_rank_resume_rng(config, rank, start_step, device, generator)
            log_for_0(f"Resumed MLFM checkpoint from step {start_step}")
            if world_size > 1:
                log_for_0(
                    "Resume RNG handling: restored rank-0 checkpoint RNG exactly; "
                    "nonzero ranks were deterministically reseeded from seed, rank, and step."
                )
            restored_diagnostic_batches = _loss_diagnostic_batches_for_fit(loss_diagnostic_state, config)
            if restored_diagnostic_batches:
                diagnostic_aggregate = _aggregate_loss_diagnostic_window(restored_diagnostic_batches)
                log_for_0(
                    "Restored loss diagnostic state: "
                    f"estimator={_loss_diagnostic_estimator(config)}, "
                    f"summaries={len(restored_diagnostic_batches)}, "
                    f"examples={diagnostic_aggregate.get('examples', 0):,}, "
                    f"total_examples={diagnostic_aggregate.get('total_examples', 0):,}"
                )
    if use_ema and not ema_state:
        ema_state = _init_ema_adapter_state(backbone)
    if use_ema:
        log_for_0(f"Adapter EMA enabled: decay={ema_decay}, eval_use_ema={eval_use_ema}")

    if bool(getattr(config, "compile", False)):
        log_for_0("Compiling MLFM backbone with torch.compile")
        backbone = torch.compile(backbone)

    if world_size > 1:
        backbone = DDP(backbone, device_ids=[device.index] if device.type == "cuda" else None, find_unused_parameters=True)

    if getattr(config, "global_batch_size", None) is not None:
        if config.global_batch_size % world_size != 0:
            raise ValueError("global_batch_size must be divisible by world_size.")
        local_batch_size = config.global_batch_size // world_size
    elif getattr(config, "batch_size", None) is not None:
        local_batch_size = config.batch_size
    else:
        raise ValueError("Either global_batch_size or batch_size must be set.")

    train_loader = load_mlfm_dataloader(
        config,
        tokenizer=tokenizer,
        batch_size=local_batch_size,
        train=True,
        distributed=True,
    )
    train_loader_stage = str(getattr(config, "training_stage", "pretrain") or "pretrain").lower()
    if train_loader_stage == "sft" and hasattr(train_loader, "source_names"):
        source_names = list(getattr(train_loader, "source_names", []))
        schedule = list(getattr(train_loader, "schedule", []))
        if schedule:
            counts = {source_names[idx] if idx < len(source_names) else f"source_{idx}": schedule.count(idx) for idx in sorted(set(schedule))}
            log_for_0(
                "SFT source schedule: "
                f"mode={getattr(config, 'sft_batching_mode', 'mixed_concat')}, "
                f"slots={len(schedule)}, counts={counts}, "
                f"dynamic_crop={bool(getattr(config, 'sft_dynamic_crop', False))}, "
                f"crop_multiple={int(getattr(config, 'sft_dynamic_crop_multiple', 64) or 64)}"
            )
        elif source_names:
            log_for_0(f"SFT source loaders: {source_names}")
    _log_empirical_embedding_geometry(backbone, train_loader, device, config, wandb_run=wandb_run)
    eval_loader = None
    if str(getattr(config, "training_stage", "pretrain") or "pretrain").lower() == "sft":
        eval_paths = _paths(getattr(config, "sft_eval_data_paths", None))
    else:
        eval_paths = _paths(getattr(config, "eval_data_paths", None), fallback=getattr(config, "eval_data_path", None))
    if eval_paths:
        eval_loader = load_mlfm_dataloader(
            config,
            tokenizer=tokenizer,
            batch_size=local_batch_size,
            train=False,
            distributed=True,
            drop_last=False,
        )

    max_steps = int(getattr(config, "max_train_steps", 100000))
    training_stage = str(getattr(config, "training_stage", "pretrain") or "pretrain").lower()
    if training_stage == "sft":
        sft_max_steps = int(getattr(config, "sft_max_steps", 0) or max_steps)
        sft_total_tokens = int(getattr(config, "sft_total_tokens", 0) or 0)
        if sft_total_tokens > 0:
            if getattr(config, "global_batch_size", None) is not None:
                global_batch = int(config.global_batch_size)
            else:
                global_batch = int(local_batch_size) * world_size
            tokens_per_step = global_batch * int(getattr(config, "grad_accum_steps", 1)) * int(getattr(config, "sft_max_length", None) or config.max_length)
            token_limited_steps = int(math.ceil(float(sft_total_tokens) / float(max(tokens_per_step, 1))))
            max_steps = min(sft_max_steps, token_limited_steps)
            log_for_0(
                "SFT token budget: "
                f"target_tokens={sft_total_tokens:,}, tokens_per_step={tokens_per_step:,}, "
                f"token_limited_steps={token_limited_steps:,}, max_steps={max_steps:,}"
            )
        else:
            max_steps = sft_max_steps
    if start_step >= max_steps:
        log_for_0(
            "Training start step is already at or beyond max_steps: "
            f"start_step={start_step:,}, max_steps={max_steps:,}. "
            "No gradient steps will run unless the resume state is reset or max_steps is increased.",
            level=logging.WARNING,
        )
        if training_stage == "sft" and getattr(config, "resume", None) and not reset_resume_training_state:
            log_for_0(
                "For SFT initialized from a pretraining checkpoint, set "
                "reset_resume_training_state: true so the adapter weights are loaded "
                "but the SFT step counter and learning-rate schedule start from zero.",
                level=logging.WARNING,
            )
    warmup_steps = int(getattr(config, "warmup_steps", 0) or max(2000, int(0.03 * max_steps)))
    min_lr_ratio = float(getattr(config, "min_lr_ratio", 0.1))
    grad_accum_steps = int(getattr(config, "grad_accum_steps", 1))
    log_freq = int(getattr(config, "log_freq", 20))
    eval_freq = int(getattr(config, "eval_freq", 1000))
    save_freq = int(getattr(config, "save_freq", 2000))
    random_length_prob = max(0.0, min(1.0, float(getattr(config, "random_length_prob", 0.0) or 0.0)))
    if training_stage == "pretrain" and random_length_prob > 0.0:
        random_length_max = int(getattr(config, "random_length_max", 0) or int(getattr(config, "max_length", 0) or 0))
        if random_length_max <= 0:
            random_length_max = int(getattr(config, "max_length", 0) or 0)
        expected_length = _random_length_expected_length(config, int(getattr(config, "max_length", random_length_max) or random_length_max))
        log_for_0(
            "Random-length pretraining enabled: "
            f"prob={random_length_prob:.4f}, min_length={int(getattr(config, 'random_length_min', 1) or 1)}, "
            f"max_length={random_length_max}, expected_length={expected_length:.2f}. "
            "When selected, the whole packed microbatch is cropped to one uniformly sampled prefix length before corruption."
        )
    os.makedirs(config.output_dir, exist_ok=True)
    if is_main_process():
        with open(os.path.join(config.output_dir, "config.yml"), "w", encoding="utf-8") as f:
            json.dump(_as_config_dict(config), f, indent=2, sort_keys=True)
    barrier()

    step = start_step
    restore_exact_iterator = bool(getattr(config, "restore_train_iterator_state", False))
    max_resume_skip = int(getattr(config, "resume_train_iterator_max_skip_batches", 2048))
    iterator_epoch, consumed_in_epoch, requested_resume_skip, resume_iterator_mode = _resume_train_iterator_plan(
        len(train_loader),
        epoch,
        start_step,
        grad_accum_steps,
        restore_exact=restore_exact_iterator,
        max_skip_batches=max_resume_skip,
    )
    train_iter, epoch = _restore_train_iterator(train_loader, iterator_epoch, consumed_in_epoch)
    if start_step > 0:
        if resume_iterator_mode == "exact":
            log_for_0(f"Restored train iterator exactly at epoch={epoch}, microbatch_offset={consumed_in_epoch}")
        elif resume_iterator_mode == "skip_limit":
            log_for_0(
                "Skipped exact train iterator replay on resume because requested offset "
                f"{requested_resume_skip:,} exceeds resume_train_iterator_max_skip_batches={max_resume_skip:,}. "
                f"Starting from next sampler epoch={epoch} instead.",
                level=logging.WARNING,
            )
        else:
            log_for_0(
                "Resume train iterator exact replay disabled; "
                f"requested offset would have been {requested_resume_skip:,} microbatches. "
                f"Starting from next sampler epoch={epoch} instead."
            )
    last_log_time = time.time()
    pending_metrics = []
    sample_diagnostics_batches_since_log = int(loss_diagnostic_state.get("batches_since_log", 0) or 0)
    sample_diagnostics_logging_enabled = bool(getattr(config, "use_wandb", False)) and bool(getattr(config, "sample_diagnostics", True))
    gamma_curve_adaptation_enabled = _gamma_curve_adaptation_enabled(config)
    gamma_curve_no_eos_source_enabled = _gamma_curve_uses_no_eos_ce(config) and gamma_curve_adaptation_enabled
    sample_diagnostics_enabled = (
        sample_diagnostics_logging_enabled
        or gamma_curve_adaptation_enabled
    )
    pbar = tqdm(total=max_steps, initial=start_step, disable=not is_main_process(), desc="mlfm")
    backbone.train()
    optimizer.zero_grad(set_to_none=True)

    while step < max_steps:
        lr_mult = _lr_multiplier(step, max_steps, warmup_steps, min_lr_ratio)
        set_group_lrs(optimizer, lr_mult)
        for accum_idx in range(grad_accum_steps):
            try:
                batch = next(train_iter)
            except StopIteration:
                epoch += 1
                if hasattr(train_loader, "set_epoch"):
                    train_loader.set_epoch(epoch)
                train_iter = iter(train_loader)
                batch = next(train_iter)
            batch = move_batch_to_device(batch, device)
            batch, random_length_metrics = _maybe_apply_random_length_training(batch, config, generator)
            batch, sft_crop_metrics = _maybe_apply_sft_dynamic_crop(batch, config)
            bridge_noise_sampler.update_from_batch(_unwrap(backbone), batch, special_token_ids=_as_special_ids(backbone, config))
            should_sync = accum_idx == grad_accum_steps - 1
            sync_ctx = backbone.no_sync() if hasattr(backbone, "no_sync") and not should_sync else torch.enable_grad()
            with sync_ctx:
                with autocast_context(device, amp_dtype):
                    loss, metrics = compute_mlfm_loss(
                        backbone,
                        batch,
                        config,
                        generator,
                        bridge_noise_sampler=bridge_noise_sampler,
                    )
                    metrics.update(random_length_metrics)
                    metrics.update(sft_crop_metrics)
                    scaled_loss = loss / grad_accum_steps
                scaler.scale(scaled_loss).backward()
            pending_metrics.append(metrics)

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_([param for group in optimizer.param_groups for param in group["params"]], float(getattr(config, "grad_clip", 1.0)))
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        step += 1
        if ema_state is not None:
            _update_ema_adapter_state(ema_state, backbone, ema_decay)
        pbar.update(1)

        if step % log_freq == 0:
            if sample_diagnostics_enabled:
                gathered_loss_diagnostic_batches = _gather_loss_diagnostic_batches(
                    _sample_loss_diagnostic_batches_from_pending(pending_metrics, config)
                )
                gathered_loss_diagnostic_no_eos_batches = (
                    _gather_loss_diagnostic_batches(
                        _sample_loss_diagnostic_batches_from_pending(
                            pending_metrics,
                            config,
                            ce_metric_key="sample_ce_no_eos",
                        )
                    )
                            if sample_diagnostics_logging_enabled or gamma_curve_no_eos_source_enabled
                    else []
                )
            else:
                gathered_loss_diagnostic_batches = []
                gathered_loss_diagnostic_no_eos_batches = []
            if is_main_process() and gathered_loss_diagnostic_batches:
                _update_loss_diagnostic_state(loss_diagnostic_state, gathered_loss_diagnostic_batches, config)
                sample_diagnostics_batches_since_log = int(loss_diagnostic_state.get("batches_since_log", 0) or 0)
            if is_main_process() and gathered_loss_diagnostic_no_eos_batches:
                _update_loss_diagnostic_state(
                    _loss_diagnostic_no_eos_state(loss_diagnostic_state, config),
                    gathered_loss_diagnostic_no_eos_batches,
                    config,
                )
            loss_diagnostic_batches = _loss_diagnostic_batches_for_fit(loss_diagnostic_state, config)
            loss_diagnostic_no_eos_batches = _loss_diagnostic_batches_for_fit(
                _loss_diagnostic_no_eos_state(loss_diagnostic_state, config),
                config,
            )
            loss_diagnostic_progress_metrics = _loss_diagnostic_original_bin_average_metrics(
                loss_diagnostic_batches,
                config,
                prefix="train",
            )
            gamma_curve_metrics = (
                _adapt_active_gamma_curve(
                    config,
                    _gamma_curve_batches_for_adaptation(
                        config,
                        loss_diagnostic_batches,
                        loss_diagnostic_no_eos_batches,
                    )[0],
                    step,
                    device,
                )
                if gamma_curve_adaptation_enabled
                else {}
            )
            aggregate = _aggregate_train_metrics(pending_metrics, config)
            now = time.time()
            elapsed = max(now - last_log_time, 1e-8)
            steps_per_sec = log_freq / elapsed
            _update_token_counters(token_counters, aggregate, elapsed)
            current_lrs = {f"lr_group_{idx}": group["lr"] for idx, group in enumerate(optimizer.param_groups)}
            gpu_metrics = _reduce_gpu_runtime_metrics(_local_gpu_runtime_metrics(device), device)
            record = {
                "type": "train",
                "step": step,
                "epoch": epoch,
                **aggregate,
                "steps_per_sec": steps_per_sec,
                **gpu_metrics,
                **current_lrs,
                **gamma_curve_metrics,
                **loss_diagnostic_progress_metrics,
                "train/loss_diagnostic_estimator": 1.0 if _loss_diagnostic_estimator(config) == "ema" else 0.0,
                "train/loss_diagnostic_ema_updates": float(loss_diagnostic_state.get("ema_updates", 0) or 0),
                **bridge_noise_sampler.metrics(),
            }
            log_for_0(record)
            pbar.set_postfix(loss=f"{aggregate['loss']:.4f}", corrupt=f"{aggregate['corrupt_fraction']:.3f}")
            if is_main_process():
                _log_jsonl(config.output_dir, record)
                if wandb_run is not None:
                    wandb_run.log(record, step=step)
                    update_every_batches = int(
                        getattr(config, "sample_diagnostics_update_every_batches", None) or max(1, log_freq)
                    )
                    if sample_diagnostics_logging_enabled and sample_diagnostics_batches_since_log >= update_every_batches:
                        _log_wandb_loss_diagnostics(
                            wandb_run,
                            step,
                            loss_diagnostic_batches,
                            config,
                            no_eos_batch_window=loss_diagnostic_no_eos_batches,
                        )
                        sample_diagnostics_batches_since_log = 0
                        loss_diagnostic_state["batches_since_log"] = 0
            pending_metrics = []
            last_log_time = now

        should_run_eval = eval_freq > 0 and step % eval_freq == 0 and (
            eval_loader is not None
            or (
                bool(getattr(config, "run_generation_validation", False))
                and training_stage == "sft"
            )
        )
        if should_run_eval:
            eval_generator = _make_generator(eval_seed_base + step, device)
            ema_ctx = _using_ema_adapter_weights(backbone, ema_state) if eval_use_ema and ema_state is not None else nullcontext()
            with ema_ctx:
                eval_metrics = {}
                if eval_loader is not None:
                    eval_metrics = evaluate_corrupted_ce(
                        backbone,
                        eval_loader,
                        config,
                        eval_generator,
                        device,
                        max_batches=int(getattr(config, "eval_max_batches", 32)),
                        amp_dtype=amp_dtype,
                        bridge_noise_sampler=bridge_noise_sampler,
                    )
                if bool(getattr(config, "run_generation_validation", False)):
                    generation_sample_rows = [] if is_main_process() and training_stage != "sft" else None
                    unconditional_sample_rows = [] if is_main_process() and training_stage != "sft" else None
                    gsm8k_sample_rows = [] if is_main_process() else None
                    sft_prompt_sample_rows = [] if is_main_process() else None
                    generation_ppl_texts = [] if is_main_process() and training_stage != "sft" and bool(getattr(config, "online_eval", True)) else None
                    unconditional_ppl_texts = [] if is_main_process() and training_stage != "sft" and bool(getattr(config, "online_eval", True)) else None
                    gsm8k_ppl_texts = [] if is_main_process() and bool(getattr(config, "online_eval", True)) else None
                    sft_prompt_ppl_texts = [] if is_main_process() and bool(getattr(config, "online_eval", True)) else None
                    generation_ppl_limit = int(
                        getattr(config, "generation_ppl_sample_count", None)
                        or getattr(config, "val_num_generation_samples", 64)
                        or 64
                    )
                    if eval_loader is not None and training_stage != "sft":
                        eval_metrics.update(
                            evaluate_generation_smoke(
                                backbone,
                                eval_loader,
                                config,
                                eval_generator,
                                device,
                                max_samples=int(getattr(config, "val_num_generation_samples", 64)),
                                amp_dtype=amp_dtype,
                                sample_rows=generation_sample_rows,
                                sample_limit=int(getattr(config, "wandb_generation_sample_count", 8)),
                                ppl_texts=generation_ppl_texts,
                                ppl_limit=generation_ppl_limit,
                                bridge_noise_sampler=bridge_noise_sampler,
                            )
                        )
                    if training_stage == "sft":
                        eval_metrics.update(
                            evaluate_sft_prompt_conditional_generations(
                                backbone,
                                train_loader,
                                config,
                                eval_generator,
                                device,
                                amp_dtype=amp_dtype,
                                sample_rows=sft_prompt_sample_rows,
                                sample_limit_per_type=int(
                                    getattr(config, "val_sft_prompt_generation_samples_per_type", 2)
                                ),
                                max_batches=int(getattr(config, "val_sft_prompt_generation_max_batches", 16)),
                                ppl_texts=sft_prompt_ppl_texts,
                                ppl_limit=generation_ppl_limit,
                                bridge_noise_sampler=bridge_noise_sampler,
                            )
                        )
                    else:
                        eval_metrics.update(
                            evaluate_unconditional_generations(
                                backbone,
                                config,
                                eval_generator,
                                device,
                                amp_dtype=amp_dtype,
                                sample_rows=unconditional_sample_rows,
                                sample_limit=int(getattr(config, "val_unconditional_generation_samples", 8)),
                                ppl_texts=unconditional_ppl_texts,
                                ppl_limit=generation_ppl_limit,
                                bridge_noise_sampler=bridge_noise_sampler,
                            )
                        )
                    eval_metrics.update(
                        evaluate_gsm8k_conditional_generations(
                            backbone,
                            config,
                            eval_generator,
                            device,
                            amp_dtype=amp_dtype,
                            sample_rows=gsm8k_sample_rows,
                            sample_limit=int(getattr(config, "val_gsm8k_generation_samples", 8)),
                            ppl_texts=gsm8k_ppl_texts,
                            ppl_limit=generation_ppl_limit,
                            bridge_noise_sampler=bridge_noise_sampler,
                        )
                    )
                else:
                    generation_sample_rows = None
                    unconditional_sample_rows = None
                    gsm8k_sample_rows = None
                    sft_prompt_sample_rows = None
                    generation_ppl_texts = None
                    unconditional_ppl_texts = None
                    gsm8k_ppl_texts = None
                    sft_prompt_ppl_texts = None
                generation_ppl_metrics = {}
                if bool(getattr(config, "run_generation_validation", False)) and is_main_process():
                    generation_ppl_metrics = _compute_generative_ppl_metrics(
                        config,
                        device,
                        {
                            "validation": generation_ppl_texts or [],
                            "sft_prompt_response": sft_prompt_ppl_texts or [],
                            "unconditional": unconditional_ppl_texts or [],
                            "gsm8k_answer": gsm8k_ppl_texts or [],
                        },
                    )
            eval_metrics = _reduce_float_metrics(eval_metrics, device)
            record = {"type": "eval", "step": step, **eval_metrics, **generation_ppl_metrics}
            log_for_0(record)
            if is_main_process():
                _log_jsonl(config.output_dir, record)
                if wandb_run is not None:
                    if generation_ppl_metrics:
                        wandb_run.summary["eval/generative_ppl_model"] = str(getattr(config, "eval_ppl_model", "gpt2-large"))
                    wandb_run.log(record, step=step)
                    _log_wandb_generation_samples(wandb_run, step, generation_sample_rows)
                    _log_wandb_unconditional_generations(wandb_run, step, unconditional_sample_rows)
                    _log_wandb_gsm8k_generations(
                        wandb_run,
                        step,
                        gsm8k_sample_rows,
                        table_key="eval_sft_gsm8k_generations" if training_stage == "sft" else "eval_gsm8k_conditional_generations",
                    )
                    _log_wandb_sft_prompt_generations(wandb_run, step, sft_prompt_sample_rows)
            backbone.train()

        if save_freq > 0 and step % save_freq == 0:
            if is_main_process():
                ckpt = save_mlfm_checkpoint(
                    backbone,
                    optimizer,
                    scaler,
                    config,
                    config.output_dir,
                    step,
                    epoch,
                    generator=generator,
                    token_counters=token_counters,
                    ema_state=ema_state,
                    bridge_noise_sampler=bridge_noise_sampler,
                    loss_diagnostic_state=loss_diagnostic_state,
                )
                log_for_0(f"Saved MLFM checkpoint: {ckpt}")
            barrier()

    pbar.close()
    if is_main_process():
        ckpt = save_mlfm_checkpoint(
            backbone,
            optimizer,
            scaler,
            config,
            config.output_dir,
            step,
            epoch,
            generator=generator,
            token_counters=token_counters,
            ema_state=ema_state,
            bridge_noise_sampler=bridge_noise_sampler,
            loss_diagnostic_state=loss_diagnostic_state,
        )
        log_for_0(f"Final MLFM checkpoint: {ckpt}")
        if wandb_run is not None:
            wandb_run.finish()
    barrier()


def run_mlfm_evaluation(config, checkpoint_path: str, seed: Optional[int] = None):
    """Evaluate a saved MLFM adapter checkpoint with corrupted-token CE/PPL."""
    device = setup_distributed_and_device(config)
    rank = get_rank()
    if seed is not None:
        config.seed = int(seed)
    torch.manual_seed(int(config.seed) + rank)
    generator = _make_generator(int(config.seed) + rank * 100003, device)
    precision, amp_dtype = resolve_precision(config, device)
    torch_dtype = amp_dtype if amp_dtype in (torch.bfloat16, torch.float16) else None

    wandb_run = None
    if bool(getattr(config, "use_wandb", False)) and is_main_process():
        try:
            import wandb

            wandb_run = wandb.init(
                project=getattr(config, "wandb_project", "MLFM"),
                entity=getattr(config, "wandb_entity", None),
                name=getattr(config, "wandb_run_name", None),
                group=getattr(config, "wandb_group", None),
                job_type=getattr(config, "wandb_job_type", None),
                tags=(getattr(config, "wandb_tag", None) or "").split(",") if getattr(config, "wandb_tag", None) else None,
                config=_as_config_dict(config),
                dir=_resolve_wandb_dir(config),
            )
        except Exception as exc:
            log_for_0(f"WandB init failed, continuing without WandB: {exc}", level=logging.WARNING)

    backbone = load_mlfm_backbone(config, torch_dtype=torch_dtype).to(device)
    bridge_noise_sampler = BridgeNoiseSampler.from_config(
        config,
        hidden_dim=hidden_dim_from_backbone(backbone),
        device=device,
    )
    eval_use_ema = bool(getattr(config, "eval_use_ema", True))
    ema_state = {} if eval_use_ema else None
    checkpoint_step, _ = load_mlfm_checkpoint(
        checkpoint_path,
        backbone,
        optimizer=None,
        scaler=None,
        device=device,
        config=config,
        ema_state=ema_state,
        bridge_noise_sampler=bridge_noise_sampler,
    )
    backbone.eval()
    tokenizer = backbone.tokenizer
    if tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer must expose a pad token for packed MLFM batches.")

    if str(getattr(config, "training_stage", "pretrain") or "pretrain").lower() == "sft":
        eval_paths = _paths(getattr(config, "sft_eval_data_paths", None))
    else:
        eval_paths = _paths(getattr(config, "eval_data_paths", None), fallback=getattr(config, "eval_data_path", None))
    if not eval_paths:
        raise ValueError("MLFM evaluation requires eval data paths for the configured training_stage.")
    world_size = get_world_size()
    if getattr(config, "global_batch_size", None) is not None:
        if config.global_batch_size % world_size != 0:
            raise ValueError("global_batch_size must be divisible by world_size.")
        local_batch_size = config.global_batch_size // world_size
    else:
        local_batch_size = int(getattr(config, "batch_size", 1) or 1)

    eval_loader = load_mlfm_dataloader(
        config,
        tokenizer=tokenizer,
        batch_size=local_batch_size,
        train=False,
        distributed=True,
        drop_last=False,
    )
    ema_ctx = _using_ema_adapter_weights(backbone, ema_state) if eval_use_ema and ema_state else nullcontext()
    with ema_ctx:
        metrics = evaluate_corrupted_ce(
            backbone,
            eval_loader,
            config,
            generator,
            device,
            max_batches=int(getattr(config, "eval_max_batches", 32)),
            amp_dtype=amp_dtype,
            bridge_noise_sampler=bridge_noise_sampler,
        )
        if bool(getattr(config, "run_generation_validation", False)):
            generation_sample_rows = [] if is_main_process() else None
            unconditional_sample_rows = [] if is_main_process() else None
            gsm8k_sample_rows = [] if is_main_process() else None
            generation_ppl_texts = [] if is_main_process() and bool(getattr(config, "online_eval", True)) else None
            unconditional_ppl_texts = [] if is_main_process() and bool(getattr(config, "online_eval", True)) else None
            gsm8k_ppl_texts = [] if is_main_process() and bool(getattr(config, "online_eval", True)) else None
            generation_ppl_limit = int(
                getattr(config, "generation_ppl_sample_count", None)
                or getattr(config, "val_num_generation_samples", 64)
                or 64
            )
            metrics.update(
                evaluate_generation_smoke(
                    backbone,
                    eval_loader,
                    config,
                    generator,
                    device,
                    max_samples=int(getattr(config, "val_num_generation_samples", 64)),
                    amp_dtype=amp_dtype,
                    sample_rows=generation_sample_rows,
                    sample_limit=int(getattr(config, "wandb_generation_sample_count", 8)),
                    ppl_texts=generation_ppl_texts,
                    ppl_limit=generation_ppl_limit,
                    bridge_noise_sampler=bridge_noise_sampler,
                )
            )
            metrics.update(
                evaluate_unconditional_generations(
                    backbone,
                    config,
                    generator,
                    device,
                    amp_dtype=amp_dtype,
                    sample_rows=unconditional_sample_rows,
                    sample_limit=int(getattr(config, "val_unconditional_generation_samples", 8)),
                    ppl_texts=unconditional_ppl_texts,
                    ppl_limit=generation_ppl_limit,
                    bridge_noise_sampler=bridge_noise_sampler,
                )
            )
            metrics.update(
                evaluate_gsm8k_conditional_generations(
                    backbone,
                    config,
                    generator,
                    device,
                    amp_dtype=amp_dtype,
                    sample_rows=gsm8k_sample_rows,
                    sample_limit=int(getattr(config, "val_gsm8k_generation_samples", 8)),
                    ppl_texts=gsm8k_ppl_texts,
                    ppl_limit=generation_ppl_limit,
                    bridge_noise_sampler=bridge_noise_sampler,
                )
            )
        else:
            generation_sample_rows = None
            unconditional_sample_rows = None
            gsm8k_sample_rows = None
            generation_ppl_texts = None
            unconditional_ppl_texts = None
            gsm8k_ppl_texts = None
        generation_ppl_metrics = {}
        if bool(getattr(config, "run_generation_validation", False)) and is_main_process():
            generation_ppl_metrics = _compute_generative_ppl_metrics(
                config,
                device,
                {
                    "validation": generation_ppl_texts or [],
                    "unconditional": unconditional_ppl_texts or [],
                    "gsm8k_answer": gsm8k_ppl_texts or [],
                },
            )
    final_metrics = _reduce_float_metrics(metrics, device)
    record = {
        "type": "eval",
        "step": checkpoint_step,
        "checkpoint": checkpoint_path,
        "precision": precision,
        **final_metrics,
        **generation_ppl_metrics,
        **bridge_noise_sampler.metrics(),
    }
    log_for_0(record)
    if is_main_process():
        _log_jsonl(config.output_dir, record)
        if wandb_run is not None:
            if generation_ppl_metrics:
                wandb_run.summary["eval/generative_ppl_model"] = str(getattr(config, "eval_ppl_model", "gpt2-large"))
            wandb_run.log(record, step=checkpoint_step)
            _log_wandb_generation_samples(wandb_run, checkpoint_step, generation_sample_rows)
            _log_wandb_unconditional_generations(wandb_run, checkpoint_step, unconditional_sample_rows)
            _log_wandb_gsm8k_generations(wandb_run, checkpoint_step, gsm8k_sample_rows)
            wandb_run.finish()
    barrier()
    return final_metrics
