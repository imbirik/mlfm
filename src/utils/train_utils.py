"""Torch training utilities: DDP setup, train state, optimizer/schedule helpers."""

import math
import os
import queue
import threading
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.optim import Optimizer

from utils.logging_utils import log_for_0


def distributed_available() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if distributed_available() else 0


def get_world_size() -> int:
    return dist.get_world_size() if distributed_available() else 1


def is_main_process() -> bool:
    return get_rank() == 0


def barrier():
    if distributed_available():
        dist.barrier()


def setup_distributed_and_device(config):
    """Initialize DDP when launched by torchrun and return this rank's device."""
    is_torchrun = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if is_torchrun and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)

    if config.device == "auto":
        if torch.cuda.is_available():
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(config.device)
        if device.type == "cuda":
            torch.cuda.set_device(device)
    return device


def cleanup_distributed():
    if distributed_available():
        dist.destroy_process_group()


def resolve_precision(config, device: torch.device):
    precision = getattr(config, "precision", "bf16")
    if device.type != "cuda":
        return "fp32", None
    if precision == "bf16" and torch.cuda.is_bf16_supported():
        return "bf16", torch.bfloat16
    if precision == "fp16":
        return "fp16", torch.float16
    return "fp32", None


def autocast_context(device: torch.device, amp_dtype):
    if device.type == "cuda" and amp_dtype is not None:
        # PyTorch deprecated torch.cuda.amp.autocast in favor of the unified
        # torch.amp.autocast(device_type, ...) API. Keeping the helper here
        # prevents warnings across training, validation, sampling, and notebooks.
        return torch.amp.autocast("cuda", dtype=amp_dtype)
    return nullcontext()


@dataclass
class TrainState:
    model: nn.Module
    ema_model: Optional[nn.Module]
    optimizer: Optional[Optimizer]
    scheduler: Optional[Any]
    scaler: Optional[Any]
    step: int
    epoch: int
    device: torch.device

    def unwrapped_model(self):
        model = self.model.module if hasattr(self.model, "module") else self.model
        return model._orig_mod if hasattr(model, "_orig_mod") else model


def copy_model_for_ema(model: nn.Module) -> nn.Module:
    import copy

    ema_model = copy.deepcopy(model)
    ema_model.eval()
    for p in ema_model.parameters():
        p.requires_grad_(False)
    return ema_model


@torch.no_grad()
def update_ema(ema_model: nn.Module, model: nn.Module, decay: float):
    model = model.module if hasattr(model, "module") else model
    model = model._orig_mod if hasattr(model, "_orig_mod") else model
    ema_state = ema_model.state_dict()
    model_state = model.state_dict()
    for key, ema_value in ema_state.items():
        model_value = model_state[key]
        if torch.is_floating_point(ema_value):
            ema_value.mul_(decay).add_(model_value.detach(), alpha=1.0 - decay)
        else:
            ema_value.copy_(model_value)


def reduce_metrics(metrics):
    reduced = {}
    for key, value in metrics.items():
        tensor = value if isinstance(value, torch.Tensor) else torch.tensor(value)
        tensor = tensor.detach()
        if distributed_available():
            tensor = tensor.clone()
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            tensor /= get_world_size()
        reduced[key] = tensor
    return reduced


def prefetch_to_device(iterator, size=2):
    """Prefetch batches asynchronously on the host side."""
    q = queue.Queue(maxsize=size)

    def enqueue():
        for item in iterator:
            q.put(item)
        q.put(None)

    threading.Thread(target=enqueue, daemon=True).start()
    while True:
        item = q.get()
        if item is None:
            break
        yield item


def _orthogonalize_update(update: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Newton-Schulz orthogonalization used by Muon-style optimizers."""
    original_shape = update.shape
    update = update.reshape(update.shape[0], -1)
    transposed = False
    if update.shape[0] > update.shape[1]:
        update = update.T
        transposed = True
    update = update / (update.norm() + eps)
    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(steps):
        gram = update @ update.T
        update = a * update + (b * gram + c * gram @ gram) @ update
    if transposed:
        update = update.T
    return update.reshape(original_shape)


class Muon(Optimizer):
    """Small local Muon optimizer with AdamW fallback for vector parameters."""

    def __init__(self, params, lr=1e-3, weight_decay=0.0, momentum=0.95, betas=(0.9, 0.95), eps=1e-8):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum, betas=betas, eps=eps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            wd = group["weight_decay"]
            momentum = group["momentum"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if wd != 0:
                    p.mul_(1.0 - lr * wd)

                state = self.state[p]
                if p.ndim == 2:
                    buf = state.get("momentum_buffer")
                    if buf is None:
                        buf = torch.zeros_like(p)
                        state["momentum_buffer"] = buf
                    buf.mul_(momentum).add_(grad)
                    update = _orthogonalize_update(buf)
                    scale = math.sqrt(max(1.0, p.shape[0] / max(1, p.numel() // p.shape[0])))
                    p.add_(update, alpha=-lr * scale)
                else:
                    if len(state) == 0:
                        state["step"] = torch.tensor(0, device=p.device, dtype=torch.long)
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                    state["step"] += 1
                    exp_avg = state["exp_avg"]
                    exp_avg_sq = state["exp_avg_sq"]
                    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                    step = int(state["step"].item())
                    bias_correction1 = 1 - beta1**step
                    bias_correction2 = 1 - beta2**step
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
                    p.addcdiv_(exp_avg / bias_correction1, denom, value=-lr)
        return loss


def get_optimizer(config, params):
    """Build Torch AdamW or local Muon optimizer."""
    if config.optimizer == "muon":
        log_for_0("Using Muon optimizer")
        return Muon(params, lr=config.lr, weight_decay=config.weight_decay, betas=(config.adam_b1, config.adam_b2))
    if config.optimizer == "adamw":
        log_for_0("Using AdamW optimizer")
        return torch.optim.AdamW(
            params,
            lr=config.lr,
            weight_decay=config.weight_decay,
            betas=(config.adam_b1, config.adam_b2),
        )
    raise ValueError(f"Unknown optimizer: {config.optimizer}. Choose 'adamw' or 'muon'.")


def create_learning_rate_fn(
    num_train_steps: int,
    num_warmup_steps: int,
    learning_rate: float,
    schedule: str = "constant",
    min_lr: float = 0.0,
):
    """Create a Python learning-rate schedule over optimizer steps."""

    def lr_fn(step: int):
        if num_warmup_steps > 0 and step < num_warmup_steps:
            return learning_rate * (step + 1) / num_warmup_steps
        if schedule == "cosine":
            denom = max(num_train_steps - num_warmup_steps, 1)
            progress = min(max((step - num_warmup_steps) / denom, 0.0), 1.0)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr + (learning_rate - min_lr) * cosine
        return learning_rate

    return lr_fn


def set_optimizer_lr(optimizer: Optimizer, lr: float):
    for group in optimizer.param_groups:
        group["lr"] = lr
