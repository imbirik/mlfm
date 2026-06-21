"""Adapter helpers for MLFM backbones."""

from __future__ import annotations

import logging
import math
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


logger = logging.getLogger(__name__)


DEFAULT_LORA_TARGETS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    "fc1",
    "fc2",
    "wi",
    "wo",
)

ADALN_MODES = ("vanilla", "dit")


@dataclass
class OutputHeadTieInfo:
    """Description of whether input embeddings and output head share storage."""

    tied: bool
    reason: str
    input_weight: torch.Tensor
    output_weight: Optional[torch.Tensor]


def freeze_module(module: nn.Module):
    """Freeze all parameters in a module in place."""
    for param in module.parameters():
        param.requires_grad_(False)


def _storage_ptr(tensor: Optional[torch.Tensor]):
    if tensor is None:
        return None
    if hasattr(tensor, "untyped_storage"):
        return tensor.untyped_storage().data_ptr()
    return tensor.storage().data_ptr()


def detect_tied_weights(input_embedding: nn.Module, output_embedding: Optional[nn.Module]) -> OutputHeadTieInfo:
    """Return whether input and output embedding/head weights are tied."""
    input_weight = getattr(input_embedding, "weight", None)
    output_weight = getattr(output_embedding, "weight", None) if output_embedding is not None else None
    if input_weight is None:
        raise ValueError("Input embedding module has no weight parameter.")
    if output_weight is None:
        return OutputHeadTieInfo(False, "output head has no weight", input_weight, None)
    if input_weight is output_weight:
        return OutputHeadTieInfo(True, "same Parameter object", input_weight, output_weight)
    if _storage_ptr(input_weight) == _storage_ptr(output_weight):
        return OutputHeadTieInfo(True, "shared storage", input_weight, output_weight)
    return OutputHeadTieInfo(False, "distinct storage", input_weight, output_weight)


class TiedOutputLoRA(nn.Module):
    """LoRA delta for an output projection without mutating the frozen base weight."""

    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
    ):
        super().__init__()
        if rank <= 0:
            raise ValueError("Output-head LoRA rank must be positive.")
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = float(alpha) / float(rank)
        self.dropout = nn.Dropout(dropout)
        self.lora_a = nn.Linear(hidden_size, rank, bias=False)
        self.lora_b = nn.Linear(rank, vocab_size, bias=False)
        nn.init.kaiming_uniform_(self.lora_a.weight, a=5**0.5)
        nn.init.zeros_(self.lora_b.weight)

    def forward(
        self,
        hidden_states: torch.Tensor,
        base_weight: torch.Tensor,
        base_bias: Optional[torch.Tensor] = None,
        base_logits: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if base_logits is None:
            base_logits = F.linear(
                hidden_states,
                base_weight.detach(),
                base_bias.detach() if base_bias is not None else None,
            )
        lora_input = self.dropout(hidden_states).to(dtype=self.lora_a.weight.dtype)
        delta = self.lora_b(self.lora_a(lora_input)) * self.scaling
        return base_logits + delta.to(dtype=base_logits.dtype)


class LoRALinear(nn.Module):
    """Small local LoRA wrapper for non-HuggingFace modules."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float = 0.0, freeze_base: bool = True):
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive.")
        self.base = base
        if bool(freeze_base):
            for param in self.base.parameters():
                param.requires_grad_(False)
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = float(alpha) / float(rank)
        self.dropout = nn.Dropout(dropout)
        self.lora_a = nn.Linear(base.in_features, rank, bias=False)
        self.lora_b = nn.Linear(rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_a.weight, a=5**0.5)
        nn.init.zeros_(self.lora_b.weight)

    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_output = self.base(x)
        lora_input = self.dropout(x).to(dtype=self.lora_a.weight.dtype)
        delta = self.lora_b(self.lora_a(lora_input)) * self.scaling
        return base_output + delta.to(dtype=base_output.dtype)


def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    """Sinusoidal scalar timestep embedding used by MLFM/DiT-style conditioners."""
    if dim <= 0:
        raise ValueError("AdaLN timestep embedding dimension must be positive.")
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(0, half, device=t.device, dtype=torch.float32) / max(half, 1)
    )
    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        pad = torch.zeros(t.shape[0], 1, device=t.device, dtype=embedding.dtype)
        embedding = torch.cat([embedding, pad], dim=-1)
    return embedding


class AdaLNWrapper(nn.Module):
    """Timestep-conditioned shift/scale wrapper around a normalization module."""

    def __init__(
        self,
        norm: nn.Module,
        hidden_size: int,
        mode: str = "vanilla",
        time_embed_dim: int = 256,
        adaln_hidden_size: Optional[int] = None,
    ):
        super().__init__()
        self.norm = norm
        self.hidden_size = int(hidden_size)
        self.mode = str(mode or "vanilla").lower()
        if self.mode not in ADALN_MODES:
            raise ValueError(f"Unknown AdaLN mode {mode!r}; expected one of {ADALN_MODES}.")
        if self.mode == "dit":
            raise ValueError("AdaLNWrapper implements only norm-site adapters; use block-level `dit` wrapping.")
        self.time_embed_dim = int(time_embed_dim)
        if self.time_embed_dim <= 0:
            raise ValueError("AdaLN `time_embed_dim` must be positive.")
        self.adaln_hidden_size = int(adaln_hidden_size or hidden_size)
        self.time_mlp = nn.Sequential(
            nn.Linear(self.time_embed_dim, self.adaln_hidden_size),
            nn.SiLU(),
            nn.Linear(self.adaln_hidden_size, self.adaln_hidden_size),
        )
        self.adaln_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.adaln_hidden_size, 2 * self.hidden_size),
        )
        nn.init.zeros_(self.adaln_modulation[-1].weight)
        nn.init.zeros_(self.adaln_modulation[-1].bias)
        self._current_t: Optional[torch.Tensor] = None

    def set_timestep(self, t: Optional[torch.Tensor]):
        self._current_t = t

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm(x)
        if self._current_t is None:
            return y
        t = self._current_t.to(device=y.device, dtype=torch.float32)
        if t.ndim == 0:
            t = t.reshape(1)
        while t.ndim > 1:
            t = t.reshape(t.shape[0], -1)[:, 0]
        embed_dtype = self.time_mlp[0].weight.dtype
        t_emb = timestep_embedding(t, self.time_embed_dim).to(device=y.device, dtype=embed_dtype)
        cond = self.adaln_modulation(self.time_mlp(t_emb)).to(dtype=y.dtype)

        shift, scale = cond.chunk(2, dim=-1)
        while scale.ndim < y.ndim:
            shift = shift.unsqueeze(1)
            scale = scale.unsqueeze(1)
        return y * (1.0 + scale) + shift


def _first_attr(module: nn.Module, names: Sequence[str]):
    for name in names:
        if hasattr(module, name):
            return name, getattr(module, name)
    return None, None


class DiTBlockWrapper(nn.Module):
    """DiT-style timestep adapter for pre-norm Transformer blocks.

    The wrapper keeps the frozen block's branch structure, modulates each
    normalization input with shift/scale, and gates the attention/MLP residual
    branches. The gate is applied as `(1 + gate)` so zero initialization
    preserves a pretrained backbone exactly.
    """

    norm1_names = ("norm_1", "norm1", "input_layernorm", "attn_norm")
    norm2_names = ("norm_2", "norm2", "post_attention_layernorm", "ff_norm", "ffn_norm")
    attn_names = ("attn", "self_attn", "attention")
    mlp_names = ("mlp", "feed_forward", "ffn")

    def __init__(
        self,
        block: nn.Module,
        hidden_size: int,
        time_embed_dim: int = 256,
        adaln_hidden_size: Optional[int] = None,
    ):
        super().__init__()
        self.block = block
        self.hidden_size = int(hidden_size)
        self.time_embed_dim = int(time_embed_dim)
        if self.time_embed_dim <= 0:
            raise ValueError("DiT `time_embed_dim` must be positive.")
        self.adaln_hidden_size = int(adaln_hidden_size or hidden_size)
        self.time_mlp = nn.Sequential(
            nn.Linear(self.time_embed_dim, self.adaln_hidden_size),
            nn.SiLU(),
            nn.Linear(self.adaln_hidden_size, self.adaln_hidden_size),
        )
        self.adaln_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.adaln_hidden_size, 6 * self.hidden_size),
        )
        nn.init.zeros_(self.adaln_modulation[-1].weight)
        nn.init.zeros_(self.adaln_modulation[-1].bias)
        self._current_t: Optional[torch.Tensor] = None

        self.norm1_name, norm1 = _first_attr(block, self.norm1_names)
        self.norm2_name, norm2 = _first_attr(block, self.norm2_names)
        self.attn_name, attn = _first_attr(block, self.attn_names)
        self.mlp_name, mlp = _first_attr(block, self.mlp_names)
        if norm1 is None or norm2 is None or attn is None or mlp is None:
            raise ValueError(f"Block {block.__class__.__name__} is not a supported pre-norm DiT block.")

    def set_timestep(self, t: Optional[torch.Tensor]):
        self._current_t = t

    def _conditioning(self, x: torch.Tensor):
        if self._current_t is None:
            return None
        t = self._current_t.to(device=x.device, dtype=torch.float32)
        if t.ndim == 0:
            t = t.reshape(1)
        while t.ndim > 1:
            t = t.reshape(t.shape[0], -1)[:, 0]
        embed_dtype = self.time_mlp[0].weight.dtype
        t_emb = timestep_embedding(t, self.time_embed_dim).to(device=x.device, dtype=embed_dtype)
        cond = self.adaln_modulation(self.time_mlp(t_emb)).to(dtype=x.dtype)
        return cond.chunk(6, dim=-1)

    @staticmethod
    def _expand_to_tokens(value: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        while value.ndim < target.ndim:
            value = value.unsqueeze(1)
        return value

    def _modulate(self, x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        shift = self._expand_to_tokens(shift, x)
        scale = self._expand_to_tokens(scale, x)
        return x * (1.0 + scale) + shift

    def _apply_gate(self, residual_branch: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        gate = self._expand_to_tokens(gate, residual_branch)
        return residual_branch * (1.0 + gate)

    def _call_attention(self, x: torch.Tensor, *args, **kwargs):
        output = getattr(self.block, self.attn_name)(x, *args, **kwargs)
        if isinstance(output, tuple):
            return output[0], output[1:]
        return output, ()

    def _call_mlp(self, x: torch.Tensor, **kwargs):
        mlp = getattr(self.block, self.mlp_name)
        try:
            return mlp(x, deterministic=kwargs["deterministic"])
        except (KeyError, TypeError):
            return mlp(x)

    def forward(self, x: torch.Tensor, *args, **kwargs):
        cond = self._conditioning(x)
        if cond is None:
            return self.block(x, *args, **kwargs)
        shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = cond

        config = getattr(self.block, "config", None)
        parallel_residual = bool(getattr(config, "parallel_residual", False))
        shared_attention_norm = bool(getattr(config, "shared_attention_norm", False))

        norm1 = getattr(self.block, self.norm1_name)
        norm2 = getattr(self.block, self.norm2_name)

        attn_input = self._modulate(norm1(x), shift_attn, scale_attn)
        attn_output, attn_tail = self._call_attention(attn_input, *args, **kwargs)

        if parallel_residual:
            if shared_attention_norm:
                mlp_input = attn_input
            else:
                mlp_input = self._modulate(norm2(x), shift_mlp, scale_mlp)
            x = x + self._apply_gate(attn_output, gate_attn) + self._apply_gate(self._call_mlp(mlp_input, **kwargs), gate_mlp)
        else:
            x = x + self._apply_gate(attn_output, gate_attn)
            mlp_input = self._modulate(norm2(x), shift_mlp, scale_mlp)
            x = x + self._apply_gate(self._call_mlp(mlp_input, **kwargs), gate_mlp)

        if attn_tail:
            return (x, *attn_tail)
        return x


def set_adaln_timestep(module: nn.Module, t: Optional[torch.Tensor]):
    """Propagate a timestep tensor to all AdaLN wrappers in a model."""
    for child in module.modules():
        if isinstance(child, (AdaLNWrapper, DiTBlockWrapper)):
            child.set_timestep(t)


@contextmanager
def adaln_timestep(module: nn.Module, t: Optional[torch.Tensor]):
    set_adaln_timestep(module, t)
    try:
        yield
    finally:
        set_adaln_timestep(module, None)


def _is_norm_like(module: nn.Module) -> bool:
    if isinstance(module, nn.LayerNorm):
        return True
    class_name = module.__class__.__name__.lower()
    has_weight = getattr(module, "weight", None) is not None
    return has_weight and ("rmsnorm" in class_name or "layernorm" in class_name or class_name.endswith("norm"))


def _hidden_size_for_norm(module: nn.Module) -> Optional[int]:
    weight = getattr(module, "weight", None)
    if weight is not None and weight.ndim == 1:
        return int(weight.shape[0])
    normalized_shape = getattr(module, "normalized_shape", None)
    if normalized_shape is None:
        return None
    if isinstance(normalized_shape, int):
        return int(normalized_shape)
    return int(normalized_shape[-1])


def _module_matches_any(name: str, module: nn.Module, patterns: Sequence[str]) -> bool:
    if not patterns:
        return True
    lowered = name.lower()
    if any(pattern in lowered for pattern in patterns):
        return True
    child_names = (
        *DiTBlockWrapper.norm1_names,
        *DiTBlockWrapper.norm2_names,
        *DiTBlockWrapper.attn_names,
        *DiTBlockWrapper.mlp_names,
    )
    return any(hasattr(module, child) and any(pattern in f"{lowered}.{child}" for pattern in patterns) for child in child_names)


def _is_dit_block_candidate(module: nn.Module) -> bool:
    if isinstance(module, (AdaLNWrapper, DiTBlockWrapper)):
        return False
    return (
        _first_attr(module, DiTBlockWrapper.norm1_names)[1] is not None
        and _first_attr(module, DiTBlockWrapper.norm2_names)[1] is not None
        and _first_attr(module, DiTBlockWrapper.attn_names)[1] is not None
        and _first_attr(module, DiTBlockWrapper.mlp_names)[1] is not None
    )


def _get_parent_module(root: nn.Module, dotted_name: str) -> Tuple[nn.Module, str]:
    parts = dotted_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def attach_local_lora(
    model: nn.Module,
    rank: int,
    alpha: float,
    dropout: float,
    target_modules: Optional[Sequence[str]] = None,
    freeze_base: bool = True,
) -> List[str]:
    """Attach local LoRA wrappers to matching Linear modules."""
    if rank <= 0:
        return []
    targets = set(target_modules or infer_lora_target_modules(model, preferred_targets=DEFAULT_LORA_TARGETS + ("attn", "proj", "fc")))
    replacements = []
    for name, module in list(model.named_modules()):
        if not name or not isinstance(module, nn.Linear):
            continue
        lowered = name.lower()
        if any(skip in lowered for skip in ("embed", "embedding", "wte", "lm_head", "output")):
            continue
        if name.split(".")[-1] in targets:
            replacements.append((name, module))
    for name, module in replacements:
        parent, child_name = _get_parent_module(model, name)
        setattr(parent, child_name, LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout, freeze_base=freeze_base))
    return [name for name, _ in replacements]


def attach_dit_to_blocks(
    model: nn.Module,
    include_patterns: Optional[Sequence[str]] = None,
    exclude_patterns: Optional[Sequence[str]] = None,
    time_embed_dim: int = 256,
    adaln_hidden_size: Optional[int] = None,
    expected_hidden_size: Optional[int] = None,
) -> List[str]:
    """Wrap supported pre-norm transformer blocks with DiT branch gating."""
    include_patterns = tuple(pattern.lower() for pattern in (include_patterns or ()))
    exclude_patterns = tuple(pattern.lower() for pattern in (exclude_patterns or ("embed", "embedding", "lm_head", "output")))
    replacements = []
    for name, module in list(model.named_modules()):
        if not name or not _is_dit_block_candidate(module):
            continue
        lowered = name.lower()
        if any(pattern in lowered for pattern in exclude_patterns):
            continue
        if include_patterns and not _module_matches_any(name, module, include_patterns):
            continue
        _, norm1 = _first_attr(module, DiTBlockWrapper.norm1_names)
        hidden_size = _hidden_size_for_norm(norm1)
        if hidden_size is None:
            continue
        if expected_hidden_size not in (None, 0) and hidden_size != int(expected_hidden_size):
            logger.warning(
                "DiT wrapping %s with norm hidden size %s, but configured backbone_hidden_dim is %s.",
                name,
                hidden_size,
                expected_hidden_size,
            )
        replacements.append((name, module, hidden_size))

    wrapped_names = []
    for name, module, hidden_size in replacements:
        parent, child_name = _get_parent_module(model, name)
        setattr(
            parent,
            child_name,
            DiTBlockWrapper(
                module,
                hidden_size=hidden_size,
                time_embed_dim=time_embed_dim,
                adaln_hidden_size=adaln_hidden_size,
            ),
        )
        wrapped_names.append(name)
    if not wrapped_names:
        raise ValueError(
            "`adaln_mode: dit` requires supported pre-norm transformer blocks with "
            "attention, MLP, and two normalization modules."
        )
    return wrapped_names


def attach_adaln_to_norms(
    model: nn.Module,
    include_patterns: Optional[Sequence[str]] = None,
    exclude_patterns: Optional[Sequence[str]] = None,
    mode: str = "vanilla",
    time_embed_dim: int = 256,
    adaln_hidden_size: Optional[int] = None,
    expected_hidden_size: Optional[int] = None,
) -> List[str]:
    """Wrap normalization modules with AdaLN and return wrapped module names."""
    mode = str(mode or "vanilla").lower()
    if mode == "dit":
        return attach_dit_to_blocks(
            model,
            include_patterns=include_patterns,
            exclude_patterns=exclude_patterns,
            time_embed_dim=time_embed_dim,
            adaln_hidden_size=adaln_hidden_size,
            expected_hidden_size=expected_hidden_size,
        )
    include_patterns = tuple(pattern.lower() for pattern in (include_patterns or ()))
    exclude_patterns = tuple(pattern.lower() for pattern in (exclude_patterns or ("embed", "embedding", "lm_head", "output")))
    replacements = []
    for name, module in list(model.named_modules()):
        if not name or isinstance(module, AdaLNWrapper):
            continue
        lowered = name.lower()
        if include_patterns and not any(pattern in lowered for pattern in include_patterns):
            continue
        if any(pattern in lowered for pattern in exclude_patterns):
            continue
        hidden_size = _hidden_size_for_norm(module)
        if hidden_size is None or not _is_norm_like(module):
            continue
        if expected_hidden_size not in (None, 0) and hidden_size != int(expected_hidden_size):
            logger.warning(
                "AdaLN wrapping %s with norm hidden size %s, but configured backbone_hidden_dim is %s.",
                name,
                hidden_size,
                expected_hidden_size,
            )
        replacements.append((name, module, hidden_size))

    wrapped_names = []
    for name, module, hidden_size in replacements:
        parent, child_name = _get_parent_module(model, name)
        setattr(
            parent,
            child_name,
            AdaLNWrapper(
                module,
                hidden_size,
                mode=mode,
                time_embed_dim=time_embed_dim,
                adaln_hidden_size=adaln_hidden_size,
            ),
        )
        wrapped_names.append(name)
    return wrapped_names


def infer_lora_target_modules(model: nn.Module, preferred_targets: Optional[Iterable[str]] = None) -> List[str]:
    """Return target module suffixes that actually occur in a model."""
    preferred = tuple(preferred_targets or DEFAULT_LORA_TARGETS)
    found = set()
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        suffix = name.split(".")[-1]
        if suffix in preferred:
            found.add(suffix)
    return sorted(found)


def attach_peft_lora(
    model: nn.Module,
    rank: int,
    alpha: float,
    dropout: float,
    target_modules: Optional[Sequence[str]] = None,
):
    """Attach PEFT LoRA modules to a Hugging Face model."""
    if rank <= 0:
        return model
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as exc:
        raise ImportError(
            "PEFT is required for mlfm LoRA training. Install `peft` or set `lora_rank: 0`."
        ) from exc

    targets = list(target_modules or infer_lora_target_modules(model))
    if not targets:
        raise ValueError(
            "Could not infer LoRA target modules. Set `lora_target_modules` explicitly in the config."
        )
    peft_config = LoraConfig(
        r=int(rank),
        lora_alpha=float(alpha),
        lora_dropout=float(dropout),
        bias="none",
        target_modules=targets,
    )
    return get_peft_model(model, peft_config)


def iter_trainable_named_parameters(module: nn.Module):
    for name, param in module.named_parameters():
        if param.requires_grad:
            yield name, param


def trainable_state_dict(module: nn.Module):
    """Return a state dict containing only trainable parameters plus buffers from trainable modules."""
    names = {name for name, _ in iter_trainable_named_parameters(module)}
    full = module.state_dict()
    return {name: value for name, value in full.items() if name in names or "lora" in name.lower() or "adaln" in name.lower()}
