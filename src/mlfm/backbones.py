"""Backbone adapters for MLFM."""

from __future__ import annotations

import inspect
import logging
import os
import sys
import types
from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoModelForMaskedLM, AutoTokenizer

from mlfm.adapters import (
    TiedOutputLoRA,
    adaln_timestep,
    attach_adaln_to_norms,
    attach_local_lora,
    attach_peft_lora,
    detect_tied_weights,
    freeze_module,
    trainable_state_dict,
)
from mlfm.defaults import fill_mlfm_defaults, get_backbone_defaults, resolve_smdm_size


logger = logging.getLogger(__name__)
OUTPUT_LORA_POLICIES = {"output_delta", "tied_delta", "untie_forbidden", "allow_untie"}


def _install_smdm_eval_dependency_fallbacks() -> None:
    """Install tiny eval-only fallbacks for optional official SMDM deps.

    The official SMDM LitGPT code imports flash-attn, xformers.ops.SwiGLU,
    fused rotary embedding, and a fused cross-entropy CUDA extension at module
    import time. Some evaluation clusters do not have matching CUDA/runtime
    wheels for those packages, and the fused CE extension is not needed for
    checkpoint loading or inference.
    """
    try:
        from flash_attn import flash_attn_func  # noqa: F401
    except (ImportError, OSError):
        flash_module = types.ModuleType("flash_attn")
        interface_module = types.ModuleType("flash_attn.flash_attn_interface")

        def flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False, *args, **kwargs):
            del args, kwargs
            q_heads = q.shape[2]
            k_heads = k.shape[2]
            if q_heads != k_heads:
                if q_heads % k_heads:
                    raise RuntimeError(f"Cannot expand {k_heads} KV heads to {q_heads} query heads.")
                k = k.repeat_interleave(q_heads // k_heads, dim=2)
                v = v.repeat_interleave(q_heads // v.shape[2], dim=2)
            q_t = q.transpose(1, 2)
            k_t = k.transpose(1, 2)
            v_t = v.transpose(1, 2)
            out = torch.nn.functional.scaled_dot_product_attention(
                q_t,
                k_t,
                v_t,
                dropout_p=float(dropout_p or 0.0),
                scale=softmax_scale,
                is_causal=bool(causal),
            )
            return out.transpose(1, 2)

        flash_module.flash_attn_func = flash_attn_func
        interface_module.flash_attn_func = flash_attn_func
        sys.modules["flash_attn"] = flash_module
        sys.modules["flash_attn.flash_attn_interface"] = interface_module

    try:
        from xformers.ops import SwiGLU  # noqa: F401
    except (ImportError, OSError):
        xformers_module = sys.modules.setdefault("xformers", types.ModuleType("xformers"))
        ops_module = types.ModuleType("xformers.ops")

        class SwiGLU(nn.Module):
            def __init__(self, in_features, hidden_features, out_features=None, bias=True, _pack_weights=False):
                super().__init__()
                if _pack_weights:
                    raise NotImplementedError("SMDM eval fallback supports only _pack_weights=False.")
                out_features = int(out_features if out_features is not None else in_features)
                self.w1 = nn.Linear(int(in_features), int(hidden_features), bias=bias)
                self.w2 = nn.Linear(int(in_features), int(hidden_features), bias=bias)
                self.w3 = nn.Linear(int(hidden_features), out_features, bias=bias)

            def forward(self, x):
                return self.w3(torch.nn.functional.silu(self.w1(x)) * self.w2(x))

        ops_module.SwiGLU = SwiGLU
        xformers_module.ops = ops_module
        sys.modules["xformers.ops"] = ops_module

    if "rotary_emb" not in sys.modules:
        try:
            __import__("rotary_emb")
        except (ImportError, OSError):
            rotary_stub = types.ModuleType("rotary_emb")

            def apply_rotary(x1, x2, cos, sin, out1, out2, conjugate=False):
                if conjugate:
                    sin = -sin
                out1.copy_(x1 * cos - x2 * sin)
                out2.copy_(x1 * sin + x2 * cos)

            rotary_stub.apply_rotary = apply_rotary
            sys.modules["rotary_emb"] = rotary_stub

    if "xentropy_cuda_lib" not in sys.modules:
        try:
            __import__("xentropy_cuda_lib")
        except (ImportError, OSError):
            fused_stub = types.ModuleType("xentropy_cuda_lib")

            def _missing_fused_ce(*args, **kwargs):
                raise RuntimeError("xentropy_cuda_lib is unavailable; fused CE is training-only for SMDM eval.")

            fused_stub.forward = _missing_fused_ce
            fused_stub.backward = _missing_fused_ce
            sys.modules["xentropy_cuda_lib"] = fused_stub

    if "dropout_layer_norm" not in sys.modules:
        try:
            __import__("dropout_layer_norm")
        except (ImportError, OSError):
            norm_stub = types.ModuleType("dropout_layer_norm")

            def _dropout_add_ln_fwd(
                x0mat,
                residualmat,
                gamma,
                beta,
                rowscale,
                colscale,
                x0_subset,
                out_subset,
                dropout_p,
                epsilon,
                rowscale_const,
                out_numrows,
                reserved,
                residual_in_fp32,
                is_rms_norm,
            ):
                if x0_subset is not None or out_subset is not None or out_numrows:
                    raise RuntimeError("dropout_layer_norm subset fallback is unavailable for SMDM eval.")
                x = x0mat.float()
                if residualmat is not None:
                    x = x + residualmat.float()
                if rowscale is not None:
                    x = x * rowscale.float().reshape(-1, 1)
                if colscale is not None:
                    x = x * colscale.float()
                if float(dropout_p or 0.0) > 0.0:
                    x = torch.nn.functional.dropout(x, p=float(dropout_p), training=True)

                if is_rms_norm:
                    mu = torch.zeros((x.shape[0], 1), device=x.device, dtype=torch.float32)
                    rsigma = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + float(epsilon))
                    z = x * rsigma
                else:
                    mu = x.mean(dim=-1, keepdim=True)
                    centered = x - mu
                    rsigma = torch.rsqrt(centered.pow(2).mean(dim=-1, keepdim=True) + float(epsilon))
                    z = centered * rsigma

                z = z * gamma.float()
                if beta is not None:
                    z = z + beta.float()
                return z.to(dtype=x0mat.dtype), x.to(dtype=x0mat.dtype), None, mu.squeeze(-1), rsigma.squeeze(-1)

            def _dropout_add_ln_bwd(*args, **kwargs):
                raise RuntimeError("dropout_layer_norm backward fallback is unavailable for SMDM eval.")

            def _dropout_add_ln_parallel_residual_fwd(*args, **kwargs):
                raise RuntimeError("dropout_layer_norm parallel fallback is unavailable for SMDM eval.")

            norm_stub.dropout_add_ln_fwd = _dropout_add_ln_fwd
            norm_stub.dropout_add_ln_bwd = _dropout_add_ln_bwd
            norm_stub.dropout_add_ln_parallel_residual_fwd = _dropout_add_ln_parallel_residual_fwd
            norm_stub.dropout_add_ln_parallel_residual_bwd = _dropout_add_ln_bwd
            sys.modules["dropout_layer_norm"] = norm_stub


def _adaln_mode(config):
    return str(getattr(config, "adaln_mode", "vanilla") or "vanilla")


def _adaln_time_embed_dim(config):
    return int(getattr(config, "adaln_time_embed_dim", 256) or 256)


def _adaln_hidden_size(config):
    return getattr(config, "adaln_hidden_dim", None)


def _backbone_hidden_dim(config):
    return int(getattr(config, "backbone_hidden_dim", 0) or 0)


class MLFMBackbone(nn.Module, ABC):
    """Minimal interface required by the MLFM objective."""

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        t: torch.Tensor,
        observed_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.forward_from_embeddings(inputs_embeds, attention_mask, t, observed_mask)

    @abstractmethod
    def embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def mask_embedding(self) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def forward_from_embeddings(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        t: torch.Tensor,
        observed_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def decode_logits(self, logits: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def supports_inputs_embeds(self) -> bool:
        raise NotImplementedError


def _as_list(value):
    if value is None:
        return None
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return list(value)


def _model_forward_accepts(model: nn.Module, name: str) -> bool:
    try:
        signature = inspect.signature(model.forward)
    except (TypeError, ValueError):
        return True
    return name in signature.parameters or any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values()
    )


def _load_hf_lm(model_name_or_path: str, trust_remote_code: bool, torch_dtype=None):
    kwargs = {"trust_remote_code": trust_remote_code}
    if torch_dtype is not None:
        kwargs["torch_dtype"] = torch_dtype
    try:
        return AutoModelForCausalLM.from_pretrained(model_name_or_path, **kwargs)
    except Exception as causal_error:
        try:
            return AutoModelForMaskedLM.from_pretrained(model_name_or_path, **kwargs)
        except Exception:
            raise causal_error


def _resolve_mask_token_id(config, tokenizer, model=None):
    configured = getattr(config, "mask_token_id", None)
    if configured is not None:
        return int(configured)
    tokenizer_mask = getattr(tokenizer, "mask_token_id", None)
    if tokenizer_mask is not None:
        return int(tokenizer_mask)
    model_config = getattr(model, "config", None)
    model_mask = getattr(model_config, "mask_token_id", None)
    if model_mask is not None:
        return int(model_mask)
    defaults = get_backbone_defaults(getattr(config, "backbone_type", "auto"))
    if defaults is not None:
        return int(defaults.mask_token_id)
    raise ValueError("`mask_token_id` is required when tokenizer/model defaults do not expose one.")


class HFMLFMBackbone(MLFMBackbone):
    """Hugging Face LM adapter used for LLaDA, SMDM, and compatible backbones."""

    def __init__(self, config, torch_dtype=None):
        super().__init__()
        self.config = fill_mlfm_defaults(config)
        self.model_name_or_path = getattr(config, "backbone_model_name_or_path", None)
        if not self.model_name_or_path:
            raise ValueError("`backbone_model_name_or_path` is required for mlfm.")

        tokenizer_name = getattr(config, "tokenizer_name_or_path", None) or self.model_name_or_path
        trust_remote_code = bool(getattr(config, "trust_remote_code", True))
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=trust_remote_code)
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = _load_hf_lm(self.model_name_or_path, trust_remote_code=trust_remote_code, torch_dtype=torch_dtype)
        if not self.supports_inputs_embeds():
            raise ValueError(f"Backbone {self.model_name_or_path!r} does not expose `inputs_embeds`.")

        self.mask_token_id = _resolve_mask_token_id(config, self.tokenizer, self.model)

        if bool(getattr(config, "freeze_backbone", True)):
            freeze_module(self.model)
        elif bool(getattr(config, "freeze_embeddings", True)):
            freeze_module(self.input_embeddings)

        if bool(getattr(config, "gradient_checkpointing", True)) and hasattr(self.model, "gradient_checkpointing_enable"):
            try:
                self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            except TypeError:
                self.model.gradient_checkpointing_enable()
            if hasattr(self.model.config, "use_cache"):
                self.model.config.use_cache = False

        lora_rank = int(getattr(config, "lora_rank", 0) or 0)
        if lora_rank > 0:
            self.model = attach_peft_lora(
                self.model,
                rank=lora_rank,
                alpha=float(getattr(config, "lora_alpha", 16.0)),
                dropout=float(getattr(config, "lora_dropout", 0.05)),
                target_modules=_as_list(getattr(config, "lora_target_modules", None)),
            )

        self.adaln_wrapped = []
        if bool(getattr(config, "adaln", True)):
            self.adaln_wrapped = attach_adaln_to_norms(
                self.model,
                include_patterns=_as_list(getattr(config, "adaln_include_patterns", None)),
                exclude_patterns=_as_list(getattr(config, "adaln_exclude_patterns", None)),
                mode=_adaln_mode(config),
                time_embed_dim=_adaln_time_embed_dim(config),
                adaln_hidden_size=_adaln_hidden_size(config),
                expected_hidden_size=_backbone_hidden_dim(config),
            )

        self.output_lora: Optional[TiedOutputLoRA] = None
        self.tie_info = detect_tied_weights(self.input_embeddings, self.output_embeddings)
        if bool(getattr(config, "lora_output_head", False)):
            self._init_output_head_lora(config)

    @property
    def input_embeddings(self):
        embeddings = self.model.get_input_embeddings()
        if embeddings is None:
            raise ValueError("Backbone does not expose input embeddings.")
        return embeddings

    @property
    def output_embeddings(self):
        if hasattr(self.model, "get_output_embeddings"):
            return self.model.get_output_embeddings()
        return getattr(self.model, "lm_head", None)

    @property
    def special_token_ids(self):
        ids = set(getattr(self.tokenizer, "all_special_ids", []) or [])
        ids.add(int(self.mask_token_id))
        pad_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_id is not None:
            ids.add(int(pad_id))
        return sorted(ids)

    def _init_output_head_lora(self, config):
        policy = getattr(config, "lora_output_tied_policy", "tied_delta")
        if policy not in OUTPUT_LORA_POLICIES:
            raise ValueError(f"Unknown lora_output_tied_policy: {policy}")
        if policy == "untie_forbidden" and not self.tie_info.tied:
            raise ValueError("Output-head LoRA requested with `untie_forbidden`, but base weights are not tied.")
        if policy == "allow_untie" and self.tie_info.tied:
            logger.warning("`allow_untie` selected; preserving base weights and adding an output delta adapter.")
        if policy == "tied_delta" and not self.tie_info.tied:
            logger.info("`tied_delta` requested for untied weights; using output-delta behavior.")

        if self.tie_info.output_weight is not None:
            vocab_size, hidden_size = self.tie_info.output_weight.shape
            vocab_size, hidden_size = int(vocab_size), int(hidden_size)
        else:
            hidden_size = int(getattr(self.input_embeddings, "embedding_dim"))
            vocab_size = int(getattr(self.input_embeddings, "num_embeddings"))
        self.output_lora = TiedOutputLoRA(
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            rank=int(getattr(config, "lora_output_rank", 8)),
            alpha=float(getattr(config, "lora_output_alpha", 16.0)),
            dropout=float(getattr(config, "lora_output_dropout", 0.0)),
        )

    def supports_inputs_embeds(self) -> bool:
        return _model_forward_accepts(self.model, "inputs_embeds")

    def embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        embedding_layer = self.input_embeddings
        if bool(getattr(self.config, "freeze_embeddings", True)):
            with torch.no_grad():
                return embedding_layer(input_ids)
        return embedding_layer(input_ids)

    def mask_embedding(self) -> torch.Tensor:
        mask_id = torch.tensor([int(self.mask_token_id)], device=self.input_embeddings.weight.device)
        return self.input_embeddings(mask_id)[0]

    def _extract_logits(self, outputs):
        if hasattr(outputs, "logits"):
            return outputs.logits
        if isinstance(outputs, dict) and "logits" in outputs:
            return outputs["logits"]
        if isinstance(outputs, (tuple, list)) and outputs:
            return outputs[0]
        raise ValueError("Backbone forward output does not contain logits.")

    def _extract_hidden_states(self, outputs):
        hidden_states = None
        if hasattr(outputs, "hidden_states"):
            hidden_states = outputs.hidden_states
        elif isinstance(outputs, dict):
            hidden_states = outputs.get("hidden_states")
        elif isinstance(outputs, (tuple, list)) and len(outputs) > 1:
            hidden_states = outputs[-1]
        if hidden_states is None or len(hidden_states) == 0:
            raise ValueError("Output-head LoRA requires hidden states from the backbone.")
        return hidden_states[-1]

    def forward_from_embeddings(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        t: torch.Tensor,
        observed_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        kwargs = {
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "use_cache": False,
        }
        if self.output_lora is not None:
            kwargs["output_hidden_states"] = True
        if not _model_forward_accepts(self.model, "use_cache"):
            kwargs.pop("use_cache", None)
        if "output_hidden_states" in kwargs and not _model_forward_accepts(self.model, "output_hidden_states"):
            kwargs.pop("output_hidden_states", None)

        with adaln_timestep(self.model, t):
            outputs = self.model(**kwargs)
        logits = self._extract_logits(outputs)

        if self.output_lora is None:
            return logits

        hidden_states = self._extract_hidden_states(outputs)
        output_head = self.output_embeddings
        base_weight = self.tie_info.input_weight if self.tie_info.tied else getattr(output_head, "weight", None)
        if base_weight is None:
            base_weight = self.tie_info.input_weight
        base_bias = getattr(output_head, "bias", None) if output_head is not None else None
        return self.output_lora(hidden_states, base_weight=base_weight, base_bias=base_bias, base_logits=logits)

    def decode_logits(self, logits: torch.Tensor) -> torch.Tensor:
        return logits.argmax(dim=-1)

    def adapter_state_dict(self):
        return trainable_state_dict(self)

    def load_adapter_state_dict(self, state_dict, strict: bool = False):
        return self.load_state_dict(state_dict, strict=strict)


class LLaDAMLFMBackbone(HFMLFMBackbone):
    """Named adapter for LLaDA-style Hugging Face checkpoints."""


def _download_hf_file_if_needed(path: str) -> str:
    if os.path.exists(path):
        return path
    parts = path.split("/")
    if len(parts) < 3:
        return path
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError("huggingface_hub is required to download SMDM checkpoints from HF.") from exc
    repo_id = "/".join(parts[:2])
    filename = "/".join(parts[2:])
    return hf_hub_download(repo_id=repo_id, filename=filename)


def _load_state_file(path: str):
    path = _download_hf_file_if_needed(path)
    if path.endswith(".safetensors"):
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise ImportError("safetensors is required to load official SMDM safetensors checkpoints.") from exc
        return load_file(path)
    payload = torch.load(path, map_location="cpu")
    return payload.get("model", payload.get("state_dict", payload)) if isinstance(payload, dict) else payload


class SMDMLitGPTBackbone(MLFMBackbone):
    """Adapter for official ML-GSAI/SMDM LitGPT-style checkpoints."""

    def __init__(self, config, torch_dtype=None):
        super().__init__()
        self.config = fill_mlfm_defaults(config)
        code_path = getattr(config, "smdm_code_path", None)
        if code_path and code_path not in sys.path:
            sys.path.insert(0, code_path)

        try:
            _install_smdm_eval_dependency_fallbacks()
            from lit_gpt.config import Config as SMDMConfig
            from lit_gpt.diffmodel import TransEncoder
        except ImportError as exc:
            raise ImportError(
                "SMDM `model_loader: smdm_litgpt` requires the official ML-GSAI/SMDM code on "
                "`PYTHONPATH` or `smdm_code_path`, plus its runtime deps. Clone "
                "https://github.com/ML-GSAI/SMDM and set `smdm_code_path` to that checkout, "
                "or set `model_loader: hf` with a converted HF-compatible SMDM model."
            ) from exc

        tokenizer_name = getattr(config, "tokenizer_name_or_path", None)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=bool(getattr(config, "trust_remote_code", True)))
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_size = resolve_smdm_size(config)
        smdm_config = SMDMConfig.from_name(f"Diff_LLaMA_{model_size}M")
        config.smdm_size = model_size
        if not getattr(config, "backbone_hidden_dim", 0):
            config.backbone_hidden_dim = int(smdm_config.n_embd)
        self.model = TransEncoder(smdm_config)
        checkpoint_path = getattr(config, "smdm_checkpoint_path", None)
        if checkpoint_path:
            self.model.load_state_dict(_load_state_file(checkpoint_path), strict=True)
        if torch_dtype is not None:
            self.model.to(dtype=torch_dtype)

        self.mask_token_id = _resolve_mask_token_id(config, self.tokenizer, self.model)
        if bool(getattr(config, "freeze_backbone", True)):
            freeze_module(self.model)
        elif bool(getattr(config, "freeze_embeddings", True)):
            freeze_module(self.input_embeddings)

        lora_rank = int(getattr(config, "lora_rank", 0) or 0)
        if lora_rank > 0:
            self.local_lora_wrapped = attach_local_lora(
                self.model,
                rank=lora_rank,
                alpha=float(getattr(config, "lora_alpha", 16.0)),
                dropout=float(getattr(config, "lora_dropout", 0.05)),
                target_modules=_as_list(getattr(config, "lora_target_modules", None)),
                freeze_base=bool(getattr(config, "lora_freeze_base", True)),
            )
        else:
            self.local_lora_wrapped = []

        self.adaln_wrapped = []
        if bool(getattr(config, "adaln", True)):
            self.adaln_wrapped = attach_adaln_to_norms(
                self.model,
                include_patterns=_as_list(getattr(config, "adaln_include_patterns", None)),
                exclude_patterns=_as_list(getattr(config, "adaln_exclude_patterns", None)),
                mode=_adaln_mode(config),
                time_embed_dim=_adaln_time_embed_dim(config),
                adaln_hidden_size=_adaln_hidden_size(config),
                expected_hidden_size=_backbone_hidden_dim(config),
            )

        self.output_lora: Optional[TiedOutputLoRA] = None
        self.tie_info = detect_tied_weights(self.input_embeddings, self.output_embeddings)
        if bool(getattr(config, "lora_output_head", False)):
            self._init_output_head_lora(config)

    @property
    def input_embeddings(self):
        return self.model.transformer.wte

    @property
    def output_embeddings(self):
        return self.model.lm_head

    @property
    def special_token_ids(self):
        ids = set(getattr(self.tokenizer, "all_special_ids", []) or [])
        ids.add(int(self.mask_token_id))
        pad_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_id is not None:
            ids.add(int(pad_id))
        return sorted(ids)

    def _init_output_head_lora(self, config):
        policy = getattr(config, "lora_output_tied_policy", "output_delta")
        if policy not in OUTPUT_LORA_POLICIES:
            raise ValueError(f"Unknown lora_output_tied_policy: {policy}")
        if policy == "untie_forbidden" and not self.tie_info.tied:
            raise ValueError("Output-head LoRA requested with `untie_forbidden`, but base weights are not tied.")
        if policy == "tied_delta" and not self.tie_info.tied:
            logger.info("`tied_delta` requested for untied SMDM weights; using output-delta behavior.")
        vocab_size, hidden_size = self.output_embeddings.weight.shape
        self.output_lora = TiedOutputLoRA(
            hidden_size=int(hidden_size),
            vocab_size=int(vocab_size),
            rank=int(getattr(config, "lora_output_rank", 8)),
            alpha=float(getattr(config, "lora_output_alpha", 16.0)),
            dropout=float(getattr(config, "lora_output_dropout", 0.0)),
        )

    def supports_inputs_embeds(self) -> bool:
        return True

    def embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        if bool(getattr(self.config, "freeze_embeddings", True)):
            with torch.no_grad():
                return self.input_embeddings(input_ids)
        return self.input_embeddings(input_ids)

    def mask_embedding(self) -> torch.Tensor:
        mask_id = torch.tensor([int(self.mask_token_id)], device=self.input_embeddings.weight.device)
        return self.input_embeddings(mask_id)[0]

    def forward_from_embeddings(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        t: torch.Tensor,
        observed_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        del attention_mask, observed_mask
        batch_size, seq_len = inputs_embeds.shape[:2]
        if (
            self.model.rope_cache is None
            or self.model.rope_cache[0].device != inputs_embeds.device
            or self.model.rope_cache[0].shape[0] < seq_len
        ):
            dummy_ids = torch.empty((batch_size, seq_len), dtype=torch.long, device=inputs_embeds.device)
            self.model.rope_cache = self.model.build_rope_cache(dummy_ids)
        cos, sin = self.model.rope_cache
        cos, sin = cos[:seq_len], sin[:seq_len]
        with adaln_timestep(self.model, t):
            hidden_states = inputs_embeds
            for block in self.model.transformer.h:
                hidden_states = block(hidden_states, (cos, sin))
            hidden_states = self.model.transformer.ln_f(hidden_states)
        logits = self.model.lm_head(hidden_states)
        if self.output_lora is None:
            return logits
        return self.output_lora(
            hidden_states,
            base_weight=self.output_embeddings.weight,
            base_bias=getattr(self.output_embeddings, "bias", None),
            base_logits=logits,
        )

    def decode_logits(self, logits: torch.Tensor) -> torch.Tensor:
        return logits.argmax(dim=-1)

    def adapter_state_dict(self):
        return trainable_state_dict(self)

    def load_adapter_state_dict(self, state_dict, strict: bool = False):
        return self.load_state_dict(state_dict, strict=strict)


class SMDMMLFMBackbone(HFMLFMBackbone):
    """HF-compatible adapter for converted SMDM checkpoints."""


def create_mlfm_backbone(config, torch_dtype=None) -> HFMLFMBackbone:
    config = fill_mlfm_defaults(config)
    backbone_type = str(getattr(config, "backbone_type", "auto")).lower()
    if backbone_type == "llada":
        return LLaDAMLFMBackbone(config, torch_dtype=torch_dtype)
    if backbone_type == "smdm":
        if getattr(config, "model_loader", None) == "smdm_litgpt":
            return SMDMLitGPTBackbone(config, torch_dtype=torch_dtype)
        return SMDMMLFMBackbone(config, torch_dtype=torch_dtype)
    if backbone_type == "auto":
        return HFMLFMBackbone(config, torch_dtype=torch_dtype)
    raise ValueError(f"Unknown backbone_type: {backbone_type}")
