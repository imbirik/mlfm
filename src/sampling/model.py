"""Checkpoint-backed model facade for standalone sampling."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Optional

import torch

from sampling.config import SamplingExperimentConfig
from mlfm.adapters import AdaLNWrapper, DiTBlockWrapper, LoRALinear, TiedOutputLoRA
from mlfm.loaders import load_mlfm_backbone
from mlfm.noise_geometry import BridgeNoiseSampler, hidden_dim_from_backbone
from mlfm.runner import _normalize_adapter_weight_source, load_mlfm_checkpoint
from mlfm.train_step import _as_special_ids, unwrap_backbone_helpers
from utils.train_utils import autocast_context, resolve_precision


def _clone_training_config(config):
    cloned = copy.copy(config)
    return cloned


@torch.no_grad()
def disable_adapters(model) -> None:
    """Force local adapters to identity/zero-delta behavior in-place."""
    module = unwrap_backbone_helpers(model)
    for child in module.modules():
        if isinstance(child, (LoRALinear, TiedOutputLoRA)):
            child.lora_a.weight.zero_()
            child.lora_b.weight.zero_()
        elif isinstance(child, (AdaLNWrapper, DiTBlockWrapper)):
            final = child.adaln_modulation[-1]
            final.weight.zero_()
            if final.bias is not None:
                final.bias.zero_()
    for name, param in module.named_parameters():
        if "lora_" in name.lower():
            param.zero_()


@dataclass
class SamplingModel:
    backbone: torch.nn.Module
    config: object
    sampling_config: SamplingExperimentConfig
    device: torch.device
    amp_dtype: Optional[torch.dtype]
    bridge_noise_sampler: Optional[BridgeNoiseSampler] = None
    checkpoint_step: int = 0
    checkpoint_epoch: int = 0

    @classmethod
    def from_config_checkpoint(
        cls,
        config,
        sampling_config: SamplingExperimentConfig,
        checkpoint_path: Optional[str],
        device: Optional[torch.device] = None,
    ) -> "SamplingModel":
        if device is None:
            configured = getattr(config, "device", "auto")
            if configured and configured != "auto":
                device = torch.device(configured)
            else:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model_config = _clone_training_config(config)
        adapter_mode = str(sampling_config.adapter_mode or "finetuned").lower()
        if adapter_mode == "base":
            model_config.lora_rank = 0
            model_config.adaln = False
            model_config.lora_output_head = False
        elif adapter_mode not in {"finetuned", "disabled"}:
            raise ValueError(f"Unknown adapter_mode: {adapter_mode}")

        _, amp_dtype = resolve_precision(model_config, device)
        torch_dtype = amp_dtype if amp_dtype in (torch.bfloat16, torch.float16) else None
        backbone = load_mlfm_backbone(model_config, torch_dtype=torch_dtype).to(device)
        backbone.eval()
        bridge_noise_sampler = BridgeNoiseSampler.from_config(
            model_config,
            hidden_dim=hidden_dim_from_backbone(unwrap_backbone_helpers(backbone)),
            device=device,
        )

        step = epoch = 0
        if checkpoint_path:
            checkpoint_weight_source = _normalize_adapter_weight_source(
                getattr(sampling_config, "checkpoint_weight_source", "ema")
            )
            step, epoch = load_mlfm_checkpoint(
                checkpoint_path,
                backbone,
                optimizer=None,
                scaler=None,
                device=device,
                config=model_config,
                bridge_noise_sampler=bridge_noise_sampler,
                restore_rng=False,
                adapter_weight_source=checkpoint_weight_source,
            )
        if adapter_mode == "disabled":
            disable_adapters(backbone)
        return cls(
            backbone=backbone,
            config=model_config,
            sampling_config=sampling_config,
            device=device,
            amp_dtype=amp_dtype,
            bridge_noise_sampler=bridge_noise_sampler,
            checkpoint_step=int(step),
            checkpoint_epoch=int(epoch),
        )

    @property
    def module(self):
        return unwrap_backbone_helpers(self.backbone)

    @property
    def tokenizer(self):
        return self.module.tokenizer

    @property
    def mask_token_id(self) -> int:
        return int(getattr(self.module, "mask_token_id", getattr(self.config, "mask_token_id", -1)))

    @property
    def eos_token_id(self) -> Optional[int]:
        value = getattr(self.tokenizer, "eos_token_id", None)
        return None if value is None else int(value)

    @property
    def special_token_ids(self) -> list[int]:
        return _as_special_ids(self.backbone, self.config)

    def embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.module.embed(input_ids)

    def mask_embedding(self) -> torch.Tensor:
        return self.module.mask_embedding()

    def forward_embeddings(
        self,
        embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
        t: torch.Tensor,
        observed_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        forward_embeddings = embeddings
        if self.amp_dtype in (torch.bfloat16, torch.float16):
            forward_embeddings = embeddings.to(dtype=self.amp_dtype)
        with autocast_context(self.device, self.amp_dtype):
            return self.module.forward_from_embeddings(forward_embeddings, attention_mask, t=t, observed_mask=observed_mask)

    def encode_prompts(self, prompts: list[str], max_length: Optional[int] = None) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        max_length = int(max_length or self.sampling_config.max_length)
        mask_id = self.mask_token_id
        input_ids = torch.full((len(prompts), max_length), mask_id, device=self.device, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        corrupt_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for row, prompt in enumerate(prompts):
            ids = self.tokenizer(str(prompt), add_special_tokens=False)["input_ids"]
            if len(ids) >= max_length:
                ids = ids[-max(1, max_length - 1):]
            prefix_len = len(ids)
            if prefix_len:
                input_ids[row, :prefix_len] = torch.tensor(ids, device=self.device, dtype=torch.long)
            corrupt_mask[row, prefix_len:] = True
        return {"input_ids": input_ids, "attention_mask": attention_mask}, corrupt_mask

    def decode(self, input_ids: torch.Tensor, skip_special_tokens: bool = True) -> list[str]:
        return self.tokenizer.batch_decode(input_ids.detach().cpu().tolist(), skip_special_tokens=skip_special_tokens)
