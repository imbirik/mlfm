"""Convenience loaders for MLFM models and data."""

from __future__ import annotations

from typing import Optional

from mlfm.backbones import create_mlfm_backbone
from mlfm.data_mix import get_mlfm_dataloader
from mlfm.defaults import fill_mlfm_defaults
from mlfm.sft_data import ObjectiveMixtureDataLoader, get_sft_dataloader, has_sft_data


def load_mlfm_backbone(config, torch_dtype=None):
    """Resolve default paths/ids and instantiate the configured backbone."""
    fill_mlfm_defaults(config)
    return create_mlfm_backbone(config, torch_dtype=torch_dtype)


def load_mlfm_dataloader(
    config,
    tokenizer,
    batch_size: int,
    train: bool = True,
    distributed: bool = True,
    drop_last: Optional[bool] = None,
):
    """Resolve default packed dataset paths and instantiate the mixture dataloader."""
    fill_mlfm_defaults(config)
    max_length = int(getattr(config, "sft_max_length", None) or config.max_length)
    eos_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.pad_token_id
    sft_stage = str(getattr(config, "training_stage", "pretrain") or "pretrain").lower() == "sft"
    paths = getattr(config, "data_paths" if train else "eval_data_paths", None)
    fallback = getattr(config, "data_path" if train else "eval_data_path", None)
    if paths is None and fallback is not None:
        paths = fallback
    if drop_last is None:
        drop_last = train

    if sft_stage:
        return get_sft_dataloader(
            config,
            batch_size=batch_size,
            max_length=max_length,
            sft_pad_token_id=eos_token_id,
            num_workers=int(getattr(config, "num_workers", 0)),
            shuffle=train,
            drop_last=drop_last,
            distributed=distributed,
            train=train,
        )

    packed_loader = get_mlfm_dataloader(
        data_paths=paths,
        mix_weights=getattr(config, "data_mix_weights", None),
        batch_size=batch_size,
        max_length=int(config.max_length),
        pad_token_id=tokenizer.pad_token_id,
        num_workers=int(getattr(config, "num_workers", 0)),
        shuffle=train,
        drop_last=drop_last,
        distributed=distributed,
    )
    sft_mix_weight = float(getattr(config, "sft_mix_weight", 0.0) or 0.0)
    if train and sft_mix_weight > 0.0 and has_sft_data(config, train=True):
        sft_loader = get_sft_dataloader(
            config,
            batch_size=batch_size,
            max_length=max_length,
            sft_pad_token_id=eos_token_id,
            num_workers=int(getattr(config, "num_workers", 0)),
            shuffle=True,
            drop_last=True,
            distributed=distributed,
            train=True,
        )
        return ObjectiveMixtureDataLoader(
            [packed_loader, sft_loader],
            weights=[max(0.0, 1.0 - sft_mix_weight), sft_mix_weight],
            names=["packed", "sft"],
        )
    return packed_loader
