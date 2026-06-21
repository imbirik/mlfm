"""Default paths and ids for MLFM experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


DEFAULT_TOKENIZED_DATA_ROOT = "./data/mlfm"
DEFAULT_SMDM_SIZE = 1028

SMDM_CHECKPOINT_FILENAMES: Dict[int, str] = {
    170: "mdm-170M-100e18.safetensors",
    231: "mdm-231M-100e18.safetensors",
    336: "mdm-336M-100e18.safetensors",
    472: "mdm-472M-100e18.safetensors",
    551: "mdm-551M-60e18.safetensors",
    629: "mdm-629M-100e18.safetensors",
    717: "mdm-717M-60e18.safetensors",
    831: "mdm-831M-100e18.safetensors",
    944: "mdm-944M-60e18.safetensors",
    1028: "mdm-1028M-3300e18-rsl-0.01-bs-1024.safetensors",
    1476: "mdm-1476M-100e18.safetensors",
}


@dataclass(frozen=True)
class BackboneDefaults:
    model_name_or_path: str
    tokenizer_name_or_path: str
    mask_token_id: int
    data_subdir: str
    model_loader: str = "hf"
    checkpoint_path: Optional[str] = None
    smdm_size: Optional[int] = None

    def train_data_paths(self, root: str = DEFAULT_TOKENIZED_DATA_ROOT) -> List[str]:
        return [
            f"{root}/{self.data_subdir}/owt/train",
            f"{root}/{self.data_subdir}/proof_pile2/train",
        ]

    def eval_data_paths(self, root: str = DEFAULT_TOKENIZED_DATA_ROOT) -> List[str]:
        return [
            f"{root}/{self.data_subdir}/owt/validation",
            f"{root}/{self.data_subdir}/proof_pile2/validation",
        ]


BACKBONE_DEFAULTS: Dict[str, BackboneDefaults] = {
    "llada": BackboneDefaults(
        model_name_or_path="GSAI-ML/LLaDA-8B-Base",
        tokenizer_name_or_path="GSAI-ML/LLaDA-8B-Base",
        mask_token_id=126336,
        data_subdir="llada",
    ),
    "smdm": BackboneDefaults(
        model_name_or_path="nieshen/SMDM",
        tokenizer_name_or_path="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
        mask_token_id=32000,
        data_subdir="smdm",
        model_loader="smdm_litgpt",
        smdm_size=DEFAULT_SMDM_SIZE,
    ),
}


def get_backbone_defaults(backbone_type: str) -> Optional[BackboneDefaults]:
    return BACKBONE_DEFAULTS.get(str(backbone_type or "auto").lower())


def normalize_smdm_size(size) -> int:
    """Normalize SMDM size values like 551, "551", or "551M"."""
    if size in (None, ""):
        return DEFAULT_SMDM_SIZE
    if isinstance(size, str):
        size = size.strip()
        if size.endswith("M") or size.endswith("m"):
            size = size[:-1]
    size = int(size)
    if size not in SMDM_CHECKPOINT_FILENAMES:
        supported = ", ".join(f"{key}M" for key in sorted(SMDM_CHECKPOINT_FILENAMES))
        raise ValueError(f"Unsupported SMDM size {size}M. Supported sizes: {supported}.")
    return size


def smdm_checkpoint_path_for_size(size, repo_or_dir: str = "nieshen/SMDM") -> str:
    size = normalize_smdm_size(size)
    filename = SMDM_CHECKPOINT_FILENAMES[size]
    return f"{repo_or_dir.rstrip('/')}/mdm_safetensors/{filename}"


def resolve_smdm_size(config) -> int:
    return normalize_smdm_size(getattr(config, "smdm_size", None))


def fill_mlfm_defaults(config):
    """Fill missing model/tokenizer/mask/data fields from known backbone defaults."""
    backbone_type = str(getattr(config, "backbone_type", "auto") or "auto").lower()
    defaults = get_backbone_defaults(backbone_type)
    if defaults is None:
        return config
    data_root = getattr(config, "tokenized_data_root", DEFAULT_TOKENIZED_DATA_ROOT) or DEFAULT_TOKENIZED_DATA_ROOT
    if not getattr(config, "backbone_model_name_or_path", None):
        config.backbone_model_name_or_path = defaults.model_name_or_path
    if not getattr(config, "tokenizer_name_or_path", None):
        config.tokenizer_name_or_path = defaults.tokenizer_name_or_path
    if getattr(config, "mask_token_id", None) is None:
        config.mask_token_id = defaults.mask_token_id
    if not getattr(config, "model_loader", None):
        config.model_loader = defaults.model_loader
    if backbone_type == "smdm":
        size = resolve_smdm_size(config)
        config.smdm_size = size
        for attr in ("output_dir", "wandb_run_name"):
            value = getattr(config, attr, None)
            if isinstance(value, str) and "{smdm_size}" in value:
                setattr(config, attr, value.format(smdm_size=size))
        checkpoint_path = getattr(config, "smdm_checkpoint_path", None)
        if not checkpoint_path or checkpoint_path == "auto":
            repo_or_dir = getattr(config, "backbone_model_name_or_path", None) or defaults.model_name_or_path
            config.smdm_checkpoint_path = smdm_checkpoint_path_for_size(size, repo_or_dir=repo_or_dir)
    elif not getattr(config, "smdm_checkpoint_path", None) and defaults.checkpoint_path is not None:
        config.smdm_checkpoint_path = defaults.checkpoint_path
    if not getattr(config, "data_paths", None) and not getattr(config, "data_path", None):
        config.data_paths = defaults.train_data_paths(data_root)
    if not getattr(config, "eval_data_paths", None) and not getattr(config, "eval_data_path", None):
        config.eval_data_paths = defaults.eval_data_paths(data_root)
    return config
