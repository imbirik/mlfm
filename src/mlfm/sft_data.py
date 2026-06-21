"""Prompt/response SFT data helpers for MLFM."""

from __future__ import annotations

import os
from typing import Dict, Iterator, List, Optional, Sequence

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from mlfm.data_mix import BalancedMixtureDataLoader, _allocate_batch_sizes
from utils.data_utils import load_dataset_split, pad_and_truncate


SFT_SOURCE_TYPE_IDS = {
    "general": 0,
    "math": 1,
    "code": 2,
}


def _dist_info():
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size(), dist.get_rank()
    return 1, 0


def _normalize_paths(paths):
    if paths is None:
        return []
    if isinstance(paths, str):
        return [item.strip() for item in paths.split(",") if item.strip()]
    return list(paths)


def _as_weight_dict(value, default: Dict[str, float]) -> Dict[str, float]:
    if value is None:
        return dict(default)
    if isinstance(value, str):
        items = {}
        for part in value.split(","):
            if not part.strip():
                continue
            key, weight = part.split(":", 1)
            items[key.strip()] = float(weight)
        return items
    return {str(key): float(weight) for key, weight in dict(value).items()}


def _add_entries(entries, paths, source_type: str, total_weight: float):
    paths = _normalize_paths(paths)
    if not paths or total_weight <= 0.0:
        return
    per_path = float(total_weight) / float(len(paths))
    for path in paths:
        entries.append(
            {
                "path": path,
                "weight": per_path,
                "source_type": source_type,
            }
        )


def sft_source_entries_from_config(config, train: bool = True) -> List[Dict]:
    """Return concrete SFT dataset entries with normalized source weights."""
    if not train:
        eval_paths = _normalize_paths(getattr(config, "sft_eval_data_paths", None))
        return [{"path": path, "weight": 1.0 / len(eval_paths), "source_type": "general"} for path in eval_paths] if eval_paths else []

    source_weights = _as_weight_dict(getattr(config, "sft_source_weights", None), {"general": 0.35, "math": 0.45, "code": 0.2})
    math_weights = _as_weight_dict(
        getattr(config, "sft_math_source_weights", None),
        {"gsm8k_aug": 0.0, "gsm8k_aug_nl": 0.0, "metamathqa": 0.5, "math_reasoning": 0.5},
    )

    entries = []
    _add_entries(entries, getattr(config, "sft_general_data_paths", None), "general", source_weights.get("general", 0.0))

    math_total = source_weights.get("math", 0.0)
    _add_entries(
        entries,
        getattr(config, "sft_math_gsm8k_data_paths", None),
        "math",
        math_total * math_weights.get("gsm8k_aug", 0.0),
    )
    _add_entries(
        entries,
        getattr(config, "sft_math_gsm8k_nl_data_paths", None),
        "math",
        math_total * math_weights.get("gsm8k_aug_nl", 0.0),
    )
    _add_entries(
        entries,
        getattr(config, "sft_math_metamathqa_data_paths", None),
        "math",
        math_total * math_weights.get("metamathqa", 0.0),
    )
    generic_math = _normalize_paths(getattr(config, "sft_math_data_paths", None))
    if generic_math:
        generic_weight = math_weights.get("math_reasoning", math_weights.get("generic", math_weights.get("other")))
        if generic_weight is None:
            used_subweights = 0.0
            if _normalize_paths(getattr(config, "sft_math_gsm8k_data_paths", None)):
                used_subweights += math_weights.get("gsm8k_aug", 0.0)
            if _normalize_paths(getattr(config, "sft_math_gsm8k_nl_data_paths", None)):
                used_subweights += math_weights.get("gsm8k_aug_nl", 0.0)
            if _normalize_paths(getattr(config, "sft_math_metamathqa_data_paths", None)):
                used_subweights += math_weights.get("metamathqa", 0.0)
            generic_weight = max(0.0, 1.0 - used_subweights)
        _add_entries(entries, generic_math, "math", math_total * float(generic_weight))

    _add_entries(entries, getattr(config, "sft_code_data_paths", None), "code", source_weights.get("code", 0.0))

    total = sum(float(entry["weight"]) for entry in entries)
    if total > 0.0:
        for entry in entries:
            entry["weight"] = float(entry["weight"]) / total
    return entries


def has_sft_data(config, train: bool = True) -> bool:
    return bool(sft_source_entries_from_config(config, train=train))


def make_sft_collate_fn(max_length: int, sft_pad_token_id: int, source_id: int, source_type: str):
    """Collate prompt/response SFT rows.

    SFT attention masks intentionally stay all-ones after EOS padding because
    response-side EOS padding is part of response length modeling.
    """

    source_type_id = int(SFT_SOURCE_TYPE_IDS.get(str(source_type), -1))

    def collate(batch_list):
        ids_list = [np.asarray(item["input_ids"], dtype=np.int64) for item in batch_list]
        ids, _ = pad_and_truncate(ids_list, max_length, sft_pad_token_id)
        prompt_lengths = []
        true_lengths = []
        for item, row_ids in zip(batch_list, ids_list):
            prompt_length = item.get("prompt_length", item.get("prompt_lengths", item.get("input_length", 0)))
            prompt_length = int(np.asarray(prompt_length).item())
            prompt_lengths.append(max(0, min(prompt_length, max_length - 1)))

            true_length = item.get("true_length", item.get("length", min(len(row_ids), max_length)))
            true_length = int(np.asarray(true_length).item())
            true_lengths.append(max(1, min(true_length, max_length)))

        prompt_lengths = torch.tensor(prompt_lengths, dtype=torch.long)
        true_lengths = torch.tensor(true_lengths, dtype=torch.long)
        positions = torch.arange(max_length, dtype=torch.long).unsqueeze(0)
        response_mask = positions >= prompt_lengths.unsqueeze(1)
        return {
            "input_ids": torch.from_numpy(ids.astype(np.int64)),
            "attention_mask": torch.ones((len(batch_list), max_length), dtype=torch.long),
            "prompt_lengths": prompt_lengths,
            "true_lengths": true_lengths,
            "sft_response_mask": response_mask,
            "source_id": torch.full((len(batch_list),), int(source_id), dtype=torch.long),
            "sft_source_type": torch.full((len(batch_list),), source_type_id, dtype=torch.long),
            "is_sft": torch.ones((len(batch_list),), dtype=torch.long),
        }

    return collate


def make_sft_source_dataloader(
    dataset,
    batch_size: int,
    max_length: int,
    sft_pad_token_id: int,
    source_id: int,
    source_type: str,
    shuffle: bool,
    num_workers: int,
    drop_last: bool,
    distributed: bool,
):
    common = dict(
        batch_size=batch_size,
        collate_fn=make_sft_collate_fn(max_length, sft_pad_token_id, source_id, source_type),
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )
    if distributed and dist.is_available() and dist.is_initialized():
        world_size, rank = _dist_info()
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=shuffle, drop_last=drop_last)
        return DataLoader(dataset, sampler=sampler, **common)
    return DataLoader(dataset, shuffle=shuffle, **common)


def get_sft_dataloader(
    config,
    batch_size: int,
    max_length: int,
    sft_pad_token_id: int,
    num_workers: int = 0,
    shuffle: bool = True,
    drop_last: bool = True,
    distributed: bool = True,
    train: bool = True,
):
    entries = sft_source_entries_from_config(config, train=train)
    if not entries:
        raise ValueError("No SFT data paths configured.")

    batching_mode = str(getattr(config, "sft_batching_mode", "mixed_concat") or "mixed_concat").lower()
    source_schedule_modes = {"source_scheduled", "source_schedule", "whole_source", "scheduled"}
    mixed_concat_modes = {"mixed_concat", "concat", "balanced_concat", "mixed"}
    if batching_mode in source_schedule_modes and train:
        source_batch_sizes = [int(batch_size)] * len(entries)
    elif batching_mode in mixed_concat_modes:
        source_batch_sizes = _allocate_batch_sizes(batch_size, [entry["weight"] for entry in entries])
    else:
        raise ValueError(
            f"Unknown sft_batching_mode={batching_mode!r}. "
            "Expected 'mixed_concat' or 'source_schedule'."
        )
    loaders = []
    source_names = []
    for source_id, (entry, source_batch_size) in enumerate(zip(entries, source_batch_sizes)):
        path = str(entry["path"])
        if path.startswith((".", "/")) and not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing pretokenized SFT dataset for source '{entry['source_type']}': {path}. "
                "Run runs/tokenize_smdm_sft/submit.sh, or set this source weight to 0 and remove the path."
            )
        dataset = load_dataset_split(path)
        source_name = f"{entry['source_type']}:{os.path.basename(os.path.normpath(entry['path'])) or source_id}"
        source_names.append(source_name)
        loaders.append(
            make_sft_source_dataloader(
                dataset,
                batch_size=source_batch_size,
                max_length=max_length,
                sft_pad_token_id=sft_pad_token_id,
                source_id=source_id,
                source_type=entry["source_type"],
                shuffle=shuffle,
                num_workers=num_workers,
                drop_last=drop_last,
                distributed=distributed,
            )
        )
    if batching_mode in source_schedule_modes and train:
        return SourceScheduledSFTDataLoader(
            loaders,
            weights=[entry["weight"] for entry in entries],
            source_names=source_names,
            schedule_slots=int(getattr(config, "sft_source_schedule_slots", 50) or 50),
        )
    return BalancedMixtureDataLoader(loaders, source_names=source_names)


def _weighted_schedule(weights: Sequence[float], slots: int = 100) -> List[int]:
    weights = np.asarray(weights, dtype=np.float64)
    weights = weights / weights.sum()
    raw = weights * int(slots)
    counts = np.floor(raw).astype(np.int64)
    remainder = int(slots) - int(counts.sum())
    order = np.argsort(-(raw - counts))
    for idx in order[:remainder]:
        counts[idx] += 1
    schedule = []
    cursors = np.zeros_like(counts)
    for _ in range(int(slots)):
        scores = (cursors + 1.0) / np.maximum(counts, 1)
        scores[counts <= 0] = np.inf
        idx = int(np.argmin(scores))
        schedule.append(idx)
        cursors[idx] += 1
    return schedule


class SourceScheduledSFTDataLoader:
    """Yield one SFT source per microbatch using a deterministic weighted schedule.

    All ranks construct the same source schedule locally. Individual examples are
    still sharded by each source dataloader's DistributedSampler.
    """

    def __init__(
        self,
        loaders: Sequence[DataLoader],
        weights: Sequence[float],
        source_names: Optional[Sequence[str]] = None,
        schedule_slots: int = 50,
    ):
        if not loaders:
            raise ValueError("At least one source dataloader is required.")
        if len(loaders) != len(weights):
            raise ValueError("loaders and weights must have the same length.")
        self.loaders = list(loaders)
        self.weights = [float(weight) for weight in weights]
        self.source_names = list(source_names or [f"source_{idx}" for idx in range(len(loaders))])
        self.schedule = _weighted_schedule(self.weights, slots=int(schedule_slots))

    def __len__(self):
        return max(1, max(len(loader) for loader in self.loaders))

    def set_epoch(self, epoch: int):
        for loader in self.loaders:
            sampler = getattr(loader, "sampler", None)
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        iterators = [iter(loader) for loader in self.loaders]
        for idx in range(len(self)):
            source_idx = int(self.schedule[idx % len(self.schedule)])
            try:
                batch = next(iterators[source_idx])
            except StopIteration:
                iterators[source_idx] = iter(self.loaders[source_idx])
                batch = next(iterators[source_idx])
            batch["sft_scheduled_source_id"] = torch.full(
                (int(batch["input_ids"].shape[0]),),
                source_idx,
                dtype=torch.long,
            )
            batch["sft_num_sources"] = torch.as_tensor(len(self.loaders), dtype=torch.long)
            batch["sft_scheduled_source_name"] = self.source_names[source_idx]
            yield batch


class ObjectiveMixtureDataLoader:
    """Yield whole microbatches from multiple objective loaders by fixed weights."""

    def __init__(self, loaders: Sequence, weights: Sequence[float], names: Optional[Sequence[str]] = None):
        if not loaders:
            raise ValueError("At least one loader is required.")
        if len(loaders) != len(weights):
            raise ValueError("loaders and weights must have the same length.")
        self.loaders = list(loaders)
        self.weights = [float(weight) for weight in weights]
        self.names = list(names or [f"objective_{idx}" for idx in range(len(loaders))])
        self.schedule = _weighted_schedule(self.weights)

    def __len__(self):
        return max(1, max(len(loader) for loader in self.loaders))

    def set_epoch(self, epoch: int):
        for loader in self.loaders:
            if hasattr(loader, "set_epoch"):
                loader.set_epoch(epoch)

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        iterators = [iter(loader) for loader in self.loaders]
        for idx in range(len(self)):
            loader_idx = self.schedule[idx % len(self.schedule)]
            try:
                yield next(iterators[loader_idx])
            except StopIteration:
                iterators[loader_idx] = iter(self.loaders[loader_idx])
                yield next(iterators[loader_idx])
