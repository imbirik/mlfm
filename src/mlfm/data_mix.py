"""Packed dataset loading and 50/50 source mixing for MLFM."""

from __future__ import annotations

import os
from typing import Iterable, List, Optional, Sequence

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from utils.data_utils import load_dataset_split, pad_and_truncate


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


def _allocate_batch_sizes(batch_size: int, weights: Sequence[float]) -> List[int]:
    weights = np.asarray(weights, dtype=np.float64)
    weights = weights / weights.sum()
    raw = weights * batch_size
    sizes = np.floor(raw).astype(np.int64)
    remainder = batch_size - int(sizes.sum())
    order = np.argsort(-(raw - sizes))
    for idx in order[:remainder]:
        sizes[idx] += 1
    if batch_size >= len(sizes):
        for idx, weight in enumerate(weights):
            if weight > 0 and sizes[idx] == 0:
                donor = int(np.argmax(sizes))
                sizes[donor] -= 1
                sizes[idx] = 1
    return [int(size) for size in sizes]


def make_packed_collate_fn(max_length: int, pad_token_id: int, source_id: int):
    """Collate fixed-block token datasets, padding/truncating if needed."""

    def collate(batch_list):
        ids_list = [np.asarray(item["input_ids"], dtype=np.int64) for item in batch_list]
        ids, lengths = pad_and_truncate(ids_list, max_length, pad_token_id)
        if "attention_mask" in batch_list[0]:
            attn_list = [np.asarray(item["attention_mask"], dtype=np.int64) for item in batch_list]
            attn, _ = pad_and_truncate(attn_list, max_length, 0)
        else:
            pos = np.arange(max_length)[None, :]
            attn = (pos < lengths[:, None]).astype(np.int64)
        return {
            "input_ids": torch.from_numpy(ids.astype(np.int64)),
            "attention_mask": torch.from_numpy(attn.astype(np.int64)),
            "source_id": torch.full((len(batch_list),), int(source_id), dtype=torch.long),
        }

    return collate


def make_source_dataloader(
    dataset,
    batch_size: int,
    max_length: int,
    pad_token_id: int,
    source_id: int,
    shuffle: bool,
    num_workers: int,
    drop_last: bool,
    distributed: bool,
):
    common = dict(
        batch_size=batch_size,
        collate_fn=make_packed_collate_fn(max_length, pad_token_id, source_id),
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


class BalancedMixtureDataLoader:
    """Yield batches assembled from fixed source proportions."""

    def __init__(self, loaders: Sequence[DataLoader], source_names: Optional[Sequence[str]] = None):
        if not loaders:
            raise ValueError("At least one source dataloader is required.")
        self.loaders = list(loaders)
        self.source_names = list(source_names or [f"source_{idx}" for idx in range(len(loaders))])

    def __len__(self):
        return min(len(loader) for loader in self.loaders)

    def set_epoch(self, epoch: int):
        for loader in self.loaders:
            sampler = getattr(loader, "sampler", None)
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)

    def __iter__(self):
        iterators = [iter(loader) for loader in self.loaders]
        for _ in range(len(self)):
            batches = []
            for idx, iterator in enumerate(iterators):
                try:
                    batch = next(iterator)
                except StopIteration:
                    iterators[idx] = iter(self.loaders[idx])
                    batch = next(iterators[idx])
                batches.append(batch)
            yield self._merge_batches(batches)

    @staticmethod
    def _merge_batches(batches):
        keys = batches[0].keys()
        merged = {}
        for key in keys:
            values = [batch[key] for batch in batches if key in batch]
            if torch.is_tensor(values[0]):
                merged[key] = torch.cat(values, dim=0)
            else:
                merged[key] = sum(values, [])
        return merged


def get_mlfm_dataloader(
    data_paths: Sequence[str],
    mix_weights: Optional[Sequence[float]],
    batch_size: int,
    max_length: int,
    pad_token_id: int,
    num_workers: int = 0,
    shuffle: bool = True,
    drop_last: bool = True,
    distributed: bool = True,
):
    paths = _normalize_paths(data_paths)
    if not paths:
        raise ValueError("No data paths provided.")
    if mix_weights is None:
        mix_weights = [1.0 / len(paths)] * len(paths)
    if isinstance(mix_weights, str):
        mix_weights = [float(item.strip()) for item in mix_weights.split(",") if item.strip()]
    if len(mix_weights) != len(paths):
        raise ValueError("`data_mix_weights` must have the same length as `data_paths`.")
    source_batch_sizes = _allocate_batch_sizes(batch_size, mix_weights)
    loaders = []
    source_names = []
    for source_id, (path, source_batch_size) in enumerate(zip(paths, source_batch_sizes)):
        dataset = load_dataset_split(path)
        source_names.append(os.path.basename(os.path.normpath(path)) or f"source_{source_id}")
        loaders.append(
            make_source_dataloader(
                dataset,
                batch_size=source_batch_size,
                max_length=max_length,
                pad_token_id=pad_token_id,
                source_id=source_id,
                shuffle=shuffle,
                num_workers=num_workers,
                drop_last=drop_last,
                distributed=distributed,
            )
        )
    return BalancedMixtureDataLoader(loaders, source_names=source_names)
