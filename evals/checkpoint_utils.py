"""Shared checkpoint path helpers for standalone eval scripts."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def expand_config_template(value: str, config: Any) -> str:
    """Expand simple config placeholders such as ``{smdm_size}``."""

    try:
        return value.format(**vars(config))
    except Exception:
        return value


def unique_paths(paths: list[Path]) -> list[Path]:
    seen = set()
    result = []
    for path in paths:
        key = os.path.normpath(str(path))
        if key in seen:
            continue
        seen.add(key)
        result.append(path)
    return result


def resolve_output_dir_candidates(
    output_dir: str,
    *,
    repo_root: str,
    training_config_path: str,
    repo_path_fn,
) -> list[Path]:
    """Return candidate output dirs for a possibly relative training output_dir.

    Training jobs are normally submitted from ``PROJECTDIR`` while eval scripts
    live under ``PROJECTDIR/llada_flm``. For configs with ``./runs/...`` output
    dirs, prefer ``PROJECTDIR`` so checkpoints outside the repo are found.
    """

    expanded = os.path.expandvars(os.path.expanduser(output_dir))
    path = Path(expanded)
    if path.is_absolute():
        return [path]

    candidates: list[Path] = []
    project_dir = os.environ.get("PROJECTDIR")
    if project_dir:
        candidates.append(Path(project_dir) / path)

    candidates.append(Path.cwd() / path)
    candidates.append(Path(repo_root).parent / path)
    candidates.append(Path(repo_root) / path)

    resolved_config_path = Path(repo_path_fn(training_config_path) or training_config_path)
    if resolved_config_path.exists():
        candidates.append(resolved_config_path.parent / path)
    return unique_paths(candidates)


def infer_checkpoint_from_training_config(
    config: Any,
    training_config_path: str,
    *,
    repo_root: str,
    repo_path_fn,
) -> str:
    """Infer ``checkpoint_latest.pt`` from a training/run config output_dir."""

    output_dir = getattr(config, "output_dir", None)
    if not output_dir:
        raise ValueError(
            "No checkpoint was provided and the training config has no output_dir. "
            "Pass --checkpoint or set model.checkpoint."
        )
    output_dir = expand_config_template(str(output_dir), config)
    candidates = [
        directory / "checkpoint_latest.pt"
        for directory in resolve_output_dir_candidates(
            output_dir,
            repo_root=repo_root,
            training_config_path=training_config_path,
            repo_path_fn=repo_path_fn,
        )
    ]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    formatted = "\n  ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "No checkpoint was provided, and checkpoint_latest.pt was not found from "
        f"training config output_dir={output_dir!r}. Checked:\n  {formatted}\n"
        "Set PROJECTDIR to the run root or pass --checkpoint explicitly."
    )
