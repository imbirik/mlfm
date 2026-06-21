#!/usr/bin/env python
"""Train the camera-ready MLFM/SMDM runs."""

from __future__ import annotations

import argparse
import logging
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from configs.config import apply_config_overrides, load_config_from_yaml
from mlfm.runner import run_mlfm_training


logging.basicConfig(
    format="%(levelname)s - %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO,
    force=True,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train MLFM on an SMDM backbone.")
    parser.add_argument("--config", required=True, help="Path to a YAML training config.")
    parser.add_argument(
        "--config_override",
        action="append",
        default=[],
        help="Override config values as field=value. Can be repeated.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config_from_yaml(args.config)
    config = apply_config_overrides(config, args.config_override)
    objective = getattr(config, "training_objective", None)
    if objective != "mlfm":
        raise ValueError(
            "camera_ready/src/train.py only supports training_objective="
            f"'mlfm', got {objective!r}."
        )
    run_mlfm_training(config)


if __name__ == "__main__":
    main()
