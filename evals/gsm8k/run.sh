#!/usr/bin/env bash
# Run GSM8K on one GPU.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
CONFIG_PATH="${CONFIG_PATH:-$REPO_ROOT/evals/gsm8k/config.yml}"
PROJECTDIR="${PROJECTDIR:-$REPO_ROOT}"
export PROJECTDIR

PROJECT_TMP="${PROJECT_TMP:-$PROJECTDIR/tmp/evals_gsm8k}"
mkdir -p "$PROJECT_TMP" "$PROJECT_TMP/wandb" "$PROJECT_TMP/matplotlib" "$PROJECT_TMP/xdg_cache"

export TMPDIR="$PROJECT_TMP"
export WANDB_DIR="$PROJECT_TMP/wandb"
export HF_HOME="${HF_HOME:-$PROJECTDIR/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export MPLCONFIGDIR="$PROJECT_TMP/matplotlib"
export XDG_CACHE_HOME="$PROJECT_TMP/xdg_cache"
export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
  else
    echo "Could not find python. Set PYTHON_BIN." >&2
    exit 1
  fi
fi

SMDM_CODE_DIR="${SMDM_CODE_DIR:-$REPO_ROOT/external/SMDM}"
CMD=("$PYTHON_BIN" -s "$REPO_ROOT/evals/gsm8k/evaluate_gsm8k.py" --config "$CONFIG_PATH")

[[ -n "${TRAINING_CONFIG:-}" ]] && CMD+=(--training_config "$TRAINING_CONFIG")
[[ -n "${CHECKPOINT:-}" ]] && CMD+=(--checkpoint "$CHECKPOINT")
[[ -n "${OUTPUT_JSONL:-}" ]] && CMD+=(--output_jsonl "$OUTPUT_JSONL")
[[ -n "${DATASET_PATH:-}" ]] && CMD+=(--dataset_path "$DATASET_PATH")
[[ -n "${MAX_EXAMPLES:-}" ]] && CMD+=(--max_examples "$MAX_EXAMPLES")
[[ -n "${SUBSET_SIZE:-}" ]] && CMD+=(--subset_size "$SUBSET_SIZE")
[[ -n "${SUBSET_PERCENTAGE:-}" ]] && CMD+=(--subset_percentage "$SUBSET_PERCENTAGE")
[[ -n "${SUBSET_SEED:-}" ]] && CMD+=(--subset_seed "$SUBSET_SEED")
[[ -n "${SUBSET_INDICES:-}" ]] && CMD+=(--subset_indices "$SUBSET_INDICES")
[[ -n "${SUBSET_INDICES_FILE:-}" ]] && CMD+=(--subset_indices_file "$SUBSET_INDICES_FILE")
[[ -n "${BATCH_SIZE:-}" ]] && CMD+=(--batch_size "$BATCH_SIZE")
[[ -n "${NUM_FEWSHOTS:-}" ]] && CMD+=(--num_fewshots "$NUM_FEWSHOTS")
CMD+=(--config_override "smdm_code_path=$SMDM_CODE_DIR")
CMD+=("$@")

printf 'Command:'
printf ' %q' "${CMD[@]}"
printf '\n'

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  exit 0
fi

cd "$REPO_ROOT"
"${CMD[@]}"
