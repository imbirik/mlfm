#!/usr/bin/env bash
# Generate MT-Bench first-turn answers on one GPU.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
CONFIG_PATH="${CONFIG_PATH:-$REPO_ROOT/evals/mt_bench/config.yml}"
PROJECTDIR="${PROJECTDIR:-$REPO_ROOT}"
export PROJECTDIR

PROJECT_TMP="${PROJECT_TMP:-$PROJECTDIR/tmp/evals_mt_bench}"
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
CMD=("$PYTHON_BIN" -s "$REPO_ROOT/evals/mt_bench/evaluate_mt_bench.py" --config "$CONFIG_PATH")

[[ -n "${TRAINING_CONFIG:-}" ]] && CMD+=(--training_config "$TRAINING_CONFIG")
[[ -n "${CHECKPOINT:-}" ]] && CMD+=(--checkpoint "$CHECKPOINT")
[[ -n "${MODEL_ID:-}" ]] && CMD+=(--model_id "$MODEL_ID")
[[ -n "${ANSWER_FILE:-}" ]] && CMD+=(--answer_file "$ANSWER_FILE")
[[ -n "${DETAILS_JSONL:-}" ]] && CMD+=(--details_jsonl "$DETAILS_JSONL")
[[ -n "${QUESTION_FILE:-}" ]] && CMD+=(--question_file "$QUESTION_FILE")
[[ -n "${MAX_EXAMPLES:-}" ]] && CMD+=(--max_examples "$MAX_EXAMPLES")
[[ -n "${SUBSET_SIZE:-}" ]] && CMD+=(--subset_size "$SUBSET_SIZE")
[[ -n "${SUBSET_PERCENTAGE:-}" ]] && CMD+=(--subset_percentage "$SUBSET_PERCENTAGE")
[[ -n "${SUBSET_SEED:-}" ]] && CMD+=(--subset_seed "$SUBSET_SEED")
[[ -n "${SUBSET_INDICES:-}" ]] && CMD+=(--subset_indices "$SUBSET_INDICES")
[[ -n "${SUBSET_QUESTION_IDS:-}" ]] && CMD+=(--subset_question_ids "$SUBSET_QUESTION_IDS")
[[ -n "${BATCH_SIZE:-}" ]] && CMD+=(--batch_size "$BATCH_SIZE")
[[ -n "${JUDGMENT_JSONL:-}" ]] && CMD+=(--judgment_jsonl "$JUDGMENT_JSONL")
[[ "${SCORE_ONLY:-0}" == "1" ]] && CMD+=(--score_only)
[[ "${RUN_FASTCHAT_JUDGE:-0}" == "1" ]] && CMD+=(--run_fastchat_judge)
[[ -n "${FASTCHAT_LLM_JUDGE_DIR:-}" ]] && CMD+=(--fastchat_llm_judge_dir "$FASTCHAT_LLM_JUDGE_DIR")
[[ -n "${FASTCHAT_BENCH_NAME:-}" ]] && CMD+=(--fastchat_bench_name "$FASTCHAT_BENCH_NAME")
[[ -n "${FASTCHAT_PYTHON:-}" ]] && CMD+=(--fastchat_python "$FASTCHAT_PYTHON")
[[ -n "${JUDGE_MODEL:-}" ]] && CMD+=(--judge_model "$JUDGE_MODEL")
[[ -n "${JUDGE_PARALLEL:-}" ]] && CMD+=(--judge_parallel "$JUDGE_PARALLEL")
[[ "${DOWNLOAD_QUESTIONS:-0}" == "1" ]] && CMD+=(--download_questions)
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
