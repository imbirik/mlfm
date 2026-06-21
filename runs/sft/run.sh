#!/usr/bin/env bash
# Plain bash launcher for the final SFT run.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
CONFIG_PATH="${CONFIG_PATH:-$REPO_ROOT/runs/sft/config.yml}"
PROJECTDIR="${PROJECTDIR:-$REPO_ROOT}"
export PROJECTDIR

PROJECT_TMP="${PROJECT_TMP:-$PROJECTDIR/tmp}"
mkdir -p "$PROJECT_TMP" "$PROJECT_TMP/wandb" "$PROJECT_TMP/matplotlib" "$PROJECT_TMP/xdg_cache"

export TMPDIR="$PROJECT_TMP"
export WANDB_DIR="$PROJECT_TMP/wandb"
export HF_HOME="${HF_HOME:-$PROJECTDIR/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export MPLCONFIGDIR="$PROJECT_TMP/matplotlib"
export XDG_CACHE_HOME="$PROJECT_TMP/xdg_cache"
export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

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

infer_nproc_per_node() {
  if [[ -n "${NPROC_PER_NODE:-}" ]]; then
    echo "$NPROC_PER_NODE"
  elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" && "$CUDA_VISIBLE_DEVICES" != "NoDevFiles" ]]; then
    awk -F, '{print NF}' <<<"$CUDA_VISIBLE_DEVICES"
  else
    echo 1
  fi
}

NPROC_PER_NODE="$(infer_nproc_per_node)"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"
export MASTER_ADDR MASTER_PORT
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$OMP_NUM_THREADS}"

DATA_ROOT="${DATA_ROOT:-$PROJECTDIR/datasets/smdm}"
SFT_DATA_ROOT="${SFT_DATA_ROOT:-$PROJECTDIR/datasets/sft}"
SMDM_CODE_DIR="${SMDM_CODE_DIR:-$REPO_ROOT/external/SMDM}"
DEFAULT_OUTPUT_DIR="$PROJECTDIR/runs/sft/output_smdm_{smdm_size}M_sft"
OUTPUT_DIR="${OUTPUT_DIR:-$DEFAULT_OUTPUT_DIR}"
USE_WANDB="${USE_WANDB:-false}"
TRAINING_STAGE="${TRAINING_STAGE:-sft}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"

CMD=(
  "$PYTHON_BIN" -s -m torch.distributed.run
  --nnodes="$NNODES"
  --nproc_per_node="$NPROC_PER_NODE"
  --node_rank="$NODE_RANK"
  --master_addr="$MASTER_ADDR"
  --master_port="$MASTER_PORT"
  "$REPO_ROOT/src/train.py"
  --config "$CONFIG_PATH"
  --config_override "smdm_code_path=$SMDM_CODE_DIR"
  --config_override "data_paths=[$DATA_ROOT/owt/train,$DATA_ROOT/proof_pile2/train]"
  --config_override "eval_data_paths=[$DATA_ROOT/owt/validation,$DATA_ROOT/proof_pile2/validation]"
  --config_override "sft_general_data_paths=[$SFT_DATA_ROOT/sharegpt_first_round/train]"
  --config_override "sft_math_data_paths=[$SFT_DATA_ROOT/numinamath_cot/train]"
  --config_override "sft_math_gsm8k_data_paths=[]"
  --config_override "sft_math_gsm8k_nl_data_paths=[$SFT_DATA_ROOT/gsm8k_aug_nl/train]"
  --config_override "sft_math_metamathqa_data_paths=[$SFT_DATA_ROOT/metamathqa/train]"
  --config_override "sft_code_data_paths=[$SFT_DATA_ROOT/opencodeinstruct_short/train]"
  --config_override "training_stage=$TRAINING_STAGE"
  --config_override "gsm8k_eval_path=data/gsm8k/test.jsonl"
  --config_override "output_dir=$OUTPUT_DIR"
  --config_override "use_wandb=$USE_WANDB"
)
if [[ -n "$RESUME_CHECKPOINT" ]]; then
  CMD+=(--config_override "resume=$RESUME_CHECKPOINT")
fi
CMD+=("$@")

printf 'Command:'
printf ' %q' "${CMD[@]}"
printf '\n'

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  exit 0
fi

cd "$REPO_ROOT"
"${CMD[@]}"
