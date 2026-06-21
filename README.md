# MLFM: Masked Language Flow Models

<!-- [arXiv](TODO) -->

[Blog post](https://kiaashour.github.io/blog/masked-language-flow-models/)

Official implementation of Masked Language Flow Models.

## Environment Setup

Use Python 3.11 and install dependencies with pip:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu126
```

FlashAttention is intentionally not included in `requirements.txt`; install it separately after PyTorch is installed:

```bash
MAX_JOBS=4 python -m pip install flash-attn==2.8.3 --no-build-isolation
```

If your cluster provides a prebuilt FlashAttention wheel or module, use that instead. The SMDM compatibility loader has inference fallbacks for some optional fused kernels, but training should use the real CUDA kernels when available.

## Checkpoints

Please find our 1028B SFT MLFM model [here](https://drive.google.com/drive/folders/1G1eI6DGMploU8rrt1pNhzbB-umH18Je3?usp=sharing).

The run configs are:

- `runs/train/config.yml`: final pretraining/adaptation config.
- `runs/sft/config.yml`: final SFT config.

## GSM8K

Run a released SFT checkpoint with the camera-ready SDE sampler and online token promotion:

```bash
CHECKPOINT=<CHECKPOINT> \
TRAINING_CONFIG=runs/sft/config.yml \
DATASET_PATH=data/gsm8k/test.jsonl \
bash evals/gsm8k/run.sh
```

## MT-Bench

Generate MT-Bench answers from a released SFT checkpoint:

```bash
CHECKPOINT=<CHECKPOINT> \
TRAINING_CONFIG=runs/sft/config.yml \
MODEL_ID=mlfm_sft \
bash evals/mt_bench/run.sh
```

For MT-Bench scoring, generate answers first, then run the FastChat judge with your own OpenAI API key and FastChat checkout.

## Training

The scripts below are plain bash launchers. They use `torch.distributed.run` and can run single-node or multi-node when `NNODES`, `NODE_RANK`, `MASTER_ADDR`, and `MASTER_PORT` are set.

```bash
bash runs/train/run.sh
```

```bash
RESUME_CHECKPOINT=<PRETRAIN_CHECKPOINT> bash runs/sft/run.sh
```

Useful environment overrides:

- `PROJECTDIR`: run/data root. Defaults to the repository root.
- `DATA_ROOT`: pretokenized packed data root.
- `SFT_DATA_ROOT`: pretokenized SFT data root.
- `SMDM_CODE_DIR`: path to `external/SMDM`.
- `OUTPUT_DIR`: output directory for checkpoints and metrics.
- `NPROC_PER_NODE`, `NNODES`, `NODE_RANK`, `MASTER_ADDR`, `MASTER_PORT`: distributed launch controls.
- `USE_WANDB=true`: enable WandB logging.

## Data Layout

The included benchmark data lives under `data/`:

- `data/gsm8k/test.jsonl`
- `data/mt_bench/question.jsonl`

Large pretraining/SFT datasets and checkpoints are not included.
