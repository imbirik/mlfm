#!/usr/bin/env python
"""Generate FastChat-compatible MT-Bench answers for SMDM/MLFM checkpoints.

MT-Bench is not an exact-match benchmark.  The model first writes answers to the
official question set, then a strong LLM judge scores those answers.  This
script owns the first step for this repository's masked-diffusion sampler and
emits the answer JSONL shape expected by FastChat's ``gen_judgment.py``.

The paper protocol implemented here is first-turn-only: each standard MT-Bench
question has two user turns, but only the first turn receives a generated model
answer.  Later answer turns are kept as empty strings so the output remains
compatible with the FastChat judge/result scripts.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import time
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import torch
import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC_ROOT = os.path.join(REPO_ROOT, "src")
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from configs.config import apply_config_overrides, load_config_from_yaml
from evals.checkpoint_utils import infer_checkpoint_from_training_config
from sampling import OnlineTokenPromotionSampler, SamplingModel, load_sampling_experiment_config
from mlfm.defaults import fill_mlfm_defaults


DEFAULT_QUESTION_URL = (
    "https://raw.githubusercontent.com/lm-sys/FastChat/main/"
    "fastchat/llm_judge/data/mt_bench/question.jsonl"
)

VICUNA_SYSTEM_MESSAGE = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)

DEFAULT_STOP_STRINGS = (
    "</s>",
    "\nUSER:",
    "\nASSISTANT:",
    "\nUser:",
    "\nAssistant:",
)


@dataclass
class MTBenchQuestion:
    """Normalized MT-Bench question row."""

    index: int
    question_id: Any
    category: str
    turns: list[str]
    raw: dict[str, Any]


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate MT-Bench answer generation with one SMDM/MLFM SDE sampler.")
    parser.add_argument("--config", default="evals/mt_bench/config.yml")
    parser.add_argument("--training_config", default=None, help="Training/model config, e.g. runs/sft/config.yml.")
    parser.add_argument("--checkpoint", default=None, help="Adapter checkpoint. Optional when sampling.adapter_mode=base.")
    parser.add_argument("--sampling_config", default=None, help="Optional YAML containing `sampling:` overrides.")
    parser.add_argument(
        "--question_file",
        "--dataset_path",
        dest="question_file",
        default=None,
        help="Local MT-Bench question JSONL/JSON file. Defaults to evaluation.question_file.",
    )
    parser.add_argument(
        "--answer_file",
        "--output_jsonl",
        dest="answer_file",
        default=None,
        help="FastChat-format answer JSONL path.",
    )
    parser.add_argument("--details_jsonl", default=None, help="Optional per-question generation details JSONL path.")
    parser.add_argument("--model_id", default=None, help="Model name written into FastChat answer rows.")
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--subset_size", type=int, default=None)
    parser.add_argument("--subset_percentage", type=float, default=None)
    parser.add_argument("--subset_seed", type=int, default=None)
    parser.add_argument("--subset_indices", default=None, help="Comma-separated original MT-Bench row indices.")
    parser.add_argument("--subset_question_ids", default=None, help="Comma-separated MT-Bench question IDs, e.g. 81,82.")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--judgment_jsonl", default=None, help="Optional FastChat judgment JSONL to summarize.")
    parser.add_argument("--score_only", action="store_true", help="Only summarize --judgment_jsonl; do not load a model.")
    parser.add_argument(
        "--run_fastchat_judge",
        action="store_true",
        help=(
            "After answer generation, run FastChat gen_judgment.py with evaluation.judge_model. "
            "This makes OpenAI API calls and requires OPENAI_API_KEY."
        ),
    )
    parser.add_argument(
        "--fastchat_llm_judge_dir",
        default=None,
        help="Path to FastChat/fastchat/llm_judge, used when --run_fastchat_judge is set.",
    )
    parser.add_argument("--fastchat_bench_name", default=None, help="FastChat benchmark name, usually mt_bench.")
    parser.add_argument("--fastchat_python", default=None, help="Python executable for FastChat judging. Defaults to this interpreter.")
    parser.add_argument("--judge_model", default=None, help="FastChat judge model. Defaults to evaluation.judge_model.")
    parser.add_argument("--judge_parallel", type=int, default=None, help="FastChat gen_judgment.py --parallel value.")
    parser.add_argument(
        "--download_questions",
        action="store_true",
        help="Download the official FastChat MT-Bench questions if the local question file is missing.",
    )
    parser.add_argument("--config_override", action="append", default=[])
    parser.add_argument("--sampling_override", action="append", default=[], help="Nested override, e.g. cfg.scale=0.0.")
    return parser.parse_args()


def _repo_path(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    expanded = os.path.expandvars(os.path.expanduser(path))
    return expanded if os.path.isabs(expanded) else os.path.join(REPO_ROOT, expanded)


def _load_eval_config(path: str) -> dict[str, Any]:
    resolved = _repo_path(path)
    if not resolved or not os.path.exists(resolved):
        raise FileNotFoundError(f"Evaluation config not found: {path}")
    with open(resolved, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, Mapping):
        raise ValueError(f"Expected mapping in evaluation config: {path}")
    return dict(data)


def _read_json_records(path: str) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        if path.endswith(".jsonl"):
            return [json.loads(line) for line in handle if line.strip()]
        data = json.load(handle)
    if isinstance(data, list):
        return [dict(item) for item in data]
    if isinstance(data, Mapping):
        for key in ("questions", "data", "examples", "records", "mt_bench"):
            if isinstance(data.get(key), list):
                return [dict(item) for item in data[key]]
    raise ValueError(f"Could not find a question list in {path}")


def _download_file(url: str, destination: str) -> None:
    """Download a small text file used by the benchmark question loader."""

    Path(destination).parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=60) as response:
        payload = response.read()
    with open(destination, "wb") as handle:
        handle.write(payload)


def _parse_int_list(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, str):
        if not value.strip():
            return []
        return [int(chunk) for chunk in re.split(r"[\s,]+", value.strip()) if chunk]
    return [int(item) for item in value]


def _parse_scalar_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        if not value.strip():
            return []
        return [chunk for chunk in re.split(r"[\s,]+", value.strip()) if chunk]
    return [str(item) for item in value]


def _normalize_question(raw: Mapping[str, Any], index: int) -> MTBenchQuestion:
    turns = raw.get("turns")
    if not isinstance(turns, list) or not turns:
        raise ValueError(f"MT-Bench record {index} must contain a non-empty `turns` list.")
    normalized_turns = [str(turn).strip() for turn in turns]
    if not normalized_turns[0]:
        raise ValueError(f"MT-Bench record {index} has an empty first turn.")
    return MTBenchQuestion(
        index=index,
        question_id=raw.get("question_id", raw.get("id", index)),
        category=str(raw.get("category", "unknown") or "unknown"),
        turns=normalized_turns,
        raw=dict(raw),
    )


def _question_file_from_config(eval_cfg: Mapping[str, Any], question_file: Optional[str], download_questions: bool) -> str:
    configured = question_file or eval_cfg.get("question_file") or eval_cfg.get("dataset_path")
    if not configured:
        configured = "data/mt_bench/question.jsonl"
    resolved = _repo_path(str(configured)) or str(configured)
    if os.path.exists(resolved):
        return resolved

    should_download = bool(download_questions or eval_cfg.get("download_if_missing", False))
    if should_download:
        url = str(eval_cfg.get("question_url") or DEFAULT_QUESTION_URL)
        print(f"Downloading MT-Bench questions from {url} to {resolved}", file=sys.stderr)
        _download_file(url, resolved)
        return resolved

    raise FileNotFoundError(
        f"MT-Bench question file not found: {resolved}\n"
        "Set evaluation.question_file, pass --question_file, or rerun with --download_questions."
    )


def load_mt_bench_questions(eval_cfg: Mapping[str, Any], question_file: Optional[str], download_questions: bool) -> list[MTBenchQuestion]:
    resolved = _question_file_from_config(eval_cfg, question_file, download_questions)
    raw_records = _read_json_records(resolved)
    return [_normalize_question(raw, idx) for idx, raw in enumerate(raw_records)]


def select_eval_questions(questions: list[MTBenchQuestion], eval_cfg: Mapping[str, Any]) -> list[MTBenchQuestion]:
    """Select a deterministic MT-Bench subset before answer generation."""

    selected = list(questions)

    question_ids = _parse_scalar_list(eval_cfg.get("subset_question_ids"))
    if question_ids:
        by_id = {str(question.question_id): question for question in selected}
        missing = [qid for qid in question_ids if qid not in by_id]
        if missing:
            raise ValueError(f"Requested MT-Bench question IDs are unavailable: {missing[:10]}")
        return [by_id[qid] for qid in question_ids]

    explicit_indices = _parse_int_list(eval_cfg.get("subset_indices"))
    if explicit_indices:
        by_index = {int(question.index): question for question in selected}
        missing = [idx for idx in explicit_indices if idx not in by_index]
        if missing:
            raise ValueError(f"Requested MT-Bench row indices are unavailable: {missing[:10]}")
        return [by_index[idx] for idx in explicit_indices]

    subset_size = int(eval_cfg.get("subset_size", 0) or 0)
    if subset_size > 0:
        rng = random.Random(int(eval_cfg.get("subset_seed", eval_cfg.get("seed", 0)) or 0))
        subset_size = min(subset_size, len(selected))
        positions = sorted(rng.sample(range(len(selected)), subset_size))
        selected = [selected[pos] for pos in positions]
    else:
        raw_percentage = eval_cfg.get("subset_percentage", 100.0)
        percentage = 100.0 if raw_percentage is None or raw_percentage == "" else float(raw_percentage)
        if percentage <= 0.0 or percentage > 100.0:
            raise ValueError(f"evaluation.subset_percentage must be in (0, 100], got {percentage}.")
        if selected and percentage < 100.0:
            subset_size = max(1, int(math.ceil(len(selected) * percentage / 100.0)))
            subset_size = min(subset_size, len(selected))
            rng = random.Random(int(eval_cfg.get("subset_seed", eval_cfg.get("seed", 0)) or 0))
            positions = sorted(rng.sample(range(len(selected)), subset_size))
            selected = [selected[pos] for pos in positions]

    max_examples = int(eval_cfg.get("max_examples", 0) or 0)
    if max_examples > 0:
        selected = selected[:max_examples]
    return selected


def _fastchat_prompt(first_turn: str, template_name: str) -> Optional[str]:
    """Build a prompt with FastChat when the package is installed locally."""

    try:
        from fastchat.model import get_conversation_template
    except Exception:
        # FastChat is an optional convenience dependency here.  Some notebook
        # environments have a partially importable FastChat/transformers stack
        # (for example a broken flash_attn module spec), which can raise
        # ValueError/RuntimeError before ImportError is reached.  Treat those
        # optional-import failures the same as "FastChat unavailable" so the
        # Vicuna fallback below preserves answer generation.
        return None

    try:
        conv = get_conversation_template(template_name)
        conv.append_message(conv.roles[0], first_turn)
        conv.append_message(conv.roles[1], None)
        return conv.get_prompt()
    except Exception:
        # Keep non-Vicuna template requests strict in build_prompt(): returning
        # None there will raise unless the local Vicuna fallback is appropriate.
        return None


def _local_vicuna_prompt(first_turn: str) -> str:
    """Fallback for the Vicuna v1.x template used by FastChat MT-Bench."""

    return f"{VICUNA_SYSTEM_MESSAGE} USER: {first_turn.strip()} ASSISTANT:"


def build_prompt(question: MTBenchQuestion, eval_cfg: Mapping[str, Any]) -> str:
    """Construct the conditional prompt for the first MT-Bench turn.

    ``prompt_style=vicuna`` matches the paper plan and tries FastChat's
    ``get_conversation_template("models/vicuna-7b-v1.5")`` first.  If FastChat
    is unavailable, the local Vicuna-format fallback keeps answer generation
    self-contained.
    """

    first_turn = question.turns[0].strip()
    prompt_style = str(eval_cfg.get("prompt_style", "vicuna") or "vicuna").lower()
    if prompt_style in {"vicuna", "fastchat"}:
        template_name = str(eval_cfg.get("fastchat_template", "models/vicuna-7b-v1.5"))
        if bool(eval_cfg.get("use_fastchat_template", True)):
            prompt = _fastchat_prompt(first_turn, template_name)
            if prompt is not None:
                return prompt
        if "vicuna" not in template_name.lower() and prompt_style == "fastchat":
            raise RuntimeError(
                "FastChat is not installed, and the requested template is not the built-in Vicuna fallback. "
                "Install FastChat or set evaluation.prompt_style=vicuna/sft_chat/plain."
            )
        return _local_vicuna_prompt(first_turn)

    if prompt_style == "sft_chat":
        return f"USER:\n{first_turn}\nASSISTANT:\n"
    if prompt_style == "plain":
        instruction = str(eval_cfg.get("instruction", "") or "").strip()
        answer_prefix = str(eval_cfg.get("answer_prefix", "") or "")
        parts = []
        if instruction:
            parts.append(instruction)
            parts.append("\n\n")
        parts.append(first_turn)
        if answer_prefix:
            parts.append(answer_prefix)
        return "".join(parts)
    raise ValueError(f"Unknown MT-Bench prompt_style: {prompt_style}")


def _tokenizer_ids(tokenizer, text: str) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=False)
    if isinstance(encoded, Mapping):
        return list(encoded.get("input_ids", []))
    return list(getattr(encoded, "input_ids", []))


def _optional_positive_int(value: Any) -> Optional[int]:
    if value in {None, "", 0, "0"}:
        return None
    parsed = int(value)
    return parsed if parsed > 0 else None


def encode_batch(
    model: SamplingModel,
    prompts: list[str],
    answer_tokens: Optional[int],
    max_length: int,
    pad_to_max_length: bool,
    min_answer_tokens: int,
):
    """Encode prompts and mask the answer span.

    With ``answer_tokens=None`` the answer span fills the remaining fixed context
    after the prompt, so prompt plus generated answer never exceeds
    ``max_sequence_length``.  This matches the conservative 1024-token total
    context recommended in the MT-Bench plan.
    """

    prompt_ids = [_tokenizer_ids(model.tokenizer, prompt) for prompt in prompts]
    min_answer_tokens = max(1, int(min_answer_tokens))
    max_length = max(min_answer_tokens + 1, int(max_length))
    if answer_tokens is None:
        max_prompt_tokens = max(1, max_length - min_answer_tokens)
    else:
        answer_tokens = max(1, int(answer_tokens))
        max_prompt_tokens = max(1, max_length - answer_tokens)

    prompt_ids = [ids[-max_prompt_tokens:] if len(ids) > max_prompt_tokens else ids for ids in prompt_ids]

    if answer_tokens is None or pad_to_max_length:
        seq_len = max_length
    else:
        max_prompt = max((len(ids) for ids in prompt_ids), default=0)
        seq_len = min(max_length, max_prompt + answer_tokens)

    mask_id = int(model.mask_token_id)
    input_ids = torch.full((len(prompts), seq_len), mask_id, device=model.device, dtype=torch.long)
    attention_mask = torch.zeros_like(input_ids)
    corrupt_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    spans = []

    for row, ids in enumerate(prompt_ids):
        prefix_len = min(len(ids), max(0, seq_len - min_answer_tokens))
        if prefix_len:
            input_ids[row, :prefix_len] = torch.tensor(ids[:prefix_len], device=model.device, dtype=torch.long)
        answer_end = seq_len if answer_tokens is None else min(seq_len, prefix_len + answer_tokens)
        attention_mask[row, :answer_end] = 1
        corrupt_mask[row, prefix_len:answer_end] = True
        spans.append((prefix_len, answer_end))
    return {"input_ids": input_ids, "attention_mask": attention_mask}, corrupt_mask, spans


def _flatten_special_token_strings(tokenizer) -> list[str]:
    values = []
    for token in getattr(tokenizer, "special_tokens_map", {}).values():
        if isinstance(token, list):
            values.extend(str(item) for item in token)
        elif token is not None:
            values.append(str(token))
    return [value for value in values if value]


def _trim_at_token_ids(token_ids: torch.Tensor, stop_token_ids: set[int]) -> torch.Tensor:
    if not stop_token_ids:
        return token_ids
    ids = token_ids.detach().cpu().tolist()
    for pos, token_id in enumerate(ids):
        if int(token_id) in stop_token_ids:
            return token_ids[:pos]
    return token_ids


def _trim_at_stop_strings(text: str, stop_strings: list[str]) -> str:
    stops = [text.find(stop) for stop in stop_strings if stop and text.find(stop) >= 0]
    if not stops:
        return text
    return text[: min(stops)]


def _clean_completion(model: SamplingModel, token_ids: torch.Tensor, eval_cfg: Mapping[str, Any]) -> tuple[str, int]:
    """Decode one answer span and remove template/special-token artifacts."""

    stop_token_ids = set(int(item) for item in eval_cfg.get("stop_token_ids", []) or [])
    if model.eos_token_id is not None:
        stop_token_ids.add(int(model.eos_token_id))
    trimmed_ids = _trim_at_token_ids(token_ids, stop_token_ids)
    text = model.tokenizer.decode(trimmed_ids.detach().cpu().tolist(), skip_special_tokens=True)

    stop_strings = list(eval_cfg.get("stop_strings", DEFAULT_STOP_STRINGS) or [])
    text = _trim_at_stop_strings(text, [str(item) for item in stop_strings])
    for special_token in _flatten_special_token_strings(model.tokenizer):
        text = text.replace(special_token, "")
    for prefix in ("ASSISTANT:", "Assistant:", "assistant:"):
        if text.strip().startswith(prefix):
            text = text.strip()[len(prefix) :]
            break
    return text.strip(), int(trimmed_ids.numel())


def _history_json(history):
    return [
        {
            "promotion_step": item.promotion_step,
            "remaining_before": item.remaining_before,
            "promoted": item.promoted,
            "promotion_fraction": item.promotion_fraction,
            "promoted_fraction": [
                float(promoted) / float(max(1, remaining))
                for promoted, remaining in zip(item.promoted, item.remaining_before)
            ],
        }
        for item in history
    ]


@torch.no_grad()
def generate_answers(
    questions: list[MTBenchQuestion],
    model: SamplingModel,
    sampler: OnlineTokenPromotionSampler,
    eval_cfg: Mapping[str, Any],
    seed: int,
):
    """Generate first-turn MT-Bench answers for a normalized question list."""

    if not bool(eval_cfg.get("first_turn_only", True)):
        raise NotImplementedError("This evaluator implements the paper-style first-turn-only MT-Bench protocol.")

    generator_device = model.device if model.device.type == "cuda" else torch.device("cpu")
    generator = torch.Generator(device=generator_device)
    generator.manual_seed(int(seed))

    batch_size = max(1, int(eval_cfg.get("batch_size", 1) or 1))
    answer_tokens = _optional_positive_int(eval_cfg.get("answer_max_tokens"))
    max_length = int(eval_cfg.get("max_sequence_length") or model.sampling_config.max_length)
    pad_to_max_length = bool(eval_cfg.get("pad_to_max_length", True))
    min_answer_tokens = max(1, int(eval_cfg.get("min_answer_tokens", 1) or 1))
    progress_enabled = bool(eval_cfg.get("progress", True))

    try:
        from tqdm.auto import tqdm
    except ImportError:
        tqdm = None
    iterator = range(0, len(questions), batch_size)
    if progress_enabled and tqdm is not None:
        iterator = tqdm(iterator, desc="MT-Bench", unit="batch")

    rows = []
    for start in iterator:
        chunk = questions[start : start + batch_size]
        prompts = [build_prompt(question, eval_cfg) for question in chunk]
        batch, corrupt_mask, spans = encode_batch(
            model,
            prompts,
            answer_tokens=answer_tokens,
            max_length=max_length,
            pad_to_max_length=pad_to_max_length,
            min_answer_tokens=min_answer_tokens,
        )
        result = sampler.sample_batch(batch, corrupt_mask, generator)
        for row_idx, question in enumerate(chunk):
            answer_start, answer_end = spans[row_idx]
            completion, completion_tokens = _clean_completion(
                model,
                result.generated_ids[row_idx, answer_start:answer_end],
                eval_cfg,
            )
            turn_answers = [completion] + ["" for _ in question.turns[1:]]
            sample = model.tokenizer.decode(
                result.generated_ids[row_idx, :answer_end].detach().cpu().tolist(),
                skip_special_tokens=True,
            ).strip()
            rows.append(
                {
                    "index": question.index,
                    "question_id": question.question_id,
                    "category": question.category,
                    "question_turns": question.turns,
                    "answer_turns": turn_answers,
                    "first_turn_answer": completion,
                    "prompt": prompts[row_idx],
                    "sample": sample,
                    "prompt_tokens": int(answer_start),
                    "answer_span_tokens": int(answer_end - answer_start),
                    "completion_tokens_before_stop": completion_tokens,
                    "answer_chars": len(completion),
                    "promotion_steps": _history_json(result.history),
                }
            )
    return rows


def _qid_sort_key(value: Any):
    try:
        return (0, int(value))
    except (TypeError, ValueError):
        return (1, str(value))


def _stable_answer_id(model_id: str, question_id: Any, answer_turns: list[str]) -> str:
    payload = json.dumps(
        {"model_id": model_id, "question_id": question_id, "turns": answer_turns},
        ensure_ascii=False,
        sort_keys=True,
    )
    return uuid.uuid5(uuid.NAMESPACE_URL, payload).hex


def _answer_json(row: Mapping[str, Any], model_id: str, timestamp: float) -> dict[str, Any]:
    turns = [str(item) for item in row["answer_turns"]]
    return {
        "question_id": row["question_id"],
        "answer_id": _stable_answer_id(model_id, row["question_id"], turns),
        "model_id": model_id,
        "choices": [{"index": 0, "turns": turns}],
        "tstamp": timestamp,
    }


def write_answer_file(rows: list[dict[str, Any]], answer_file: str, model_id: str) -> None:
    """Write one deduplicated FastChat answer object per MT-Bench question."""

    answer_path = Path(_repo_path(answer_file) or answer_file)
    answer_path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = time.time()
    by_qid = {str(row["question_id"]): row for row in rows}
    sorted_rows = sorted(by_qid.values(), key=lambda row: _qid_sort_key(row["question_id"]))
    with answer_path.open("w", encoding="utf-8") as handle:
        for row in sorted_rows:
            handle.write(json.dumps(_answer_json(row, model_id, timestamp), ensure_ascii=False, sort_keys=True) + "\n")


def write_details_file(rows: list[dict[str, Any]], details_jsonl: Optional[str]) -> None:
    if not details_jsonl:
        return
    details_path = Path(_repo_path(details_jsonl) or details_jsonl)
    details_path.parent.mkdir(parents=True, exist_ok=True)
    sorted_rows = sorted(rows, key=lambda row: _qid_sort_key(row["question_id"]))
    with details_path.open("w", encoding="utf-8") as handle:
        for row in sorted_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _mean(values: list[float]) -> Optional[float]:
    if not values:
        return None
    return float(sum(values) / len(values))


def _resolve_fastchat_judge_dir(eval_cfg: Mapping[str, Any]) -> Optional[Path]:
    """Return the configured FastChat ``fastchat/llm_judge`` directory.

    FastChat's MT-Bench judge scripts expect to be run from this directory so
    that relative paths like ``data/mt_bench/model_answer`` and
    ``data/judge_prompts.jsonl`` resolve exactly as in the official workflow.
    """

    value = eval_cfg.get("fastchat_llm_judge_dir") or eval_cfg.get("llm_judge_dir")
    resolved = _repo_path(str(value)) if value else None
    return None if not resolved else Path(resolved)


def _fastchat_judgment_jsonl(eval_cfg: Mapping[str, Any]) -> Path:
    """Default FastChat single-answer judgment file for the configured judge."""

    judge_dir = _resolve_fastchat_judge_dir(eval_cfg)
    if judge_dir is None:
        raise ValueError("Set evaluation.fastchat_llm_judge_dir or pass --fastchat_llm_judge_dir.")
    bench_name = str(eval_cfg.get("fastchat_bench_name", "mt_bench") or "mt_bench")
    judge_model = str(eval_cfg.get("judge_model", "gpt-4o-2024-05-13") or "gpt-4o-2024-05-13")
    return judge_dir / "data" / bench_name / "model_judgment" / f"{judge_model}_single.jsonl"


def _write_fastchat_first_turn_questions(question_src: Path, question_dst: Path) -> None:
    """Stage an MT-Bench question file that only exposes the first turn.

    FastChat creates second-turn/multi-turn judge requests whenever
    ``data/<bench>/question.jsonl`` contains two turns, even for the normal
    single-answer ``gen_judgment.py`` mode.  The comparison paper reports the
    first-turn MT-Bench score, so we overwrite the staged judge question file
    with a one-turn copy of the generation questions before calling FastChat.
    The model answer JSONL may still keep empty later turns for compatibility.
    """

    question_dst.parent.mkdir(parents=True, exist_ok=True)
    with question_src.open("r", encoding="utf-8") as src, question_dst.open("w", encoding="utf-8") as dst:
        for line_number, line in enumerate(src, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            turns = row.get("turns") or []
            if not turns:
                raise ValueError(f"MT-Bench question row {line_number} has no turns: {question_src}")
            row["turns"] = [str(turns[0])]
            dst.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _prune_existing_fastchat_judgments(judgment_jsonl: Path, model_ids: list[str]) -> int:
    """Remove stale judgment rows for models that are about to be rescored.

    FastChat appends to ``<judge_model>_single.jsonl``. If the same model ID was
    previously judged with a two-turn question file, old turn-2 rows would stay
    next to the new first-turn-only rows. We back up the full file, keep other
    models' judgments, and clear only the rows for the active model IDs.
    """

    if not judgment_jsonl.exists():
        return 0

    model_id_set = {str(model_id) for model_id in model_ids}
    kept_lines: list[str] = []
    removed = 0
    with judgment_jsonl.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                kept_lines.append(line.rstrip("\n"))
                continue
            if str(row.get("model")) in model_id_set:
                removed += 1
            else:
                kept_lines.append(line.rstrip("\n"))

    if removed:
        backup_jsonl = judgment_jsonl.with_name(f"{judgment_jsonl.name}.bak-{int(time.time())}")
        shutil.copy2(judgment_jsonl, backup_jsonl)
        with judgment_jsonl.open("w", encoding="utf-8") as handle:
            for line in kept_lines:
                handle.write(f"{line}\n")
        print(
            f"Removed {removed} stale FastChat judgment rows for {sorted(model_id_set)}; "
            f"backup: {backup_jsonl}",
            file=sys.stderr,
        )
    return removed


def _stage_fastchat_inputs(
    *,
    eval_cfg: Mapping[str, Any],
    answer_file: str,
    question_file: str,
    model_id: str,
) -> tuple[Path, Path]:
    """Copy local evaluator outputs into FastChat's expected data layout.

    ``gen_judgment.py`` looks for model answers under
    ``data/<bench_name>/model_answer/<model_id>.jsonl`` relative to
    ``fastchat/llm_judge``.  This helper stages only the files generated by this
    evaluator.  Judge prompts and reference answers should remain the versions
    bundled with the user's FastChat checkout.
    """

    judge_dir = _resolve_fastchat_judge_dir(eval_cfg)
    if judge_dir is None:
        raise ValueError("Set evaluation.fastchat_llm_judge_dir or pass --fastchat_llm_judge_dir.")
    script = judge_dir / "gen_judgment.py"
    if not script.exists():
        raise FileNotFoundError(f"FastChat gen_judgment.py not found: {script}")

    bench_name = str(eval_cfg.get("fastchat_bench_name", "mt_bench") or "mt_bench")
    answer_src = Path(_repo_path(answer_file) or answer_file)
    question_src = Path(_repo_path(question_file) or question_file)
    if not answer_src.exists():
        raise FileNotFoundError(f"Model answer JSONL not found: {answer_src}")
    if not question_src.exists():
        raise FileNotFoundError(f"MT-Bench question JSONL not found: {question_src}")

    data_dir = judge_dir / "data" / bench_name
    answer_dir = data_dir / "model_answer"
    answer_dir.mkdir(parents=True, exist_ok=True)
    staged_answer = answer_dir / f"{model_id}.jsonl"
    if answer_src.resolve() != staged_answer.resolve():
        shutil.copy2(answer_src, staged_answer)

    # Stage a first-turn-only question file every time. FastChat decides whether
    # to create multi-turn judge prompts from the number of turns in this file,
    # so an existing two-turn question.jsonl would make extra second-turn API
    # calls even though this evaluator reports the paper's first-turn score.
    staged_question = data_dir / "question.jsonl"
    if bool(eval_cfg.get("fastchat_first_turn_only_questions", True)):
        _write_fastchat_first_turn_questions(question_src, staged_question)
    elif bool(eval_cfg.get("fastchat_copy_question_if_missing", True)) and not staged_question.exists():
        staged_question.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(question_src, staged_question)

    # FastChat keys reference answers by filename, not by the `model_id` inside
    # each JSONL row. Its bundled MT-Bench reference file is gpt-4.jsonl, but
    # judging with gpt-4o-2024-05-13 makes common.py look for
    # ref_answers["gpt-4o-2024-05-13"]. Reuse the official GPT-4 references
    # under the judge-model filename so newer OpenAI judges work unchanged.
    judge_model = str(eval_cfg.get("judge_model", "gpt-4o-2024-05-13") or "gpt-4o-2024-05-13")
    ref_dir = data_dir / "reference_answer"
    source_ref = ref_dir / "gpt-4.jsonl"
    judge_ref = ref_dir / f"{judge_model}.jsonl"
    if judge_model != "gpt-4" and source_ref.exists() and not judge_ref.exists():
        shutil.copy2(source_ref, judge_ref)
    return judge_dir, staged_answer


def run_fastchat_judge(
    *,
    eval_cfg: Mapping[str, Any],
    answer_file: str,
    question_file: str,
    model_id: str,
) -> dict[str, Any]:
    """Run FastChat single-answer judging and return execution metadata.

    This is deliberately opt-in because it calls the configured OpenAI judge
    model, by default ``gpt-4o-2024-05-13``.  The caller must provide
    ``OPENAI_API_KEY`` in the environment, matching FastChat's own scripts.
    """

    if not os.environ.get("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY must be set before running FastChat GPT judging.")

    judge_dir, staged_answer = _stage_fastchat_inputs(
        eval_cfg=eval_cfg,
        answer_file=answer_file,
        question_file=question_file,
        model_id=model_id,
    )
    python_exe = str(eval_cfg.get("fastchat_python") or sys.executable)
    bench_name = str(eval_cfg.get("fastchat_bench_name", "mt_bench") or "mt_bench")
    judge_model = str(eval_cfg.get("judge_model", "gpt-4o-2024-05-13") or "gpt-4o-2024-05-13")
    parallel = int(eval_cfg.get("judge_parallel", 1) or 1)
    judgment_jsonl = _fastchat_judgment_jsonl(eval_cfg)
    _prune_existing_fastchat_judgments(judgment_jsonl, [model_id])

    command = [
        python_exe,
        "gen_judgment.py",
        "--bench-name",
        bench_name,
        "--judge-model",
        judge_model,
        "--model-list",
        model_id,
        "--parallel",
        str(parallel),
    ]
    print("Running FastChat MT-Bench judge:", " ".join(command), file=sys.stderr)
    # gen_judgment.py prints match stats and asks for Enter before starting.
    # Supplying a newline keeps notebook/script runs non-interactive.
    subprocess.run(command, cwd=str(judge_dir), check=True, input="\n", text=True)
    if not judgment_jsonl.exists():
        raise FileNotFoundError(
            f"FastChat judging finished but did not create the expected judgment file: {judgment_jsonl}"
        )
    return {
        "fastchat_llm_judge_dir": str(judge_dir),
        "fastchat_bench_name": bench_name,
        "judge_model": judge_model,
        "judge_parallel": parallel,
        "staged_answer_file": str(staged_answer),
        "judgment_jsonl": str(judgment_jsonl),
        "command": command,
    }


def summarize_judgments(
    judgment_jsonl: str,
    *,
    model_id: Optional[str] = None,
    question_categories: Optional[Mapping[str, str]] = None,
) -> dict[str, Any]:
    """Summarize FastChat single-answer judgment JSONL.

    FastChat writes one row per scored turn with fields including ``model``,
    ``score`` and ``turn``.  The paper-compatible number is the mean score for
    ``turn == 1``.
    """

    path = Path(_repo_path(judgment_jsonl) or judgment_jsonl)
    if not path.exists():
        raise FileNotFoundError(f"Judgment JSONL not found: {path}")

    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if model_id and str(row.get("model")) != str(model_id):
                continue
            try:
                score = float(row.get("score"))
            except (TypeError, ValueError):
                continue
            if score < 0:
                continue
            turn = int(row.get("turn", 1) or 1)
            rows.append({"question_id": row.get("question_id"), "turn": turn, "score": score, "model": row.get("model")})

    first_turn_scores = [row["score"] for row in rows if row["turn"] == 1]
    second_turn_scores = [row["score"] for row in rows if row["turn"] == 2]
    summary: dict[str, Any] = {
        "path": str(path),
        "model_id": model_id,
        "valid_judgments": len(rows),
        "first_turn_count": len(first_turn_scores),
        "second_turn_count": len(second_turn_scores),
        "first_turn_score": _mean(first_turn_scores),
        "second_turn_score": _mean(second_turn_scores),
        "average_score": _mean([row["score"] for row in rows]),
    }

    if question_categories:
        category_turn_1: dict[str, list[float]] = {}
        for row in rows:
            if row["turn"] != 1:
                continue
            category = question_categories.get(str(row["question_id"]), "unknown")
            category_turn_1.setdefault(category, []).append(row["score"])
        summary["first_turn_category_scores"] = {
            category: _mean(scores) for category, scores in sorted(category_turn_1.items())
        }
    return summary


def _resolve_answer_file(eval_cfg: Mapping[str, Any], answer_file: Optional[str], model_id: str) -> str:
    configured = answer_file or eval_cfg.get("answer_file") or eval_cfg.get("output_jsonl")
    if configured:
        return str(configured).format(model_id=model_id)
    return f"data/mt_bench/model_answer/{model_id}.jsonl"


def _resolve_details_file(eval_cfg: Mapping[str, Any], details_jsonl: Optional[str], model_id: str) -> Optional[str]:
    if details_jsonl is not None:
        return details_jsonl.format(model_id=model_id)
    value = eval_cfg.get("details_jsonl")
    return None if value in {None, ""} else str(value).format(model_id=model_id)


def _apply_cli_eval_overrides(args, eval_cfg: dict[str, Any]) -> dict[str, Any]:
    if args.seed is not None:
        eval_cfg["seed"] = int(args.seed)
    if args.batch_size is not None:
        eval_cfg["batch_size"] = int(args.batch_size)
    if args.max_examples is not None:
        eval_cfg["max_examples"] = int(args.max_examples)
    if args.subset_size is not None:
        eval_cfg["subset_size"] = int(args.subset_size)
    if args.subset_percentage is not None:
        eval_cfg["subset_percentage"] = float(args.subset_percentage)
    if args.subset_seed is not None:
        eval_cfg["subset_seed"] = int(args.subset_seed)
    if args.subset_indices is not None:
        eval_cfg["subset_indices"] = args.subset_indices
    if args.subset_question_ids is not None:
        eval_cfg["subset_question_ids"] = args.subset_question_ids
    if args.judgment_jsonl is not None:
        eval_cfg["judgment_jsonl"] = args.judgment_jsonl
    if args.run_fastchat_judge:
        eval_cfg["run_fastchat_judge"] = True
    if args.fastchat_llm_judge_dir is not None:
        eval_cfg["fastchat_llm_judge_dir"] = args.fastchat_llm_judge_dir
    if args.fastchat_bench_name is not None:
        eval_cfg["fastchat_bench_name"] = args.fastchat_bench_name
    if args.fastchat_python is not None:
        eval_cfg["fastchat_python"] = args.fastchat_python
    if args.judge_model is not None:
        eval_cfg["judge_model"] = args.judge_model
    if args.judge_parallel is not None:
        eval_cfg["judge_parallel"] = int(args.judge_parallel)
    if args.download_questions:
        eval_cfg["download_if_missing"] = True
    return eval_cfg


def _base_metrics(
    *,
    model_id: str,
    question_file: str,
    questions: list[MTBenchQuestion],
    selected_questions: list[MTBenchQuestion],
    eval_cfg: Mapping[str, Any],
    answer_file: str,
    details_jsonl: Optional[str],
) -> dict[str, Any]:
    return {
        "task": "mt_bench",
        "protocol": "first_turn_single_answer_grading",
        "model_id": model_id,
        "question_file": question_file,
        "question_url": str(eval_cfg.get("question_url") or DEFAULT_QUESTION_URL),
        "total_questions_available": len(questions),
        "total": len(selected_questions),
        "categories": sorted({question.category for question in selected_questions}),
        "first_turn_only": bool(eval_cfg.get("first_turn_only", True)),
        "judge_model": str(eval_cfg.get("judge_model", "gpt-4o-2024-05-13")),
        "answer_file": str(Path(_repo_path(answer_file) or answer_file)),
        "details_jsonl": None if not details_jsonl else str(Path(_repo_path(details_jsonl) or details_jsonl)),
        "evaluation": dict(eval_cfg),
    }


def _write_metrics(metrics: Mapping[str, Any], answer_file: str) -> Path:
    answer_path = Path(_repo_path(answer_file) or answer_file)
    metrics_path = answer_path.with_name(f"{answer_path.stem}_metrics.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return metrics_path


def main():
    args = parse_args()
    eval_file_cfg = _load_eval_config(args.config)
    model_cfg = dict(eval_file_cfg.get("model", {}) or {})
    eval_cfg = _apply_cli_eval_overrides(args, dict(eval_file_cfg.get("evaluation", {}) or {}))

    model_id = args.model_id or eval_cfg.get("model_id") or model_cfg.get("model_id") or "smdm"
    model_id = str(model_id)
    answer_file = _resolve_answer_file(eval_cfg, args.answer_file, model_id)
    details_jsonl = _resolve_details_file(eval_cfg, args.details_jsonl, model_id)

    question_file = _question_file_from_config(eval_cfg, args.question_file, bool(args.download_questions))
    questions = load_mt_bench_questions(eval_cfg, question_file, bool(args.download_questions))
    selected_questions = select_eval_questions(questions, eval_cfg)
    question_categories = {str(question.question_id): question.category for question in questions}

    metrics = _base_metrics(
        model_id=model_id,
        question_file=question_file,
        questions=questions,
        selected_questions=selected_questions,
        eval_cfg=eval_cfg,
        answer_file=answer_file,
        details_jsonl=details_jsonl,
    )

    if args.score_only:
        if bool(eval_cfg.get("run_fastchat_judge", False)):
            metrics["fastchat_judge"] = run_fastchat_judge(
                eval_cfg=eval_cfg,
                answer_file=answer_file,
                question_file=question_file,
                model_id=model_id,
            )
            judgment_jsonl = metrics["fastchat_judge"]["judgment_jsonl"]
            eval_cfg["judgment_jsonl"] = judgment_jsonl
        else:
            judgment_jsonl = eval_cfg.get("judgment_jsonl")
        if not judgment_jsonl:
            raise ValueError(
                "Pass --judgment_jsonl, set evaluation.judgment_jsonl, or enable --run_fastchat_judge "
                "when using --score_only."
            )
        metrics["judgment"] = summarize_judgments(
            str(judgment_jsonl),
            model_id=model_id,
            question_categories=question_categories,
        )
        _write_metrics(metrics, answer_file)
        print(json.dumps(metrics, sort_keys=True))
        return

    training_config_path = args.training_config or model_cfg.get("training_config")
    if not training_config_path:
        raise ValueError("Set model.training_config in the eval config or pass --training_config.")
    training_config = load_config_from_yaml(_repo_path(str(training_config_path)) or str(training_config_path))
    config_overrides = list(model_cfg.get("config_overrides", []) or []) + list(args.config_override or [])
    if config_overrides:
        training_config = apply_config_overrides(training_config, config_overrides)
    training_config = fill_mlfm_defaults(training_config)

    if args.seed is not None:
        training_config.seed = int(args.seed)
    seed = int(eval_cfg.get("seed", getattr(training_config, "seed", 0)) or 0)
    training_config.seed = seed

    if args.device is not None:
        training_config.device = args.device

    sampling_config = load_sampling_experiment_config(
        config_path=_repo_path(args.config),
        sampling_config_path=_repo_path(args.sampling_config) if args.sampling_config else None,
        overrides=args.sampling_override,
    )
    checkpoint = args.checkpoint if args.checkpoint is not None else model_cfg.get("checkpoint")
    adapter_mode = str(sampling_config.adapter_mode or "finetuned").lower()
    if not checkpoint and adapter_mode == "finetuned":
        checkpoint = infer_checkpoint_from_training_config(
            training_config,
            str(training_config_path),
            repo_root=REPO_ROOT,
            repo_path_fn=_repo_path,
        )
        print(f"Inferred checkpoint from training config: {checkpoint}", file=sys.stderr)

    device = torch.device(
        getattr(training_config, "device", "auto")
        if getattr(training_config, "device", "auto") not in {None, "auto"}
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    model = SamplingModel.from_config_checkpoint(training_config, sampling_config, checkpoint, device=device)
    sampler = OnlineTokenPromotionSampler(model)
    rows = generate_answers(selected_questions, model, sampler, eval_cfg, seed)
    rows.sort(key=lambda row: _qid_sort_key(row["question_id"]))

    write_answer_file(rows, answer_file, model_id)
    write_details_file(rows, details_jsonl)

    nonempty = sum(1 for row in rows if str(row["first_turn_answer"]).strip())
    metrics.update(
        {
            "training_config": str(training_config_path),
            "checkpoint": checkpoint,
            "checkpoint_step": int(model.checkpoint_step),
            "adapter_mode": adapter_mode,
            "seed": seed,
            "sampling": sampling_config.to_dict(),
            "nonempty_first_turn_answers": nonempty,
            "empty_first_turn_answers": len(rows) - nonempty,
            "mean_first_turn_answer_chars": _mean([float(row["answer_chars"]) for row in rows]),
            "mean_completion_tokens_before_stop": _mean(
                [float(row["completion_tokens_before_stop"]) for row in rows]
            ),
        }
    )

    judgment_jsonl = eval_cfg.get("judgment_jsonl")
    if bool(eval_cfg.get("run_fastchat_judge", False)):
        metrics["fastchat_judge"] = run_fastchat_judge(
            eval_cfg=eval_cfg,
            answer_file=answer_file,
            question_file=question_file,
            model_id=model_id,
        )
        judgment_jsonl = metrics["fastchat_judge"]["judgment_jsonl"]
        eval_cfg["judgment_jsonl"] = judgment_jsonl

    if judgment_jsonl and Path(_repo_path(str(judgment_jsonl)) or str(judgment_jsonl)).exists():
        metrics["judgment"] = summarize_judgments(
            str(judgment_jsonl),
            model_id=model_id,
            question_categories=question_categories,
        )

    _write_metrics(metrics, answer_file)
    print(json.dumps(metrics, sort_keys=True))


if __name__ == "__main__":
    main()
