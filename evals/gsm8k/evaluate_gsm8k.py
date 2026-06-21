#!/usr/bin/env python
"""Self-contained GSM8K evaluation for SMDM/MLFM checkpoints.

The evaluator mirrors ``evals/arc_easy``: it loads a training config plus an
optional adapter checkpoint, runs the SDE online-token-promotion sampler, writes
per-example JSONL predictions, and emits a metrics JSON beside the predictions.
GSM8K differs from ARC-Easy in two important ways:

* the model generates a free-form solution rather than a choice letter; and
* the primary metric exact-matches the normalized number that appears after a
  ``###`` or ``####`` final-answer marker.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from fractions import Fraction
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


# GSM8K gold answers conventionally end with "#### <number>".  The strict
# extractor intentionally follows that convention because the paper's released
# evaluator searches for the answer after this marker.
STRICT_ANSWER_RE = re.compile(r"#{3,4}\s*([^\n\r]*)")

# A permissive numeric pattern used for gold fallbacks and diagnostic flexible
# accuracy.  It accepts integers, decimals, comma separators, and simple
# fractions such as "3/4".
NUMBER_RE = re.compile(r"[-+]?(?:\d[\d,]*(?:\.\d*)?|\.\d+)(?:\s*/\s*[-+]?\d[\d,]*)?")


@dataclass
class GSM8KRecord:
    """Normalized GSM8K row used by the evaluator."""

    index: int
    record_id: str
    question: str
    answer: str
    gold_answer: str


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate GSM8K with a one-GPU SMDM/MLFM sampler.")
    parser.add_argument("--config", default="evals/gsm8k/config.yml")
    parser.add_argument("--training_config", default=None, help="Training/model config, e.g. runs/train/config.yml.")
    parser.add_argument("--checkpoint", default=None, help="Adapter checkpoint. Optional when sampling.adapter_mode=base.")
    parser.add_argument("--sampling_config", default=None, help="Optional YAML containing `sampling:` overrides.")
    parser.add_argument("--dataset_path", default=None, help="Local JSONL/JSON file. If omitted, loads openai/gsm8k.")
    parser.add_argument("--output_jsonl", default=None)
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--subset_size", type=int, default=None)
    parser.add_argument("--subset_percentage", type=float, default=None, help="Deterministic percentage of records to evaluate, e.g. 10 for 10%.")
    parser.add_argument("--subset_seed", type=int, default=None)
    parser.add_argument("--subset_indices", default=None, help="Comma-separated original GSM8K record indices to evaluate.")
    parser.add_argument("--subset_indices_file", default=None, help="JSON/JSONL/text file containing original GSM8K record indices.")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_fewshots", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--config_override", action="append", default=[])
    parser.add_argument("--sampling_override", action="append", default=[], help="Nested override, e.g. cfg.scale=0.0.")
    return parser.parse_args()


def _repo_path(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    return path if os.path.isabs(path) else os.path.join(REPO_ROOT, path)


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
        return data
    if isinstance(data, Mapping):
        for key in ("data", "examples", "records", "train", "test"):
            if isinstance(data.get(key), list):
                return list(data[key])
    raise ValueError(f"Could not find a record list in {path}")


def _read_index_file(path: str) -> list[int]:
    resolved = _repo_path(path) or path
    with open(resolved, "r", encoding="utf-8") as handle:
        text = handle.read().strip()
    if not text:
        return []
    if resolved.endswith(".json"):
        parsed = json.loads(text)
        if isinstance(parsed, Mapping):
            for key in ("indices", "subset_indices", "data"):
                if key in parsed:
                    parsed = parsed[key]
                    break
        return [int(item) for item in parsed]
    if resolved.endswith(".jsonl"):
        values = []
        for line in text.splitlines():
            if not line.strip():
                continue
            parsed = json.loads(line)
            values.append(int(parsed["index"] if isinstance(parsed, Mapping) else parsed))
        return values
    values = []
    for chunk in re.split(r"[\s,]+", text):
        if chunk:
            values.append(int(chunk))
    return values


def _parse_indices(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, str):
        if not value.strip():
            return []
        return [int(chunk) for chunk in re.split(r"[\s,]+", value.strip()) if chunk]
    return [int(item) for item in value]


def _raw_records_from_hf(dataset_name: str, dataset_config: Optional[str], split: str) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install `datasets` or pass --dataset_path with local GSM8K records.") from exc

    if dataset_config:
        dataset = load_dataset(dataset_name, dataset_config, split=split)
    else:
        dataset = load_dataset(dataset_name, split=split)
    return [dict(item) for item in dataset]


def _strip_wrappers(text: str) -> str:
    """Remove common answer wrappers without changing the numeric value."""

    cleaned = str(text).strip()
    cleaned = cleaned.replace("$", "").replace(",", "")
    cleaned = cleaned.replace("\\boxed", "")
    cleaned = cleaned.strip(" `~*=:_;,.!?)(")
    if cleaned.startswith("{") and cleaned.endswith("}"):
        cleaned = cleaned[1:-1].strip()
    return cleaned


def _canonical_decimal(value: Decimal) -> str:
    """Convert a Decimal into a stable exact-match string."""

    if value == value.to_integral_value():
        return str(int(value))
    normalized = value.normalize()
    # Decimal may switch to exponent notation; fixed-point strings are easier to
    # compare and read in JSONL outputs.
    return format(normalized, "f").rstrip("0").rstrip(".")


def canonicalize_number(value: Optional[str]) -> Optional[str]:
    """Return a canonical numeric string, or None when no number is present.

    GSM8K mostly uses integer final answers, but this handles decimals and
    simple fractions so local variants of the benchmark do not need a separate
    scorer.
    """

    if value is None:
        return None
    cleaned = _strip_wrappers(str(value))
    match = NUMBER_RE.search(cleaned)
    if not match:
        return None
    token = match.group(0).replace(" ", "").replace(",", "")
    try:
        if "/" in token:
            numerator, denominator = token.split("/", 1)
            fraction = Fraction(int(numerator), int(denominator))
            if fraction.denominator == 1:
                return str(fraction.numerator)
            return f"{fraction.numerator}/{fraction.denominator}"
        return _canonical_decimal(Decimal(token))
    except (InvalidOperation, ValueError, ZeroDivisionError):
        return None


def extract_strict_answer(text: str) -> Optional[str]:
    """Extract the final answer after the last ``####`` marker.

    This is the primary paper-style extraction path.  Returning None for outputs
    without the marker is deliberate: it exposes formatting failures instead of
    quietly granting credit from an incidental number in the reasoning.
    """

    matches = STRICT_ANSWER_RE.findall(str(text or ""))
    if not matches:
        return None
    return canonicalize_number(matches[-1])


def extract_flexible_answer(text: str) -> Optional[str]:
    """Extract the last numeric value in a completion.

    This is reported as a diagnostic because it is useful while developing
    prompts or checkpoints, but it is not the default GSM8K accuracy.
    """

    matches = NUMBER_RE.findall(str(text or ""))
    if not matches:
        return None
    return canonicalize_number(matches[-1])


def extract_gold_answer(answer: str) -> Optional[str]:
    """Normalize a GSM8K gold answer from either raw or final-only formats."""

    return extract_strict_answer(answer) or extract_flexible_answer(answer)


def _normalize_record(raw: Mapping[str, Any], index: int) -> GSM8KRecord:
    question = str(raw.get("question", raw.get("query", raw.get("prompt", "")))).strip()
    answer = str(raw.get("answer", raw.get("target", raw.get("final_answer", "")))).strip()
    if not question or not answer:
        raise ValueError(f"GSM8K record {index} is missing a question or answer.")
    gold = extract_gold_answer(answer)
    if gold is None:
        raise ValueError(f"GSM8K record {index} has no parseable gold answer: {answer!r}")
    return GSM8KRecord(
        index=index,
        record_id=str(raw.get("id", raw.get("idx", index))),
        question=question,
        answer=answer,
        gold_answer=gold,
    )


def load_gsm8k_records(eval_cfg: Mapping[str, Any], dataset_path: Optional[str]) -> list[GSM8KRecord]:
    if dataset_path:
        raw_records = _read_json_records(dataset_path)
    elif eval_cfg.get("dataset_path"):
        raw_records = _read_json_records(_repo_path(str(eval_cfg["dataset_path"])) or str(eval_cfg["dataset_path"]))
    else:
        dataset_config = eval_cfg.get("dataset_config", "main")
        raw_records = _raw_records_from_hf(
            str(eval_cfg.get("dataset_name", "openai/gsm8k")),
            None if dataset_config in {None, ""} else str(dataset_config),
            str(eval_cfg.get("split", "test")),
        )
    return [_normalize_record(raw, idx) for idx, raw in enumerate(raw_records)]


def _format_one_prompt(record: GSM8KRecord, eval_cfg: Mapping[str, Any], include_answer: bool) -> str:
    question_prefix = str(eval_cfg.get("question_prefix", "Question: "))
    answer_prefix = str(eval_cfg.get("answer_prefix", "") or "")
    text = f"{question_prefix}{record.question}"
    if include_answer:
        if answer_prefix:
            text += answer_prefix
        else:
            text += "\n"
        text += record.answer.strip()
    elif answer_prefix:
        text += answer_prefix
    return text


def _format_sft_chat_prompt(record: GSM8KRecord, include_answer: bool) -> str:
    text = f"USER:\n{record.question.strip()}\nASSISTANT:\n"
    if include_answer:
        text += record.answer.strip()
    return text


def build_prompt(record: GSM8KRecord, eval_cfg: Mapping[str, Any], fewshot_records: list[GSM8KRecord]) -> str:
    """Build the conditional prompt for one GSM8K example.

    ``prompt_style=sft_chat`` matches the current SFT math format:
    ``USER:\n<raw problem>\nASSISTANT:\n``.  The ``plain`` style keeps
    the configurable ``Question:`` / ``Answer:`` prefixes for ablations.
    """

    prompt_style = str(eval_cfg.get("prompt_style", "plain") or "plain").lower()
    if prompt_style == "sft_chat":
        separator = str(eval_cfg.get("fewshot_separator", "\n\n"))
        parts = []
        for shot in fewshot_records:
            parts.append(_format_sft_chat_prompt(shot, include_answer=True))
            parts.append(separator)
        parts.append(_format_sft_chat_prompt(record, include_answer=False))
        return "".join(parts)

    instruction = str(eval_cfg.get("instruction", "") or "").strip()
    separator = str(eval_cfg.get("fewshot_separator", "\n\n"))
    parts = []
    if instruction:
        parts.append(instruction)
        parts.append(separator)
    for shot in fewshot_records:
        parts.append(_format_one_prompt(shot, eval_cfg, include_answer=True))
        parts.append(separator)
    parts.append(_format_one_prompt(record, eval_cfg, include_answer=False))
    prompt = "".join(parts)
    if prompt_style == "chat":
        return f"USER:\n{prompt.strip()}\nASSISTANT:\n"
    return prompt


def select_eval_records(records: list[GSM8KRecord], eval_cfg: Mapping[str, Any]) -> list[GSM8KRecord]:
    """Select deterministic eval subsets from already few-shot-stripped records.

    Precedence is explicit indices, fixed subset size, percentage subset, then
    max_examples.  Percentage and fixed-size subsets are random but reproducible:
    the same loaded dataset order plus the same `subset_seed` always produces the
    same row indices.  Sampled positions are sorted back into original order so
    JSONL outputs remain stable and easy to diff across runs.
    """

    selected = list(records)
    explicit_indices = _parse_indices(eval_cfg.get("subset_indices"))
    if eval_cfg.get("subset_indices_file"):
        explicit_indices.extend(_read_index_file(str(eval_cfg["subset_indices_file"])))
    if explicit_indices:
        by_index = {int(record.index): record for record in selected}
        missing = [idx for idx in explicit_indices if idx not in by_index]
        if missing:
            raise ValueError(f"Requested GSM8K subset indices are unavailable after few-shot split: {missing[:10]}")
        return [by_index[idx] for idx in explicit_indices]

    subset_size = int(eval_cfg.get("subset_size", 0) or 0)
    if subset_size > 0:
        import random

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
            import random

            subset_size = max(1, int(math.ceil(len(selected) * percentage / 100.0)))
            subset_size = min(subset_size, len(selected))
            rng = random.Random(int(eval_cfg.get("subset_seed", eval_cfg.get("seed", 0)) or 0))
            positions = sorted(rng.sample(range(len(selected)), subset_size))
            selected = [selected[pos] for pos in positions]

    max_examples = int(eval_cfg.get("max_examples", 0) or 0)
    if max_examples > 0:
        selected = selected[:max_examples]
    return selected


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

    When ``answer_tokens`` is None, every token after the prompt up to
    ``max_length`` is treated as the answer span. This matches the current SFT
    GSM8K layout: a fixed prompt+masked-answer row with the observed question at
    the front and masks everywhere else.
    """

    prompt_ids = [_tokenizer_ids(model.tokenizer, prompt) for prompt in prompts]
    min_answer_tokens = max(1, int(min_answer_tokens))
    max_length = max(min_answer_tokens + 1, int(max_length))
    if answer_tokens is None:
        max_prompt_tokens = max(1, max_length - min_answer_tokens)
    else:
        answer_tokens = max(1, int(answer_tokens))
        max_prompt_tokens = max(1, max_length - answer_tokens)

    # Left-truncation preserves the current problem when few-shot examples make
    # a prompt too long for the fixed GSM8K context.
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


def _decode_tokens(model: SamplingModel, token_ids: torch.Tensor) -> str:
    return model.tokenizer.decode(token_ids.detach().cpu().tolist(), skip_special_tokens=True).strip()


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


def _primary_prediction(strict: Optional[str], flexible: Optional[str], mode: str) -> Optional[str]:
    mode = str(mode or "strict").lower()
    if mode in {"strict", "exact", "paper"}:
        return strict
    if mode in {"flex", "flexible", "last_number"}:
        return flexible
    raise ValueError(f"Unknown GSM8K primary_extraction: {mode}")


@torch.no_grad()
def evaluate_records(
    records: list[GSM8KRecord],
    fewshot_records: list[GSM8KRecord],
    model: SamplingModel,
    sampler: OnlineTokenPromotionSampler,
    eval_cfg: Mapping[str, Any],
    seed: int,
):
    generator_device = model.device if model.device.type == "cuda" else torch.device("cpu")
    generator = torch.Generator(device=generator_device)
    generator.manual_seed(int(seed))

    batch_size = max(1, int(eval_cfg.get("batch_size", 1) or 1))
    answer_tokens = _optional_positive_int(eval_cfg.get("answer_max_tokens"))
    max_length = int(eval_cfg.get("max_sequence_length") or model.sampling_config.max_length)
    pad_to_max_length = bool(eval_cfg.get("pad_to_max_length", True))
    min_answer_tokens = max(1, int(eval_cfg.get("min_answer_tokens", 1) or 1))
    primary_mode = str(eval_cfg.get("primary_extraction", "strict") or "strict")
    progress_enabled = bool(eval_cfg.get("progress", True))

    try:
        from tqdm.auto import tqdm
    except ImportError:
        tqdm = None
    iterator = range(0, len(records), batch_size)
    if progress_enabled and tqdm is not None:
        iterator = tqdm(iterator, desc="GSM8K", unit="batch")

    rows = []
    for start in iterator:
        chunk = records[start : start + batch_size]
        prompts = [build_prompt(record, eval_cfg, fewshot_records) for record in chunk]
        batch, corrupt_mask, spans = encode_batch(
            model,
            prompts,
            answer_tokens=answer_tokens,
            max_length=max_length,
            pad_to_max_length=pad_to_max_length,
            min_answer_tokens=min_answer_tokens,
        )
        result = sampler.sample_batch(batch, corrupt_mask, generator)
        for row_idx, record in enumerate(chunk):
            answer_start, answer_end = spans[row_idx]
            completion = _decode_tokens(model, result.generated_ids[row_idx, answer_start:answer_end])
            sample = model.tokenizer.decode(
                result.generated_ids[row_idx, :answer_end].detach().cpu().tolist(),
                skip_special_tokens=True,
            ).strip()
            strict = extract_strict_answer(completion)
            flexible = extract_flexible_answer(completion)
            prediction = _primary_prediction(strict, flexible, primary_mode)
            rows.append(
                {
                    "index": record.index,
                    "id": record.record_id,
                    "question": record.question,
                    "answer": record.answer,
                    "gold": record.gold_answer,
                    "prediction": prediction,
                    "strict_prediction": strict,
                    "flex_prediction": flexible,
                    "correct": bool(prediction is not None and prediction == record.gold_answer),
                    "strict_match": bool(strict is not None and strict == record.gold_answer),
                    "flex_match": bool(flexible is not None and flexible == record.gold_answer),
                    "prompt": prompts[row_idx],
                    "completion": completion,
                    "sample": sample,
                    "prompt_tokens": int(answer_start),
                    "generated_tokens": int(answer_end - answer_start),
                    "promotion_steps": _history_json(result.history),
                }
            )
    return rows


def main():
    args = parse_args()
    eval_file_cfg = _load_eval_config(args.config)
    model_cfg = dict(eval_file_cfg.get("model", {}) or {})
    eval_cfg = dict(eval_file_cfg.get("evaluation", {}) or {})

    training_config_path = args.training_config or model_cfg.get("training_config")
    if not training_config_path:
        raise ValueError("Set model.training_config in the eval config or pass --training_config.")
    training_config = load_config_from_yaml(_repo_path(str(training_config_path)) or str(training_config_path))
    config_overrides = list(model_cfg.get("config_overrides", []) or []) + list(args.config_override or [])
    if config_overrides:
        training_config = apply_config_overrides(training_config, config_overrides)
    training_config = fill_mlfm_defaults(training_config)

    if args.seed is not None:
        eval_cfg["seed"] = int(args.seed)
        training_config.seed = int(args.seed)
    seed = int(eval_cfg.get("seed", getattr(training_config, "seed", 0)) or 0)
    training_config.seed = seed

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
    if args.subset_indices_file is not None:
        eval_cfg["subset_indices_file"] = args.subset_indices_file
    if args.num_fewshots is not None:
        eval_cfg["num_fewshots"] = int(args.num_fewshots)
    if args.output_jsonl is not None:
        eval_cfg["output_jsonl"] = args.output_jsonl
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

    records = load_gsm8k_records(eval_cfg, args.dataset_path)
    num_fewshots = max(0, int(eval_cfg.get("num_fewshots", 0) or 0))
    if num_fewshots >= len(records):
        raise ValueError(f"num_fewshots={num_fewshots} leaves no GSM8K examples to evaluate.")
    fewshot_records = records[:num_fewshots]
    eval_records = records[num_fewshots:]
    eval_records = select_eval_records(eval_records, eval_cfg)

    model = SamplingModel.from_config_checkpoint(training_config, sampling_config, checkpoint, device=device)
    sampler = OnlineTokenPromotionSampler(model)
    rows = evaluate_records(eval_records, fewshot_records, model, sampler, eval_cfg, seed)
    rows.sort(key=lambda item: item["index"])

    total = len(rows)
    correct = sum(1 for row in rows if row["correct"])
    correct_strict = sum(1 for row in rows if row["strict_match"])
    correct_flex = sum(1 for row in rows if row["flex_match"])
    no_answer = sum(1 for row in rows if row["prediction"] is None)
    strict_no_answer = sum(1 for row in rows if row["strict_prediction"] is None)
    flex_no_answer = sum(1 for row in rows if row["flex_prediction"] is None)
    metrics = {
        "task": "gsm8k",
        "dataset": {
            "name": str(eval_cfg.get("dataset_name", "openai/gsm8k")),
            "config": eval_cfg.get("dataset_config", "main"),
            "split": str(eval_cfg.get("split", "test")),
            "path": args.dataset_path or eval_cfg.get("dataset_path"),
        },
        "training_config": str(training_config_path),
        "checkpoint": checkpoint,
        "checkpoint_step": int(model.checkpoint_step),
        "adapter_mode": adapter_mode,
        "total": total,
        "correct": correct,
        "no_answer": no_answer,
        "accuracy": correct / max(total, 1),
        "primary_extraction": str(eval_cfg.get("primary_extraction", "strict") or "strict"),
        "correct_strict": correct_strict,
        "correct_flex": correct_flex,
        "strict_no_answer": strict_no_answer,
        "flex_no_answer": flex_no_answer,
        "strict_accuracy": correct_strict / max(total, 1),
        "flex_accuracy": correct_flex / max(total, 1),
        "num_fewshots": num_fewshots,
        "seed": seed,
        "evaluation": eval_cfg,
        "sampling": sampling_config.to_dict(),
    }

    output_jsonl = eval_cfg.get("output_jsonl") or "outputs/evals/gsm8k/predictions.jsonl"
    output_path = Path(_repo_path(str(output_jsonl)) or str(output_jsonl))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    metrics_path = output_path.with_name(f"{output_path.stem}_metrics.json")
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(metrics, sort_keys=True))


if __name__ == "__main__":
    main()
