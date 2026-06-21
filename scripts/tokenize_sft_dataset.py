#!/usr/bin/env python
"""Pretokenize prompt/response SFT datasets for MLFM."""

from __future__ import annotations

import argparse
import ast
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, Iterator, Optional

from datasets import Dataset, Features, Sequence, Value, load_dataset
from huggingface_hub.errors import EntryNotFoundError
from transformers import AutoTokenizer


_SKIPPED_TOKENIZATION_WARNINGS = 0


def parse_args():
    parser = argparse.ArgumentParser(description="Tokenize prompt/response SFT data into save_to_disk Arrow.")
    parser.add_argument(
        "--format",
        required=True,
        choices=["sharegpt", "gsm8k_aug", "metamathqa", "math_reasoning", "code", "code_instruction"],
        help="Input conversion format.",
    )
    parser.add_argument("--dataset_name", default=None, help="HF dataset name/path for datasets.load_dataset.")
    parser.add_argument("--dataset_config", default=None, help="Optional HF dataset config.")
    parser.add_argument("--dataset_file", default=None, help="Optional file inside an HF dataset repo to download and parse locally.")
    parser.add_argument("--split", default="train", help="HF split.")
    parser.add_argument("--input_path", default=None, help="Local json/jsonl/txt input path.")
    parser.add_argument("--output_dir", required=True, help="save_to_disk output directory.")
    parser.add_argument("--tokenizer_name_or_path", required=True)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--limit", type=int, default=0, help="Optional max number of emitted examples.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prompt_field", default="prompt")
    parser.add_argument("--response_field", default="response")
    parser.add_argument("--question_field", default="question")
    parser.add_argument("--answer_field", default="answer")
    parser.add_argument("--code_field", default=None, help="Code text column; auto-detected when omitted.")
    parser.add_argument("--code_windows_per_doc", type=int, default=1)
    parser.add_argument("--code_min_response_tokens", type=int, default=64)
    parser.add_argument("--code_prefix_min_frac", type=float, default=0.25)
    parser.add_argument("--code_prefix_max_frac", type=float, default=0.75)
    parser.add_argument("--code_min_unit_tests", type=int, default=1)
    parser.add_argument("--code_max_unit_tests", type=int, default=3)
    parser.add_argument("--code_min_average_test_score", type=float, default=None)
    parser.add_argument("--code_require_all_tests_pass", action="store_true")
    parser.add_argument("--code_allowed_domains", default="", help="Comma-separated OpenCodeInstruct domains to keep.")
    parser.add_argument(
        "--code_allowed_generation_algorithms",
        default="",
        help="Comma-separated OpenCodeInstruct generation_algorithm values to keep.",
    )
    parser.add_argument("--code_max_prompt_tokens", type=int, default=0)
    parser.add_argument("--code_max_response_tokens", type=int, default=0)
    parser.add_argument("--code_max_true_tokens", type=int, default=0)
    parser.add_argument("--code_min_instruction_response_tokens", type=int, default=0)
    parser.add_argument("--no_add_special_tokens", action="store_true", help="Disable tokenizer special tokens for prompt/response pieces.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip_existing", action="store_true", help="Exit successfully if output_dir already exists.")
    return parser.parse_args()


def _load_local_json_records(path: str) -> Iterator[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        if path.endswith(".jsonl"):
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)
        else:
            data = json.load(f)
            if isinstance(data, dict):
                data = data.get("data", data.get("instances", []))
            for item in data:
                yield item


def _load_records(args) -> Iterable:
    if args.input_path:
        if args.format == "gsm8k_aug" and args.input_path.endswith(".txt"):
            with open(args.input_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        yield line
            return
        yield from _load_local_json_records(args.input_path)
        return
    if not args.dataset_name:
        raise ValueError("Either --dataset_name or --input_path is required.")
    if args.dataset_file:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise ImportError("huggingface_hub is required for --dataset_file.") from exc

        filenames = [item.strip() for item in str(args.dataset_file).split(",") if item.strip()]
        last_error = None
        path = None
        for filename in filenames:
            try:
                path = hf_hub_download(repo_id=args.dataset_name, filename=filename, repo_type="dataset")
                break
            except EntryNotFoundError as exc:
                last_error = exc
        if path is None:
            raise last_error or FileNotFoundError(f"No dataset_file entries found for {args.dataset_name}: {args.dataset_file}")
        if args.format == "gsm8k_aug" and path.endswith(".txt"):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        yield line
            return
        yield from _load_local_json_records(path)
        return
    kwargs = {"split": args.split}
    if args.dataset_config:
        dataset = load_dataset(args.dataset_name, args.dataset_config, **kwargs)
    else:
        dataset = load_dataset(args.dataset_name, **kwargs)
    for row in dataset:
        yield row


def _coerce_text(text) -> str:
    if text is None:
        return ""
    if isinstance(text, bytes):
        value = text.decode("utf-8", errors="replace")
    elif isinstance(text, (list, tuple)):
        value = "\n".join(_coerce_text(item) for item in text)
    elif isinstance(text, dict):
        value = json.dumps(text, ensure_ascii=False, sort_keys=True)
    else:
        value = str(text)
    return value.encode("utf-8", errors="replace").decode("utf-8", errors="replace").replace("\x00", " ")


def _tokenize(tokenizer, text: str, add_special_tokens: bool) -> list[int]:
    value = _coerce_text(text)
    return tokenizer(value, add_special_tokens=bool(add_special_tokens))["input_ids"]


def _warn_skipped_tokenization(format_name: str, row_idx: int, exc: Exception) -> None:
    global _SKIPPED_TOKENIZATION_WARNINGS
    if _SKIPPED_TOKENIZATION_WARNINGS < 20:
        print(
            f"[tokenize_sft_dataset] skip {format_name} row {row_idx}: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )
    elif _SKIPPED_TOKENIZATION_WARNINGS == 20:
        print(
            "[tokenize_sft_dataset] further tokenization skip warnings suppressed",
            file=sys.stderr,
            flush=True,
        )
    _SKIPPED_TOKENIZATION_WARNINGS += 1


def _chat_prompt(prompt: str) -> str:
    return f"USER:\n{str(prompt).strip()}\nASSISTANT:\n"


def _vicuna_prompt(prompt: str) -> str:
    return _chat_prompt(prompt)


def _append_eos(ids: list[int], eos_token_id: Optional[int]) -> list[int]:
    if eos_token_id is None:
        return ids
    return list(ids) + [int(eos_token_id)]


def _math_prompt(question: str) -> str:
    return _chat_prompt(question)


def _math_target(reasoning: str, answer: Optional[str] = None) -> str:
    target = str(reasoning).strip()
    if answer is not None and str(answer).strip():
        target = f"{target}\n### {str(answer).strip()}" if target else f"### {str(answer).strip()}"
    return target.replace("####", "###")


def _make_example(
    prompt_ids: list[int],
    response_ids: list[int],
    eos_token_id: Optional[int],
    max_length: int,
    source_type: str,
):
    response_ids = _append_eos(response_ids, eos_token_id)
    true_length = len(prompt_ids) + len(response_ids)
    if len(prompt_ids) <= 0 or len(response_ids) <= 0 or true_length > max_length:
        return None
    pad_id = int(eos_token_id) if eos_token_id is not None else 0
    input_ids = list(prompt_ids) + list(response_ids) + [pad_id] * (max_length - true_length)
    return {
        "input_ids": input_ids,
        "prompt_length": int(len(prompt_ids)),
        "true_length": int(true_length),
        "source_type": str(source_type),
    }


def _sharegpt_examples(args, tokenizer) -> Iterator[Dict]:
    add_special = not args.no_add_special_tokens
    for row in _load_records(args):
        conversations = row.get("conversations", row.get("messages", []))
        if len(conversations) < 2:
            continue
        first, second = conversations[0], conversations[1]
        first_role = first.get("from", first.get("role"))
        second_role = second.get("from", second.get("role"))
        if first_role not in {"human", "user"} or second_role not in {"gpt", "assistant"}:
            continue
        prompt = first.get("value", first.get("content", ""))
        response = second.get("value", second.get("content", ""))
        example = _make_example(
            _tokenize(tokenizer, _vicuna_prompt(prompt), add_special),
            _tokenize(tokenizer, response, add_special),
            tokenizer.eos_token_id,
            args.max_length,
            "general",
        )
        if example is not None:
            yield example


def _gsm8k_aug_examples(args, tokenizer) -> Iterator[Dict]:
    add_special = not args.no_add_special_tokens
    for row in _load_records(args):
        if isinstance(row, str):
            if "||" not in row or "####" not in row:
                continue
            question, rest = row.split("||", 1)
            thought, answer = rest.split("####", 1)
            pairs = [(_math_prompt(question), _math_target(thought, answer))]
        else:
            question = row.get(args.question_field, row.get("query", ""))
            answer = str(row.get(args.answer_field, ""))
            steps = row.get("steps")
            if isinstance(steps, list) and steps:
                thought = "\n".join(str(step) for step in steps if str(step).strip())
                if not thought.strip() or not answer.strip():
                    continue
                pairs = [(_math_prompt(question), _math_target(thought, answer))]
            else:
                response = str(row.get("response", row.get("solution", answer)))
                if "####" not in response:
                    continue
                thought, final_answer = response.split("####", 1)
                if not thought.strip() or not final_answer.strip():
                    continue
                pairs = [(_math_prompt(question), _math_target(thought, final_answer))]
        for prompt, response in pairs:
            example = _make_example(
                _tokenize(tokenizer, prompt, add_special),
                _tokenize(tokenizer, response, add_special),
                tokenizer.eos_token_id,
                args.max_length,
                "math",
            )
            if example is not None:
                yield example


def _metamathqa_examples(args, tokenizer) -> Iterator[Dict]:
    add_special = not args.no_add_special_tokens
    for row in _load_records(args):
        prompt = row.get("query", row.get(args.question_field, row.get(args.prompt_field, "")))
        response = row.get("response", row.get(args.response_field, row.get("solution", row.get("output", ""))))
        answer = row.get(args.answer_field, row.get("final_answer", ""))
        if not str(response).strip() and str(answer).strip():
            response = answer
            answer = ""
        if str(answer).strip() == str(response).strip():
            answer = ""
        if not str(prompt).strip() or not str(response).strip():
            continue
        example = _make_example(
            _tokenize(tokenizer, _math_prompt(prompt), add_special),
            _tokenize(tokenizer, _math_target(response, answer), add_special),
            tokenizer.eos_token_id,
            args.max_length,
            "math",
        )
        if example is not None:
            yield example


def _math_reasoning_text(row: Dict, preferred_field: str, fallback_fields: tuple[str, ...]) -> str:
    if preferred_field and preferred_field in row and row[preferred_field] is not None:
        return str(row[preferred_field])
    for field in fallback_fields:
        if field in row and row[field] is not None:
            return str(row[field])
    messages = row.get("messages")
    if isinstance(messages, list) and len(messages) >= 2:
        for message in messages:
            role = str(message.get("role", message.get("from", ""))).lower()
            if role in {"assistant", "gpt"} and "solution" in fallback_fields:
                return str(message.get("content", message.get("value", "")))
            if role in {"user", "human"} and "problem" in fallback_fields:
                return str(message.get("content", message.get("value", "")))
    return ""


def _math_reasoning_examples(args, tokenizer) -> Iterator[Dict]:
    """Generic prompt/solution math SFT rows with explicit reasoning traces."""

    add_special = not args.no_add_special_tokens
    for row in _load_records(args):
        prompt = _math_reasoning_text(
            row,
            args.question_field,
            ("question", "query", "problem", "prompt", "instruction"),
        )
        response = _math_reasoning_text(
            row,
            args.response_field,
            ("generated_solution", "solution", "response", "output"),
        )
        answer = _math_reasoning_text(row, args.answer_field, ("answer", "final_answer", "target"))
        if answer.strip() == response.strip():
            answer = ""
        if not prompt.strip() or not response.strip():
            continue
        example = _make_example(
            _tokenize(tokenizer, _math_prompt(prompt), add_special),
            _tokenize(tokenizer, _math_target(response, answer), add_special),
            tokenizer.eos_token_id,
            args.max_length,
            "math",
        )
        if example is not None:
            yield example


def _code_text(row: Dict, preferred_field: Optional[str]) -> str:
    if preferred_field and preferred_field in row:
        return str(row[preferred_field])
    for field in ("code", "whole_func_string", "func_code_string", "content", "text"):
        if field in row and row[field]:
            return str(row[field])
    return ""


def _strip_after_markdown_input(text: str) -> str:
    text = str(text)
    markers = ("**Input**", "**Input:**")
    cut = None
    for marker in markers:
        idx = text.find(marker)
        if idx >= 0:
            cut = idx if cut is None else min(cut, idx)
    if cut is not None:
        text = text[:cut]
    return text.strip()


def _parse_unit_tests(value) -> list[str]:
    if value is None:
        return []
    parsed = value
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return []
        try:
            parsed = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                return [line.strip() for line in value.splitlines() if line.strip()]
    if isinstance(parsed, str):
        parsed = [parsed]
    if not isinstance(parsed, (list, tuple)):
        return []
    return [str(test).strip() for test in parsed if str(test).strip()]


def _parse_string_list(value) -> list[str]:
    if value is None:
        return []
    parsed = value
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return []
        try:
            parsed = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                parsed = [part.strip() for part in value.split(",") if part.strip()]
    if isinstance(parsed, str):
        parsed = [parsed]
    if not isinstance(parsed, (list, tuple)):
        return []
    return [str(item).strip() for item in parsed if str(item).strip()]


def _csv_set(value: str) -> set[str]:
    return {item.strip().lower() for item in str(value or "").split(",") if item.strip()}


def _code_instruction_row_passes_filters(args, row: Dict) -> bool:
    domains = _csv_set(getattr(args, "code_allowed_domains", ""))
    if domains:
        domain = str(row.get("domain", "")).strip().lower()
        if domain not in domains:
            return False

    algorithms = _csv_set(getattr(args, "code_allowed_generation_algorithms", ""))
    if algorithms:
        algorithm = str(row.get("generation_algorithm", "")).strip().lower()
        if algorithm not in algorithms:
            return False

    min_score = getattr(args, "code_min_average_test_score", None)
    if min_score is not None:
        try:
            score = float(row.get("average_test_score", 0.0))
        except (TypeError, ValueError):
            return False
        if score < float(min_score):
            return False

    if bool(getattr(args, "code_require_all_tests_pass", False)):
        statuses = _parse_string_list(row.get("tests_execution_status"))
        if not statuses or any(status.strip().lower() != "pass" for status in statuses):
            return False

    return True


def _sample_unit_tests(
    tests: list[str],
    rng: random.Random,
    min_tests: int = 1,
    max_tests: int = 3,
) -> list[str]:
    if not tests:
        return []
    max_tests = max(0, int(max_tests))
    min_tests = max(0, int(min_tests))
    if max_tests <= 0:
        return []
    min_tests = min(min_tests, max_tests, len(tests))
    max_tests = min(max_tests, len(tests))
    if max_tests <= 0:
        return []
    count = rng.randint(min_tests, max_tests) if max_tests > min_tests else max_tests
    return list(tests[:count])


def _code_instruction_prompt(instruction: str, unit_tests: Optional[list[str]] = None) -> str:
    prompt = str(_strip_after_markdown_input(instruction))
    tests = [str(test).strip() for test in (unit_tests or []) if str(test).strip()]
    if tests:
        prompt += "\n\n"
        if len(tests) == 1:
            prompt += "Code should pass this test:\n"
        else:
            prompt += "Code should pass these tests:\n"
        prompt += "\n".join(tests)
    return _chat_prompt(prompt)


def _strip_surrounding_code_fence(text: str) -> str:
    text = str(text).strip()
    fenced = re.search(r"```(?:python)?\s*\n(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        return fenced.group(1).strip()
    lines = text.splitlines()
    if lines and lines[0].strip().startswith("```"):
        body = []
        for line in lines[1:]:
            if line.strip().startswith("```"):
                break
            body.append(line)
        if body:
            return "\n".join(body).strip()
    return text


def _code_instruction_target(response: str) -> str:
    code = _strip_surrounding_code_fence(_strip_after_markdown_input(response))
    code = code.strip()
    if not code:
        return ""
    return f"```\n{code}\n```"


def _code_examples(args, tokenizer) -> Iterator[Dict]:
    add_special = not args.no_add_special_tokens
    rng = random.Random(int(args.seed))
    min_frac = float(args.code_prefix_min_frac)
    max_frac = float(args.code_prefix_max_frac)
    if not 0.0 < min_frac <= max_frac < 1.0:
        raise ValueError("Expected 0 < code_prefix_min_frac <= code_prefix_max_frac < 1.")
    max_content = max(2, int(args.max_length) - (1 if tokenizer.eos_token_id is not None else 0))
    min_response = max(1, int(args.code_min_response_tokens))
    for row_idx, row in enumerate(_load_records(args)):
        text = _code_text(row, args.code_field)
        if not text.strip():
            continue
        ids = _tokenize(tokenizer, text, add_special)
        if len(ids) < min_response + 2:
            continue
        for window_idx in range(max(1, int(args.code_windows_per_doc))):
            local_rng = random.Random(rng.randint(0, 2**31 - 1) + row_idx * 1009 + window_idx)
            window = ids
            if len(window) > max_content:
                start = local_rng.randint(0, len(window) - max_content)
                window = window[start : start + max_content]
            if len(window) < min_response + 2:
                continue
            low = max(1, int(len(window) * min_frac))
            high = min(len(window) - min_response, int(len(window) * max_frac))
            if high < low:
                continue
            split = local_rng.randint(low, high)
            example = _make_example(
                window[:split],
                window[split:],
                tokenizer.eos_token_id,
                args.max_length,
                "code",
            )
            if example is not None:
                yield example


def _code_instruction_examples(args, tokenizer) -> Iterator[Dict]:
    add_special = not args.no_add_special_tokens
    rng = random.Random(int(args.seed))
    for row_idx, row in enumerate(_load_records(args)):
        if not _code_instruction_row_passes_filters(args, row):
            continue
        instruction = row.get("input", row.get(args.prompt_field, row.get("instruction", row.get("question", ""))))
        response = row.get("output", row.get(args.response_field, row.get("code", "")))
        tests = _parse_unit_tests(row.get("unit_tests", row.get("tests", row.get("test_list"))))
        if not str(instruction).strip() or not str(response).strip() or not tests:
            continue
        local_rng = random.Random(rng.randint(0, 2**31 - 1) + row_idx * 1009)
        prompt_tests = _sample_unit_tests(
            tests,
            local_rng,
            min_tests=int(args.code_min_unit_tests),
            max_tests=int(args.code_max_unit_tests),
        )
        prompt = _code_instruction_prompt(instruction, prompt_tests)
        target = _code_instruction_target(response)
        if not target.strip():
            continue
        try:
            prompt_ids = _tokenize(tokenizer, prompt, add_special)
            target_ids = _tokenize(tokenizer, target, add_special)
        except (TypeError, ValueError, UnicodeError) as exc:
            _warn_skipped_tokenization("code_instruction", row_idx, exc)
            continue
        if int(getattr(args, "code_max_prompt_tokens", 0) or 0) > 0 and len(prompt_ids) > int(args.code_max_prompt_tokens):
            continue
        if int(getattr(args, "code_max_response_tokens", 0) or 0) > 0 and len(target_ids) > int(args.code_max_response_tokens):
            continue
        if (
            int(getattr(args, "code_min_instruction_response_tokens", 0) or 0) > 0
            and len(target_ids) < int(args.code_min_instruction_response_tokens)
        ):
            continue
        if int(getattr(args, "code_max_true_tokens", 0) or 0) > 0:
            extra_eos = 1 if tokenizer.eos_token_id is not None else 0
            if len(prompt_ids) + len(target_ids) + extra_eos > int(args.code_max_true_tokens):
                continue
        example = _make_example(
            prompt_ids,
            target_ids,
            tokenizer.eos_token_id,
            args.max_length,
            "code",
        )
        if example is not None:
            yield example


def _example_iter(args, tokenizer) -> Iterator[Dict]:
    if args.format == "sharegpt":
        iterator = _sharegpt_examples(args, tokenizer)
    elif args.format == "gsm8k_aug":
        iterator = _gsm8k_aug_examples(args, tokenizer)
    elif args.format == "metamathqa":
        iterator = _metamathqa_examples(args, tokenizer)
    elif args.format == "math_reasoning":
        iterator = _math_reasoning_examples(args, tokenizer)
    elif args.format == "code":
        iterator = _code_examples(args, tokenizer)
    elif args.format == "code_instruction":
        iterator = _code_instruction_examples(args, tokenizer)
    else:
        raise ValueError(f"Unknown format: {args.format}")

    emitted = 0
    for example in iterator:
        yield example
        emitted += 1
        if args.limit and emitted >= int(args.limit):
            break


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        if args.skip_existing and not args.overwrite:
            print(json.dumps({"output_dir": str(output_dir), "skipped_existing": True}, sort_keys=True))
            return
        if not args.overwrite:
            raise FileExistsError(f"Output already exists: {output_dir}. Use --overwrite to replace it.")
        import shutil

        shutil.rmtree(output_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    features = Features(
        {
            "input_ids": Sequence(Value("int64")),
            "prompt_length": Value("int64"),
            "true_length": Value("int64"),
            "source_type": Value("string"),
        }
    )
    dataset = Dataset.from_generator(lambda: _example_iter(args, tokenizer), features=features)
    if len(dataset) == 0:
        raise RuntimeError("No SFT examples were emitted.")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(output_dir))
    metadata = {
        "format": args.format,
        "dataset_name": args.dataset_name,
        "dataset_config": args.dataset_config,
        "dataset_file": args.dataset_file,
        "split": args.split,
        "input_path": args.input_path,
        "tokenizer_name_or_path": args.tokenizer_name_or_path,
        "max_length": args.max_length,
        "num_examples": len(dataset),
    }
    with open(output_dir / "sft_manifest.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)
        f.write("\n")
    print(json.dumps(metadata, sort_keys=True))


if __name__ == "__main__":
    main()
