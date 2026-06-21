#!/usr/bin/env python
"""Pretokenize and fixed-block pack text datasets for MLFM.

The pipeline is intentionally two-stage:
1. Tokenize raw documents into resumable document shards.
2. Pack tokenized shards into fixed-length block shards with carry across shards.

The committed output remains a normal Hugging Face Dataset saved with
`save_to_disk`, containing only `input_ids` and `attention_mask`.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from datasets import Dataset, Features, Value, concatenate_datasets, load_dataset, load_from_disk
from transformers import AutoTokenizer


FORMAT_VERSION = 1
SUCCESS_MARKER = "_SUCCESS"
SCRIPT_VERSION = "slimpajama-loader-v5"
PROOF_PILE_2_DATASET = "EleutherAI/proof-pile-2"
PROOF_PILE_2_BASE_URL = "https://huggingface.co/datasets/EleutherAI/proof-pile-2/resolve/main"
SLIMPAJAMA_DATASET = "cerebras/SlimPajama-627B"

_PROOF_PILE_2_ARXIV_FILES = {
    "train": [f"arXiv_{i:03}.jsonl.zst" for i in range(100)],
    "validation": [f"arXiv_{i:03}.jsonl.zst" for i in range(100)],
    "test": [f"arXiv_{i:03}.jsonl.zst" for i in range(100)],
}
_PROOF_PILE_2_OWM_FILES = {
    "train": [f"shard-{i:04}.jsonl.zst" for i in range(63)],
    "validation": ["val.jsonl.zst"],
    "test": ["test.jsonl.zst"],
}
_PROOF_PILE_2_ALGEBRAIC_STACK_FILES = {
    "train": (
        ["agda0000.jsonl.zst", "c0000.jsonl.zst"]
        + [f"cpp{i:04}.jsonl.zst" for i in range(5)]
        + [f"fortran{i:04}.jsonl.zst" for i in range(4)]
        + ["gap0000.jsonl.zst"]
        + [f"github-MATLAB-train-{i:04}.jsonl.zst" for i in range(4)]
        + [f"github-coq-train-{i:04}.jsonl.zst" for i in range(3)]
        + ["github-isabelle-train-0000.jsonl.zst", "github-lean-train-0000.jsonl.zst"]
        + ["haskell0000.jsonl.zst", "idris0000.jsonl.zst", "isa_proofsteps.jsonl.zst"]
        + [f"julia{i:04}.jsonl.zst" for i in range(6)]
        + ["jupyter-notebook0000.jsonl.zst", "lean_proofsteps.jsonl.zst", "maple0000.jsonl.zst"]
        + [f"python{i:04}.jsonl.zst" for i in range(42)]
        + ["r0000.jsonl.zst"]
        + [f"tex{i:04}.jsonl.zst" for i in range(3)]
    ),
    "validation": [
        "agda-validation.jsonl.zst",
        "c-validation.jsonl.zst",
        "cpp-validation.jsonl.zst",
        "fortran-validation.jsonl.zst",
        "gap-validation.jsonl.zst",
        "github-MATLAB-validation-0000.jsonl.zst",
        "github-coq-validation-0000.jsonl.zst",
        "github-isabelle-validation-0000.jsonl.zst",
        "github-lean-validation-0000.jsonl.zst",
        "haskell-validation.jsonl.zst",
        "idris-validation.jsonl.zst",
        "isa_proofsteps.jsonl.zst",
        "julia-validation.jsonl.zst",
        "jupyter-notebook-validation.jsonl.zst",
        "lean_proofsteps.jsonl.zst",
        "maple-validation.jsonl.zst",
        "python-validation.jsonl.zst",
        "r-validation.jsonl.zst",
        "tex-validation.jsonl.zst",
    ],
    "test": [
        "agda-test.jsonl.zst",
        "c-test.jsonl.zst",
        "cpp-test.jsonl.zst",
        "fortran-test.jsonl.zst",
        "gap-test.jsonl.zst",
        "github-MATLAB-test-0000.jsonl.zst",
        "github-coq-test-0000.jsonl.zst",
        "github-isabelle-test-0000.jsonl.zst",
        "github-lean-test-0000.jsonl.zst",
        "haskell-test.jsonl.zst",
        "idris-test.jsonl.zst",
        "isa_proofsteps.jsonl.zst",
        "julia-test.jsonl.zst",
        "jupyter-notebook-test.jsonl.zst",
        "lean_proofsteps.jsonl.zst",
        "maple-test.jsonl.zst",
        "python-test.jsonl.zst",
        "r-test.jsonl.zst",
        "tex-test.jsonl.zst",
    ],
}
_PROOF_PILE_2_FILES = {
    "arxiv": _PROOF_PILE_2_ARXIV_FILES,
    "open-web-math": _PROOF_PILE_2_OWM_FILES,
    "algebraic-stack": _PROOF_PILE_2_ALGEBRAIC_STACK_FILES,
}
_PROOF_PILE_2_CONFIG_SUBSETS = {
    None: ["arxiv", "open-web-math", "algebraic-stack"],
    "default": ["arxiv", "open-web-math", "algebraic-stack"],
    "arxiv": ["arxiv"],
    "open-web-math": ["open-web-math"],
    "algebraic-stack": ["algebraic-stack"],
}


def _log(message: str):
    print(message, flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Tokenize and pack a text dataset into fixed-length blocks.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {SCRIPT_VERSION}")
    parser.add_argument("--dataset_name", required=True, help="HF dataset name or local dataset script/path.")
    parser.add_argument("--dataset_config", default=None, help="Optional HF dataset config name.")
    parser.add_argument("--split", default="train", help="Dataset split to tokenize.")
    parser.add_argument(
        "--data_files",
        default=None,
        help="Optional comma-separated data files/patterns forwarded to datasets.load_dataset.",
    )
    parser.add_argument("--text_column", default="text", help="Column containing raw text.")
    parser.add_argument("--tokenizer_name_or_path", required=True, help="Tokenizer matching the target backbone.")
    parser.add_argument("--output_dir", required=True, help="Directory for datasets.save_to_disk output.")
    parser.add_argument("--max_length", type=int, default=2048, help="Packed block length.")
    parser.add_argument("--num_proc", type=int, default=1, help="Parallel tokenizer map processes per shard.")
    parser.add_argument("--append_eos", action="store_true", help="Append EOS between documents when available.")
    parser.add_argument("--resume", action="store_true", help="Resume from completed work shards.")
    parser.add_argument(
        "--resume_skip_raw_if_tokenized_complete",
        action="store_true",
        help=(
            "In resume mode, skip load_dataset/HF cache work when the existing work manifest is compatible "
            "and every tokenized shard is already complete. Safe for resuming packing/commit only."
        ),
    )
    parser.add_argument("--overwrite", action="store_true", help="Delete existing output/work state and start clean.")
    parser.add_argument("--work_dir", default=None, help="Scratch directory for tokenized and packed shards.")
    parser.add_argument("--docs_per_shard", type=int, default=100000, help="Raw documents per tokenized shard.")
    parser.add_argument("--packed_blocks_per_shard", type=int, default=10000, help="Packed blocks per output shard.")
    parser.add_argument(
        "--max_documents",
        type=int,
        default=0,
        help="Optional cap on raw documents after loading. 0 means use all loaded documents.",
    )
    parser.add_argument(
        "--max_files",
        type=int,
        default=0,
        help="Optional cap on matched input files for file-backed loaders such as SlimPajama. 0 means all files.",
    )
    parser.add_argument(
        "--slimpajama_chunks",
        default=None,
        help="For cerebras/SlimPajama-627B, comma/range selector such as '1', '1,2,4', or '1-3'.",
    )
    parser.add_argument("--keep_intermediate", action="store_true", help="Keep tokenized shards after final commit.")
    return parser.parse_args()


def _read_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json_atomic(path: Path, payload: Dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp_path, path)


def _remove_path(path: Path):
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def _rename_dir_atomic(src: Path, dst: Path):
    if dst.exists():
        raise FileExistsError(f"Cannot atomically rename to existing path: {dst}")
    src.rename(dst)


@dataclass
class ShardedTextDataset:
    dataset: object
    text_column: str
    docs_per_shard: int

    def __post_init__(self):
        if self.docs_per_shard <= 0:
            raise ValueError("`docs_per_shard` must be positive.")
        if self.text_column not in self.dataset.column_names:
            raise ValueError(
                f"Text column {self.text_column!r} not found. Available columns: {self.dataset.column_names}"
            )
        try:
            self.num_documents = len(self.dataset)
        except TypeError as exc:
            raise ValueError("Resumable tokenization requires a non-streaming dataset with known length.") from exc
        self.num_shards = int(math.ceil(self.num_documents / float(self.docs_per_shard)))

    def shard_range(self, shard_id: int) -> Tuple[int, int]:
        if shard_id < 0 or shard_id >= self.num_shards:
            raise IndexError(f"Shard {shard_id} out of range for {self.num_shards} shards.")
        start = shard_id * self.docs_per_shard
        end = min(start + self.docs_per_shard, self.num_documents)
        return start, end

    def select_shard(self, shard_id: int):
        start, end = self.shard_range(shard_id)
        return self.dataset.select(range(start, end))


class ShardStore:
    """Durable HF Dataset shard storage with `.done.json` resume markers."""

    def __init__(self, base_dir: Path, kind: str):
        self.base_dir = Path(base_dir)
        self.kind = kind

    def shard_name(self, shard_id: int) -> str:
        return f"shard_{int(shard_id):06d}"

    def shard_path(self, shard_id: int) -> Path:
        return self.base_dir / self.shard_name(shard_id)

    def tmp_path(self, shard_id: int) -> Path:
        return self.base_dir / f"{self.shard_name(shard_id)}.tmp"

    def done_path(self, shard_id: int) -> Path:
        return self.base_dir / f"{self.shard_name(shard_id)}.done.json"

    def cleanup_tmp(self):
        if not self.base_dir.exists():
            return
        for path in self.base_dir.glob("*.tmp"):
            _remove_path(path)
        for path in self.base_dir.glob("*.done.json.tmp"):
            _remove_path(path)

    def read_done(self, shard_id: int) -> Optional[Dict]:
        done_path = self.done_path(shard_id)
        if not done_path.exists():
            return None
        return _read_json(done_path)

    def is_done(self, shard_id: int) -> bool:
        metadata = self.read_done(shard_id)
        return bool(
            metadata
            and metadata.get("kind") == self.kind
            and int(metadata.get("shard_id", -1)) == int(shard_id)
            and self.shard_path(shard_id).is_dir()
        )

    def completed_ids(self) -> List[int]:
        if not self.base_dir.exists():
            return []
        ids = []
        for done_path in self.base_dir.glob("shard_*.done.json"):
            stem = done_path.name.split(".")[0]
            try:
                shard_id = int(stem.split("_")[1])
            except (IndexError, ValueError):
                continue
            if self.is_done(shard_id):
                ids.append(shard_id)
        return sorted(ids)

    def load(self, shard_id: int):
        if not self.is_done(shard_id):
            raise FileNotFoundError(f"{self.kind} shard {shard_id} is not complete.")
        return load_from_disk(str(self.shard_path(shard_id)))

    def write_atomic(self, shard_id: int, dataset, metadata: Dict):
        self.base_dir.mkdir(parents=True, exist_ok=True)
        final_path = self.shard_path(shard_id)
        tmp_path = self.tmp_path(shard_id)
        done_path = self.done_path(shard_id)
        _remove_path(tmp_path)
        if final_path.exists() and not self.is_done(shard_id):
            _remove_path(final_path)

        dataset.save_to_disk(str(tmp_path))
        loaded = load_from_disk(str(tmp_path))
        if len(loaded) != len(dataset):
            raise RuntimeError(f"Validation failed for {self.kind} shard {shard_id}: saved row count changed.")

        if final_path.exists():
            _remove_path(final_path)
        _rename_dir_atomic(tmp_path, final_path)
        payload = {
            "kind": self.kind,
            "shard_id": int(shard_id),
            **metadata,
        }
        _write_json_atomic(done_path, payload)


class PackedBlockBuilder:
    """Build fixed-length token blocks while carrying leftovers across shards."""

    def __init__(self, max_length: int):
        if max_length <= 0:
            raise ValueError("`max_length` must be positive.")
        self.max_length = int(max_length)
        self.carry: List[int] = []

    def add_tokens(self, token_ids: Iterable[int]) -> List[List[int]]:
        self.carry.extend(int(token_id) for token_id in token_ids)
        blocks = []
        while len(self.carry) >= self.max_length:
            blocks.append(self.carry[: self.max_length])
            del self.carry[: self.max_length]
        return blocks

    @property
    def dropped_tokens(self) -> int:
        return len(self.carry)


def _validate_args(args):
    if args.max_length <= 0:
        raise ValueError("`max_length` must be positive.")
    if args.num_proc <= 0:
        raise ValueError("`num_proc` must be positive.")
    if args.docs_per_shard <= 0:
        raise ValueError("`docs_per_shard` must be positive.")
    if args.packed_blocks_per_shard <= 0:
        raise ValueError("`packed_blocks_per_shard` must be positive.")
    if args.max_documents < 0:
        raise ValueError("`max_documents` must be non-negative.")
    if args.max_files < 0:
        raise ValueError("`max_files` must be non-negative.")
    if args.resume and args.overwrite:
        raise ValueError("Use either `--resume` or `--overwrite`, not both.")


def _split_base_name(split: str) -> str:
    return split.split("[", 1)[0]


def _split_csv(value: Optional[str]) -> Optional[List[str]]:
    if value is None:
        return None
    items = [item.strip() for item in str(value).split(",") if item.strip()]
    return items or None


def _natural_sort_key(value: str) -> list:
    return [int(piece) if piece.isdigit() else piece for piece in re.split(r"(\d+)", value)]


def _proof_pile_2_data_files(dataset_config: Optional[str], split: str) -> Tuple[str, List[str]]:
    split_name = _split_base_name(split)
    if split_name not in {"train", "validation", "test"}:
        raise ValueError(
            f"Proof-Pile-2 script-free loading supports train/validation/test splits, got {split!r}."
        )
    if dataset_config not in _PROOF_PILE_2_CONFIG_SUBSETS:
        valid = ", ".join(str(key) for key in _PROOF_PILE_2_CONFIG_SUBSETS if key is not None)
        raise ValueError(f"Unsupported Proof-Pile-2 config {dataset_config!r}. Valid configs: {valid}.")

    paths = []
    for subset in _PROOF_PILE_2_CONFIG_SUBSETS[dataset_config]:
        paths.extend(
            f"{subset}/{split_name}/{filename}"
            for filename in _PROOF_PILE_2_FILES[subset][split_name]
        )
    return split_name, paths


def _download_proof_pile_2_files(paths: List[str]) -> List[str]:
    from huggingface_hub import hf_hub_download

    token = _hf_token()
    local_files = []
    for index, path in enumerate(paths, start=1):
        print(f"[proof-pile-2] file {index}/{len(paths)} {path}", flush=True)
        local_files.append(
            hf_hub_download(
                repo_id=PROOF_PILE_2_DATASET,
                filename=path,
                repo_type="dataset",
                token=token,
            )
        )
    return local_files


def _iter_proof_pile_2_text(data_files: List[str]):
    import zstandard as zstd

    for path in data_files:
        with open(path, "rb") as raw:
            with zstd.open(raw, "rt", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    instance = json.loads(line)
                    text = instance.get("text")
                    if text is not None:
                        yield {"text": text}


def _load_proof_pile_2_without_script(args):
    try:
        import zstandard  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Proof-Pile-2 is stored as .jsonl.zst files. Install `zstandard` in this environment "
            "before tokenizing it, for example `pip install zstandard` or reinstall this package's requirements."
        ) from exc

    split_name, remote_paths = _proof_pile_2_data_files(args.dataset_config, args.split)
    print(
        f"Loading {PROOF_PILE_2_DATASET} split={split_name} through the text-only zstd generator "
        f"with {len(remote_paths)} files."
    )
    local_files = _download_proof_pile_2_files(remote_paths)
    return Dataset.from_generator(
        _iter_proof_pile_2_text,
        gen_kwargs={"data_files": local_files},
        features=Features({"text": Value("string")}),
    )


def _parse_slimpajama_chunks(selector: Optional[str]) -> Optional[set[str]]:
    items = _split_csv(selector)
    if not items:
        return None
    chunks = set()
    for item in items:
        item = item.strip().lower()
        if item.startswith("chunk"):
            item = item[len("chunk") :]
        if "-" in item:
            left, right = item.split("-", 1)
            start = int(left)
            end = int(right)
            if end < start:
                raise ValueError(f"Invalid SlimPajama chunk range {selector!r}: {start}-{end}")
            chunks.update(f"chunk{idx}" for idx in range(start, end + 1))
        else:
            chunks.add(f"chunk{int(item)}")
    return chunks


def _filter_slimpajama_repo_files(
    repo_files: Sequence[str],
    split: str,
    chunks: Optional[set[str]] = None,
    max_files: int = 0,
) -> List[str]:
    split_name = _split_base_name(split)
    if split_name != split:
        raise ValueError(
            "The script-free SlimPajama-627B loader does not support split slicing. "
            "Use --max_documents/--max_files, or use a smaller regular HF dataset such as DKYoon/SlimPajama-6B."
        )
    if split_name not in {"train", "validation", "test"}:
        raise ValueError(f"SlimPajama-627B split must be train/validation/test, got {split!r}.")
    prefix = f"{split_name}/"
    paths = [
        path
        for path in repo_files
        if path.startswith(prefix) and path.endswith(".jsonl.zst")
    ]
    if chunks:
        paths = [
            path
            for path in paths
            if any(part in chunks for part in path.split("/"))
        ]
    paths = sorted(paths, key=_natural_sort_key)
    if max_files > 0:
        paths = paths[: int(max_files)]
    return paths


def _hf_token() -> Optional[str]:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token is None:
        return None
    token = token.strip()
    return token or None


def _hf_download_token_candidates() -> List[object]:
    token = _hf_token()
    if token:
        # Try the explicit batch token first. If it is stale but the dataset is public,
        # retry anonymously instead of failing with Hugging Face's confusing repo-not-found 401.
        return [token, False]
    return [None]


def _download_hf_dataset_files(repo_id: str, paths: List[str]) -> List[str]:
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import HfHubHTTPError, RepositoryNotFoundError

    token_candidates = _hf_download_token_candidates()
    local_files = []
    for index, path in enumerate(paths, start=1):
        print(f"[{repo_id}] file {index}/{len(paths)} {path}", flush=True)
        last_exc = None
        for token in token_candidates:
            try:
                local_files.append(
                    hf_hub_download(
                        repo_id=repo_id,
                        filename=path,
                        repo_type="dataset",
                        token=token,
                    )
                )
                break
            except (RepositoryNotFoundError, HfHubHTTPError) as exc:
                last_exc = exc
                if token is False:
                    raise
                print(
                    f"[{repo_id}] explicit HF token failed for {path}; retrying anonymously.",
                    flush=True,
                )
        else:
            raise RuntimeError(
                f"Could not download {repo_id}/{path} from Hugging Face with either the explicit token "
                "or anonymous access."
            ) from last_exc
    return local_files


def _iter_jsonl_zst_text(data_files: List[str], text_column: str):
    import zstandard as zstd

    for path in data_files:
        with open(path, "rb") as raw:
            with zstd.open(raw, "rt", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    instance = json.loads(line)
                    text = instance.get(text_column)
                    if text is not None:
                        yield {text_column: str(text)}


def _load_slimpajama_without_script(args):
    try:
        import zstandard  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "SlimPajama-627B is stored as .jsonl.zst files. Install `zstandard` in this environment "
            "before tokenizing it, for example `pip install zstandard` or reinstall this package's requirements."
        ) from exc

    data_files = _split_csv(args.data_files)
    if data_files:
        local_files = data_files
        print(f"Loading {SLIMPAJAMA_DATASET} from {len(local_files)} explicit data file(s).")
    else:
        from huggingface_hub import HfApi
        from huggingface_hub.errors import HfHubHTTPError, RepositoryNotFoundError

        chunks = _parse_slimpajama_chunks(args.slimpajama_chunks)
        print(f"Listing {SLIMPAJAMA_DATASET} files for split={args.split} chunks={args.slimpajama_chunks or 'all'}")
        repo_files = None
        last_exc = None
        for token in _hf_download_token_candidates():
            try:
                repo_files = HfApi(token=token).list_repo_files(repo_id=SLIMPAJAMA_DATASET, repo_type="dataset")
                if token is False:
                    print(f"Listed {SLIMPAJAMA_DATASET} anonymously after explicit HF token failed.")
                break
            except (RepositoryNotFoundError, HfHubHTTPError) as exc:
                last_exc = exc
                if token is False:
                    break
                print(
                    f"Explicit HF token could not list {SLIMPAJAMA_DATASET}; retrying anonymously.",
                    flush=True,
                )
        if repo_files is None:
            raise RuntimeError(
                f"Could not list {SLIMPAJAMA_DATASET} files from Hugging Face. "
                "Tried the explicit HF_TOKEN/HUGGING_FACE_HUB_TOKEN when present and anonymous access. "
                "If anonymous access fails, the repository requires authentication or the environment cannot "
                "access Hugging Face. Set a valid read token, accept any dataset access terms for that account, "
                "or run `huggingface-cli login` in the job environment."
            ) from last_exc
        remote_paths = _filter_slimpajama_repo_files(
            repo_files,
            split=args.split,
            chunks=chunks,
            max_files=int(args.max_files),
        )
        if not remote_paths:
            raise FileNotFoundError(
                f"No SlimPajama files matched split={args.split!r}, chunks={args.slimpajama_chunks!r}, "
                f"max_files={args.max_files}."
            )
        print(f"Loading {SLIMPAJAMA_DATASET} through the text-only zstd generator with {len(remote_paths)} files.")
        local_files = _download_hf_dataset_files(SLIMPAJAMA_DATASET, remote_paths)
    return Dataset.from_generator(
        _iter_jsonl_zst_text,
        gen_kwargs={"data_files": local_files, "text_column": args.text_column},
        features=Features({args.text_column: Value("string")}),
    )


def _maybe_limit_documents(dataset, max_documents: int):
    if max_documents <= 0:
        return dataset
    limit = min(int(max_documents), len(dataset))
    print(f"Limiting raw documents to first {limit:,} of {len(dataset):,}.")
    return dataset.select(range(limit))


def _load_raw_dataset(args) -> ShardedTextDataset:
    _log(
        "[load] preparing raw dataset: "
        f"dataset={args.dataset_name}, config={args.dataset_config}, split={args.split}, "
        f"data_files={args.data_files or 'none'}, text_column={args.text_column}"
    )
    dataset_kwargs = {"split": args.split}
    data_files = _split_csv(args.data_files)
    if data_files:
        dataset_kwargs["data_files"] = data_files[0] if len(data_files) == 1 else data_files
    if args.dataset_name == PROOF_PILE_2_DATASET:
        dataset = _load_proof_pile_2_without_script(args)
    elif args.dataset_name == SLIMPAJAMA_DATASET:
        dataset = _load_slimpajama_without_script(args)
    elif args.dataset_config:
        _log(f"[load] calling datasets.load_dataset with config: {args.dataset_config}")
        dataset = load_dataset(args.dataset_name, args.dataset_config, **dataset_kwargs)
    else:
        _log("[load] calling datasets.load_dataset")
        dataset = load_dataset(args.dataset_name, **dataset_kwargs)
    _log(f"[load] raw dataset loaded: columns={list(dataset.column_names)}, rows={len(dataset):,}")
    dataset = _maybe_limit_documents(dataset, int(args.max_documents))
    wrapped = ShardedTextDataset(dataset=dataset, text_column=args.text_column, docs_per_shard=args.docs_per_shard)
    _log(
        "[load] sharded raw dataset: "
        f"documents={wrapped.num_documents:,}, docs_per_shard={wrapped.docs_per_shard:,}, "
        f"num_shards={wrapped.num_shards:,}"
    )
    return wrapped


def _manifest(args, tokenizer, raw_dataset: ShardedTextDataset) -> Dict:
    return {
        "format_version": FORMAT_VERSION,
        "dataset": {
            "dataset_name": args.dataset_name,
            "dataset_config": args.dataset_config,
            "split": args.split,
            "data_files": _split_csv(args.data_files),
            "text_column": args.text_column,
            "num_documents": raw_dataset.num_documents,
            "max_documents": int(args.max_documents),
            "max_files": int(args.max_files),
            "slimpajama_chunks": args.slimpajama_chunks,
        },
        "tokenizer": {
            "name_or_path": args.tokenizer_name_or_path,
            "vocab_size": len(tokenizer),
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
        },
        "packing": {
            "max_length": int(args.max_length),
            "append_eos": bool(args.append_eos),
            "docs_per_shard": int(args.docs_per_shard),
            "packed_blocks_per_shard": int(args.packed_blocks_per_shard),
        },
    }


def _assert_manifest_compatible(existing: Dict, current: Dict, where: Path):
    if existing != current:
        raise ValueError(
            f"Existing tokenization manifest at {where} is incompatible with the current arguments. "
            "Use `--overwrite` to start over."
        )


def _manifest_without_raw(args, tokenizer) -> Dict:
    return {
        "format_version": FORMAT_VERSION,
        "dataset": {
            "dataset_name": args.dataset_name,
            "dataset_config": args.dataset_config,
            "split": args.split,
            "data_files": _split_csv(args.data_files),
            "text_column": args.text_column,
            "max_documents": int(args.max_documents),
            "max_files": int(args.max_files),
            "slimpajama_chunks": args.slimpajama_chunks,
        },
        "tokenizer": {
            "name_or_path": args.tokenizer_name_or_path,
            "vocab_size": len(tokenizer),
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
        },
        "packing": {
            "max_length": int(args.max_length),
            "append_eos": bool(args.append_eos),
            "docs_per_shard": int(args.docs_per_shard),
            "packed_blocks_per_shard": int(args.packed_blocks_per_shard),
        },
    }


def _assert_manifest_matches_args_without_raw(existing: Dict, args, tokenizer, where: Path):
    expected = _manifest_without_raw(args, tokenizer)
    comparable_existing = {
        "format_version": existing.get("format_version"),
        "dataset": {
            key: existing.get("dataset", {}).get(key)
            for key in expected["dataset"]
        },
        "tokenizer": existing.get("tokenizer", {}),
        "packing": existing.get("packing", {}),
    }
    if comparable_existing != expected:
        raise ValueError(
            f"Existing tokenization manifest at {where} is incompatible with the current arguments. "
            "Use `--overwrite` to start over."
        )


def _num_tokenized_shards_from_manifest(manifest: Dict) -> int:
    dataset_info = manifest.get("dataset", {})
    packing_info = manifest.get("packing", {})
    num_documents = int(dataset_info.get("num_documents", 0) or 0)
    docs_per_shard = int(packing_info.get("docs_per_shard", 0) or 0)
    if num_documents <= 0 or docs_per_shard <= 0:
        raise ValueError("Existing manifest is missing positive dataset.num_documents or packing.docs_per_shard.")
    return int(math.ceil(float(num_documents) / float(docs_per_shard)))


def _tokenized_completion_stats(tokenized_store: ShardStore, num_shards: int) -> Dict[str, int]:
    completed_ids = tokenized_store.completed_ids()
    completed_set = set(completed_ids)
    contiguous = 0
    total_tokens = 0
    for shard_id in range(num_shards):
        if shard_id not in completed_set:
            continue
        metadata = tokenized_store.read_done(shard_id) or {}
        total_tokens += int(metadata.get("token_count", 0))
        contiguous += 1
    return {
        "completed": int(contiguous),
        "expected": int(num_shards),
        "tokenized_tokens": int(total_tokens),
    }


def _maybe_resume_from_complete_tokenized(args, tokenizer, output_dir: Path, work_dir: Path):
    """Return (manifest, num_shards, stats) when raw HF loading can be safely skipped."""
    if not args.resume or not args.resume_skip_raw_if_tokenized_complete or args.overwrite:
        return None

    output_tmp = Path(str(output_dir) + ".tmp")
    work_manifest = work_dir / "manifest.json"
    final_manifest = output_dir / "manifest.json"

    if output_dir.exists() and (output_dir / SUCCESS_MARKER).exists():
        if not final_manifest.exists():
            raise FileExistsError(
                f"Completed output is missing manifest metadata: {final_manifest}. Use `--overwrite` to rebuild it."
            )
        manifest = _read_json(final_manifest)
        _assert_manifest_matches_args_without_raw(manifest, args, tokenizer, final_manifest)
        _log(f"[resume-fast] output dataset already complete: {output_dir}")
        return manifest, 0, {"already_complete": 1}

    if not work_manifest.exists():
        _log("[resume-fast] no work manifest found; raw dataset loading is required.")
        return None

    manifest = _read_json(work_manifest)
    _assert_manifest_matches_args_without_raw(manifest, args, tokenizer, work_manifest)
    num_shards = _num_tokenized_shards_from_manifest(manifest)
    tokenized_store = ShardStore(work_dir / "tokenized", kind="tokenized")
    stats = _tokenized_completion_stats(tokenized_store, num_shards)
    _log(
        "[resume-fast] tokenized shard state: "
        f"{stats['completed']}/{stats['expected']} complete, tokens={stats['tokenized_tokens']:,}"
    )
    if stats["completed"] != stats["expected"]:
        _log("[resume-fast] tokenized shards are incomplete; raw dataset loading is required for missing shards.")
        return None

    if output_dir.exists():
        _log(f"[resume-fast] removing partial final output before resume commit: {output_dir}")
        _remove_path(output_dir)
    if output_tmp.exists():
        _log(f"[resume-fast] removing stale temporary final output: {output_tmp}")
        _remove_path(output_tmp)
    _log("[resume-fast] all tokenized shards are complete; skipping load_dataset/HF cache preparation.")
    return manifest, num_shards, stats


def _prepare_state(args, output_dir: Path, work_dir: Path, manifest: Dict) -> bool:
    """Prepare output/work dirs. Return True when final output already exists."""
    output_tmp = Path(str(output_dir) + ".tmp")
    work_manifest = work_dir / "manifest.json"
    final_manifest = output_dir / "manifest.json"

    if args.overwrite:
        _remove_path(output_dir)
        _remove_path(output_tmp)
        _remove_path(work_dir)

    if output_dir.exists():
        if (output_dir / SUCCESS_MARKER).exists():
            if final_manifest.exists():
                _assert_manifest_compatible(_read_json(final_manifest), manifest, final_manifest)
            else:
                raise FileExistsError(
                    f"Completed output is missing manifest metadata: {final_manifest}. "
                    "Use `--overwrite` to rebuild it."
                )
            print(f"Output dataset already complete: {output_dir}")
            return True
        if not args.resume:
            raise FileExistsError(
                f"Output directory exists without {SUCCESS_MARKER}: {output_dir}. "
                "Use `--resume` to replace the partial final output or `--overwrite` to start clean."
            )
        print(f"Removing partial final output before resume commit: {output_dir}")
        _remove_path(output_dir)

    if output_tmp.exists():
        if not args.resume:
            raise FileExistsError(f"Temporary output exists: {output_tmp}. Use `--resume` or `--overwrite`.")
        print(f"Removing stale temporary final output: {output_tmp}")
        _remove_path(output_tmp)

    if work_manifest.exists():
        if not args.resume:
            raise FileExistsError(f"Work directory already exists: {work_dir}. Use `--resume` or `--overwrite`.")
        _assert_manifest_compatible(_read_json(work_manifest), manifest, work_manifest)
    else:
        if work_dir.exists() and any(work_dir.iterdir()):
            if not args.resume:
                raise FileExistsError(f"Non-empty work directory exists: {work_dir}. Use `--resume` or `--overwrite`.")
            print(f"Removing work directory without manifest before resume: {work_dir}")
            _remove_path(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        _write_json_atomic(work_manifest, manifest)

    return False


def _tokenize_shards(args, tokenizer, raw_dataset: ShardedTextDataset, tokenized_store: ShardStore) -> Dict[str, int]:
    tokenized_store.cleanup_tmp()
    total_tokens = 0
    completed = 0

    def tokenize_fn(examples):
        tokenized = tokenizer(examples[args.text_column], add_special_tokens=False)
        input_ids = tokenized["input_ids"]
        if args.append_eos and tokenizer.eos_token_id is not None:
            input_ids = [ids + [tokenizer.eos_token_id] for ids in input_ids]
        return {"input_ids": input_ids}

    for shard_id in range(raw_dataset.num_shards):
        start, end = raw_dataset.shard_range(shard_id)
        if tokenized_store.is_done(shard_id):
            metadata = tokenized_store.read_done(shard_id) or {}
            total_tokens += int(metadata.get("token_count", 0))
            completed += 1
            print(f"[tokenize] skip shard {shard_id + 1}/{raw_dataset.num_shards} rows={start}:{end}")
            continue

        print(f"[tokenize] shard {shard_id + 1}/{raw_dataset.num_shards} rows={start}:{end}")
        raw_shard = raw_dataset.select_shard(shard_id)
        map_kwargs = {
            "batched": True,
            "remove_columns": raw_shard.column_names,
            "desc": f"Tokenizing shard {shard_id:06d}",
        }
        if args.num_proc > 1:
            map_kwargs["num_proc"] = args.num_proc
        tokenized = raw_shard.map(tokenize_fn, **map_kwargs)
        token_count = sum(len(ids) for ids in tokenized["input_ids"])
        tokenized_store.write_atomic(
            shard_id,
            tokenized,
            {
                "row_start": int(start),
                "row_end": int(end),
                "document_count": int(len(tokenized)),
                "token_count": int(token_count),
            },
        )
        total_tokens += int(token_count)
        completed += 1

    return {"tokenized_shards": completed, "tokenized_tokens": total_tokens}


def _flush_packed_shard(
    packed_store: ShardStore,
    shard_id: int,
    blocks: List[List[int]],
    max_length: int,
    force_write: bool = False,
):
    if not blocks:
        return
    if packed_store.is_done(shard_id) and not force_write:
        print(f"[pack] skip packed shard {shard_id:06d} blocks={len(blocks)}")
        return
    print(f"[pack] write packed shard {shard_id:06d} blocks={len(blocks)}")
    attention = [1] * max_length
    dataset = Dataset.from_dict(
        {
            "input_ids": blocks,
            "attention_mask": [attention for _ in blocks],
        }
    )
    packed_store.write_atomic(
        shard_id,
        dataset,
        {
            "block_count": int(len(blocks)),
            "token_count": int(len(blocks) * max_length),
        },
    )


def _tokenized_token_counts(tokenized_store: ShardStore, num_tokenized_shards: int) -> List[int]:
    counts = []
    for shard_id in range(num_tokenized_shards):
        metadata = tokenized_store.read_done(shard_id)
        if not metadata:
            raise FileNotFoundError(f"tokenized shard {shard_id} is not complete.")
        counts.append(int(metadata.get("token_count", 0)))
    return counts


def _locate_token_offset(token_counts: Sequence[int], token_offset: int) -> Tuple[int, int]:
    if token_offset <= 0:
        return 0, 0
    seen = 0
    for shard_id, count in enumerate(token_counts):
        next_seen = seen + int(count)
        if token_offset < next_seen:
            return shard_id, int(token_offset - seen)
        seen = next_seen
    return len(token_counts), 0


def _iter_token_rows_after_skip(tokenized, skip_tokens: int):
    remaining = int(skip_tokens)
    for row in tokenized:
        ids = row["input_ids"]
        if remaining >= len(ids):
            remaining -= len(ids)
            continue
        if remaining > 0:
            ids = ids[remaining:]
            remaining = 0
        yield ids
    if remaining > 0:
        raise RuntimeError(f"Could not skip requested token offset; {remaining} tokens remained.")


def _packed_resume_plan(
    tokenized_store: ShardStore,
    packed_store: ShardStore,
    num_tokenized_shards: int,
    max_length: int,
    packed_blocks_per_shard: int,
) -> Dict[str, int]:
    token_counts = _tokenized_token_counts(tokenized_store, num_tokenized_shards)
    total_tokens = int(sum(token_counts))
    expected_blocks = total_tokens // int(max_length)
    expected_shards = int(math.ceil(expected_blocks / float(packed_blocks_per_shard))) if expected_blocks else 0
    completed = set(packed_store.completed_ids())

    completed_blocks = 0
    first_missing = 0
    while first_missing in completed:
        metadata = packed_store.read_done(first_missing) or {}
        completed_blocks += int(metadata.get("block_count", packed_blocks_per_shard))
        first_missing += 1

    if expected_shards > 0 and all(shard_id in completed for shard_id in range(expected_shards)):
        return {
            "all_complete": 1,
            "packed_shard_id": expected_shards,
            "total_blocks": expected_blocks,
            "start_tokenized_shard_id": num_tokenized_shards,
            "skip_tokens_in_start_shard": 0,
            "dropped_tail_tokens": total_tokens - expected_blocks * int(max_length),
        }

    if first_missing == 0:
        return {
            "all_complete": 0,
            "packed_shard_id": 0,
            "total_blocks": 0,
            "start_tokenized_shard_id": 0,
            "skip_tokens_in_start_shard": 0,
            "dropped_tail_tokens": 0,
        }

    skip_tokens = int(completed_blocks) * int(max_length)
    start_shard, skip_in_shard = _locate_token_offset(token_counts, skip_tokens)
    return {
        "all_complete": 0,
        "packed_shard_id": int(first_missing),
        "total_blocks": int(completed_blocks),
        "start_tokenized_shard_id": int(start_shard),
        "skip_tokens_in_start_shard": int(skip_in_shard),
        "dropped_tail_tokens": 0,
    }


def pack_tokenized_shards(
    tokenized_store: ShardStore,
    packed_store: ShardStore,
    num_tokenized_shards: int,
    max_length: int,
    packed_blocks_per_shard: int,
) -> Dict[str, int]:
    packed_store.cleanup_tmp()
    builder = PackedBlockBuilder(max_length=max_length)
    pending_blocks: List[List[int]] = []
    resume_plan = _packed_resume_plan(
        tokenized_store=tokenized_store,
        packed_store=packed_store,
        num_tokenized_shards=num_tokenized_shards,
        max_length=max_length,
        packed_blocks_per_shard=packed_blocks_per_shard,
    )
    if resume_plan["all_complete"]:
        print(
            "[pack] all packed shards already complete: "
            f"packed_shards={resume_plan['packed_shard_id']:,}, blocks={resume_plan['total_blocks']:,}"
        )
        return {
            "packed_shards": int(resume_plan["packed_shard_id"]),
            "packed_blocks": int(resume_plan["total_blocks"]),
            "dropped_tail_tokens": int(resume_plan["dropped_tail_tokens"]),
        }

    packed_shard_id = int(resume_plan["packed_shard_id"])
    total_blocks = int(resume_plan["total_blocks"])
    start_tokenized_shard_id = int(resume_plan["start_tokenized_shard_id"])
    skip_tokens_in_start_shard = int(resume_plan["skip_tokens_in_start_shard"])
    if packed_shard_id > 0:
        print(
            "[pack] fast-forward resume: "
            f"next_packed_shard={packed_shard_id:06d}, completed_blocks={total_blocks:,}, "
            f"start_tokenized_shard={start_tokenized_shard_id + 1}/{num_tokenized_shards}, "
            f"skip_tokens_in_start_shard={skip_tokens_in_start_shard:,}"
        )

    for tokenized_shard_id in range(start_tokenized_shard_id, num_tokenized_shards):
        tokenized = tokenized_store.load(tokenized_shard_id)
        skip_tokens = skip_tokens_in_start_shard if tokenized_shard_id == start_tokenized_shard_id else 0
        if skip_tokens > 0:
            print(
                f"[pack] consume tokenized shard {tokenized_shard_id + 1}/{num_tokenized_shards} "
                f"(skip {skip_tokens:,} tokens)"
            )
        else:
            print(f"[pack] consume tokenized shard {tokenized_shard_id + 1}/{num_tokenized_shards}")
        for token_ids in _iter_token_rows_after_skip(tokenized, skip_tokens):
            for block in builder.add_tokens(token_ids):
                pending_blocks.append(block)
                total_blocks += 1
                if len(pending_blocks) == packed_blocks_per_shard:
                    _flush_packed_shard(packed_store, packed_shard_id, pending_blocks, max_length)
                    pending_blocks = []
                    packed_shard_id += 1

    if pending_blocks:
        _flush_packed_shard(packed_store, packed_shard_id, pending_blocks, max_length)
        packed_shard_id += 1

    return {
        "packed_shards": packed_shard_id,
        "packed_blocks": total_blocks,
        "dropped_tail_tokens": builder.dropped_tokens,
    }


def _commit_final_dataset(output_dir: Path, packed_store: ShardStore, manifest: Dict):
    packed_ids = packed_store.completed_ids()
    if not packed_ids:
        raise RuntimeError("No packed shards were produced; cannot commit an empty dataset.")
    if packed_ids != list(range(len(packed_ids))):
        raise RuntimeError(f"Packed shard ids are not contiguous from zero: {packed_ids[:10]}...")

    print(f"[commit] loading {len(packed_ids)} packed shards")
    datasets = [packed_store.load(shard_id) for shard_id in packed_ids]
    final_dataset = datasets[0] if len(datasets) == 1 else concatenate_datasets(datasets)

    output_tmp = Path(str(output_dir) + ".tmp")
    output_tmp.parent.mkdir(parents=True, exist_ok=True)
    _remove_path(output_tmp)
    final_dataset.save_to_disk(str(output_tmp))
    if output_dir.exists():
        _remove_path(output_dir)
    _rename_dir_atomic(output_tmp, output_dir)
    _write_json_atomic(output_dir / "manifest.json", manifest)
    (output_dir / SUCCESS_MARKER).write_text("ok\n", encoding="utf-8")
    print(f"[commit] wrote final dataset with {len(final_dataset)} blocks to {output_dir}")


def main():
    _log(f"tokenize_packed_dataset.py version={SCRIPT_VERSION} path={Path(__file__).resolve()}")
    args = parse_args()
    _validate_args(args)

    output_dir = Path(args.output_dir)
    work_dir = Path(args.work_dir) if args.work_dir else Path(str(output_dir) + ".work")
    _log(
        "[startup] "
        f"output_dir={output_dir}, work_dir={work_dir}, resume={args.resume}, overwrite={args.overwrite}, "
        f"resume_skip_raw_if_tokenized_complete={args.resume_skip_raw_if_tokenized_complete}"
    )
    _log(
        "[startup] HF caches: "
        f"HF_HOME={os.environ.get('HF_HOME', '')}, "
        f"HF_DATASETS_CACHE={os.environ.get('HF_DATASETS_CACHE', '')}, "
        f"HF_HUB_CACHE={os.environ.get('HF_HUB_CACHE', '')}"
    )

    _log(f"[tokenizer] loading tokenizer: {args.tokenizer_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    _log(
        "[tokenizer] loaded: "
        f"vocab_size={len(tokenizer):,}, eos_token_id={tokenizer.eos_token_id}, pad_token_id={tokenizer.pad_token_id}"
    )

    fast_resume = _maybe_resume_from_complete_tokenized(args, tokenizer, output_dir, work_dir)
    if fast_resume is not None and fast_resume[2].get("already_complete"):
        return

    tokenized_store = ShardStore(work_dir / "tokenized", kind="tokenized")
    packed_store = ShardStore(work_dir / "packed", kind="packed")

    if fast_resume is not None:
        manifest, num_tokenized_shards, resume_stats = fast_resume
        tokenized_stats = {
            "tokenized_shards": int(resume_stats["completed"]),
            "tokenized_tokens": int(resume_stats["tokenized_tokens"]),
            "tokenized_resume_fast_path": 1,
        }
    else:
        raw_dataset = _load_raw_dataset(args)
        manifest = _manifest(args, tokenizer, raw_dataset)
        _log("[state] preparing output/work state")
        if _prepare_state(args, output_dir=output_dir, work_dir=work_dir, manifest=manifest):
            return
        num_tokenized_shards = raw_dataset.num_shards
        tokenized_stats = _tokenize_shards(args, tokenizer, raw_dataset, tokenized_store)

    _log(
        "[pack] starting packing: "
        f"num_tokenized_shards={num_tokenized_shards:,}, packed_blocks_per_shard={int(args.packed_blocks_per_shard):,}"
    )
    packed_stats = pack_tokenized_shards(
        tokenized_store=tokenized_store,
        packed_store=packed_store,
        num_tokenized_shards=num_tokenized_shards,
        max_length=int(args.max_length),
        packed_blocks_per_shard=int(args.packed_blocks_per_shard),
    )
    _log(f"[summary] {json.dumps({**tokenized_stats, **packed_stats}, sort_keys=True)}")

    _commit_final_dataset(output_dir=output_dir, packed_store=packed_store, manifest=manifest)
    if not args.keep_intermediate:
        _log(f"[cleanup] removing intermediate tokenized shards: {tokenized_store.base_dir}")
        _remove_path(tokenized_store.base_dir)


if __name__ == "__main__":
    main()
