import argparse
import json
import os
from pathlib import Path
from time import time
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

import torch
from huggingface_hub import hf_hub_download, list_repo_files, snapshot_download
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.llm_services import parse_json_from_string, parse_llm_response
from src.logger_config import setup_logger

RESULTS_DIR = "results"
LOG_DIR = "logging_data/qwen3"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
log_path = os.path.join(LOG_DIR, f"log_infer_{int(time())}.txt")
logger = setup_logger(__name__, log_file=log_path)

SPIDER_SYSTEM_PROMPT_PATH = Path("prompts/single_turn/generator/system_prompt.txt")
SPIDER_USER_PROMPT_PATH = Path("prompts/single_turn/generator/user_prompt.txt")
BIRD_SYSTEM_PROMPT_PATH = Path("prompts/single_turn/bird_generator/system_prompt.txt")
BIRD_USER_PROMPT_PATH = Path("prompts/single_turn/bird_generator/user_prompt.txt")
SPIDER_STYLE_BENCHMARKS = {
    "spider_data": {
        "splits": {
            "train": "train_spider.json",
            "dev": "dev.json",
            "test": "test.json",
        },
        "tables": {
            "train": "tables.json",
            "dev": "tables.json",
            "test": "test_tables.json",
        },
        "question_fields": ("question",),
    },
    "spider_dk": {
        "splits": {
            "test": "test.json",
        },
        "tables": "tables.json",
        "question_fields": ("question",),
    },
    "spider_realistic": {
        "splits": {
            "test": "test.json",
        },
        "tables": "../spider_data/tables.json",
        "question_fields": ("question",),
    },
    "spider_syn": {
        "splits": {
            "train": "train_spider.json",
            "test": "test.json",
        },
        "tables": "../spider_data/tables.json",
        "question_fields": ("SpiderSynQuestion", "question", "SpiderQuestion"),
    },
    "bird": {
        "splits": {
            "dev": "dev/dev.json",
        },
        "tables": {
            "dev": "dev/dev_tables.json",
        },
        "question_fields": ("question",),
        "gold_sql_fields": ("SQL",),
    },
}
DIALOG_BENCHMARKS = {"sparc", "cosql_dataset"}
SUPPORTED_BENCHMARKS = tuple(sorted(set(SPIDER_STYLE_BENCHMARKS) | DIALOG_BENCHMARKS))

DIALOG_SYSTEM_PROMPT = """You are a SQL Query Generator for conversational text-to-SQL tasks.

Your task is to generate a valid SQLite SQL query for the current user turn using the provided schema and conversation context.

Rules:
- Use only tables, columns, and relationships that exist in the provided schema.
- Do not invent schema elements.
- Generate a single executable SQL query.
- Use the conversation history to resolve references such as pronouns, ellipsis, and follow-up constraints.
- When prior SQL context is provided, use it only as context; still produce the full SQL query for the current turn.
- Return only the columns needed to answer the current turn.
- Use DISTINCT when duplicates are possible and the question implies unique results.
- If the question requires aggregation, sorting, grouping, filtering, or limiting, use the correct SQL clauses.
- Use JOINs only when needed, and use schema-consistent join keys.
- Qualify column names when they may be ambiguous.
- Prefer concise, conventional SQLite SQL.
- Ensure the query is syntactically valid.

Output format:
{
  "sql": "The complete SQL query"
}

Return only the JSON object and nothing else."""

DIALOG_USER_TEMPLATE = """SCHEMA:
{schema}

CONVERSATION HISTORY:
{history}

CURRENT QUESTION:
{question}

Generate a valid SQLite SQL query for the current question using the schema and the conversation context.
Return only the JSON object in the required format."""


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run text-to-SQL inference on Spider, SParC, or CoSQL using local benchmark files."
    )
    parser.add_argument(
        "--benchmark",
        default="spider_data",
        choices=SUPPORTED_BENCHMARKS,
        help="Benchmark name under benchmarks_2/.",
    )
    parser.add_argument(
        "--split",
        default="dev",
        choices=["train", "dev", "test"],
        help="Data split to run inference on.",
    )
    parser.add_argument(
        "--db",
        default=None,
        help='Optional db_id filter. If omitted or set to "full", "all", or the benchmark name, use all databases.',
    )
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B", help="Base model name or path.")
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=None,
        help="Local path, HF repo id, hf:// path, or HF URL of a LoRA/full fine-tuned checkpoint.",
    )
    parser.add_argument(
        "--ckpt_revision",
        type=str,
        default=None,
        help="Optional Hugging Face revision for --ckpt_path.",
    )
    parser.add_argument(
        "--device",
        default=None,
        choices=["cpu", "cuda", "auto"],
        help="Device to run the model on.",
    )
    parser.add_argument("--max-length", type=int, default=512, help="Max new tokens to generate.")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for Spider inference.")
    parser.add_argument("--temperature", type=float, default=0.5, help="Sampling temperature.")
    parser.add_argument("--top-p", type=float, default=0.95, help="Top-p sampling.")
    parser.add_argument("--top-k", type=int, default=0, help="Top-k sampling.")
    parser.add_argument("--limit", type=int, default=None, help="Optional sample limit after filtering.")
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Output JSON path. Defaults to results/<benchmark>/<split>_<db>_sqls_<model>.json.",
    )
    parser.add_argument(
        "--flush-every",
        type=int,
        default=None,
        help="Write partial output every N processed samples.",
    )
    parser.add_argument(
        "--dialog-history-mode",
        choices=["pred", "gold", "none"],
        default="pred",
        help="How to supply previous SQL context for SParC/CoSQL.",
    )
    return parser.parse_args()


def _parse_hf_source(source: str, revision: Optional[str] = None) -> Dict[str, Optional[str]]:
    raw = str(source).strip()
    normalized = raw.rstrip("/")

    if os.path.isdir(raw):
        return {
            "kind": "local",
            "path": raw,
            "repo_id": None,
            "revision": None,
            "subfolder": None,
        }

    def _pack_hf(repo_id: str, rev: Optional[str], subfolder: Optional[str]) -> Dict[str, Optional[str]]:
        return {
            "kind": "hf",
            "path": None,
            "repo_id": repo_id,
            "revision": rev,
            "subfolder": subfolder,
        }

    if normalized.startswith("hf://"):
        content = normalized[len("hf://"):].strip("/")
        parts = [p for p in content.split("/") if p]
        if len(parts) < 2:
            raise ValueError(
                f"Invalid HF path '{source}'. Expected: hf://<owner>/<repo>/<optional/subfolder>"
            )
        repo_id = f"{parts[0]}/{parts[1]}"
        subfolder = "/".join(parts[2:]) if len(parts) > 2 else None
        return _pack_hf(repo_id, revision, subfolder)

    parsed = urlparse(normalized)
    if parsed.scheme in {"http", "https"} and parsed.netloc in {"huggingface.co", "www.huggingface.co"}:
        parts = [p for p in parsed.path.split("/") if p]
        if len(parts) >= 2:
            repo_id = f"{parts[0]}/{parts[1]}"
            resolved_revision = revision
            subfolder = None
            if len(parts) >= 4 and parts[2] in {"tree", "blob"}:
                resolved_revision = revision or parts[3]
                subfolder = "/".join(parts[4:]) if len(parts) > 4 else None
            elif len(parts) > 2:
                subfolder = "/".join(parts[2:])
            return _pack_hf(repo_id, resolved_revision, subfolder)

    shorthand = normalized.strip("/")
    parts = [p for p in shorthand.split("/") if p]
    if len(parts) >= 4 and parts[2] in {"tree", "blob"}:
        repo_id = f"{parts[0]}/{parts[1]}"
        resolved_revision = revision or parts[3]
        subfolder = "/".join(parts[4:]) if len(parts) > 4 else None
        return _pack_hf(repo_id, resolved_revision, subfolder)

    if len(parts) >= 3 and not normalized.startswith(".") and not os.path.exists(raw):
        repo_id = f"{parts[0]}/{parts[1]}"
        subfolder = "/".join(parts[2:])
        return _pack_hf(repo_id, revision, subfolder)

    return _pack_hf(normalized, revision, None)


def _materialize_ckpt_source(source: str, revision: Optional[str] = None) -> Tuple[str, Optional[str]]:
    parsed = _parse_hf_source(source, revision)
    if parsed["kind"] == "local":
        return parsed["path"], None  # type: ignore[return-value]

    repo_id = parsed["repo_id"]  # type: ignore[assignment]
    repo_revision = parsed["revision"]
    subfolder = parsed["subfolder"]

    if not subfolder:
        return repo_id, repo_revision  # type: ignore[return-value]

    token = os.getenv("HF_READ_TOKEN") or os.getenv("HF_TOKEN")
    snapshot_dir = snapshot_download(
        repo_id=repo_id,  # type: ignore[arg-type]
        revision=repo_revision,
        allow_patterns=[f"{subfolder}/*", f"{subfolder}/**"],
        token=token,
    )
    resolved = os.path.join(snapshot_dir, subfolder)
    return resolved, None


CKPT_MARKER_FILES = (
    "adapter_config.json",
    "config.json",
    "model.safetensors",
    "pytorch_model.bin",
    "adapter_model.bin",
    "adapter_model.safetensors",
)


def _local_has_ckpt_marker(path: str | Path) -> bool:
    path = Path(path)
    return path.is_dir() and any((path / filename).exists() for filename in CKPT_MARKER_FILES)


def _find_latest_local_ckpt_dir(path: str | Path) -> Optional[str]:
    root = Path(path)
    if not root.is_dir():
        return None
    if _local_has_ckpt_marker(root):
        return str(root)

    candidates = []
    for child in root.rglob("*"):
        if not child.is_dir() or not _local_has_ckpt_marker(child):
            continue
        step = int(child.name) if child.name.isdigit() else -1
        candidates.append((step, child.stat().st_mtime, child))

    if not candidates:
        return None
    return str(max(candidates, key=lambda item: (item[0], item[1]))[2])


def _resolve_ckpt_dir(source: str, revision: Optional[str] = None) -> Tuple[str, Optional[str]]:
    resolved_source, resolved_revision = _materialize_ckpt_source(source, revision)
    if os.path.isdir(resolved_source):
        nested_ckpt = _find_latest_local_ckpt_dir(resolved_source)
        if nested_ckpt is not None and nested_ckpt != resolved_source:
            logger.info(f"Resolved checkpoint directory: {nested_ckpt}")
            print(f"Resolved checkpoint directory: {nested_ckpt}")
            return nested_ckpt, None
    return resolved_source, resolved_revision


def _local_or_hf_has_file(source: str, filename: str, revision: Optional[str] = None) -> bool:
    resolved_source, resolved_revision = _resolve_ckpt_dir(source, revision)

    if os.path.isdir(resolved_source):
        return os.path.exists(os.path.join(resolved_source, filename))

    try:
        repo_files = list_repo_files(repo_id=resolved_source, revision=resolved_revision)
        return filename in repo_files
    except Exception:
        return False


def _resolve_ckpt_file(source: str, filename: str, revision: Optional[str] = None) -> Optional[str]:
    resolved_source, resolved_revision = _resolve_ckpt_dir(source, revision)

    if os.path.isdir(resolved_source):
        candidate = os.path.join(resolved_source, filename)
        return candidate if os.path.exists(candidate) else None

    try:
        return hf_hub_download(repo_id=resolved_source, filename=filename, revision=resolved_revision)
    except Exception:
        return None


def init_model(model_name_or_path, ckpt_path=None, ckpt_revision=None, device=None):
    if device is None or device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    is_peft = False
    resolved_ckpt_source = ckpt_path
    resolved_ckpt_revision = ckpt_revision
    if ckpt_path:
        if _local_or_hf_has_file(ckpt_path, "adapter_config.json", ckpt_revision):
            is_peft = True
            resolved_ckpt_source, resolved_ckpt_revision = _resolve_ckpt_dir(ckpt_path, ckpt_revision)
            print("This is LoRA finetune")
        elif _local_or_hf_has_file(ckpt_path, "adapter_model.safetensors", ckpt_revision) or _local_or_hf_has_file(
            ckpt_path, "adapter_model.bin", ckpt_revision
        ):
            raise RuntimeError(
                "Found adapter weights but missing adapter_config.json at --ckpt_path. "
                "Please point --ckpt_path to the exact LoRA checkpoint directory."
            )
        elif _local_or_hf_has_file(ckpt_path, "config.json", ckpt_revision):
            resolved_ckpt_source, resolved_ckpt_revision = _resolve_ckpt_dir(ckpt_path, ckpt_revision)
            model_name_or_path = resolved_ckpt_source
            print("This is a full finetune")

    logger.info(f"Loading tokenizer from {model_name_or_path}")
    print(f"Loading tokenizer from {model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        padding_side="left",
        trust_remote_code=True,
        revision=resolved_ckpt_revision if model_name_or_path == resolved_ckpt_source else None,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info(f"Loading model from {model_name_or_path} on {device}")
    print(f"Loading model from {model_name_or_path} on {device}")
    if device == "cpu":
        dtype = torch.float32
    else:
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float16

    device_map = "auto" if device == "cuda" else {"": device}

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=True,
        revision=resolved_ckpt_revision if model_name_or_path == resolved_ckpt_source else None,
    )

    if ckpt_path and is_peft:
        print(f"Loading PEFT checkpoint weights from {resolved_ckpt_source}")
        logger.info(f"Loading PEFT checkpoint weights from {resolved_ckpt_source}")
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, resolved_ckpt_source, revision=resolved_ckpt_revision)
        model = model.merge_and_unload()
        print("Successfully loaded and merged LoRA weights.")
    elif ckpt_path and model_name_or_path != ckpt_path:
        import safetensors.torch

        safetensor_path = _resolve_ckpt_file(ckpt_path, "model.safetensors", ckpt_revision)
        bin_path = _resolve_ckpt_file(ckpt_path, "pytorch_model.bin", ckpt_revision)

        def _load_weights_tolerant(state_dict):
            incompatible = model.load_state_dict(state_dict, strict=False)
            missing_keys = set(incompatible.missing_keys)
            unexpected_keys = set(incompatible.unexpected_keys)
            allowed_missing = {"lm_head.weight"}
            bad_missing = sorted(missing_keys - allowed_missing)
            bad_unexpected = sorted(unexpected_keys)
            if bad_missing or bad_unexpected:
                raise RuntimeError(
                    f"Missing keys: {sorted(missing_keys)}\nUnexpected keys: {sorted(unexpected_keys)}"
                )
            if hasattr(model, "tie_weights"):
                model.tie_weights()

        if safetensor_path:
            _load_weights_tolerant(safetensors.torch.load_file(safetensor_path))
            print("Loaded full model weights from safetensors.")
        elif bin_path:
            _load_weights_tolerant(torch.load(bin_path, map_location="cpu"))
            print("Loaded full model weights from pytorch_model.bin.")

    model.eval()
    return tokenizer, model


def generate_response_batch(tokenizer, model, batch_messages, max_length=512, temperature=0.5, top_p=0.95, top_k=0):
    def _apply_template(messages):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

    texts = [_apply_template(messages) for messages in batch_messages]
    tokenizer.padding_side = "left"
    inputs = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )

    generated_ids = outputs[:, inputs["input_ids"].shape[-1] :]
    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)


def is_full_db(db: Optional[str], benchmark: Optional[str] = None) -> bool:
    if db is None:
        return True

    normalized = str(db).strip().lower()
    full_names = {"full", "all", ""}
    if benchmark:
        full_names.add(str(benchmark).strip().lower())
    return normalized in full_names


def is_spider_style_benchmark(benchmark: str) -> bool:
    return benchmark in SPIDER_STYLE_BENCHMARKS


def get_configured_split(config: Dict, benchmark: str, split: str) -> str:
    split_files = config["splits"]
    if split not in split_files:
        available = ", ".join(sorted(split_files))
        raise ValueError(f"{benchmark} does not include split {split!r}. Available splits: {available}")
    return split_files[split]


def get_configured_tables(config: Dict, split: str) -> str:
    tables = config["tables"]
    if isinstance(tables, dict):
        return tables[split]
    return tables


def get_spider_question(item: Dict, question_fields: Tuple[str, ...]) -> str:
    for field in question_fields:
        value = item.get(field)
        if value:
            return str(value)
    raise KeyError(f"missing question field; expected one of: {', '.join(question_fields)}")


def get_gold_sql(item: Dict, gold_sql_fields: Tuple[str, ...]) -> Optional[str]:
    for field in gold_sql_fields:
        value = item.get(field)
        if value:
            return str(value)
    return None


def get_benchmark_paths(benchmark: str, split: str) -> Tuple[Path, Path]:
    if is_spider_style_benchmark(benchmark):
        config = SPIDER_STYLE_BENCHMARKS[benchmark]
        benchmark_dir = Path("benchmarks_2") / benchmark
        split_file = get_configured_split(config, benchmark, split)
        tables_file = get_configured_tables(config, split)
        return benchmark_dir / split_file, benchmark_dir / tables_file

    if benchmark == "sparc":
        if split == "test":
            raise ValueError("Local SParC benchmark does not include a test split.")
        split_file = {"train": "train.json", "dev": "dev.json"}[split]
        return Path("benchmarks_2") / benchmark / split_file, Path("benchmarks_2") / benchmark / "tables.json"

    if benchmark == "cosql_dataset":
        if split == "test":
            raise ValueError("Local CoSQL benchmark does not include a test split.")
        split_file = {"train": "cosql_train.json", "dev": "cosql_dev.json"}[split]
        return (
            Path("benchmarks_2") / benchmark / "sql_state_tracking" / split_file,
            Path("benchmarks_2") / benchmark / "tables.json",
        )

    raise ValueError(f"Unsupported benchmark: {benchmark}")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def serialize_schema_entry(entry: Dict) -> str:
    table_names = entry.get("table_names_original") or entry.get("table_names") or []
    column_names = entry.get("column_names_original") or entry.get("column_names") or []
    foreign_keys = entry.get("foreign_keys") or []

    table_to_columns: Dict[int, List[str]] = {idx: [] for idx in range(len(table_names))}
    for table_idx, col_name in column_names:
        if table_idx == -1:
            continue
        table_to_columns.setdefault(table_idx, []).append(str(col_name))

    table_lines = []
    for idx, table_name in enumerate(table_names):
        cols = ", ".join(table_to_columns.get(idx, []))
        table_lines.append(f"- {table_name}({cols})")

    fk_lines = []
    for src_idx, dst_idx in foreign_keys:
        src_table_idx, src_col = column_names[src_idx]
        dst_table_idx, dst_col = column_names[dst_idx]
        src_table = table_names[src_table_idx]
        dst_table = table_names[dst_table_idx]
        fk_lines.append(f"- {src_table}.{src_col} -> {dst_table}.{dst_col}")

    schema_parts = ["Tables:"] + table_lines
    if fk_lines:
        schema_parts.extend(["", "Foreign keys:"] + fk_lines)
    return "\n".join(schema_parts)


def load_schema_lookup(tables_path: Path) -> Dict[str, str]:
    return {entry["db_id"]: serialize_schema_entry(entry) for entry in read_json(tables_path)}


def load_prompt_templates() -> Tuple[str, str]:
    return load_text(SPIDER_SYSTEM_PROMPT_PATH), load_text(SPIDER_USER_PROMPT_PATH)


def load_single_turn_prompt_templates(benchmark: str) -> Tuple[str, str]:
    if benchmark == "bird":
        return load_text(BIRD_SYSTEM_PROMPT_PATH), load_text(BIRD_USER_PROMPT_PATH)
    return load_prompt_templates()


def load_records(benchmark: str, split: str, db_filter: Optional[str], limit: Optional[int]) -> List[Dict]:
    data_path, tables_path = get_benchmark_paths(benchmark, split)
    raw_data = read_json(data_path)
    schema_lookup = load_schema_lookup(tables_path)
    use_all = is_full_db(db_filter, benchmark)
    records: List[Dict] = []
    available_db_ids = set()

    if is_spider_style_benchmark(benchmark):
        question_fields = SPIDER_STYLE_BENCHMARKS[benchmark]["question_fields"]
        gold_sql_fields = SPIDER_STYLE_BENCHMARKS[benchmark].get("gold_sql_fields", ("query",))
        for idx, item in enumerate(raw_data):
            db_id = item["db_id"]
            available_db_ids.add(db_id)
            if not use_all and db_id != db_filter:
                continue
            records.append(
                {
                    "benchmark": benchmark,
                    "split": split,
                    "sample_id": f"{benchmark}:{split}:{idx}",
                    "db_id": db_id,
                    "question": get_spider_question(item, question_fields),
                    "evidence": str(item.get("evidence") or ""),
                    "gold_sql": get_gold_sql(item, gold_sql_fields),
                    "schema": schema_lookup[db_id],
                    "is_dialog": False,
                    "dialog_id": None,
                    "turn_index": 0,
                }
            )
    else:
        for dialog_idx, item in enumerate(raw_data):
            db_id = item["database_id"]
            available_db_ids.add(db_id)
            if not use_all and db_id != db_filter:
                continue
            turns = item.get("interaction", [])
            for turn_index, turn in enumerate(turns):
                history_utterances = [prev["utterance"] for prev in turns[:turn_index]]
                gold_history_sql = [prev["query"] for prev in turns[:turn_index]]
                records.append(
                    {
                        "benchmark": benchmark,
                        "split": split,
                        "sample_id": f"{benchmark}:{split}:{dialog_idx}:{turn_index}",
                        "db_id": db_id,
                        "question": turn["utterance"],
                        "gold_sql": turn.get("query"),
                        "schema": schema_lookup[db_id],
                        "is_dialog": True,
                        "dialog_id": f"{benchmark}:{split}:{dialog_idx}",
                        "turn_index": turn_index,
                        "history_utterances": history_utterances,
                        "gold_history_sql": gold_history_sql,
                    }
                )

    if not records and not use_all and raw_data:
        examples = ", ".join(sorted(available_db_ids)[:10])
        suffix = f" Available db_id examples: {examples}" if examples else ""
        raise ValueError(
            f"No records found for --db {db_filter!r} in benchmark={benchmark}, split={split}.{suffix}"
        )

    if limit is not None:
        records = records[:limit]
    return records


def build_dialog_history(history_utterances: List[str], history_sql: List[str]) -> str:
    if not history_utterances:
        return "None"
    lines = []
    for idx, utterance in enumerate(history_utterances):
        lines.append(f"Turn {idx + 1} User: {utterance}")
        if idx < len(history_sql) and history_sql[idx]:
            lines.append(f"Turn {idx + 1} SQL: {history_sql[idx]}")
    return "\n".join(lines)


def build_messages_for_record(record: Dict, single_turn_system: str, single_turn_user: str, history_sql: Optional[List[str]] = None):
    if not record["is_dialog"]:
        user_prompt = single_turn_user.format(
            question=record["question"],
            evidence=record.get("evidence", ""),
            schema=record["schema"],
        )
        return [
            {"role": "system", "content": single_turn_system},
            {"role": "user", "content": user_prompt},
        ]

    prior_sql = history_sql if history_sql is not None else []
    history_text = build_dialog_history(record.get("history_utterances", []), prior_sql)
    user_prompt = DIALOG_USER_TEMPLATE.format(
        schema=record["schema"],
        history=history_text,
        question=record["question"],
    )
    return [
        {"role": "system", "content": DIALOG_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def extract_sql_from_response(raw_response: str) -> Tuple[str, str, str]:
    parsed = parse_llm_response(raw_response)
    final_answer = parsed.get("final_answer", "").strip()
    parsed_json = parse_json_from_string(final_answer)
    if parsed_json and isinstance(parsed_json, dict) and "sql" in parsed_json:
        return parsed.get("think", ""), final_answer, str(parsed_json["sql"]).strip()

    # Fallback: accept plain SQL if the model skipped the JSON wrapper.
    if final_answer:
        return parsed.get("think", ""), final_answer, final_answer

    raise ValueError("Failed to parse SQL from model response.")


def compact_output(results: List[Dict]) -> List[Dict]:
    rows = []
    for row in results:
        compact = {
            "sample_id": row["sample_id"],
            "benchmark": row["benchmark"],
            "split": row["split"],
            "db_id": row["db_id"],
            "question": row["question"],
            "gold_sql": row["gold_sql"],
            "pred_sql": row["pred_sql"],
            "success": row["success"],
            "error": row["error"],
        }
        if "evidence" in row:
            compact["evidence"] = row["evidence"]
        if row["dialog_id"] is not None:
            compact["dialog_id"] = row["dialog_id"]
            compact["turn_index"] = row["turn_index"]
        rows.append(compact)
    return rows


def write_output_snapshot(results: List[Dict], output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(f"{output_path.name}.tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(compact_output(results), f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, output_path)


def run_spider_inference(
    records: List[Dict],
    tokenizer,
    model,
    max_length: int,
    batch_size: int,
    temperature: float,
    top_p: float,
    top_k: int,
    output_path: Optional[Path],
    flush_every: Optional[int],
) -> Tuple[List[Dict], List[Dict]]:
    benchmark = records[0]["benchmark"] if records else ""
    single_turn_system, single_turn_user = load_single_turn_prompt_templates(benchmark)
    results: List[Dict] = []
    errors: List[Dict] = []
    next_flush_at = flush_every if flush_every else None

    progress_bar = tqdm(total=len(records), desc=f"Running {benchmark or 'spider-style'}")
    for i in range(0, len(records), batch_size):
        batch_records = records[i : i + batch_size]
        batch_messages = [
            build_messages_for_record(record, single_turn_system, single_turn_user)
            for record in batch_records
        ]
        raw_responses = generate_response_batch(
            tokenizer,
            model,
            batch_messages,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )

        for record, raw_response in zip(batch_records, raw_responses):
            try:
                think, final_answer, pred_sql = extract_sql_from_response(raw_response)
                results.append(
                    {
                        **record,
                        "raw_response": raw_response,
                        "think": think,
                        "final_answer": final_answer,
                        "pred_sql": pred_sql,
                        "success": True,
                        "error": None,
                    }
                )
            except Exception as exc:
                errors.append({"sample_id": record["sample_id"], "error": str(exc)})
                results.append(
                    {
                        **record,
                        "raw_response": raw_response,
                        "think": "",
                        "final_answer": "",
                        "pred_sql": None,
                        "success": False,
                        "error": str(exc),
                    }
                )

        progress_bar.update(len(batch_records))
        processed_count = min(i + len(batch_records), len(records))
        if output_path and flush_every and processed_count >= next_flush_at:
            write_output_snapshot(results, output_path)
            while next_flush_at <= processed_count:
                next_flush_at += flush_every

    return results, errors


def run_dialog_inference(
    records: List[Dict],
    tokenizer,
    model,
    max_length: int,
    temperature: float,
    top_p: float,
    top_k: int,
    output_path: Optional[Path],
    flush_every: Optional[int],
    history_mode: str,
) -> Tuple[List[Dict], List[Dict]]:
    single_turn_system, single_turn_user = load_prompt_templates()
    results: List[Dict] = []
    errors: List[Dict] = []
    predicted_sql_history: Dict[str, List[str]] = {}
    next_flush_at = flush_every if flush_every else None

    progress_bar = tqdm(total=len(records), desc="Running dialog benchmark")
    for idx, record in enumerate(records, start=1):
        dialog_id = record["dialog_id"]
        if history_mode == "gold":
            prior_sql = record.get("gold_history_sql", [])
        elif history_mode == "none":
            prior_sql = []
        else:
            prior_sql = predicted_sql_history.get(dialog_id, [])

        messages = build_messages_for_record(record, single_turn_system, single_turn_user, prior_sql)
        raw_response = generate_response_batch(
            tokenizer,
            model,
            [messages],
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )[0]

        try:
            think, final_answer, pred_sql = extract_sql_from_response(raw_response)
            predicted_sql_history.setdefault(dialog_id, []).append(pred_sql)
            results.append(
                {
                    **record,
                    "raw_response": raw_response,
                    "think": think,
                    "final_answer": final_answer,
                    "pred_sql": pred_sql,
                    "success": True,
                    "error": None,
                }
            )
        except Exception as exc:
            predicted_sql_history.setdefault(dialog_id, []).append("")
            errors.append({"sample_id": record["sample_id"], "error": str(exc)})
            results.append(
                {
                    **record,
                    "raw_response": raw_response,
                    "think": "",
                    "final_answer": "",
                    "pred_sql": None,
                    "success": False,
                    "error": str(exc),
                }
            )

        progress_bar.update(1)
        if output_path and flush_every and idx >= next_flush_at:
            write_output_snapshot(results, output_path)
            while next_flush_at <= idx:
                next_flush_at += flush_every

    return results, errors


def main():
    args = parse_args()
    if args.flush_every is not None and args.flush_every <= 0:
        raise ValueError("--flush-every must be a positive integer")

    records = load_records(args.benchmark, args.split, args.db, args.limit)
    db_name = args.db if not is_full_db(args.db, args.benchmark) else "full"
    tokenizer, model = init_model(args.model, args.ckpt_path, args.ckpt_revision, device=args.device)

    print(f"Running benchmark={args.benchmark}, split={args.split}, db={db_name}, samples={len(records)}")
    output_path = (
        Path(args.output_path)
        if args.output_path
        else Path(RESULTS_DIR) / args.benchmark / f"{args.split}_{db_name}_sqls_{args.model.split('/')[-1]}.json"
    )

    if is_spider_style_benchmark(args.benchmark):
        results, errors = run_spider_inference(
            records=records,
            tokenizer=tokenizer,
            model=model,
            max_length=args.max_length,
            batch_size=args.batch_size,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            output_path=output_path,
            flush_every=args.flush_every,
        )
    else:
        if args.batch_size != 1:
            logger.warning("Dialog inference is run turn-by-turn; ignoring --batch-size > 1.")
        results, errors = run_dialog_inference(
            records=records,
            tokenizer=tokenizer,
            model=model,
            max_length=args.max_length,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            output_path=output_path,
            flush_every=args.flush_every,
            history_mode=args.dialog_history_mode,
        )

    write_output_snapshot(results, output_path)
    print(f"Saved results to: {output_path}")
    print(f"Success: {len(results) - len(errors)} | Failed: {len(errors)}")


if __name__ == "__main__":
    main()
