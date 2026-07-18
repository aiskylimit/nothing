#!/usr/bin/env python3
"""Generate and validate SynID-SQL augmentation candidates using independent retries."""

from __future__ import annotations

import argparse
import json
import sys
import gc
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.synid_sql.augmentation.bird_records import (
    build_bird_teacher_user_prompt,
    load_bird_teacher_templates,
    load_bird_train_records,
)
from infer import init_model
from src.synid_sql.augmentation.io import read_jsonl, write_json, write_jsonl
from src.synid_sql.augmentation.spider_records import (
    build_teacher_user_prompt,
    load_spider_train_records,
    load_synid_teacher_templates,
)
from src.synid_sql.augmentation.sql_extract import extract_sql
from src.synid_sql.augmentation.validator import validate_candidate


DEFAULT_TEACHER_PEFT_PATHS = {
    "spider": "hf://distillation-sql/baselines/qwen3/sft_sft_qwen3_4b_spider_lora/e5-bs4-lr0.0001-G4-N2-NN1-lora-32-64-0.1/1090",
    "bird": "hf://distillation-sql/bird_baselines/qwen3/sft_sft_qwen3_4b_bird_lora/e5-bs2-lr0.0001-G8-N2-NN1-lora-32-64-0.1/1470",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=["spider", "bird"], default="spider")
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--db-root", type=Path, default=None)
    parser.add_argument("--teacher-prompt-dir", type=Path, default=Path("prompts/single_turn/synid_teacher"))
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument(
        "--teacher-peft-path",
        default=None,
        help="Teacher LoRA adapter. Defaults to the benchmark-specific Qwen3-4B teacher LoRA.",
    )
    parser.add_argument("--tensor-parallel-size", "-tp", type=int, default=1, help="Number of GPUs for vLLM")
    parser.add_argument("--num-loops", type=int, default=5, help="Number of independent retries for failed candidates")
    parser.add_argument("--gamma", type=float, default=0.8)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max-input-tokens", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7, help="Must be > 0 for independent retries to differ")
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.root is None:
        args.root = Path("benchmarks_2/bird") if args.benchmark == "bird" else Path("benchmarks_2/spider_data")
    if args.output_root is None:
        args.output_root = args.root / "synid_aug_v2_lora"
    if args.db_root is None:
        args.db_root = (
            Path("benchmarks/bird/train/train_databases")
            if args.benchmark == "bird"
            else Path("benchmarks/spider_data/database")
        )
    if args.teacher_peft_path is None:
        args.teacher_peft_path = DEFAULT_TEACHER_PEFT_PATHS[args.benchmark]
    return args


def _failure_detail(row: dict[str, Any]) -> str:
    fields = []
    for key in (
        "failure_stage",
        "repairable",
        "error",
        "evaluator",
        "gold_row_count",
        "candidate_row_count",
        "jaccard",
        "gamma",
    ):
        if key in row:
            fields.append(f"{key}={row[key]}")
    return "; ".join(fields) if fields else "No additional detail."


def _build_messages(
    tokenizer,
    system_prompt: str,
    user_prompt: str,
) -> str:
    """Build chat messages using the base prompt without injecting previous errors."""
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        add_generation_prompt=True,
        tokenize=False,
    )


def generate_with_vllm(
    *,
    llm: LLM,
    tokenizer,
    system_prompt: str,
    user_prompts: list[str],
    args: argparse.Namespace,
) -> list[tuple[str, str]]:
    """Generate responses using vLLM purely from base prompts (independent retry)."""
    
    prompts = [
        _build_messages(tokenizer, system_prompt, user_prompt)
        for user_prompt in user_prompts
    ]

    do_sample = args.temperature > 0
    sampling_params = SamplingParams(
        temperature=args.temperature if do_sample else 0.0,
        top_p=args.top_p if do_sample else 1.0,
        top_k=args.top_k if args.top_k > 0 else -1, 
        max_tokens=args.max_new_tokens,
        skip_special_tokens=True,
    )

    outputs = llm.generate(prompts, sampling_params)

    results = []
    for output in outputs:
        raw_response = output.outputs[0].text
        results.append((raw_response, extract_sql(raw_response)))

    return results


def _load_existing_loop(loop_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]] | None:
    accepted_path = loop_dir / "accepted.jsonl"
    rejected_path = loop_dir / "rejected.jsonl"
    repair_path = loop_dir / "repair_queue.jsonl"
    if not accepted_path.exists() or not rejected_path.exists() or not repair_path.exists():
        return None
    return read_jsonl(accepted_path), read_jsonl(rejected_path), read_jsonl(repair_path)


def _summarize(loop_index: int, candidates: list[dict[str, Any]], accepted: list[dict[str, Any]], rejected: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "loop": loop_index,
        "candidates": len(candidates),
        "accepted": len(accepted),
        "rejected": len(rejected),
        "rejection_reasons": dict(Counter(row.get("reason", "unknown") for row in rejected)),
    }


def _count_by(rows: list[dict[str, Any]], key: str, default: str = "unknown") -> dict[str, int]:
    return dict(Counter(str(row.get(key, default)) for row in rows))


def _write_error_stats(
    path: Path,
    *,
    benchmark: str,
    source_total_records: int,
    run_total_records: int,
    loop_summaries: list[dict[str, Any]],
    all_rejections: list[dict[str, Any]],
    terminal_rejected: list[dict[str, Any]],
    final_rejected: list[dict[str, Any]] | None = None,
) -> None:
    payload: dict[str, Any] = {
        "updated_at": _timestamp(),
        "benchmark": benchmark,
        "source_total_records": source_total_records,
        "run_total_records": run_total_records,
        "loops": loop_summaries,
        "cumulative_rejections": {
            "total": len(all_rejections),
            "by_reason": _count_by(all_rejections, "reason"),
            "by_failure_stage": _count_by(all_rejections, "failure_stage"),
            "by_repairable": _count_by(all_rejections, "repairable"),
        },
        "terminal_rejections": {
            "total": len(terminal_rejected),
            "by_reason": _count_by(terminal_rejected, "reason"),
            "by_failure_stage": _count_by(terminal_rejected, "failure_stage"),
        },
    }
    if final_rejected is not None:
        payload["final_rejections"] = {
            "total": len(final_rejected),
            "by_reason": _count_by(final_rejected, "reason"),
            "by_failure_stage": _count_by(final_rejected, "failure_stage"),
            "by_repairable": _count_by(final_rejected, "repairable"),
        }
    write_json(path, payload)


def _timestamp() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _append_log(log_path: Path, message: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as fout:
        fout.write(f"[{_timestamp()}] {message}\n")


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fout:
        fout.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    if args.num_loops <= 0:
        raise ValueError("--num-loops must be positive")
    
    if args.temperature <= 0.0 and args.num_loops > 1:
        print("WARNING: Temperature is 0. Independent retries will yield the exact same SQL. Consider setting --temperature > 0.")

    if args.benchmark == "bird":
        records = load_bird_train_records(args.root)
        system_prompt, user_template = load_bird_teacher_templates()
        user_prompt_builder = build_bird_teacher_user_prompt
    else:
        records = load_spider_train_records(args.root)
        system_prompt, user_template = load_synid_teacher_templates(args.teacher_prompt_dir)
        user_prompt_builder = build_teacher_user_prompt

    source_total_records = len(records)
    if args.limit is not None:
        records = records[: args.limit]
    run_total_records = len(records)
    records_by_id = {int(row["id"]): row for row in records}
    log_path = args.output_root / "aug_loops.log"
    progress_path = args.output_root / "loop_progress.jsonl"
    error_stats_path = args.output_root / "error_stats.json"

    start_summary = {
        "event": "start",
        "benchmark": args.benchmark,
        "root": str(args.root),
        "output_root": str(args.output_root),
        "db_root": str(args.db_root),
        "model": args.model,
        "teacher_peft_path": args.teacher_peft_path,
        "num_loops": args.num_loops,
        "gamma": args.gamma,
        "timeout": args.timeout,
        "max_input_tokens": args.max_input_tokens,
        "max_new_tokens": args.max_new_tokens,
        "tensor_parallel_size": args.tensor_parallel_size,
        "source_total_records": source_total_records,
        "run_total_records": run_total_records,
        "limit": args.limit,
        "resume": args.resume,
    }
    _append_log(log_path, "START " + json.dumps(start_summary, ensure_ascii=False))
    _append_jsonl(progress_path, {"timestamp": _timestamp(), **start_summary})

    # ==========================================
    # MERGE LORA INTO BASE MODEL (IF PROVIDED)
    # ==========================================
    model_name_or_path = args.model
    if args.teacher_peft_path:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM
        
        clean_peft_path = args.teacher_peft_path.replace("hf://", "")
        merged_model_path = args.output_root / "merged_model"
        
        if not merged_model_path.exists():
            print(f"Merging LoRA from {clean_peft_path} into base model {args.model}...")
            print("Loading base model on CPU to prevent VRAM OOM...")
            tokenizer, model = init_model(args.model, args.teacher_peft_path, device="cpu")
            model.save_pretrained(merged_model_path)
            tokenizer.save_pretrained(merged_model_path)

            del model
            del tokenizer
            gc.collect()
            torch.cuda.empty_cache()

            print("Merge completed successfully.")
        else:
            print(f"Found existing merged model at {merged_model_path}. Skipping merge step.")
        
        # Override the model path so vLLM loads the merged version
        model_name_or_path = str(merged_model_path)

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    
    print("Loading vLLM Engine...")
    llm = LLM(
        model=model_name_or_path,
        enable_lora=False, # LoRA is already merged statically
        max_model_len=args.max_input_tokens + args.max_new_tokens,
        trust_remote_code=True,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=0.8
    )

    accepted_by_id: dict[int, dict[str, Any]] = {}
    terminal_rejected_by_id: dict[int, dict[str, Any]] = {}
    all_rejections: list[dict[str, Any]] = []
    remaining: list[dict[str, Any]] = records
    loop_summaries = []

    for loop_index in range(1, args.num_loops + 5):
        loop_dir = args.output_root / "loops" / f"loop_{loop_index}"
        loop_dir.mkdir(parents=True, exist_ok=True)

        existing = _load_existing_loop(loop_dir) if args.resume else None
        if existing is not None:
            accepted, rejected, repair_queue = existing
            candidates = read_jsonl(loop_dir / "candidates.jsonl")
        else:
            candidates = []
            accepted = []
            rejected = []
            
            base_records = [records_by_id[int(item["id"])] for item in remaining]
            user_prompts = [
                user_prompt_builder(user_template, base_record)
                for base_record in base_records
            ]
            
            print(f"Loop {loop_index}: Generating {len(remaining)} candidates via vLLM (Independent Rejection Sampling)...")
            generated = generate_with_vllm(
                llm=llm,
                tokenizer=tokenizer,
                system_prompt=system_prompt,
                user_prompts=user_prompts,
                args=args,
            )

            for base_record, (raw_response, candidate_sql) in tqdm(
                zip(base_records, generated), 
                total=len(base_records), 
                desc=f"loop {loop_index} validate"
            ):
                candidate = {
                    **base_record,
                    "loop": loop_index,
                    "raw_response": raw_response,
                    "candidate_sql": candidate_sql,
                }

                gamma = args.gamma if loop_index < args.num_loops else 1.0
                ok, validation_row = validate_candidate(
                    candidate,
                    benchmark=args.benchmark,
                    db_root=args.db_root,
                    gamma=gamma,
                    timeout_s=args.timeout,
                )
                candidates.append(candidate)
                if ok:
                    accepted.append(validation_row)
                else:
                    rejected.append(validation_row)

            accepted_ids = {int(row["id"]) for row in accepted}
            for row in rejected:
                if not row.get("repairable", True) and int(row["id"]) not in accepted_ids:
                    terminal_rejected_by_id.setdefault(int(row["id"]), row)
            
            repair_queue = [
                {
                    **row,
                    "previous_candidate_sql": row.get("candidate_sql", ""),
                    "failure_reason": row.get("reason", "unknown"),
                    "failure_detail": _failure_detail(row),
                }
                for row in rejected
                if int(row["id"]) not in accepted_ids and row.get("repairable", True)
            ]

            write_jsonl(loop_dir / "candidates.jsonl", candidates)
            write_jsonl(loop_dir / "accepted.jsonl", accepted)
            write_jsonl(loop_dir / "rejected.jsonl", rejected)
            write_jsonl(loop_dir / ("final_rejected.jsonl" if loop_index == args.num_loops else "repair_queue.jsonl"), repair_queue)
            if loop_index == args.num_loops:
                write_jsonl(loop_dir / "repair_queue.jsonl", [])

        for row in accepted:
            accepted_by_id.setdefault(int(row["id"]), row)

        remaining = [row for row in repair_queue if int(row["id"]) not in accepted_by_id]
        summary = _summarize(loop_index, candidates, accepted, rejected)
        kept_total = len(accepted_by_id)
        summary["kept_total"] = kept_total
        summary["source_total_records"] = source_total_records
        summary["run_total_records"] = run_total_records
        summary["kept_over_source"] = f"{kept_total}/{source_total_records}"
        summary["kept_over_run"] = f"{kept_total}/{run_total_records}"
        summary["kept_source_ratio"] = kept_total / source_total_records if source_total_records else 0.0
        summary["kept_run_ratio"] = kept_total / run_total_records if run_total_records else 0.0
        summary["terminal_rejected_total"] = len(terminal_rejected_by_id)
        summary["remaining_after_loop"] = len(remaining)
        summary["rejection_stages"] = _count_by(rejected, "failure_stage")
        summary["rejection_repairable"] = _count_by(rejected, "repairable")
        loop_summaries.append(summary)
        all_rejections.extend(rejected)
        
        write_json(loop_dir / "summary.json", summary)
        _write_error_stats(
            error_stats_path,
            benchmark=args.benchmark,
            source_total_records=source_total_records,
            run_total_records=run_total_records,
            loop_summaries=loop_summaries,
            all_rejections=all_rejections,
            terminal_rejected=list(terminal_rejected_by_id.values()),
        )
        
        _append_jsonl(progress_path, {"timestamp": _timestamp(), "event": "loop_done", **summary})
        progress_line = (
            f"LOOP {loop_index} DONE "
            f"accepted_this_loop={len(accepted)} rejected_this_loop={len(rejected)} "
            f"kept={kept_total}/{source_total_records} source_records "
            f"({summary['kept_source_ratio']:.4f}); "
            f"run_subset={kept_total}/{run_total_records}; "
            f"terminal={len(terminal_rejected_by_id)}; "
            f"remaining={len(remaining)}"
        )
        print(progress_line)
        _append_log(log_path, progress_line)

        if not remaining:
            break

    accepted_all = [accepted_by_id[row_id] for row_id in sorted(accepted_by_id)]
    final_rejected_by_id = dict(terminal_rejected_by_id)
    for row in remaining:
        final_rejected_by_id.setdefault(int(row["id"]), row)
    rejected_final = [final_rejected_by_id[row_id] for row_id in sorted(final_rejected_by_id)]
    
    _write_error_stats(
        error_stats_path,
        benchmark=args.benchmark,
        source_total_records=source_total_records,
        run_total_records=run_total_records,
        loop_summaries=loop_summaries,
        all_rejections=all_rejections,
        terminal_rejected=list(terminal_rejected_by_id.values()),
        final_rejected=rejected_final,
    )
    write_jsonl(args.output_root / "accepted_all.jsonl", accepted_all)
    write_jsonl(args.output_root / "rejected_final.jsonl", rejected_final)
    write_json(
        args.output_root / "summary.json",
        {
            "source_total_records": source_total_records,
            "run_total_records": run_total_records,
            "accepted": len(accepted_all),
            "rejected_final": len(rejected_final),
            "terminal_rejected": len(terminal_rejected_by_id),
            "kept_over_source": f"{len(accepted_all)}/{source_total_records}",
            "kept_over_run": f"{len(accepted_all)}/{run_total_records}",
            "loops": loop_summaries,
        },
    )
    _append_log(
        log_path,
        f"DONE kept={len(accepted_all)}/{source_total_records} source_records; "
        f"run_subset={len(accepted_all)}/{run_total_records}; "
        f"terminal={len(terminal_rejected_by_id)}; rejected_final={len(rejected_final)}",
    )
    print(json.dumps({"accepted": len(accepted_all), "rejected_final": len(rejected_final)}, indent=2))


if __name__ == "__main__":
    main()