#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MAC-SQL multi-agent text-to-SQL inference.")
    parser.add_argument("--benchmark", default="spider_data")
    parser.add_argument("--split", default="test", choices=["train", "dev", "test"])
    parser.add_argument("--db", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--db-root", type=Path, default=None)
    parser.add_argument("--prompt-dir", type=Path, default=Path("prompts/macsql/default"))
    parser.add_argument("--output_path", type=Path, default=None)
    parser.add_argument("--flush-every", type=int, default=None)

    parser.add_argument("--teacher-base", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--teacher-sft-ckpt", default=None)
    parser.add_argument("--teacher-lora-adapter", default=None)
    parser.add_argument("--teacher-ckpt-revision", default=None)
    parser.add_argument("--student-base", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--student-sft-ckpt", default=None)
    parser.add_argument("--student-lora-adapters", default=None)
    parser.add_argument("--student-ckpt-revision", default=None)
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda", "auto"])

    parser.add_argument("--selector-model", choices=["teacher", "student", "none"], default="student")
    parser.add_argument("--decomposer-model", choices=["teacher", "student"], default="teacher")
    parser.add_argument("--refiner-model", choices=["teacher", "student", "none"], default="student")

    parser.add_argument("--selector-threshold-tokens", type=int, default=3500)
    parser.add_argument("--max-refine-rounds", type=int, default=3)
    parser.add_argument("--execution-timeout", type=float, default=30.0)
    parser.add_argument("--no-refine-empty-result", action="store_true")
    parser.add_argument("--value-examples", type=int, default=5)
    parser.add_argument(
        "--max-new-tokens",
        default="auto",
        help="Max generation tokens, or 'auto' to match running.sh benchmark defaults.",
    )
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=0)
    return parser.parse_args()


def resolve_max_new_tokens(value: str, benchmark: str, split: str) -> int:
    if value != "auto":
        try:
            parsed = int(value)
        except ValueError as exc:
            raise ValueError("--max-new-tokens must be 'auto' or a positive integer") from exc
        if parsed <= 0:
            raise ValueError("--max-new-tokens must be 'auto' or a positive integer")
        return parsed

    by_benchmark_split = {
        ("spider_data", "train"): 1612,
        ("spider_data", "dev"): 755,
        ("spider_data", "test"): 856,
        ("spider_syn", "test"): 756,
        ("spider_realistic", "test"): 755,
        ("spider_dk", "test"): 663,
    }
    return by_benchmark_split.get((benchmark, split), 1024)


def build_registry(args: argparse.Namespace):
    from src.macsql.models import ModelRegistry, ModelSpec, split_adapters

    return ModelRegistry(
        [
            ModelSpec(
                name="teacher",
                base=args.teacher_base,
                sft_ckpt=args.teacher_sft_ckpt,
                sft_revision=args.teacher_ckpt_revision,
                lora_adapters=split_adapters(args.teacher_lora_adapter),
                lora_revision=args.teacher_ckpt_revision,
                device=args.device,
            ),
            ModelSpec(
                name="student",
                base=args.student_base,
                sft_ckpt=args.student_sft_ckpt,
                sft_revision=args.student_ckpt_revision,
                lora_adapters=split_adapters(args.student_lora_adapters),
                lora_revision=args.student_ckpt_revision,
                device=args.device,
            ),
        ]
    )


def main() -> None:
    args = parse_args()
    from src.macsql.agents import DecomposerAgent, GenerationConfig, RefinerAgent, SelectorAgent
    from src.macsql.prompts import load_prompt_set
    from src.macsql.runner import MacSqlConfig, MacSqlPipeline, run_pipeline
    from src.macsql.schema import load_macsql_records

    prompts = load_prompt_set(args.prompt_dir)
    records = load_macsql_records(
        benchmark=args.benchmark,
        split=args.split,
        db_filter=args.db,
        limit=args.limit,
        db_root=args.db_root,
        value_examples=args.value_examples,
    )
    registry = build_registry(args)
    max_new_tokens = resolve_max_new_tokens(args.max_new_tokens, args.benchmark, args.split)
    gen = GenerationConfig(
        max_new_tokens=max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
    )

    selector = None
    if args.selector_model != "none":
        selector = SelectorAgent(
            name="selector",
            loaded_model=registry.get(args.selector_model),
            system_prompt=prompts.selector_system,
            user_template=prompts.selector_user,
            gen=gen,
        )
    decomposer = DecomposerAgent(
        name="decomposer",
        loaded_model=registry.get(args.decomposer_model),
        system_prompt=prompts.decomposer_system,
        user_template=prompts.decomposer_user,
        gen=gen,
    )
    refiner = None
    if args.refiner_model != "none":
        refiner = RefinerAgent(
            name="refiner",
            loaded_model=registry.get(args.refiner_model),
            system_prompt=prompts.refiner_system,
            user_template=prompts.refiner_user,
            gen=gen,
        )

    db_name = args.db if args.db not in (None, "", "full", "all", args.benchmark) else "full"
    output_path = args.output_path or Path("results/macsql") / args.benchmark / f"{args.split}_{db_name}_sql_result.json"
    pipeline = MacSqlPipeline(
        selector=selector,
        decomposer=decomposer,
        refiner=refiner,
        config=MacSqlConfig(
            selector_threshold_tokens=args.selector_threshold_tokens,
            max_refine_rounds=args.max_refine_rounds,
            execution_timeout=args.execution_timeout,
            refine_empty_result=not args.no_refine_empty_result,
            value_examples=args.value_examples,
        ),
    )
    results = run_pipeline(
        pipeline=pipeline,
        records=records,
        output_path=output_path,
        flush_every=args.flush_every,
    )
    failed = sum(1 for row in results if not row.get("success"))
    print(f"Saved MAC-SQL results to {output_path}")
    print(f"Rows={len(results)} failed={failed} max_new_tokens={max_new_tokens}")


if __name__ == "__main__":
    main()
