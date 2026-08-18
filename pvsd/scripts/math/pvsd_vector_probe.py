"""Falsification probe for the PVSD privilege vector (no training involved).

The method only works if the purified vector carries the *content* of the correct
reference rather than the *format* of the privileged view. This script measures
that directly: it steers the student with

  * ``correct``    - v_transfer built from the reference that belongs to the problem
  * ``mismatched`` - v_transfer built from another problem's reference, purified the
                     same way (so it has the same format and the same construction)
  * ``random``     - a random direction rescaled to the same norm as ``correct``

and reports the change in the student's log-probability of the *gold* solution.

Expected outcome if PVSD is sound::

    correct  >>  mismatched  ~=  random  ~=  0

If ``mismatched`` helps about as much as ``correct``, the vector is transporting
the view template and the purification is not isolating content.

Example::

    python scripts/math/pvsd_vector_probe.py \
      --model_name Qwen/Qwen3-4B \
      --num_examples 8 --alphas 0.5,1.0,2.0 \
      --output ~/outputs/results/pvsd/probe_qwen3_4b.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from pvsd.common.pie import compute_privilege_indirect_effect, parse_candidate_layers
from pvsd.common.privilege_vectors import (
    all_heads_in_layers,
    cosine_similarity_rowwise,
    describe_model,
    extract_privilege_vector,
    fuse_view_vectors,
    inject_at_layer,
    position_ids_from_mask,
    purify_privilege_vector,
)
from pvsd.common.token_log_probs import sampled_token_log_probs
from pvsd.math.data_collator import SelfDistillationDataCollator

DTYPES = {
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float16": torch.float16,
    "fp16": torch.float16,
    "float32": torch.float32,
    "fp32": torch.float32,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model_name", default="Qwen/Qwen3-4B")
    parser.add_argument(
        "--problems_file",
        default=None,
        help="Optional JSONL with {'problem','solution'} rows. Defaults to the training dataset.",
    )
    parser.add_argument("--training_dataset", default="openthought")
    parser.add_argument("--num_examples", type=int, default=8)
    parser.add_argument("--views", default="full_solution")
    parser.add_argument(
        "--num_corrupt",
        type=int,
        default=2,
        help="Corrupted contexts per example. Must be >= 2 so the mismatched arm can be purified too.",
    )
    parser.add_argument("--pvsd_layer", type=int, default=None)
    parser.add_argument("--pvsd_layer_fraction", default="quarter", choices=("quarter", "third", "half"))
    parser.add_argument("--alphas", default="1.0", help="Comma-separated steering strengths to sweep.")
    parser.add_argument("--top_k_heads", type=int, default=10)
    parser.add_argument("--no_pie", action="store_true", help="Use every head of the injection layer instead of PIE.")
    parser.add_argument("--pie_head_chunk", type=int, default=8)
    parser.add_argument("--pie_num_examples", type=int, default=2)
    parser.add_argument("--pie_layers", default="all", help="'all', a range like '8:24', or '9,12,15'.")
    parser.add_argument("--max_length", type=int, default=8192)
    parser.add_argument("--max_gold_tokens", type=int, default=1024)
    parser.add_argument("--extract_micro_batch", type=int, default=4)
    parser.add_argument("--torch_dtype", default="bfloat16", choices=sorted(DTYPES))
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def load_examples(args) -> list[dict[str, str]]:
    if args.problems_file:
        rows = []
        with Path(args.problems_file).open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    else:
        from pvsd.math.training_datasets import load_training_dataset

        dataset, _ = load_training_dataset(training_dataset=args.training_dataset)
        rows = [dataset[index] for index in range(min(args.num_examples, len(dataset)))]
    examples = [
        {"problem": str(row["problem"]), "solution": str(row["solution"])}
        for row in rows[: args.num_examples]
    ]
    if len(examples) < args.num_corrupt + 1:
        raise ValueError(
            f"need at least num_corrupt + 1 = {args.num_corrupt + 1} examples, got {len(examples)}"
        )
    return examples


@torch.no_grad()
def gold_log_prob(
    model,
    tokenizer,
    student_prompt: str,
    solution: str,
    max_gold_tokens: int,
    layer: int | None = None,
    vector: torch.Tensor | None = None,
    alpha: float = 1.0,
) -> float:
    """Mean per-token log-prob of the gold solution, optionally under steering."""

    device = next(model.parameters()).device
    prompt_ids = tokenizer(student_prompt, return_tensors="pt", add_special_tokens=False).input_ids
    gold_ids = tokenizer(solution, return_tensors="pt", add_special_tokens=False).input_ids
    gold_ids = gold_ids[:, :max_gold_tokens]
    if gold_ids.shape[1] == 0:
        raise ValueError("gold solution tokenised to zero tokens")

    input_ids = torch.cat([prompt_ids, gold_ids], dim=1).to(device)
    attention_mask = torch.ones_like(input_ids)
    prompt_len = prompt_ids.shape[1]
    position_ids = position_ids_from_mask(attention_mask)

    def run():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
        )
        logits = outputs.logits[:, prompt_len - 1 : -1, :]
        return sampled_token_log_probs(logits, input_ids[:, prompt_len:], temperature=1.0)

    if vector is None:
        log_probs = run()
    else:
        with inject_at_layer(model, layer, vector, alpha=alpha, start_index=prompt_len - 1):
            log_probs = run()
    return float(log_probs.mean())


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    if args.num_corrupt < 2:
        raise ValueError("--num_corrupt must be >= 2 so the mismatched arm gets the same purification.")

    dtype = DTYPES[args.torch_dtype]
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    views = tuple(view.strip() for view in args.views.split(",") if view.strip())
    alphas = [float(item) for item in args.alphas.split(",") if item.strip()]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=dtype).to(device)
    model.eval()

    topology = describe_model(model)
    layer = args.pvsd_layer
    if layer is None:
        divisors = {"quarter": 4, "third": 3, "half": 2}
        layer = topology.num_layers // divisors[args.pvsd_layer_fraction]
    print(f"[probe] topology={topology} injection layer={layer}")

    examples = load_examples(args)
    collator = SelfDistillationDataCollator(
        tokenizer=tokenizer,
        max_length=args.max_length,
        reason_first=False,
        multi_view_mode="single",
        pvsd_views=views,
        pvsd_num_corrupt=args.num_corrupt,
    )
    batch = collator(examples)
    batch = {
        key: (value.to(device) if isinstance(value, torch.Tensor) else value)
        for key, value in batch.items()
    }

    # --- Stage 1: head localisation -----------------------------------------
    if args.no_pie:
        heads = all_heads_in_layers(topology, [layer])
        pie_log = {"mode": "no_pie", "heads": [list(head) for head in heads]}
    else:
        first_view = views[0]
        result = compute_privilege_indirect_effect(
            model,
            topology,
            batch[f"pvsd_{first_view}_input_ids"],
            batch[f"pvsd_{first_view}_attention_mask"],
            batch[f"pvsd_{first_view}_corrupt_input_ids"][:, 0, :],
            batch[f"pvsd_{first_view}_corrupt_attention_mask"][:, 0, :],
            top_k_heads=args.top_k_heads,
            candidate_layers=parse_candidate_layers(args.pie_layers, topology.num_layers),
            head_chunk_size=args.pie_head_chunk,
            max_examples=args.pie_num_examples,
        )
        heads = result.top_heads
        pie_log = {"mode": "pie", "heads": [list(head) for head in heads], **result.as_log_dict()}
    print(f"[probe] head set: {list(heads)}")

    # --- Stage 2: vectors ----------------------------------------------------
    correct_views, mismatched_views, diagnostics = [], [], {}
    for view in views:
        real_ids = batch[f"pvsd_{view}_input_ids"]
        real_mask = batch[f"pvsd_{view}_attention_mask"]
        corrupt_ids = batch[f"pvsd_{view}_corrupt_input_ids"]
        corrupt_mask = batch[f"pvsd_{view}_corrupt_attention_mask"]
        rows, num_corrupt, seq_len = corrupt_ids.shape

        raw = extract_privilege_vector(
            model, topology, real_ids, real_mask, heads, micro_batch_size=args.extract_micro_batch
        )
        corrupt = extract_privilege_vector(
            model,
            topology,
            corrupt_ids.reshape(rows * num_corrupt, seq_len),
            corrupt_mask.reshape(rows * num_corrupt, seq_len),
            heads,
            micro_batch_size=args.extract_micro_batch,
        ).view(rows, num_corrupt, -1)

        transfer = purify_privilege_vector(raw, corrupt)
        # Mismatched arm: treat corrupted context 0 as if it were the reference and
        # purify it against the remaining corrupted contexts - same construction,
        # same format, wrong content.
        transfer_mismatched = purify_privilege_vector(corrupt[:, 0, :], corrupt[:, 1:, :])

        correct_views.append(transfer)
        mismatched_views.append(transfer_mismatched)
        diagnostics[view] = {
            "raw_norm": float(raw.norm(dim=-1).mean()),
            "corrupt_norm": float(corrupt.mean(dim=1).norm(dim=-1).mean()),
            "transfer_norm": float(transfer.norm(dim=-1).mean()),
            "transfer_ratio": float(
                (transfer.norm(dim=-1) / raw.norm(dim=-1).clamp_min(1e-8)).mean()
            ),
            "cos_raw_corrupt": float(cosine_similarity_rowwise(raw, corrupt.mean(dim=1)).mean()),
            "cos_correct_mismatched": float(
                cosine_similarity_rowwise(transfer, transfer_mismatched).mean()
            ),
        }
        print(f"[probe] {view}: {json.dumps(diagnostics[view], indent=2)}")

    correct = fuse_view_vectors(torch.stack(correct_views, dim=1))
    mismatched = fuse_view_vectors(torch.stack(mismatched_views, dim=1))
    random_vector = torch.randn_like(correct)
    random_vector = random_vector * (
        correct.norm(dim=-1, keepdim=True) / random_vector.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    )

    # --- Stage 3: does steering raise log P(gold)? ---------------------------
    student_prompts = tokenizer.batch_decode(batch["student_prompts"], skip_special_tokens=False)
    if tokenizer.pad_token:
        student_prompts = [prompt.replace(tokenizer.pad_token, "") for prompt in student_prompts]

    arms = {"correct": correct, "mismatched": mismatched, "random": random_vector}
    records = []
    for index, example in enumerate(examples):
        baseline = gold_log_prob(
            model, tokenizer, student_prompts[index], example["solution"], args.max_gold_tokens
        )
        record = {"index": index, "baseline_gold_logprob": baseline, "deltas": {}}
        for alpha in alphas:
            for name, vectors in arms.items():
                steered = gold_log_prob(
                    model,
                    tokenizer,
                    student_prompts[index],
                    example["solution"],
                    args.max_gold_tokens,
                    layer=layer,
                    vector=vectors[index : index + 1],
                    alpha=alpha,
                )
                record["deltas"][f"alpha={alpha}/{name}"] = steered - baseline
        records.append(record)
        print(f"[probe] example {index}: baseline={baseline:.4f} deltas={record['deltas']}")

    summary = {}
    for alpha in alphas:
        for name in arms:
            key = f"alpha={alpha}/{name}"
            values = [record["deltas"][key] for record in records]
            summary[key] = sum(values) / len(values)

    report = {
        "model_name": args.model_name,
        "injection_layer": layer,
        "views": list(views),
        "num_examples": len(examples),
        "alphas": alphas,
        "pie": pie_log,
        "vector_diagnostics": diagnostics,
        "mean_gold_logprob_delta": summary,
        "per_example": records,
    }

    print("\n=== mean delta log P(gold) vs no steering ===")
    for key, value in summary.items():
        print(f"  {key:32s} {value:+.4f}")
    print(
        "\nPVSD is only supported if 'correct' is clearly above both 'mismatched' and 'random'."
    )

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nSaved report to {output_path}")


if __name__ == "__main__":
    main()
