"""Audit PVSD decisive positions on teacher and student rollouts.

This is the T0/T1 validation script from PVSD_IMPLEMENTATION.md: compare
decisive positions found while teacher-forcing the privileged prompt on a
teacher greedy trace versus on the student's sampled on-policy rollout.
"""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from pvsd.common.chat_template import apply_chat_template_compat
from pvsd.common.decisive_positions import find_decisive_positions, per_position_gap
from pvsd.math.privileged_views import build_teacher_user_message, get_view_payload


MODEL_NAME = "Qwen/Qwen3-4B"
VIEW_TYPE = "full_solution"
TOP_N = 8
MIN_POS = 20
DTYPES = {
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float16": torch.float16,
    "fp16": torch.float16,
    "float32": torch.float32,
    "fp32": torch.float32,
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name", default=MODEL_NAME)
    parser.add_argument("--problem", required=True)
    parser.add_argument("--solution", required=True)
    parser.add_argument("--view_type", default=VIEW_TYPE)
    parser.add_argument("--top_n", type=int, default=TOP_N)
    parser.add_argument("--min_pos", type=int, default=MIN_POS)
    parser.add_argument("--partial_solution_ratio", type=float, default=0.5)
    parser.add_argument("--max_new_tokens", type=int, default=160)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--torch_dtype", choices=sorted(DTYPES), default="bfloat16")
    return parser.parse_args()


def model_device(model):
    return next(model.parameters()).device


def build_prompts(tokenizer, problem, solution, view_type, partial_solution_ratio):
    student_message = f"Problem: {problem}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}."
    student_prompt = apply_chat_template_compat(
        tokenizer,
        [{"role": "user", "content": student_message}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    payload = get_view_payload(
        problem,
        solution,
        view_type=view_type,
        partial_solution_ratio=partial_solution_ratio,
    )
    teacher_message = build_teacher_user_message(problem, payload, view_type=view_type)
    teacher_prompt = apply_chat_template_compat(
        tokenizer,
        [{"role": "user", "content": teacher_message}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    return student_prompt, teacher_prompt


def encode_prompt(tokenizer, prompt, device):
    return tokenizer(prompt, return_tensors="pt").input_ids.to(device)


def decisive_positions_for_rollout(
    model,
    tokenizer,
    student_prompt,
    teacher_prompt,
    rollout_ids,
    top_n,
    min_pos,
):
    device = rollout_ids.device
    student_full = encode_prompt(tokenizer, student_prompt, device)
    teacher_full = encode_prompt(tokenizer, teacher_prompt, device)
    student_input = torch.cat([student_full, rollout_ids], dim=1)
    teacher_input = torch.cat([teacher_full, rollout_ids], dim=1)

    with torch.no_grad():
        student_logits = model(student_input).logits[:, student_full.shape[1] - 1 : -1, :]
        teacher_logits = model(teacher_input).logits[:, teacher_full.shape[1] - 1 : -1, :]

    student_logp = F.log_softmax(student_logits, dim=-1)
    teacher_logp = F.log_softmax(teacher_logits, dim=-1)
    delta = per_position_gap(teacher_logp, student_logp, rollout_ids)
    labels = torch.zeros_like(rollout_ids)
    return find_decisive_positions(delta, labels, top_n=top_n, min_pos=min_pos)[0]


def tokens_at_positions(tokenizer, rollout_ids, positions):
    return [
        {
            "position": int(pos.item()),
            "token": tokenizer.decode(rollout_ids[0, pos : pos + 1]),
        }
        for pos in positions
    ]


def main():
    args = parse_args()
    dtype = DTYPES[args.torch_dtype]
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
        device_map="auto",
    )
    model.eval()

    student_prompt, teacher_prompt = build_prompts(
        tokenizer,
        args.problem,
        args.solution,
        view_type=args.view_type,
        partial_solution_ratio=args.partial_solution_ratio,
    )
    device = model_device(model)

    teacher_full = encode_prompt(tokenizer, teacher_prompt, device)
    teacher_trace = model.generate(
        teacher_full,
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
    )[:, teacher_full.shape[1] :]
    positions_on_teacher_trace = decisive_positions_for_rollout(
        model,
        tokenizer,
        student_prompt,
        teacher_prompt,
        teacher_trace,
        top_n=args.top_n,
        min_pos=args.min_pos,
    )

    student_full = encode_prompt(tokenizer, student_prompt, device)
    student_rollout = model.generate(
        student_full,
        max_new_tokens=args.max_new_tokens,
        do_sample=True,
        temperature=args.temperature,
        top_p=args.top_p,
    )[:, student_full.shape[1] :]
    positions_on_student_rollout = decisive_positions_for_rollout(
        model,
        tokenizer,
        student_prompt,
        teacher_prompt,
        student_rollout,
        top_n=args.top_n,
        min_pos=args.min_pos,
    )

    print("Decisive positions on teacher greedy trace:", positions_on_teacher_trace.tolist())
    print("Tokens at teacher-trace positions:", tokens_at_positions(tokenizer, teacher_trace, positions_on_teacher_trace))
    print("Decisive positions on student on-policy rollout:", positions_on_student_rollout.tolist())
    print("Tokens at student-rollout positions:", tokens_at_positions(tokenizer, student_rollout, positions_on_student_rollout))


if __name__ == "__main__":
    main()
