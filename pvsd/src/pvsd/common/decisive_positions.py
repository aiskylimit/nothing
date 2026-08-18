"""Decisive-position identification from already-computed log-probs."""

from __future__ import annotations

import torch


def per_position_gap(
    teacher_log_probs: torch.Tensor,
    student_log_probs: torch.Tensor,
    sampled_token_ids: torch.Tensor,
) -> torch.Tensor:
    """Return log P_teacher(y_t) - log P_student(y_t) for each sampled token."""

    if teacher_log_probs.dim() == 3:
        teacher_log_probs = torch.gather(
            teacher_log_probs,
            dim=-1,
            index=sampled_token_ids.unsqueeze(-1),
        ).squeeze(-1)
    if student_log_probs.dim() == 3:
        student_log_probs = torch.gather(
            student_log_probs,
            dim=-1,
            index=sampled_token_ids.unsqueeze(-1),
        ).squeeze(-1)
    return teacher_log_probs - student_log_probs


def find_decisive_positions(
    delta: torch.Tensor,
    labels: torch.Tensor,
    top_n: int,
    min_pos: int,
) -> list[torch.Tensor]:
    """Return up to top_n high-gap rollout-relative positions for each row."""

    batch_size, _ = delta.shape
    valid = labels != -100
    if min_pos > 0:
        valid[:, :min_pos] = False

    checkpoints = []
    for row in range(batch_size):
        row_delta = delta[row].masked_fill(~valid[row], float("-inf"))
        n_valid = int(valid[row].sum().item())
        k = min(top_n, n_valid)
        if k == 0:
            checkpoints.append(torch.empty(0, dtype=torch.long, device=delta.device))
            continue
        _, idx = torch.topk(row_delta, k=k)
        checkpoints.append(idx.sort().values)
    return checkpoints
