"""The objective is reverse KL ``D_KL(student || steered teacher)`` at ``beta = 1``.

``generalized_jsd_loss`` builds its divergences with ``F.kl_div(..., log_target=True)``,
whose argument order is the reverse of the mathematical convention. These tests pin
down that direction so the ``--beta 1.0`` recommendation cannot silently become a
forward KL.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F


def reference_kl(p_log_probs: torch.Tensor, q_log_probs: torch.Tensor) -> torch.Tensor:
    """``D_KL(p || q)`` from log-probabilities, textbook definition."""

    return (p_log_probs.exp() * (p_log_probs - q_log_probs)).sum(dim=-1)


def test_kl_div_with_log_target_reverses_its_arguments():
    torch.manual_seed(0)
    student = F.log_softmax(torch.randn(4, 7), dim=-1)
    teacher = F.log_softmax(torch.randn(4, 7), dim=-1)

    # This is exactly the expression generalized_jsd_loss uses for beta == 1.
    as_used = F.kl_div(teacher, student, reduction="none", log_target=True).sum(dim=-1)
    torch.testing.assert_close(as_used, reference_kl(student, teacher), atol=1e-6, rtol=1e-5)

    # ... and it is *not* the forward KL, unless the two agree.
    assert not torch.allclose(as_used, reference_kl(teacher, student))


def test_beta_zero_is_the_forward_kl():
    torch.manual_seed(1)
    student = F.log_softmax(torch.randn(3, 5), dim=-1)
    teacher = F.log_softmax(torch.randn(3, 5), dim=-1)
    as_used = F.kl_div(student, teacher, reduction="none", log_target=True).sum(dim=-1)
    torch.testing.assert_close(as_used, reference_kl(teacher, student), atol=1e-6, rtol=1e-5)


def test_identical_distributions_give_exactly_zero():
    logits = torch.randn(2, 6)
    log_probs = F.log_softmax(logits, dim=-1)
    value = F.kl_div(log_probs, log_probs, reduction="none", log_target=True)
    assert float(value.abs().max()) == 0.0


def test_real_generalized_jsd_loss_matches_reverse_kl_at_beta_one():
    """Runs only where ``trl`` is installed (i.e. on the training machine)."""

    pytest.importorskip("trl")
    from pvsd.math.trainer import AVSDTrainer

    torch.manual_seed(2)
    student_logits = torch.randn(2, 3, 11)
    teacher_logits = torch.randn(2, 3, 11)
    labels = torch.full((2, 3), 1, dtype=torch.long)
    labels[0, 2] = -100

    loss = AVSDTrainer.generalized_jsd_loss(
        student_logits=student_logits,
        teacher_logits=teacher_logits,
        labels=labels,
        beta=1.0,
        temperature=1.0,
        token_clip=None,
    )

    mask = labels != -100
    student_log_probs = F.log_softmax(student_logits, dim=-1)
    teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)
    expected = reference_kl(student_log_probs, teacher_log_probs)[mask].sum() / mask.sum()
    torch.testing.assert_close(loss, expected, atol=1e-5, rtol=1e-4)
