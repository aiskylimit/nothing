"""End-to-end PVSD step on CPU: PIE -> extract -> purify -> fuse -> steer -> loss.

This mirrors ``PVSDTrainer._compute_loss_single_view`` and calls the very same
functions it calls, so a regression in the vector pipeline shows up here. It runs
one backward pass on a ~10k-parameter model; it does not train anything.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from pvsd.common.pie import compute_privilege_indirect_effect
from pvsd.common.privilege_vectors import (
    all_heads_in_layers,
    build_privilege_vectors_from_batch,
    extract_privilege_vector,
    inject_at_layer,
    position_ids_from_mask,
    purify_privilege_vector,
)

from conftest import TINY_CONFIG, make_padded_batch


VOCAB = TINY_CONFIG["vocab_size"]
PROMPT_LEN = 4
COMPLETION_LEN = 5
REAL_PROMPT_LENGTHS = (3, 4, 4)  # row 0 leaves a pad gap between prompt and rollout


def make_student_batch():
    """``[prompt][pad][rollout]`` exactly as ``training_step`` assembles it."""

    torch.manual_seed(7)
    batch_size = len(REAL_PROMPT_LENGTHS)
    total = PROMPT_LEN + COMPLETION_LEN
    input_ids = torch.zeros(batch_size, total, dtype=torch.long)
    attention_mask = torch.zeros(batch_size, total, dtype=torch.long)
    for row, length in enumerate(REAL_PROMPT_LENGTHS):
        input_ids[row, :length] = torch.randint(2, VOCAB, (length,))
        attention_mask[row, :length] = 1
    input_ids[:, PROMPT_LEN:] = torch.randint(2, VOCAB, (batch_size, COMPLETION_LEN))
    attention_mask[:, PROMPT_LEN:] = 1

    labels = input_ids.clone()
    for row, length in enumerate(REAL_PROMPT_LENGTHS):
        labels[row, :length] = -100
    labels[input_ids == 0] = -100
    return input_ids, attention_mask, labels


def make_pvsd_batch(views=("full_solution",), num_corrupt=2, identical_corrupt=False):
    """Privileged and corrupted prompt tensors in the collator's layout."""

    torch.manual_seed(11)
    batch_size = len(REAL_PROMPT_LENGTHS)
    batch = {"pvsd_view_names": list(views), "pvsd_num_corrupt": num_corrupt}
    for view_index, view in enumerate(views):
        real_sequences = [
            torch.randint(2, VOCAB, (5 + row + view_index,)).tolist() for row in range(batch_size)
        ]
        real_ids, real_mask = make_padded_batch(real_sequences)
        batch[f"pvsd_{view}_input_ids"] = real_ids
        batch[f"pvsd_{view}_attention_mask"] = real_mask
        batch[f"pvsd_{view}_lengths"] = real_mask.sum(dim=-1)

        if identical_corrupt:
            corrupt_sequences = [
                list(real_sequences[row]) for row in range(batch_size) for _ in range(num_corrupt)
            ]
        else:
            corrupt_sequences = [
                torch.randint(2, VOCAB, (5 + row,)).tolist()
                for row in range(batch_size)
                for _ in range(num_corrupt)
            ]
        corrupt_ids, corrupt_mask = make_padded_batch(corrupt_sequences)
        batch[f"pvsd_{view}_corrupt_input_ids"] = corrupt_ids.view(batch_size, num_corrupt, -1)
        batch[f"pvsd_{view}_corrupt_attention_mask"] = corrupt_mask.view(batch_size, num_corrupt, -1)
        batch[f"pvsd_{view}_corrupt_lengths"] = corrupt_mask.sum(dim=-1).view(
            batch_size, num_corrupt
        )
    return batch


def reverse_kl_loss(student_logits, teacher_logits, labels):
    """``D_KL(student || teacher)`` masked and averaged, i.e. beta = 1."""

    student_log_probs = F.log_softmax(student_logits.float(), dim=-1)
    teacher_log_probs = F.log_softmax(teacher_logits.float(), dim=-1)
    per_element = F.kl_div(teacher_log_probs, student_log_probs, reduction="none", log_target=True)
    mask = labels != -100
    return per_element[mask].sum() / mask.sum()


def run_pvsd_step(model, topology, heads, alpha=1.0, views=("full_solution",), identical_corrupt=False):
    """One PVSD loss computation, in the same order as the trainer."""

    student_ids, student_mask, labels = make_student_batch()
    pvsd_batch = make_pvsd_batch(views=views, identical_corrupt=identical_corrupt)
    position_ids = position_ids_from_mask(student_mask)
    shifted_labels = labels[:, PROMPT_LEN:]

    outputs = model(
        input_ids=student_ids, attention_mask=student_mask, position_ids=position_ids
    )
    student_logits = outputs.logits[:, PROMPT_LEN - 1 : -1, :]

    with torch.no_grad():
        fused, per_view = build_privilege_vectors_from_batch(
            model, topology, views, pvsd_batch, heads, micro_batch_size=2
        )

    with torch.no_grad(), inject_at_layer(
        model, topology.num_layers // 4, fused, alpha=alpha, start_index=PROMPT_LEN - 1
    ) as hook:
        steered_logits = model(
            input_ids=student_ids, attention_mask=student_mask, position_ids=position_ids
        ).logits[:, PROMPT_LEN - 1 : -1, :]
    assert hook.call_count == 1

    loss = reverse_kl_loss(student_logits, steered_logits, shifted_labels)
    return {
        "loss": loss,
        "fused": fused,
        "per_view": per_view,
        "student_logits": student_logits,
        "steered_logits": steered_logits,
        "labels": shifted_labels,
    }


@pytest.fixture
def heads(tiny_lm, tiny_topology):
    """Head set from a real PIE calibration on the privileged/corrupted prompts."""

    batch = make_pvsd_batch()
    result = compute_privilege_indirect_effect(
        tiny_lm,
        tiny_topology,
        batch["pvsd_full_solution_input_ids"],
        batch["pvsd_full_solution_attention_mask"],
        batch["pvsd_full_solution_corrupt_input_ids"][:, 0, :],
        batch["pvsd_full_solution_corrupt_attention_mask"][:, 0, :],
        top_k_heads=4,
        head_chunk_size=4,
        max_examples=2,
    )
    return result.top_heads


def test_pipeline_gives_a_finite_positive_loss_with_gradients(tiny_lm, tiny_topology, heads):
    outcome = run_pvsd_step(tiny_lm, tiny_topology, heads, alpha=1.0)
    loss = outcome["loss"]

    assert torch.isfinite(loss)
    assert float(loss.detach()) > 0.0, "a non-zero privilege vector must move the teacher"
    assert outcome["fused"].shape == (len(REAL_PROMPT_LENGTHS), tiny_topology.resid_dim)
    assert outcome["fused"].dtype == torch.float32
    assert not outcome["fused"].requires_grad, "the teacher signal must be stop-gradient"

    loss.backward()
    grads = [p.grad for p in tiny_lm.parameters() if p.grad is not None]
    assert grads, "no parameter received a gradient"
    assert all(torch.isfinite(grad).all() for grad in grads)
    assert any(float(grad.abs().sum()) > 0 for grad in grads)


def test_zero_alpha_reduces_to_a_no_op_teacher(tiny_lm, tiny_topology, heads):
    """alpha = 0 must give an identical teacher, hence exactly zero loss."""

    outcome = run_pvsd_step(tiny_lm, tiny_topology, heads, alpha=0.0)
    torch.testing.assert_close(
        outcome["steered_logits"], outcome["student_logits"], atol=0, rtol=0
    )
    assert float(outcome["loss"].detach()) == 0.0

    # d/dtheta KL(p_theta || q) is analytically zero at p_theta == q, but the
    # per-vocabulary terms only cancel to float32 precision, so this is noise
    # (~1e-9) rather than an exact zero.
    outcome["loss"].backward()
    grads = [p.grad for p in tiny_lm.parameters() if p.grad is not None]
    assert grads
    assert all(float(grad.abs().max()) < 1e-7 for grad in grads)


def test_identical_corrupt_context_purifies_the_vector_to_zero(tiny_lm, tiny_topology, heads):
    """The semantic end-to-end check of contrastive purification.

    If the corrupted context *is* the real context, the two read-outs are the same
    vector, so the purified vector must vanish and the teacher must collapse onto
    the student. Any leak of an un-purified term would break this.
    """

    outcome = run_pvsd_step(tiny_lm, tiny_topology, heads, alpha=1.0, identical_corrupt=True)
    torch.testing.assert_close(
        outcome["fused"], torch.zeros_like(outcome["fused"]), atol=1e-5, rtol=0
    )
    assert float(outcome["loss"].detach()) == pytest.approx(0.0, abs=1e-9)


def test_multiple_views_fuse_to_the_mean_of_their_transfers(tiny_lm, tiny_topology):
    views = ("full_solution", "partial_solution", "answer_only")
    heads = all_heads_in_layers(tiny_topology, [1])
    outcome = run_pvsd_step(tiny_lm, tiny_topology, heads, views=views)

    per_view = outcome["per_view"]
    assert [item.view for item in per_view] == list(views)
    expected = torch.stack([item.transfer for item in per_view], dim=1).mean(dim=1)
    torch.testing.assert_close(outcome["fused"], expected, atol=1e-6, rtol=1e-6)
    # Each view really is a different prompt, so the transfers must differ.
    assert not torch.allclose(per_view[0].transfer, per_view[1].transfer)


def test_batch_pipeline_matches_manual_extraction(tiny_lm, tiny_topology, heads):
    """``build_privilege_vectors_from_batch`` == extract + purify done by hand."""

    batch = make_pvsd_batch(num_corrupt=2)
    fused, per_view = build_privilege_vectors_from_batch(
        tiny_lm, tiny_topology, ("full_solution",), batch, heads, micro_batch_size=3
    )

    raw = extract_privilege_vector(
        tiny_lm,
        tiny_topology,
        batch["pvsd_full_solution_input_ids"],
        batch["pvsd_full_solution_attention_mask"],
        heads,
    )
    corrupt_ids = batch["pvsd_full_solution_corrupt_input_ids"]
    corrupt_mask = batch["pvsd_full_solution_corrupt_attention_mask"]
    rows, num_corrupt, seq_len = corrupt_ids.shape
    corrupt = extract_privilege_vector(
        tiny_lm,
        tiny_topology,
        corrupt_ids.reshape(rows * num_corrupt, seq_len),
        corrupt_mask.reshape(rows * num_corrupt, seq_len),
        heads,
    ).view(rows, num_corrupt, -1)

    torch.testing.assert_close(per_view[0].raw, raw, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(per_view[0].corrupt, corrupt, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(
        fused, purify_privilege_vector(raw, corrupt), atol=1e-5, rtol=1e-4
    )


def test_pipeline_rejects_a_flat_corrupt_tensor(tiny_lm, tiny_topology, heads):
    batch = make_pvsd_batch()
    batch["pvsd_full_solution_corrupt_input_ids"] = batch["pvsd_full_solution_corrupt_input_ids"][
        :, 0, :
    ]
    with pytest.raises(ValueError, match=r"\[batch, num_corrupt, seq\]"):
        build_privilege_vectors_from_batch(
            tiny_lm, tiny_topology, ("full_solution",), batch, heads
        )


def test_per_view_head_sets_are_used_for_their_own_view(tiny_lm, tiny_topology):
    """PIE localises one A^(m) per view; each view must read from its own heads."""

    views = ("full_solution", "partial_solution")
    heads_by_view = {
        "full_solution": ((0, 0),),
        "partial_solution": ((2, 3),),
    }
    batch = make_pvsd_batch(views=views)
    _, per_view = build_privilege_vectors_from_batch(
        tiny_lm, tiny_topology, views, batch, heads_by_view
    )
    assert [item.heads for item in per_view] == [((0, 0),), ((2, 3),)]

    # Reading the same view with a different head set must give a different vector.
    _, swapped = build_privilege_vectors_from_batch(
        tiny_lm, tiny_topology, views, batch, {view: ((1, 1),) for view in views}
    )
    assert not torch.allclose(per_view[0].raw, swapped[0].raw)


def test_purification_ablations_change_only_the_injected_signal(tiny_lm, tiny_topology, heads):
    batch = make_pvsd_batch()
    outputs = {}
    for mode in ("contrast", "none", "template_only"):
        fused, per_view = build_privilege_vectors_from_batch(
            tiny_lm, tiny_topology, ("full_solution",), batch, heads, purification=mode
        )
        outputs[mode] = (fused, per_view[0])

    # The raw read-out is identical across modes: the ablations are cost-matched.
    torch.testing.assert_close(outputs["contrast"][1].raw, outputs["none"][1].raw)
    torch.testing.assert_close(outputs["contrast"][1].corrupt, outputs["template_only"][1].corrupt)
    # ... and the injected signals decompose exactly.
    torch.testing.assert_close(
        outputs["contrast"][0], outputs["none"][0] - outputs["template_only"][0], atol=1e-6, rtol=1e-6
    )
    assert outputs["none"][1].purification == "none"
    with pytest.raises(ValueError):
        build_privilege_vectors_from_batch(
            tiny_lm, tiny_topology, ("full_solution",), batch, heads, purification="bogus"
        )


def test_collator_output_feeds_the_vector_pipeline_directly(
    tiny_lm, tiny_topology, fake_tokenizer, math_features
):
    """The seam: whatever the collator emits must be exactly what the builder reads."""

    from pvsd.math.data_collator import SelfDistillationDataCollator

    views = ("full_solution", "answer_only")
    collator = SelfDistillationDataCollator(
        tokenizer=fake_tokenizer,
        max_length=256,
        reason_first=False,
        multi_view_mode="single",
        pvsd_views=views,
        pvsd_num_corrupt=2,
    )
    batch = collator(math_features)
    # The fake tokenizer's vocabulary is unbounded; fold ids into the tiny model's.
    for key, value in list(batch.items()):
        if key.endswith("input_ids") and isinstance(value, torch.Tensor):
            batch[key] = value % VOCAB

    fused, per_view = build_privilege_vectors_from_batch(
        tiny_lm,
        tiny_topology,
        tuple(batch["pvsd_view_names"]),
        batch,
        all_heads_in_layers(tiny_topology, [0]),
        micro_batch_size=4,
    )
    assert fused.shape == (len(math_features), tiny_topology.resid_dim)
    assert [item.view for item in per_view] == list(views)
    assert torch.isfinite(fused).all()


def test_larger_alpha_moves_the_teacher_further(tiny_lm, tiny_topology, heads):
    losses = [
        float(run_pvsd_step(tiny_lm, tiny_topology, heads, alpha=alpha)["loss"].detach())
        for alpha in (0.5, 2.0)
    ]
    assert losses[1] > losses[0]
