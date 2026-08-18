"""The collator's PVSD prompt sets: real vs corrupted contexts."""

from __future__ import annotations

import pytest
import torch

from pvsd.math.data_collator import SelfDistillationDataCollator


def build_collator(
    tokenizer,
    views=("full_solution",),
    num_corrupt=2,
    max_length=512,
    corrupt_match="cycle",
):
    return SelfDistillationDataCollator(
        tokenizer=tokenizer,
        max_length=max_length,
        reason_first=False,
        multi_view_mode="single",
        pvsd_views=views,
        pvsd_num_corrupt=num_corrupt,
        pvsd_corrupt_match=corrupt_match,
    )


def test_real_prompt_carries_its_own_reference(fake_tokenizer, math_features):
    collator = build_collator(fake_tokenizer)
    batch = collator(math_features)

    decoded = fake_tokenizer.batch_decode(batch["pvsd_full_solution_input_ids"])
    for index, text in enumerate(decoded):
        assert f"problem_{index}" in text
        assert f"step_{index}_one" in text
        assert f"answer_{index}}}" in text or f"answer_{index}" in text


def test_corrupted_prompt_keeps_the_question_and_swaps_the_reference(fake_tokenizer, math_features):
    collator = build_collator(fake_tokenizer, num_corrupt=2)
    batch = collator(math_features)
    corrupt_ids = batch["pvsd_full_solution_corrupt_input_ids"]
    assert corrupt_ids.shape[:2] == (4, 2)

    for index in range(4):
        for shift in range(2):
            text = fake_tokenizer.decode(corrupt_ids[index, shift])
            partner = (index + shift + 1) % 4
            # same question ...
            assert f"problem_{index}" in text
            # ... other problem's reference, and not its own
            assert f"step_{partner}_one" in text
            assert f"step_{index}_one" not in text


def test_real_and_corrupted_prompts_share_the_view_wording(fake_tokenizer, math_features):
    collator = build_collator(fake_tokenizer, num_corrupt=1)
    batch = collator(math_features)
    real = fake_tokenizer.decode(batch["pvsd_full_solution_input_ids"][0])
    corrupt = fake_tokenizer.decode(batch["pvsd_full_solution_corrupt_input_ids"][0, 0])
    for marker in ("Privileged", "Reference", "(full_solution)", "<|assistant|>"):
        assert marker in real, marker
        assert marker in corrupt, marker


def test_lengths_and_padding_are_consistent(fake_tokenizer, math_features):
    collator = build_collator(fake_tokenizer, num_corrupt=2)
    batch = collator(math_features)

    ids = batch["pvsd_full_solution_input_ids"]
    mask = batch["pvsd_full_solution_attention_mask"]
    lengths = batch["pvsd_full_solution_lengths"]
    torch.testing.assert_close(mask.sum(dim=-1), lengths)
    for row, length in enumerate(lengths.tolist()):
        # right padding: everything after `length` is padding
        assert bool((ids[row, length:] == fake_tokenizer.pad_token_id).all())
        assert bool((mask[row, :length] == 1).all())

    corrupt_mask = batch["pvsd_full_solution_corrupt_attention_mask"]
    corrupt_lengths = batch["pvsd_full_solution_corrupt_lengths"]
    assert corrupt_lengths.shape == (4, 2)
    torch.testing.assert_close(corrupt_mask.sum(dim=-1), corrupt_lengths)


def test_multiple_views_are_all_emitted(fake_tokenizer, math_features):
    views = ("full_solution", "partial_solution", "answer_only")
    collator = build_collator(fake_tokenizer, views=views)
    batch = collator(math_features)

    assert tuple(batch["pvsd_view_names"]) == views
    assert batch["pvsd_num_corrupt"] == 2
    for view in views:
        assert f"pvsd_{view}_input_ids" in batch
        assert f"pvsd_{view}_corrupt_input_ids" in batch

    answer_only = fake_tokenizer.decode(batch["pvsd_answer_only_input_ids"][0])
    assert "answer_0" in answer_only
    # answer_only must not leak the intermediate steps
    assert "step_0_one" not in answer_only


def test_pvsd_mode_does_not_build_a_teacher_sequence(fake_tokenizer, math_features):
    collator = build_collator(fake_tokenizer)
    batch = collator(math_features)
    assert "teacher_prompts" not in batch
    assert "teacher_prompt_length" not in batch
    # the student side is untouched
    assert "student_prompts" in batch
    assert batch["student_prompt_length"] == batch["student_prompts"].shape[1]


def test_num_corrupt_is_clamped_to_the_batch(fake_tokenizer, math_features):
    collator = build_collator(fake_tokenizer, num_corrupt=8)
    batch = collator(math_features)
    assert batch["pvsd_num_corrupt"] == 3  # batch_size - 1
    assert batch["pvsd_full_solution_corrupt_input_ids"].shape[1] == 3


def test_single_example_batch_is_rejected(fake_tokenizer, math_features):
    collator = build_collator(fake_tokenizer)
    with pytest.raises(ValueError, match="in-batch corrupted context"):
        collator(math_features[:1])


def test_collator_validates_pvsd_settings(fake_tokenizer):
    with pytest.raises(ValueError, match="Unknown PVSD view"):
        build_collator(fake_tokenizer, views=("nonexistent_view",))
    with pytest.raises(ValueError, match="pvsd_num_corrupt"):
        build_collator(fake_tokenizer, num_corrupt=0)
    with pytest.raises(ValueError, match="pvsd_corrupt_match"):
        build_collator(fake_tokenizer, corrupt_match="nearest_topic")


def test_length_matching_picks_the_closest_donor(fake_tokenizer):
    """The corrupted reference should match the real one in approximate length."""

    features = [
        {"problem": "p0", "solution": "a " * 2},
        {"problem": "p1", "solution": "b " * 40},
        {"problem": "p2", "solution": "c " * 3},
        {"problem": "p3", "solution": "d " * 41},
    ]
    collator = build_collator(fake_tokenizer, num_corrupt=1, corrupt_match="length")
    partners = collator._corrupt_partners([row["solution"] for row in features], 1)
    assert partners == [[2], [3], [0], [1]]

    # cycle mode ignores length and just rotates
    cycle = build_collator(fake_tokenizer, num_corrupt=1, corrupt_match="cycle")
    assert cycle._corrupt_partners([row["solution"] for row in features], 1) == [[1], [2], [3], [0]]


def test_length_matching_never_picks_itself_and_is_deterministic(fake_tokenizer):
    solutions = ["x " * 10 for _ in range(4)]  # all ties
    collator = build_collator(fake_tokenizer, num_corrupt=3, corrupt_match="length")
    partners = collator._corrupt_partners(solutions, 3)
    for index, donors in enumerate(partners):
        assert index not in donors
        assert len(set(donors)) == 3
    assert partners == collator._corrupt_partners(solutions, 3)


def test_length_matched_corrupt_prompts_are_closer_in_length(fake_tokenizer):
    features = [
        {"problem": f"p{index}", "solution": "w " * length}
        for index, length in enumerate((2, 60, 3, 61))
    ]
    deltas = {}
    for mode in ("cycle", "length"):
        batch = build_collator(fake_tokenizer, num_corrupt=1, corrupt_match=mode)(features)
        real = batch["pvsd_full_solution_lengths"].float()
        corrupt = batch["pvsd_full_solution_corrupt_lengths"].float()
        deltas[mode] = float((corrupt - real.unsqueeze(-1)).abs().mean())
    assert deltas["length"] < deltas["cycle"]


def test_legacy_single_view_path_still_works(fake_tokenizer, math_features):
    """Without pvsd_views the collator behaves exactly as before."""

    collator = SelfDistillationDataCollator(
        tokenizer=fake_tokenizer,
        max_length=512,
        reason_first=False,
        multi_view_mode="single",
        single_view_pi="full_solution",
    )
    batch = collator(math_features)
    assert "teacher_prompts" in batch
    assert "pvsd_view_names" not in batch
    assert batch["teacher_prompt_length"] == batch["teacher_prompts"].shape[1]
