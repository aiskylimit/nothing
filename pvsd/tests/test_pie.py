"""Stage 1: PIE head localisation by causal mediation."""

from __future__ import annotations

import pytest
import torch

from pvsd.common.pie import (
    compute_privilege_indirect_effect,
    estimate_pie_forward_passes,
    parse_candidate_layers,
    patch_heads,
)
from pvsd.common.privilege_vectors import (
    capture_head_activations,
    last_real_token_index,
    position_ids_from_mask,
)

from conftest import make_padded_batch


def _logits(model, input_ids, attention_mask):
    return model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids_from_mask(attention_mask),
        use_cache=False,
    ).logits


def _clean_and_corrupt():
    """Two privileged prompts and two 'corrupted' ones of similar length."""

    clean_ids, clean_mask = make_padded_batch([[5, 6, 7, 8, 9], [10, 11, 12, 13]])
    corrupt_ids, corrupt_mask = make_padded_batch([[5, 6, 31, 32, 33], [10, 11, 41, 42]])
    return clean_ids, clean_mask, corrupt_ids, corrupt_mask


def test_patching_with_its_own_activations_is_a_no_op(tiny_lm, tiny_topology):
    """Sanity of the patch mechanics: replacing a value with itself changes nothing."""

    input_ids, attention_mask = make_padded_batch([[3, 4, 5, 6]])
    positions = last_real_token_index(attention_mask)
    layers = [0, 1, 2]

    with capture_head_activations(tiny_lm, layers, positions) as store:
        baseline = _logits(tiny_lm, input_ids, attention_mask)
    own = {layer: activation[0] for layer, activation in store.items()}

    assignments = {layer: [(0, head)] for layer, head in [(0, 1), (1, 2), (2, 3)]}
    with patch_heads(tiny_lm, tiny_topology, int(positions[0]), assignments, own) as hooks:
        patched = _logits(tiny_lm, input_ids, attention_mask)
    assert all(hook.call_count == 1 for hook in hooks.values())
    torch.testing.assert_close(patched, baseline, atol=0, rtol=0)


def test_patching_changes_the_output_on_a_corrupted_input(tiny_lm, tiny_topology):
    clean_ids, clean_mask, corrupt_ids, corrupt_mask = _clean_and_corrupt()
    clean_ids, clean_mask = clean_ids[:1], clean_mask[:1]
    corrupt_ids, corrupt_mask = corrupt_ids[:1], corrupt_mask[:1]

    clean_positions = last_real_token_index(clean_mask)
    with capture_head_activations(tiny_lm, [1], clean_positions) as store:
        _logits(tiny_lm, clean_ids, clean_mask)
    replacement = {1: store[1][0]}

    position = int(last_real_token_index(corrupt_mask)[0])
    baseline = _logits(tiny_lm, corrupt_ids, corrupt_mask)
    with patch_heads(tiny_lm, tiny_topology, position, {1: [(0, 0)]}, replacement):
        patched = _logits(tiny_lm, corrupt_ids, corrupt_mask)
    assert not torch.allclose(patched, baseline)


def test_patch_only_touches_the_assigned_rows(tiny_lm, tiny_topology):
    input_ids, attention_mask = make_padded_batch([[3, 4, 5], [6, 7, 8]])
    baseline = _logits(tiny_lm, input_ids, attention_mask)
    replacement = {0: torch.full((tiny_topology.attn_inner_dim,), 3.0)}
    with patch_heads(tiny_lm, tiny_topology, 2, {0: [(1, 0)]}, replacement):
        patched = _logits(tiny_lm, input_ids, attention_mask)
    torch.testing.assert_close(patched[0], baseline[0], atol=0, rtol=0)
    assert not torch.allclose(patched[1], baseline[1])


def test_pie_returns_a_valid_head_set(tiny_lm, tiny_topology):
    clean_ids, clean_mask, corrupt_ids, corrupt_mask = _clean_and_corrupt()
    result = compute_privilege_indirect_effect(
        tiny_lm,
        tiny_topology,
        clean_ids,
        clean_mask,
        corrupt_ids,
        corrupt_mask,
        top_k_heads=5,
        head_chunk_size=4,
    )
    assert len(result.top_heads) == 5
    assert len(set(result.top_heads)) == 5
    for layer_idx, head_idx in result.top_heads:
        assert 0 <= layer_idx < tiny_topology.num_layers
        assert 0 <= head_idx < tiny_topology.num_heads
    assert result.scores.shape == (tiny_topology.num_layers, tiny_topology.num_heads)
    assert torch.isfinite(result.scores).all()
    assert torch.isfinite(result.kl).all()
    assert bool((result.kl >= 0).all())
    assert result.num_examples == 2
    assert result.num_forward_passes == estimate_pie_forward_passes(
        2, tiny_topology.num_layers * tiny_topology.num_heads, 4
    )


def test_pie_restricted_to_candidate_layers_never_selects_others(tiny_lm, tiny_topology):
    clean_ids, clean_mask, corrupt_ids, corrupt_mask = _clean_and_corrupt()
    result = compute_privilege_indirect_effect(
        tiny_lm,
        tiny_topology,
        clean_ids,
        clean_mask,
        corrupt_ids,
        corrupt_mask,
        top_k_heads=3,
        candidate_layers=[2],
        head_chunk_size=2,
    )
    assert {layer for layer, _ in result.top_heads} == {2}
    assert torch.isinf(result.scores[0]).all() and torch.isinf(result.kl[0]).all()
    assert torch.isfinite(result.scores[2]).all()


def test_pie_head_chunking_is_equivalent(tiny_lm, tiny_topology):
    clean_ids, clean_mask, corrupt_ids, corrupt_mask = _clean_and_corrupt()
    kwargs = dict(
        top_k_heads=4,
        candidate_layers=[0, 1, 2],
    )
    one = compute_privilege_indirect_effect(
        tiny_lm, tiny_topology, clean_ids, clean_mask, corrupt_ids, corrupt_mask,
        head_chunk_size=1, **kwargs,
    )
    many = compute_privilege_indirect_effect(
        tiny_lm, tiny_topology, clean_ids, clean_mask, corrupt_ids, corrupt_mask,
        head_chunk_size=6, **kwargs,
    )
    torch.testing.assert_close(one.kl, many.kl, atol=1e-5, rtol=1e-3)
    assert one.num_forward_passes > many.num_forward_passes


def test_pie_is_degenerate_when_the_corrupt_context_equals_the_clean_one(tiny_lm, tiny_topology):
    """If nothing is corrupted, no head can be causally responsible for anything.

    Patching one head with the (single-example) clean mean then reproduces the clean
    distribution exactly, so every KL is 0 and no head stands out.
    """

    clean_ids, clean_mask = make_padded_batch([[5, 6, 7, 8]])
    result = compute_privilege_indirect_effect(
        tiny_lm,
        tiny_topology,
        clean_ids,
        clean_mask,
        clean_ids.clone(),
        clean_mask.clone(),
        top_k_heads=2,
        head_chunk_size=3,
    )
    assert float(result.kl.max()) < 1e-6


def test_pie_score_modes_agree_on_a_single_example(tiny_lm, tiny_topology):
    """With one example, mean(1/kl) and mean(-kl) are both monotone in kl."""

    clean_ids, clean_mask, corrupt_ids, corrupt_mask = _clean_and_corrupt()
    shared = dict(
        clean_input_ids=clean_ids[:1],
        clean_attention_mask=clean_mask[:1],
        corrupt_input_ids=corrupt_ids[:1],
        corrupt_attention_mask=corrupt_mask[:1],
        top_k_heads=1,
        head_chunk_size=4,
    )
    inverse = compute_privilege_indirect_effect(
        tiny_lm, tiny_topology, score_mode="inverse_kl", **shared
    )
    negative = compute_privilege_indirect_effect(
        tiny_lm, tiny_topology, score_mode="neg_kl", **shared
    )
    assert inverse.top_heads == negative.top_heads
    # The winner is the head whose patch recovers the clean distribution best.
    layer_idx, head_idx = inverse.top_heads[0]
    assert float(inverse.kl[layer_idx, head_idx]) == pytest.approx(float(inverse.kl.min()))


def test_pie_log_dict_is_scalar_only(tiny_lm, tiny_topology):
    clean_ids, clean_mask, corrupt_ids, corrupt_mask = _clean_and_corrupt()
    result = compute_privilege_indirect_effect(
        tiny_lm, tiny_topology, clean_ids, clean_mask, corrupt_ids, corrupt_mask,
        top_k_heads=2, head_chunk_size=4,
    )
    logs = result.as_log_dict()
    assert logs["pvsd/pie/num_examples"] == 2.0
    assert logs["pvsd/pie/num_heads_scanned"] == float(
        tiny_topology.num_layers * tiny_topology.num_heads
    )
    assert all(isinstance(value, float) for value in logs.values())


def test_pie_validates_arguments(tiny_lm, tiny_topology):
    clean_ids, clean_mask, corrupt_ids, corrupt_mask = _clean_and_corrupt()
    base = (tiny_lm, tiny_topology, clean_ids, clean_mask, corrupt_ids, corrupt_mask)

    with pytest.raises(ValueError):
        compute_privilege_indirect_effect(*base, top_k_heads=2, score_mode="nope")
    with pytest.raises(ValueError):
        compute_privilege_indirect_effect(*base, top_k_heads=0)
    with pytest.raises(ValueError):
        compute_privilege_indirect_effect(*base, top_k_heads=10_000)
    with pytest.raises(IndexError):
        compute_privilege_indirect_effect(*base, top_k_heads=2, candidate_layers=[99])
    with pytest.raises(ValueError):
        compute_privilege_indirect_effect(
            tiny_lm, tiny_topology, clean_ids, clean_mask, corrupt_ids[:1], corrupt_mask[:1],
            top_k_heads=2,
        )


def test_pie_is_invariant_to_extra_padding(tiny_lm, tiny_topology):
    """Rows are trimmed to their real tokens, so padding width must not matter."""

    clean = [[5, 6, 7, 8, 9], [10, 11, 12, 13]]
    corrupt = [[5, 6, 31, 32, 33], [10, 11, 41, 42]]
    kwargs = dict(top_k_heads=3, head_chunk_size=4, candidate_layers=[0, 1, 2])

    tight = compute_privilege_indirect_effect(
        tiny_lm, tiny_topology, *make_padded_batch(clean), *make_padded_batch(corrupt), **kwargs
    )
    # Same content, padded much wider.
    padded_clean = make_padded_batch(clean + [[1] * 12])
    padded_corrupt = make_padded_batch(corrupt + [[1] * 12])
    loose = compute_privilege_indirect_effect(
        tiny_lm,
        tiny_topology,
        padded_clean[0][:2],
        padded_clean[1][:2],
        padded_corrupt[0][:2],
        padded_corrupt[1][:2],
        **kwargs,
    )
    torch.testing.assert_close(tight.kl, loose.kl, atol=1e-6, rtol=1e-5)
    assert tight.top_heads == loose.top_heads


def test_pie_rejects_an_all_padding_row(tiny_lm, tiny_topology):
    clean_ids, clean_mask, corrupt_ids, corrupt_mask = _clean_and_corrupt()
    corrupt_mask = corrupt_mask.clone()
    corrupt_mask[1] = 0
    with pytest.raises(ValueError, match="all padding"):
        compute_privilege_indirect_effect(
            tiny_lm, tiny_topology, clean_ids, clean_mask, corrupt_ids, corrupt_mask,
            top_k_heads=2, head_chunk_size=4,
        )


def test_parse_candidate_layers():
    assert parse_candidate_layers(None) is None
    assert parse_candidate_layers("all") is None
    assert parse_candidate_layers("all", num_layers=3) == (0, 1, 2)
    assert parse_candidate_layers("8:12") == (8, 9, 10, 11)
    assert parse_candidate_layers("8:", num_layers=10) == (8, 9)
    assert parse_candidate_layers(":3") == (0, 1, 2)
    assert parse_candidate_layers("9,12,9") == (9, 12)
    assert parse_candidate_layers([3, 1, 3]) == (1, 3)

    with pytest.raises(ValueError):
        parse_candidate_layers("12:8")
    with pytest.raises(ValueError):
        parse_candidate_layers("8:")
    with pytest.raises(ValueError):
        parse_candidate_layers([])


def test_estimate_pie_forward_passes():
    # 1152 heads in chunks of 8 -> 144 patched passes plus 1 clean pass, per example.
    assert estimate_pie_forward_passes(2, 1152, 8) == 2 * (144 + 1)
    assert estimate_pie_forward_passes(1, 10, 4) == 1 * (3 + 1)
