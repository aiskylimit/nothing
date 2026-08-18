"""Head read-out, purification and fusion."""

from __future__ import annotations

import pytest
import torch

from pvsd.common.privilege_vectors import (
    ModelTopology,
    all_heads_in_layers,
    apply_purification,
    capture_head_activations,
    cosine_similarity_rowwise,
    head_set_jaccard,
    resolve_heads_by_view,
    default_injection_layer,
    describe_model,
    extract_privilege_vector,
    fuse_view_vectors,
    get_attn_out_proj,
    get_decoder_layers,
    head_contribution_to_residual,
    last_real_token_index,
    normalize_head_set,
    out_proj_head_weight,
    position_ids_from_mask,
    privilege_vector_from_activations,
    purify_privilege_vector,
)

from conftest import TINY_CONFIG, make_padded_batch


# ---------------------------------------------------------------------------
# topology
# ---------------------------------------------------------------------------


def test_describe_model_reads_head_dim_from_out_projection(tiny_lm):
    topology = describe_model(tiny_lm)
    assert topology.num_layers == TINY_CONFIG["num_hidden_layers"]
    assert topology.num_heads == TINY_CONFIG["num_attention_heads"]
    assert topology.resid_dim == TINY_CONFIG["hidden_size"]
    # The point of the test: head_dim must come from o_proj, not from the hidden size.
    assert topology.head_dim == TINY_CONFIG["head_dim"]
    assert topology.head_dim != TINY_CONFIG["hidden_size"] // TINY_CONFIG["num_attention_heads"]
    assert topology.attn_inner_dim == get_attn_out_proj(tiny_lm, 0).in_features


def test_head_slice_covers_the_projection_input_exactly(tiny_topology):
    covered = torch.zeros(tiny_topology.attn_inner_dim, dtype=torch.bool)
    for head_idx in range(tiny_topology.num_heads):
        covered[tiny_topology.head_slice(head_idx)] = True
    assert bool(covered.all())
    with pytest.raises(IndexError):
        tiny_topology.head_slice(tiny_topology.num_heads)


def test_get_decoder_layers_and_out_proj(tiny_lm):
    layers = get_decoder_layers(tiny_lm)
    assert len(layers) == TINY_CONFIG["num_hidden_layers"]
    assert get_attn_out_proj(tiny_lm, 1) is layers[1].self_attn.o_proj


def test_default_injection_layer():
    assert default_injection_layer(36, "quarter") == 9
    assert default_injection_layer(36, "third") == 12
    assert default_injection_layer(36, "half") == 18
    with pytest.raises(ValueError):
        default_injection_layer(36, "fifth")


# ---------------------------------------------------------------------------
# the head decomposition is exact
# ---------------------------------------------------------------------------


def test_head_contributions_sum_to_the_full_projection(tiny_lm, tiny_topology):
    """Summing all heads must reproduce o_proj(z): the decomposition is exact."""

    torch.manual_seed(0)
    activation = torch.randn(2, tiny_topology.attn_inner_dim)
    out_proj = get_attn_out_proj(tiny_lm, 0)
    assert out_proj.bias is None, "bias handling assumption changed"

    reference = out_proj(activation)
    total = torch.zeros_like(reference)
    for head_idx in range(tiny_topology.num_heads):
        head_slice = tiny_topology.head_slice(head_idx)
        total = total + head_contribution_to_residual(
            tiny_lm, tiny_topology, 0, head_idx, activation[:, head_slice]
        )
    torch.testing.assert_close(total, reference, atol=1e-5, rtol=1e-5)


def test_privilege_vector_is_the_sum_of_selected_heads(tiny_lm, tiny_topology):
    torch.manual_seed(1)
    activations = {0: torch.randn(3, tiny_topology.attn_inner_dim)}
    heads = [(0, 1), (0, 3)]
    vector = privilege_vector_from_activations(tiny_lm, tiny_topology, activations, heads)

    expected = torch.zeros(3, tiny_topology.resid_dim)
    for layer_idx, head_idx in heads:
        expected = expected + head_contribution_to_residual(
            tiny_lm,
            tiny_topology,
            layer_idx,
            head_idx,
            activations[layer_idx][:, tiny_topology.head_slice(head_idx)],
        )
    torch.testing.assert_close(vector, expected, atol=1e-6, rtol=1e-6)
    assert vector.dtype == torch.float32


def test_privilege_vector_rejects_missing_layers(tiny_lm, tiny_topology):
    activations = {0: torch.randn(1, tiny_topology.attn_inner_dim)}
    with pytest.raises(KeyError):
        privilege_vector_from_activations(tiny_lm, tiny_topology, activations, [(1, 0)])


# ---------------------------------------------------------------------------
# activation capture
# ---------------------------------------------------------------------------


def test_capture_head_activations_matches_a_manual_hook(tiny_lm, tiny_topology):
    input_ids, attention_mask = make_padded_batch([[3, 4, 5, 6], [7, 8, 9, 10]])
    positions = torch.tensor([2, 3])

    manual = {}

    def manual_hook(module, args):
        del module
        manual["value"] = args[0].detach().clone()
        return None

    handle = get_attn_out_proj(tiny_lm, 1).register_forward_pre_hook(manual_hook)
    try:
        with capture_head_activations(tiny_lm, [1], positions) as store:
            tiny_lm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids_from_mask(attention_mask),
                use_cache=False,
            )
    finally:
        handle.remove()

    expected = torch.stack([manual["value"][0, 2], manual["value"][1, 3]]).float()
    torch.testing.assert_close(store[1], expected, atol=1e-6, rtol=1e-6)
    assert store[1].shape == (2, tiny_topology.attn_inner_dim)


def test_capture_removes_all_hooks(tiny_lm):
    out_proj = get_attn_out_proj(tiny_lm, 0)
    before = len(out_proj._forward_pre_hooks)
    with capture_head_activations(tiny_lm, [0, 2], torch.tensor([0])):
        assert len(out_proj._forward_pre_hooks) == before + 1
    assert len(out_proj._forward_pre_hooks) == before


def test_read_out_position_ignores_right_padding(tiny_lm, tiny_topology):
    """A padded row must give the same read-out as the same sequence unpadded."""

    short = [11, 12, 13]
    long = [21, 22, 23, 24, 25]
    input_ids, attention_mask = make_padded_batch([short, long])
    heads = [(2, 0), (1, 2)]

    batched = extract_privilege_vector(tiny_lm, tiny_topology, input_ids, attention_mask, heads)

    alone_ids = torch.tensor([short], dtype=torch.long)
    alone_mask = torch.ones_like(alone_ids)
    alone = extract_privilege_vector(tiny_lm, tiny_topology, alone_ids, alone_mask, heads)

    torch.testing.assert_close(batched[0], alone[0], atol=1e-4, rtol=1e-4)


def test_extract_micro_batching_is_equivalent(tiny_lm, tiny_topology):
    input_ids, attention_mask = make_padded_batch([[3, 4], [5, 6, 7], [8, 9, 10, 11], [12, 13]])
    heads = [(0, 0), (2, 3)]
    whole = extract_privilege_vector(tiny_lm, tiny_topology, input_ids, attention_mask, heads)
    chunked = extract_privilege_vector(
        tiny_lm, tiny_topology, input_ids, attention_mask, heads, micro_batch_size=1
    )
    torch.testing.assert_close(whole, chunked, atol=1e-4, rtol=1e-4)


def test_extract_does_not_require_grad(tiny_lm, tiny_topology):
    input_ids, attention_mask = make_padded_batch([[3, 4, 5], [6, 7, 8]])
    vector = extract_privilege_vector(tiny_lm, tiny_topology, input_ids, attention_mask, [(0, 0)])
    assert not vector.requires_grad


# ---------------------------------------------------------------------------
# positions
# ---------------------------------------------------------------------------


def test_position_ids_from_mask_skips_pad_gaps():
    # prompt(3) + pad(2) + completion(2): the completion must continue from 3,
    # not from 5, so the training pass matches how the rollout was sampled.
    attention_mask = torch.tensor([[1, 1, 1, 0, 0, 1, 1]])
    torch.testing.assert_close(
        position_ids_from_mask(attention_mask),
        torch.tensor([[0, 1, 2, 2, 2, 3, 4]]),
    )


def test_position_ids_are_zero_based_per_row():
    attention_mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1]])
    positions = position_ids_from_mask(attention_mask)
    assert positions[0, 0].item() == 0 and positions[1, 0].item() == 0
    assert positions[0, 1].item() == 1 and positions[1, 3].item() == 3


def test_last_real_token_index():
    attention_mask = torch.tensor([[1, 1, 1, 0], [1, 0, 0, 0]])
    torch.testing.assert_close(last_real_token_index(attention_mask), torch.tensor([2, 0]))
    with pytest.raises(ValueError):
        last_real_token_index(torch.tensor([[0, 0]]))


# ---------------------------------------------------------------------------
# purification and fusion
# ---------------------------------------------------------------------------


def test_purify_subtracts_the_corrupted_mean():
    raw = torch.tensor([[3.0, 6.0], [1.0, 1.0]])
    corrupt = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[0.0, 0.0], [2.0, 2.0]]])
    torch.testing.assert_close(
        purify_privilege_vector(raw, corrupt),
        torch.tensor([[1.0, 3.0], [0.0, 0.0]]),
    )


def test_purify_is_zero_when_the_contexts_are_identical():
    raw = torch.randn(3, 5)
    corrupt = raw.unsqueeze(1).repeat(1, 4, 1)
    torch.testing.assert_close(
        purify_privilege_vector(raw, corrupt), torch.zeros_like(raw), atol=1e-6, rtol=0
    )


@pytest.mark.parametrize(
    "raw_shape, corrupt_shape",
    [((2, 4), (2, 4)), ((2, 4), (3, 1, 4)), ((2, 4), (2, 1, 5)), ((2, 4), (2, 0, 4))],
)
def test_purify_validates_shapes(raw_shape, corrupt_shape):
    with pytest.raises(ValueError):
        purify_privilege_vector(torch.zeros(raw_shape), torch.zeros(corrupt_shape))


def test_fuse_uniform_is_the_mean():
    views = torch.tensor([[[1.0, 1.0], [3.0, 5.0]]])
    torch.testing.assert_close(fuse_view_vectors(views), torch.tensor([[2.0, 3.0]]))


def test_fuse_with_weights_normalises():
    views = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    weights = torch.tensor([[3.0, 1.0]])
    torch.testing.assert_close(
        fuse_view_vectors(views, weights), torch.tensor([[0.75, 0.25]])
    )


def test_fuse_validates_shapes():
    with pytest.raises(ValueError):
        fuse_view_vectors(torch.zeros(2, 4))
    with pytest.raises(ValueError):
        fuse_view_vectors(torch.zeros(2, 3, 4), torch.zeros(2, 2))


def test_purification_modes_select_the_right_signal():
    raw = torch.tensor([[10.0, 0.0]])
    corrupt = torch.tensor([[[2.0, 0.0], [4.0, 0.0]]])  # mean = [3, 0]

    torch.testing.assert_close(
        apply_purification(raw, corrupt, "contrast"), torch.tensor([[7.0, 0.0]])
    )
    torch.testing.assert_close(apply_purification(raw, corrupt, "none"), raw)
    torch.testing.assert_close(
        apply_purification(raw, corrupt, "template_only"), torch.tensor([[3.0, 0.0]])
    )
    # contrast = none - template_only, so the ablations decompose the method exactly
    torch.testing.assert_close(
        apply_purification(raw, corrupt, "contrast"),
        apply_purification(raw, corrupt, "none") - apply_purification(raw, corrupt, "template_only"),
    )
    with pytest.raises(ValueError):
        apply_purification(raw, corrupt, "invalid")


def test_head_set_jaccard():
    assert head_set_jaccard([(0, 1), (0, 2)], [(0, 1), (0, 2)]) == 1.0
    assert head_set_jaccard([(0, 1)], [(0, 2)]) == 0.0
    assert head_set_jaccard([(0, 1), (0, 2)], [(0, 2), (0, 3)]) == pytest.approx(1 / 3)
    assert head_set_jaccard([], []) == 0.0


def test_resolve_heads_by_view_accepts_shared_or_per_view():
    views = ("a", "b")
    shared = resolve_heads_by_view(views, [(0, 1)])
    assert shared == {"a": ((0, 1),), "b": ((0, 1),)}

    per_view = resolve_heads_by_view(views, {"a": [(0, 1)], "b": [(2, 3)]})
    assert per_view == {"a": ((0, 1),), "b": ((2, 3),)}

    with pytest.raises(KeyError):
        resolve_heads_by_view(views, {"a": [(0, 1)]})


def test_cosine_similarity_rowwise():
    left = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    right = torch.tensor([[2.0, 0.0], [0.0, 5.0]])
    torch.testing.assert_close(
        cosine_similarity_rowwise(left, right), torch.tensor([1.0, 0.0]), atol=1e-6, rtol=0
    )


# ---------------------------------------------------------------------------
# head set bookkeeping
# ---------------------------------------------------------------------------


def test_normalize_head_set_dedups_and_validates(tiny_topology):
    heads = normalize_head_set([(0, 1), (0, 1), (2, 3)], tiny_topology)
    assert heads == ((0, 1), (2, 3))
    with pytest.raises(IndexError):
        normalize_head_set([(99, 0)], tiny_topology)
    with pytest.raises(IndexError):
        normalize_head_set([(0, 99)], tiny_topology)
    with pytest.raises(ValueError):
        normalize_head_set([], tiny_topology)


def test_all_heads_in_layers(tiny_topology):
    heads = all_heads_in_layers(tiny_topology, [1])
    assert len(heads) == tiny_topology.num_heads
    assert {layer for layer, _ in heads} == {1}


# ---------------------------------------------------------------------------
# LoRA-aware output projection
# ---------------------------------------------------------------------------


def test_out_proj_head_weight_includes_the_lora_delta(tiny_lm, tiny_topology):
    peft = pytest.importorskip("peft")

    config = peft.LoraConfig(r=4, lora_alpha=8, target_modules=["o_proj"], lora_dropout=0.0)
    wrapped = peft.get_peft_model(tiny_lm, config)
    module = get_attn_out_proj(wrapped, 0)
    assert hasattr(module, "lora_A"), "expected o_proj to be LoRA-wrapped"

    # Random adapter weights: peft initialises lora_B to zeros, which would make
    # the delta trivially zero and the test vacuous.
    with torch.no_grad():
        for adapter in module.lora_B:
            module.lora_B[adapter].weight.normal_(std=0.3)
            module.lora_A[adapter].weight.normal_(std=0.3)

    head_slice = tiny_topology.head_slice(2)
    adapter = list(module.lora_A)[0]
    scaling = module.scaling[adapter]
    expected = module.base_layer.weight[:, head_slice].float() + scaling * (
        module.lora_B[adapter].weight.float() @ module.lora_A[adapter].weight.float()[:, head_slice]
    )
    torch.testing.assert_close(
        out_proj_head_weight(module, head_slice), expected, atol=1e-6, rtol=1e-6
    )

    # And the per-head decomposition still reproduces the wrapped module's output.
    topology = describe_model(wrapped)
    activation = torch.randn(2, topology.attn_inner_dim)
    reference = module(activation)
    total = torch.zeros_like(reference)
    for head_idx in range(topology.num_heads):
        total = total + head_contribution_to_residual(
            wrapped, topology, 0, head_idx, activation[:, topology.head_slice(head_idx)]
        )
    torch.testing.assert_close(total, reference, atol=1e-5, rtol=1e-5)


def test_out_proj_head_weight_skips_merged_adapters(tiny_lm, tiny_topology):
    peft = pytest.importorskip("peft")

    config = peft.LoraConfig(r=4, lora_alpha=8, target_modules=["o_proj"], lora_dropout=0.0)
    wrapped = peft.get_peft_model(tiny_lm, config)
    module = get_attn_out_proj(wrapped, 0)
    with torch.no_grad():
        for adapter in module.lora_B:
            module.lora_B[adapter].weight.normal_(std=0.3)

    head_slice = tiny_topology.head_slice(0)
    unmerged = out_proj_head_weight(module, head_slice)
    module.merge()
    merged = out_proj_head_weight(module, head_slice)
    # After merging, the delta lives in base_layer.weight and must not be added twice.
    torch.testing.assert_close(unmerged, merged, atol=1e-5, rtol=1e-5)


def test_describe_model_works_through_a_peft_wrapper(tiny_lm):
    peft = pytest.importorskip("peft")

    config = peft.LoraConfig(r=4, lora_alpha=8, target_modules=["o_proj", "q_proj"], lora_dropout=0.0)
    wrapped = peft.get_peft_model(tiny_lm, config)
    assert describe_model(wrapped) == ModelTopology(
        num_layers=TINY_CONFIG["num_hidden_layers"],
        num_heads=TINY_CONFIG["num_attention_heads"],
        head_dim=TINY_CONFIG["head_dim"],
        resid_dim=TINY_CONFIG["hidden_size"],
    )
    assert len(get_decoder_layers(wrapped)) == TINY_CONFIG["num_hidden_layers"]
