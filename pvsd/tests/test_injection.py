"""Residual-stream injection: exactness, position alignment, hook hygiene."""

from __future__ import annotations

import pytest
import torch

from pvsd.common.privilege_vectors import (
    ResidualSteerHook,
    get_decoder_layers,
    inject_at_layer,
    position_ids_from_mask,
)

from conftest import make_padded_batch


def _forward(model, input_ids, attention_mask):
    return model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids_from_mask(attention_mask),
        use_cache=False,
    ).logits


def _layer_output(model, layer_idx, input_ids, attention_mask):
    captured = {}

    def hook(module, inputs, output):
        del module, inputs
        captured["value"] = (output[0] if isinstance(output, tuple) else output).detach().clone()

    handle = get_decoder_layers(model)[layer_idx].register_forward_hook(hook)
    try:
        _forward(model, input_ids, attention_mask)
    finally:
        handle.remove()
    return captured["value"]


def test_zero_vector_leaves_logits_unchanged(tiny_lm, tiny_topology):
    input_ids, attention_mask = make_padded_batch([[3, 4, 5, 6], [7, 8, 9, 10]])
    baseline = _forward(tiny_lm, input_ids, attention_mask)
    zeros = torch.zeros(2, tiny_topology.resid_dim)
    with torch.no_grad(), inject_at_layer(tiny_lm, 1, zeros, alpha=1.0, start_index=0):
        steered = _forward(tiny_lm, input_ids, attention_mask)
    torch.testing.assert_close(steered, baseline, atol=0, rtol=0)


def test_injection_adds_exactly_alpha_times_vector_at_steered_positions(tiny_lm, tiny_topology):
    input_ids, attention_mask = make_padded_batch([[3, 4, 5, 6, 7]])
    layer_idx = 1
    start = 2
    alpha = 2.5
    vector = torch.randn(1, tiny_topology.resid_dim)

    clean = _layer_output(tiny_lm, layer_idx, input_ids, attention_mask)
    with torch.no_grad(), inject_at_layer(
        tiny_lm, layer_idx, vector, alpha=alpha, start_index=start
    ):
        steered = _layer_output(tiny_lm, layer_idx, input_ids, attention_mask)

    delta = steered - clean
    torch.testing.assert_close(
        delta[:, start:, :],
        (alpha * vector).unsqueeze(1).expand(-1, delta.shape[1] - start, -1),
        atol=1e-5,
        rtol=1e-5,
    )
    torch.testing.assert_close(delta[:, :start, :], torch.zeros_like(delta[:, :start, :]))


def test_steering_is_causal_and_starts_exactly_at_start_index(tiny_lm, tiny_topology):
    """The guard against an off-by-one between measurement and injection.

    With ``start_index = p``, the logits at sequence index ``p`` (which predict the
    token at ``p + 1``) must already move, and every logit before ``p`` must be
    bit-identical, because attention is causal.
    """

    input_ids, attention_mask = make_padded_batch([[3, 4, 5, 6, 7, 8]])
    start = 3
    vector = torch.randn(1, tiny_topology.resid_dim) * 5.0

    baseline = _forward(tiny_lm, input_ids, attention_mask)
    with torch.no_grad(), inject_at_layer(tiny_lm, 0, vector, alpha=1.0, start_index=start):
        steered = _forward(tiny_lm, input_ids, attention_mask)

    torch.testing.assert_close(steered[:, :start, :], baseline[:, :start, :], atol=0, rtol=0)
    assert not torch.allclose(steered[:, start, :], baseline[:, start, :])
    assert not torch.allclose(steered[:, -1, :], baseline[:, -1, :])


def test_completion_scope_covers_the_first_loss_position(tiny_lm, tiny_topology):
    """``start_index = prompt_len - 1`` must steer the first completion logit.

    The loss slices ``logits[:, prompt_len - 1 : -1]``, so index ``prompt_len - 1``
    is the first position the loss uses; if steering started at ``prompt_len`` the
    first supervised token would get no teacher signal.
    """

    prompt_len = 3
    input_ids, attention_mask = make_padded_batch([[3, 4, 5, 6, 7, 8]])
    vector = torch.randn(1, tiny_topology.resid_dim) * 5.0

    baseline = _forward(tiny_lm, input_ids, attention_mask)[:, prompt_len - 1 : -1, :]
    with torch.no_grad(), inject_at_layer(
        tiny_lm, 0, vector, alpha=1.0, start_index=prompt_len - 1
    ):
        steered = _forward(tiny_lm, input_ids, attention_mask)[:, prompt_len - 1 : -1, :]

    per_position_changed = [
        not torch.allclose(steered[:, index, :], baseline[:, index, :])
        for index in range(steered.shape[1])
    ]
    assert all(per_position_changed), per_position_changed


def test_alpha_scales_the_delta_linearly(tiny_lm, tiny_topology):
    input_ids, attention_mask = make_padded_batch([[3, 4, 5]])
    vector = torch.randn(1, tiny_topology.resid_dim)
    clean = _layer_output(tiny_lm, 2, input_ids, attention_mask)
    deltas = []
    for alpha in (1.0, 3.0):
        with torch.no_grad(), inject_at_layer(tiny_lm, 2, vector, alpha=alpha, start_index=0):
            deltas.append(_layer_output(tiny_lm, 2, input_ids, attention_mask) - clean)
    torch.testing.assert_close(deltas[1], 3.0 * deltas[0], atol=1e-5, rtol=1e-5)


def test_hook_is_removed_and_fires_once(tiny_lm, tiny_topology):
    input_ids, attention_mask = make_padded_batch([[3, 4, 5]])
    layer = get_decoder_layers(tiny_lm)[1]
    before = len(layer._forward_hooks)
    with torch.no_grad(), inject_at_layer(
        tiny_lm, 1, torch.zeros(1, tiny_topology.resid_dim)
    ) as hook:
        assert len(layer._forward_hooks) == before + 1
        _forward(tiny_lm, input_ids, attention_mask)
        assert hook.call_count == 1
    assert len(layer._forward_hooks) == before


def test_inject_at_layer_validates_the_layer_index(tiny_lm, tiny_topology):
    with pytest.raises(IndexError):
        with inject_at_layer(tiny_lm, 99, torch.zeros(1, tiny_topology.resid_dim)):
            pass


def test_steer_hook_validates_shapes():
    with pytest.raises(ValueError):
        ResidualSteerHook(torch.zeros(4))
    with pytest.raises(ValueError):
        ResidualSteerHook(torch.zeros(1, 4), start_index=-1)

    hook = ResidualSteerHook(torch.zeros(2, 4))
    with pytest.raises(ValueError):
        hook(None, None, torch.zeros(3, 5, 4))


def test_steer_hook_handles_tensor_and_tuple_outputs():
    vector = torch.ones(1, 3)
    hook = ResidualSteerHook(vector, alpha=2.0, start_index=1)
    hidden = torch.zeros(1, 2, 3)

    as_tensor = hook(None, None, hidden)
    torch.testing.assert_close(as_tensor, torch.tensor([[[0.0, 0.0, 0.0], [2.0, 2.0, 2.0]]]))

    as_tuple = hook(None, None, (hidden, "extra"))
    assert as_tuple[1] == "extra"
    torch.testing.assert_close(as_tuple[0], as_tensor)


def test_injection_dtype_follows_the_hidden_state(tiny_lm, tiny_topology):
    hook = ResidualSteerHook(torch.randn(1, tiny_topology.resid_dim), start_index=0)
    hidden = torch.zeros(1, 2, tiny_topology.resid_dim, dtype=torch.bfloat16)
    assert hook(None, None, hidden).dtype == torch.bfloat16
