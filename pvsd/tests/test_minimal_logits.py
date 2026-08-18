"""The prefill passes must not materialise logits for every position.

A batch of 8 privileged prompts of 2k tokens over a 150k vocabulary is ~5 GB of
logits in bf16. PVSD's prefills need head activations (captured by hooks) and at
most the final next-token distribution, so they ask the model for one position.
"""

from __future__ import annotations

import pytest
import torch

from pvsd.common import privilege_vectors as pv
from pvsd.common.privilege_vectors import (
    extract_privilege_vector,
    forward_with_minimal_logits,
    position_ids_from_mask,
)

from conftest import make_padded_batch


@pytest.fixture(autouse=True)
def restore_kwarg_cache():
    """The detected kwarg name is cached globally; keep tests independent."""

    original = pv._LOGITS_TO_KEEP_KWARG
    yield
    pv._LOGITS_TO_KEEP_KWARG = original


def test_only_one_position_of_logits_is_computed(tiny_lm):
    input_ids, attention_mask = make_padded_batch([[3, 4, 5, 6, 7], [8, 9, 10, 11, 12]])
    outputs = forward_with_minimal_logits(tiny_lm, input_ids, attention_mask)
    assert outputs.logits.shape[1] == 1, "the pass should keep a single position of logits"


def test_the_kept_logits_match_a_full_forward(tiny_lm):
    input_ids, attention_mask = make_padded_batch([[3, 4, 5, 6], [7, 8, 9, 10]])
    full = tiny_lm(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids_from_mask(attention_mask),
        use_cache=False,
    ).logits
    minimal = forward_with_minimal_logits(tiny_lm, input_ids, attention_mask).logits
    torch.testing.assert_close(minimal[:, -1], full[:, -1], atol=1e-5, rtol=1e-5)


def test_the_detected_kwarg_is_cached(tiny_lm):
    input_ids, attention_mask = make_padded_batch([[3, 4, 5]])
    pv._LOGITS_TO_KEEP_KWARG = None
    forward_with_minimal_logits(tiny_lm, input_ids, attention_mask)
    assert pv._LOGITS_TO_KEEP_KWARG in {"logits_to_keep", "num_logits_to_keep"}


def test_falls_back_to_full_logits_when_unsupported(tiny_lm):
    """Older/custom models without the kwarg must still work."""

    class NoKeepKwarg(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner
            self.calls = 0

        def forward(self, *, logits_to_keep=None, num_logits_to_keep=None, **kwargs):
            for name, value in (
                ("logits_to_keep", logits_to_keep),
                ("num_logits_to_keep", num_logits_to_keep),
            ):
                if value is not None:
                    raise TypeError(f"forward() got an unexpected keyword argument '{name}'")
            self.calls += 1
            return self.inner(**kwargs)

    wrapped = NoKeepKwarg(tiny_lm)
    input_ids, attention_mask = make_padded_batch([[3, 4, 5, 6]])
    outputs = forward_with_minimal_logits(wrapped, input_ids, attention_mask)
    assert outputs.logits.shape[1] == input_ids.shape[1]  # full logits fallback
    assert wrapped.calls == 1
    assert pv._LOGITS_TO_KEEP_KWARG is None


def test_a_real_type_error_from_the_forward_is_not_swallowed(tiny_lm):
    class Broken(torch.nn.Module):
        def forward(self, **kwargs):
            raise TypeError("something inside the model went wrong")

    with pytest.raises(TypeError, match="something inside the model"):
        forward_with_minimal_logits(
            Broken(), torch.zeros(1, 3, dtype=torch.long), torch.ones(1, 3, dtype=torch.long)
        )


def test_extraction_is_unaffected_by_the_optimisation(tiny_lm, tiny_topology):
    """The vector comes from hooks, so restricting logits must not change it."""

    input_ids, attention_mask = make_padded_batch([[3, 4, 5, 6], [7, 8, 9, 10, 11]])
    heads = [(0, 1), (2, 2)]

    with_optimisation = extract_privilege_vector(
        tiny_lm, tiny_topology, input_ids, attention_mask, heads
    )

    pv._LOGITS_TO_KEEP_KWARG = None
    original = pv.forward_with_minimal_logits
    try:
        # Force the full-logits path and compare.
        pv.forward_with_minimal_logits = lambda model, ids, mask, position_ids=None: model(
            input_ids=ids,
            attention_mask=mask,
            position_ids=position_ids
            if position_ids is not None
            else position_ids_from_mask(mask),
            use_cache=False,
        )
        without_optimisation = extract_privilege_vector(
            tiny_lm, tiny_topology, input_ids, attention_mask, heads
        )
    finally:
        pv.forward_with_minimal_logits = original

    torch.testing.assert_close(with_optimisation, without_optimisation, atol=1e-6, rtol=1e-6)
