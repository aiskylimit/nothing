"""Chunked sampled-token log-probabilities."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from pvsd.common.token_log_probs import sampled_token_log_probs


def reference(logits, token_ids, temperature=1.0):
    log_probs = F.log_softmax(logits.float() / temperature, dim=-1)
    return log_probs.gather(dim=-1, index=token_ids.unsqueeze(-1)).squeeze(-1)


def test_matches_log_softmax_and_gather():
    torch.manual_seed(0)
    logits = torch.randn(3, 17, 29)
    token_ids = torch.randint(0, 29, (3, 17))
    torch.testing.assert_close(
        sampled_token_log_probs(logits, token_ids, chunk_size=5),
        reference(logits, token_ids),
        atol=1e-6,
        rtol=1e-5,
    )


@pytest.mark.parametrize("chunk_size", [1, 2, 7, 64, 1000])
def test_chunking_does_not_change_the_result(chunk_size):
    torch.manual_seed(1)
    logits = torch.randn(2, 13, 8)
    token_ids = torch.randint(0, 8, (2, 13))
    torch.testing.assert_close(
        sampled_token_log_probs(logits, token_ids, chunk_size=chunk_size),
        reference(logits, token_ids),
        atol=1e-6,
        rtol=1e-5,
    )


def test_temperature_is_applied():
    torch.manual_seed(2)
    logits = torch.randn(1, 4, 6)
    token_ids = torch.randint(0, 6, (1, 4))
    torch.testing.assert_close(
        sampled_token_log_probs(logits, token_ids, temperature=0.6),
        reference(logits, token_ids, temperature=0.6),
        atol=1e-6,
        rtol=1e-5,
    )


def test_bf16_logits_are_normalised_in_float32():
    torch.manual_seed(3)
    logits = torch.randn(1, 6, 40)
    result = sampled_token_log_probs(logits.bfloat16(), torch.randint(0, 40, (1, 6)))
    assert result.dtype == torch.float32
    assert torch.isfinite(result).all()


def test_log_probs_are_negative_and_sum_to_one_over_the_vocab():
    torch.manual_seed(4)
    logits = torch.randn(1, 1, 5)
    all_tokens = torch.arange(5).view(1, 5)
    per_token = sampled_token_log_probs(logits.expand(1, 5, 5).contiguous(), all_tokens)
    assert bool((per_token <= 0).all())
    torch.testing.assert_close(per_token.exp().sum(), torch.tensor(1.0), atol=1e-5, rtol=1e-5)


def test_validates_shapes():
    with pytest.raises(ValueError):
        sampled_token_log_probs(torch.randn(2, 3), torch.zeros(2, 3, dtype=torch.long))
    with pytest.raises(ValueError):
        sampled_token_log_probs(torch.randn(2, 3, 4), torch.zeros(2, 5, dtype=torch.long))
    with pytest.raises(ValueError):
        sampled_token_log_probs(
            torch.randn(2, 3, 4), torch.zeros(2, 3, dtype=torch.long), temperature=0.0
        )
