"""Memory-bounded log-probability read-out for sampled tokens."""

from __future__ import annotations

import torch


def sampled_token_log_probs(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    temperature: float = 1.0,
    chunk_size: int = 64,
) -> torch.Tensor:
    """``log p(token_ids)`` without materialising a ``[batch, seq, vocab]`` copy.

    ``logits`` is ``[batch, seq, vocab]`` and ``token_ids`` is ``[batch, seq]``. The
    softmax normaliser is computed in float32 over chunks of the time axis, so a
    4k-token rollout over a 150k vocabulary stays within a few hundred MB instead
    of allocating several GB.
    """

    if logits.dim() != 3:
        raise ValueError("logits must have shape [batch, seq, vocab].")
    if logits.shape[:2] != token_ids.shape:
        raise ValueError("logits and token_ids disagree on [batch, seq].")
    if temperature <= 0:
        raise ValueError("temperature must be positive.")

    outputs = []
    seq_len = logits.shape[1]
    step = max(1, int(chunk_size))
    for start in range(0, seq_len, step):
        stop = min(start + step, seq_len)
        block = logits[:, start:stop, :].float() / temperature
        gathered = block.gather(dim=-1, index=token_ids[:, start:stop].unsqueeze(-1)).squeeze(-1)
        outputs.append(gathered - torch.logsumexp(block, dim=-1))
        del block
    return torch.cat(outputs, dim=1)
