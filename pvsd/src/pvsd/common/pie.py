"""Stage 1 of PVSD: Privilege Indirect Effect (PIE) head localisation.

Causal mediation adapted from function-vector analysis. For every candidate head
``(l, j)`` we ask: if the model is fed a *corrupted* privileged context (the same
question and view format, but another problem's reference) and we patch only that
head's activation with the clean-context mean, how much of the clean teacher
distribution is recovered?

    CIE(a_lj | r~_i) = D_KL( P(. | x_i, r_i) || P(. | x_i, r~_i)[a_lj <- a_bar_lj] )^-1
    PIE(a_lj)        = mean_i CIE(a_lj | r~_i)

The head set ``A = TopK(PIE)`` is what the privilege vector is read from. PIE is a
slowly-changing property of the model, so it is recomputed only every N steps.

Pure ``torch``: importable and testable on CPU.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from pvsd.common.privilege_vectors import (
    ModelTopology,
    capture_head_activations,
    forward_with_minimal_logits,
    get_attn_out_proj,
    normalize_head_set,
)


PIE_SCORE_MODES = ("inverse_kl", "neg_kl")


def parse_candidate_layers(spec, num_layers: int | None = None):
    """Parse a candidate-layer spec: ``None``/``'all'``, ``'8:24'`` or ``'9,12,15'``.

    Returns ``None`` for "every layer" so callers can keep that meaning lazily until
    the model's depth is known.
    """

    if spec is None:
        return None
    if not isinstance(spec, str):
        layers = tuple(sorted({int(item) for item in spec}))
        if not layers:
            raise ValueError("candidate layer spec must name at least one layer.")
        return layers

    text = spec.strip().lower()
    if not text or text == "all":
        return None if num_layers is None else tuple(range(num_layers))
    if ":" in text:
        start_text, stop_text = text.split(":", 1)
        start = int(start_text) if start_text else 0
        if stop_text:
            stop = int(stop_text)
        elif num_layers is not None:
            stop = int(num_layers)
        else:
            raise ValueError("a layer range needs an explicit end, e.g. '8:24'.")
        if stop <= start:
            raise ValueError(f"layer range '{spec}' is empty.")
        return tuple(range(start, stop))
    layers = tuple(sorted({int(item) for item in text.split(",") if item.strip()}))
    if not layers:
        raise ValueError("candidate layer spec must name at least one layer.")
    return layers


@dataclass(frozen=True)
class PIEResult:
    """Outcome of one calibration pass."""

    scores: torch.Tensor  # [num_layers, num_heads], -inf for heads not scanned
    kl: torch.Tensor  # [num_layers, num_heads] mean patched-vs-clean KL, +inf if unscanned
    top_heads: tuple[tuple[int, int], ...]
    num_forward_passes: int
    num_examples: int

    def as_log_dict(self, prefix: str = "pvsd/pie") -> dict[str, float]:
        finite = torch.isfinite(self.kl)
        logs = {
            f"{prefix}/num_forward_passes": float(self.num_forward_passes),
            f"{prefix}/num_examples": float(self.num_examples),
            f"{prefix}/num_heads_scanned": float(int(finite.sum())),
        }
        if torch.any(finite):
            logs[f"{prefix}/kl_min"] = float(self.kl[finite].min())
            logs[f"{prefix}/kl_median"] = float(self.kl[finite].median())
        selected = torch.zeros_like(self.kl, dtype=torch.bool)
        for layer_idx, head_idx in self.top_heads:
            selected[layer_idx, head_idx] = True
        if torch.any(selected & finite):
            logs[f"{prefix}/kl_selected_mean"] = float(self.kl[selected & finite].mean())
            logs[f"{prefix}/selected_layer_mean"] = float(
                sum(layer for layer, _ in self.top_heads) / len(self.top_heads)
            )
        return logs


class _HeadPatch:
    """Forward pre-hook that overwrites chosen ``(row, head)`` activations."""

    def __init__(
        self,
        topology: ModelTopology,
        position: int,
        assignments: list[tuple[int, int]],
        replacement: torch.Tensor,
    ):
        self.topology = topology
        self.position = int(position)
        self.assignments = assignments
        self.replacement = replacement  # [attn_inner_dim]
        self.call_count = 0

    def __call__(self, module, args, kwargs=None):
        del module
        hidden = args[0] if isinstance(args, tuple) else args
        patched = hidden.clone()
        for row, head_idx in self.assignments:
            head_slice = self.topology.head_slice(head_idx)
            patched[row, self.position, head_slice] = self.replacement[head_slice].to(
                dtype=patched.dtype, device=patched.device
            )
        self.call_count += 1
        return (patched,)


@contextmanager
def patch_heads(
    model: torch.nn.Module,
    topology: ModelTopology,
    position: int,
    assignments_by_layer: dict[int, list[tuple[int, int]]],
    mean_activations: dict[int, torch.Tensor],
):
    """Patch ``{layer: [(row, head), ...]}`` with clean-context means for one pass."""

    handles = []
    hooks = {}
    try:
        for layer_idx, assignments in assignments_by_layer.items():
            hook = _HeadPatch(topology, position, assignments, mean_activations[layer_idx])
            hooks[layer_idx] = hook
            handles.append(get_attn_out_proj(model, layer_idx).register_forward_pre_hook(hook))
        yield hooks
    finally:
        for handle in handles:
            handle.remove()


def _trim_to_real_tokens(input_ids: torch.Tensor, attention_mask: torch.Tensor, row: int):
    """Drop the right padding of one row: ``(ids[1, len], mask[1, len], len - 1)``.

    Every row of a PIE forward is the same example replicated, so trimming keeps the
    read-out at the final position. That lets the pass compute a single position's
    logits instead of a ``[rows, seq, vocab]`` tensor, and removes pad tokens from the
    computation entirely.
    """

    length = int(attention_mask[row].to(dtype=torch.long).sum())
    if length == 0:
        raise ValueError(f"row {row} of the PIE batch is all padding.")
    return input_ids[row : row + 1, :length], attention_mask[row : row + 1, :length], length - 1


def _readout_log_probs(model: torch.nn.Module, input_ids: torch.Tensor, attention_mask: torch.Tensor):
    """Next-token log-probs at the final position: ``[batch, vocab]``."""

    outputs = forward_with_minimal_logits(model, input_ids, attention_mask)
    return F.log_softmax(outputs.logits[:, -1].float(), dim=-1)


def estimate_pie_forward_passes(
    num_examples: int,
    num_candidate_heads: int,
    head_chunk_size: int,
) -> int:
    """Forward passes one calibration will cost (clean passes included)."""

    chunk = max(1, int(head_chunk_size))
    per_example = (num_candidate_heads + chunk - 1) // chunk
    return int(num_examples) * (per_example + 1)


@torch.no_grad()
def compute_privilege_indirect_effect(
    model: torch.nn.Module,
    topology: ModelTopology,
    clean_input_ids: torch.Tensor,
    clean_attention_mask: torch.Tensor,
    corrupt_input_ids: torch.Tensor,
    corrupt_attention_mask: torch.Tensor,
    *,
    top_k_heads: int,
    candidate_layers=None,
    head_chunk_size: int = 8,
    max_examples: int | None = None,
    score_mode: str = "inverse_kl",
    eps: float = 1e-6,
) -> PIEResult:
    """Localise the heads that transport privileged content into the output.

    ``clean_*`` are the privileged prompts ``(x_i, r_i)``; ``corrupt_*`` are the
    matched corrupted prompts ``(x_i, r~_i)``. Both are right-padded batches.
    """

    if score_mode not in PIE_SCORE_MODES:
        raise ValueError(f"score_mode must be one of: {', '.join(PIE_SCORE_MODES)}")
    if top_k_heads <= 0:
        raise ValueError("top_k_heads must be positive.")
    if clean_input_ids.shape[0] != corrupt_input_ids.shape[0]:
        raise ValueError("clean and corrupt batches must have the same number of examples.")

    if candidate_layers is None:
        candidate_layers = range(topology.num_layers)
    candidate_layers = sorted({int(layer) for layer in candidate_layers})
    for layer_idx in candidate_layers:
        if not 0 <= layer_idx < topology.num_layers:
            raise IndexError(f"candidate layer {layer_idx} out of range.")
    candidate_heads = [
        (layer_idx, head_idx)
        for layer_idx in candidate_layers
        for head_idx in range(topology.num_heads)
    ]
    if top_k_heads > len(candidate_heads):
        raise ValueError(
            f"top_k_heads ({top_k_heads}) exceeds the number of candidate heads ({len(candidate_heads)})."
        )

    num_examples = clean_input_ids.shape[0]
    if max_examples is not None:
        num_examples = min(num_examples, max(1, int(max_examples)))

    # --- clean pass: teacher distribution + clean-context mean head activations ---
    clean_log_probs = []
    clean_activation_sums: dict[int, torch.Tensor] = {}
    for index in range(num_examples):
        ids, mask, read_out = _trim_to_real_tokens(clean_input_ids, clean_attention_mask, index)
        positions = torch.tensor([read_out], dtype=torch.long, device=ids.device)
        with capture_head_activations(model, candidate_layers, positions) as store:
            clean_log_probs.append(_readout_log_probs(model, ids, mask)[0])
        for layer_idx, activation in store.items():
            row = activation[0]
            if layer_idx in clean_activation_sums:
                clean_activation_sums[layer_idx] = clean_activation_sums[layer_idx] + row
            else:
                clean_activation_sums[layer_idx] = row.clone()

    mean_activations = {
        layer_idx: total / float(num_examples) for layer_idx, total in clean_activation_sums.items()
    }

    # --- corrupted passes with one head patched per batch row ---
    chunk_size = max(1, int(head_chunk_size))
    kl_sums = torch.zeros(topology.num_layers, topology.num_heads, dtype=torch.float64)
    score_sums = torch.zeros(topology.num_layers, topology.num_heads, dtype=torch.float64)
    num_forward_passes = num_examples  # clean passes already done

    for index in range(num_examples):
        target_log_probs = clean_log_probs[index]
        target_probs = target_log_probs.exp()
        base_ids, base_mask, position = _trim_to_real_tokens(
            corrupt_input_ids, corrupt_attention_mask, index
        )

        for start in range(0, len(candidate_heads), chunk_size):
            chunk = candidate_heads[start : start + chunk_size]
            rows = len(chunk)
            ids = base_ids.expand(rows, -1).contiguous()
            mask = base_mask.expand(rows, -1).contiguous()

            assignments_by_layer: dict[int, list[tuple[int, int]]] = {}
            for row, (layer_idx, head_idx) in enumerate(chunk):
                assignments_by_layer.setdefault(layer_idx, []).append((row, head_idx))

            with patch_heads(model, topology, position, assignments_by_layer, mean_activations):
                patched_log_probs = _readout_log_probs(model, ids, mask)
            num_forward_passes += 1

            # KL( clean || patched ) per row
            kl = (target_probs.unsqueeze(0) * (target_log_probs.unsqueeze(0) - patched_log_probs)).sum(
                dim=-1
            )
            kl = kl.clamp_min(0.0).double().cpu()
            for row, (layer_idx, head_idx) in enumerate(chunk):
                kl_value = kl[row]
                kl_sums[layer_idx, head_idx] += kl_value
                if score_mode == "inverse_kl":
                    score_sums[layer_idx, head_idx] += 1.0 / (kl_value + eps)
                else:
                    score_sums[layer_idx, head_idx] += -kl_value

    scanned = torch.zeros(topology.num_layers, topology.num_heads, dtype=torch.bool)
    for layer_idx, head_idx in candidate_heads:
        scanned[layer_idx, head_idx] = True

    scores = torch.full_like(score_sums, float("-inf"))
    kl_mean = torch.full_like(kl_sums, float("inf"))
    scores[scanned] = score_sums[scanned] / float(num_examples)
    kl_mean[scanned] = kl_sums[scanned] / float(num_examples)

    flat_scores = scores.flatten()
    top_indices = torch.topk(flat_scores, k=top_k_heads, largest=True).indices
    top_heads = normalize_head_set(
        [
            (int(index // topology.num_heads), int(index % topology.num_heads))
            for index in top_indices
        ],
        topology,
    )

    return PIEResult(
        scores=scores.float(),
        kl=kl_mean.float(),
        top_heads=top_heads,
        num_forward_passes=num_forward_passes,
        num_examples=num_examples,
    )
