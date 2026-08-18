"""Function-vector style privilege signal: extraction, purification, fusion, injection.

This module implements the mechanical core of PVSD (Stage 2 of the method):

1. **Live raw vector extraction.** For a privileged prompt ``(x, r)`` we read the
   activations of a selected set of attention heads at the *last input token* and
   map each head back into the residual stream through its slice of ``W_O``::

       v_r = sum_{(l, j) in A} W_O^{(l)}[:, j] @ a_{l,j}(last(x, r))

   This is the Function Vector construction of Todd et al., where the head
   activation ``a_{l,j}`` is the *input* of ``self_attn.o_proj`` split by head.

2. **Contrastive purification.** The raw vector mixes transferable reasoning
   signal with a view-template term. We estimate the template *per example* with
   corrupted contexts ``(x, r~_k)`` that keep the question, the view format and
   the read-out position but swap in another problem's reference::

       v_transfer = v_r - mean_k v_{r~_k}

   A running/stream average is deliberately **not** used: its limit is
   ``v_template + E[v_reason]``, so it would remove the reasoning component that
   is shared across examples together with the template.

3. **Fusion.** The purified per-view vectors are combined into a single vector
   *before* any forward pass, so downstream layers process the fused signal
   through their full nonlinear computation.

4. **Injection.** ``h_l* <- h_l* + alpha * v*`` on the student's residual stream.

Only ``torch`` is required here so the module stays importable (and testable) on
CPU without ``trl``/``vllm``/``deepspeed``.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass

import torch


# ---------------------------------------------------------------------------
# model topology
# ---------------------------------------------------------------------------


def _unwrap_module(model: torch.nn.Module) -> torch.nn.Module:
    while hasattr(model, "module"):
        model = model.module
    return model


def get_base_causal_lm(model: torch.nn.Module) -> torch.nn.Module:
    """Unwrap DDP/DeepSpeed/PEFT wrappers down to the HF causal LM."""

    base = _unwrap_module(model)
    get_base_model = getattr(base, "get_base_model", None)
    if callable(get_base_model):
        base = _unwrap_module(get_base_model())
    return base


def get_decoder_layers(model: torch.nn.Module) -> torch.nn.ModuleList:
    """Return the decoder layer list, unwrapping training and PEFT wrappers."""

    base = get_base_causal_lm(model)
    inner = getattr(base, "model", None)
    layers = getattr(inner, "layers", None)
    if layers is None:
        # transformers>=5 nests one more level for some architectures.
        layers = getattr(getattr(inner, "model", None), "layers", None)
    if layers is None:
        raise AttributeError(
            "Could not locate decoder layers at model.model.layers. "
            f"Unsupported model type for PVSD: {type(base).__name__}"
        )
    return layers


def get_attn_out_proj(model: torch.nn.Module, layer_idx: int) -> torch.nn.Linear:
    """Return the attention output projection (``o_proj``) of one decoder layer."""

    layer = get_decoder_layers(model)[layer_idx]
    attn = getattr(layer, "self_attn", None) or getattr(layer, "attention", None)
    if attn is None:
        raise AttributeError(f"Decoder layer {layer_idx} has no self_attn/attention module.")
    for name in ("o_proj", "out_proj", "dense", "c_proj"):
        proj = getattr(attn, name, None)
        if proj is not None:
            return proj
    raise AttributeError(f"Could not locate the attention output projection on layer {layer_idx}.")


def out_proj_head_weight(module: torch.nn.Module, head_slice: slice) -> torch.Tensor:
    """Effective ``W_O`` columns for one head, LoRA delta included.

    The attention output projection is linear in the concatenated per-head vector,
    so head ``j``'s contribution to the residual stream is
    ``W_eff[:, slice_j] @ a_j``. With PEFT the effective projection is
    ``W + scaling * B @ A``, and only the requested columns of the delta are
    materialised (``B @ A[:, slice]``), which keeps this cheap enough to call per
    head and per step.
    """

    base = getattr(module, "base_layer", module)
    weight = base.weight[:, head_slice].float()

    lora_a = getattr(module, "lora_A", None)
    lora_b = getattr(module, "lora_B", None)
    if lora_a is None or lora_b is None:
        return weight
    if getattr(module, "disable_adapters", False):
        return weight

    active = getattr(module, "active_adapters", None) or []
    merged = set(getattr(module, "merged_adapters", None) or [])
    use_dora = getattr(module, "use_dora", {}) or {}
    for name in active:
        if name in merged:
            # Already folded into base.weight; counting it again would double it.
            continue
        if name not in lora_a or name not in lora_b:
            continue
        if use_dora.get(name, False):
            raise NotImplementedError(
                "PVSD head read-out does not support DoRA adapters on the attention "
                "output projection; use plain LoRA or exclude o_proj from the adapters."
            )
        down = lora_a[name].weight.float()  # [r, in_features]
        up = lora_b[name].weight.float()  # [out_features, r]
        scaling = float(module.scaling[name]) if hasattr(module, "scaling") else 1.0
        weight = weight + scaling * (up @ down[:, head_slice])
    return weight


@dataclass(frozen=True)
class ModelTopology:
    """Shapes PVSD needs, derived from the model rather than from hard-coded names."""

    num_layers: int
    num_heads: int
    head_dim: int
    resid_dim: int

    @property
    def attn_inner_dim(self) -> int:
        return self.num_heads * self.head_dim

    def head_slice(self, head_idx: int) -> slice:
        if not 0 <= head_idx < self.num_heads:
            raise IndexError(f"head index {head_idx} out of range for {self.num_heads} heads")
        return slice(head_idx * self.head_dim, (head_idx + 1) * self.head_dim)


def describe_model(model: torch.nn.Module) -> ModelTopology:
    """Read layer/head/head_dim/resid_dim off the model.

    ``head_dim`` is derived from ``o_proj.in_features`` and **not** from
    ``hidden_size // num_heads``: models such as Qwen3 set ``head_dim`` explicitly
    (Qwen3-4B: hidden 2560, 32 heads, head_dim 128), so the division would be wrong.
    """

    base = get_base_causal_lm(model)
    config = base.config
    layers = get_decoder_layers(model)
    out_proj = get_attn_out_proj(model, 0)
    out_proj_base = getattr(out_proj, "base_layer", out_proj)

    num_heads = int(getattr(config, "num_attention_heads"))
    resid_dim = int(out_proj_base.out_features)
    inner_dim = int(out_proj_base.in_features)
    if inner_dim % num_heads != 0:
        raise ValueError(
            f"o_proj.in_features ({inner_dim}) is not divisible by num_attention_heads ({num_heads})."
        )
    head_dim = inner_dim // num_heads

    config_head_dim = getattr(config, "head_dim", None)
    if config_head_dim is not None and int(config_head_dim) != head_dim:
        raise ValueError(
            f"Derived head_dim ({head_dim}) disagrees with config.head_dim ({int(config_head_dim)})."
        )

    return ModelTopology(
        num_layers=len(layers),
        num_heads=num_heads,
        head_dim=head_dim,
        resid_dim=resid_dim,
    )


def normalize_head_set(heads, topology: ModelTopology) -> tuple[tuple[int, int], ...]:
    """Validate and de-duplicate a ``[(layer, head), ...]`` head set, keeping order."""

    normalized: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for entry in heads:
        layer_idx, head_idx = int(entry[0]), int(entry[1])
        if not 0 <= layer_idx < topology.num_layers:
            raise IndexError(f"layer index {layer_idx} out of range for {topology.num_layers} layers")
        topology.head_slice(head_idx)  # range check
        key = (layer_idx, head_idx)
        if key not in seen:
            seen.add(key)
            normalized.append(key)
    if not normalized:
        raise ValueError("head set must contain at least one (layer, head) pair.")
    return tuple(normalized)


def all_heads_in_layers(topology: ModelTopology, layers) -> tuple[tuple[int, int], ...]:
    """Every head in the given layers - used as the no-PIE ablation head set."""

    return normalize_head_set(
        [(layer_idx, head_idx) for layer_idx in layers for head_idx in range(topology.num_heads)],
        topology,
    )


# ---------------------------------------------------------------------------
# position handling
# ---------------------------------------------------------------------------


def position_ids_from_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    """Positions that ignore padding, so real tokens always start at 0.

    Prompts in this codebase are right-padded and completions are concatenated
    after the padding, which leaves pad gaps *inside* the sequence. Without
    explicit ``position_ids`` the model would fall back to ``arange`` and shift
    every completion token of the shorter rows, so the training forward pass
    would not match the positions the rollout was sampled at.
    """

    mask = attention_mask.to(dtype=torch.long)
    return (mask.cumsum(dim=-1) - 1).clamp_min(0)


_LOGITS_TO_KEEP_KWARG: str | None = "logits_to_keep"


def forward_with_minimal_logits(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor | None = None,
):
    """Forward pass that computes logits for the last position only.

    PVSD's prefill passes need head activations (captured by hooks) and at most the
    final next-token distribution - never the logits of every position. Without this,
    a batch of 8 privileged prompts of 2k tokens would materialise a
    ``[8, 2000, 151936]`` logits tensor (~5 GB in bf16) on every extraction and on
    every PIE patch, next to the training model and vLLM's reserved memory.

    ``logits_to_keep`` was called ``num_logits_to_keep`` in older transformers, and
    the working spelling is cached after the first call.
    """

    global _LOGITS_TO_KEEP_KWARG

    if position_ids is None:
        position_ids = position_ids_from_mask(attention_mask)
    kwargs = dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_cache=False,
    )

    candidates = (
        [_LOGITS_TO_KEEP_KWARG] if _LOGITS_TO_KEEP_KWARG else []
    ) + ["logits_to_keep", "num_logits_to_keep"]
    for name in dict.fromkeys(candidates):
        try:
            outputs = model(**kwargs, **{name: 1})
        except TypeError as exc:
            # Only swallow "this model has no such argument"; a TypeError raised from
            # inside the forward pass must not be mistaken for an unsupported kwarg.
            if name not in str(exc):
                raise
            continue
        _LOGITS_TO_KEEP_KWARG = name
        return outputs

    # Older/custom models: fall back to full logits rather than failing.
    _LOGITS_TO_KEEP_KWARG = None
    return model(**kwargs)


def last_real_token_index(attention_mask: torch.Tensor) -> torch.Tensor:
    """Index of the last attended token in each row (works for right padding)."""

    mask = attention_mask.to(dtype=torch.long)
    lengths = mask.sum(dim=-1)
    if torch.any(lengths == 0):
        raise ValueError("attention_mask has an all-padding row; cannot pick a read-out position.")
    return lengths - 1


# ---------------------------------------------------------------------------
# head activation read-out
# ---------------------------------------------------------------------------


class _HeadActivationCapture:
    """Forward pre-hook that stores ``o_proj`` inputs at one position per row."""

    def __init__(self, layer_idx: int, positions: torch.Tensor, store: dict[int, torch.Tensor]):
        self.layer_idx = layer_idx
        self.positions = positions
        self.store = store

    def __call__(self, module, args, kwargs=None):
        del module
        hidden = args[0] if isinstance(args, tuple) else args
        rows = torch.arange(hidden.shape[0], device=hidden.device)
        positions = self.positions.to(device=hidden.device, dtype=torch.long)
        # [batch, attn_inner_dim] - only the read-out position is kept, so this
        # stays tiny even for long prompts.
        self.store[self.layer_idx] = hidden[rows, positions].detach().float()
        return None


@contextmanager
def capture_head_activations(model: torch.nn.Module, layers, positions: torch.Tensor):
    """Capture ``o_proj`` inputs of ``layers`` at ``positions`` for one forward pass."""

    store: dict[int, torch.Tensor] = {}
    handles = []
    try:
        for layer_idx in sorted(set(int(layer) for layer in layers)):
            hook = _HeadActivationCapture(layer_idx, positions, store)
            handles.append(get_attn_out_proj(model, layer_idx).register_forward_pre_hook(hook))
        yield store
    finally:
        for handle in handles:
            handle.remove()


def head_contribution_to_residual(
    model: torch.nn.Module,
    topology: ModelTopology,
    layer_idx: int,
    head_idx: int,
    activation: torch.Tensor,
) -> torch.Tensor:
    """Map one head's activation into the residual stream via its ``W_O`` slice.

    ``activation`` is ``[..., head_dim]``. The projection bias (when a model has
    one) is intentionally excluded: it is a constant shared by every head, it is
    not part of the head's contribution, and adding it once per selected head
    would scale it by ``|A|``.

    The matmul runs in float32 even for bf16 models: the slices are small
    (``resid_dim x head_dim``) so the cost is negligible, while summing many heads
    in bf16 would lose precision in a vector that is later differenced.
    """

    out_proj = get_attn_out_proj(model, layer_idx)
    weight = out_proj_head_weight(out_proj, topology.head_slice(head_idx))
    return activation.float() @ weight.t()


def privilege_vector_from_activations(
    model: torch.nn.Module,
    topology: ModelTopology,
    activations: dict[int, torch.Tensor],
    heads,
) -> torch.Tensor:
    """Sum the residual-stream contributions of ``heads``: ``[batch, resid_dim]``."""

    heads = normalize_head_set(heads, topology)
    missing = sorted({layer for layer, _ in heads} - set(activations))
    if missing:
        raise KeyError(f"missing captured activations for layers {missing}")

    reference = next(iter(activations.values()))
    vector = torch.zeros(
        reference.shape[0],
        topology.resid_dim,
        dtype=torch.float32,
        device=reference.device,
    )
    for layer_idx, head_idx in heads:
        activation = activations[layer_idx][:, topology.head_slice(head_idx)]
        vector += head_contribution_to_residual(
            model, topology, layer_idx, head_idx, activation
        ).float()
    return vector


@torch.no_grad()
def extract_privilege_vector(
    model: torch.nn.Module,
    topology: ModelTopology,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    heads,
    positions: torch.Tensor | None = None,
    micro_batch_size: int | None = None,
) -> torch.Tensor:
    """Read the privilege vector from prefill passes over privileged prompts.

    Returns ``[batch, resid_dim]`` in float32. One prefill per micro-batch; no
    rollout and no gradient are involved.
    """

    heads = normalize_head_set(heads, topology)
    layers = sorted({layer for layer, _ in heads})
    if positions is None:
        positions = last_real_token_index(attention_mask)

    total = input_ids.shape[0]
    step = total if not micro_batch_size or micro_batch_size <= 0 else int(micro_batch_size)
    chunks = []
    for start in range(0, total, step):
        stop = min(start + step, total)
        ids = input_ids[start:stop]
        mask = attention_mask[start:stop]
        with capture_head_activations(model, layers, positions[start:stop]) as store:
            # The logits are unused here: the vector comes from the captured
            # o_proj inputs, so only one position's logits are computed.
            forward_with_minimal_logits(model, ids, mask)
        chunks.append(privilege_vector_from_activations(model, topology, store, heads))
    return torch.cat(chunks, dim=0)


# ---------------------------------------------------------------------------
# purification and fusion
# ---------------------------------------------------------------------------


PURIFICATION_MODES = ("contrast", "none", "template_only")


def apply_purification(
    raw_vector: torch.Tensor,
    corrupt_vectors: torch.Tensor,
    mode: str = "contrast",
) -> torch.Tensor:
    """Select what gets steered with, for the method and its two ablations.

    * ``contrast`` - ``v_r - mean_k v_{r~_k}``: the method.
    * ``none`` - the raw vector, i.e. steering without template subtraction.
    * ``template_only`` - the discarded component alone, which should *hurt*.

    The corrupted contexts are extracted in every mode, so the ablations cost the
    same as the method and differ only in the signal that is injected.
    """

    if mode not in PURIFICATION_MODES:
        raise ValueError(f"purification mode must be one of: {', '.join(PURIFICATION_MODES)}")
    if mode == "none":
        return raw_vector
    if mode == "template_only":
        return corrupt_vectors.mean(dim=1)
    return purify_privilege_vector(raw_vector, corrupt_vectors)


def head_set_jaccard(left, right) -> float:
    """Jaccard overlap of two head sets (the cross-view head-overlap diagnostic)."""

    left_set = {(int(layer), int(head)) for layer, head in left}
    right_set = {(int(layer), int(head)) for layer, head in right}
    union = left_set | right_set
    if not union:
        return 0.0
    return len(left_set & right_set) / len(union)


def purify_privilege_vector(
    raw_vector: torch.Tensor,
    corrupt_vectors: torch.Tensor,
) -> torch.Tensor:
    """``v_transfer = v_r - mean_k v_{r~_k}`` (per example, not over the stream).

    ``raw_vector``: ``[batch, resid]``; ``corrupt_vectors``: ``[batch, k, resid]``.
    """

    if corrupt_vectors.dim() != 3:
        raise ValueError("corrupt_vectors must have shape [batch, num_corrupt, resid_dim].")
    if corrupt_vectors.shape[0] != raw_vector.shape[0]:
        raise ValueError("raw_vector and corrupt_vectors disagree on batch size.")
    if corrupt_vectors.shape[-1] != raw_vector.shape[-1]:
        raise ValueError("raw_vector and corrupt_vectors disagree on resid_dim.")
    if corrupt_vectors.shape[1] == 0:
        raise ValueError(
            "contrastive purification needs at least one corrupted context; "
            "use per_device_train_batch_size >= 2 so an in-batch contrast exists."
        )
    return raw_vector - corrupt_vectors.mean(dim=1)


def fuse_view_vectors(
    view_vectors: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fuse ``[batch, num_views, resid]`` into ``[batch, resid]`` (uniform default)."""

    if view_vectors.dim() != 3:
        raise ValueError("view_vectors must have shape [batch, num_views, resid_dim].")
    if weights is None:
        return view_vectors.mean(dim=1)
    if weights.shape != view_vectors.shape[:2]:
        raise ValueError("weights must have shape [batch, num_views].")
    normalizer = weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
    return (view_vectors * (weights / normalizer).unsqueeze(-1)).sum(dim=1)


def cosine_similarity_rowwise(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Row-wise cosine similarity, used for the template/content diagnostics."""

    return torch.nn.functional.cosine_similarity(left.float(), right.float(), dim=-1)


@dataclass(frozen=True)
class ViewVectors:
    """Raw / corrupted / purified vectors of a single privileged view."""

    view: str
    raw: torch.Tensor  # [batch, resid]
    corrupt: torch.Tensor  # [batch, num_corrupt, resid]
    transfer: torch.Tensor  # [batch, resid] - the signal that is injected
    heads: tuple[tuple[int, int], ...] = ()
    purification: str = "contrast"

    @property
    def corrupt_mean(self) -> torch.Tensor:
        return self.corrupt.mean(dim=1)


def resolve_heads_by_view(views, heads) -> dict[str, tuple[tuple[int, int], ...]]:
    """Accept either one shared head set or a per-view mapping ``A^(m)``."""

    if hasattr(heads, "keys"):
        missing = [view for view in views if view not in heads]
        if missing:
            raise KeyError(f"no head set for view(s) {missing}")
        return {view: tuple(tuple(head) for head in heads[view]) for view in views}
    shared = tuple(tuple(head) for head in heads)
    return {view: shared for view in views}


@torch.no_grad()
def build_privilege_vectors_from_batch(
    model: torch.nn.Module,
    topology: ModelTopology,
    views,
    batch,
    heads,
    micro_batch_size: int = 8,
    purification: str = "contrast",
) -> tuple[torch.Tensor, list[ViewVectors]]:
    """Full Stage-2 pipeline for one batch: extract, purify, fuse.

    ``batch`` must provide, for every view, the keys written by
    ``SelfDistillationDataCollator`` when ``pvsd_views`` is set:
    ``pvsd_{view}_input_ids`` / ``_attention_mask`` and
    ``pvsd_{view}_corrupt_input_ids`` / ``_corrupt_attention_mask``.

    ``heads`` is either one shared head set or a ``{view: head set}`` mapping, since
    PIE localises a separate ``A^(m)`` per view.

    Returns the fused vector ``[batch, resid]`` and the per-view intermediates so
    callers can log diagnostics without recomputing anything.
    """

    if purification not in PURIFICATION_MODES:
        raise ValueError(f"purification mode must be one of: {', '.join(PURIFICATION_MODES)}")
    heads_by_view = resolve_heads_by_view(views, heads)

    per_view: list[ViewVectors] = []
    for view in views:
        heads = heads_by_view[view]
        real_ids = batch[f"pvsd_{view}_input_ids"]
        real_mask = batch[f"pvsd_{view}_attention_mask"]
        raw = extract_privilege_vector(
            model, topology, real_ids, real_mask, heads, micro_batch_size=micro_batch_size
        )

        corrupt_ids = batch[f"pvsd_{view}_corrupt_input_ids"]
        corrupt_mask = batch[f"pvsd_{view}_corrupt_attention_mask"]
        if corrupt_ids.dim() != 3:
            raise ValueError(
                f"pvsd_{view}_corrupt_input_ids must be [batch, num_corrupt, seq]; "
                f"got {tuple(corrupt_ids.shape)}."
            )
        rows, num_corrupt, seq_len = corrupt_ids.shape
        corrupt = extract_privilege_vector(
            model,
            topology,
            corrupt_ids.reshape(rows * num_corrupt, seq_len),
            corrupt_mask.reshape(rows * num_corrupt, seq_len),
            heads,
            micro_batch_size=micro_batch_size,
        ).view(rows, num_corrupt, -1)

        per_view.append(
            ViewVectors(
                view=view,
                raw=raw,
                corrupt=corrupt,
                transfer=apply_purification(raw, corrupt, purification),
                heads=heads,
                purification=purification,
            )
        )

    fused = fuse_view_vectors(torch.stack([item.transfer for item in per_view], dim=1))
    return fused, per_view


# ---------------------------------------------------------------------------
# injection
# ---------------------------------------------------------------------------


class ResidualSteerHook:
    """Adds ``alpha * vector`` to a decoder layer's residual output.

    ``start_index`` is the first sequence position that gets steered. The trainer
    passes ``student_prompt_length - 1``: that is the position whose hidden state
    produces the logits for the *first* completion token, so every position the
    loss is computed on is steered, and no earlier position is touched.
    """

    def __init__(self, vector: torch.Tensor, alpha: float = 1.0, start_index: int = 0):
        if vector.dim() != 2:
            raise ValueError("steering vector must have shape [batch, resid_dim].")
        if start_index < 0:
            raise ValueError("start_index must be non-negative.")
        self.vector = vector
        self.alpha = float(alpha)
        self.start_index = int(start_index)
        self.call_count = 0

    def _delta(self, hidden: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = hidden.shape
        if self.vector.shape[0] != batch:
            raise ValueError(
                f"steering vector batch ({self.vector.shape[0]}) != hidden batch ({batch})."
            )
        vector = (self.alpha * self.vector).to(dtype=hidden.dtype, device=hidden.device)
        positions = torch.arange(seq_len, device=hidden.device)
        gate = (positions >= self.start_index).to(dtype=hidden.dtype).view(1, seq_len, 1)
        return gate * vector.unsqueeze(1)

    def __call__(self, module, inputs, output):
        del module, inputs
        self.call_count += 1
        if isinstance(output, tuple):
            hidden = output[0]
            return (hidden + self._delta(hidden),) + output[1:]
        return output + self._delta(output)


@contextmanager
def inject_at_layer(
    model: torch.nn.Module,
    layer_idx: int,
    vector: torch.Tensor,
    alpha: float = 1.0,
    start_index: int = 0,
):
    """Steer layer ``layer_idx`` with ``vector`` for the duration of the context."""

    layers = get_decoder_layers(model)
    if not 0 <= layer_idx < len(layers):
        raise IndexError(f"pvsd_layer {layer_idx} out of range for {len(layers)} decoder layers.")
    hook = ResidualSteerHook(vector, alpha=alpha, start_index=start_index)
    handle = layers[layer_idx].register_forward_hook(hook)
    try:
        yield hook
    finally:
        handle.remove()


def default_injection_layer(num_layers: int, fraction: str = "quarter") -> int:
    """``l*`` candidates from the method description: ``L/4``, ``L/3``, ``L/2``."""

    divisors = {"quarter": 4, "third": 3, "half": 2}
    if fraction not in divisors:
        raise ValueError(f"fraction must be one of: {', '.join(sorted(divisors))}")
    return num_layers // divisors[fraction]
