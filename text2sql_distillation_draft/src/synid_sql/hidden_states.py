from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class LastHiddenStateCapture:
    """Capture the final normalized hidden state entering a causal LM head."""

    def __init__(self, model: nn.Module):
        output_embeddings = model.get_output_embeddings()
        if output_embeddings is None:
            raise ValueError("Cannot capture hidden states: model has no output embedding layer.")
        self.hidden_states: Optional[torch.Tensor] = None
        self._handle = output_embeddings.register_forward_pre_hook(self._capture)

    def _capture(self, _module: nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
        if not inputs or not isinstance(inputs[0], torch.Tensor):
            raise RuntimeError("Causal LM head did not receive hidden states as its first input.")
        self.hidden_states = inputs[0]

    def pop(self) -> torch.Tensor:
        if self.hidden_states is None:
            raise RuntimeError("No hidden states were captured during the latest model forward.")
        hidden_states = self.hidden_states
        self.hidden_states = None
        return hidden_states

    def clear(self) -> None:
        self.hidden_states = None

    def close(self) -> None:
        self._handle.remove()
        self.hidden_states = None


def parse_layer_ids(value: str | list[int] | tuple[int, ...]) -> list[int]:
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",") if part.strip()]
        if not parts:
            raise ValueError("Layer list must not be empty.")
        layer_ids = [int(part) for part in parts]
    else:
        layer_ids = [int(layer_id) for layer_id in value]
        if not layer_ids:
            raise ValueError("Layer list must not be empty.")

    invalid = [layer_id for layer_id in layer_ids if layer_id < -1]
    if invalid:
        raise ValueError(f"Layer ids must be -1 or non-negative, got {invalid}.")
    return layer_ids


def resolve_decoder_layers(model: nn.Module) -> nn.ModuleList | list[nn.Module]:
    candidates = [
        lambda module: module.model.layers,
        lambda module: module.base_model.model.model.layers,
        lambda module: module.base_model.model.layers,
        lambda module: module.transformer.h,
    ]

    seen = set()
    current = model
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        for getter in candidates:
            try:
                layers = getter(current)
            except AttributeError:
                continue
            if layers is not None:
                return layers
        current = getattr(current, "module", None)

    raise ValueError("Cannot resolve decoder layers for selected SynID layer capture.")


class SelectedHiddenStateCapture:
    """Capture selected decoder hidden states, with -1 as LM-head input."""

    def __init__(self, model: nn.Module, layer_ids: str | list[int] | tuple[int, ...]):
        self.layer_ids = parse_layer_ids(layer_ids)
        self.hidden_states: list[Optional[torch.Tensor]] = [None] * len(self.layer_ids)
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

        decoder_layers = None
        if any(layer_id >= 0 for layer_id in self.layer_ids):
            decoder_layers = resolve_decoder_layers(model)

        for slot, layer_id in enumerate(self.layer_ids):
            if layer_id == -1:
                output_embeddings = model.get_output_embeddings()
                if output_embeddings is None:
                    raise ValueError("Cannot capture hidden states: model has no output embedding layer.")
                self._handles.append(output_embeddings.register_forward_pre_hook(self._make_lm_head_hook(slot)))
            else:
                assert decoder_layers is not None
                if layer_id >= len(decoder_layers):
                    raise ValueError(
                        f"Layer id {layer_id} is out of range for model with {len(decoder_layers)} decoder layers."
                    )
                self._handles.append(decoder_layers[layer_id].register_forward_hook(self._make_layer_hook(slot)))

    def _make_lm_head_hook(self, slot: int):
        def hook(_module: nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
            if not inputs or not isinstance(inputs[0], torch.Tensor):
                raise RuntimeError("Causal LM head did not receive hidden states as its first input.")
            self.hidden_states[slot] = inputs[0]

        return hook

    def _make_layer_hook(self, slot: int):
        def hook(_module: nn.Module, _inputs: tuple[torch.Tensor, ...], output) -> None:
            hidden_states = output[0] if isinstance(output, tuple) else output
            if not isinstance(hidden_states, torch.Tensor):
                raise RuntimeError("Decoder layer did not return hidden states as a tensor.")
            self.hidden_states[slot] = hidden_states

        return hook

    def pop_all(self) -> list[torch.Tensor]:
        missing = [self.layer_ids[idx] for idx, hidden in enumerate(self.hidden_states) if hidden is None]
        if missing:
            raise RuntimeError(f"No hidden states were captured for layers {missing} during the latest model forward.")
        hidden_states = [hidden for hidden in self.hidden_states if hidden is not None]
        self.clear()
        return hidden_states

    def pop(self) -> torch.Tensor:
        hidden_states = self.pop_all()
        if len(hidden_states) != 1:
            raise RuntimeError("pop() is only valid when exactly one hidden state was captured; use pop_all().")
        return hidden_states[0]

    def clear(self) -> None:
        self.hidden_states = [None] * len(self.layer_ids)

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []
        self.clear()
