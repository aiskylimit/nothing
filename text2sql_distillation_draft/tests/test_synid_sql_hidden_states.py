import pytest
import torch
import torch.nn as nn

from src.synid_sql import LastHiddenStateCapture, SelectedHiddenStateCapture, parse_layer_ids


class TinyCausalLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(8, 4)
        self.lm_head = nn.Linear(4, 8, bias=False)

    def get_output_embeddings(self):
        return self.lm_head

    def forward(self, input_ids):
        return self.lm_head(self.embedding(input_ids))


class TinyLayeredCausalLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(8, 4)
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([nn.Linear(4, 4), nn.Linear(4, 4)])
        self.lm_head = nn.Linear(4, 8, bias=False)

    def get_output_embeddings(self):
        return self.lm_head

    def forward(self, input_ids):
        hidden = self.embedding(input_ids)
        for layer in self.model.layers:
            hidden = layer(hidden)
        return self.lm_head(hidden)


def test_last_hidden_state_capture_preserves_gradients():
    model = TinyCausalLM()
    capture = LastHiddenStateCapture(model)

    logits = model(torch.tensor([[1, 2, 3]]))
    hidden_states = capture.pop()
    logits.sum().backward()

    assert hidden_states.shape == (1, 3, 4)
    assert hidden_states.requires_grad
    assert model.embedding.weight.grad is not None
    capture.close()


def test_last_hidden_state_capture_requires_a_forward():
    model = TinyCausalLM()
    capture = LastHiddenStateCapture(model)

    with pytest.raises(RuntimeError, match="No hidden states"):
        capture.pop()

    capture.close()


def test_parse_layer_ids_accepts_last_and_layer_indices():
    assert parse_layer_ids("0, 3, -1") == [0, 3, -1]


def test_selected_hidden_state_capture_preserves_order_and_last_hidden():
    model = TinyLayeredCausalLM()
    capture = SelectedHiddenStateCapture(model, [0, -1])

    logits = model(torch.tensor([[1, 2, 3]]))
    layer_hidden, last_hidden = capture.pop_all()
    logits.sum().backward()

    assert layer_hidden.shape == (1, 3, 4)
    assert last_hidden.shape == (1, 3, 4)
    assert layer_hidden.requires_grad
    assert last_hidden.requires_grad
    assert not torch.allclose(layer_hidden, last_hidden)
    assert model.embedding.weight.grad is not None
    capture.close()


def test_selected_hidden_state_capture_rejects_out_of_range_layer():
    model = TinyLayeredCausalLM()

    with pytest.raises(ValueError, match="out of range"):
        SelectedHiddenStateCapture(model, [2])
