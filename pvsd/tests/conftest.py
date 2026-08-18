"""Shared fixtures for the PVSD test suite.

Everything here runs on CPU in float32 on a randomly initialised 3-layer model of
~10k parameters, so the whole suite is a few seconds and needs no network access
and no GPU. Nothing in this suite trains a model.
"""

from __future__ import annotations

import pytest
import torch

transformers = pytest.importorskip("transformers")

from transformers import AutoModelForCausalLM, Qwen3Config  # noqa: E402


# Deliberately chosen so that hidden_size // num_attention_heads (4) differs from
# head_dim (8): that is the Qwen3 layout, and it catches any code that derives the
# head dimension from the hidden size.
TINY_CONFIG = dict(
    vocab_size=97,
    hidden_size=16,
    intermediate_size=32,
    num_hidden_layers=3,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=8,
    max_position_embeddings=128,
    tie_word_embeddings=False,
    rms_norm_eps=1e-6,
)


@pytest.fixture
def tiny_lm():
    """A small real Qwen3 causal LM (random weights, eager attention, float32)."""

    config = Qwen3Config(**TINY_CONFIG)
    config._attn_implementation = "eager"
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(config)
    model.eval()
    return model


@pytest.fixture
def tiny_topology(tiny_lm):
    from pvsd.common.privilege_vectors import describe_model

    return describe_model(tiny_lm)


@pytest.fixture
def rng():
    return torch.Generator().manual_seed(1234)


def make_padded_batch(sequences, pad_id=0):
    """Right-pad a list of id lists into ``(input_ids, attention_mask)``."""

    max_len = max(len(sequence) for sequence in sequences)
    input_ids = torch.full((len(sequences), max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(sequences), max_len), dtype=torch.long)
    for row, sequence in enumerate(sequences):
        input_ids[row, : len(sequence)] = torch.tensor(sequence, dtype=torch.long)
        attention_mask[row, : len(sequence)] = 1
    return input_ids, attention_mask


class FakeTokenizer:
    """Word-level stand-in for a HF tokenizer (offline, deterministic).

    Implements only what ``SelfDistillationDataCollator`` uses: ``__call__`` with
    padding/truncation, ``apply_chat_template``, ``padding_side`` and ``decode``.
    """

    def __init__(self):
        self.padding_side = "right"
        self.pad_token = "<pad>"
        self.pad_token_id = 0
        self.chat_template = "{# fake template #}"
        self._vocab = {self.pad_token: 0}
        self._inverse = {0: self.pad_token}

    # -- chat template -------------------------------------------------------
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, **kwargs):
        del kwargs
        body = "\n".join(message["content"] for message in messages)
        rendered = f"<|im_start|> {body} <|im_end|>"
        if add_generation_prompt:
            rendered += " <|assistant|>"
        if tokenize:
            return self._encode(rendered)
        return rendered

    # -- tokenisation --------------------------------------------------------
    def _token_id(self, token):
        if token not in self._vocab:
            index = len(self._vocab)
            self._vocab[token] = index
            self._inverse[index] = token
        return self._vocab[token]

    def _encode(self, text):
        return [self._token_id(token) for token in str(text).split()]

    def decode(self, ids, skip_special_tokens=False):
        del skip_special_tokens
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return " ".join(self._inverse.get(int(index), "<unk>") for index in ids)

    def batch_decode(self, batch, skip_special_tokens=False):
        return [self.decode(row, skip_special_tokens=skip_special_tokens) for row in batch]

    def __call__(
        self,
        texts,
        padding=False,
        truncation=False,
        max_length=None,
        return_tensors=None,
    ):
        if isinstance(texts, str):
            texts = [texts]
        encoded = [self._encode(text) for text in texts]
        if truncation and max_length is not None:
            encoded = [ids[:max_length] for ids in encoded]

        if padding == "max_length":
            width = max_length if max_length is not None else max(len(ids) for ids in encoded)
        elif padding in {True, "longest"}:
            width = max(len(ids) for ids in encoded)
        else:
            width = None

        attention = [[1] * len(ids) for ids in encoded]
        if width is not None:
            for row, ids in enumerate(encoded):
                missing = width - len(ids)
                if missing > 0:
                    if self.padding_side == "right":
                        encoded[row] = ids + [self.pad_token_id] * missing
                        attention[row] = attention[row] + [0] * missing
                    else:
                        encoded[row] = [self.pad_token_id] * missing + ids
                        attention[row] = [0] * missing + attention[row]

        if return_tensors == "pt":
            return {
                "input_ids": torch.tensor(encoded, dtype=torch.long),
                "attention_mask": torch.tensor(attention, dtype=torch.long),
            }
        return {"input_ids": encoded, "attention_mask": attention}


@pytest.fixture
def fake_tokenizer():
    return FakeTokenizer()


@pytest.fixture
def math_features():
    """Four distinct (problem, solution) rows with easily greppable content."""

    return [
        {
            "problem": f"problem_{index} compute value",
            "solution": (
                f"step_{index}_one step_{index}_two therefore the answer is "
                f"\\boxed{{answer_{index}}}"
            ),
        }
        for index in range(4)
    ]
