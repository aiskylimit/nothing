from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .parsing import extract_selector_schema, extract_sql, parse_json_object


@dataclass
class GenerationConfig:
    max_new_tokens: int = 512
    temperature: float = 0.2
    top_p: float = 0.95
    top_k: int = 0


class ChatAgent:
    def __init__(self, *, name: str, loaded_model: Any, system_prompt: str, user_template: str, gen: GenerationConfig):
        self.name = name
        self.loaded_model = loaded_model
        self.system_prompt = system_prompt
        self.user_template = user_template
        self.gen = gen

    def generate(self, **kwargs: Any) -> str:
        return self.generate_many([kwargs])[0]

    def generate_many(self, batch_kwargs: list[dict[str, Any]]) -> list[str]:
        batch_messages = []
        for kwargs in batch_kwargs:
            user_prompt = self.user_template.format(**kwargs)
            batch_messages.append(
                [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
            )
        return self.loaded_model.generate_many(
            batch_messages,
            max_new_tokens=self.gen.max_new_tokens,
            temperature=self.gen.temperature,
            top_p=self.gen.top_p,
            top_k=self.gen.top_k,
        )


class SelectorAgent(ChatAgent):
    def select(self, **kwargs: Any) -> tuple[dict[str, Any], str]:
        raw = self.generate(**kwargs)
        return extract_selector_schema(raw), raw

    def select_many(self, batch_kwargs: list[dict[str, Any]]) -> list[tuple[dict[str, Any], str]]:
        return [(extract_selector_schema(raw), raw) for raw in self.generate_many(batch_kwargs)]


class DecomposerAgent(ChatAgent):
    def decompose(self, **kwargs: Any) -> tuple[str, list[dict[str, Any]], str]:
        raw = self.generate(**kwargs)
        return self._parse_response(raw)

    def decompose_many(self, batch_kwargs: list[dict[str, Any]]) -> list[tuple[str, list[dict[str, Any]], str]]:
        return [self._parse_response(raw) for raw in self.generate_many(batch_kwargs)]

    @staticmethod
    def _parse_response(raw: str) -> tuple[str, list[dict[str, Any]], str]:
        parsed = parse_json_object(raw)
        steps = []
        if parsed and isinstance(parsed.get("sub_questions"), list):
            steps = [item for item in parsed["sub_questions"] if isinstance(item, dict)]
        return extract_sql(raw), steps, raw


class RefinerAgent(ChatAgent):
    def refine(self, **kwargs: Any) -> tuple[str, str]:
        raw = self.generate(**kwargs)
        return extract_sql(raw), raw

