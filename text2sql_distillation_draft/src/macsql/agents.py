from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from infer import generate_response_batch

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
        user_prompt = self.user_template.format(**kwargs)
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return generate_response_batch(
            self.loaded_model.tokenizer,
            self.loaded_model.model,
            [messages],
            max_length=self.gen.max_new_tokens,
            temperature=self.gen.temperature,
            top_p=self.gen.top_p,
            top_k=self.gen.top_k,
        )[0]


class SelectorAgent(ChatAgent):
    def select(self, **kwargs: Any) -> tuple[dict[str, Any], str]:
        raw = self.generate(**kwargs)
        return extract_selector_schema(raw), raw


class DecomposerAgent(ChatAgent):
    def decompose(self, **kwargs: Any) -> tuple[str, list[dict[str, Any]], str]:
        raw = self.generate(**kwargs)
        parsed = parse_json_object(raw)
        steps = []
        if parsed and isinstance(parsed.get("sub_questions"), list):
            steps = [item for item in parsed["sub_questions"] if isinstance(item, dict)]
        return extract_sql(raw), steps, raw


class RefinerAgent(ChatAgent):
    def refine(self, **kwargs: Any) -> tuple[str, str]:
        raw = self.generate(**kwargs)
        return extract_sql(raw), raw

