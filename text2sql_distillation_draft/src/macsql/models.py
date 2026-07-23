from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from infer import _resolve_ckpt_dir, init_model


@dataclass(frozen=True)
class ModelSpec:
    name: str
    base: str
    sft_ckpt: str | None = None
    sft_revision: str | None = None
    lora_adapters: tuple[str, ...] = ()
    lora_revision: str | None = None
    device: str = "cuda"


@dataclass
class LoadedModel:
    name: str
    tokenizer: object
    model: object


def split_adapters(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(part.strip() for part in value.split(",") if part.strip())


def load_model_from_spec(spec: ModelSpec) -> LoadedModel:
    ckpt_path = spec.sft_ckpt
    ckpt_revision = spec.sft_revision
    tokenizer, model = init_model(
        spec.base,
        ckpt_path=ckpt_path,
        ckpt_revision=ckpt_revision,
        device=spec.device,
    )
    if spec.lora_adapters:
        from peft import PeftModel

        for adapter in spec.lora_adapters:
            resolved_source, resolved_revision = _resolve_ckpt_dir(adapter, spec.lora_revision)
            print(f"Loading extra LoRA adapter for {spec.name}: {resolved_source}")
            model = PeftModel.from_pretrained(model, resolved_source, revision=resolved_revision)
            model = model.merge_and_unload()
        model.eval()
    return LoadedModel(name=spec.name, tokenizer=tokenizer, model=model)


class ModelRegistry:
    def __init__(self, specs: Iterable[ModelSpec]):
        self._specs = {spec.name: spec for spec in specs}
        self._loaded: dict[str, LoadedModel] = {}

    def get(self, name: str) -> LoadedModel:
        if name not in self._specs:
            available = ", ".join(sorted(self._specs))
            raise KeyError(f"Unknown model alias {name!r}. Available aliases: {available}")
        if name not in self._loaded:
            self._loaded[name] = load_model_from_spec(self._specs[name])
        return self._loaded[name]

