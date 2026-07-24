from __future__ import annotations

import json
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Iterable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from infer import _resolve_ckpt_dir, generate_response_batch, init_model


@dataclass(frozen=True)
class ModelSpec:
    name: str
    base: str
    backend: str = "hf"
    sft_ckpt: str | None = None
    sft_revision: str | None = None
    lora_adapters: tuple[str, ...] = ()
    lora_revision: str | None = None
    device: str = "cuda"
    vllm_base_url: str | None = None
    vllm_model: str | None = None
    vllm_api_key: str = "EMPTY"
    vllm_timeout: float = 120.0
    vllm_concurrency: int = 8
    vllm_disable_thinking: bool = True


@dataclass
class LoadedModel:
    name: str
    backend: str
    tokenizer: object | None = None
    model: object | None = None
    vllm_base_url: str | None = None
    vllm_model: str | None = None
    vllm_api_key: str = "EMPTY"
    vllm_timeout: float = 120.0
    vllm_concurrency: int = 8
    vllm_disable_thinking: bool = True

    def generate_many(
        self,
        batch_messages: list[list[dict[str, str]]],
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
    ) -> list[str]:
        if self.backend == "hf":
            if self.tokenizer is None or self.model is None:
                raise RuntimeError(f"HF model {self.name!r} was not loaded correctly.")
            return generate_response_batch(
                self.tokenizer,
                self.model,
                batch_messages,
                max_length=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )
        if self.backend == "vllm":
            return self._generate_vllm_many(
                batch_messages,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )
        raise ValueError(f"Unsupported model backend: {self.backend}")

    def _generate_vllm_many(
        self,
        batch_messages: list[list[dict[str, str]]],
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
    ) -> list[str]:
        concurrency = max(1, int(self.vllm_concurrency))
        if concurrency == 1 or len(batch_messages) <= 1:
            return [
                self._generate_vllm_one(
                    messages,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                )
                for messages in batch_messages
            ]

        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [
                executor.submit(
                    self._generate_vllm_one,
                    messages,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                )
                for messages in batch_messages
            ]
            return [future.result() for future in futures]

    def _generate_vllm_one(
        self,
        messages: list[dict[str, str]],
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
    ) -> str:
        if not self.vllm_base_url:
            raise RuntimeError(f"Missing vLLM base URL for model alias {self.name!r}.")
        model_name = self.vllm_model or self.name
        payload: dict[str, Any] = {
            "model": model_name,
            "messages": messages,
            "max_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }
        if top_k > 0:
            payload["top_k"] = top_k
        if self.vllm_disable_thinking:
            payload["chat_template_kwargs"] = {"enable_thinking": False}

        url = self.vllm_base_url.rstrip("/") + "/chat/completions"
        body = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.vllm_api_key}",
        }
        request = Request(url, data=body, headers=headers, method="POST")
        try:
            with urlopen(request, timeout=self.vllm_timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"vLLM request failed for {model_name!r}: HTTP {exc.code}: {detail}") from exc
        except URLError as exc:
            raise RuntimeError(f"Could not reach vLLM server at {url}: {exc}") from exc

        try:
            message = data["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Unexpected vLLM response for {model_name!r}: {data}") from exc
        return str(message.get("content") or message.get("reasoning") or "")


def split_adapters(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(part.strip() for part in value.split(",") if part.strip())


def load_model_from_spec(spec: ModelSpec) -> LoadedModel:
    if spec.backend == "vllm":
        base_url = spec.vllm_base_url or "http://localhost:8000/v1"
        served_model = spec.vllm_model or spec.name
        print(f"Using vLLM backend for {spec.name}: model={served_model} base_url={base_url}")
        return LoadedModel(
            name=spec.name,
            backend="vllm",
            vllm_base_url=base_url,
            vllm_model=served_model,
            vllm_api_key=spec.vllm_api_key,
            vllm_timeout=spec.vllm_timeout,
            vllm_concurrency=spec.vllm_concurrency,
            vllm_disable_thinking=spec.vllm_disable_thinking,
        )
    if spec.backend != "hf":
        raise ValueError(f"Unsupported model backend for {spec.name}: {spec.backend}")

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
    return LoadedModel(name=spec.name, backend="hf", tokenizer=tokenizer, model=model)


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

