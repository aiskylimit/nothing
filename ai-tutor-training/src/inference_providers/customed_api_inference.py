################################################################
# Simple inference class for OpenAI-compatible custom APIs
################################################################

import concurrent.futures
import os
import random
import time
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import ChatCompletion
from vllm import CompletionOutput, RequestOutput, SamplingParams

from src.utils.utils import init_logger

load_dotenv()
logger = init_logger()

class CustomedAPIInference:
    """
    Inference adapter for OpenAI-compatible API providers with custom base URL.

    API key and URL are read from:
    - CUSTOMED_API_KEY
    - CUSTOMED_API_BASE_URL
    """

    def __init__(self, model_name: str):
        load_dotenv()
        self.model_name = model_name
        self.api_key = os.getenv("CUSTOMED_API_KEY")
        self.base_url = os.getenv("CUSTOMED_API_BASE_URL")

        if not self.api_key:
            raise ValueError("CUSTOMED_API_KEY is not set in the environment.")
        if not self.base_url:
            raise ValueError("CUSTOMED_API_BASE_URL is not set in the environment.")

    def create_client(self) -> OpenAI:
        return OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
        )

    @staticmethod
    def _message_content_to_text(content: Any) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict):
                    parts.append(str(item.get("text", "")))
                else:
                    parts.append(str(item))
            return "".join(parts)
        return str(content)

    def run_batch(
        self,
        conversations: list,
        sampling_params: SamplingParams,
        meta=None,
    ) -> list[RequestOutput]:
        def _execute_one(conversation):
            max_retries = 3
            backoff = 1.0
            n = getattr(sampling_params, "n", 1) or 1

            for attempt in range(1, max_retries + 1):
                try:
                    client = self.create_client()
                    completion_outputs: list[CompletionOutput] = []

                    for index in range(n):
                        completion: ChatCompletion = client.chat.completions.create(
                            model=self.model_name,
                            messages=conversation,
                            temperature=getattr(sampling_params, "temperature", None),
                            max_tokens=getattr(sampling_params, "max_tokens", None),
                            top_p=getattr(sampling_params, "top_p", None),
                            reasoning_effort="low"
                        )
                        text = self._message_content_to_text(
                            completion.choices[0].message.content
                        )
                        completion_outputs.append(
                            CompletionOutput(
                                index=index,
                                text=text,
                                token_ids=[],
                                cumulative_logprob=0.0,
                                logprobs=[],
                            )
                        )

                    return RequestOutput(
                        request_id="",
                        prompt="",
                        outputs=completion_outputs,
                        prompt_token_ids=[],
                        prompt_logprobs=[],
                        finished=True,
                    )
                except Exception as exc:
                    logger.warning(f"Attempt {attempt} failed: {exc}")
                    if attempt == max_retries:
                        logger.error("All custom API attempts failed.")
                        return RequestOutput(
                            request_id="",
                            prompt="",
                            outputs=[
                                CompletionOutput(
                                    index=index,
                                    text="",
                                    token_ids=[],
                                    cumulative_logprob=0.0,
                                    logprobs=[],
                                )
                                for index in range(n)
                            ],
                            prompt_token_ids=[],
                            prompt_logprobs=[],
                            finished=False,
                        )

                    time.sleep(backoff + random.uniform(0, 5))
                    backoff = min(backoff * 2, 30)

        with concurrent.futures.ThreadPoolExecutor(max_workers=80) as executor:
            futures = [
                executor.submit(_execute_one, conversation)
                for conversation in conversations
            ]
            return [future.result() for future in futures]

    def sleep(self):
        pass
