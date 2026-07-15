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

        return str(content)

    def run_batch(
        self,
        conversations: list,
        sampling_params: SamplingParams,
        meta=None,
    ) -> list[RequestOutput]:
        pass

    def sleep(self):
        pass
