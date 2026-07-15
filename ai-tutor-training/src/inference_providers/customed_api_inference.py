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
    def __init__(self, model_name: str):
        pass

    def create_client(self) -> OpenAI:
        pass

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
