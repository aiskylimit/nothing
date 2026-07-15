################################################################
# Simple Inference class for OpenRouter API
################################################################

import os
import time
import random
import concurrent.futures
from openai import OpenAI
from openai.types.chat import ChatCompletion
from vllm import SamplingParams, RequestOutput, CompletionOutput
from src.utils.utils import init_logger
logger = init_logger()

class OpenRouterInference:
    def __init__(self, model_name: str):
        self.model_name = model_name
        # Initial API key from environment variable.
        initial_api_key = os.getenv("OPENROUTER_API_KEY")
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=initial_api_key
        )

    def create_client(self, api_key: str):
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key
        )
        return client

    def run_batch(self, conversations: list, sampling_params: SamplingParams, meta=None):
        pass

    def sleep(self):
        pass
