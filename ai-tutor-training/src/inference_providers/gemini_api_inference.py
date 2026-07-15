################################################################
# Simple Inference class for Google Gemini API
################################################################

import os
import time
import random
import concurrent.futures

from dotenv import load_dotenv
import google.generativeai as genai
from vllm import SamplingParams, RequestOutput, CompletionOutput
from src.utils.utils import init_logger
logger = init_logger()

class GeminiInference:
    def __init__(self, model_name: str):
        load_dotenv()
        self.model_name = model_name
        primary_key = os.getenv("GEMINI_API_KEY")
        if not primary_key:
            raise ValueError("GEMINI_API_KEY is not set in the environment")
        genai.configure(api_key=primary_key)

    def run_batch(self, conversations: list, sampling_params: SamplingParams, meta=None, max_retries=10000000):
        pass

    def sleep(self):
        pass
