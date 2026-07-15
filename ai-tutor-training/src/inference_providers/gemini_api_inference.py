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
        pass

    def run_batch(self, conversations: list, sampling_params: SamplingParams, meta=None):
        pass

    def sleep(self):
        pass
