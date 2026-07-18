import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json


def read_json_file(filepath):
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"File {filepath} does not exist.")
        return None
    except json.JSONDecodeError:
        print(f"File {filepath} is not a valid JSON.")
        return None


def load_prompt(filepath: str) -> str:
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


def build_messages(
    question: str, schema: str, prompts_dir: str = "prompts/generator"
) -> list:
    system_prompt = load_prompt(f"{prompts_dir}/system_prompt.txt")
    user_prompt_template = load_prompt(f"{prompts_dir}/user_prompt.txt")

    # inject variables vào user prompt
    user_prompt = user_prompt_template.format(question=question, schema=schema)

    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": user_prompt,
        },
    ]

    return messages
