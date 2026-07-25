import os


ROOT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

IGNORE_INDEX = -100

MODEL_PATH = os.path.join(ROOT_PATH, "models")
ADAPTER_PATH = os.path.join(ROOT_PATH, "output", "adapter")
DATA_PATH = os.path.join(ROOT_PATH, "data")

INPUT_PROMPT = "Question: {}"
INSTRUCTION_PROMPT = "Database schema:\n{}"
INSTRUCTION_ONE_SHOT_PROMPT = "Database schema:\n{}\nUse the provided schema to write the SQL query."

LOG_FILE_NAME = "trainer.log"
VALUE_HEAD_FILE_NAME = "value_head.bin"
FINETUNING_ARGS_NAME = "finetuning_args.json"

LAYERNORM_NAMES = ["norm", "ln_f", "ln_attn", "ln_mlp"]

EXT2TYPE = {
    "csv": "csv",
    "json": "json",
    "jsonl": "json",
    "txt": "text",
}
