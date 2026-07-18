import random
import torch
import os
import json
import numpy as np
from functools import lru_cache
from torch.utils.data import Dataset
from .distributed_indexed import DistributedMMapIndexedDataset
from huggingface_hub import snapshot_download

from torch.distributed import get_rank, get_world_size
from utils import print_rank

HF_DATA_PATH_ALIASES = {
    "./processed_data/benchmarks/Cypherbench/qwen": "hf://fisherman611/text_to_cypher_distillation/benchmarks/Cypherbench/qwen",
    "./processed_data/benchmarks/Cypherbench/qwen/": "hf://fisherman611/text_to_cypher_distillation/benchmarks/Cypherbench/qwen",
    "processed_data/benchmarks/Cypherbench/qwen": "hf://fisherman611/text_to_cypher_distillation/benchmarks/Cypherbench/qwen",
    "processed_data/benchmarks/Cypherbench/qwen/": "hf://fisherman611/text_to_cypher_distillation/benchmarks/Cypherbench/qwen",
}


def _ensure_trailing_sep(path):
    return path if path.endswith(os.sep) else path + os.sep


def _parse_hf_path(path):
    normalized = path[len("hf://"):].strip("/")
    parts = normalized.split("/")
    if len(parts) < 2:
        raise ValueError(
            f"Invalid Hugging Face path '{path}'. Expected format: "
            "hf://<owner>/<repo>/<optional/subdir>"
        )
    repo_id = "/".join(parts[:2])
    subdir = "/".join(parts[2:])
    return repo_id, subdir


@lru_cache(maxsize=None)
def resolve_data_path(path):
    if path is None:
        return path

    normalized_path = path.replace("\\", "/").rstrip("/")
    if normalized_path in HF_DATA_PATH_ALIASES and not os.path.exists(path):
        hf_path = HF_DATA_PATH_ALIASES[normalized_path]
        print_rank(f"Local dataset path '{path}' not found. Falling back to '{hf_path}'.")
        path = hf_path

    if not path.startswith("hf://"):
        return _ensure_trailing_sep(path)

    repo_id, subdir = _parse_hf_path(path)
    allow_patterns = None
    if subdir:
        allow_patterns = [f"{subdir}/*", f"{subdir}/**"]

    token = os.getenv("HF_READ_TOKEN") or os.getenv("HF_TOKEN")
    last_error = None
    for repo_type in (None, "dataset"):
        try:
            snapshot_dir = snapshot_download(
                repo_id=repo_id,
                repo_type=repo_type,
                allow_patterns=allow_patterns,
                token=token,
            )
            resolved_path = os.path.join(snapshot_dir, subdir) if subdir else snapshot_dir
            if not os.path.isdir(resolved_path):
                raise FileNotFoundError(
                    f"Resolved Hugging Face path does not exist locally: {resolved_path}"
                )
            print_rank(f"Using dataset from Hugging Face cache: {resolved_path}")
            return _ensure_trailing_sep(resolved_path)
        except Exception as exc:
            last_error = exc

    raise RuntimeError(
        f"Failed to download dataset path '{path}' from Hugging Face Hub."
    ) from last_error


def find_prompt_response_separator(input_ids):
    # Preferred separators for new/uint32/int32 datasets.
    for sep in (-1, 4294967295):
        sep_pos = np.where(input_ids == sep)[0]
        if sep_pos.size > 0:
            return int(sep_pos[0])

    # Legacy separator used by older uint16 datasets.
    sep_pos = np.where(input_ids == 65535)[0]
    if sep_pos.size > 0:
        return int(sep_pos[0])

    return None


def extract_event_span_offsets(full_text, response_str):
    try:
        response_json = json.loads(response_str)
    except Exception:
        return []

    values_to_find = []
    for event in response_json.get("events", []):
        values_to_find.append(event[0])
        values_to_find.append(event[1])

        if len(event) > 3:
            for arg in event[2]:
                values_to_find.append(arg[0])
                values_to_find.append(arg[1])
            values_to_find.append(event[3])
        else:
            values_to_find.append(event[2])

    result_tuples = []
    search_start_idx = 0
    for val in values_to_find:
        search_str = f"{val}"
        char_start = full_text.find(search_str, search_start_idx)
        if char_start != -1:
            char_end = char_start + len(search_str)
            result_tuples.append((char_start, char_end))
            search_start_idx = char_end + 1

    return result_tuples


class LMTrainDataset(Dataset):
    def __init__(self, args, tokenizer, path, split, num, ratio, rng_sample: random.Random):
        self.args = args
        self.tokenizer = tokenizer
        self.split = split
        self.pad_id = self.tokenizer.eos_token_id
        self.ratio = ratio
        self.max_length = args.max_length
        self.max_prompt_length = args.max_prompt_length
        self.rng_sample = rng_sample
        self.has_span_metadata = False
        path = resolve_data_path(path)
        self.lm_ctx = DistributedMMapIndexedDataset(path, f"{split}", get_rank(), get_world_size())
        self.t_lm_ctx = None

        if (
            getattr(args, "type", None) == "synid"
            and os.path.exists(os.path.join(path, f"teacher_train_0.bin"))
            and split == "train"
        ):
            self.t_lm_ctx = DistributedMMapIndexedDataset(path, f"teacher_train", get_rank(), get_world_size())

        if os.path.exists(os.path.join(path, f"{split}.jsonl")):
            with open(os.path.join(path, f"{split}.jsonl")) as f:
                self.raw = [json.loads(line) for line in f.readlines()]
                self.answers = [x["response"] if isinstance(x["response"], list) else [x["response"]] for x in self.raw]
                self.full_texts = []
                for item in self.raw:
                    response = item["response"][0] if isinstance(item["response"], list) else item["response"]
                    self.full_texts.append(item["prompt"] + response)
                self.offset_mapping = [
                    tokenizer(
                        text,
                        return_offsets_mapping=True,
                        truncation=True,
                        max_length=self.max_length,
                        padding="max_length",
                        add_special_tokens=False,
                        return_tensors="pt",
                    )["offset_mapping"]
                    for text in self.full_texts
                ]
                self.get_span_offsets()
                self.has_span_metadata = True
        
        print_rank(len(self.lm_ctx))
        if num == -1:
            self.num = len(self.lm_ctx)
        else:
            self.num = num

        print_rank(f"Num LM instances: {len(self.lm_ctx)}")

    def __len__(self):
        return self.num

    def get_span_offsets(self):
        self.span_offsets = []
        for item, full_text in zip(self.raw, self.full_texts):
            response_str = item["response"][0] if isinstance(item["response"], list) else item["response"]
            self.span_offsets.append(extract_event_span_offsets(full_text, response_str))
   
    def __getitem__(self, index):
        return self._get_lm(index)
    
    def _get_lm(self, index):
        data = self.lm_ctx[index]
        input_ids = data.astype(int)

        t_input_ids = None
        if self.t_lm_ctx is not None:
            t_data = self.t_lm_ctx[index]
            t_input_ids = t_data.astype(int)

        sample = {
            "input_ids": input_ids,
            "t_input_ids": t_input_ids,
        }
        if self.has_span_metadata:
            sample["span_offsets"] = self.span_offsets[index]
            sample["offset_mapping"] = self.offset_mapping[index]
        return sample

    def _process_lm(self, i, samp, model_data, no_model_data, gen_data, max_length=None):
        if max_length is None:
            max_length = self.max_length

        input_ids = samp["input_ids"]
        source_len = 1
        
        prompt = None
        sep_idx = find_prompt_response_separator(input_ids)
        if sep_idx is not None:
            source_len = sep_idx
            prompt = input_ids[:source_len]
            input_ids = np.concatenate([input_ids[:source_len], input_ids[source_len+1:]], axis=0)
        
        input_ids = input_ids[:max_length]
        input_len = len(input_ids)
        model_data["input_ids"][i][:input_len-1] = torch.tensor(input_ids[:-1], dtype=torch.long)
        model_data["attention_mask"][i][:input_len-1] = 1.0
        if self.args.model_type in ["gpt2"]:
            model_data["position_ids"][i][:input_len-1] = torch.arange(0, input_len-1, dtype=torch.long)
        no_model_data["label"][i][:input_len-1] = torch.tensor(input_ids[1:], dtype=torch.long)
        no_model_data["label"][i][:source_len-1] = -100
        if "loss_mask" in no_model_data:
            no_model_data["loss_mask"][i][:input_len-1] = 1.0
            no_model_data["loss_mask"][i][:source_len-1] = 0
        
        if prompt is not None and gen_data is not None:
            gen_data["input_ids"][i][-len(prompt):] = torch.tensor(prompt, dtype=torch.long)
            gen_data["attention_mask"][i][-len(prompt):] = 1.0

    def move_to_device(self, model_data, no_model_data, gen_data, device):
        for k in model_data:
            model_data[k] = model_data[k].to(device)

        for k in no_model_data:
            if isinstance(no_model_data[k], torch.Tensor):
                no_model_data[k] = no_model_data[k].to(device)

        if gen_data is not None:
            for k in gen_data:
                gen_data[k] = gen_data[k].to(device)

        return model_data, no_model_data, gen_data

    def collate(self, samples):
        bs = len(samples)

        max_length = self.max_length
        
        model_data = {
            "input_ids": torch.ones(bs, max_length, dtype=torch.long) * self.pad_id,
            "attention_mask": torch.zeros(bs, max_length),
        }
        
        if self.args.model_type in ["gpt2"]:
            model_data["position_ids"] = torch.zeros(bs, max_length, dtype=torch.long)
            
        no_model_data = {
            "label": torch.ones(bs, max_length, dtype=torch.long) * -100,
            "loss_mask": torch.zeros(bs, max_length),
        }
        if "span_offsets" in samples[0]:
            no_model_data["span_offsets"] = [sample["span_offsets"] for sample in samples]
            no_model_data["offset_mapping"] = torch.concat([sample["offset_mapping"] for sample in samples])
        
        gen_data = {
            "input_ids": torch.ones(bs, self.max_prompt_length, dtype=torch.long) * self.pad_id,
            "attention_mask": torch.zeros(bs, self.max_prompt_length, dtype=torch.long),
        }

        for i, samp in enumerate(samples):
            self._process_lm(i, samp, model_data, no_model_data, gen_data)

        t_model_data, t_no_model_data = None, None
        if samples[0]["t_input_ids"] is not None:
            t_model_data = {
                "input_ids": torch.ones(bs, self.args.t_max_length, dtype=torch.long) * self.pad_id,
                "attention_mask": torch.zeros(bs, self.args.t_max_length),
            }
            
            if self.args.model_type in ["gpt2"]:
                t_model_data["position_ids"] = torch.zeros(bs, self.args.t_max_length, dtype=torch.long)
                
            t_no_model_data = {
                "label": torch.ones(bs, self.args.t_max_length, dtype=torch.long) * -100,
            }

            for i, samp in enumerate(samples):
                self._process_lm(
                    i,
                    {"input_ids": samp["t_input_ids"]},
                    t_model_data,
                    t_no_model_data,
                    None,
                    max_length=self.args.t_max_length,
                )
        
        return model_data, no_model_data, gen_data, t_model_data, t_no_model_data


class LMEvalDataset(Dataset):
    def __init__(self, args, tokenizer, path, split, rng_sample: random.Random):
        self.args = args
        self.tokenizer = tokenizer
        self.split = split
        self.pad_id = self.tokenizer.eos_token_id
        self.max_length = args.max_length
        self.max_prompt_length = args.max_prompt_length
        self.rng_sample = rng_sample
        path = resolve_data_path(path)
        self.lm_ctx = DistributedMMapIndexedDataset(path, f"{split}", 0, 1)

        if os.path.exists(os.path.join(path, f"{split}.jsonl")):
            with open(os.path.join(path, f"{split}.jsonl")) as f:
                self.raw = [json.loads(line) for line in f.readlines()]
                self.answers = [x["response"] if isinstance(x["response"], list) else [x["response"]] for x in self.raw]
        
        self.num = len(self.lm_ctx)

        print(f"Num LM instances: {len(self.lm_ctx)}")

    def __len__(self):
        return self.num
   
    def __getitem__(self, index):
        return self._get_lm(index)
    
    def _get_lm(self, index):
        data = self.lm_ctx[index]
        input_ids = data.astype(int)
        return {
            "input_ids": input_ids
        }

    def _process_lm(self, i, samp, model_data, no_model_data, gen_data):
        input_ids = samp["input_ids"]
        source_len = 1
        
        prompt = None
        sep_idx = find_prompt_response_separator(input_ids)
        if sep_idx is not None:
            source_len = sep_idx
            prompt = input_ids[:source_len]
            input_ids = np.concatenate([input_ids[:source_len], input_ids[source_len+1:]], axis=0)
        
        input_ids = input_ids[:self.max_length]
        input_len = len(input_ids)
        model_data["input_ids"][i][:input_len-1] = torch.tensor(input_ids[:-1], dtype=torch.long)
        model_data["attention_mask"][i][:input_len-1] = 1.0
        if self.args.model_type in ["gpt2"]:
            model_data["position_ids"][i][:input_len-1] = torch.arange(0, input_len-1, dtype=torch.long)
        no_model_data["label"][i][:input_len-1] = torch.tensor(input_ids[1:], dtype=torch.long)
        no_model_data["label"][i][:source_len-1] = -100
        no_model_data["loss_mask"][i][:input_len-1] = 1.0
        no_model_data["loss_mask"][i][:source_len-1] = 0
        
        if prompt is not None:
            gen_data["input_ids"][i][-len(prompt):] = torch.tensor(prompt, dtype=torch.long)
            gen_data["attention_mask"][i][-len(prompt):] = 1.0

    def move_to_device(self, model_data, no_model_data, gen_data, device):
        for k in model_data:
            model_data[k] = model_data[k].to(device)

        for k in no_model_data:
            no_model_data[k] = no_model_data[k].to(device)

        for k in gen_data:
            gen_data[k] = gen_data[k].to(device)

        return model_data, no_model_data, gen_data

    def collate(self, samples):
        bs = len(samples)

        max_length = self.max_length
        
        model_data = {
            "input_ids": torch.ones(bs, max_length, dtype=torch.long) * self.pad_id,
            "attention_mask": torch.zeros(bs, max_length),
        }
        
        if self.args.model_type in ["gpt2"]:
            model_data["position_ids"] = torch.zeros(bs, max_length, dtype=torch.long)
            
        no_model_data = {
            "label": torch.ones(bs, max_length, dtype=torch.long) * -100,
            "loss_mask": torch.zeros(bs, max_length)
        }
        
        gen_data = {
            "input_ids": torch.ones(bs, self.max_prompt_length, dtype=torch.long) * self.pad_id,
            "attention_mask": torch.zeros(bs, self.max_prompt_length, dtype=torch.long),
        }

        for i, samp in enumerate(samples):
            self._process_lm(i, samp, model_data, no_model_data, gen_data)
        
        return model_data, no_model_data, gen_data
