from functools import partial

import datasets
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from transformers import PreTrainedTokenizerBase

from octopus.distributed import get_global_rank, get_world_size

def build_dataset(
    dataset_name: str,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
    batch_size: int,
    num_workers: int = 0,
    max_sample: int | None = None,
):
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    if "alpaca" in dataset_name:
        dataset = AlpacaCleanedDataset(tokenizer, max_length, num_proc=16)
    else:
        dataset = OpenR1Dataset(dataset_name, tokenizer, max_length, max_sample)

    sampler = DistributedSampler(
        dataset,
        num_replicas=get_world_size(),
        rank=get_global_rank(),
        shuffle=True,
        drop_last=False,
    )
    dataloader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    return dataloader
    
# def collate_fn(features, tokenizer: PreTrainedTokenizerBase):
#     batch = tokenizer.pad(
#         {"input_ids": [f["input_ids"] for f in features]},
#         padding=True,
#         return_tensors="pt",
#     )
#     return {
#         "input_ids": batch["input_ids"].to(dtype=torch.long),
#         "attention_mask": batch["attention_mask"].to(dtype=torch.long),
#     }
def collate_fn(features):
    batch = {}
    batch["input_ids"] = torch.tensor([f["input_ids"] for f in features], dtype=torch.long)
    batch["sequence_ids"] = torch.tensor([f["sequence_ids"] for f in features], dtype=torch.long)
    return batch

def tokenize_batch(examples, tokenizer: PreTrainedTokenizerBase):
    # 'examples' is a dict of lists: {"problem": [...], "solution": [...]}
    
    # create conversation format for the whole batch
    if "messages" in examples:
        conversations = examples["messages"]
    elif "problem" in examples and "solution" in examples:
        conversations = []
        for problem, solution in zip(examples["problem"], examples["solution"]):
            conversations.append([
                {"role": "user", "content": problem},
                {"role": "assistant", "content": solution},
            ])
    else:
        raise ValueError(f"Unsupported data format: {examples.keys()}")

    input_ids = []
    sequence_ids = []
    for seq_id, conv in enumerate(conversations):
        ids = tokenizer.apply_chat_template(conv, tokenize=True)["input_ids"]
        input_ids.append(ids)
        sequence_ids.append([seq_id] * len(ids))

    return {"input_ids": input_ids, "sequence_ids": sequence_ids}

def tokenize(sample, tokenizer):
    conversation = [
        {"role": "user", "content": sample["problem"]},
        {"role": "assistant", "content": sample["solution"]},
    ]
    input_ids = tokenizer.apply_chat_template(conversation, tokenize=True)
    return {"input_ids": input_ids}

def group_texts(examples, max_length):
    # Concatenate all input_ids in the batch
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])

    # Option A: Drop the remainder to ensure exactly max_length (Prevents collate_fn errors)
    if total_length >= max_length:
        total_length = (total_length // max_length) * max_length
    
    # Split by chunks of max_length
    result = {
        k: [t[i : i + max_length] for i in range(0, total_length, max_length)]
        for k, t in concatenated_examples.items()
    }
    return result

class OpenR1Dataset(Dataset):
    def __init__(
        self,
        dataset_name: str,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 1024,
        max_sample: int | None = None,
        num_proc: int = 16,
    ):
        rank = get_global_rank()
        
        # 1. Load raw data
        ds = datasets.load_dataset(dataset_name, "all", split="train")
        if max_sample is not None:
            ds = ds.select(range(max_sample))

        # 2. Define processing arguments
        # We must ensure arguments are IDENTICAL across ranks so the hash/fingerprint matches.
        map_kwargs = {
            "batched": True,
            "remove_columns": ds.column_names,
            # "fn_kwargs": {"tokenizer": tokenizer, "max_length": max_length},
        }

        # 3. SYNCHRONIZATION LOGIC
        if rank == 0:
            # Rank 0 does the heavy lifting
            print(f"Rank 0: Processing dataset with {num_proc} workers...")
            ds = ds.map(partial(tokenize_batch, tokenizer=tokenizer), num_proc=num_proc, **map_kwargs)
            ds = ds.map(partial(group_texts, max_length=max_length), num_proc=num_proc, batched=True)
            print("Rank 0: Processing complete.")
        
        # 4. BARRIER: Everyone waits here until Rank 0 finishes writing to cache
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        if rank != 0:
            # Ranks 1-7 run the EXACT same map command but with num_proc=1.
            # Because Rank 0 just finished it, 'datasets' will find the cache 
            # on disk and load it instantly without re-computing.
            print(f"Rank {rank}: Loading processed dataset from cache...")
            ds = ds.map(partial(tokenize_batch, tokenizer=tokenizer), num_proc=1, **map_kwargs)
            ds = ds.map(partial(group_texts, max_length=max_length), num_proc=1, batched=True)

        self.dataset = ds

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # The dataset is now backed by Arrow, we can return directly
        return self.dataset[idx]


def tokenize_alpaca(examples, tokenizer: PreTrainedTokenizerBase):
    # 'examples' is a dict of lists: {"problem": [...], "solution": [...]}
    conversations = []
    for instruction, inp, output in zip(examples["instruction"], examples["input"], examples["output"]):
        input_prompt = instruction
        input_prompt += "\n" + inp if inp else ""
        conversations.append([
            {"role": "user", "content": input_prompt},
            {"role": "assistant", "content": output},
        ])

    input_ids = []
    sequence_ids = []  # <-- NEW
    for seq_id, conv in enumerate(conversations):
        ids = tokenizer.apply_chat_template(conv, tokenize=True)["input_ids"]
        input_ids.append(ids)
        sequence_ids.append([seq_id] * len(ids))  # <-- tag every token with its sample index

    return {"input_ids": input_ids, "sequence_ids": sequence_ids}

class AlpacaCleanedDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        max_length: int = 2048,
        num_proc: int = 16,
    ):
        ds = datasets.load_dataset("unsloth/alpaca-cleaned", split="train")
        
        rank = get_global_rank()
        map_kwargs = {
            "batched": True,
            "remove_columns": ds.column_names,
            # "fn_kwargs": {"tokenizer": tokenizer, "max_length": max_length},
        }

        if rank == 0:
            # Rank 0 does the heavy lifting
            print(f"Rank 0: Processing dataset with {num_proc} workers...")
            ds = ds.map(partial(tokenize_alpaca, tokenizer=tokenizer), num_proc=num_proc, **map_kwargs)
            ds = ds.map(partial(group_texts, max_length=max_length), num_proc=num_proc, batched=True)
            print("Rank 0: Processing complete.")
        
        # 4. BARRIER: Everyone waits here until Rank 0 finishes writing to cache
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        if rank != 0:
            print(f"Rank {rank}: Loading processed dataset from cache...")
            ds = ds.map(partial(tokenize_alpaca, tokenizer=tokenizer), num_proc=1, **map_kwargs)
            ds = ds.map(partial(group_texts, max_length=max_length), num_proc=1, batched=True)

        self.dataset = ds
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]
