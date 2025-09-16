from distiller import Distiller, DistillationCollator, DistillationDataset
from arguments import DataArguments, MTEBArguments, TrainingArguments, ModelArguments
import time 
import os
import sys

import torch
import torch.nn as nn 
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler
from torch.optim import AdamW

# Todo

def prepare_dataset(data_args, model_args):
    dataset = DistillationDataset(data_args, model_args)
    return dataset

def finetune(
    model_args: ModelArguments, 
    data_args: DataArguments,
    training_args: TrainingArguments,
    distiller: Distiller, 
    train_dataset: DistillationDataset,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
    collator: DistillationCollator,
):
    print("Start finetuning...")
    