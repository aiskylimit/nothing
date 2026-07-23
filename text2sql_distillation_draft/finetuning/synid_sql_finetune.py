import time
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim import AdamW
import deepspeed
from huggingface_hub import login

hf_token = os.getenv("HF_READ_TOKEN")

if hf_token:
    login(token=hf_token, add_to_git_credential=False)

import random
import json
from tqdm import tqdm
import math

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    GenerationConfig)

from transformers import get_constant_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup, get_cosine_schedule_with_warmup
from torch.optim.lr_scheduler import CosineAnnealingLR

try:
    from ._bootstrap import configure_project_paths
except ImportError:
    from _bootstrap import configure_project_paths

configure_project_paths()

from arguments import get_args

from data_utils.lm_datasets import LMTrainDataset
from utils import get_optimizer_params, get_optimizer_params_peft, print_args, initialize
from utils import print_rank, get_rank, OverheadTracker
from utils import save_rank
from utils import all_gather
from utils import get_tokenizer, get_model, resolve_hf_path

from distillm import forward_kl, reverse_kl, js_distance, tv_distance
from distillm import skewed_forward_kl, skewed_reverse_kl, csd
from distillm import ab_div, bdkd, AKL, wsd, alphanet, amid
from distillm import SampleGenerator, ReplayBuffer

from src.synid_sql import SelectedHiddenStateCapture, combine_synid_with_ce, parse_layer_ids, synid_loss
from rouge_metric import compute_metrics

from peft import PeftModel

torch.set_num_threads(4)


def get_teacher_model(args, device):
    teacher_model_path = resolve_hf_path(args.teacher_model_path)
    config = AutoConfig.from_pretrained(teacher_model_path)
    if args.model_parallel:
        raise NotImplementedError
    else:
        config.is_model_parallel = False
        try: model = AutoModelForCausalLM.from_pretrained(teacher_model_path, config=config, device_map={"": device}, torch_dtype=torch.bfloat16)
        except:
            model = AutoModelForCausalLM.from_pretrained(teacher_model_path, config=config, device_map={"": device}, torch_dtype=torch.float32)
            model = model.half()
        
        if args.teacher_peft_path is not None:
            teacher_peft_path = resolve_hf_path(args.teacher_peft_path)
            model = PeftModel.from_pretrained(model, teacher_peft_path)
            model = model.merge_and_unload()
            print("merge_and_unload")

        if dist.get_rank() == 0:
            print(' > number of parameters: {}'.format(
                sum([p.nelement() for p in model.parameters()])), flush=True)

    model.eval()
    
    return model


def get_optimizer(args, model):
    """Set up the optimizer."""

    # Build parameter groups (weight decay and non-decay).
    while isinstance(model, DDP):
        model = model.module

    if args.peft is not None:
        param_groups = get_optimizer_params_peft(args, model)
    else:
        param_groups = get_optimizer_params(args, model)

    # Use AdamW.
    optimizer = AdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)
    print_rank(f'Optimizer = {optimizer.__class__.__name__}')
    return optimizer


def _hidden_size(model):
    config = getattr(model, "config", None)
    if config is not None and hasattr(config, "hidden_size"):
        return config.hidden_size
    base_model = getattr(model, "base_model", None)
    config = getattr(base_model, "config", None)
    if config is not None and hasattr(config, "hidden_size"):
        return config.hidden_size
    raise ValueError("Cannot infer model hidden_size for SynID-SQL projector.")


def attach_synid_projector(args, model, teacher_model, device):
    if teacher_model is None:
        return model

    student_hidden_size = _hidden_size(model)
    teacher_hidden_size = _hidden_size(teacher_model)
    student_layer_ids = parse_layer_ids(args.synid_student_layers)
    teacher_layer_ids = parse_layer_ids(args.synid_teacher_layers)
    if len(student_layer_ids) != len(teacher_layer_ids):
        raise ValueError(
            "SynID student and teacher layer lists must have equal length, "
            f"got {student_layer_ids} and {teacher_layer_ids}."
        )
    num_layer_pairs = len(student_layer_ids)
    dtype = next(model.parameters()).dtype

    def make_projector():
        if student_hidden_size == teacher_hidden_size:
            return nn.Identity()
        projector = nn.Linear(student_hidden_size, teacher_hidden_size, bias=False).to(
            device=device,
            dtype=dtype,
        )
        with torch.no_grad():
            projector.weight.normal_(mean=0.0, std=1e-3)
        return projector

    model.synid_projectors = nn.ModuleList([make_projector() for _ in range(num_layer_pairs)])
    print_rank(
        "SynID projectors: "
        f"{num_layer_pairs} layer-specific projectors "
        f"({student_hidden_size} -> {teacher_hidden_size})"
    )
    return model


def get_learning_rate_scheduler(args, optimizer):
    if args.total_iters is None:
        args.total_iters = args.train_iters_per_epoch * args.epochs
    if args.lr_decay_style == "constant":
        lr_scheduler = get_constant_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_iters)
    elif args.lr_decay_style == "cosine":
        lr_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=args.total_iters,
            eta_min=args.lr_min)
    elif args.lr_decay_style == "noam":
        lr_scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_iters,
            num_training_steps=args.total_iters,
            power=0.5)
    elif args.lr_decay_style == "wrmup_cosine":
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_ratio * args.total_iters,
            num_training_steps=args.total_iters)
    else:
        raise ValueError(f"lr_scheduler of type {args.lr_decay_style} is not supported yet.")

    return lr_scheduler


def setup_model_and_optimizer(args, ds_config, device, set_optim=True, teacher_model=None):
    # get the model
    model = get_model(args, device)
    model = attach_synid_projector(args, model, teacher_model, device)
    # get the optimizer and lr_scheduler
    if set_optim:
        optimizer = get_optimizer(args, model)
        lr_scheduler = get_learning_rate_scheduler(args, optimizer)
    else:
        optimizer, lr_scheduler = None, None
        
    model, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        args=args,
        lr_scheduler=lr_scheduler,
        mpu=None,
        config_params=ds_config
    )
    
    # get the memory usage
    print_rank("Model mem\n", torch.cuda.memory_summary())
    return model, optimizer, lr_scheduler


def prepare_dataset(args, tokenizer):
    data = {}
    rng_sample = random.Random(args.seed)
    
    from torch.utils.data import Subset
    
    if args.do_train:
        if not args.slice_data:
            # Full data
            data["train"] = LMTrainDataset(args, tokenizer, args.data_dir, "train", args.train_num, args.train_ratio, rng_sample)
            data["dev"] = LMTrainDataset(args, tokenizer, args.data_dir, "valid", args.dev_num, args.dev_ratio, rng_sample)
        else:
            # Sliced data for testing
            full_train = LMTrainDataset(args, tokenizer, args.data_dir, "train", args.train_num, args.train_ratio, rng_sample)
            data["train"] = Subset(full_train, range(min(100, len(full_train))))
            data["train"].collate = full_train.collate
            data["train"].move_to_device = full_train.move_to_device
            print_rank("train num", len(data["train"]))
            
            full_dev = LMTrainDataset(args, tokenizer, args.data_dir, "valid", args.dev_num, args.dev_ratio, rng_sample)
            data["dev"] = Subset(full_dev, range(min(20, len(full_dev))))
            data["dev"].collate = full_dev.collate
            data["dev"].move_to_device = full_dev.move_to_device
            if hasattr(full_dev, 'answers'):
                data["dev"].answers = [full_dev.answers[i] for i in data["dev"].indices]

    # if not args.slice_data:
    #     # Full data
    #     data["test"] = LMTrainDataset(args, tokenizer, args.data_dir, "test", args.dev_num, args.dev_ratio, rng_sample)
    # else:
    #     # Sliced data
    #     full_test = LMTrainDataset(args, tokenizer, args.data_dir, "test", args.dev_num, args.dev_ratio, rng_sample)
    #     data["test"] = Subset(full_test, range(min(20, len(full_test))))
    #     data["test"].collate = full_test.collate
    #     data["test"].move_to_device = full_test.move_to_device
    #     if hasattr(full_test, 'answers'):
    #         data["test"].answers = [full_test.answers[i] for i in data["test"].indices]
        
    # pre-trained dataset
    if args.do_train and args.lm_data_dir is not None:
        data["pt_train"] = LMTrainDataset(args, tokenizer, args.lm_data_dir, "train", args.train_num, args.train_ratio, rng_sample)
        print_rank("train num", len(data["pt_train"]))
    return data


def pt_loss(args, model, model_batch, no_model_batch):
    loss_mask = (no_model_batch["label"] != -100).int()
    outputs = model(**model_batch, return_dict=True, use_cache=False)
    logits = outputs.logits
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    lm_loss = loss_fn(logits.view(-1, logits.size(-1)), no_model_batch["label"].view(-1))
    return lm_loss


def move_batch_to_device(batch, device):
    if batch is None:
        return None
    for key in batch:
        if isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].to(device)
    return batch


def get_distil_loss(args, tokenizer, model, teacher_model, model_batch, no_model_batch, logits):
    with torch.no_grad():
        teacher_model.eval()
        teacher_outputs = teacher_model(**model_batch, use_cache=False)
        teacher_logits = teacher_outputs.logits
    if args.model_parallel:
        raise NotImplementedError
    else:
        if "sfkl" in args.type:
            distil_loss = skewed_forward_kl(logits, teacher_logits, no_model_batch, lam=args.skew_alpha)
        elif "srkl" in args.type:
            distil_loss = skewed_reverse_kl(logits, teacher_logits, no_model_batch, lam=args.skew_alpha)
        elif "jsd" in args.type:
            distil_loss = js_distance(logits, teacher_logits, no_model_batch)
        elif "tvd" in args.type:
            distil_loss = tv_distance(logits, teacher_logits, no_model_batch)
        elif "fkl" in args.type or args.type == "kd":
            distil_loss = forward_kl(logits, teacher_logits, no_model_batch)
        elif "rkl" in args.type:
            distil_loss = reverse_kl(logits, teacher_logits, no_model_batch)
        elif "csd" in args.type:
            distil_loss = csd(logits, teacher_logits, no_model_batch)
        elif "bdkd" in args.type:
            distil_loss = bdkd(logits, teacher_logits, no_model_batch)
        elif "akl" in args.type:
            distil_loss = AKL(teacher_logits, logits, no_model_batch)
        elif "wsd" in args.type:
            distil_loss = wsd(logits, teacher_logits, no_model_batch)
        elif "alphanet" in args.type:
            distil_loss = alphanet(logits, teacher_logits, no_model_batch, args.alphanet_alpha, args.alphanet_beta)
        elif "amid" in args.type:
            distil_loss = amid(logits, teacher_logits, no_model_batch, args)
        elif "ab" in args.type:
            distil_loss = ab_div(logits, teacher_logits, no_model_batch, args.ab_alpha, args.ab_beta)
        else:
            raise NotImplementedError
    return distil_loss


def get_teacher_lm_loss(args, tokenizer, model, teacher_model, model_batch):
    with torch.no_grad():
        t_gen_out = teacher_model.generate(
            **model_batch,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            max_length=args.max_length,
            top_k=0,
            top_p=1,
            temperature=1.0,
            do_sample=True,
            return_dict_in_generate=True,
            output_scores=False)
    
    full_ids = t_gen_out.sequences
    
    input_ids = full_ids[:, :-1]
    mask = (input_ids != tokenizer.pad_token_id).long()
    labels = full_ids[:, 1:]    
    labels = torch.masked_fill(labels, mask==0, -100)
    labels[:, :model_batch["input_ids"].size(1)-1] = -100
    loss_mask = (labels != -100).float()
    
    new_batch = {
        "input_ids": input_ids,
        "attention_mask": mask,
    }
    
    if args.model_type in ["gpt2"]:
        position_ids = torch.cumsum(mask, dim=-1) - 1
        position_ids = torch.masked_fill(position_ids, mask==0, 0)    
        new_batch["position_ids"] = position_ids    
    
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    outputs = model(**new_batch, return_dict=True, use_cache=False)
    logits = outputs.logits
    lm_loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))

    return lm_loss


def finetune(args, tokenizer: AutoTokenizer, model: deepspeed.DeepSpeedEngine, optimizer: AdamW, lr_scheduler, dataset, device, teacher_model=None):
    print_rank("Start Fine-tuning")
    if args.kd_ratio is None:
        args.kd_ratio = 1.0
    if not 0.0 <= args.kd_ratio <= 1.0:
        raise ValueError(f"SynID KD ratio must be in [0, 1], got {args.kd_ratio}.")

    # print_inspect(model, '*')
    if args.model_parallel:
        raise NotImplementedError
    else:
        dp_world_size = dist.get_world_size()
        dp_rank = dist.get_rank()
        dp_group = None
        loss_func = nn.CrossEntropyLoss()

    sampler = DistributedSampler(dataset["train"], shuffle=True, drop_last=True, rank=dp_rank, num_replicas=dp_world_size)
    train_dataloader = DataLoader(
        dataset['train'], sampler=sampler, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=dataset["train"].collate)
    if teacher_model is not None and args.type == "synid":
        has_privileged_teacher_inputs = getattr(dataset["train"], "t_lm_ctx", None) is not None
        if args.synid_use_privileged_teacher_input and not has_privileged_teacher_inputs:
            raise ValueError(
                "SynID privileged teacher training requires processed "
                "teacher_train_0.bin/teacher_train_0.idx in --data-dir. "
                "Run process_data.py on a data_dir that contains teacher_train.jsonl, "
                "or pass --synid-use-privileged-teacher-input false."
            )
        if args.synid_use_privileged_teacher_input and has_privileged_teacher_inputs:
            print_rank("Using privileged teacher_train_0 dataset for SynID teacher inputs.")
        else:
            print_rank("Using student train_0 dataset for SynID teacher inputs.")
    
    if "pt_train" in dataset:
        pt_sampler = DistributedSampler(dataset["pt_train"], shuffle=True, drop_last=True, rank=dp_rank, num_replicas=dp_world_size)
        pt_train_dataloader = DataLoader(
        dataset['pt_train'], sampler=pt_sampler, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=dataset["pt_train"].collate)
        pt_train_iter = iter(pt_train_dataloader)
        
    student_generator = SampleGenerator(args, tokenizer)
    student_hidden_capture = None
    teacher_hidden_capture = None
    if teacher_model is not None:
        student_layer_ids = parse_layer_ids(args.synid_student_layers)
        teacher_layer_ids = parse_layer_ids(args.synid_teacher_layers)
        if len(student_layer_ids) != len(teacher_layer_ids):
            raise ValueError(
                "SynID student and teacher layer lists must have equal length, "
                f"got {student_layer_ids} and {teacher_layer_ids}."
            )
        print_rank(f"SynID student layers: {student_layer_ids}")
        print_rank(f"SynID teacher layers: {teacher_layer_ids}")
        student_hidden_capture = SelectedHiddenStateCapture(model.module, student_layer_ids)
        teacher_hidden_capture = SelectedHiddenStateCapture(teacher_model, teacher_layer_ids)

    step, global_step = 1, 1
    total_loss, total_ce_loss, total_distil_loss, total_con1_loss, total_con2_loss, total_time = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    
    adaptive_threshold = args.init_threshold if "adaptive" in args.type else -1.0
    run_train_eval = args.do_valid and not args.log_overhead_metrics
    prev_avg_loss = (
        evaluate(args, tokenizer, model, dataset["dev"], "dev", 0, device, adaptive_threshold)
        if run_train_eval
        else float("inf")
    )
    replay_buffer = ReplayBuffer(args)
    if args.synid_projector_warmup_epochs < 0:
        raise ValueError(
            "SynID projector warmup epochs must be non-negative, "
            f"got {args.synid_projector_warmup_epochs}."
        )
    
    overhead_tracker = OverheadTracker(
        enabled=args.log_overhead_metrics,
        method_name=args.overhead_method_name or args.type,
        save_path=args.save,
        device=device,
    )

    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        overhead_tracker.start_epoch(epoch)

        model.train()
        detach_student_contrastive = teacher_model is not None and epoch < args.synid_projector_warmup_epochs
        if teacher_model is not None:
            contrastive_mode = "projector-only" if detach_student_contrastive else "model+projector"
            print_rank(f"SynID contrastive mode: {contrastive_mode} (epoch {epoch + 1}/{args.epochs})")
        for it, (model_batch, no_model_batch, gen_data, t_model_data, t_no_model_data) in enumerate(train_dataloader):
            dataset["train"].move_to_device(model_batch, no_model_batch, gen_data, device)
            move_batch_to_device(t_model_data, device)
            move_batch_to_device(t_no_model_data, device)
            
            if args.lm_data_dir is not None:
                try:
                    pt_model_batch, pt_no_model_batch, pt_gen_data = next(pt_train_iter)
                    # pt_model_batch, pt_no_model_batch, pt_gen_data = pt_train_iter.next()
                except:
                    pt_train_iter = iter(pt_train_dataloader)
                    # pt_model_batch, pt_no_model_batch, pt_gen_data = pt_train_iter.next()
                    pt_model_batch, pt_no_model_batch, pt_gen_data = next(pt_train_iter)
                    
                dataset["pt_train"].move_to_device(pt_model_batch, pt_no_model_batch, pt_gen_data, device)
            
            torch.cuda.synchronize()
            st_time = time.time()
            
            # # sampling ratio:
            samp_threshold = adaptive_threshold * (1 - global_step / args.total_iters)
            if "adaptive" in args.type:
                if args.replay_ratio == "constant":
                    samp_threshold = adaptive_threshold * 0.5
                elif args.replay_ratio == "increasing":
                    samp_threshold = adaptive_threshold * global_step / args.total_iters
                else:
                    samp_threshold = adaptive_threshold * (1 - global_step / args.total_iters)
            
            # data generation
            if args.student_gen:
                r = np.random.uniform(0, 1)
                if "mixed" in args.type and r < args.mixed_alpha:
                    model_batch = student_generator.run_sample(model, gen_data)
                    no_model_batch["label"] = model_batch.pop("no_model_batch")
                    
                    replay_buffer.move_to_memory(model_batch, no_model_batch)
                    model_batch, no_model_batch, gen_data = replay_buffer.sample()
                    model_batch, no_model_batch = replay_buffer.move_to_device(model_batch, no_model_batch, gen_data, device)
                    
                elif "adaptive" in args.type and (r < samp_threshold or (r < adaptive_threshold and len(replay_buffer) < args.capacity)):

                    model_batch = student_generator.run_sample(model, gen_data)
                    no_model_batch["label"] = model_batch.pop("no_model_batch")
                    
                    if args.model_type in ["opt"]:
                        model_batch.pop('position_ids')
                        
                    replay_buffer.move_to_memory(model_batch, no_model_batch, gen_data)
                    
                elif "adaptive" in args.type and r < adaptive_threshold:
                    model_batch, no_model_batch, gen_data = replay_buffer.sample()
                    model_batch, no_model_batch, gen_data = replay_buffer.move_to_device(model_batch, no_model_batch, gen_data, device)
                    
                model.train()

            if student_hidden_capture is not None:
                student_hidden_capture.clear()
            outputs = model(**model_batch, use_cache=False, output_hidden_states=False)
            
            logits = outputs.logits
            if args.model_parallel:
                raise NotImplementedError
            else:
                lm_loss = loss_func(logits.float().view(-1, logits.shape[-1]), no_model_batch["label"].view(-1))
            
            if teacher_model is not None:
                use_privileged_teacher_batch = (
                    args.synid_use_privileged_teacher_input
                    and t_model_data is not None
                    and t_no_model_data is not None
                )
                teacher_batch = t_model_data if use_privileged_teacher_batch else model_batch
                teacher_no_model_batch = t_no_model_data if use_privileged_teacher_batch else no_model_batch
                with torch.no_grad():
                    teacher_model.eval()
                    teacher_hidden_capture.clear()
                    teacher_outputs = teacher_model(
                        **teacher_batch,
                        use_cache=False,
                        output_hidden_states=False,
                    )
                student_hidden_states = student_hidden_capture.pop_all()
                teacher_hidden_states = teacher_hidden_capture.pop_all()
                loss_parts = synid_loss(
                    args=args,
                    tokenizer=tokenizer,
                    student_outputs=outputs,
                    teacher_outputs=teacher_outputs,
                    student_batch=model_batch,
                    student_no_model_batch=no_model_batch,
                    teacher_batch=teacher_batch,
                    teacher_no_model_batch=teacher_no_model_batch,
                    student_projector=getattr(model.module, "synid_projector", None),
                    student_projectors=getattr(model.module, "synid_projectors", None),
                    student_hidden_states=student_hidden_states,
                    teacher_hidden_states=teacher_hidden_states,
                    detach_student_contrastive=detach_student_contrastive,
                )
                distil_loss = loss_parts.kd
                con1_loss = loss_parts.con1
                con2_loss = loss_parts.con2
                loss = combine_synid_with_ce(
                    lm_loss,
                    loss_parts,
                    kd_ratio=args.kd_ratio,
                    con1_weight=args.synid_alpha,
                    con2_weight=args.synid_beta,
                )
            else:
                loss = lm_loss
                distil_loss = torch.zeros_like(loss)
                con1_loss = torch.zeros_like(loss)
                con2_loss = torch.zeros_like(loss)
                
            if args.lm_data_dir is not None:
                assert args.lm_coef is not None
                loss += args.lm_coef * pt_loss(args, model, pt_model_batch, pt_no_model_batch)
                
            model.backward(loss)
            model.step()
            if student_hidden_capture is not None:
                student_hidden_capture.clear()
                teacher_hidden_capture.clear()
             
            dist.all_reduce(loss, dist.ReduceOp.SUM, group=dp_group)
            global_loss = loss.item() / dp_world_size

            global_distil_loss = 0
            global_ce_loss = 0
            global_con1_loss = 0
            global_con2_loss = 0
            if teacher_model is not None:
                dist.all_reduce(lm_loss, dist.ReduceOp.SUM, group=dp_group)
                dist.all_reduce(distil_loss, dist.ReduceOp.SUM, group=dp_group)
                dist.all_reduce(con1_loss, dist.ReduceOp.SUM, group=dp_group)
                dist.all_reduce(con2_loss, dist.ReduceOp.SUM, group=dp_group)
                global_ce_loss = lm_loss.item() / dp_world_size
                global_distil_loss = distil_loss.item() / dp_world_size
                global_con1_loss = con1_loss.item() / dp_world_size
                global_con2_loss = con2_loss.item() / dp_world_size
                total_ce_loss += global_ce_loss
                total_distil_loss += global_distil_loss
                total_con1_loss += global_con1_loss
                total_con2_loss += global_con2_loss
    
            torch.cuda.synchronize()
            elapsed_time = time.time() - st_time

            total_loss += global_loss
            total_time += elapsed_time
            overhead_tracker.record_step(
                elapsed_time,
                is_optimizer_step=(step % args.gradient_accumulation_steps == 0),
            )

            # Logging
            def get_log(log_loss, log_ce_loss, log_distil_loss, log_con1_loss, log_con2_loss, log_time):
                return "train | epoch {:3d} | Iter: {:6d}/{:6d} | global iter: {:6d}/{:6d} | loss: {:.4f} | ce_loss: {:.4f} | kd_loss: {:.4f} | con1_loss: {:.4f} | con2_loss: {:.4f} | lr: {:.4e} | scale: {:10.4f} | micro time: {:.3f} | step time: {:.3f}".format(
                    epoch,
                    step,
                    args.total_iters * args.gradient_accumulation_steps,
                    global_step,
                    args.total_iters,
                    log_loss,
                    log_ce_loss,
                    log_distil_loss,
                    log_con1_loss,
                    log_con2_loss,
                    lr_scheduler.get_last_lr()[0],
                    optimizer.cur_scale if hasattr(optimizer, "cur_scale") else 0,
                    elapsed_time,
                    log_time,
                )

            if args.mid_log_num > 0:
                mid_log_step = args.gradient_accumulation_steps // args.mid_log_num
                mid_log_step = 1 if mid_log_step == 0 else mid_log_step
                if step % mid_log_step == 0:
                    print_rank(get_log(global_loss, global_ce_loss, global_distil_loss, global_con1_loss, global_con2_loss, 0))

            if global_step % args.log_interval == 0 and step % args.gradient_accumulation_steps == 0:
                log_str = get_log(
                    total_loss / (args.log_interval * args.gradient_accumulation_steps),
                    total_ce_loss / (args.log_interval * args.gradient_accumulation_steps),
                    total_distil_loss / (args.log_interval * args.gradient_accumulation_steps),
                    total_con1_loss / (args.log_interval * args.gradient_accumulation_steps),
                    total_con2_loss / (args.log_interval * args.gradient_accumulation_steps),
                    total_time / (args.log_interval))
                print_rank("*" * 100)
                print_rank(log_str)
                print_rank(args.save)
                print_rank("*" * 100)
                save_rank(log_str, os.path.join(args.save, "log.txt"))
                total_loss, total_ce_loss, total_distil_loss, total_con1_loss, total_con2_loss, total_time = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            
            # Checkpointing
            if args.save and args.save_interval and global_step % args.save_interval == 0 and step % args.gradient_accumulation_steps == 0:
                save_dir_path = os.path.join(args.save, str(global_step))
                if args.model_parallel:
                    raise NotImplementedError
                else:
                    if dist.get_rank() == 0:
                        os.makedirs(save_dir_path, exist_ok=True)
                        print_rank(f"Model save to {save_dir_path}")
                        tokenizer.save_pretrained(save_dir_path)
                        model.module.save_pretrained(save_dir_path, safe_serialization=False)
                dist.barrier()

            # Evaluation
            if run_train_eval and args.eval_interval and global_step % args.eval_interval == 0 and step % args.gradient_accumulation_steps == 0:
                curr_avg_loss = evaluate(args, tokenizer, model, dataset["dev"], "dev", epoch, device, adaptive_threshold)
                if "adaptive" in args.type:
                    if curr_avg_loss >= prev_avg_loss + args.loss_eps:
                        adaptive_threshold += 0.1
                        adaptive_threshold = min(adaptive_threshold, 1.0)
                        prev_avg_loss = curr_avg_loss

                # evaluate(args, tokenizer, model, dataset["test"], "test", epoch, device)
                    
                model.train()
                
            step += 1
            if step % args.gradient_accumulation_steps == 0:
                global_step += 1
            
            if global_step > args.total_iters:
                break

        overhead_tracker.finish_epoch(epoch)
            
    if student_hidden_capture is not None:
        student_hidden_capture.close()
        teacher_hidden_capture.close()
    return model


def evaluate(args, tokenizer, model, dataset: LMTrainDataset, split, epoch, device, adaptive_threshold=None):
    
    collate_fn = dataset.collate

    if args.model_parallel:
        raise NotImplementedError
    else:
        dp_world_size = dist.get_world_size()
        dp_rank = dist.get_rank()
        dp_group = None
        loss_func = nn.CrossEntropyLoss()

    print_rank("dp size", dp_world_size)

    eos_token_id = tokenizer.eos_token_id
    if args.model_type == "qwen":
        eos_token_id = [tokenizer.eos_token_id, 151643]

    generation_config = GenerationConfig(
        do_sample=args.do_sample,
        top_p=args.top_p,
        top_k=args.top_k,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
        max_length=args.max_length,
        min_length=None,
        eos_token_id=eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        return_dict_in_generate=True,
        output_scores=False
    )

    sampler = DistributedSampler(dataset, shuffle=False, drop_last=False, rank=dp_rank, num_replicas=dp_world_size)
    dataloader = DataLoader(
        dataset, sampler=sampler, batch_size=args.eval_batch_size, num_workers=args.num_workers, collate_fn=collate_fn)

    model.eval()
    all_loss = 0.0
    step = 0
    
    all_response_ids = []
    
    with torch.no_grad():
        for it, (model_batch, no_model_batch, gen_data, _, _) in enumerate(tqdm(dataloader, desc="Evaluating", disable=(dist.get_rank() != 0))):
            print_rank(f"{it}/{len(dataloader)}")
            dataset.move_to_device(model_batch, no_model_batch, gen_data, device)
            logits = model(**model_batch).logits
            if args.model_parallel:
                raise NotImplementedError
            else:
                loss = loss_func(logits.view(-1, logits.shape[-1]), no_model_batch["label"].view(-1))
            
            max_new_tokens = args.max_length - gen_data["input_ids"].size(1)
            
            if args.eval_gen:            
                gen_out = model.generate(
                    **gen_data,
                    generation_config=generation_config,
                    max_new_tokens=max_new_tokens)
                
                full_ids = gen_out.sequences
                
                full_ids = F.pad(
                    full_ids,
                    (0, args.max_length - full_ids.shape[1]),
                    value=tokenizer.pad_token_id,
                )
                
                response_ids = full_ids[:, gen_data["input_ids"].size(1):]
                all_response_ids.append(response_ids)
                    
            dist.all_reduce(loss, dist.ReduceOp.SUM, group=dp_group)
            loss = loss / dp_world_size
            all_loss += loss.item()
            step += 1
    
    if args.eval_gen:
        all_response_ids = torch.cat(all_response_ids, dim=0)
        all_response_ids = all_gather(all_response_ids, dim=1, world_size=dp_world_size, group=dp_group, op="stack")
        all_response_ids = all_response_ids.view(-1, all_response_ids.size(-1))
        
        responses = tokenizer.batch_decode(all_response_ids, skip_special_tokens=True)
    
    if get_rank() == 0:
        if args.eval_gen:
            references = dataset.answers
            responses = responses[:len(references)]
            
            res = compute_metrics(responses, references)
        
            eval_dir = os.path.join(args.save, "eval", str(epoch))
            print_rank(eval_dir)
            os.makedirs(eval_dir, exist_ok=True)
            with open(os.path.join(eval_dir, "answers.jsonl"), "w") as f:
                for resp in responses:
                    f.write(json.dumps({"text": resp}) + "\n")
        else:
            res = {}
    
        avg_loss = all_loss / step
        
        if "adaptive" in args.type:
            log_str = f"{split} | avg_loss: {avg_loss} | {res} | threshold: {adaptive_threshold}"
        else:
            log_str = f"{split} | avg_loss: {avg_loss} | {res}"
        print_rank(log_str)
        save_rank(log_str, os.path.join(args.save, "log.txt"))
        
    return all_loss / step


def main():
    torch.backends.cudnn.enabled = False
    
    args = get_args()
    if args.kd_ratio is None:
        args.kd_ratio = 1.0
    if not 0.0 <= args.kd_ratio <= 1.0:
        raise ValueError(f"SynID KD ratio must be in [0, 1], got {args.kd_ratio}.")
    initialize(args)
    
    if dist.get_rank() == 0:
        print_args(args)
        with open(os.path.join(args.save, "args.json"), "w") as f:
            json.dump(vars(args), f)
    
    device = torch.cuda.current_device()
    cur_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    save_rank("\n\n" + "="*30 + f" EXP at {cur_time} " + "="*30, os.path.join(args.save, "log.txt"))
    
    with open(args.deepspeed_config, "r") as f:
        ds_config = json.load(f)

    ds_config["gradient_accumulation_steps"] = args.gradient_accumulation_steps
    ds_config["train_micro_batch_size_per_gpu"] = args.batch_size
    ds_config["gradient_clipping"] = args.clip_grad
    ds_config["steps_per_print"] = 10000000
    
    if not args.do_train:
        ds_config["zero_optimization"]["stage"] = 0
    
    args.fp32 = not ds_config["fp16"]["enabled"]  
    args.bf16 = "bf16" in ds_config and ds_config["bf16"]["enabled"]  
    args.deepspeed_config = None
    
    # get the tokenizer
    tokenizer = get_tokenizer(args)
    print(type(tokenizer))

    dataset = prepare_dataset(
        args,
        tokenizer,
    )
    
    dp_world_size = dist.get_world_size()
    
    if args.do_train:
        args.train_iters_per_epoch = int(len(dataset["train"]) / (args.batch_size * dp_world_size * args.gradient_accumulation_steps))
        print_rank("Train iters per epoch", args.train_iters_per_epoch)
        if args.total_iters is None:
            args.total_iters = args.train_iters_per_epoch * args.epochs
        if args.epochs is None:
            args.epochs = math.ceil(args.total_iters / args.train_iters_per_epoch)
        print_rank("total_iters", args.total_iters)
        
        if args.save_interval == -1:
            args.save_interval = args.train_iters_per_epoch
        
        if args.eval_interval == -1:
            args.eval_interval = args.train_iters_per_epoch
    
    if args.teacher_model_type is None:
        args.teacher_model_type = args.model_type
    
    if args.teacher_model_path is not None:
        teacher_model = get_teacher_model(args, device)
    else:
        teacher_model = None

    model, optimizer, lr_scheduler = setup_model_and_optimizer(
        args,
        ds_config,
        device,
        set_optim=args.do_train,
        teacher_model=teacher_model,
    )
    
    if args.do_train:
        model = finetune(args, tokenizer, model, optimizer, lr_scheduler, dataset, device, teacher_model=teacher_model)
   
    if args.do_eval:
        # evaluate(args, tokenizer, model, dataset["test"], "test", 0, device)
        pass
        
    
if __name__ == "__main__":
    main()
