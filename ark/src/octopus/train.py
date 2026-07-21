import os
import torch
import torch.nn.functional as F
import torch.distributed as torch_dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import get_state_dict, StateDictOptions
import argparse

from transformers import AutoTokenizer

from octopus.models import OctopusQwen3ForCausalLM, OctopusLlamaForCausalLM, OctopusLlamaConfig, OctopusQwen3Config
from octopus.models.qwen3.modeling_octopus_qwen3 import OctopusQwen3DecoderLayer
from octopus.models.llama.modeling_octopus_llama import OctopusLlamaDecoderLayer

import octopus.distributed as dist
from octopus.logger import get_logger
from octopus.data import build_dataset
from octopus.optimizer import build_optimizer, build_scheduler
import octopus.utils as utils

def get_model_class(model_name: str):
    """Return the appropriate model class and decoder layer class based on model name."""
    model_name_lower = model_name.lower()
    if "llama" in model_name_lower:
        return OctopusLlamaForCausalLM, OctopusLlamaDecoderLayer, OctopusLlamaConfig
    elif "qwen" in model_name_lower:
        return OctopusQwen3ForCausalLM, OctopusQwen3DecoderLayer, OctopusQwen3Config
    else:
        raise ValueError(f"Unknown model type for: {model_name}. Supported: llama, qwen")


def setup_model(model_name, logger, gradient_checkpointing: bool = False, separate_portion_score_layers: bool = False):
    model_class, decoder_layer_class, config_class = get_model_class(model_name)
    logger.info(f"Using model class: {model_class.__name__}, decoder layer: {decoder_layer_class.__name__}")
    
    config = config_class.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        use_cache=False,
        attn_implementation="eager",
    )
    config.separate_portion_score_layers = separate_portion_score_layers
    model = model_class.from_pretrained(
        model_name,
        config=config,
        dtype=torch.bfloat16,
    )
    logger.info(f"Model config: {model.config}")
    
    for name, param in model.named_parameters():
        if "gated_layer" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    
    dist.setup_distributed()
    local_rank = dist.get_local_rank()
    torch.cuda.set_device(local_rank)
    
    dist.apply_fdsp_2(model, modules_to_shard=[decoder_layer_class], activation_checkpointing=gradient_checkpointing)
    
    # # Compile AFTER FSDP
    # model = torch.compile(
    #     model,
    #     mode="max-autotune",      # or "reduce-overhead"
    #     fullgraph=False,
    #     dynamic=False,
    # )

    logger.info(
        f"Model size: {utils.get_model_size(model)} | "
        f"Number of parameters: {utils.get_num_params(model) / 1e9:.2f}B | "
        f"Trainable parameters: {utils.get_trainable_params(model) / 1e6:.2f}M"
    )
    
    return model


def save_consolidated_checkpoint(tokenizer, model, output_dir: str, logger, cpu_offload: bool = True):
    logger.info(f"Saving consolidated checkpoint to {output_dir}")
    
    # DCP requires options to gather the full state dict to CPU
    sd_options = StateDictOptions(full_state_dict=True, cpu_offload=cpu_offload)
    state_dict, _ = get_state_dict(model, optimizers=(), options=sd_options)
    
    if dist.get_global_rank() == 0:
        model.save_pretrained(output_dir, state_dict=state_dict, safe_serialization=True)
        tokenizer.save_pretrained(output_dir)


def forward_kl(logits, teacher_logits, labels):
    teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
    inf_mask = torch.isinf(logits) | torch.isinf(teacher_logits)
    teacher_logprobs = F.log_softmax(teacher_logits, dim=-1, dtype=torch.float32)
    student_logprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
    prod_probs = torch.masked_fill(teacher_probs * student_logprobs, inf_mask, 0)
    prod_probs -= torch.masked_fill(teacher_probs * teacher_logprobs, inf_mask, 0)
    x = torch.sum(prod_probs, dim=-1).view(-1)
    mask = (labels != -100).int()
    distil_loss = -torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)
    return distil_loss.to(dtype=logits.dtype)

def reverse_kl(logits, teacher_logits, labels):
    student_probs = F.softmax(logits, dim=-1, dtype=torch.float32)
    student_logprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
    teacher_logprobs = F.log_softmax(teacher_logits, dim=-1, dtype=torch.float32)
    inf_mask = torch.isinf(teacher_logits) | torch.isinf(logits)
    prod_probs = torch.masked_fill(student_probs * teacher_logprobs, inf_mask, 0)
    prod_probs -= torch.masked_fill(student_probs * student_logprobs, inf_mask, 0)
    x = torch.sum(prod_probs, dim=-1).view(-1)
    mask = (labels != -100).int()
    distil_loss = -torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)
    return distil_loss.to(dtype=logits.dtype)

def skewed_forward_kl(logits, teacher_logits, labels, lam=0.1):
    teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
    student_probs = F.softmax(logits, dim=-1, dtype=torch.float32)
    mixed_probs = lam * teacher_probs + (1-lam) * student_probs
    mixed_logprobs = torch.log(mixed_probs)
    teacher_logprobs = F.log_softmax(teacher_logits, dim=-1, dtype=torch.float32)
    
    mask = (labels != -100).int()
    inf_mask = torch.isinf(logits) | torch.isinf(teacher_logits)

    prod_probs = torch.masked_fill(teacher_probs * mixed_logprobs, inf_mask, 0)
    prod_probs -= torch.masked_fill(teacher_probs * teacher_logprobs, inf_mask, 0)
    x = torch.sum(prod_probs, dim=-1).view(-1)
    distil_loss = -torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)
    return distil_loss

def skewed_reverse_kl(logits, teacher_logits, labels, lam=0.1):
    teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
    student_probs = F.softmax(logits, dim=-1, dtype=torch.float32)
    mixed_probs = (1-lam) * teacher_probs + lam * student_probs
    
    student_logprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
    mixed_logprobs = torch.log(mixed_probs)

    mask = (labels != -100).int()
    inf_mask = torch.isinf(logits) | torch.isinf(teacher_logits)

    prod_probs = torch.masked_fill(student_probs * mixed_logprobs, inf_mask, 0)
    prod_probs -= torch.masked_fill(student_probs * student_logprobs, inf_mask, 0)
    x = torch.sum(prod_probs, dim=-1).view(-1)
    distil_loss = -torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)
    return distil_loss

def regularize_sink_portion_scores(portion_scores: torch.Tensor, is_sink_token: torch.Tensor, var_weight=1.0):
    # portion_scores shape: [layer, batch_size, num_heads, seq_len]
    # is_sink_token shape: [layer, batch_size, seq_len]
    
    # 2 parts: score minimization and entropy/gini
    # entropy only on non-sink tokens
    minimization_term = portion_scores.clone().to(dtype=torch.float32)
    
    # gini_term = 0.0
    # numel = 0
    # layers, batch_size, num_heads, _ = portion_scores.shape
    # for l in range(layers):
    #     for b in range(batch_size):
    #         _is_non_sink_token = ~is_sink_token[l,b]
    #         if _is_non_sink_token.any().int().item() == 0:
    #             continue
    #         _portion_scores_non_sink = portion_scores[l, b, :, _is_non_sink_token].clone().to(dtype=torch.float32)
    #         _gini = _portion_scores_non_sink * (1 - _portion_scores_non_sink)
    #         # _gini = _gini.mean(dim=-1)
    #         gini_term += _gini.sum()
    #         numel += _gini.numel()

    # return (minimization_term + gini_term / max(numel, 1)).to(dtype=portion_scores.dtype)
    
    # entropy on all tokens
    gini_term = minimization_term.clone()
    gini_term = gini_term * (1 - gini_term)
    
    return (minimization_term + gini_term).mean().to(dtype=portion_scores.dtype)

def attention_distillation_loss(input: torch.Tensor, target: torch.Tensor):
    return (
        torch.nn.functional.mse_loss(input.to(torch.float32), target.to(torch.float32), reduction="none")
        .mean(dim=tuple(list(range(1, input.dim()))))
        .sum()
        .to(input.dtype)
    )


def train(
    model,
    tokenizer,
    dataloader,
    optimizer,
    scheduler,
    logger,
    num_epochs,
    kd_ratio: float,
    regularization_weight: float,
    attn_loss_weight: float,
    kl_type: str,
    grad_clip: float,
    total_steps: int,
    use_distillation: bool = True,
    two_phase_training: bool = False,
    attn_distillation: bool = False,
    phase1_epochs: int = 1,
    log_interval: int = 10,
    gradient_accumulation_steps: int = 1,
    reg_var_weight: float = 1.0,
):
    
    def reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
        if not torch_dist.is_available() or not torch_dist.is_initialized():
            return tensor
        tensor = tensor.clone()
        torch_dist.all_reduce(tensor, op=torch_dist.ReduceOp.AVG)
        return tensor
    
    device = torch.device("cuda", dist.get_local_rank()) if torch.cuda.is_available() else torch.device("cpu")
    model.train()
    global_step = 0
    if kl_type == "rkl":
        kl_func = reverse_kl
    elif kl_type == "kl" or kl_type == "fkl":
        kl_func = forward_kl
    elif kl_type == "skl" or kl_type == "sfkl":
        kl_func = skewed_forward_kl
    elif kl_type == "srkl":
        kl_func = skewed_reverse_kl
    else:
        raise ValueError(f"Unsupported divergence function {kl_type}")
    
    for epoch in range(num_epochs):
        if hasattr(dataloader, "sampler") and isinstance(dataloader.sampler, torch.utils.data.distributed.DistributedSampler):
            dataloader.sampler.set_epoch(epoch)
        
        # Determine if we should use distillation for this epoch
        # Two-phase training: Phase 1 uses distillation, Phase 2 uses cross entropy only
        if two_phase_training:
            current_phase = 1 if epoch < phase1_epochs else 2
            use_distillation_this_epoch = (current_phase == 1)
            if dist.get_global_rank() == 0 and (epoch == 0 or epoch == phase1_epochs):
                phase_name = "Phase 1 (distillation + gate loss)" if current_phase == 1 else "Phase 2 (cross entropy + gate loss)"
                logger.info(f"Starting {phase_name} at epoch {epoch}")
        else:
            current_phase = None
            use_distillation_this_epoch = use_distillation
        
        for _step, batch in enumerate(dataloader):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            step = epoch * len(dataloader) + _step
            
            labels = batch["input_ids"].clone()
            # labels[batch["attention_mask"] == 0] = -100
            
            # With left-padding + left-shift, the last PAD in each row becomes the
            # context for predicting the first real token — mask it out too.
            # first_real_token_idx = batch["attention_mask"].argmax(dim=1)  # index of first 1 in each row
            # first_real_token_mask = torch.zeros_like(labels, dtype=torch.bool).scatter_(
            #     1, first_real_token_idx.unsqueeze(1), True
            # )
            # labels[first_real_token_mask] = -100
            
            outputs = model(
                batch["input_ids"],
                attention_mask=batch.get("attention_mask", None),
                labels=labels,
                output_attentions=True,
                return_dict=True,
                sequence_ids=batch.get("sequence_ids", None)
            )
            lm_loss = outputs.loss
            
            # regularization loss on portion scores
            portion_scores_outputs = outputs.attentions # tuple length = number of layers, each element is tuple (portion scores, is sink token)
            portion_scores = torch.stack([item[0] for item in portion_scores_outputs], dim=0)
            is_sink_token = torch.stack([item[1] for item in portion_scores_outputs], dim=0)
            redist_attn_output = torch.stack([item[2] for item in portion_scores_outputs], dim=0) # shape [layer, batch, heads, seq, seq]
            reg_loss = regularize_sink_portion_scores(portion_scores, is_sink_token, reg_var_weight)
            
            if use_distillation_this_epoch:
                # Phase 1: include distillation to stabilize module training
                # Loss = cross entropy + distillation loss (2 forward passes)
                with torch.no_grad():
                    teacher_outputs = model(
                        batch["input_ids"],
                        attention_mask=batch.get("attention_mask", None),
                        # labels=labels,
                        output_attentions=True,
                        return_dict=True,
                        use_base_attention=True
                    )
                base_attn_output = torch.stack([item[2] for item in teacher_outputs.attentions], dim=0)
                attn_distill_loss = torch.tensor(0.0, device=device, dtype=outputs.logits.dtype)
                if attn_distillation:
                    attn_distill_loss = attention_distillation_loss(redist_attn_output, base_attn_output)
                
                # compute distillation loss; need to shift positions to manually compute loss
                logits = outputs.logits[:, :-1, :].contiguous()
                teacher_logits = teacher_outputs.logits[:, :-1, :].contiguous()
                kd_loss = kl_func(logits, teacher_logits, labels[:, 1:].contiguous()) # forward or reverse
                
                total_loss = (1 - kd_ratio) * lm_loss\
                    + kd_ratio * kd_loss\
                    + attn_loss_weight * attn_distill_loss\
                    + regularization_weight * reg_loss # include kd_ratio
            else:
                # Phase 2: full language modeling
                # Loss = cross entropy (1 forward pass)
                kd_loss = torch.tensor(0.0, device=device)
                attn_distill_loss = torch.tensor(0.0, device=device)
                
                total_loss = lm_loss + regularization_weight * reg_loss
            
            total_loss = total_loss / gradient_accumulation_steps
            total_loss.backward()
            
            if (step + 1) % gradient_accumulation_steps == 0:
                if grad_clip is not None and grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            
            # Metrics (reduce across ranks for consistent logging)
            lm_loss_val = reduce_mean(lm_loss.detach())
            kd_loss_val = reduce_mean(kd_loss.detach())
            attn_loss_val = reduce_mean(attn_distill_loss.detach())
            reg_loss_val = reduce_mean(reg_loss.detach())
            total_loss_val = reduce_mean(total_loss.detach())
            lr = scheduler.get_last_lr()[0]
            progress = 100.0 * (global_step + 1) / max(total_steps, 1)
            
            # Optional: monitor gating activation statistics
            # gate_means = torch.stack([g.exp().mean() for _, _, g in outputs.attentions]).mean().detach()
            # gate_means = reduce_mean(gate_means)
            
            if dist.get_global_rank() == 0:
                if global_step % log_interval == 0 \
                    and step % gradient_accumulation_steps == 0: # log on first forward step only
                    phase_str = f" | phase={current_phase}" if two_phase_training else ""
                    logger.info(
                        f"epoch={epoch} | step={step}{phase_str} | total_loss={total_loss_val.item():.4f} | "
                        f"lm_loss={lm_loss_val.item():.4f} | kd_loss={kd_loss_val.item():.4f} | "
                        f"attn_loss={attn_loss_val.item():.4f} | reg_loss={reg_loss_val.item():.4f} | "
                        f"lr={lr:.6f} | progress={progress:.2f}%"
                    )

            if (step + 1) % gradient_accumulation_steps == 0:
                global_step += 1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--dataset-name", type=str, default="unsloth/alpaca-cleaned")
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--max-sample", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--num-epochs", type=int, default=2)
    parser.add_argument("--phase1-epochs", type=int, default=2)
    parser.add_argument("--kd-ratio", type=float, default=0.9)
    parser.add_argument("--regularization-weight", type=float, default=0.5)
    parser.add_argument("--regularization-variance-weight", type=float, default=1.0)
    parser.add_argument("--attn-loss-weight", type=float, default=1)
    parser.add_argument("--kl-type", type=str, default="fkl")
    parser.add_argument("--no-distill", action="store_false")
    parser.add_argument("--no-attn-distill", action="store_false")
    parser.add_argument("--separate-portion-score-layers", action="store_true")
    parser.add_argument("--sink-token-value-threshold", type=int, default=20)
    parser.add_argument("--output-dir", type=str, default="checkpoints/llama-8b-alpaca-cleaned")
    args = parser.parse_args()
    
    # MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
    # MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
    MODEL_NAME = args.model_name
    # DATASET_NAME = "open-r1/Mixture-of-Thoughts"
    # DATASET_NAME = "unsloth/alpaca-cleaned"
    DATASET_NAME = args.dataset_name
    # MAX_SEQ_LENGTH = 24576
    # MAX_SEQ_LENGTH = 2048
    MAX_SEQ_LENGTH = args.max_seq_length
    # MAX_SAMPLE = 20000 # mixture-of-thoughts
    # MAX_SAMPLE = None # alpaca
    MAX_SAMPLE = args.max_sample
    BATCH_SIZE = args.batch_size
    GRADIENT_ACCUMULATION_STEPS = args.grad_accum # adjust to fulfill global batch size compensating for less GPUs
    LEARNING_RATE = 1e-4
    MIN_LR = 1e-6
    WEIGHT_DECAY = 0.01
    BETAS = (0.9, 0.95)
    EPS = 1e-8
    NUM_EPOCHS = args.num_epochs
    WARMUP_RATIO = 0.05
    KD_RATIO = args.kd_ratio
    REGULARIZATION_WEIGHT = args.regularization_weight
    REGULARIZATION_VARIANCE_WEIGHT = args.regularization_variance_weight
    ATTENTION_LOSS_WEIGHT = args.attn_loss_weight
    KL_TYPE = args.kl_type
    GRAD_CLIP = 1.0
    USE_DISTILLATION = False
    TWO_PHASE_TRAINING = args.no_distill  # Enable two-phase training
    ATTENTION_DISTILLATION = args.no_attn_distill # enable attention distillation
    SEPARATE_PORTION_SCORE_LAYERS = args.separate_portion_score_layers
    PHASE1_EPOCHS = args.phase1_epochs  # Number of epochs for Phase 1 (distillation + lm loss)
    GRADIENT_CHECKPOINTING = True
    OUTPUT_DIR = os.getenv("OUTPUT_DIR", args.output_dir)
    
    # notes on num. epochs: Octopus use 1 epoch per phase
    # phase 1 warms up modules, phase 2 trains modules on Next Token Prediction
    # ours only train on NTP, maybe should keep 2 epochs to ensure well training
    # maybe follow 2-phase training: phase 1 includes KD loss term; phase 2 only LM loss
    
    utils.set_seed()
    logger = get_logger()
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = setup_model(MODEL_NAME, logger, GRADIENT_CHECKPOINTING, SEPARATE_PORTION_SCORE_LAYERS)
    # setup model config
    model.config.sink_token_value_threshold = args.sink_token_value_threshold
    
    dataloader = build_dataset(DATASET_NAME, tokenizer, MAX_SEQ_LENGTH, BATCH_SIZE, max_sample=MAX_SAMPLE)
    optimizer = build_optimizer(model, LEARNING_RATE, WEIGHT_DECAY, BETAS, EPS)
    
    num_training_steps = (len(dataloader) // GRADIENT_ACCUMULATION_STEPS) * NUM_EPOCHS
    num_warmup_steps = int(num_training_steps * WARMUP_RATIO)
    logger.info(f"Number of training steps: {num_training_steps} | Number of warmup steps: {num_warmup_steps}")
    scheduler = build_scheduler(optimizer, num_warmup_steps, num_training_steps, MIN_LR)
    
    train(
        model,
        tokenizer,
        dataloader,
        optimizer,
        scheduler,
        logger,
        NUM_EPOCHS,
        KD_RATIO,
        REGULARIZATION_WEIGHT,
        ATTENTION_LOSS_WEIGHT,
        KL_TYPE,
        GRAD_CLIP,
        num_training_steps,
        use_distillation=USE_DISTILLATION,
        two_phase_training=TWO_PHASE_TRAINING,
        attn_distillation=ATTENTION_DISTILLATION,
        phase1_epochs=PHASE1_EPOCHS,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        reg_var_weight=REGULARIZATION_VARIANCE_WEIGHT,
    )
    
    save_consolidated_checkpoint(tokenizer, model, OUTPUT_DIR, logger, cpu_offload=False)
    dist.cleanup_distributed()
    
if __name__ == "__main__":
    main()
