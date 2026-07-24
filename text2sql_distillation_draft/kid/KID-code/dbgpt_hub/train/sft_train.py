import os
import sys

ROOT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ROOT_PATH)
from typing import TYPE_CHECKING, Any, Dict, List, Optional
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainingArguments
from trl import SFTTrainer
from dbgpt_hub.llm_base.loggings import LogCallback, get_logger
from dbgpt_hub.llm_base.config_parser import get_train_args
from dbgpt_hub.llm_base.load_tokenizer import load_model_and_tokenizer
from dbgpt_hub.data_process.data_utils import (
    get_dataset,
    DataCollatorWithSource,
    preprocess_dataset,
    split_dataset,
)
from dbgpt_hub.configs.config import IGNORE_INDEX
from dbgpt_hub.llm_base.model_trainer import (
    Seq2SeqPeftTrainer,
    ComputeMetrics,
    get_logits_processor,
    plot_loss,
)
import torch
import transformers
from peft import PeftModel
from distill_trainer import DistillTrainer


if TYPE_CHECKING:
    from transformers import TrainerCallback
    from dbgpt_hub.configs.model_args import (
        ModelArguments,
        FinetuningArguments,
        GeneratingArguments,
    )
    from dbgpt_hub.configs.data_args import DataArguments


logger = get_logger(__name__)


def _normalize_hf_peft_path(path: Optional[str]) -> tuple[Optional[str], dict]:
    if not path:
        return None, {}

    if path.startswith("hf://"):
        parts = path[len("hf://") :].strip("/").split("/")
        if len(parts) < 2:
            raise ValueError(f"Invalid hf:// PEFT path: {path}")
        repo_id = "/".join(parts[:2])
        kwargs = {}
        if len(parts) > 2:
            kwargs["subfolder"] = "/".join(parts[2:])
        return repo_id, kwargs

    prefix = "https://huggingface.co/"
    if path.startswith(prefix):
        parts = path[len(prefix) :].strip("/").split("/")
        if len(parts) < 2:
            raise ValueError(f"Invalid Hugging Face PEFT URL: {path}")
        repo_id = "/".join(parts[:2])
        kwargs = {}
        rest = parts[2:]
        if rest[:2] == ["tree", "main"]:
            kwargs["revision"] = "main"
            rest = rest[2:]
        elif rest[:1] == ["tree"] and len(rest) >= 2:
            kwargs["revision"] = rest[1]
            rest = rest[2:]
        if rest:
            kwargs["subfolder"] = "/".join(rest)
        return repo_id, kwargs

    return path, {}


def run_sft(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    generating_args: "GeneratingArguments",
    callbacks: Optional[List["TrainerCallback"]] = None,
):
    dataset = get_dataset(model_args, data_args)
    model, tokenizer = load_model_and_tokenizer(
        model_args, finetuning_args, training_args.do_train
    )
    model.to(model_args.compute_dtype)
    if finetuning_args.use_kd and finetuning_args.sample_source in ["mix_token", "mix_request_teacher", "mix_request_gt", "student", "mask_student"]:
        if "qwen" in model_args.model_name_or_path.lower():
            tokenizer.pad_token_id=151645
            tokenizer.pad_token='<|im_end|>'
            tokenizer.bos_token_id=151644
            tokenizer.bos_token='<|im_start|>'
        elif "codes" in model_args.model_name_or_path.lower():
            tokenizer.pad_token_id=4
            tokenizer.pad_token='<fim_pad>'
            tokenizer.bos_token_id=1
            tokenizer.bos_token='<fim_prefix>'
        tokenizer.padding_side='left'
        training_args.remove_unused_columns=False

    dataset = preprocess_dataset(dataset, tokenizer, data_args, training_args, finetuning_args, "sft")
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        label_pad_token_id=IGNORE_INDEX
        if data_args.ignore_pad_token_for_loss
        else tokenizer.pad_token_id,
    )

    # Override the decoding parameters of Seq2SeqTrainer
    training_args_dict = training_args.to_dict()
    training_args_dict.update(
        dict(
            generation_max_length=training_args.generation_max_length
            or data_args.max_target_length,
            generation_num_beams=data_args.eval_num_beams
            or training_args.generation_num_beams,
        )
    )
    training_args = Seq2SeqTrainingArguments(**training_args_dict)

    # Initialize our Trainer
    if finetuning_args.use_kd:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        teacher_config = transformers.AutoConfig.from_pretrained(
            model_args.teacher_model_path,
        )
        teacher_model = transformers.AutoModelForCausalLM.from_pretrained(
            model_args.teacher_model_path,
            config=teacher_config,
            torch_dtype=model_args.compute_dtype,
            use_safetensors=True
        ).to(device)
        if model_args.teacher_peft_path:
            peft_path, peft_kwargs = _normalize_hf_peft_path(model_args.teacher_peft_path)
            teacher_model = PeftModel.from_pretrained(
                teacher_model,
                peft_path,
                is_trainable=False,
                **peft_kwargs,
            )
        teacher_model.eval()
        teacher_model.requires_grad_(False)

        if finetuning_args.sample_source in ["mix_token", "mix_request_teacher", "mix_request_gt", "student", "mask_student"]:
            copy_model, _ = load_model_and_tokenizer(model_args, finetuning_args, training_args.do_train)
            copy_model.to(model_args.compute_dtype)
        else:
            copy_model=None
        
        trainer = DistillTrainer(
            teacher_model=teacher_model, copy_model=copy_model, assistant_model=None,
            model=model,
            finetuning_args=finetuning_args,
            args=training_args,
            tokenizer=tokenizer,
            data_collator=data_collator,
            callbacks=callbacks,
            compute_metrics=ComputeMetrics(tokenizer)
            if training_args.predict_with_generate
            else None,
            **split_dataset(dataset, data_args, training_args)
        )
    else:
        trainer = Seq2SeqPeftTrainer(
            finetuning_args=finetuning_args,
            model=model,
            args=training_args,
            tokenizer=tokenizer,
            data_collator=data_collator,
            callbacks=callbacks,
            compute_metrics=ComputeMetrics(tokenizer)
            if training_args.predict_with_generate
            else None,
            **split_dataset(dataset, data_args, training_args)
        )

    # Keyword arguments for `model.generate`
    gen_kwargs = generating_args.to_dict()
    gen_kwargs["eos_token_id"] = list(
        set([tokenizer.eos_token_id] + tokenizer.additional_special_tokens_ids)
    )
    gen_kwargs["pad_token_id"] = tokenizer.pad_token_id
    gen_kwargs["logits_processor"] = get_logits_processor()

    # Training
    if training_args.do_train:
        train_result = trainer.train(
            resume_from_checkpoint=training_args.resume_from_checkpoint
        )
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
        trainer.save_model()
        # if trainer.is_world_process_zero() and model_args.plot_loss:
        if model_args.plot_loss:
            plot_loss(training_args.output_dir, keys=["loss"])

    # Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate(metric_key_prefix="eval", **gen_kwargs)
        if (
            training_args.predict_with_generate
        ):  # eval_loss will be wrong if predict_with_generate is enabled
            metrics.pop("eval_loss", None)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Predict
    if training_args.do_predict:
        predict_results = trainer.predict(
            dataset, metric_key_prefix="predict", **gen_kwargs
        )
        if (
            training_args.predict_with_generate
        ):  # predict_loss will be wrong if predict_with_generate is enabled
            predict_results.metrics.pop("predict_loss", None)
        trainer.log_metrics("predict", predict_results.metrics)
        trainer.save_metrics("predict", predict_results.metrics)
        trainer.save_predictions(predict_results)


def train(
    args: Optional[Dict[str, Any]] = None,
    callbacks: Optional[List["TrainerCallback"]] = None,
):
    (
        model_args,
        data_args,
        training_args,
        finetuning_args,
        generating_args,
    ) = get_train_args(args)
    callbacks = [LogCallback()] if callbacks is None else callbacks

    run_sft(
        model_args,
        data_args,
        training_args,
        finetuning_args,
        generating_args,
        callbacks,
    )


def export_model(
    args: Optional[Dict[str, Any]] = None, max_shard_size: Optional[str] = "10GB"
):
    model_args, _, training_args, finetuning_args, _ = get_train_args(args)
    model, tokenizer = load_model_and_tokenizer(model_args, finetuning_args)
    model.save_pretrained(training_args.output_dir, max_shard_size=max_shard_size)
    try:
        tokenizer.save_pretrained(training_args.output_dir)
    except:
        logger.warning("Cannot save tokenizer, please copy the files manually.")


if __name__ == "__main__":
    train()
