import torch
from transformers import Trainer, TrainerCallback
from transformers.trainer_pt_utils import LabelSmoother
import wandb
from specInfer.generator import Generator
from specInfer.common import pad_to_2d
from enum import Enum
import random
from torch.utils.data import DataLoader
import torch.nn as nn
import transformers
import copy
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from dbgpt_hub.llm_base.model_trainer import Seq2SeqPeftTrainer

IGNORE_TOKEN_ID = LabelSmoother.ignore_index


class KLMethod(Enum):
    Forward = 1
    Reverse = 2
    JSD = 3
    SeqKD=4
    TVD=5


KL_METHOD_MAP = {
    "forward": KLMethod.Forward,
    "reverse": KLMethod.Reverse,
    "jsd": KLMethod.JSD,
    "tvd": KLMethod.TVD,
    "seqkd": KLMethod.SeqKD,
}

eval_cnt = 0


import torch.nn.functional as F
def entropy_loss(logits):
    probs =F.softmax(logits, dim=-1)
    log_probs =F.log_softmax(logits, dim=-1)
    entropy = -torch.sum(probs * log_probs, dim=-1)
    return entropy

class DistillTrainer(Seq2SeqPeftTrainer):
    def __init__(self, teacher_model, copy_model, assistant_model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        args = kwargs["args"]
        self.teacher_model = teacher_model
        self.generator = Generator(
            self.model, self.teacher_model, self.tokenizer, 5
        )
        self.train_step_cnt = 0
        self.topk=0.5
        self.beta=0.5
        self.alpha=0.2
        self.setting=kwargs["finetuning_args"].kl_setting
        self.copy_model=copy_model
        self.assistant_model=assistant_model
        self.cross_loss_fct = CrossEntropyLoss()
        self.sample_source = kwargs["finetuning_args"].sample_source
        self.mask_strategy = kwargs["finetuning_args"].mask_strategy
        self.mask_ratio = float(kwargs["finetuning_args"].mask_ratio)
        self.kl_method = KL_METHOD_MAP[kwargs["finetuning_args"].kl_method]

    def training_step(self, model, inputs):
        max_new_tokens = 128
        self.train_step_cnt += 1
        student_temperature = 1.0
        teacher_temperature = 1.0
        student_token_ratio=0.5
        student_request_ratio=0.5

        if self.kl_method == KLMethod.JSD:
            fwd_loss_ratio = 0.5


        # if self.sample_source == "student" and self.train_step_cnt%2==0:
        if self.sample_source == "student":
            try:
                self.copy_model.load_state_dict(model.module.state_dict())
                self.copy_model.to(torch.bfloat16).eval()
                if self.copy_model.device != model.module.device:
                    self.copy_model.to(model.module.device)
            except:
                self.copy_model.load_state_dict(model.state_dict())
                self.copy_model.to(torch.bfloat16).eval()
                if self.copy_model.device != model.device:
                    self.copy_model.to(model.device)
                
            generated_ids, _ = self.get_generated_ids(
                self.copy_model,
                self.tokenizer,
                inputs["all_source_ids"],
                inputs["source_attention_mask"],
                max_new_tokens,
                False,
            )
            generated_ids = generated_ids.clone().detach()
            prompt_len = inputs["all_source_ids"].shape[-1]
            attention_mask = generated_ids != self.tokenizer.pad_token_id
            output_mask = generated_ids[..., :] == self.tokenizer.pad_token_id
            output_mask[..., :prompt_len-1] = True

        elif self.sample_source == "mask_student":
            try:
                self.copy_model.load_state_dict(model.module.state_dict())
                self.copy_model.to(torch.bfloat16).eval()
                if self.copy_model.device != model.module.device:
                    self.copy_model.to(model.module.device)
            except:
                self.copy_model.load_state_dict(model.state_dict())
                self.copy_model.to(torch.bfloat16).eval()
                if self.copy_model.device != model.device:
                    self.copy_model.to(model.device)

            # mask_input_ids=torch.cat((inputs["all_source_ids"], inputs["mask_target_ids"]), dim=-1)

            prompt_len = inputs["all_source_ids"].shape[-1]

            if self.mask_strategy == "uniform":
                mask=[False if i%5==0 else True for i in range(inputs["input_ids"].shape[-1]-inputs["all_source_ids"].shape[-1])]
            elif self.mask_strategy == "random":
                mask=[True if random.random()<(1-self.mask_ratio) else False for i in range(inputs["input_ids"].shape[-1]-inputs["all_source_ids"].shape[-1])]
            elif self.mask_strategy == "hard":
                with torch.no_grad():
                    logits=self.copy_model(inputs["input_ids"]).logits[...,:-1,:].contiguous()
                entropy=entropy_loss(logits.squeeze())[prompt_len:]
                _, indices = entropy.sort(descending=True)
                num=int(self.mask_ratio*(inputs["input_ids"].shape[-1]-inputs["all_source_ids"].shape[-1]))
                mask=[False if i in indices[:num] else True for i in range(inputs["input_ids"].shape[-1]-inputs["all_source_ids"].shape[-1])]
            elif self.mask_strategy == "easy":
                with torch.no_grad():
                    logits=self.copy_model(inputs["input_ids"]).logits[...,:-1,:].contiguous()
                entropy=entropy_loss(logits.squeeze())[prompt_len:]
                _, indices = entropy.sort(descending=False)
                num=int(self.mask_ratio*(inputs["input_ids"].shape[-1]-inputs["all_source_ids"].shape[-1]))
                mask=[False if i in indices[:num] else True for i in range(inputs["input_ids"].shape[-1]-inputs["all_source_ids"].shape[-1])]
            else:
                assert 0, f"Error! The mask strategy is not supportted."

            mask[-1]=True
            mask[0]=True
            mask=[True]*inputs["all_source_ids"].shape[-1]+mask
            inputs["input_ids"]=inputs["input_ids"].masked_fill(~torch.tensor(mask).to(inputs["input_ids"].device), self.tokenizer.bos_token_id)

            with torch.no_grad():
                mask_tokens=torch.argmax(self.copy_model(inputs["input_ids"]).logits[...,:-1,:].contiguous(), -1)
                mask_tokens=torch.cat([inputs["input_ids"][..., 0].unsqueeze(0), mask_tokens], dim=-1)
            generated_ids=torch.where(torch.tensor(mask).to(inputs["input_ids"].device)==True, inputs["input_ids"], mask_tokens)

            # generated_ids = inputs["input_ids"].clone().detach()

            # prompt_len = inputs["all_source_ids"].shape[-1]
            attention_mask = generated_ids != self.tokenizer.pad_token_id
            output_mask = generated_ids[..., :] == self.tokenizer.pad_token_id
            output_mask[..., :prompt_len-1] = True
            output_mask[..., prompt_len-1:] = False
            labels=inputs["labels"]

        elif self.sample_source in ["mix_request_teacher", "mix_request_gt"]:
            if random.random() < student_request_ratio:
                try:
                    self.copy_model.load_state_dict(model.module.state_dict())
                    self.copy_model.to(torch.bfloat16).eval()
                    if self.copy_model.device != model.module.device:
                        self.copy_model.to(model.module.device)
                except:
                    self.copy_model.load_state_dict(model.state_dict())
                    self.copy_model.to(torch.bfloat16).eval()
                    if self.copy_model.device != model.device:
                        self.copy_model.to(model.device)
                    
                generated_ids, _ = self.get_generated_ids(
                    self.copy_model,
                    self.tokenizer,
                    inputs["all_source_ids"],
                    inputs["source_attention_mask"],
                    max_new_tokens,
                    False,
                )
                generated_ids = generated_ids.clone().detach()
                prompt_len = inputs["all_source_ids"].shape[-1]
                attention_mask = generated_ids != self.tokenizer.pad_token_id
                output_mask = generated_ids[..., :] == self.tokenizer.pad_token_id
                output_mask[..., :prompt_len-1] = True
            else:
                generated_ids = inputs["input_ids"]
                labels=inputs["labels"]
                attention_mask = inputs["attention_mask"]
                output_mask = (inputs["labels"][..., :] == IGNORE_TOKEN_ID)
        elif self.sample_source == "mix_token":
            max_new_tokens = 128
            try:
                self.copy_model.load_state_dict(model.module.state_dict())
            except:
                self.copy_model.load_state_dict(model.state_dict())
            generated_ids = self.get_mix_generated_ids(
                self.copy_model,
                self.teacher_model,
                self.tokenizer,
                inputs["prompt_input_ids"],
                inputs["prompt_attention_mask"],
                max_new_tokens,
                student_token_ratio
            )
            generated_ids = generated_ids.clone().detach()
            prompt_len = inputs["prompt_input_ids"].shape[-1]
            attention_mask = generated_ids != self.tokenizer.pad_token_id
            output_mask = generated_ids[..., :] == self.tokenizer.pad_token_id
            output_mask[..., :prompt_len-1] = True
        else:
            generated_ids = inputs["input_ids"]
            labels=inputs["labels"]
            attention_mask = inputs["attention_mask"]
            output_mask = (inputs["labels"][..., :] == IGNORE_TOKEN_ID)
            
        # get student/teacher logits
        student_logits = self.get_logits(model, generated_ids, attention_mask)
        student_logits = student_logits.float()

        if self.sample_source == "mask_student" or self.kl_method == KLMethod.Reverse or self.kl_method == KLMethod.JSD:
            if self.sample_source != "student":
                lm_loss=self.get_mle_loss(student_logits, labels, student_temperature)

        # other KD methods
        with torch.no_grad():
            teacher_logits = self.get_logits(
                self.teacher_model, generated_ids, attention_mask)
            teacher_logits = teacher_logits.float()

        # calculate loss
        if self.kl_method == KLMethod.Forward:
            student_logits, teacher_logits=student_logits[...,:-1,:].contiguous(), teacher_logits[...,:-1,:].contiguous()
            output_mask=output_mask[...,1:]
            loss = self.get_kl(
                student_logits / student_temperature,
                teacher_logits / teacher_temperature,
                output_mask
            )
        elif self.kl_method == KLMethod.Reverse:
            student_logits, teacher_logits=student_logits[...,:-1,:].contiguous(), teacher_logits[...,:-1,:].contiguous()
            output_mask=output_mask[...,1:]
            loss = self.get_kl(
                teacher_logits / teacher_temperature,
                student_logits / student_temperature,
                output_mask
            )
        elif self.kl_method == KLMethod.TVD:
            student_logits, teacher_logits=student_logits[...,:-1,:].contiguous(), teacher_logits[...,:-1,:].contiguous()
            output_mask=output_mask[...,1:]
            loss = self.get_tvd(
                teacher_logits / teacher_temperature,
                student_logits / student_temperature,
                output_mask
            )
        elif self.kl_method == KLMethod.JSD:
            student_logits, teacher_logits=student_logits[...,:-1,:].contiguous(), teacher_logits[...,:-1,:].contiguous()
            output_mask=output_mask[...,1:]
            reverse_loss = self.get_kl(
                teacher_logits / teacher_temperature,
                student_logits / student_temperature,
                output_mask
            )
            fwd_loss = self.get_kl(
                student_logits / student_temperature,
                teacher_logits / teacher_temperature,
                output_mask
            )
            loss = fwd_loss_ratio * fwd_loss + \
                (1 - fwd_loss_ratio) * reverse_loss

        if self.sample_source == "mask_student" or self.kl_method == KLMethod.Reverse or self.kl_method == KLMethod.JSD:
            if self.sample_source != "student":
                loss=lm_loss+loss

        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        self.accelerator.backward(loss)
        # loss.backward()
        return loss.detach()

    def log(self, logs):
        # Remove the 'loss' entry with value 0 before calling the superclass method
        if 'loss' in logs and logs['loss'] == -1:
            del logs['loss']

        # Call the original `log` method of the `Trainer` class
        super().log(logs)

    ###################### Helper Functions #############################

    def get_mle_loss(self, predicts, labels, temperature):
        lm_logits = predicts / temperature
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        lm_loss = self.cross_loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        return lm_loss

    def soft_cross_entropy(self, predicts, targets, padding_mask):
        predict_log_prob = torch.nn.functional.log_softmax(predicts, dim=-1)
        targets_prob = torch.nn.functional.softmax(targets, dim=-1)
        entropy = -targets_prob * predict_log_prob
        expand_mask = padding_mask.unsqueeze(-1).expand_as(entropy)
        entropy.masked_fill_(expand_mask, 0)
        mean_entropy = entropy.sum() / (~padding_mask).sum()
        return mean_entropy  

    def get_kl(self, predicts, targets, padding_mask, reduce=True):
        kl_loss = torch.nn.KLDivLoss(reduction="none", log_target=True)
        predict_prob = torch.nn.functional.log_softmax(predicts, dim=-1)
        targets_prob = torch.nn.functional.log_softmax(targets, dim=-1)
        output = kl_loss(predict_prob, targets_prob)
        if reduce:
            expand_mask = padding_mask.unsqueeze(-1).expand_as(output)
            output.masked_fill_(expand_mask, 0)
            mean_output = output.sum() / (~padding_mask).sum()
            return mean_output
        else:
            return output
    
    def get_tvd(self, s_logits, t_logits, padding_mask):
        s_logits = torch.nn.functional.softmax(s_logits, dim=-1)
        t_logits = torch.nn.functional.softmax(t_logits, dim=-1)
        sel_mask = padding_mask[:, :, None].expand_as(s_logits)
        vocab_size = s_logits.size(-1)
        s_logits_slct = torch.masked_select(s_logits, sel_mask)  # (bs * seq_length * voc_size) modulo the 1s in mask
        t_logits_slct = torch.masked_select(t_logits, sel_mask)  # (bs * seq_length * voc_size) modulo the 1s in mask
        s_logits_slct = s_logits_slct.view(-1, vocab_size)  # (bs * seq_length, voc_size) modulo the 1s in mask
        t_logits_slct = t_logits_slct.view(-1, vocab_size)  # (bs * seq_length, voc_size) modulo the 1s in mask
        assert t_logits_slct.size() == s_logits_slct.size()
        loss_tvd = (0.5 * torch.abs(s_logits_slct-t_logits_slct)).sum(dim=-1).mean()
        return loss_tvd


    def get_logits(self, model, input_ids, attention_mask):
        return model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).logits


    @torch.inference_mode()
    def get_generated_ids(
        self,
        model,
        tokenizer,
        input_ids,
        attention_mask,
        max_new_tokens,
        require_logits,
    ):
        model.generation_config.use_cache=True
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            output_scores=require_logits,
            return_dict_in_generate=True,
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        if require_logits:
            logits = torch.cat(
                [score.unsqueeze(1) for score in outputs["scores"]], dim=1
            )
        else:
            logits = None
        return outputs["sequences"], logits
    
    @torch.inference_mode()
    def get_mix_generated_ids(
        self,
        student_model,
        teacher_model,
        tokenizer,
        input_ids,
        attention_mask,
        max_new_tokens,
        mix_ratio
    ):
        org_input_ids = input_ids.clone()
        def sample_token_from_logits(logits):
            tau = 0.001 # argmax
            distribution = torch.softmax(logits / tau, dim=-1)
            next_token_id = torch.multinomial(distribution, num_samples=1)
            return next_token_id
    
        def generate_one(model, input_ids, attention_mask, past_key_values):
            if past_key_values is None:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=True,
                    use_cache=True,
                )
            else:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=True,
                    use_cache=True,
                    past_key_values=past_key_values,
                )
            past_key_values = outputs.past_key_values
            next_token = sample_token_from_logits(outputs.logits[:, -1, :])
            return next_token, past_key_values

        bsz, prompt_len = input_ids.shape
        # always generate the first token for teacher/student to get the kv cache
        student_first_token, student_key_values = generate_one(
            student_model, input_ids, attention_mask, None)
        teacher_first_token, teacher_key_values = generate_one(
            teacher_model, input_ids, attention_mask, None)
        
        torch.manual_seed(1)
        input_ids = student_first_token if random.random() < mix_ratio else teacher_first_token
        attention_mask = torch.cat([attention_mask, torch.ones(
                bsz, 1, dtype=torch.long, device='cuda')], dim=1)

        for i in range(max_new_tokens - 1):
            sample_model, past_key_values = (student_model, student_key_values) if random.random(
            ) < mix_ratio else (teacher_model, teacher_key_values)
            next_token, _ = generate_one(sample_model, input_ids, 
                                        attention_mask, past_key_values)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            attention_mask = torch.cat([attention_mask, torch.ones(
                bsz, 1, dtype=torch.long, device='cuda')], dim=1)

        # mask eos
        eos_positions = (input_ids == tokenizer.eos_token_id).nonzero(as_tuple=True)
        mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for row, col in zip(*eos_positions):
            mask[row, col+1:] = True
        input_ids[mask] = tokenizer.pad_token_id
        return torch.cat((org_input_ids, input_ids), dim=-1).cuda()
    
