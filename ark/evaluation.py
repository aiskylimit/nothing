import torch

import transformers
from transformers import AutoTokenizer

from octopus.models import OctopusQwen3ForCausalLM, OctopusLlamaForCausalLM, OctopusLlamaConfig, OctopusQwen3Config

from lm_eval.api.model import LM
from lm_eval.models.huggingface import HFLM
from lm_eval.api.registry import register_model
from lm_eval.__main__ import cli_evaluate

from src.octopus.cache_utils_pruned import OctopusPrunedCache


def get_model_class(pretrained: str):
    """Return the appropriate model class based on pretrained path/name."""
    pretrained_lower = pretrained.lower()
    if "llama" in pretrained_lower:
        return OctopusLlamaForCausalLM, OctopusLlamaConfig
    elif "qwen" in pretrained_lower:
        return OctopusQwen3ForCausalLM, OctopusQwen3Config
    else:
        raise ValueError(f"Unknown model type for: {pretrained}. Supported: llama, qwen")


@register_model("octopus")
class OctopusEvalWrapper(HFLM):

    def __init__(self, pretrained="checkpoints/llama-8b-alpaca-cleaned", max_length=2048, batch_size=None, device="cuda",
                 dtype=torch.bfloat16, cache_budget=128, cache_recent_window=32, cache_sink_tokens=4,
                 sink_token_value_threshold=20, separate_portion_score_layers=False):
        LM.__init__(self)
        model_class, config_class = get_model_class(pretrained)
        config = config_class.from_pretrained(
            pretrained,
            dtype=torch.bfloat16,
            attn_implementation="eager",
        )
        config.separate_portion_score_layers = separate_portion_score_layers
        self._model = model_class.from_pretrained(pretrained, config=config, device_map={"": device}, torch_dtype=dtype)
        self._model.eval()
        self.AUTO_MODEL_CLASS = model_class
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained)
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.vocab_size = self.tokenizer.vocab_size
        self._batch_size = int(batch_size) if batch_size is not None else 64
        self._max_length = max_length
        self._device = torch.device(device)
        
        self.add_bos_token = getattr(self.tokenizer, "add_bos_token", False)   # or True, depending on your model/tokenizer
        self.add_eos_token = False
        self.backend = "causal"
        # self.prefix_token_id = None
        self.tokenizer_mode = "auto"
        # self._max_gen_toks = max_length // 2
        self.mixed_precision_dtype = None   # set to torch.bfloat16 if you want autocast
        self.truncation = True
        self.logits_cache = True
        self.softmax_dtype = torch.float32
        self.custom_prefix_token_id = None  # NOT self.prefix_token_id — that's a @property
        self.think_end_token = None
        self.chat_template_args = {}
        self.pretrained = pretrained
        self.revision = "main"
        self.peft = None
        self.delta = None
        
        self._model.config.use_base_attention = True
        self._model.config.kv_cache_budget = cache_budget
        self._model.config.kv_cache_recent_window = cache_recent_window
        self._model.config.kv_cache_sink_tokens = cache_sink_tokens
        self._model.config.sink_token_value_threshold = sink_token_value_threshold

    @property
    def batch_size(self):
        return self._batch_size
    
    def _model_generate(self, context, max_length, stop, **generation_kwargs):
        with torch.autocast(
            device_type=self._device.type,
            dtype=self.mixed_precision_dtype,
            enabled=self.mixed_precision_dtype is not None,
        ):
            return self._model.generate(
                input_ids=context,
                attention_mask=generation_kwargs.pop("attention_mask", None),
                max_length=max_length,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
                past_key_values=OctopusPrunedCache(),
            )
    

if __name__ == "__main__":
    cli_evaluate()