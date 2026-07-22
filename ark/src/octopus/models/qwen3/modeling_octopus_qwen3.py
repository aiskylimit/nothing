from typing import Optional, Union

import torch
from torch import nn

from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3RMSNorm,
    Qwen3MLP,
    Qwen3RotaryEmbedding,
    Qwen3Attention,
    GenerationMixin,
    Cache,
    GradientCheckpointingLayer,
    Qwen3PreTrainedModel,
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)

from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb, create_causal_mask, create_sliding_window_causal_mask

from octopus.cache_utils import OctopusDynamicCache
from .configuration_octopus_qwen3 import OctopusQwen3Config
# from .attention import compiled_flex_octopus_attention, uncompiled_flex_octopus_attention, flash_attention_2, get_q_start_pos
from ...var import identify_sink_tokens
from ...utils import apply_attention_mask, repeat_kv, repeat_is_sink_token


class OctopusQwen3Attention(Qwen3Attention):
    """OctopusQwen3Attention is a wrapper around Qwen3Attention that adds octopus-specific functionality.
    """
    
    def __init__(self, config: OctopusQwen3Config, layer_idx: int):
        super().__init__(config, layer_idx)
        
        num_heads = self.config.num_key_value_heads if self.config.num_key_value_heads is not None else self.config.num_attention_heads
        if config.separate_portion_score_layers:
            self.gated_layer_sink = nn.Sequential(
                nn.Linear(self.config.hidden_size, self.config.hidden_size, bias=False, dtype=self.q_proj.weight.dtype, device=self.q_proj.weight.device),
                nn.SiLU(),
                nn.Linear(self.config.hidden_size, num_heads, bias=False, dtype=self.q_proj.weight.dtype, device=self.q_proj.weight.device),
            )
            self.gated_layer_non_sink = nn.Sequential(
                nn.Linear(self.config.hidden_size, self.config.hidden_size, bias=False, dtype=self.q_proj.weight.dtype, device=self.q_proj.weight.device),
                nn.SiLU(),
                nn.Linear(self.config.hidden_size, num_heads, bias=False, dtype=self.q_proj.weight.dtype, device=self.q_proj.weight.device),
            )
            for m in self.gated_layer_sink.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
            for m in self.gated_layer_non_sink.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
        else:
            self.gated_layer = nn.Sequential(
                nn.Linear(self.config.hidden_size, self.config.hidden_size, bias=False, dtype=self.q_proj.weight.dtype, device=self.q_proj.weight.device),
                nn.SiLU(),
                nn.Linear(self.config.hidden_size, num_heads, bias=False, dtype=self.q_proj.weight.dtype, device=self.q_proj.weight.device),
            )
            for m in self.gated_layer.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
        self.separate_modules = config.separate_portion_score_layers
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size, seq_len, hidden_size = hidden_states.shape
        hidden_shape = (batch_size, seq_len, -1, self.head_dim)
        
        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        
        if self.separate_modules:
            gated_states_sink = self.gated_layer_sink(hidden_states).transpose(1, 2)  # (batch_size, num_heads, seq_len)
            gated_states_non_sink = self.gated_layer_non_sink(hidden_states).transpose(1, 2)
            gated_states_sink = torch.sigmoid(gated_states_sink.float()).to(dtype=key_states.dtype)
            gated_states_non_sink = torch.sigmoid(gated_states_non_sink.float()).to(dtype=key_states.dtype)
        else:
            gated_states_sink = self.gated_layer(hidden_states).transpose(1, 2)
            gated_states_sink = nn.functional.logsigmoid(gated_states_sink.float()).to(dtype=key_states.dtype)
            gated_states_non_sink = gated_states_sink.clone()
        
        is_sink_token = identify_sink_tokens(hidden_states, self.config.sink_token_value_threshold) # (batch_size, seq_len)
        # _is_sink_token = is_sink_token.clone()
        _is_sink_token = torch.zeros_like(is_sink_token)
                
        # extend is_sink_token to store sink token information per KV head
        # since each KV head keeps different tokens, corresponding is_sink_token must also be kept accordingly
        is_sink_token = is_sink_token.unsqueeze(1).repeat(1, key_states.shape[1], 1)
        
        # 1. Apply RoPE
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        
        # cache attention mask: reconstruct from causal mask
        # during prefilling: attention mask should be square (seq_len, seq_len); Cache returns it intact
        # during generation: overwrite the causal mask with Cache-constructed causal mask
        # assert attention_mask is not None, "No causal mask is passed!!!"
        # dummy attn mask; during inference it is constructed by the cache anyways
        if attention_mask is None:
            # cases: no padding -> fine, no padding mask; 1 token only (generate step) -> fine, just dummy and cache rebuilds
            attention_mask = torch.ones(batch_size, 1, seq_len, seq_len, dtype=torch.bool, device=hidden_states.device)
            attention_mask = torch.tril(attention_mask)
        if attention_mask.dtype == torch.bool:
            attention_mask = torch.where(
                attention_mask,
                0.0,
                -torch.inf
            )
            attention_mask = attention_mask.to(dtype=self.q_proj.weight.dtype, device=self.q_proj.weight.device)
            attention_mask = attention_mask.nan_to_num(neginf=torch.finfo(self.q_proj.weight.dtype).min)
        if past_key_values is not None:
            # Update cache with keys, values, and gates
            cache_kwargs = {
                "sin": sin, "cos": cos, "cache_position": cache_position, 
                "gates_sink": gated_states_sink, "gates_non_sink": gated_states_non_sink,
                "is_sink_token": is_sink_token,
                "causal_mask": attention_mask,
                "num_heads": self.config.num_attention_heads,
            }
            key_states, value_states, gated_states_sink, gated_states_non_sink, is_sink_token, attention_mask = past_key_values.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )
        
        kv_seq_len = key_states.shape[-2]
        
        # merge portion scores of sink and non-sink tokens into unified tensor
        # for single-module case, this basically returns the module's output
        gated_states = torch.where(
            is_sink_token,
            gated_states_sink,
            gated_states_non_sink
        )
        # for returning
        _gated_states = gated_states.clone()
        
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        is_sink_token = repeat_is_sink_token(is_sink_token, self.num_key_value_groups)
        gated_states = repeat_is_sink_token(gated_states, self.num_key_value_groups)
        # gated_states_sink = repeat_is_sink_token(gated_states_sink, self.num_key_value_groups)
        # gated_states_non_sink = repeat_is_sink_token(gated_states_non_sink, self.num_key_value_groups)
        
        attention_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling
        use_base_attention = kwargs.get("use_base_attention", False)
        if not use_base_attention:
            gated_states = gated_states[:, :, None, :].repeat(1, 1, seq_len, 1)
            attention_weights = attention_weights + gated_states

        if attention_weights.size() != (batch_size, self.config.num_attention_heads, seq_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(batch_size, self.config.num_attention_heads, seq_len, kv_seq_len)}, but is"
                f" {attention_weights.size()}"
            )

        attention_weights = apply_attention_mask(
            attention_weights,
            attention_mask,
            batch_size=batch_size,
            seq_len=seq_len,
            kv_seq_len=kv_seq_len,
        )

        # upcast attention to fp32
        attention_weights = nn.functional.softmax(attention_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        
        
        # attention redistribution
        attn_redistributed = False
        reduce_sink, increase_non_sink = None, None
        # use_base_attention = kwargs.get("use_base_attention", False)
        # if not use_base_attention:
            # if is_sink_token.size() != (batch_size, self.config.num_attention_heads, kv_seq_len):
            #     raise ValueError(
            #         f"is_sink_token should be of size {(batch_size, self.config.num_attention_heads, kv_seq_len)}, but is {is_sink_token.size()}"
            #     )
            
            # gated_states = gated_states[:, :, None, :].repeat(1, 1, seq_len, 1)
            
            # if attention_mask is not None:
            #     # apply causal mask onto gating scores
            #     causal_mask = attention_mask[:, :, :, :kv_seq_len]
            #     gated_states = torch.where(
            #         causal_mask == 0,
            #         gated_states,
            #         0.0
            #     ).to(dtype=gated_states.dtype, device=gated_states.device)
            
            # compute in float32 for higher precision
            # gated_attention_weights = attention_weights.float() * gated_states.float()
            # attention_budget = (attention_weights.float() - gated_attention_weights).sum(dim=-1, keepdim=True)
            # portion_scores = gated_attention_weights.clone()
                        
            # # portion_scores = gated_states.clone().float()
            # sequence_ids = kwargs.get("sequence_ids", None)
            # if sequence_ids is not None:
            #     # build and apply packed seq attn mask on gated states to compute portion scores
            #     packed_attention_mask = build_packed_attention_mask(sequence_ids)[:, :, :, :kv_seq_len]
            #     portion_scores = torch.where(
            #         packed_attention_mask,
            #         portion_scores,
            #         0.0
            #     ).to(dtype=portion_scores.dtype, device=portion_scores.device)
            
            # only redist. to non-sink tokens; after applying packed sequence attn mask
            # is_sink_token = is_sink_token.unsqueeze(-2)
            # _portion_scores = torch.where(
            #     is_sink_token,
            #     0.0,
            #     portion_scores
            # ).to(dtype=portion_scores.dtype, device=portion_scores.device) # filter portion scores of sink tokens
            # recover rows of original portion_scores where all 0 after filtering (i.e. all sink tokens)
            # all_sink_tokens = torch.all(_portion_scores == 0, dim=-1, keepdim=True)
            # portion_scores = torch.where(
            #     all_sink_tokens,
            #     portion_scores,
            #     _portion_scores
            # )
            # del _portion_scores
            
            # portion_scores = portion_scores / (portion_scores.sum(dim=-1, keepdim=True) 
            #                                          + torch.tensor(1e-9, dtype=torch.float32, device=attention_weights.device))
            # portion_scores = nn.functional.softmax(portion_scores, dim=-1, dtype=torch.float32)
            # redistributed_weights = attention_budget * portion_scores
            # complete redist. and downcast
            # attention_weights = (gated_attention_weights + attention_budget * portion_scores).to(dtype=attention_weights.dtype)
            
            # sink_portions = torch.where(
            #     is_sink_token,
            #     gated_states_sink,
            #     -torch.inf
            # ).to(dtype=gated_states_sink.dtype, device=gated_states_sink.device)
            # sink_portions = sink_portions.unsqueeze(-2).repeat(1, 1, seq_len, 1)
            # if attention_mask is not None:
            #     causal_mask = attention_mask[:, :, :, :kv_seq_len]
            #     # sink_portions = sink_portions + causal_mask
            #     sink_portions = torch.where(
            #         causal_mask == torch.finfo(causal_mask.dtype).min,
            #         -torch.inf,
            #         sink_portions
            #     ).to(dtype=sink_portions.dtype, device=sink_portions.device)
            #     sink_portions = sink_portions.nan_to_num(neginf=torch.finfo(sink_portions.dtype).min)
            # sink_portions = nn.functional.sigmoid(sink_portions)
            # if torch.any(sink_portions != 0):
            #     reduce_sink = attention_weights * sink_portions
                
            #     non_sink_portions = torch.where(
            #         is_sink_token,
            #         -torch.inf,
            #         gated_states_non_sink,
            #     ).to(dtype=gated_states_non_sink.dtype, device=gated_states_non_sink.device)
            #     non_sink_portions = non_sink_portions.unsqueeze(-2).repeat(1, 1, seq_len, 1)
            #     if attention_mask is not None:
            #         causal_mask = attention_mask[:, :, :, :kv_seq_len]
            #         # non_sink_portions = non_sink_portions + causal_mask
            #         non_sink_portions = torch.where(
            #             causal_mask == torch.finfo(causal_mask.dtype).min,
            #             -torch.inf,
            #             non_sink_portions
            #         ).to(dtype=non_sink_portions.dtype, device=non_sink_portions.device)
            #         non_sink_portions = non_sink_portions.nan_to_num(neginf=torch.finfo(non_sink_portions.dtype).min)
                
            #     # for rows with sink tokens only, no redistribution
            #     # handle non-sink token portions: set all to 0 to receive none from budget
            #     all_min = (non_sink_portions == torch.finfo(non_sink_portions.dtype).min).all(dim=-1, keepdim=True)
            #     non_sink_portions = nn.functional.softmax(non_sink_portions, dim=-1, dtype=torch.float32).to(dtype=non_sink_portions.dtype)
            #     non_sink_portions = torch.where(all_min, torch.zeros_like(non_sink_portions), non_sink_portions)
                
            #     if torch.any(non_sink_portions != 0):
            #         attn_redistributed = True
            #         # handle sink token portions: no redistribution ~ no sink tokens' attn scores reduction + no budget formed
            #         reduce_sink = torch.where(all_min, torch.zeros_like(reduce_sink), reduce_sink)
            #         attention_budget = reduce_sink.clone().sum(dim=-1, keepdim=True)
            #         increase_non_sink = non_sink_portions * attention_budget
            #         attention_weights = attention_weights - reduce_sink + increase_non_sink
        
        
        if past_key_values is not None:
            past_key_values.update_utility_scores(
                attention_weights,
                increase_non_sink,
                self.num_key_value_groups,
                attn_redistributed,
                self.layer_idx
            )
        
        attention_output = torch.matmul(attention_weights, value_states)

        if attention_output.size() != (batch_size, self.config.num_attention_heads, seq_len, self.head_dim):
            raise ValueError(
                f"`attention_output` should be of size {(batch_size, self.config.num_attention_heads, seq_len, self.head_dim)}, but is"
                f" {attention_output.size()}"
            )

        attention_output = attention_output.transpose(1, 2).contiguous()
        _attention_output = attention_output.detach().clone() if use_base_attention else attention_output.clone()
        attention_output = attention_output.reshape(batch_size, seq_len, self.head_dim*self.config.num_attention_heads)
        attention_output = self.o_proj(attention_output)
        
        return attention_output, (_gated_states, _is_sink_token, _attention_output)
        

class OctopusQwen3DecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: OctopusQwen3Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = OctopusQwen3Attention(config=config, layer_idx=layer_idx)

        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attention_type = config.layer_types[layer_idx]
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        hidden_states, attention_outputs = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return (hidden_states, attention_outputs)

class OctopusQwen3Model(Qwen3PreTrainedModel):
    def __init__(self, config: OctopusQwen3Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [OctopusQwen3DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.has_sliding_layers = "sliding_attention" in self.config.layer_types

        # Initialize weights and apply final processing
        self.post_init()
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = OctopusDynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # It may already have been prepared by e.g. `generate`
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            # Prepare mask arguments
            mask_kwargs = {
                "config": self.config,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            # Create the masks
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
            }
            # The sliding window alternating layers are not always activated depending on the config
            if self.has_sliding_layers:
                causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        
        attentions = ()
        kv_cache_budget = getattr(self.config, "kv_cache_budget", None)
        kv_cache_recent_window = getattr(self.config, "kv_cache_recent_window", 64)
        kv_cache_sink_tokens = getattr(self.config, "kv_cache_sink_tokens", 4)
        
        for layer_idx, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            hidden_states, attention_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs,
            )
            attentions = attentions + (attention_outputs,)
            
            # Apply KV cache pruning based on cumulative attention if budget is specified
            if kv_cache_budget is not None and past_key_values is not None:
                past_key_values.prune_by_utility_scores(
                    budget=kv_cache_budget,
                    layer_idx=layer_idx,
                    recent_window=kv_cache_recent_window,
                    sink_tokens=kv_cache_sink_tokens,
                )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            attentions=attentions,
        )
        
class OctopusQwen3ForCausalLM(Qwen3PreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}
    _no_split_modules = ["OctopusQwen3DecoderLayer"]
    
    def __init__(self, config):
        super().__init__(config)
        self.model = OctopusQwen3Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()
        self.config.use_base_attention = True
        
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
        
        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        
        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )