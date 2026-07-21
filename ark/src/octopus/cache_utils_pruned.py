"""
Pruned KV Cache implementation for Octopus models.

Unlike the standard OctopusDynamicCache which masks pruned tokens with -inf gates,
this implementation actually removes pruned tokens from memory, saving both memory
and computation during attention.
"""

from typing import Any, Optional

import torch

from transformers.cache_utils import Cache, DynamicLayer


class OctopusPrunedCacheLayer(DynamicLayer):
    """
    A cache layer that stores keys, values, and gates, and supports
    actual token removal during pruning.
    
    Unlike OctopusDynamicLayer which masks pruned tokens, this implementation
    removes them entirely from the cache tensors.
    """
    
    def __init__(self):
        super().__init__()
        self.gates_sink: Optional[torch.Tensor] = None
        self.gates_non_sink: Optional[torch.Tensor] = None
        self.is_sink_token: Optional[torch.Tensor] = None
        self.utility_scores: Optional[torch.Tensor] = None
        
        # cache attention mask
        self.cached_attention_mask: Optional[torch.Tensor] = None
        # self._seen_tokens: int = 0
    
    def lazy_initialization(self, key_states: torch.Tensor, value_states: torch.Tensor):
        """Initialize the cache with empty tensors matching the key states."""
        super().lazy_initialization(key_states, value_states)
        self.gates_sink = torch.tensor([], dtype=self.dtype, device=self.device)
        self.gates_non_sink = torch.tensor([], dtype=self.dtype, device=self.device)
        self.is_sink_token = torch.tensor([], dtype=torch.bool, device=self.device)
        self.utility_scores = torch.tensor([], dtype=self.dtype, device=self.device)
        
        # do not init cached_attention_mask
        # use as flag for prefilling/generating stages
    
    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Update the cache with new key, value, and gate states.
        
        Returns the full cached keys, values, and gates (including new additions).
        """
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)
        
        self.keys = torch.cat([self.keys, key_states], dim=-2)
        self.values = torch.cat([self.values, value_states], dim=-2)
        # self._seen_tokens += self.keys.shape[-2]
        
        gates_sink = cache_kwargs.get("gates_sink") if cache_kwargs is not None else None
        if gates_sink is not None:
            self.gates_sink = torch.cat([self.gates_sink, gates_sink], dim=-1)
        gates_non_sink = cache_kwargs.get("gates_non_sink") if cache_kwargs is not None else None
        if gates_non_sink is not None:
            self.gates_non_sink = torch.cat([self.gates_non_sink, gates_non_sink], dim=-1)
        
        is_sink_token = cache_kwargs.get("is_sink_token") if cache_kwargs is not None else None
        if is_sink_token is not None:
            self.is_sink_token = torch.cat([self.is_sink_token, is_sink_token], dim=-1)
        
        causal_mask = cache_kwargs.get("causal_mask") if cache_kwargs is not None else None
        if causal_mask is not None:
            if self.cached_attention_mask is None:
                # prefilling -> reconstruct attn mask and return causal mask intact
                # shape: [batch_size, 1, seq_len, seq_len]
                batch_size, num_kv_heads, seq_len, _ = self.keys.shape
                if causal_mask.size() != (batch_size, 1, seq_len, seq_len):
                    raise ValueError(
                        f"Attention mask should be of size {(batch_size, 1, seq_len, seq_len)}, but is {causal_mask.size()}"
                    )
                dtype_min = torch.finfo(causal_mask.dtype).min # assured that dtype is consistent between model, cache, and causal mask
                
                # tokens that do not get attended by any other tokens are padding tokens
                # i.e. corresponding columns in causal mask is full of -inf (or dtype min)
                self.cached_attention_mask = causal_mask.clone().eq(dtype_min).all(dim=-2)
                self.cached_attention_mask = ~self.cached_attention_mask.repeat(1, num_kv_heads, 1)
            
            else:
                # generating -> add 1 to cached attn mask and construct causal mask
                # repeat from kv heads to query heads too
                assert key_states.shape[-2] == 1, "Unhandled runtime case, can only handle step-by-step token generation (1 token at a time)"
                new_pos = torch.ones(
                    *self.cached_attention_mask.shape[:-1], 1,
                    dtype=self.cached_attention_mask.dtype,
                    device=self.cached_attention_mask.device
                )
                self.cached_attention_mask = torch.cat([self.cached_attention_mask, new_pos], dim=-1)
                num_heads = cache_kwargs.get("num_heads", self.keys.shape[1]) if cache_kwargs is not None else self.keys.shape[1]
                num_kv_heads = self.keys.shape[1]
                _causal_mask = self.cached_attention_mask.clone()
                _causal_mask = _causal_mask.unsqueeze(2).repeat(1, 1, num_heads // num_kv_heads, 1)
                _causal_mask = _causal_mask.reshape(_causal_mask.shape[0], -1, _causal_mask.shape[-1])
                _causal_mask = _causal_mask.unsqueeze(-2)
                
                causal_mask = torch.where(_causal_mask, 0, -torch.inf) # assured cache dtype == model and mask dtype
                causal_mask = causal_mask.to(dtype=self.dtype, device=self.device)
                causal_mask = causal_mask.nan_to_num(neginf=torch.finfo(self.dtype).min)
        
        return self.keys, self.values, self.gates_sink, self.gates_non_sink, self.is_sink_token, causal_mask
    
    def get_seq_length(self) -> int:
        """Returns the current sequence length in the cache."""
        if not self.is_initialized or self.keys is None:
            return 0
        return self.keys.shape[-2]
        # return self._seen_tokens
    
    def update_utility_scores(
        self,
        attention_scores: torch.Tensor,
        attention_score_increases: Optional[torch.Tensor],
        num_key_value_groups: int,
        attention_redistributed: bool
    ) -> None:
        """
        Accumulate attention received by cached tokens.

        Args:
            attention_scores: Per-head attention with shape
                [batch_size, num_heads, seq_len, kv_seq_len];
            increase: Attention score increase after redistribution, same shape as attention_scores;
            attn_redistributed: flag telling if any redistribution occurred. False means no redistribution
                occurred and increase=None
        """
        if not self.is_initialized:
            return
        
        assert type(self.is_sink_token) is torch.Tensor \
            and type(self.gates_sink) is torch.Tensor \
            and type(self.gates_non_sink) is torch.Tensor
        self.utility_scores = torch.where(
            self.is_sink_token,
            self.gates_sink,
            self.gates_non_sink
        )

        # attention_scores = attention_scores.sum(dim=2)
        # batch_size, num_heads, kv_seq_len = attention_scores.shape
        # attention_scores = attention_scores.reshape(batch_size, num_heads//num_key_value_groups, num_key_value_groups, kv_seq_len)
        # attention_scores = attention_scores.sum(dim=2)
        # attention_scores = attention_scores.to(dtype=self.dtype)
        
        # if self.utility_scores is None or self.utility_scores.numel() == 0:
        #     self.utility_scores = attention_scores.clone()
        #     if attention_redistributed:
        #         assert attention_score_increases is not None, "Attention Scores redistributed but increase tensor is not passed"
        #         attention_score_increases = attention_score_increases.sum(dim=2)
        #         attention_score_increases = attention_score_increases.reshape(batch_size, num_heads//num_key_value_groups, num_key_value_groups, kv_seq_len)
        #         attention_score_increases = attention_score_increases.sum(dim=2)
        #         attention_score_increases = attention_score_increases.to(dtype=self.dtype)
        #         self.utility_scores = self.utility_scores + attention_score_increases
        #     return

        # if self.utility_scores.shape[:2] != attention_scores.shape[:2]:
        #     raise ValueError(
        #         "Cumulative attention scores should match batch/head dimensions, "
        #         f"but got cached {self.utility_scores.shape[:2]} and new {attention_scores.shape[:2]}"
        #     )

        # cached_seq_len = self.utility_scores.shape[-1]
        # score_seq_len = attention_scores.shape[-1]
        # if cached_seq_len < score_seq_len:
        #     pad_width = score_seq_len - cached_seq_len
        #     padding = attention_scores.new_zeros(*attention_scores.shape[:2], pad_width)
        #     self.utility_scores = torch.cat([self.utility_scores, padding], dim=-1)
        # elif cached_seq_len > score_seq_len:
        #     raise ValueError(
        #         "New attention scores should cover all cached tokens, "
        #         f"but got cached seq len {cached_seq_len} and new seq len {score_seq_len}"
        #     )

        # self.utility_scores = self.utility_scores + attention_scores
        
        # # add attn score increase into utility score
        # if attention_redistributed:
        #     assert attention_score_increases is not None, "Attention Scores redistributed but increase tensor is not passed"
        #     attention_score_increases = attention_score_increases.sum(dim=2)
        #     attention_score_increases = attention_score_increases.reshape(batch_size, num_heads//num_key_value_groups, num_key_value_groups, kv_seq_len)
        #     attention_score_increases = attention_score_increases.sum(dim=2)
        #     attention_score_increases = attention_score_increases.to(dtype=self.dtype)
        #     self.utility_scores = self.utility_scores + attention_score_increases
    
    def prune_by_gate_scores(
        self,
        budget: int,
        recent_window: int = 0,
        sink_tokens: int = 0,
    ) -> None:
        """
        Prune this layer's KV cache by REMOVING low-importance tokens.
        
        Unlike the masking approach, this actually removes tokens from the cache,
        reducing memory usage and computation in subsequent attention operations.
        
        Strategy:
        1. Always keep the first `sink_tokens` (attention sinks / system prompt)
        2. Always keep the last `recent_window` tokens (recent context)
        3. From remaining middle tokens, keep top-K by gate score
        
        Args:
            budget: Total number of tokens to keep per KV head.
            recent_window: Number of recent tokens to always keep.
            sink_tokens: Number of initial tokens to always keep.
        """
        self.gates = torch.tensor([], device=self.device, dtype=self.dtype)
        if not self.is_initialized or self.gates is None or self.gates.numel() == 0:
            return
        
        batch_size, num_kv_heads, seq_len = self.gates.shape
        
        # If budget >= seq_len, no pruning needed
        if budget >= seq_len:
            return
        
        # We need to select which tokens to keep
        # The selection is done per batch and per head
        
        # Build the keep mask
        keep_mask = torch.zeros(batch_size, num_kv_heads, seq_len, dtype=torch.bool, device=self.gates.device)
        
        # Always keep sink tokens
        if sink_tokens > 0:
            keep_mask[..., :sink_tokens] = True
        
        # Always keep recent tokens
        if recent_window > 0:
            keep_mask[..., -recent_window:] = True
        
        # Calculate how many tokens to select by gate score from the middle
        guaranteed_tokens = min(sink_tokens, seq_len) + min(recent_window, max(0, seq_len - sink_tokens))
        tokens_to_select = max(0, budget - guaranteed_tokens)
        
        if tokens_to_select > 0:
            # Define the middle region
            middle_start = sink_tokens
            middle_end = seq_len - recent_window if recent_window > 0 else seq_len
            
            if middle_end > middle_start:
                # Get gates for middle region
                middle_gates = self.gates[..., middle_start:middle_end]
                middle_len = middle_gates.shape[-1]
                
                # Select top-k from middle region
                k = min(tokens_to_select, middle_len)
                if k > 0:
                    _, top_indices = torch.topk(middle_gates, k=k, dim=-1, sorted=False)
                    # Adjust indices to global positions
                    top_indices = top_indices + middle_start
                    # Mark these positions
                    keep_mask.scatter_(-1, top_indices, True)
        
        # Now actually remove the tokens
        # We need to handle this carefully since different heads might want different tokens
        # For simplicity, we'll keep the union of tokens across all heads within a batch
        # This is a conservative approach that keeps more tokens but is simpler
        
        # Alternative: keep per-head selection (more memory efficient but more complex)
        # For now, let's do per-head selection since that's the whole point
        
        # Since we're doing per-head selection, we need to gather the kept tokens
        # This is tricky because different heads keep different tokens
        
        # For efficiency, let's use a simpler approach:
        # Take the union of kept tokens across heads (per batch item)
        # This ensures consistent sequence length across heads
        keep_mask_union = keep_mask.any(dim=1, keepdim=True).expand_as(keep_mask)
        
        # Count how many tokens we're keeping per batch
        num_kept = keep_mask_union[0, 0].sum().item()
        
        if num_kept >= seq_len:
            return  # No tokens to prune
        
        # Create indices for gathering
        # Shape: [batch, num_kv_heads, num_kept]
        kept_indices = keep_mask_union[0, 0].nonzero(as_tuple=True)[0]  # Same for all batch/heads
        
        # Gather the kept tokens
        # keys/values: [batch, num_kv_heads, seq_len, head_dim] -> [batch, num_kv_heads, num_kept, head_dim]
        # gates: [batch, num_kv_heads, seq_len] -> [batch, num_kv_heads, num_kept]
        
        kept_indices_kv = kept_indices.view(1, 1, -1, 1).expand(batch_size, num_kv_heads, -1, self.keys.shape[-1])
        kept_indices_g = kept_indices.view(1, 1, -1).expand(batch_size, num_kv_heads, -1)
        
        self.keys = torch.gather(self.keys, dim=2, index=kept_indices_kv)
        self.values = torch.gather(self.values, dim=2, index=kept_indices_kv)
        self.gates = torch.gather(self.gates, dim=2, index=kept_indices_g)
    
    def prune_by_utility_scores(self, budget: int, recent_window: int = 0, sink_tokens: int = 0) -> None:
        """
        Prune KV cache using per-head, per-batch top-k selection based on utility scores,
        while always preserving sink tokens and recent window tokens.
        """

        if not self.is_initialized or self.utility_scores is None or self.utility_scores.numel() == 0:
            return

        # assert self.keys is not None # for debug purposes
        
        # assert self.utility_scores.shape[0] == self.keys.shape[0], "Mismatch batch size"
        # assert self.utility_scores.shape[2] == self.keys.shape[2], "Mismatch kv_seq_len"
        batch_size, num_kv_heads, seq_len = self.utility_scores.shape
        # _, num_heads, _ = self.gates_sink.shape

        # No pruning needed
        if budget >= seq_len:
            return

        device = self.utility_scores.device
        dtype = self.utility_scores.dtype


        # ---- build mask for guaranteed tokens ----
        keep_mask = torch.zeros(
            batch_size, num_kv_heads, seq_len, dtype=torch.bool, device=device
        )
        
        # identify which sequence at which head still has padding tokens
        assert self.cached_attention_mask is not None
        num_padding_tokens_remaining = (~self.cached_attention_mask).sum(dim=-1).int()
        
        # if any item at any head has padding tokens, need to manually filter
        if torch.any(num_padding_tokens_remaining != 0).item():
            for b in range(batch_size):
                for h in range(num_kv_heads):
                    num_padding_tokens = int(num_padding_tokens_remaining[b, h].item()) # num padding tokens for item b at head h
                    actual_seq_len = seq_len - num_padding_tokens
                    if actual_seq_len <= budget:
                        # actual seq length is smaller than or equal to budget -> evict enough padding tokens (leftmost) to fit
                        keep_mask[b, h, -budget:] = True
                    else:
                        # actual seq length is larger than budget -> first evict all padding tokens
                        # Always keep sink tokens
                        if sink_tokens > 0:
                            keep_mask[b, h, num_padding_tokens:num_padding_tokens+sink_tokens] = True
                        
                        # Always keep recent tokens
                        if recent_window > 0:
                            keep_mask[b, h, -recent_window:] = True
                        
                        guaranteed_tokens = min(sink_tokens, actual_seq_len) + min(recent_window, max(0, actual_seq_len - sink_tokens))
                        tokens_to_select = max(0, budget - guaranteed_tokens)
                        
                        # ---- Step 3: select top-k from middle per head ----
                        if tokens_to_select > 0:
                            # Define the middle region
                            middle_start = num_padding_tokens + sink_tokens
                            middle_end = seq_len - recent_window if recent_window > 0 else seq_len
                            
                            if middle_end > middle_start:
                                # Get gates for middle region
                                middle_scores = self.utility_scores[b, h, middle_start:middle_end]
                                middle_len = middle_scores.shape[-1]
                                
                                # Select top-k from middle region
                                k = min(tokens_to_select, middle_len)
                                if k > 0:
                                    _, top_indices = torch.topk(middle_scores, k=k, dim=-1, sorted=False)
                                    # Adjust indices to global positions
                                    top_indices = top_indices + middle_start
                                    # Mark these positions
                                    keep_mask[b, h].scatter_(-1, top_indices, True)
        # otherwise, use vectorized pytorch ops for efficiency
        else:
            guaranteed_tokens = min(sink_tokens, seq_len) + min(recent_window, max(0, seq_len - sink_tokens))
            tokens_to_select = max(0, budget - guaranteed_tokens)
            # Always keep sink tokens
            if sink_tokens > 0:
                keep_mask[..., :sink_tokens] = True
            
            # Always keep recent tokens
            if recent_window > 0:
                keep_mask[..., -recent_window:] = True
            
            # ---- Step 3: select top-k from middle per head ----
            if tokens_to_select > 0:
                # Define the middle region
                middle_start = sink_tokens
                middle_end = seq_len - recent_window if recent_window > 0 else seq_len
                
                if middle_end > middle_start:
                    # Get gates for middle region
                    middle_scores = self.utility_scores[..., middle_start:middle_end]
                    middle_len = middle_scores.shape[-1]
                    
                    # Select top-k from middle region
                    k = min(tokens_to_select, middle_len)
                    if k > 0:
                        _, top_indices = torch.topk(middle_scores, k=k, dim=-1, sorted=False)
                        # Adjust indices to global positions
                        top_indices = top_indices + middle_start
                        # Mark these positions
                        keep_mask.scatter_(-1, top_indices, True)
        
        # ---- Step 4: convert mask -> indices per head ----
        # We now have exactly `budget` True per (batch, head) (unless seq_len < budget)
        
        # kept_indices = keep_mask.nonzero(as_tuple=False)
        # shape: [N, 3] → (batch, head, position)
        
        # We need to reshape into [batch, num_kv_heads, k]
        # First, count per head (should be uniform = budget or seq_len)
        # counts = keep_mask.sum(dim=-1)  # [batch, num_kv_heads]
        
        # k = int(counts.max().item()) # should be equal to budget since budget < seq_len by condition checking above
        # assert k == budget, "k != budget"
        
        # # Create index tensor
        # gathered_indices = torch.zeros(
        #     batch_size, num_kv_heads, k, dtype=torch.long, device=device
        # )

        # # Fill indices per (batch, head)
        # for b in range(batch_size):
        #     for h in range(num_kv_heads):
        #         idx = keep_mask[b, h].nonzero(as_tuple=True)[0]
        #         assert idx.numel() == gathered_indices.shape[-1], f"num of tokens kept for item {b} at head {h} != budget"
        #         gathered_indices[b, h, : idx.numel()] = idx

        # # ---- Step 5: (optional but recommended) sort indices ----
        # # gathered_indices, _ = torch.sort(gathered_indices, dim=-1)

        # # ---- Step 6: gather tensors ----
        # head_dim = self.keys.shape[-1]

        # idx_kv = gathered_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
        
        
        keep_mask_union = keep_mask.any(dim=(0,1), keepdim=True).expand_as(keep_mask)
        # Count how many tokens we're keeping per batch
        num_kept = keep_mask_union[0, 0].sum().item()
        if num_kept >= seq_len:
            return  # No tokens to prune
        # Create indices for gathering
        # Shape: [batch, num_kv_heads, num_kept]
        kept_indices = keep_mask_union[0, 0].nonzero(as_tuple=True)[0]  # Same for all batch/heads
        # Gather the kept tokens
        # keys/values: [batch, num_kv_heads, seq_len, head_dim] -> [batch, num_kv_heads, num_kept, head_dim]
        # gates: [batch, num_kv_heads, seq_len] -> [batch, num_kv_heads, num_kept]
        idx_kv = kept_indices.view(1, 1, -1, 1).expand(batch_size, num_kv_heads, -1, self.keys.shape[-1])
        gathered_indices = kept_indices.view(1, 1, -1).expand(batch_size, num_kv_heads, -1)

        self.keys = torch.gather(self.keys, dim=2, index=idx_kv)
        self.values = torch.gather(self.values, dim=2, index=idx_kv)
        self.utility_scores = torch.gather(self.utility_scores, dim=2, index=gathered_indices)

        # gates: note num_heads may differ from num_kv_heads
        # idx_g = gathered_indices[:, :, None, :].expand(-1, -1, num_heads // num_kv_heads, -1)
        # idx_g = idx_g.reshape(batch_size, -1, k)
        self.gates_sink = torch.gather(self.gates_sink, dim=2, index=gathered_indices)
        self.gates_non_sink = torch.gather(self.gates_non_sink, dim=2, index=gathered_indices)

        # is_sink_token: same shape as utility scores
        self.is_sink_token = torch.gather(self.is_sink_token, dim=2, index=gathered_indices)
        
        # attention mask: same shape as utility scores
        self.cached_attention_mask = torch.gather(self.cached_attention_mask, dim=2, index=gathered_indices)
    
    def reset(self) -> None:
        """Reset the cache to empty state."""
        if self.is_initialized:
            batch_size, num_kv_heads, _, head_dim = self.keys.shape
            # _, num_heads, _ = self.gates_sink.shape
            self.keys = torch.empty(batch_size, num_kv_heads, 0, head_dim, dtype=self.dtype, device=self.device)
            self.values = torch.empty(batch_size, num_kv_heads, 0, head_dim, dtype=self.dtype, device=self.device)
            self.gates_sink = torch.empty(batch_size, num_kv_heads, 0, dtype=self.dtype, device=self.device)
            self.gates_non_sink = torch.empty(batch_size, num_kv_heads, 0, dtype=self.dtype, device=self.device)
            self.is_sink_token = torch.empty(batch_size, num_kv_heads, 0, dtype=torch.bool, device=self.device)
            self.utility_scores = torch.empty(batch_size, num_kv_heads, 0, dtype=self.dtype, device=self.device)
            
            if self.cached_attention_mask is not None:
                self.cached_attention_mask = torch.empty(batch_size, num_kv_heads, 0, dtype=self.cached_attention_mask.dtype, device=self.device)


class OctopusPrunedCache(Cache):
    """
    A pruned KV cache that actually removes low-importance tokens from memory.
    
    Unlike OctopusDynamicCache which masks pruned tokens with -inf gates,
    this implementation removes them entirely, saving memory and computation.
    
    Example:
        ```python
        >>> from octopus.cache_utils_pruned import OctopusPrunedCache
        >>> cache = OctopusPrunedCache()
        >>> # Use in model generation with pruning enabled
        >>> # Tokens will be physically removed from cache when pruning occurs
        ```
    
    Note:
        When tokens are removed, the cache sequence length decreases. This is
        reflected in get_seq_length() and affects subsequent attention computations.
        The causal mask in attention must handle the reduced sequence correctly.
    """
    
    def __init__(self):
        # Initialize with our custom layer class
        super().__init__(layer_class_to_replicate=OctopusPrunedCacheLayer)
        self._seen_tokens = 0  # Track total tokens seen (not current cache size)
    
    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Updates the cache with new key, value, and gate states.
        
        Returns the full cached tensors for this layer.
        """
        while len(self.layers) <= layer_idx:
            self.layers.append(OctopusPrunedCacheLayer())
        
        # Track seen tokens (for position calculations)
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]
        
        return self.layers[layer_idx].update(key_states, value_states, cache_kwargs)
    
    def get_seq_length(self, layer_idx: int = 0) -> int:
        """Returns the current cache sequence length for the given layer."""
        if layer_idx >= len(self.layers):
            return 0
        return self.layers[layer_idx].get_seq_length()
    
    def get_max_cache_shape(self) -> Optional[int]:
        """Returns None as this cache has dynamic size."""
        return None
    
    def get_gates(self, layer_idx: int) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Returns the cached gates for the given layer."""
        if layer_idx >= len(self.layers):
            return None
        return self.layers[layer_idx].gates_sink, self.layers[layer_idx].gates_non_sink
    
    def update_utility_scores(
        self,
        attention_scores: torch.Tensor,
        attention_score_increases: Optional[torch.Tensor],
        num_key_value_groups: int,
        attention_redistributed: bool,
        layer_idx: int
    ) -> None:
        """Accumulate per-layer utility scores."""
        if layer_idx >= len(self.layers):
            return
        self.layers[layer_idx].update_utility_scores(
            attention_scores,
            attention_score_increases,
            num_key_value_groups,
            attention_redistributed
        )

    def get_utility_scores(self, layer_idx: int) -> Optional[torch.Tensor]:
        """Returns the cached utility scores for the given layer."""
        if layer_idx >= len(self.layers):
            return None
        return self.layers[layer_idx].utility_scores
    
    def prune_by_gate_scores(
        self,
        budget: int,
        layer_idx: Optional[int] = None,
        recent_window: int = 0,
        sink_tokens: int = 0,
    ) -> None:
        """
        Prune the KV cache by REMOVING low-importance tokens from memory.
        
        This physically removes tokens from the cache, reducing memory usage
        and computation in subsequent attention operations.
        
        Args:
            budget: Total number of tokens to keep per KV head.
            layer_idx: If provided, only prune the specified layer.
            recent_window: Number of recent tokens to always keep.
            sink_tokens: Number of initial tokens to always keep.
        """
        if layer_idx is not None:
            if layer_idx < len(self.layers):
                self.layers[layer_idx].prune_by_gate_scores(budget, recent_window, sink_tokens)
        else:
            for layer in self.layers:
                layer.prune_by_gate_scores(budget, recent_window, sink_tokens)
    
    def prune_by_utility_scores(
        self,
        budget: int,
        layer_idx: Optional[int] = None,
        recent_window: int = 0,
        sink_tokens: int = 0,
    ) -> None:
        """Prune the KV cache based on utility scores."""
        if layer_idx is not None:
            if layer_idx < len(self.layers):
                self.layers[layer_idx].prune_by_utility_scores(budget, recent_window, sink_tokens)
        else:
            for layer in self.layers:
                layer.prune_by_utility_scores(budget, recent_window, sink_tokens)
    
    def __iter__(self):
        """Yields (keys, values, gates) for each layer."""
        for layer in self.layers:
            yield layer.keys, layer.values, layer.utility_scores, layer.gates_sink, layer.gates_non_sink, layer.is_sink_token
    
    def __len__(self) -> int:
        """Returns the number of layers in the cache."""
        return len(self.layers)
    
    def reset(self) -> None:
        """Reset all layers to empty state."""
        for layer in self.layers:
            layer.reset()
        self._seen_tokens = 0

