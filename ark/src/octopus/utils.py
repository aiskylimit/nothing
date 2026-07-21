from typing import Optional

import torch

def human_readable_size(num_bytes: int) -> str:
    """Converts a number of bytes into a human-readable string (e.g., "1.21 GB")."""
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    if num_bytes == 0:
        return "0 B"
    
    # Calculate the exponent for the unit
    exponent = int(torch.log2(torch.tensor(num_bytes, dtype=torch.float64)) / 10)
    # Ensure exponent is within the range of units
    exponent = min(exponent, len(units) - 1)
    
    size = num_bytes / (1024 ** exponent)
    unit = units[exponent]
    
    return f"{size:.2f} {unit}"

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

def repeat_is_sink_token(is_sink_token: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    repeat_kv reimplemented for is_sink_token
    """
    batch, num_key_value_heads, slen = is_sink_token.shape
    if n_rep == 1:
        return is_sink_token
    is_sink_token = is_sink_token[:, :, None, :].expand(batch, num_key_value_heads, n_rep, slen)
    return is_sink_token.reshape(batch, num_key_value_heads * n_rep, slen)

def apply_attention_mask(
    attention_weights: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    batch_size: int,
    seq_len: int,
    kv_seq_len: int,
) -> torch.Tensor:
    if attention_mask is None:
        return attention_weights

    causal_mask = attention_mask[:, :, :, :kv_seq_len]
    causal_mask_shape = causal_mask.size()
    if causal_mask_shape != (batch_size, 1, seq_len, kv_seq_len) \
        and (causal_mask_shape != (batch_size, attention_weights.shape[1], seq_len, kv_seq_len)):
        raise ValueError(
            f"Attention mask should be of size {(batch_size, 1, seq_len, kv_seq_len)} "
            f"or {(batch_size, attention_weights.shape[1], seq_len, kv_seq_len)}, but is {causal_mask.size()}"
        )

    # attention_weights = attention_weights + causal_mask
    attention_weights = torch.where(
        causal_mask == 0,
        attention_weights,
        -torch.inf
    ).to(dtype=attention_weights.dtype, device=attention_weights.device)
    return attention_weights.nan_to_num(neginf=torch.finfo(attention_weights.dtype).min)

def build_packed_attention_mask(sequence_ids: torch.Tensor) -> torch.Tensor:
    """
    Args:
        sequence_ids: (B, L) integer tensor from the batch
    Returns:
        mask: (B, L, L) bool tensor — True where attention is allowed
              (same sequence AND causal order)
    """
    # Same-sequence mask: token i can attend to token j iff they share a sequence id
    same_seq = sequence_ids.unsqueeze(2) == sequence_ids.unsqueeze(1)  # (B, L, L)

    # Causal mask: token i can only attend to positions j <= i
    L = sequence_ids.size(1)
    causal = torch.ones(L, L, dtype=torch.bool, device=sequence_ids.device).tril()  # (L, L)

    return (same_seq & causal).unsqueeze(1)  # (B, 1, L, L)

def get_model_size(model: torch.nn.Module):
    model_size = sum(p.numel() * p.element_size() for p in model.parameters())
    return human_readable_size(model_size)

def get_num_params(model: torch.nn.Module):
    return sum(p.numel() for p in model.parameters())

def get_trainable_params(model: torch.nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def freeze_model(model: torch.nn.Module):
    for param in model.parameters():
        param.requires_grad = False
        
def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    import random
    import numpy as np
    import torch
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
