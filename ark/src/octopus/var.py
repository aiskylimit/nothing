import torch

def identify_sink_tokens(x: torch.Tensor, sink_token_value_threshold=20):
    assert len(x.shape) == 3, "Input batch is incorrect"
    # filter sink tokens
    x_detached = x.detach()
    sink_token_values_x = torch.abs(torch.max(x_detached, dim=-1)[0]) / torch.sqrt(torch.mean(torch.square(x_detached), dim=-1))
    is_sink_token_x = sink_token_values_x >= sink_token_value_threshold # (batch, sequence)
    return is_sink_token_x