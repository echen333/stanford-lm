import torch
from torch.types import Tensor


def cross_entropy(inputs: Tensor, targets: Tensor):
    shifted_inputs = inputs - inputs.max(dim=-1, keepdim=True)[0]
    seq_len = inputs.shape[-2]
    batch = inputs.shape[0] if inputs.ndim == 3 else 1
    numer = shifted_inputs[torch.arange(seq_len), targets]
    denom = torch.logsumexp(shifted_inputs, dim=-1)

    return (-numer.sum() + denom.sum()) / seq_len / batch
