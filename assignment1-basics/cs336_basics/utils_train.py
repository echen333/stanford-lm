import torch
from torch.types import Tensor
from jaxtyping import Float, Int


def cross_entropy(
    inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]
) -> Float[Tensor, ""]:
    inputs = inputs.to(torch.float32)
    shifted_inputs = inputs - inputs.max(dim=-1, keepdim=True)[0]
    seq_len = inputs.shape[-2]
    batch = inputs.shape[0] if inputs.ndim == 3 else 1
    numer = shifted_inputs[torch.arange(seq_len), targets]
    denom = torch.logsumexp(shifted_inputs, dim=-1)

    return (-numer.sum() + denom.sum()) / seq_len / batch


import torch.optim as optim
from typing import Optional, Callable
import math
import numpy.typing as npt


class AdamW(optim.Optimizer):
    def __init__(self, params, lr, weight_decay, betas, num_steps=40000, eps=1e-8, warmup_steps=400):
        defaults = {"lr": lr, "betas": betas, "weight_decay": weight_decay, "eps": eps}
        self.warmup_steps = warmup_steps
        self.lr = lr
        self.num_steps = num_steps
        self.cur_step = 0
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        self.cur_step += 1
        for group in self.param_groups:
            beta_1, beta_2 = group["betas"][0], group["betas"][1]
            alpha = get_lr_cosine_schedule(self.cur_step, self.lr, 0, self.warmup_steps, self.num_steps)
            for p in group["params"]:  # p of type nn.Parameter
                if p.grad is None:
                    pass

                state = self.state[p]
                t = state.get("t", 1)
                g = p.grad
                state["m"] = beta_1 * state.get("m", torch.zeros_like(g)) + (1 - beta_1) * g
                state["v"] = beta_2 * state.get("v", torch.zeros_like(g)) + (1 - beta_2) * torch.square(g)
                alpha_t = alpha * math.sqrt(1 - beta_2**t) / (1 - beta_1**t)

                p.data -= alpha_t * state["m"] / (torch.sqrt(state["v"]) + group["eps"])
                p.data -= alpha * group["weight_decay"] * p.data
                state["t"] = t + 1
        return loss


def get_lr_cosine_schedule(t, a_max, a_min, T_w, T_c):
    if t < T_w:
        return t / T_w * a_max
    elif t > T_c:
        return a_min
    return a_min + (1 + math.cos((t - T_w) / (T_c - T_w) * math.pi)) * (a_max - a_min) / 2


def clip_gradients(params, max_l2_norm, eps=1e-6):
    total_norm = torch.sqrt(
        sum(torch.norm(param.grad, 2) ** 2 for param in params if param.grad is not None and param.requires_grad)
    )

    if total_norm > max_l2_norm:
        for param in params:
            if param.grad is not None:
                param.grad = param.grad * (max_l2_norm / (total_norm + eps))

def get_batch(dataset: npt.NDArray, batch_size: int, context_length: int, device: str):
    """
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding
    labels from the dataset.

    Args:
        dataset (np.array): 1D numpy array of integer token IDs in the dataset.
        batch_size (int): Desired batch size to sample.
        context_length (int): Desired context length of each sampled example.
        device (str): PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
            to place the sampled input sequences and labels on.

    Returns:
        Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
        is the sampled input sequences, and the second tuple item is the corresponding
        language modeling labels.
    """
    start_idxs = torch.randint(0, len(dataset) - context_length, tuple([batch_size]))
    all_idxs = start_idxs.reshape(-1, 1) + torch.arange(0, context_length)
    all_idxs.to(device)
    x = torch.tensor(dataset[all_idxs.reshape(-1, 1)].reshape(batch_size, -1), device=device, dtype=torch.long)
    ret = torch.tensor(dataset[all_idxs.reshape(-1, 1) + 1].reshape(batch_size, -1), device=device, dtype=torch.long)
    return (x, ret)


import os
import typing


def save_checkpoint(model, optimizer, iteration, out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]):
    model_dict = model.state_dict()
    optim_dict = optimizer.state_dict()
    torch.save({"model": model_dict, "optim": optim_dict, "iteration": iteration}, out)


def load_checkpoint(src, model, optimizer):
    saved_dict = torch.load(src, weights_only=False)
    model.load_state_dict(saved_dict["model"])
    optimizer.load_state_dict(saved_dict["optim"])
    return saved_dict["iteration"]
