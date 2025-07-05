import torch
from torch.types import Tensor


def cross_entropy(inputs: Tensor, targets: Tensor):
    shifted_inputs = inputs - inputs.max(dim=-1, keepdim=True)[0]
    seq_len = inputs.shape[-2]
    batch = inputs.shape[0] if inputs.ndim == 3 else 1
    numer = shifted_inputs[torch.arange(seq_len), targets]
    denom = torch.logsumexp(shifted_inputs, dim=-1)

    return (-numer.sum() + denom.sum()) / seq_len / batch


import torch.optim as optim
from typing import Optional, Callable
import math


class AdamW(optim.Optimizer):
    def __init__(self, params, lr, weight_decay, betas, eps=1e-8):
        defaults = {"lr": lr, "betas": betas, "weight_decay": weight_decay, "eps": eps}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            alpha, beta_1, beta_2 = group["lr"], group["betas"][0], group["betas"][1]
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
