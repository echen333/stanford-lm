import cs336_basics
import torch
import timeit

from cs336_basics.utils import Transformer
from cs336_basics.utils_train import get_batch, cross_entropy
import numpy as np


def forward_pass(model: Transformer, x):
    torch.cuda.synchronize()
    out = model(x)


def backward_pass(model: Transformer, loss):
    torch.cuda.synchronize()
    model.zero_grad()
    loss.backward()


def benchmark(data_path, vocab_sz, d_model, n_heads, rope_theta, context_len, n_layers, device):
    BATCH_SIZE = 4
    NUM_WARMUPS = 5

    model = Transformer(vocab_sz, d_model, n_heads, rope_theta, context_len, n_layers, device)
    with open(data_path, "r") as f:
        arr = np.load(data_path, mmap_mode="r")

    x, y = get_batch(arr, BATCH_SIZE, context_len, device)

    for _ in range(NUM_WARMUPS):
        forward_pass(model, x)
    timer = timeit.timeit("forward_pass(model, x)", number=10)
    print(timer)

    out = model(x)
    out = out.flatten(0, 1)  # of shape B C V -> (B*C) V
    y = y.flatten(0, 1)
    loss = cross_entropy(out, y)

    for _ in range(NUM_WARMUPS):
        backward_pass(model, loss)
    timer2 = timeit.timeit("backward_pass(model, loss)", number=10)
    print(timer2)


if __name__ == "__main__":
    data_path = "data/TinyStoriesV2-GPT4-valid.npy"
    benchmark(data_path, 10000, 128, 2, 10000, 32, 2, "cpu")
