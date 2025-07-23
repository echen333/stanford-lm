import cs336_basics
import torch
import timeit
import argparse

from cs336_basics.utils import Transformer
from cs336_basics.utils_train import get_batch, cross_entropy
import numpy as np
import torch.cuda.nvtx as nvtx


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument("--d_ff", type=int, default=3072)
    parser.add_argument("--num_layers", type=int, default=12)
    parser.add_argument("--num_heads", type=int, default=12)
    parser.add_argument("--data_path", type=str, default="cs336_systems/data/TinyStoriesV2-GPT4-valid.npy")
    return parser.parse_args()


def model_pass(model: Transformer, x, only_forward=True, y=None):
    if not only_forward:
        model.zero_grad(set_to_none=True)

    nvtx.range_push("forward")
    out = model(x)
    nvtx.range_pop()
    if only_forward:
        torch.cuda.synchronize()
        return
    

    out = out.flatten(0, 1)  # of shape B C V -> (B*C) V
    y = y.flatten(0, 1)
    loss = cross_entropy(out, y)

    nvtx.range_push("backward")
    loss.backward()
    torch.cuda.synchronize()
    nvtx.range_pop()

def benchmark():
    # data_path, vocab_sz, d_model, n_heads, rope_theta, context_len, n_layers, device
    import time
    time.sleep(2)


    import torch
    torch.cuda.init()
    torch.tensor([1.0], device='cuda')  # force kernel
    print("CUDA test done")

    args = get_args()
    args.vocab_sz = 10000
    args.rope_theta = 10000
    args.context_len = 128
    args.device = "cuda"
    print(args.num_heads, args.d_model, args.num_layers)
    BATCH_SIZE = 4
    NUM_WARMUPS = 5

    model = Transformer(args.vocab_sz, args.d_model, args.num_heads, args.rope_theta, args.context_len, args.num_layers, args.device)
    print("model", model.device)
    arr = np.load(args.data_path, mmap_mode="r")

    print("finished loading")
    x, y = get_batch(arr, BATCH_SIZE, args.context_len, args.device)

    for _ in range(NUM_WARMUPS):
        model_pass(model, x)
    print("hihi")
    timer = timeit.Timer(stmt=lambda: model_pass(model, x))
    arr = np.array(timer.repeat(100, number=1))
    print(arr)
    print(f"statement has mean: {arr.mean()} and std: {arr.std()}")

    for _ in range(NUM_WARMUPS):
        model_pass(model, x, only_forward=False, y=y)
    timer2 = timeit.Timer(stmt=lambda: model_pass(model, x, only_forward=False, y=y), setup="")
    arr = np.array(timer2.repeat(100, number=1))
    print(arr)
    print(f"statement has mean: {arr.mean()} and std: {arr.std()}")


if __name__ == "__main__":
    benchmark()
