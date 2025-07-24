import torch
import timeit
import argparse
from contextlib import nullcontext

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.data import get_batch

import numpy as np
import torch.cuda.nvtx as nvtx


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument("--d_ff", type=int, default=3072)
    parser.add_argument("--num_layers", type=int, default=12)
    parser.add_argument("--num_heads", type=int, default=12)
    parser.add_argument("--data_path", type=str, default="cs336_systems/data/TinyStoriesV2-GPT4-valid.npy")
    parser.add_argument("--mixed_precision", type=bool, default=False)
    return parser.parse_args()


def model_pass(model: BasicsTransformerLM, x, only_forward=True, y=None):
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
    args = get_args()
    args.vocab_sz = 10000
    args.rope_theta = 10000
    args.context_len = 128
    args.device = "cuda"
    print(args.num_heads, args.d_model, args.num_layers)
    BATCH_SIZE = 4
    NUM_WARMUPS = 5

    model = BasicsTransformerLM(args.vocab_sz, args.context_len, args.d_model, args.num_layers, args.num_heads, args.d_ff, args.rope_theta)
    model.to(args.device)
    arr = np.load(args.data_path, mmap_mode="r")

    x, y = get_batch(arr, BATCH_SIZE, args.context_len, args.device)

    with torch.amp.autocast("cuda", torch.bfloat16) if args.mixed_precision else nullcontext():
        for _ in range(NUM_WARMUPS):
            model_pass(model, x)
        timer = timeit.Timer(stmt=lambda: model_pass(model, x))
        arr = np.array(timer.repeat(10, number=1))
        print(f"statement with {args.d_model} and mixed_precision: {args.mixed_precision} has mean: {arr.mean():.3f} and std: {arr.std():.3f}")

        for _ in range(NUM_WARMUPS):
            model_pass(model, x, only_forward=False, y=y)
        timer2 = timeit.Timer(stmt=lambda: model_pass(model, x, only_forward=False, y=y), setup="")
        arr = np.array(timer2.repeat(10, number=1))
        print(f"statement with {args.d_model} and mixed_precision: {args.mixed_precision} has mean: {arr.mean():.3f} and std: {arr.std():.3f}")

if __name__ == "__main__":
    benchmark()
