from cs336_basics.utils_train import (
    get_lr_cosine_schedule,
    clip_gradients,
    AdamW,
    cross_entropy,
    get_batch,
    save_checkpoint,
    load_checkpoint,
)
from cs336_basics.utils import softmax, Transformer

import hydra, wandb, os, torch, random
from cs336_basics.tokenizer import Tokenizer
from omegaconf import OmegaConf
from hydra.utils import instantiate
import numpy as np
from torch import Tensor


# @hydra.main(config_path="conf", config_name="config", version_base=None)
@hydra.main(config_path="conf", config_name="config_small", version_base=None)
def main(cfg):
    print(cfg)
    model: Transformer = instantiate(cfg.model)
    optim: AdamW = instantiate(cfg.optimizer, model.parameters())
    # end_of_text_token = "<|endoftext|>"
    # tokenizer = Tokenizer.from_files(cfg.vocab_path, cfg.merges_path, [end_of_text_token])

    # tokenizer.enc

    print("model", model)
    print("optim", optim)
    dataset = np.load(cfg.data_path, mmap_mode="r")
    print(dataset[:100], "max", np.max(dataset))
    for step in range(cfg.max_steps):
        x, y = get_batch(dataset, cfg.batch_size, model.context_length, "cpu")
        print("x, y", x.dtype, y.dtype)
        out: Tensor = model(x)

        out = out.flatten(0, 1)  # of shape B C V -> (B*C) V
        y = y.flatten(0, 1)

        loss = cross_entropy(out, y)
        optim.zero_grad()
        loss.backward()
        optim.step()
        print(f"Loss at step {step} is {loss.item()}")


if __name__ == "__main__":
    main()
