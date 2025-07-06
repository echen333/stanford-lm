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


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg):
    print(cfg)
    model: Transformer = instantiate(cfg.model)
    optim: AdamW = instantiate(cfg.optimizer, model.parameters())
    # end_of_text_token = "<|endoftext|>"
    # tokenizer = Tokenizer.from_files(cfg.vocab_path, cfg.merges_path, [end_of_text_token])

    # tokenizer.enc

    # dataset = np.load(, mmap_mode='r')
    dataset = None
    for step in range(cfg.max_steps):
        x, y = get_batch(dataset, 32, 32, "cpu")
        out = model(x)
        loss = cross_entropy(out, y)
        optim.zero_grad()
        loss.backward()
        optim.step()
        print(f"Loss at step {step} is {loss.item()}")


if __name__ == "__main__":
    main()
