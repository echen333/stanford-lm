import torch
from utils_train import load_checkpoint
import wandb
import hydra, wandb, os, torch, random
from cs336_basics.tokenizer import Tokenizer
from omegaconf import OmegaConf
from hydra.utils import instantiate
from cs336_basics.train import Transformer, AdamW, softmax
import yaml
from torch import Tensor


def main():
    run_name = "misty-energy-168"
    steps = 40000
    checkpoint_name = f"checkpoints/owt_{steps}.pt"

    api = wandb.Api()
    runs = api.runs("stanford-lm-1")
    run = [run for run in runs if run.name == run_name][0]
    run.file("config.yaml").download(replace=True)
    run.file(checkpoint_name).download(replace=True)

    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    # Cleaning wandb values in config
    clean_cfg = {k: v["value"] if isinstance(v, dict) and set(v) == {"value"} else v for k, v in cfg.items()}

    print(clean_cfg)
    omega_cfg = OmegaConf.create(clean_cfg)
    print("omega", omega_cfg)

    model: Transformer = instantiate(omega_cfg.model)
    print("model", model)
    optim_fn = instantiate(omega_cfg.optimizer, _partial_=True)
    optim: AdamW = optim_fn(params=model.parameters())
    print("optim", optim)

    print("checkpoint", checkpoint_name)
    load_checkpoint(checkpoint_name, model, optim)

    prompt: str = "Once upon a ti"
    end_of_text_token = "<|endoftext|>"
    print("cfg", clean_cfg)
    # tokenizer = Tokenizer.from_files(clean_cfg["vocab_path"], clean_cfg["merges_path"], [end_of_text_token])
    # vocab_path = "data/tiny_stories_10000_vocab.json"
    # merges_path = "data/tiny_stories_10000_merges.pkl"
    vocab_path = "data/owt_train_32000_vocab.json"
    merges_path = "data/owt_train_32000_merges.pkl"
    tokenizer = Tokenizer.from_files(vocab_path, merges_path, [end_of_text_token])

    encoded_prompt = torch.tensor(tokenizer.encode(prompt), device=model.device)
    print("encoded_prompt", encoded_prompt, encoded_prompt.dtype)

    generated: Tensor = model.generate(encoded_prompt, max_tokens=10000)
    generated.to("cpu")
    print("generated", generated)
    text = tokenizer.decode(generated.tolist())
    print("output:", text)


if __name__ == "__main__":
    main()
