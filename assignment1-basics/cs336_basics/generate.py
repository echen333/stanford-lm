import torch
from utils_train import load_checkpoint
import wandb
import hydra, wandb, os, torch, random
from cs336_basics.tokenizer import Tokenizer
from omegaconf import OmegaConf
from hydra.utils import instantiate
from cs336_basics.train import Transformer, AdamW, softmax

@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg):
    model_path = "checkpoints/med_550.pt"
    prompt: str = "Once upon a time, "
    model: Transformer = instantiate(cfg.model)
    optim: AdamW = instantiate(cfg.optimizer, model.parameters())
    load_checkpoint(model_path, model, optim)
    print("model", model)

    end_of_text_token = "<|endoftext|>"
    tokenizer = Tokenizer.from_files(cfg.vocab_path, cfg.merges_path, [end_of_text_token])

    encoded_prompt = tokenizer.encode(prompt)

if __name__ == "__main__":
    main()