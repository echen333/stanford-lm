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
import wandb
from omegaconf import OmegaConf


def upload_dataset(path: str):
    run = wandb.init(entity="eddys", project="stanford-lm-1", job_type="add_dataset")
    artifact = wandb.Artifact(name=f"stories-models", type="dataset")
    artifact.add_file(local_path=path, name="med_700")
    artifact.save()


@hydra.main(config_path="conf", config_name="config", version_base=None)
# @hydra.main(config_path="conf", config_name="config_small", version_base=None)
def main(cfg):
    wandb_cfg = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    print(cfg, type(cfg))
    run = wandb.init(entity="eddys", project="stanford-lm-1", config=wandb_cfg)
    model: Transformer = instantiate(cfg.model)
    optim: AdamW = instantiate(cfg.optimizer, model.parameters())
    # end_of_text_token = "<|endoftext|>"
    # tokenizer = Tokenizer.from_files(cfg.vocab_path, cfg.merges_path, [end_of_text_token])

    # tokenizer.enc

    print("model", model)
    print("optim", optim)

    artifact = run.use_artifact("eddys/stanford-lm-1/tinystories-train:v0", type="dataset")
    artifact_dir = artifact.download()
    print(artifact_dir, type(artifact_dir))

    # dataset = np.load(cfg.data_path, mmap_mode="r")
    dataset = np.load(f"{artifact_dir}/stories-train", mmap_mode="r")
    print(dataset[:100], "max", np.max(dataset))
    for step in range(1, cfg.max_steps + 1):
        x, y = get_batch(dataset, cfg.batch_size, model.context_length, cfg.model.device)
        out: Tensor = model(x)

        out = out.flatten(0, 1)  # of shape B C V -> (B*C) V
        y = y.flatten(0, 1)

        loss = cross_entropy(out, y)
        optim.zero_grad()
        loss.backward()
        optim.step()
        print(f"Loss at step {step} is {loss.item()}")
        run.log({"loss": loss.item()})

        if step % cfg.checkpoint_steps == 0:
            file_path = f"{cfg.checkpoint_path}_{step}.pt"
            folder = os.path.dirname(file_path)
            os.makedirs(folder, exist_ok=True)
            print(f"Saving checkpoint at step {step} to path {file_path}")
            save_checkpoint(model, optim, step, file_path)
            wandb.save(file_path)

    run.finish()


if __name__ == "__main__":
    upload_dataset("checkpoints/med_700.pt")
    # main()
