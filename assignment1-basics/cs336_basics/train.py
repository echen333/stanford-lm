from cs336_basics.utils_train import ( get_lr_cosine_schedule,
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
import time


def upload_dataset(path: str):
    run = wandb.init(entity="eddys", project="stanford-lm-1", job_type="add_dataset")
    artifact = wandb.Artifact(name=f"tinystories-valid", type="dataset")
    artifact.add_file(local_path=path, name="stories-valid")
    artifact.save()


from torch.utils.data import Dataset

def evaluate_model(model: Transformer, ds: Dataset, batch_size, num_samples=None):
    model.eval()

    total_loss = 0
    num_items = 0
    
    if num_samples is None:
        start_idxs = torch.arange(0, len(ds) - model.context_length, model.context_length)
        for i in range(0, len(start_idxs), batch_size):
            batch_idxs = start_idxs[i: i+batch_size] # TODO: fix for ending if not divisible
            all_idxs = batch_idxs.reshape(-1, 1) + torch.arange(0, model.context_length)
            all_idxs.to(model.device)
            x = torch.tensor(ds[all_idxs.reshape(-1, 1)].reshape(batch_size, -1), device=model.device, dtype=torch.long)
            y = torch.tensor(ds[all_idxs.reshape(-1, 1) + 1].reshape(batch_size, -1), device=model.device, dtype=torch.long)

    else:
        for _ in range(num_samples):
            x, y = get_batch(ds, batch_size, model.context_length, model.device)

    out = model(x)
    out = out.flatten(0, 1)  # of shape B C V -> (B*C) V
    y = y.flatten(0, 1)
    loss = cross_entropy(out, y)
    total_loss += loss.item()
    num_items += 1

    
    model.train()
    return total_loss / num_items

@hydra.main(config_path="conf", config_name="config", version_base=None)
# @hydra.main(config_path="conf", config_name="config_small", version_base=None)
def main(cfg):
    wandb_cfg = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    print(cfg, type(cfg))
    run = wandb.init(entity="eddys", project="stanford-lm-1", config=wandb_cfg)
    model: Transformer = instantiate(cfg.model)
    optim: AdamW = instantiate(cfg.optimizer, model.parameters())
    start_time = time.time()

    print("model", model)
    print("optim", optim)

    artifact = run.use_artifact("eddys/stanford-lm-1/tinystories-train:v0", type="dataset")
    artifact2 = run.use_artifact("eddys/stanford-lm-1/tinystories-valid:v0", type="dataset")
    artifact_dir = artifact.download()
    artifact_dir2 = artifact2.download()

    # dataset = np.load(cfg.data_path, mmap_mode="r")
    train_ds = np.load(f"{artifact_dir}/stories-train", mmap_mode="r")
    validation_ds = np.load(f"{artifact_dir2}/stories-valid", mmap_mode="r")
    for step in range(1, cfg.max_steps + 1):
        x, y = get_batch(train_ds, cfg.batch_size, model.context_length, cfg.model.device)
        out: Tensor = model(x)

        out = out.flatten(0, 1)  # of shape B C V -> (B*C) V
        y = y.flatten(0, 1)

        loss = cross_entropy(out, y)
        optim.zero_grad()
        loss.backward()
        optim.step()
        step_stats = {"loss": loss.item(), "time": time.time() - start_time}
        if step % cfg.val_interval == 0:
            valid_loss = evaluate_model(model, validation_ds, cfg.batch_size, 100)
            if step == (cfg.max_steps // cfg.val_interval) * cfg.val_interval:
                print(f"Running final validation set eval at step {step}.")
                valid_loss = evaluate_model(model, validation_ds, cfg.batch_size)
            step_stats["valid_loss"] = valid_loss
            # print(f"valid took {time.time() - start_time}s with len {len(validation_ds)}")
        
        run.log(step_stats)

        if step % cfg.checkpoint_steps == 0:
            file_path = f"{cfg.checkpoint_path}_{step}.pt"
            folder = os.path.dirname(file_path)
            os.makedirs(folder, exist_ok=True)
            print(f"Saving checkpoint at step {step} to path {file_path}")
            save_checkpoint(model, optim, step, file_path)
            wandb.save(file_path)

    run.finish()


if __name__ == "__main__":
    # upload_dataset("data/TinyStoriesV2-GPT4-valid.npy")
    main()
