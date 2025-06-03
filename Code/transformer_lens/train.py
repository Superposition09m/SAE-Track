import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import wandb
import datetime
from dataclasses import dataclass
from typing import Optional, List
import os
from transformer_lens.HookedTransformer import HookedTransformer


@dataclass
class HookedTransformerTrainConfig:
    total_steps: int
    batch_size: int
    lr: float
    min_lr: float
    seed: int = 0
    momentum: float = 0.0
    max_grad_norm: float = 1.0  # Gradient clipping
    weight_decay: float = 0.1
    optimizer_name: str = "Adam"
    optimizer_params: dict = None
    scheduler_name: str = "cosine"
    warmup_steps: int = 50
    device: str = "cuda"
    save_every: int = 250
    extra_save_iters: List[int] = None  # Specific iterations for extra saves
    eval_interval: int = 500  # Interval for evaluation steps
    eval_iter_num: int = 10  # Number of batches for evaluation
    save_dir: Optional[str] = None
    wandb: bool = False
    wandb_project_name: Optional[str] = None
    train_log_interval: int = 100


def train(model: HookedTransformer, config: HookedTransformerTrainConfig, dataset):
    torch.manual_seed(config.seed)
    model.train()

    # DataLoader for streaming dataset
    loader = DataLoader(dataset, batch_size=config.batch_size)
    optimizer_params = config.optimizer_params or {"lr": config.lr, "betas": (0.9, 0.95), "eps": 1.0e-8}
    optimizer = torch.optim.AdamW(model.parameters(), **optimizer_params, weight_decay=config.weight_decay)

    def lr_lambda(step: int):
        if step < config.warmup_steps:
            return step / config.warmup_steps
        elif config.scheduler_name == "cosine":
            cosine_input = (step - config.warmup_steps) / (config.total_steps - config.warmup_steps)
            cosine_lr = 0.5 * (1 + torch.cos(torch.tensor(torch.pi) * cosine_input))
            return max(cosine_lr, config.min_lr / config.lr)
        return max(1.0 - (step - config.warmup_steps) / (config.total_steps - config.warmup_steps), config.min_lr / config.lr)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    train_losses, test_losses, steps = [], [], []
    step = 0

    if config.wandb:
        run = wandb.init(project=config.wandb_project_name, config=vars(config))
        wandb_run_name = run.name
    else:
        wandb_run_name = f"lr_{config.lr}_steps_{config.total_steps}_bs_{config.batch_size}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

    save_dir = f"{config.save_dir}/{wandb_run_name}"
    os.makedirs(save_dir, exist_ok=True)

    extra_save_iters = set(config.extra_save_iters or [])
    with tqdm(total=config.total_steps, desc="Training Progress", unit="step") as pbar:
        while step < config.total_steps:
            for idx, tokens in loader:
                tokens = tokens.to(config.device)

                # Perform evaluation at intervals
                if step % config.eval_interval == 0 and step > 0:
                    model.eval()
                    test_loss = 0.0
                    with torch.no_grad():
                        eval_steps = 0
                        for eval_idx, eval_tokens in loader:
                            if eval_steps >= config.eval_iter_num:
                                break
                            eval_tokens = eval_tokens.to(config.device)
                            test_loss += model(eval_tokens, return_type="loss").item()
                            eval_steps += 1
                    test_loss /= config.eval_iter_num
                    test_losses.append(test_loss)

                    # Log evaluation immediately
                    if config.wandb:
                        wandb.log({"test_loss": test_loss, "step": step})

                    print(f"Step {step}: Evaluation Loss: {test_loss:.4f}")

                else:  # Perform training
                    model.train()
                    loss = model(tokens, return_type="loss")
                    loss.backward()

                    # Gradient clipping
                    if config.max_grad_norm:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)

                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                    train_losses.append(loss.item())

                    # Log training loss at intervals
                    if step % config.train_log_interval == 0:
                        if config.wandb:
                            wandb.log({"train_loss": loss.item(), "step": step})

                steps.append(step)
                pbar.update(1)
                pbar.set_postfix({"Train Loss": loss.item(), "Step": step, "LR": scheduler.get_last_lr()[0]})

                # Save model periodically and for extra iterations
                if step > 0 and (step % config.save_every == 0 or step in extra_save_iters):
                    checkpoint_path = os.path.join(save_dir, f"checkpoint_step_{step}.pt")
                    model.save(checkpoint_path)

                step += 1
                if step >= config.total_steps:
                    break

    # Save final model
    final_save_dir = os.path.join(save_dir, "final_model")
    model.save(final_save_dir)
    return model
