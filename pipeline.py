import argparse
import time

import torch
import os
import sys
import json

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB__SERVICE_WAIT"] = "300"

from sae_training.config import LanguageModelSAERunnerConfig
from sae_training.lm_runner import language_model_sae_runner


# Function to run the SAE training pipeline
def run_pipeline(args, next_ckpt, prev_savepath):
    cfg = LanguageModelSAERunnerConfig(
        model_name=args.model_name,
        ckpt=next_ckpt,
        hook_point=f"blocks.{args.hook_point_layer}.hook_resid_pre",
        hook_point_layer=args.hook_point_layer,
        d_in=args.d_in,
        from_pretrained_path=prev_savepath,
        dataset_path=args.dataset_path,
        is_dataset_tokenized=False,
        expansion_factor=args.expansion_factor,
        b_dec_init_method="geometric_median",
        lr=args.lr,
        l1_coefficient=args.l1,
        lr_scheduler_name="constantwithwarmup",
        train_batch_size=4096,
        context_size=128,
        lr_warm_up_steps=args.lr_warm_up_steps,
        n_batches_in_buffer=128,
        total_training_tokens=args.total_training_tokens,
        store_batch_size=32,
        use_ghost_grads=True,
        feature_sampling_window=args.feature_sampling_window,
        dead_feature_window=args.dead_feature_window,
        dead_feature_threshold=1e-6,
        log_to_wandb=True,
        wandb_project=args.project,
        wandb_entity=None,
        wandb_log_frequency=args.wandb_log_frequency,
        device="cuda",
        seed=42,
        n_checkpoints=args.n_checkpoints,
        checkpoint_path="pipe",
        dtype=torch.float32,
        reg_coefficient=args.reg,
        w_dec_norm=args.w_dec_norm
    )

    # Run the training process and return the path where the model is saved
    sparse_autoencoder, new_savepath = language_model_sae_runner(cfg)

    print(f"SAE model saved at {new_savepath}")
    return f"{new_savepath}/final_{sparse_autoencoder.get_name()}.pt"


# Set up argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="pythia-410m-deduped")
parser.add_argument("--ckpt", type=int, default=None)
parser.add_argument("--hook_point_layer", type=int, default=4)
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--expansion_factor", type=int, default=64)
parser.add_argument("--l1", type=float, default=0.00008)
parser.add_argument("--lr", type=float, default=0.00004)#smaller for tuning
parser.add_argument("--total_training_tokens", type=int, default=1_000_000 * 15)
parser.add_argument("--n_checkpoints", type=int, default=3)
parser.add_argument("--lr_warm_up_steps", type=int, default=3)
parser.add_argument("--feature_sampling_window", type=int, default=250)
parser.add_argument("--dead_feature_window", type=int, default=100000000000)
parser.add_argument("--wandb_log_frequency", type=int, default=100)
parser.add_argument("--reg", type=float, default=0)
parser.add_argument("--num_iterations", type=int, default=6)  # Argument for number of iterations
parser.add_argument("--from_pretrained", type=str,
                    default="checkpoints/h6nzltd7/final_sae_group_pythia-410m-deduped_blocks.4.hook_resid_pre_65536.pt")  # Initial pretrained path
# parser.add_argument("--output_map_file", type=str, default="model_save_map.json")  # File to save the map
# parser.add_argument("--project", type=str, default="pipe-sae-backward")
parser.add_argument("--delta_suffix", type=int, default=20)  # Delta suffix for model name
parser.add_argument("--project", type=str, default="pythia-410m-deduped-tracking")
parser.add_argument("--d_in", type=int, default=1024)
parser.add_argument("--dataset_path", type=str, default="EleutherAI/the_pile_deduplicated")
parser.add_argument("--w_dec_norm", type=int, default=1)
args = parser.parse_args()

# Set CUDA visible devices
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

# Initial savepath (could be None or a specified path)
savepath = args.from_pretrained

ckpt = args.ckpt

# Dictionary to map model names to save paths
model_save_map = {}

output_map_file = "model_save_map" + time.strftime("%Y%m%d-%H%M%S") + ".json"
# Loop for multiple training iterations
for i in range(args.num_iterations):
    # Decrement the numeric suffix by 10 for the next iteration|||| range(154)=0,1,2,...,153
    if ckpt is None:
        ckpt_next = 153 - (i + 1) * args.delta_suffix
    else:
        ckpt_next = ckpt - (i + 1) * args.delta_suffix

    # Train the model and get the savepath
    savepath = run_pipeline(args, ckpt_next, savepath)

    # Save the model name and corresponding save path in the map
    model_save_map[ckpt_next] = savepath

    # Save the model_save_map to a JSON file after every iteration
    with open(output_map_file, 'w') as f:
        json.dump(model_save_map, f, indent=4)

    print(f"Model save map updated and saved to {output_map_file}")

print("Training completed. Final model save map has been saved.")
