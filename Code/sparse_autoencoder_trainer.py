import argparse

import torch
import os
import sys

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB__SERVICE_WAIT"] = "300"

from sae_training.config import LanguageModelSAERunnerConfig
from sae_training.lm_runner import language_model_sae_runner

# set args parser
parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="pythia-410m-deduped")
parser.add_argument("--ckpt", type=int, default=None)
parser.add_argument("--hook_point_layer", type=int, default=4)
parser.add_argument("--gpu", type=int, default=1)
parser.add_argument("--expansion_factor", type=int, default=64)
parser.add_argument("--l1", type=float, default=0.00008)
parser.add_argument("--lr", type=float, default=0.0004)
parser.add_argument("--from_pretrained", type=str, default=None)
parser.add_argument("--total_training_tokens", type=int, default=1_000_000 * 300)
parser.add_argument("--n_checkpoints", type=int, default=10)
parser.add_argument("--lr_warm_up_steps", type=int, default=5000)
parser.add_argument("--feature_sampling_window", type=int, default=1000)# feature_sampling_window = 1000,
parser.add_argument("--dead_feature_window", type=int, default=5000)# dead_feature_window = 5000,
parser.add_argument("--wandb_log_frequency", type=int, default=100)# wandb_log_frequency=100,
parser.add_argument("--reg", type=float, default=0)
parser.add_argument("--project", type=str, default="pythia_410_multilayer")
parser.add_argument("--d_in", type=int, default=1024)
parser.add_argument("--dataset_path", type=str, default="EleutherAI/the_pile_deduplicated")
parser.add_argument("--w_dec_norm", type=int, default=1)#default is normalized. but without normalization it is improved
args = parser.parse_args()
print(args)
print(args.from_pretrained)
#set cuda visible devices
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
cfg = LanguageModelSAERunnerConfig(

    # Data Generating Function (Model + Training Distibuion)
    model_name=args.model_name,
    hook_point=f"blocks.{args.hook_point_layer}.hook_resid_pre",
    hook_point_layer=args.hook_point_layer,
    d_in=args.d_in,
    from_pretrained_path=args.from_pretrained,
    dataset_path=args.dataset_path,
    is_dataset_tokenized=False,
    ckpt=args.ckpt,

    # SAE Parameters
    expansion_factor=args.expansion_factor,
    b_dec_init_method="geometric_median",
    w_dec_norm=args.w_dec_norm,
    # Training Parameters
    lr=args.lr,
    l1_coefficient=args.l1,
    lr_scheduler_name="constantwithwarmup",
    train_batch_size=4096,
    context_size=128,
    lr_warm_up_steps=args.lr_warm_up_steps,

    # Activation Store Parameters
    n_batches_in_buffer=128,
    total_training_tokens=args.total_training_tokens,
    store_batch_size=32,

    # Dead Neurons and Sparsity
    use_ghost_grads=True,
    feature_sampling_window=args.feature_sampling_window,
    dead_feature_window=args.dead_feature_window,
    dead_feature_threshold=1e-6,

    # WANDB
    log_to_wandb=True,
    wandb_project=args.project,
    wandb_entity=None,
    wandb_log_frequency=args.wandb_log_frequency,

    # Misc
    device="cuda",
    seed=42,
    n_checkpoints=args.n_checkpoints,
    checkpoint_path="checkpoints",
    dtype=torch.float32,

    #reg
    reg_coefficient=args.reg
)

sparse_autoencoder, savepath = language_model_sae_runner(cfg)
print(f"SAE model saved at {savepath}")