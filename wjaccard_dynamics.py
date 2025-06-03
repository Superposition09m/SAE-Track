import argparse
import os
import torch
import torch.nn.functional as F
from datasets import load_dataset
from sae_training.sae_group import SAEGroup
from sae_vis.model_fns import TransformerLensWrapper
from transformer_lens import HookedTransformer, utils
from feat_fns import get_feature_activations, get_features_activations
from matplotlib import pyplot as plt
from matplotlib import rcParams
import numpy as np
from tqdm import tqdm

# Centralized Plot Configuration
PLOT_CONFIG = {
    'font.size':46,           # Default font size
    'axes.titlesize': 46,      # Font size for subplot titles
    'axes.labelsize': 46,      # Font size for x and y labels
    'legend.fontsize': 35,     # Font size for legends
    'xtick.labelsize': 44,     # Font size for x-axis tick labels
    'ytick.labelsize': 44,     # Font size for y-axis tick labels
    'figure.figsize': (30, 10), # Adjusted figure size for better layout
}
rcParams.update(PLOT_CONFIG)
LINE_WIDTH = 5  # Line thickness for plots

# Parameters
parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default="cuda", help="Device to run on (e.g., 'cuda' or 'cpu')")
parser.add_argument("--model_type", type=str, required=True, help="Model type to use")
# parser.add_argument("--feature_indices", type=int, nargs='+', required=True, help="List of feature indices to visualize")

parser.add_argument("--top_k", type=int, default=25, help="Number of top-k features to visualize")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size for processing")
parser.add_argument("--token_num", type=int, default=8192*4, help="Number of tokens to process")

args = parser.parse_args()
device = args.device
model_type = args.model_type

feature_indices = [59,23,24,42,102,173, 617, 694]
feature_names = ["related to programming environment", "Sea", "suspension", "}}", "Immun", "related to speed up","charge/es/ing","insert(s)/ed/ing"]
top_k = args.top_k
token_num = args.token_num

# Load model and checkpoints based on model_type
if model_type == "stanford-gpt2-small-a":
    #TEST
    # ckpt_range = [0,4,16,200,608]
    ckpt_range = list(range(0, 39, 1)) + list(range(38, 609, 10))
    hkpt = "blocks.6.hook_resid_pre"
    ckpt_list = (
        list(range(0, 100, 10))
        + list(range(100, 2000, 50))
        + list(range(2000, 20000, 100))
        + list(range(20000, 400000 + 1, 1000))
    )
    data = load_dataset("stas/openwebtext-10k", split="train")
elif model_type == "pythia-70m-deduped":
    ckpt_range = range(154)
    hkpt = "blocks.2.hook_resid_post"
    ckpt_list = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512] + list(
        range(1000, 143000 + 1, 1000)
    )
    data = load_dataset("NeelNanda/pile-10k", split="train")
elif model_type == "pythia-410m-deduped":
    ckpt_range = list(range(12)) + list(range(13, 33, 2)) + list(range(33, 154, 20))
    # ckpt_range = [0,2,4,6,8,10,20,140]
    hkpt = "blocks.4.hook_resid_pre"
    ckpt_list = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512] + list(
        range(1000, 143000 + 1, 1000)
    )
    sae_dict_1 = {
        0: "checkpoints/921e0mgk/final_sae_group_pythia-410m-deduped_blocks.4.hook_resid_pre_65536.pt",
        1: "pipe/52lieq0l/final_sae_group_pythia-410m-deduped_blocks.4.hook_resid_pre_65536.pt",
        2: "pipe/18ekwo2k/final_sae_group_pythia-410m-deduped_blocks.4.hook_resid_pre_65536.pt",
        3: "pipe/epmdboe7/final_sae_group_pythia-410m-deduped_blocks.4.hook_resid_pre_65536.pt",
        4: "pipe/l2no0yut/final_sae_group_pythia-410m-deduped_blocks.4.hook_resid_pre_65536.pt",
        5: "pipe/e7shgo9p/final_sae_group_pythia-410m-deduped_blocks.4.hook_resid_pre_65536.pt",
        6: "pipe/ebs8r6qr/final_sae_group_pythia-410m-deduped_blocks.4.hook_resid_pre_65536.pt",
        7: "pipe/vx8xb2k3/final_sae_group_pythia-410m-deduped_blocks.4.hook_resid_pre_65536.pt",
        8: "pipe/v07opmp2/final_sae_group_pythia-410m-deduped_blocks.4.hook_resid_pre_65536.pt",
        9: "pipe/io568vlx/final_sae_group_pythia-410m-deduped_blocks.4.hook_resid_pre_65536.pt",
        10: "pipe/trbwvb9d/final_sae_group_pythia-410m-deduped_blocks.4.hook_resid_pre_65536.pt",
        11: "pipe/x9qb3koe/final_sae_group_pythia-410m-deduped_blocks.4.hook_resid_pre_65536.pt",
        13: "pipe/59zu9leh/final_sae_group_pythia-410m-deduped_blocks.4.hook_resid_pre_65536.pt",
        15: "pipe/hpt1cy9i/final_sae_group_pythia-410m-deduped_blocks.4.hook_resid_pre_65536.pt",
        17: "pipe/2i4w2133/final_sae_group_pythia-410m-deduped_blocks.4.hook_resid_pre_65536.pt",
        19: "pipe/bo3lc5so/final_sae_group_pythia-410m-deduped_blocks.4.hook_resid_pre_65536.pt",
        21: "pipe/fqohwi3s/final_sae_group_pythia-410m-deduped_blocks.4.hook_resid_pre_65536.pt",
        23: "pipe/490ovm5t/final_sae_group_pythia-410m-deduped_blocks.4.hook_resid_pre_65536.pt",
        25: "pipe/dvk5euee/final_sae_group_pythia-410m-deduped_blocks.4.hook_resid_pre_65536.pt",
        27: "pipe/0nvrwgo9/final_sae_group_pythia-410m-deduped_blocks.4.hook_resid_pre_65536.pt",
        29: "pipe/r4k7r6y6/final_sae_group_pythia-410m-deduped_blocks.4.hook_resid_pre_65536.pt",
        31: "pipe/ein8fps8/final_sae_group_pythia-410m-deduped_blocks.4.hook_resid_pre_65536.pt",
        33: "pipe/io4vov51/final_sae_group_pythia-410m-deduped_blocks.4.hook_resid_pre_65536.pt",
        53: "pipe/y4x2bx9y/final_sae_group_pythia-410m-deduped_blocks.4.hook_resid_pre_65536.pt",
        73: "pipe/ib1hrafl/final_sae_group_pythia-410m-deduped_blocks.4.hook_resid_pre_65536.pt",
        93: "pipe/3oih20d8/final_sae_group_pythia-410m-deduped_blocks.4.hook_resid_pre_65536.pt",
        113: "pipe/nh42ylwt/final_sae_group_pythia-410m-deduped_blocks.4.hook_resid_pre_65536.pt",
        133: "pipe/30737iqa/final_sae_group_pythia-410m-deduped_blocks.4.hook_resid_pre_65536.pt",
        153: "pipe/07u89qd0/final_sae_group_pythia-410m-deduped_blocks.4.hook_resid_pre_65536.pt"
    }
    data = load_dataset("NeelNanda/pile-10k", split="train")
else:
    raise ValueError(f"Invalid model type: {model_type}")

print(f"Processing checkpoints: {ckpt_range}")
print(f"model_type: {model_type}")

# Load and tokenize data
model = HookedTransformer.from_pretrained(model_type).to(device)
tokenized_data = utils.tokenize_and_concatenate(data, model.tokenizer, max_length=128)
tokenized_data = tokenized_data.shuffle(22)
all_tokens = tokenized_data["tokens"]

# Function to calculate average non-self similarity
def calculate_non_self_similarity(activations, top_k):
    # Extract activations using top_k indices
    act_vec = activations[:top_k]  # Shape: (top_k, feature_dim)

    # Compute pairwise min and max using broadcasting
    min_values = torch.min(act_vec.unsqueeze(1), act_vec.unsqueeze(0))  # Shape: (top_k, top_k, feature_dim)
    max_values = torch.max(act_vec.unsqueeze(1), act_vec.unsqueeze(0))  # Shape: (top_k, top_k, feature_dim)

    # Sum over the feature dimension
    min_sums = min_values.sum(dim=2)  # Shape: (top_k, top_k)
    max_sums = max_values.sum(dim=2)  # Shape: (top_k, top_k)

    # Compute Weighted Jaccard Similarity
    jaccard_matrix = min_sums / (max_sums + 1e-8)  # Add epsilon to avoid division by zero

    # Mask the diagonal (self-to-self similarities)
    mask = torch.eye(top_k, dtype=torch.bool, device=act_vec.device)  # Diagonal mask
    non_self_similarities = jaccard_matrix[~mask].view(top_k, -1)  # Exclude diagonal elements

    # Compute the mean of non-self similarities
    mean_similarity = non_self_similarities.mean().item()
    print(mean_similarity)
    return mean_similarity

# Gather random similarities across checkpoints
average_random_similarities = []
token_set_number = 100
token_indices = np.random.choice(all_tokens.shape[0], token_set_number, replace=False)
random_indices = torch.randint(0, all_tokens.shape[1], (token_set_number,))

# Process checkpoints
for ckpt_num in ckpt_range:
    print(f"Processing checkpoint: {ckpt_num}")

    model_df = HookedTransformer.from_pretrained(model_type, checkpoint_index=ckpt_num).to(device)
    model_df_wrapper = TransformerLensWrapper(model_df, hkpt)

    # Random similarities
    selected_tokens = all_tokens[token_indices].to(device)
    saes=SAEGroup.load_from_pretrained(sae_dict_1[ckpt_num])
    sae=saes.autoencoders[0].to(device)
    # acts = get_model_activations(model_df_wrapper, selected_tokens)
    acts = get_features_activations(model_wrapper=model_df_wrapper, tokens=selected_tokens, encoder=sae)
    act_vec = acts[torch.arange(token_set_number), random_indices]
    average_random_similarity = calculate_non_self_similarity(act_vec, token_set_number)
    average_random_similarities.append((ckpt_num, average_random_similarity))

# Process features
feature_similarities = {idx: [] for idx in feature_indices}
for feature_idx in feature_indices:
    topk_seqs_path = f"features_{model_type}/feature_{feature_idx}_topk_{top_k}_toknum_{token_num}_seqs.pt"
    topk_feats_path = f"features_{model_type}/feature_{feature_idx}_topk_{top_k}_toknum_{token_num}_feats.pt"

    if os.path.exists(topk_seqs_path) and os.path.exists(topk_feats_path):
        topk_seqs = torch.load(topk_seqs_path)
        topk_feats = torch.load(topk_feats_path)
    else:
        raise ValueError((f"No saved topk_seqs found for feature {feature_idx}. "
                          f"Please run generate_feat_seqs.py first."))

    flattened_feats = topk_feats.view(-1)
    topk_vals, flat_topk_indices = torch.topk(flattened_feats, top_k)
    topk_2d_indices = torch.stack([flat_topk_indices // 128, flat_topk_indices % 128], dim=1)

    for ckpt_num in ckpt_range:
        model_df = HookedTransformer.from_pretrained(model_type, checkpoint_index=ckpt_num).to(device)
        model_df_wrapper = TransformerLensWrapper(model_df, hkpt)
        saes = SAEGroup.load_from_pretrained(sae_dict_1[ckpt_num])
        sae = saes.autoencoders[0].to(device)
        acts = get_features_activations(model_wrapper=model_df_wrapper, tokens=topk_seqs, encoder=sae)
        act_vec = acts[topk_2d_indices[:, 0], topk_2d_indices[:, 1]]
        average_similarity = calculate_non_self_similarity(act_vec, top_k)
        feature_similarities[feature_idx].append((ckpt_num, average_similarity))

# Plotting
plt.figure(figsize=PLOT_CONFIG['figure.figsize'])

# Plot feature similarities (difference from baseline)
for i, (feature_idx, similarities) in enumerate(feature_similarities.items()):
    ckpts, avg_sims = zip(*similarities)
    baseline_similarities = [dict(average_random_similarities)[ckpt] for ckpt in ckpts]
    diff_sims = [sim - baseline for sim, baseline in zip(avg_sims, baseline_similarities)]
    x_values = [ckpt_list[ckpt] + 1 for ckpt in ckpts]
    plt.plot(x_values, diff_sims, linestyle='-', linewidth=LINE_WIDTH, label=f"{feature_names[i]}")

# Plot random baseline as a reference
ckpts, random_avg_sims = zip(*average_random_similarities)
x_values = [ckpt_list[ckpt] + 1 for ckpt in ckpts]
plt.plot(x_values, np.zeros_like(random_avg_sims), linestyle='--', linewidth=LINE_WIDTH, label="Baseline (Random Sampled)")

# Adjust plot settings
plt.xscale('log')
plt.xlabel('Training Steps (log scale)')
plt.ylabel('Progressive Measure')
plt.ylim(-0.2, 1)
# plt.title(f'{model_type}, Hook Point: {hkpt}')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.grid(False)
plt.tight_layout()


# Save and show
random_save_id = np.random.randint(1000)
plt.savefig(f"wjaccard_similarity_diff_{model_type}_{hkpt}_{random_save_id}.svg")
plt.show()
