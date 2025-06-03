import argparse
import os
import torch
import torch.nn.functional as F
from datasets import load_dataset
from sae_vis.model_fns import TransformerLensWrapper
from transformer_lens import HookedTransformer, utils
from feat_fns import get_model_activations
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm
import pickle
from matplotlib import rcParams
from cycler import cycler
import matplotlib as mpl

# For example, to use a larger set of distinct colors:
colors = plt.cm.get_cmap('tab20').colors  # 'tab20' has 20 distinct colors
mpl.rcParams['axes.prop_cycle'] = cycler(color=colors)

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
parser.add_argument("--top_k", type=int, default=25, help="Number of top-k features to visualize")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size for processing")
parser.add_argument("--token_num", type=int, default=8192 * 4, help="Number of tokens to process")
args = parser.parse_args()

device = args.device
model_type = args.model_type
top_k = args.top_k
token_num = args.token_num

# Define saving directory
save_dir = "act_data_vis_gpt2medium_3"
os.makedirs(save_dir, exist_ok=True)

feature_indices = [187,99,81,49,68,37,5]
feature_names = ["while","M/movement","up","Hel/hel","license/ed/ing","related to sports divisions","related to psychological, emotional, and personal states"]

# Load model and checkpoints based on model_type
if model_type == "stanford-gpt2-small-a":
    ckpt_range = list(range(0, 11, 1)) + list(range(11, 31, 5)) + list(range(31, 181, 10))+list(range(181, 608, 30))+[608]
    hkpt = "blocks.5.hook_resid_pre"
    ckpt_list = (
        list(range(0, 100, 10))
        + list(range(100, 2000, 50))
        + list(range(2000, 20000, 100))
        + list(range(20000, 400000 + 1, 1000))
    )
    data = load_dataset("NeelNanda/pile-10k", split="train")
elif model_type == "stanford-gpt2-medium-a":
    ckpt_range = list(range(0, 11, 1)) + list(range(11, 31, 5)) + list(range(31, 181, 10))+list(range(181, 608, 30))+[608]
    hkpt = "blocks.6.hook_resid_pre"
    ckpt_list = (
        list(range(0, 100, 10))
        + list(range(100, 2000, 50))
        + list(range(2000, 20000, 100))
        + list(range(20000, 400000 + 1, 1000))
    )
    data = load_dataset("NeelNanda/pile-10k", split="train")

elif model_type == "pythia-410m-deduped":
    ckpt_range = list(range(33)) + list(range(33, 154, 20))  # Intact ckpt_range logic
    hkpt = "blocks.4.hook_resid_pre"
    ckpt_list = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512] + list(
        range(1000, 143000 + 1, 1000)
    )  # Intact ckpt_list logic
    data = load_dataset("NeelNanda/pile-10k", split="train")
elif model_type == "pythia-160m-deduped":
    ckpt_range = list(range(12)) + list(range(13, 33, 2)) + list(range(33, 154, 20))
    hkpt = "blocks.4.hook_resid_pre"
    ckpt_list = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512] + list(
        range(1000, 143000 + 1, 1000)
    )
    data = load_dataset("NeelNanda/pile-10k", split="train")
elif model_type == "pythia-1.4b-deduped":
    # ckpt_range = list(range(12)) + list(range(13, 33, 2)) + list(range(33, 154, 20))
    ckpt_range = list(range(12)) + list(range(13, 33, 4)) + list(range(33, 154, 20))
    hkpt = "blocks.3.hook_resid_pre"
    ckpt_list = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512] + list(
        range(1000, 143000 + 1, 1000)
    )
    data = load_dataset("NeelNanda/pile-10k", split="train")
else:
    raise ValueError(f"Invalid model type: {model_type}")

print(f"Processing checkpoints: {ckpt_range}")
print(f"model_type: {model_type}")

# Load and tokenize data
model = HookedTransformer.from_pretrained(model_type).to(device)
tokenized_data = utils.tokenize_and_concatenate(data, model.tokenizer, max_length=128)
tokenized_data = tokenized_data.shuffle(42)
all_tokens = tokenized_data["tokens"]

# Function to calculate average non-self similarity
def calculate_non_self_similarity(activations, top_k):
    act_vec = activations[:top_k]
    cosine_similarity_matrix = F.cosine_similarity(
        act_vec.unsqueeze(1), act_vec.unsqueeze(0), dim=2
    ).cpu().numpy()
    mask = np.eye(len(act_vec), dtype=bool)
    non_self_similarities = cosine_similarity_matrix[~mask]
    return non_self_similarities.mean()

# Gather random similarities across checkpoints
average_random_similarities = []
token_set_number = 150
token_indices = np.random.choice(all_tokens.shape[0], token_set_number, replace=False)
random_indices = torch.randint(0, all_tokens.shape[1], (token_set_number,))

# Checkpoint-wise processing
for ckpt_num in tqdm(ckpt_range):
    random_save_path = os.path.join(save_dir, f"random_sim_ckpt_{ckpt_num}.pkl")
    if os.path.exists(random_save_path):
        with open(random_save_path, "rb") as f:
            average_random_similarity = pickle.load(f)
        print(f"Loaded random similarity for checkpoint {ckpt_num}")
    else:
        print(f"Processing checkpoint: {ckpt_num}")
        model_df = HookedTransformer.from_pretrained(model_type, checkpoint_index=ckpt_num).to(device)
        model_df_wrapper = TransformerLensWrapper(model_df, hkpt)
        selected_tokens = all_tokens[token_indices].to(device)
        acts = get_model_activations(model_df_wrapper, selected_tokens)
        act_vec = acts[torch.arange(token_set_number), random_indices]
        average_random_similarity = calculate_non_self_similarity(act_vec, token_set_number)
        with open(random_save_path, "wb") as f:
            pickle.dump(average_random_similarity, f)
    average_random_similarities.append((ckpt_num, average_random_similarity))

# Feature similarities
feature_similarities = {idx: [] for idx in feature_indices}
for feature_idx in tqdm(feature_indices):
    topk_seqs_path = f"features_{model_type}/feature_{feature_idx}_topk_{top_k}_toknum_{token_num}_seqs.pt"
    topk_feats_path = f"features_{model_type}/feature_{feature_idx}_topk_{top_k}_toknum_{token_num}_feats.pt"

    if os.path.exists(topk_seqs_path) and os.path.exists(topk_feats_path):
        topk_seqs = torch.load(topk_seqs_path)
        topk_feats = torch.load(topk_feats_path)
    else:
        raise ValueError(f"No saved topk_seqs found for feature {feature_idx}. Run generate_feat_seqs.py first.")

    flattened_feats = topk_feats.view(-1)
    topk_vals, flat_topk_indices = torch.topk(flattened_feats, top_k)
    topk_2d_indices = torch.stack([flat_topk_indices // 128, flat_topk_indices % 128], dim=1)

    for ckpt_num in ckpt_range:
        feature_save_path = os.path.join(save_dir, f"feature_{feature_idx}_ckpt_{ckpt_num}.pkl")
        if os.path.exists(feature_save_path):
            with open(feature_save_path, "rb") as f:
                average_similarity = pickle.load(f)
            print(f"Loaded feature {feature_idx} similarity for checkpoint {ckpt_num}")
        else:
            model_df = HookedTransformer.from_pretrained(model_type, checkpoint_index=ckpt_num).to(device)
            model_df_wrapper = TransformerLensWrapper(model_df, hkpt)
            acts = get_model_activations(model_df_wrapper, topk_seqs)
            act_vec = acts[topk_2d_indices[:, 0], topk_2d_indices[:, 1]]
            average_similarity = calculate_non_self_similarity(act_vec, top_k)
            with open(feature_save_path, "wb") as f:
                pickle.dump(average_similarity, f)
        feature_similarities[feature_idx].append((ckpt_num, average_similarity))

# Plotting
plt.figure(figsize=PLOT_CONFIG['figure.figsize'])

for i, (feature_idx, similarities) in enumerate(feature_similarities.items()):
    ckpts, avg_sims = zip(*similarities)
    baseline_similarities = [dict(average_random_similarities)[ckpt] for ckpt in ckpts]
    diff_sims = [sim - baseline for sim, baseline in zip(avg_sims, baseline_similarities)]
    x_values = [ckpt_list[ckpt] + 1 for ckpt in ckpts]  # Mapping checkpoints to x-coordinates
    plt.plot(x_values, diff_sims, linestyle='-', label=f"{feature_names[i]}", linewidth=LINE_WIDTH)

# Plot baseline
ckpts, random_avg_sims = zip(*average_random_similarities)
x_values = [ckpt_list[ckpt] + 1 for ckpt in ckpts]  # Mapping checkpoints to x-coordinates
plt.plot(x_values, np.zeros_like(random_avg_sims), linestyle='--', label="Baseline (Random Sampled)", linewidth=LINE_WIDTH)

# Adjust plot settings
plt.xscale('log')
plt.xlabel('Training Steps (log scale)')
plt.ylabel('Progressive Measure')
# plt.title(f'{model_type}, Hook Point: {hkpt}')
# plt.legend()
#legend to the out of the fig, right 
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout()

# Save and show
random_save_id = np.random.randint(1000)
plt.savefig(f"feature_similarity_diff_{model_type}_{hkpt}_{random_save_id}.svg", bbox_inches='tight')
# plt.show()
#Now ,save to pdf
# plt.savefig(f"feature_similarity_diff_{model_type}_{hkpt}_{random_save_id}.pdf", bbox_inches='tight', format='pdf')
