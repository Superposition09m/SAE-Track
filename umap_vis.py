import argparse
import os
import time
import pickle

import torch
import numpy as np
import umap
from matplotlib import pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.gridspec import GridSpec
from tqdm import tqdm
from sae_vis.model_fns import TransformerLensWrapper
from transformer_lens import HookedTransformer
from feat_fns import get_model_activations
from datasets import load_dataset
from matplotlib import rcParams

# Global font size configuration
rcParams.update({
    'font.size': 38,          # Default font size
    'axes.titlesize': 50,     # Font size for subplot titles
    'axes.labelsize': 38,     # Font size for x and y labels
    'legend.fontsize': 38,    # Font size for legends
    'xtick.labelsize': 38,    # Font size for x-axis tick labels
    'ytick.labelsize': 38,    # Font size for y-axis tick labels
})

# Parameters
parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default="cuda", help="Device to run on (e.g., 'cuda' or 'cpu')")
# parser.add_argument("--model_type", type=str, default="pythia-410m-deduped", help="Model type to use")
parser.add_argument("--model_type", type=str, default="stanford-gpt2-medium-a", help="Model type to use")
parser.add_argument("--top_k", type=int, default=25, help="Number of top-k activations to process")
parser.add_argument("--token_num", type=int, default=8192 * 4, help="Number of tokens to process")
parser.add_argument("--plot_width", type=int, default=58, help="Width of the plot in inches")
parser.add_argument("--plot_height", type=int, default=28, help="Height of the plot in inches")
args = parser.parse_args()

device = args.device
model_type = args.model_type
top_k = args.top_k
token_num = args.token_num
plot_width = args.plot_width
plot_height = args.plot_height

features = [
    (187, "while", "o"),
    (99, "M/movement", "o"),
    (81, "up", "o"),
    (49, "Hel/hel", "D"),
    (68, "license/ed/ing", "D"),
    (37, "related to sports divisions", "D"),
    (5, "related to psychological, emotional, and personal states", "D"),
]
# Colormap for features
colormap = get_cmap("tab20")
feature_colors = {feature_name: colormap(i / len(features)) for i, (_, feature_name, _) in enumerate(features)}

# Create directory to save UMAP embeddings and labels
data_save_dir = "umap_saved_data_stfgpt2m"
os.makedirs(data_save_dir, exist_ok=True)

# Load model and checkpoints
if model_type == "pythia-70m-deduped":
    ckpt_range = range(154)
    hkpt = "blocks.2.hook_resid_post"
    data = load_dataset("NeelNanda/pile-10k", split="train")
elif model_type == "pythia-410m-deduped":
    ckpt_range = [0, 5, 10, 15, 60, 153]
    hkpt = "blocks.4.hook_resid_pre"
    data = load_dataset("NeelNanda/pile-10k", split="train")
elif model_type == "pythia-160m-deduped":
    ckpt_range = [0, 5, 10, 15, 60, 153]
    hkpt = "blocks.4.hook_resid_pre"
    data = load_dataset("NeelNanda/pile-10k", split="train")
elif model_type == "pythia-1.4b-deduped":
    ckpt_range = [0, 5, 10, 15, 60, 153]
    hkpt = "blocks.3.hook_resid_pre"
    data = load_dataset("NeelNanda/pile-10k", split="train")
elif model_type == "stanford-gpt2-small-a":
    ckpt_range = [0, 5, 10, 15, 60, 608]
    print(ckpt_range)
    hkpt = "blocks.5.hook_resid_pre"
    data = load_dataset("stas/openwebtext-10k", split="train")
elif model_type == "stanford-gpt2-medium-a":
    ckpt_range = [0, 5, 10, 15, 60, 608]
    hkpt = "blocks.6.hook_resid_pre"
    data = load_dataset("stas/openwebtext-10k", split="train")
else:
    raise ValueError(f"Invalid model type: {model_type}")

print(f"Processing checkpoints: {ckpt_range}")
print(f"model_type: {model_type}")

# Create the main figure using GridSpec
fig = plt.figure(figsize=(plot_width, plot_height))  # Adjustable width and height via arguments
gs = GridSpec(2, 4, figure=fig, width_ratios=[1, 1, 1, 0.9])  # Allocate last column for legend

# Process each checkpoint and add subplots
axes = []
print(ckpt_range)
for idx, ckpt_num in enumerate(tqdm(ckpt_range, desc="Processing Checkpoints")):
    print(f"Processing checkpoint: {ckpt_num}")

    # File paths for saved embeddings and labels
    embedding_file = os.path.join(data_save_dir, f"checkpoint_{ckpt_num}_embeddings.pkl")
    labels_file = os.path.join(data_save_dir, f"checkpoint_{ckpt_num}_labels.pkl")

    if os.path.exists(embedding_file) and os.path.exists(labels_file):
        print(f"Loading saved data for checkpoint {ckpt_num}...")
        with open(embedding_file, "rb") as f:
            umap_embeddings = pickle.load(f)
        with open(labels_file, "rb") as f:
            labels = pickle.load(f)
    else:
        print(f"Generating data for checkpoint {ckpt_num}...")

        # Load the model checkpoint
        model_df = HookedTransformer.from_pretrained(model_type, checkpoint_index=ckpt_num).to(device)
        model_df_wrapper = TransformerLensWrapper(model_df, hkpt)

        # Storage for activations and labels
        combined_acts = []
        labels = []

        for feature_index, feature_name, _ in features:
            # Load pre-saved feature sequences and activations
            topk_seqs_path = f"features_{model_type}/feature_{feature_index}_topk_{top_k}_toknum_{token_num}_seqs.pt"
            topk_feats_path = f"features_{model_type}/feature_{feature_index}_topk_{top_k}_toknum_{token_num}_feats.pt"

            if not os.path.exists(topk_seqs_path) or not os.path.exists(topk_feats_path):
                raise ValueError((f"No saved topk_seqs found for feature {feature_index}. "
                                  f"Please run generate_feat_seqs.py first."))

            print(f"Loading pre-saved sequences and activations for Feature {feature_index} ({feature_name})...")
            topk_seqs = torch.load(topk_seqs_path)
            topk_feats = torch.load(topk_feats_path)

            # Extract top-k activations
            flattened_feats = topk_feats.view(-1)
            topk_vals, flat_topk_indices = torch.topk(flattened_feats, top_k)
            topk_2d_indices = torch.stack([flat_topk_indices // 128, flat_topk_indices % 128], dim=1)

            # Get top-k activations
            acts = get_model_activations(model_df_wrapper, topk_seqs)
            act_vec = acts[topk_2d_indices[:, 0], topk_2d_indices[:, 1]].cpu().numpy()

            # Append activations and labels
            combined_acts.append(act_vec)
            labels.extend([feature_name] * top_k)

        # Combine all activations
        combined_acts = np.vstack(combined_acts)
        labels = np.array(labels)

        # Apply UMAP
        print("Applying UMAP...")
        umap_model = umap.UMAP(n_neighbors=100, min_dist=0.0, metric="cosine", random_state=42)
        umap_embeddings = umap_model.fit_transform(combined_acts)

        # Save embeddings and labels to files
        with open(embedding_file, "wb") as f:
            pickle.dump(umap_embeddings, f)
        with open(labels_file, "wb") as f:
            pickle.dump(labels, f)

    # Add subplot for current checkpoint
    row, col = divmod(idx, 3)  # Compute row and column for 2x3 layout
    ax = fig.add_subplot(gs[row, col])  # Assign subplot to GridSpec
    for feature_index, feature_name, feature_marker in features:
        idxs = labels == feature_name
        ax.scatter(
            umap_embeddings[idxs, 0],
            umap_embeddings[idxs, 1],
            label=feature_name,
            alpha=0.7,
            s=400,
            color=feature_colors[feature_name],
            marker=feature_marker,  # Apply the manually chosen marker style
        )
    ax.set_title(f"Checkpoint {ckpt_num}", fontweight="bold")
    ax.set_xlabel("UMAP Dimension 1")
    ax.set_ylabel("UMAP Dimension 2")
    axes.append(ax)

# Add a legend in the last column of the GridSpec
legend_ax = fig.add_subplot(gs[:, 3])  # Use the entire last column for the legend
legend_ax.axis("off")  # Turn off axis for the legend
handles, labels = axes[0].get_legend_handles_labels()
legend_ax.legend(
    handles,
    labels,
    loc="center",
    title="Features",
    fontsize=rcParams['legend.fontsize'],  # Use global font size for the legend
    frameon=True
)

# Save the plot
random_number = int(time.time())
# Save the plot as a PDF
plt.savefig(f"umap_multi_ckpt_features_{model_type}_{random_number}.pdf", bbox_inches="tight", format="pdf")
print(f"Saved combined plot for all checkpoints as PDF.")
# save a png version
# plt.savefig(f"umap_multi_ckpt_features_{model_type}_{random_number}.png", bbox_inches="tight", format="png")