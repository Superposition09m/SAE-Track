import argparse
import os
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, TensorDataset
from datasets import load_dataset
from sae_vis.model_fns import TransformerLensWrapper
from sae_vis import SaeVisConfig, SaeVisData
from transformer_lens import HookedTransformer, utils
from sae_training.utils import LMSparseAutoencoderSessionloader
from feat_fns import get_feature_activations, get_model_activations, select_topk_sequences, tokens_to_text_and_print
import torch.nn.functional as F
from matplotlib import pyplot as plt
import numpy as np

"""
Use ground truth SAE to get features. Get token strings for each feature.

e.g. for feature "number", it activates on < ....... 1......>, <.....2.....>, <.....3.....> and so on.
"""

# Parameters
parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default="cuda", help="Device to run on (e.g., 'cuda' or 'cpu')")
parser.add_argument("--model_type", type=str, required=True, help="Model type to use (e.g., 'stanford-gpt2-small-a' or 'pythia-70m-deduped')")

args = parser.parse_args()

device = args.device
model_type = args.model_type

# Hard-coded checkpoint range
if model_type == "stanford-gpt2-small-a":
    # 0-37 dense , 38-608 sparse(+10)
    ckpt_range = list(range(0, 39, 1)) + list(range(38, 609, 10))
    # ckpt_range=[0,2,4,14,200,608]
    hkpt = "blocks.5.hook_resid_pre"
    ckpt_list = (
            list(range(0, 100, 10))
            + list(range(100, 2000, 50))
            + list(range(2000, 20000, 100))
            + list(range(20000, 400000 + 1, 1000))
    )
    data = load_dataset("stas/openwebtext-10k", split="train")
elif model_type == "stanford-gpt2-medium-a":
    ckpt_range = list(range(0, 39, 1)) + list(range(38, 609, 20))
    hkpt = "blocks.6.hook_resid_pre"
    ckpt_list = (
            
            list(range(0, 100, 10))
            + list(range(100, 2000, 50))
            + list(range(2000, 20000, 100))
            + list(range(20000, 400000 + 1, 1000))
    )
    data = load_dataset("stas/openwebtext-10k", split="train")
    
            


elif model_type == "solu-12l-pile":
    ckpt_range = list(range(200))
    hkpt = "blocks.2.hook_resid_post"
    ckpt_list = ckpt_range
    data = load_dataset("NeelNanda/pile-10k", split="train")
elif model_type == "pythia-160m-deduped":
    ckpt_range = list(range(12)) + list(range(13, 33, 2)) + list(range(33, 154, 20))
    hkpt = "blocks.4.hook_resid_post"
    ckpt_list = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512] + list(
        range(1000, 143000 + 1, 1000)
    )
    data = load_dataset("NeelNanda/pile-10k", split="train")
elif model_type == "pythia-410m-deduped":
    ckpt_range = list(range(12)) + list(range(13, 33, 2)) + list(range(33, 154, 20))
    hkpt = "blocks.4.hook_resid_pre"
    ckpt_list = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512] + list(
        range(1000, 143000 + 1, 1000)
    )
    data = load_dataset("NeelNanda/pile-10k", split="train")
elif model_type == "pythia-1.4b-deduped":
    ckpt_range = list(range(12)) + list(range(13, 33, 2)) + list(range(33, 154, 20))
    hkpt = "blocks.3.hook_resid_pre"
    ckpt_list = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512] + list(
        range(1000, 143000 + 1, 1000)
    )
    data = load_dataset("NeelNanda/pile-10k", split="train")
elif model_type == "solu-12l-pile":
    ckpt_range = list(range(200))
    data = load_dataset("NeelNanda/pile-10k", split="train")
    hkpt = "blocks.2.hook_resid_post"
    ckpt_list = ckpt_range
else:
    raise ValueError(f"Invalid model type: {model_type}")

print(f"Processing checkpoints: {ckpt_range}")
print(f"model_type: {model_type}")

# Load the saved topk_seqs, if it exists
model = HookedTransformer.from_pretrained(model_type).to(device)
token_set_number = 100
tokenized_data = utils.tokenize_and_concatenate(data, model.tokenizer, max_length=128)
tokenized_data = tokenized_data.shuffle(22)
all_tokens = tokenized_data["tokens"]#[N,128]
# Loop over the hard-coded checkpoints
average_non_self_similarities = []
#randomly select token_set_number tokens [token_set_number,128]
token_indices = np.random.choice(all_tokens.shape[0], token_set_number, replace=False)
# Randomly select one index per sequence across the batch independently
random_indices = torch.randint(0, all_tokens.shape[1], (token_set_number,))


with torch.no_grad():
    for ckpt_num in ckpt_range:
        print(f"Processing checkpoint: {ckpt_num}")

        # Load the model for the specific checkpoint
        model_df = HookedTransformer.from_pretrained(model_type, checkpoint_index=ckpt_num).to(device)
        model_df_wrapper = TransformerLensWrapper(model_df, hkpt)

        
        selected_tokens = all_tokens[token_indices].to(device)

        # Get model activations for the top-k sequences
        acts = get_model_activations(model_df_wrapper, selected_tokens)


        # Use advanced indexing to select the activations at the randomly chosen indices
        act_vec = acts[torch.arange(token_set_number), random_indices]

        # Calculate the cosine similarity matrix
        cosine_similarity_matrix = F.cosine_similarity(
            act_vec.unsqueeze(1),  # Shape becomes [10, 1, 768]
            act_vec.unsqueeze(0),  # Shape becomes [1, 10, 768]
            dim=2  # Calculate cosine similarity across the 768 dimensions
        ).cpu().numpy()  # Convert to NumPy array for plotting

        # Mask the diagonal (self-to-self similarities)
        num_vectors = cosine_similarity_matrix.shape[0]
        mask = np.eye(num_vectors, dtype=bool)  # Create a mask for the diagonal (True for self-to-self)
        non_self_similarities = cosine_similarity_matrix[~mask]  # Extract non-self-to-self similarities

        # Compute the average of the non-self-to-self similarities for this checkpoint
        average_similarity = non_self_similarities.mean()
        average_non_self_similarities.append((ckpt_num, average_similarity))
        print(f"Average non-self-to-self similarity for checkpoint {ckpt_num}: {average_similarity}")

    # Print all average similarities for each checkpoint
for ckpt, avg_sim in average_non_self_similarities:
    print(f"Checkpoint {ckpt}: Average non-self-to-self similarity = {avg_sim}")

# After processing all checkpoints, draw the plot
# After processing all checkpoints, draw the plot

# Define the ckpt_list (logscale)
#Moved to the top branch

import matplotlib.pyplot as plt
import matplotlib as mpl

# Apply PLOT_CONFIG settings
PLOT_CONFIG = {
    'font.size': 46,           # Default font size
    'axes.titlesize': 46,      # Font size for subplot titles
    'axes.labelsize': 46,      # Font size for x and y labels
    'legend.fontsize': 35,     # Font size for legends
    'xtick.labelsize': 44,     # Font size for x-axis tick labels
    'ytick.labelsize': 44,     # Font size for y-axis tick labels
    'figure.figsize': (30, 10) # Adjusted figure size for better layout
}
mpl.rcParams.update(PLOT_CONFIG)

# Extract checkpoint numbers and corresponding average similarities
ckpts, avg_sims = zip(*average_non_self_similarities)

# Map the ckpt numbers to corresponding values in ckpt_list
x_values = [ckpt_list[ckpt] + 1 for ckpt in ckpts]

# Plot the results
plt.figure()
plt.plot(x_values, avg_sims, linestyle='-', color='red',linewidth=5)
plt.ylim(-0.2, 1)  # Set y-range
plt.xscale('log')  # Set x-axis to log scale
plt.xlabel('Training Steps (log scale)')
plt.ylabel('Similarity')
plt.title(f'model={model_type}, hkpt={hkpt}')
plt.grid(True)
plt.tight_layout()
plt.savefig(f"activations_non_self_similarity_{model_type}_{hkpt}.svg")
