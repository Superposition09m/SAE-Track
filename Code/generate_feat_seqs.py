from sae_lens.sae import SAE
import argparse
import os
from sae_training.sae_group import SAEGroup
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, TensorDataset
from datasets import load_dataset
from sae_vis.model_fns import TransformerLensWrapper
from sae_vis import SaeVisConfig, SaeVisData
from transformer_lens import HookedTransformer, utils
from sae_training.utils import LMSparseAutoencoderSessionloader
from feat_fns import get_feature_activations, get_model_activations, select_topk_sequences, tokens_to_text_and_print
import argparse

# Parameters
parser = argparse.ArgumentParser()
parser.add_argument("--feature_idx", type=int, required=True, help="Feature index to visualize")
parser.add_argument("--model_type", type=str, default="stanford-gpt2-medium-a", help="Model type to use (e.g., 'stanford-gpt2-small-a' or 'pythia-70m-deduped')")
args = parser.parse_args()

# Parameters  66 85 95 129 248 242 192 168 117
feature_idx = args.feature_idx
top_k = 25
batch_size = 256
device = "cuda"
token_num=8192*4
model_type=args.model_type
#print Parameters
print(f"feature_idx: {feature_idx}")
print(f"top_k: {top_k}")
print(f"batch_size: {batch_size}")
print(f"device: {device}")
print(f"token_num: {token_num}")
#load the saved topk_seqs, if it exists
if os.path.exists(f"features_{model_type}/feature_{feature_idx}_topk_{top_k}_toknum_{token_num}_seqs.pt") and os.path.exists(f"features_{model_type}/feature_{feature_idx}_topk_{top_k}_toknum_{token_num}_feats.pt"):
    model = HookedTransformer.from_pretrained(model_type).to(device)
    topk_seqs = torch.load(f"features_{model_type}/feature_{feature_idx}_topk_{top_k}_toknum_{token_num}_seqs.pt")
    topk_feats = torch.load(f"features_{model_type}/feature_{feature_idx}_topk_{top_k}_toknum_{token_num}_feats.pt")
else:
    # get the seqs
    # Load our SAE model
    if model_type == "stanford-gpt2-small-a":
        _, sparse_autoencoder, _ = LMSparseAutoencoderSessionloader.load_session_from_pretrained(
            "checkpoints/f74zy27r/final_sae_group_stanford-gpt2-small-a_blocks.5.hook_resid_pre_49152.pt"
        )
        hook_point = f"blocks.5.hook_resid_pre"
        sae_ground_truth = sparse_autoencoder.autoencoders[0].to(device)
        data = load_dataset("stas/openwebtext-10k", split="train")
    elif model_type == "stanford-gpt2-medium-a":
        _, sparse_autoencoder, _ = LMSparseAutoencoderSessionloader.load_session_from_pretrained(
            "pipe/lkdiza4e/final_sae_group_stanford-gpt2-medium-a_blocks.6.hook_resid_pre_65536.pt"
        )
        hook_point = f"blocks.6.hook_resid_pre"
        sae_ground_truth = sparse_autoencoder.autoencoders[0].to(device)
        data = load_dataset("stas/openwebtext-10k", split="train")
    elif model_type == "pythia-70m-deduped":
        sae_ground_truth, _, _ = SAE.from_pretrained(
            release="pythia-70m-deduped-res-sm",  # see other options in sae_lens/pretrained_saes.yaml
            sae_id="blocks.2.hook_resid_post",  # won't always be a hook point
        )
        sae_ground_truth = sae_ground_truth.to(device)
        hook_point = "blocks.2.hook_resid_post"
        data = load_dataset("NeelNanda/pile-10k", split="train")
    elif model_type == "pythia-410m-deduped":
        _, sparse_autoencoder, _ = LMSparseAutoencoderSessionloader.load_session_from_pretrained(
            "pipe/07u89qd0/final_sae_group_pythia-410m-deduped_blocks.4.hook_resid_pre_65536.pt"
        )
        hook_point = f"blocks.4.hook_resid_pre"
        sae_ground_truth = sparse_autoencoder.autoencoders[0].to(device)
        data = load_dataset("NeelNanda/pile-10k", split="train")
    elif model_type == "pythia-160m-deduped":
        sparse_autoencoders= SAEGroup.load_from_pretrained("pipe/lig6rc8y/final_sae_group_pythia-160m-deduped_blocks.4.hook_resid_pre_49152.pt")
        sae_ground_truth = sparse_autoencoders.autoencoders[0].to(device)
        hook_point = "blocks.4.hook_resid_pre"
        data = load_dataset("NeelNanda/pile-10k", split="train")
    elif model_type == "pythia-1.4b-deduped":
        sparse_autoencoders= SAEGroup.load_from_pretrained("pipe/b19kk6hj/final_sae_group_pythia-1.4b-deduped_blocks.3.hook_resid_pre_131072.pt")
        sae_ground_truth = sparse_autoencoders.autoencoders[0].to(device)
        hook_point = "blocks.3.hook_resid_pre"
        data = load_dataset("NeelNanda/pile-10k", split="train")


    else:
        raise ValueError(f"Invalid model type: {model_type}")


    # Load our model
    model = HookedTransformer.from_pretrained(model_type).to(device)
    model_layer_wrapped = TransformerLensWrapper(model, hook_point)

    # Load our dataset
    
    tokenized_data = utils.tokenize_and_concatenate(data, model.tokenizer, max_length=128)
    tokenized_data = tokenized_data.shuffle(22)
    all_tokens = tokenized_data["tokens"]
    tokens = all_tokens[:token_num]  # Sample a smaller subset for processing

    # Create a TensorDataset and DataLoader
    dataset = TensorDataset(tokens)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    topk_seqs = None
    topk_feats = None

    # Loop through batches generated by DataLoader
    with torch.no_grad():
        for batch in tqdm(dataloader, total=len(dataloader), desc="Processing Batches"):
            batch_tokens = batch[0].to(device)  # DataLoader returns batches as lists of tensors
            feats_new = get_feature_activations(model_layer_wrapped, batch_tokens, sae_ground_truth, feature_idx)

            # Concatenate new features with previous top-k sequences
            if topk_seqs is not None:
                feats_cat = torch.cat([topk_feats, feats_new])
                seqs_cat = torch.cat([topk_seqs, batch_tokens])
            else:
                feats_cat = feats_new
                seqs_cat = batch_tokens

            topk_feats, topk_seqs = select_topk_sequences(feats_cat, top_k, seqs_cat)

            # Free up memory
            del feats_new, batch_tokens, feats_cat, seqs_cat
            torch.cuda.empty_cache()  # Explicitly release cached memory
    #mkdir 
    os.makedirs(f"features_{model_type}", exist_ok=True)
    # Save the top-k seqs to a pt.
    torch.save(topk_seqs, f"features_{model_type}/feature_{feature_idx}_topk_{top_k}_toknum_{token_num}_seqs.pt")
    # Also save the top-k feats to a pt.
    torch.save(topk_feats, f"features_{model_type}/feature_{feature_idx}_topk_{top_k}_toknum_{token_num}_feats.pt")


# Print final top-k sequences
tokens_to_text_and_print(topk_seqs, model.tokenizer)