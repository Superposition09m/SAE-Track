import torch
from datasets import load_dataset
from sae_lens.sae import SAE
from transformers import LlamaTokenizer, LlamaForCausalLM

from sae_training.sae_group import SAEGroup
from transformer_lens import utils, HookedTransformer
from sae_vis.data_storing_fns import SaeVisData
from sae_vis.data_config_classes import SaeVisConfig
from sae import Sae
import argparse
from datasets import load_dataset

from sae_vis import SaeVisConfig, SaeVisData
from transformer_lens import HookedTransformer, utils

from sae_training.utils import LMSparseAutoencoderSessionloader
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_type", type=str, default="stanford-gpt2-medium-a")
parser.add_argument("--ckpt", type=int, required=True)
parser.add_argument("--sae_path", type=str, required=True)
args = parser.parse_args()

model_type = args.model_type
ckpt = args.ckpt

print(f"ckpt: {ckpt}")
device = "cuda"
from_zero = "final"

if model_type == "stanford-gpt2-small-a":
    model = HookedTransformer.from_pretrained("stanford-gpt2-small-a", checkpoint_index=ckpt).to(device)
    data = load_dataset("stas/openwebtext-10k", split="train")
    _, sparse_autoencoder, _ = LMSparseAutoencoderSessionloader.load_session_from_pretrained(
        # "pipe/e9f3k2o0/final_sae_group_stanford-gpt2-small-a_blocks.5.hook_resid_pre_49152.pt"
        args.sae_path
    )
    sae = sparse_autoencoder.autoencoders[0].to(device)
    hkpt = "blocks.5.hook_resid_pre"
    tokenized_data = utils.tokenize_and_concatenate(data, model.tokenizer, max_length=128)
    tokenized_data = tokenized_data.shuffle(22)

    # Get the tokens as a tensor
    all_tokens = tokenized_data["tokens"]
elif model_type == "stanford-gpt2-medium-a":
    model = HookedTransformer.from_pretrained("stanford-gpt2-medium-a",checkpoint_index=ckpt).to(device)
    data = load_dataset("stas/openwebtext-10k", split="train")
    _, sparse_autoencoder, _ = LMSparseAutoencoderSessionloader.load_session_from_pretrained(
        args.sae_path
    )
    sae = sparse_autoencoder.autoencoders[0].to(device)
    hkpt = "blocks.6.hook_resid_pre"
    tokenized_data = utils.tokenize_and_concatenate(data, model.tokenizer, max_length=128)
    tokenized_data = tokenized_data.shuffle(22)

    # Get the tokens as a tensor
    all_tokens = tokenized_data["tokens"]

elif model_type == "pythia-70m-deduped":
    sae, cfg_dict, sparsity = SAE.from_pretrained(
        release="pythia-70m-deduped-res-sm",  # see other options in sae_lens/pretrained_saes.yaml
        sae_id="blocks.2.hook_resid_post",  # won't always be a hook point
    )
    print(sae)
    sae = sae.to(device)
    model = HookedTransformer.from_pretrained("EleutherAI/pythia-70m-deduped").to(device)
    data = load_dataset("NeelNanda/pile-10k", split="train")
    hkpt = "blocks.2.hook_resid_post"
    tokenized_data = utils.tokenize_and_concatenate(data, model.tokenizer, max_length=128)
    tokenized_data = tokenized_data.shuffle(22)

    # Get the tokens as a tensor
    all_tokens = tokenized_data["tokens"]

elif model_type == "pythia-160m-deduped":
    model = HookedTransformer.from_pretrained("EleutherAI/pythia-160m-deduped", checkpoint_index=ckpt).to(device)
    data = load_dataset("NeelNanda/pile-10k", split="train")
    # _, sparse_autoencoder, _ = LMSparseAutoencoderSessionloader.load_session_from_pretrained(
    #     "checkpoints/0nvrwgo9/final_sae_group_pythia-160m-deduped_blocks.4.hook_resid_pre_65536.pt"
    # )
    sparse_autoencoder = SAEGroup.load_from_pretrained(
        "checkpoints/bky32ycn/final_sae_group_pythia-160m-deduped_blocks.4.hook_resid_pre_49152.pt"
        # args.sae_path
    )
    sae = sparse_autoencoder.autoencoders[0].to(device)
    hkpt = "blocks.4.hook_resid_pre"
    tokenized_data = utils.tokenize_and_concatenate(data, model.tokenizer, max_length=128)
    tokenized_data = tokenized_data.shuffle(22)

    # Get the tokens as a tensor
    all_tokens = tokenized_data["tokens"]
elif model_type == "pythia-410m-deduped":
    model = HookedTransformer.from_pretrained("EleutherAI/pythia-410m-deduped", checkpoint_index=ckpt).to(device)
    data = load_dataset("NeelNanda/pile-10k", split="train")
    # _, sparse_autoencoder, _ = LMSparseAutoencoderSessionloader.load_session_from_pretrained(
    #     "checkpoints/srr6jh1f/final_sae_group_pythia-410m-deduped_blocks.4.hook_resid_pre_65536.pt"
    # )
    sparse_autoencoder = SAEGroup.load_from_pretrained(
        "pipe/z5sgu15x/final_sae_group_pythia-410m-deduped_blocks.12.hook_resid_pre_16384.pt"
    )
        
    sae = sparse_autoencoder.autoencoders[0].to(device)
    hkpt = "blocks.12.hook_resid_pre"
    tokenized_data = utils.tokenize_and_concatenate(data, model.tokenizer, max_length=128)
    tokenized_data = tokenized_data.shuffle(22)

    # Get the tokens as a tensor
    all_tokens = tokenized_data["tokens"]
elif model_type=="pythia-1.4b-deduped":
    model = HookedTransformer.from_pretrained("EleutherAI/pythia-1.4b-deduped", checkpoint_index=ckpt).to(device)
    data = load_dataset("NeelNanda/pile-10k", split="train")
    # _, sparse_autoencoder, _ = LMSparseAutoencoderSessionloader.load_session_from_pretrained(
    #     "pipe/xnnrwhbp/final_sae_group_pythia-1.4b-deduped_blocks.3.hook_resid_pre_131072.pt"
    # )
    sparse_autoencoder = SAEGroup.load_from_pretrained(
        args.sae_path
    )
    sae = sparse_autoencoder.autoencoders[0].to(device)
    hkpt = "blocks.3.hook_resid_pre"
    tokenized_data = utils.tokenize_and_concatenate(data, model.tokenizer, max_length=128)
    tokenized_data = tokenized_data.shuffle(22)

    # Get the tokens as a tensor
    all_tokens = tokenized_data["tokens"]
elif model_type == "our_pythia_160m":
    model = HookedTransformer.load(
                    f"checkpoints/serene-puddle-32/checkpoint_step_{ckpt}.pt"
                )
    data = load_dataset("NeelNanda/pile-10k", split="train")
    sparse_autoencoder = SAEGroup.load_from_pretrained(
        "checkpoints/biswmohm/120000512_sae_group_our_pythia_160_blocks.4.hook_resid_pre_49152.pt"
    )
    sae = sparse_autoencoder.autoencoders[0].to(device)
    hkpt = "blocks.4.hook_resid_pre"
    tokenized_data = utils.tokenize_and_concatenate(data, model.tokenizer, max_length=128)
    tokenized_data = tokenized_data.shuffle(22)

    # Get the tokens as a tensor
    all_tokens = tokenized_data["tokens"]
    
elif model_type == "Amber":
    # Load dataset in streaming mode to avoid loading everything into memory
    dataset = load_dataset(
        "LLM360/AmberDatasets",
        data_files="train/train_359.jsonl",
        split="train",
        streaming=True
    )

    # Parameters
    seq_len = 64
    num_sequences = 10000  # Desired number of sequences
    sequences = []

    # Iterate over each row in the dataset
    for row in dataset:
        tokens = row['token_ids']
        # Divide tokens into chunks of 64
        for i in range(0, len(tokens), seq_len):
            chunk = tokens[i:i + seq_len]
            # Only add complete sequences of length 64
            if len(chunk) == seq_len:
                sequences.append(chunk)
            # Stop once we have enough sequences
            if len(sequences) == num_sequences:
                break
        if len(sequences) == num_sequences:
            break

    # Convert to a PyTorch tensor
    all_tokens = torch.tensor(sequences)
    tokenizer = LlamaTokenizer.from_pretrained("LLM360/Amber")
    model_hf = LlamaForCausalLM.from_pretrained("LLM360/Amber")
    model = HookedTransformer.from_pretrained("llama-7b-hf", hf_model=model_hf)
    model.set_tokenizer(tokenizer)
    _, sparse_autoencoder, _ = LMSparseAutoencoderSessionloader.load_session_from_pretrained(
        "checkpoints/pzp4vwep/75000832_sae_group_Amber_blocks.5.hook_resid_pre_131072.pt"
    )
    sae = sparse_autoencoder.autoencoders[0].to(device)

    hkpt = "blocks.5.hook_resid_pre"



else:
    raise ValueError(f"Invalid model type: {model_type}")

assert isinstance(all_tokens, torch.Tensor)

print(all_tokens.shape)

torch.cuda.empty_cache()
import gc

gc.collect()

# test_feature_idx_gpt = [45097, 30777, 60424, 26355, 20451, 42624, 61515]
# test_feature_idx_gpt=[42577, 23276, 14408, 42532, 47939, 11399, 36237, 13780]
# test_feature_idx_gpt=range(128,1024)
test_feature_idx_gpt = range(256)
# test_feature_idx_gpt = range(1023,2047)
#random from 65536, use randomseed to make it reproducible
import random
random.seed(0)
# test_feature_idx_gpt = random.sample(range(65536), 256)

print(test_feature_idx_gpt)
feature_vis_config_gpt = SaeVisConfig(
    hook_point=hkpt,
    features=test_feature_idx_gpt,
    verbose=True,
    # minibatch_size_features= 64,
    # minibatch_size_tokens = 16
)
# print(all_tokens[:8192*5])
# exit()
sae_vis_data_gpt = SaeVisData.create(
    encoder=sae,
    model=model,
    tokens=all_tokens[:8192*2], #5??
    cfg=feature_vis_config_gpt,
)

filename = f"{model_type}_GT_{hkpt}_{ckpt}_{from_zero}.html"
sae_vis_data_gpt.save_feature_centric_vis(filename)