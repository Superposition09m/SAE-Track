import os
from typing import List

import einops
import torch
from datasets import load_dataset
from torch import Tensor, nn
import torch.nn.functional as F
from tqdm import tqdm

from sae_training.utils import LMSparseAutoencoderSessionloader
from sae_vis.model_fns import TransformerLensWrapper, AutoEncoder, AutoEncoderConfig
from transformer_lens import HookedTransformer, utils


def get_model_activations(
        model: TransformerLensWrapper,
        tokens: Tensor
) -> Tensor:
    with torch.no_grad():
        _, model_acts = model.forward(tokens, return_logits=False)
    return model_acts


import torch
import torch.nn.functional as F
import einops
from torch import Tensor, nn


def get_feature_activations(
        model_wrapper: TransformerLensWrapper,
        tokens: Tensor,
        encoder: nn.Module,
        feature: int
) -> Tensor:
    # If encoder isn't an AutoEncoder, wrap it in one
    if not isinstance(encoder, AutoEncoder):
        assert set(encoder.state_dict().keys()).issuperset(
            {"W_enc", "W_dec", "b_enc", "b_dec"}
        ), "If encoder isn't an AutoEncoder, it should have weights 'W_enc', 'W_dec', 'b_enc', 'b_dec'"

        d_in, d_hidden = encoder.state_dict()["W_enc"].shape
        device = encoder.W_enc.device

        encoder_cfg = AutoEncoderConfig(d_in=d_in, d_hidden=d_hidden)
        encoder_wrapper = AutoEncoder(encoder_cfg).to(device)
        encoder_wrapper.load_state_dict(encoder.state_dict(), strict=False)
    else:
        encoder_wrapper = encoder  # No wrapping needed if already an AutoEncoder

    # Get activations from the model
    model_activations = get_model_activations(model_wrapper, tokens)

    # Get the feature direction and bias from the wrapped encoder
    feature_act_dir = encoder_wrapper.W_enc  # (d_in, feats)
    feature_bias = encoder_wrapper.b_enc  # (feats,)

    # Center the model activations based on decoder bias (if applicable)
    x_cent = model_activations - encoder_wrapper.b_dec * encoder_wrapper.cfg.apply_b_dec_to_input

    # Calculate feature activations using einsum
    feat_acts_pre = einops.einsum(
        x_cent, feature_act_dir, "batch seq d_in, d_in feats -> batch seq feats"
    )

    # Apply bias and ReLU activation
    feature_activations = F.relu(feat_acts_pre + feature_bias)

    return feature_activations[..., feature]

def get_features_activations(
        model_wrapper: TransformerLensWrapper,
        tokens: Tensor,
        encoder: nn.Module,
) -> Tensor:
    # If encoder isn't an AutoEncoder, wrap it in one
    if not isinstance(encoder, AutoEncoder):
        assert set(encoder.state_dict().keys()).issuperset(
            {"W_enc", "W_dec", "b_enc", "b_dec"}
        ), "If encoder isn't an AutoEncoder, it should have weights 'W_enc', 'W_dec', 'b_enc', 'b_dec'"

        d_in, d_hidden = encoder.state_dict()["W_enc"].shape
        device = encoder.W_enc.device

        encoder_cfg = AutoEncoderConfig(d_in=d_in, d_hidden=d_hidden)
        encoder_wrapper = AutoEncoder(encoder_cfg).to(device)
        encoder_wrapper.load_state_dict(encoder.state_dict(), strict=False)
    else:
        encoder_wrapper = encoder  # No wrapping needed if already an AutoEncoder

    # Get activations from the model
    model_activations = get_model_activations(model_wrapper, tokens)

    # Get the feature direction and bias from the wrapped encoder
    feature_act_dir = encoder_wrapper.W_enc  # (d_in, feats)
    feature_bias = encoder_wrapper.b_enc  # (feats,)

    # Center the model activations based on decoder bias (if applicable)
    x_cent = model_activations - encoder_wrapper.b_dec * encoder_wrapper.cfg.apply_b_dec_to_input

    # Calculate feature activations using einsum
    feat_acts_pre = einops.einsum(
        x_cent, feature_act_dir, "batch seq d_in, d_in feats -> batch seq feats"
    )

    # Apply bias and ReLU activation
    feature_activations = F.relu(feat_acts_pre + feature_bias)

    return feature_activations

def select_topk_sequences(feats, top_k, tokens):
    """
    Selects sequences from the batch that contain any of the top-k tokens globally
    and returns both the top-k feature activations (feats) and their corresponding tokens.

    Args:
        feats (torch.Tensor): The input tensor of shape [batch_size, seq_len].
        top_k (int): Number of top tokens to keep across the entire batch.
        tokens (torch.Tensor): The corresponding tokens for the input sequences.

    Returns:
        torch.Tensor: The selected feature activations (feats) containing any of the top-k tokens.
        torch.Tensor: The corresponding tokens for the selected feature activations.
    """
    # Flatten the feats tensor to search for the top-k tokens across the entire batch
    flat_feats = feats.view(-1)

    # Find the top-k tokens globally (across the entire 2D tensor)
    topk_vals, topk_indices = torch.topk(flat_feats, top_k)

    # Convert the flat indices back to 2D indices (batch_size, seq_len)
    batch_size, seq_len = feats.shape
    topk_batch_indices = topk_indices // seq_len  # Which sequence in the batch
    topk_token_indices = topk_indices % seq_len  # Which token in the sequence

    # Create a mask to keep only sequences containing any top-k tokens
    mask = torch.zeros(batch_size, dtype=torch.bool)
    mask[topk_batch_indices] = True

    # Select sequences containing top-k tokens
    topk_feats = feats[mask]

    # Select corresponding tokens for those sequences
    selected_tokens = tokens[mask]

    return topk_feats, selected_tokens


def tokens_to_text_and_print(tokens, tokenizer):
    """
    Converts a tensor of token IDs into human-readable text and prints each sequence.

    Args:
        tokens (torch.Tensor): The tensor of token IDs of shape [batch_size, seq_len].
        tokenizer: The tokenizer associated with the model, used to decode tokens.
    """
    decoded_sentences = [tokenizer.decode(token_seq, skip_special_tokens=True) for token_seq in tokens]
    for idx, sentence in enumerate(decoded_sentences):
        print(f"Sequence {idx + 1}: {sentence}")


def tokens_to_word_and_print(tokens, tokenizer):
    """
    Converts a tensor of token IDs into human-readable text, with each token
    separated by "<>", and prints each sequence. "Ġ" (space) is handled properly
    within the token brackets.

    Args:
        tokens (torch.Tensor): The tensor of token IDs of shape [batch_size, seq_len].
        tokenizer: The tokenizer associated with the model, used to decode tokens.
    """
    for idx, token_seq in enumerate(tokens):
        # Decode tokens into their corresponding strings
        decoded_tokens = tokenizer.convert_ids_to_tokens(token_seq)

        # Format tokens: Add space for "Ġ" and wrap each token with "<>"
        formatted_sentence = "".join(
            f"< {token[1:]}>" if token.startswith("Ġ") else f"<{token}>"
            for token in decoded_tokens
        )

        # Print the formatted sentence
        print(f"Sequence {idx + 1}: {formatted_sentence}")
