import os
from typing import Any, Iterator, cast

import torch
from datasets import (
    Dataset,
    DatasetDict,
    IterableDataset,
    IterableDatasetDict,
    load_dataset,
)
from torch.utils.data import DataLoader
from transformer_lens import HookedTransformer

HfDataset = DatasetDict | Dataset | IterableDatasetDict | IterableDataset


class ActivationsStore:
    """
    Class for streaming tokens and generating and storing activations
    while training SAEs.
    """

    def __init__(
        self,
        cfg: Any,
        model: HookedTransformer,
        dataset: HfDataset | None = None,
        create_dataloader: bool = True,
    ):
        self.cfg = cfg
        self.model = model
        self.dataset = dataset or load_dataset(
            cfg.dataset_path, split="train", streaming=True
        )
        self.iterable_dataset = iter(self.dataset)

        # Check if dataset is tokenized
        dataset_sample = next(self.iterable_dataset)

        # Check if the dataset is tokenized
        if "tokens" in dataset_sample.keys():
            self.cfg.is_dataset_tokenized = True
            self.tokens_column = "tokens"
        elif "input_ids" in dataset_sample.keys():
            self.cfg.is_dataset_tokenized = True
            self.tokens_column = "input_ids"
        elif "text" in dataset_sample.keys():
            self.cfg.is_dataset_tokenized = False
            self.tokens_column = "text"
        elif "token_ids" in dataset_sample.keys():
            self.cfg.is_dataset_tokenized = True
            self.tokens_column = "token_ids"
        else:
            raise ValueError(
                "Dataset must have a 'tokens', 'input_ids', 'text', or 'token_ids' column."
            )

        self.iterable_dataset = iter(self.dataset)  # Reset iterator after checking

        if self.cfg.use_cached_activations:  # EDIT: load from multi-layer acts
            assert self.cfg.cached_activations_path is not None  # keep pyright happy
            # Sanity check: does the cache directory exist?
            assert os.path.exists(
                self.cfg.cached_activations_path
            ), f"Cache directory {self.cfg.cached_activations_path} does not exist. Consider double-checking your dataset, model, and hook names."

            self.next_cache_idx = 0  # which file to open next
            self.next_idx_within_buffer = 0  # where to start reading from in that file

            # Check that we have enough data on disk
            first_buffer = torch.load(f"{self.cfg.cached_activations_path}/0.pt")
            buffer_size_on_disk = first_buffer.shape[0]
            n_buffers_on_disk = len(os.listdir(self.cfg.cached_activations_path))
            # Note: we're assuming all files have the same number of tokens
            # (which seems reasonable imo since that's what our script does)
            n_activations_on_disk = buffer_size_on_disk * n_buffers_on_disk
            assert (
                n_activations_on_disk > self.cfg.total_training_tokens
            ), f"Only {n_activations_on_disk/1e6:.1f}M activations on disk, but cfg.total_training_tokens is {self.cfg.total_training_tokens/1e6:.1f}M."

            # TODO add support for "mixed loading" (ie use cache until you run out, then switch over to streaming from HF)

        if create_dataloader:
            # fill buffer half a buffer, so we can mix it with a new buffer
            self.storage_buffer = self.get_buffer(self.cfg.n_batches_in_buffer // 2)
            self.dataloader = self.get_data_loader()

    def get_batch_tokens(self):
        """
        Streams a batch of tokens from a dataset.
        """

        batch_size = self.cfg.store_batch_size
        context_size = self.cfg.context_size
        device = self.cfg.device

        batch_tokens = torch.zeros(
            size=(0, context_size), device=device, dtype=torch.long, requires_grad=False
        )

        current_batch = []
        current_length = 0

        # pbar = tqdm(total=batch_size, desc="Filling batches")
        while batch_tokens.shape[0] < batch_size:
            tokens = self._get_next_dataset_tokens()
            token_len = tokens.shape[0]

            # TODO: Fix this so that we are limiting how many tokens we get from the same context.
            assert self.model.tokenizer is not None  # keep pyright happy
            bos_token_id_tensor = torch.tensor(
                [self.model.tokenizer.bos_token_id],
                device=tokens.device,
                dtype=torch.long,
            )
            while token_len > 0 and batch_tokens.shape[0] < batch_size:
                # Space left in the current batch
                space_left = context_size - current_length

                # If the current tokens fit entirely into the remaining space
                if token_len <= space_left:
                    current_batch.append(tokens[:token_len])
                    current_length += token_len
                    break

                else:
                    # Take as much as will fit
                    current_batch.append(tokens[:space_left])

                    # Remove used part, add BOS
                    tokens = tokens[space_left:]
                    token_len -= space_left

                    # only add BOS if it's not already the first token
                    if tokens[0] != bos_token_id_tensor:
                        tokens = torch.cat(
                            (
                                bos_token_id_tensor,
                                tokens,
                            ),
                            dim=0,
                        )
                        token_len += 1
                    current_length = context_size

                # If a batch is full, concatenate and move to next batch
                if current_length == context_size:
                    full_batch = torch.cat(current_batch, dim=0)
                    batch_tokens = torch.cat(
                        (batch_tokens, full_batch.unsqueeze(0)), dim=0
                    )
                    current_batch = []
                    current_length = 0

            # pbar.n = batch_tokens.shape[0]
            # pbar.refresh()
        return batch_tokens[:batch_size]

    def get_activations(self, batch_tokens: torch.Tensor):
        """
        Returns activations of shape (batches, context, num_layers, d_in)
        """
        layers = (
            self.cfg.hook_point_layer
            if isinstance(self.cfg.hook_point_layer, list)
            else [self.cfg.hook_point_layer]
        )
        act_names = [self.cfg.hook_point.format(layer=layer) for layer in layers]
        hook_point_max_layer = max(layers)
        layerwise_activations = self.model.run_with_cache(
            batch_tokens,
            names_filter=act_names,
            stop_at_layer=hook_point_max_layer + 1,
        )[1]
        activations_list = [layerwise_activations[act_name] for act_name in act_names]
        if self.cfg.hook_point_head_index is not None:
            activations_list = [
                act[:, :, self.cfg.hook_point_head_index] for act in activations_list
            ]

        # Stack along a new dimension to keep separate layers distinct
        stacked_activations = torch.stack(activations_list, dim=2)

        return stacked_activations

    def get_buffer(self, n_batches_in_buffer: int):
        context_size = self.cfg.context_size
        batch_size = self.cfg.store_batch_size
        d_in = self.cfg.d_in
        total_size = batch_size * n_batches_in_buffer
        num_layers = (
            len(self.cfg.hook_point_layer)
            if isinstance(self.cfg.hook_point_layer, list)
            else 1
        )  # Number of hook points or layers

        if self.cfg.use_cached_activations:
            # Load the activations from disk
            buffer_size = total_size * context_size
            # Initialize an empty tensor with an additional dimension for layers
            new_buffer = torch.zeros(
                (buffer_size, num_layers, d_in),
                dtype=self.cfg.dtype,
                device=self.cfg.device,
            )
            n_tokens_filled = 0

            # Assume activations for different layers are stored separately and need to be combined
            while n_tokens_filled < buffer_size:
                if not os.path.exists(
                    f"{self.cfg.cached_activations_path}/{self.next_cache_idx}.pt"
                ):
                    print(
                        "\n\nWarning: Ran out of cached activation files earlier than expected."
                    )
                    print(
                        f"Expected to have {buffer_size} activations, but only found {n_tokens_filled}."
                    )
                    if buffer_size % self.cfg.total_training_tokens != 0:
                        print(
                            "This might just be a rounding error — your batch_size * n_batches_in_buffer * context_size is not divisible by your total_training_tokens"
                        )
                    print(f"Returning a buffer of size {n_tokens_filled} instead.")
                    print("\n\n")
                    new_buffer = new_buffer[:n_tokens_filled, ...]
                    return new_buffer

                activations = torch.load(
                    f"{self.cfg.cached_activations_path}/{self.next_cache_idx}.pt"
                )
                taking_subset_of_file = False
                if n_tokens_filled + activations.shape[0] > buffer_size:
                    activations = activations[: buffer_size - n_tokens_filled, ...]
                    taking_subset_of_file = True

                new_buffer[
                    n_tokens_filled : n_tokens_filled + activations.shape[0], ...
                ] = activations

                if taking_subset_of_file:
                    self.next_idx_within_buffer = activations.shape[0]
                else:
                    self.next_cache_idx += 1
                    self.next_idx_within_buffer = 0

                n_tokens_filled += activations.shape[0]

            return new_buffer

        refill_iterator = range(0, batch_size * n_batches_in_buffer, batch_size)
        # Initialize empty tensor buffer of the maximum required size with an additional dimension for layers
        new_buffer = torch.zeros(
            (total_size, context_size, num_layers, d_in),
            dtype=self.cfg.dtype,
            device=self.cfg.device,
        )

        for refill_batch_idx_start in refill_iterator:
            refill_batch_tokens = self.get_batch_tokens()
            refill_activations = self.get_activations(refill_batch_tokens)
            new_buffer[
                refill_batch_idx_start : refill_batch_idx_start + batch_size, ...
            ] = refill_activations

            # pbar.update(1)

        new_buffer = new_buffer.reshape(-1, num_layers, d_in)
        new_buffer = new_buffer[torch.randperm(new_buffer.shape[0])]

        return new_buffer

    def get_data_loader(
        self,
    ) -> Iterator[Any]:
        """
        Return a torch.utils.dataloader which you can get batches from.

        Should automatically refill the buffer when it gets to n % full.
        (better mixing if you refill and shuffle regularly).

        """

        batch_size = self.cfg.train_batch_size

        # 1. # create new buffer by mixing stored and new buffer
        mixing_buffer = torch.cat(
            [self.get_buffer(self.cfg.n_batches_in_buffer // 2), self.storage_buffer],
            dim=0,
        )

        mixing_buffer = mixing_buffer[torch.randperm(mixing_buffer.shape[0])]

        # 2.  put 50 % in storage
        self.storage_buffer = mixing_buffer[: mixing_buffer.shape[0] // 2]

        # 3. put other 50 % in a dataloader
        dataloader = iter(
            DataLoader(
                # TODO: seems like a typing bug?
                cast(Any, mixing_buffer[mixing_buffer.shape[0] // 2 :]),
                batch_size=batch_size,
                shuffle=True,
            )
        )

        return dataloader

    def next_batch(self):
        """
        Get the next batch from the current DataLoader.
        If the DataLoader is exhausted, refill the buffer and create a new DataLoader.
        """
        try:
            # Try to get the next batch
            return next(self.dataloader)
        except StopIteration:
            # If the DataLoader is exhausted, create a new one
            self.dataloader = self.get_data_loader()
            return next(self.dataloader)

    def _get_next_dataset_tokens(self) -> torch.Tensor:
        device = self.cfg.device
        if not self.cfg.is_dataset_tokenized:
            s = next(self.iterable_dataset)[self.tokens_column]
            tokens = self.model.to_tokens(
                s,
                truncate=True,
                move_to_device=True,
            ).squeeze(0)
            assert (
                len(tokens.shape) == 1
            ), f"tokens.shape should be 1D but was {tokens.shape}"
        else:
            tokens = torch.tensor(
                next(self.iterable_dataset)[self.tokens_column],
                dtype=torch.long,
                device=device,
                requires_grad=False,
            )
        return tokens
