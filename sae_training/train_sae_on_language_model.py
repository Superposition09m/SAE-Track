from dataclasses import dataclass
from typing import Any, NamedTuple, cast, Optional

import torch
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import LRScheduler
from tqdm import tqdm
from transformer_lens import HookedTransformer

import wandb
from sae_training.activations_store import ActivationsStore
from sae_training.evals import run_evals
from sae_training.geometric_median import compute_geometric_median
from sae_training.optim import get_scheduler
from sae_training.sae_group import SAEGroup
from sae_training.sparse_autoencoder import SparseAutoencoder


@dataclass
class SAETrainContext:
    """
    Context to track during training for a single SAE
    """

    act_freq_scores: torch.Tensor
    n_forward_passes_since_fired: torch.Tensor
    n_frac_active_tokens: int
    optimizer: Optimizer
    scheduler: LRScheduler

    @property
    def feature_sparsity(self) -> torch.Tensor:
        return self.act_freq_scores / self.n_frac_active_tokens


@dataclass
class TrainSAEGroupOutput:
    sae_group: SAEGroup
    checkpoint_paths: list[str]
    log_feature_sparsities: list[torch.Tensor]


def train_sae_on_language_model(
    model: HookedTransformer,
    sae_group: SAEGroup,
    activation_store: ActivationsStore,
    batch_size: int = 1024,
    n_checkpoints: int = 0,
    feature_sampling_window: int = 1000,  # how many training steps between resampling the features / considiring neurons dead
    dead_feature_threshold: float = 1e-8,  # how infrequently a feature has to be active to be considered dead
    use_wandb: bool = False,
    wandb_log_frequency: int = 50,
    #Optional
sae_pretrained_reg: Optional[SAEGroup] = None,
) -> SAEGroup:
    """
    @deprecated Use `train_sae_group_on_language_model` instead. This method is kept for backward compatibility.
    """
    return train_sae_group_on_language_model(
        model,
        sae_group,
        activation_store,
        batch_size,
        n_checkpoints,
        feature_sampling_window,
        use_wandb,
        wandb_log_frequency,
        sae_pretrained_reg=sae_pretrained_reg
    ).sae_group


def train_sae_group_on_language_model(
    model: HookedTransformer,
    sae_group: SAEGroup,
    activation_store: ActivationsStore,
    batch_size: int = 1024,
    n_checkpoints: int = 0,
    feature_sampling_window: int = 1000,  # how many training steps between resampling the features / considiring neurons dead
    use_wandb: bool = False,
    wandb_log_frequency: int = 50,
    #Optional
    sae_pretrained_reg: Optional[SAEGroup] = None,
) -> TrainSAEGroupOutput:
    total_training_tokens = sae_group.cfg.total_training_tokens
    total_training_steps = total_training_tokens // batch_size
    n_training_steps = 0
    n_training_tokens = 0

    checkpoint_thresholds = []
    if n_checkpoints > 0:
        checkpoint_thresholds = list(
            range(0, total_training_tokens, total_training_tokens // n_checkpoints)
        )[1:]

    all_layers = sae_group.cfg.hook_point_layer
    if not isinstance(all_layers, list):
        all_layers = [all_layers]

    train_contexts = [
        _build_train_context(sae, total_training_steps) for sae in sae_group
    ]

    # for sae in sae_group:
    #     print("SAVE!")
    #     sae.save_model("./what_is_this_1.pt")
    if sae_group.cfg.from_pretrained_path is None:
        _init_sae_group_b_decs(sae_group, activation_store, all_layers)
    else:
        print("WHAT IS THAT")
        print(sae_group.autoencoders[0].cfg)

        # exit()
    # for sae in sae_group:
    #     print("SAVE!")
    #     sae.save_model("./what_is_this_2.pt")
    # exit()
    pbar = tqdm(total=total_training_tokens, desc="Training SAE")
    checkpoint_paths: list[str] = []
    while n_training_tokens < total_training_tokens:
        # Do a training step.
        layer_acts = activation_store.next_batch()
        n_training_tokens += batch_size

        mse_losses: list[torch.Tensor] = []
        l1_losses: list[torch.Tensor] = []
        reg_losses: list[torch.Tensor] = []
        for (
            sparse_autoencoder,
            ctx,
        ) in zip(sae_group, train_contexts):
            wandb_suffix = _wandb_log_suffix(sae_group.cfg, sparse_autoencoder.cfg)
            step_output = _train_step(
                sparse_autoencoder=sparse_autoencoder,
                layer_acts=layer_acts,
                ctx=ctx,
                feature_sampling_window=feature_sampling_window,
                use_wandb=use_wandb,
                n_training_steps=n_training_steps,
                all_layers=all_layers,
                batch_size=batch_size,
                wandb_suffix=wandb_suffix,
                #TODO:use pretrained weight to do regularization
                sae_pretrained_reg=sae_pretrained_reg.autoencoders[0] if sae_pretrained_reg is not None else None,

            )
            mse_losses.append(step_output.mse_loss)
            l1_losses.append(step_output.l1_loss)
            reg_losses.append(step_output.reg_loss)
            if use_wandb:
                with torch.no_grad():
                    if (n_training_steps + 1) % wandb_log_frequency == 0:
                        wandb.log(
                            _build_train_step_log_dict(
                                sparse_autoencoder,
                                step_output,
                                ctx,
                                wandb_suffix,
                                n_training_tokens,
                            ),
                            step=n_training_steps,
                        )

                    # record loss frequently, but not all the time.
                    if (n_training_steps + 1) % (wandb_log_frequency * 10) == 0:
                        sparse_autoencoder.eval()
                        print("runeval")
                        run_evals(
                            sparse_autoencoder,
                            activation_store,
                            model,
                            n_training_steps,
                            suffix=wandb_suffix,
                        )
                        sparse_autoencoder.train()

        # checkpoint if at checkpoint frequency
        if checkpoint_thresholds and n_training_tokens > checkpoint_thresholds[0]:
            checkpoint_path = _save_checkpoint(
                sae_group,
                train_contexts=train_contexts,
                checkpoint_name=n_training_tokens,
            ).path
            checkpoint_paths.append(checkpoint_path)
            checkpoint_thresholds.pop(0)

        ###############

        n_training_steps += 1
        pbar.set_description(
            f"{n_training_steps}| MSE Loss {torch.stack(mse_losses).mean().item():.3f} | L1 {torch.stack(l1_losses).mean().item():.3f} | Reg {torch.stack(reg_losses).mean().item()}"
        )
        pbar.update(batch_size)

    # save final sae group to checkpoints folder
    final_checkpoint = _save_checkpoint(
        sae_group,
        train_contexts=train_contexts,
        checkpoint_name="final",
        wandb_aliases=["final_model"],
    )
    checkpoint_paths.append(final_checkpoint.path)

    return TrainSAEGroupOutput(
        sae_group=sae_group,
        checkpoint_paths=checkpoint_paths,
        log_feature_sparsities=final_checkpoint.log_feature_sparsities,
    )


def _wandb_log_suffix(cfg: Any, hyperparams: Any):
    # Create a mapping from cfg list keys to their corresponding hyperparams attributes
    key_mapping = {
        "hook_point_layer": "layer",
        "l1_coefficient": "coeff",
        "lp_norm": "l",
        "lr": "lr",
    }

    # Generate the suffix by iterating over the keys that have list values in cfg
    suffix = "".join(
        f"_{key_mapping.get(key, key)}{getattr(hyperparams, key, '')}"
        for key, value in vars(cfg).items()
        if isinstance(value, list)
    )
    return suffix


def _build_train_context(
    sae: SparseAutoencoder, total_training_steps: int
) -> SAETrainContext:
    act_freq_scores = torch.zeros(
        cast(int, sae.cfg.d_sae),
        device=sae.cfg.device,
    )
    n_forward_passes_since_fired = torch.zeros(
        cast(int, sae.cfg.d_sae),
        device=sae.cfg.device,
    )
    n_frac_active_tokens = 0

    optimizer = Adam(sae.parameters(), lr=sae.cfg.lr)
    scheduler = get_scheduler(
        sae.cfg.lr_scheduler_name,
        optimizer=optimizer,
        warm_up_steps=sae.cfg.lr_warm_up_steps,
        training_steps=total_training_steps,
        lr_end=sae.cfg.lr / 10,  # heuristic for now.
    )

    return SAETrainContext(
        act_freq_scores=act_freq_scores,
        n_forward_passes_since_fired=n_forward_passes_since_fired,
        n_frac_active_tokens=n_frac_active_tokens,
        optimizer=optimizer,
        scheduler=scheduler,
    )


def _init_sae_group_b_decs(
    sae_group: SAEGroup, activation_store: ActivationsStore, all_layers: list[int]
) -> None:
    """
    extract all activations at a certain layer and use for sae b_dec initialization
    """
    geometric_medians = {}
    for sae in sae_group:
        hyperparams = sae.cfg
        sae_layer_id = all_layers.index(hyperparams.hook_point_layer)
        if hyperparams.b_dec_init_method == "geometric_median":
            layer_acts = activation_store.storage_buffer.detach()[:, sae_layer_id, :]
            # get geometric median of the activations if we're using those.
            if sae_layer_id not in geometric_medians:
                median = compute_geometric_median(
                    layer_acts,
                    maxiter=100,
                ).median
                geometric_medians[sae_layer_id] = median
            sae.initialize_b_dec_with_precalculated(geometric_medians[sae_layer_id])
        elif hyperparams.b_dec_init_method == "mean":
            layer_acts = activation_store.storage_buffer.detach().cpu()[
                :, sae_layer_id, :
            ]
            sae.initialize_b_dec_with_mean(layer_acts)


@dataclass
class TrainStepOutput:
    sae_in: torch.Tensor
    sae_out: torch.Tensor
    feature_acts: torch.Tensor
    loss: torch.Tensor
    mse_loss: torch.Tensor
    l1_loss: torch.Tensor
    reg_loss: torch.Tensor
    ghost_grad_loss: torch.Tensor
    ghost_grad_neuron_mask: torch.Tensor


def _train_step(
    sparse_autoencoder: SparseAutoencoder,
    layer_acts: torch.Tensor,
    ctx: SAETrainContext,
    feature_sampling_window: int,  # how many training steps between resampling the features / considiring neurons dead
    use_wandb: bool,
    n_training_steps: int,
    all_layers: list[int],
    batch_size: int,
    wandb_suffix: str,
    sae_pretrained_reg: Optional[SparseAutoencoder] = None,  # Single SAE for regularization
) -> TrainStepOutput:
    assert sparse_autoencoder.cfg.d_sae is not None  # keep pyright happy
    hyperparams = sparse_autoencoder.cfg
    layer_id = all_layers.index(hyperparams.hook_point_layer)
    sae_in = layer_acts[:, layer_id, :]

    sparse_autoencoder.train()
    # Make sure the W_dec is still zero-norm
    if hyperparams.w_dec_norm == 1:
        # print(1)
        sparse_autoencoder.set_decoder_norm_to_unit_norm()

    # log and then reset the feature sparsity every feature_sampling_window steps
    if (n_training_steps + 1) % feature_sampling_window == 0:
        feature_sparsity = ctx.feature_sparsity
        log_feature_sparsity = _log_feature_sparsity(feature_sparsity)

        if use_wandb:
            wandb_histogram = wandb.Histogram(log_feature_sparsity.numpy())
            wandb.log(
                {
                    f"metrics/mean_log10_feature_sparsity{wandb_suffix}": log_feature_sparsity.mean().item(),
                    f"plots/feature_density_line_chart{wandb_suffix}": wandb_histogram,
                    f"sparsity/below_1e-5{wandb_suffix}": (feature_sparsity < 1e-5)
                    .sum()
                    .item(),
                    f"sparsity/below_1e-6{wandb_suffix}": (feature_sparsity < 1e-6)
                    .sum()
                    .item(),
                },
                step=n_training_steps,
            )

        ctx.act_freq_scores = torch.zeros(
            sparse_autoencoder.cfg.d_sae, device=sparse_autoencoder.cfg.device
        )
        ctx.n_frac_active_tokens = 0

    ghost_grad_neuron_mask = (
        ctx.n_forward_passes_since_fired > sparse_autoencoder.cfg.dead_feature_window
    ).bool()


    # Forward and Backward Passes
    (
        sae_out,
        feature_acts,
        loss,
        mse_loss,
        l1_loss,
        ghost_grad_loss,
    ) = sparse_autoencoder(
        sae_in,
        ghost_grad_neuron_mask,
    )
    reg_loss = torch.tensor(0.0, device=sparse_autoencoder.cfg.device)
    if sae_pretrained_reg is not None:
        # Regularize both W_enc and W_dec using the L2 norm
        encoder_reg_loss = torch.norm(sparse_autoencoder.W_enc - sae_pretrained_reg.W_enc, p=2) ** 2
        decoder_reg_loss = torch.norm(sparse_autoencoder.W_dec - sae_pretrained_reg.W_dec, p=2) ** 2
        reg_loss = encoder_reg_loss + decoder_reg_loss
        # Include the regularization in the total loss
        loss += hyperparams.reg_coefficient * reg_loss








    did_fire = (feature_acts > 0).float().sum(-2) > 0
    ctx.n_forward_passes_since_fired += 1
    ctx.n_forward_passes_since_fired[did_fire] = 0

    with torch.no_grad():
        # Calculate the sparsities, and add it to a list, calculate sparsity metrics
        ctx.act_freq_scores += (feature_acts.abs() > 0).float().sum(0)
        ctx.n_frac_active_tokens += batch_size

    ctx.optimizer.zero_grad()
    loss.backward()
    sparse_autoencoder.remove_gradient_parallel_to_decoder_directions()
    ctx.optimizer.step()
    ctx.scheduler.step()

    return TrainStepOutput(
        sae_in=sae_in,
        sae_out=sae_out,
        feature_acts=feature_acts,
        loss=loss,
        mse_loss=mse_loss,
        l1_loss=l1_loss,
        ghost_grad_loss=ghost_grad_loss,
        ghost_grad_neuron_mask=ghost_grad_neuron_mask,
        reg_loss=reg_loss,
    )


def _build_train_step_log_dict(
    sparse_autoencoder: SparseAutoencoder,
    output: TrainStepOutput,
    ctx: SAETrainContext,
    wandb_suffix: str,
    n_training_tokens: int,
) -> dict[str, Any]:
    sae_in = output.sae_in
    sae_out = output.sae_out
    feature_acts = output.feature_acts
    mse_loss = output.mse_loss
    l1_loss = output.l1_loss
    ghost_grad_loss = output.ghost_grad_loss
    reg_loss = output.reg_loss  # Add reg_loss here
    loss = output.loss
    ghost_grad_neuron_mask = output.ghost_grad_neuron_mask

    # metrics for currents acts
    l0 = (feature_acts > 0).float().sum(-1).mean()
    current_learning_rate = ctx.optimizer.param_groups[0]["lr"]

    per_token_l2_loss = (sae_out - sae_in).pow(2).sum(dim=-1).squeeze()
    total_variance = (sae_in - sae_in.mean(0)).pow(2).sum(-1)
    explained_variance = 1 - per_token_l2_loss / total_variance

    return {
        # losses
        f"losses/mse_loss{wandb_suffix}": mse_loss.item(),
        f"losses/l1_loss{wandb_suffix}": l1_loss.item()
        / sparse_autoencoder.l1_coefficient,  # normalize by l1 coefficient
        f"losses/ghost_grad_loss{wandb_suffix}": ghost_grad_loss.item(),
        f"losses/reg_loss{wandb_suffix}": reg_loss.item(),  # Log reg_loss
        f"losses/overall_loss{wandb_suffix}": loss.item(),
        # variance explained
        f"metrics/explained_variance{wandb_suffix}": explained_variance.mean().item(),
        f"metrics/explained_variance_std{wandb_suffix}": explained_variance.std().item(),
        f"metrics/l0{wandb_suffix}": l0.item(),
        # sparsity
        f"sparsity/mean_passes_since_fired{wandb_suffix}": ctx.n_forward_passes_since_fired.mean().item(),
        f"sparsity/dead_features{wandb_suffix}": ghost_grad_neuron_mask.sum().item(),
        f"details/current_learning_rate{wandb_suffix}": current_learning_rate,
        "details/n_training_tokens": n_training_tokens,
    }



class SaveCheckpointOutput(NamedTuple):
    path: str
    log_feature_sparsity_path: str
    log_feature_sparsities: list[torch.Tensor]


def _save_checkpoint(
    sae_group: SAEGroup,
    train_contexts: list[SAETrainContext],
    checkpoint_name: int | str,
    wandb_aliases: list[str] | None = None,
) -> SaveCheckpointOutput:
    path = (
        f"{sae_group.cfg.checkpoint_path}/{checkpoint_name}_{sae_group.get_name()}.pt"
    )
    for sae in sae_group:
        if sae.cfg.w_dec_norm == 1:
            # print(2)
            sae.set_decoder_norm_to_unit_norm()
    sae_group.save_model(path)
    log_feature_sparsity_path = f"{sae_group.cfg.checkpoint_path}/{checkpoint_name}_{sae_group.get_name()}_log_feature_sparsity.pt"
    log_feature_sparsities = [
        _log_feature_sparsity(ctx.feature_sparsity) for ctx in train_contexts
    ]
    torch.save(log_feature_sparsities, log_feature_sparsity_path)
    # if sae_group.cfg.log_to_wandb:
    #     model_artifact = wandb.Artifact(
    #         f"{sae_group.get_name()}",
    #         type="model",
    #         metadata=dict(sae_group.cfg.__dict__),
    #     )
    #     model_artifact.add_file(path)
    #     wandb.log_artifact(model_artifact, aliases=wandb_aliases)
    #
    #     sparsity_artifact = wandb.Artifact(
    #         f"{sae_group.get_name()}_log_feature_sparsity",
    #         type="log_feature_sparsity",
    #         metadata=dict(sae_group.cfg.__dict__),
    #     )
    #     sparsity_artifact.add_file(log_feature_sparsity_path)
    #     wandb.log_artifact(sparsity_artifact)
    return SaveCheckpointOutput(path, log_feature_sparsity_path, log_feature_sparsities)


def _log_feature_sparsity(
    feature_sparsity: torch.Tensor, eps: float = 1e-10
) -> torch.Tensor:
    return torch.log10(feature_sparsity + eps).detach().cpu()


