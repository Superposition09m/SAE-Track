import copy
from typing import Any, cast

import wandb

from sae_training.sae_group import SAEGroup
from transformer_lens import HookedTransformer

from sae_training.config import LanguageModelSAERunnerConfig

# from sae_training.activation_store import ActivationStore
from sae_training.train_sae_on_language_model import train_sae_on_language_model
from sae_training.utils import LMSparseAutoencoderSessionloader


def language_model_sae_runner(cfg: LanguageModelSAERunnerConfig):
    """ """
    print(f"SAE model saved at {cfg.checkpoint_path}")
    if cfg.from_pretrained_path is not None:
        # (
        #     _,
        #     sparse_autoencoder,
        #     activations_loader,
        # ) = LMSparseAutoencoderSessionloader.load_session_from_pretrained(
        #     cfg.from_pretrained_path
        # )

        #brandnew model, _, activations_loader
        loader = LMSparseAutoencoderSessionloader(cfg)
        model=loader.get_model(cfg.model_name, cfg.ckpt).to(cfg.device)#new
        activations_loader=loader.get_activations_loader(cfg, model)#new
        #but sae is loaded from pretrained
        sparse_autoencoder = SAEGroup.load_from_pretrained(cfg.from_pretrained_path)
        #handle attribute problem
        if not hasattr(sparse_autoencoder.autoencoders[0], "w_dec_norm"):
            sparse_autoencoder.autoencoders[0].cfg.w_dec_norm = 1
            sparse_autoencoder.autoencoders[0].w_dec_norm = 1
            sparse_autoencoder.cfg.w_dec_norm = 1
            # exit("This is a hack to stop the code here, so that I can check the model before training")
        sparse_autoencoder_reg = copy.deepcopy(sparse_autoencoder)
        sparse_autoencoder_reg.eval()

        sparse_autoencoder.cfg = cfg
        print(sparse_autoencoder.cfg)
        sparse_autoencoder._renew_autoencoders_cfg(cfg)
        # sparse_autoencoder.save_model("./what_is_this_0.pt")
        # exit(
        #     "This is a hack to stop the code here, so that I can check the model before training"
        # )
    else:
        loader = LMSparseAutoencoderSessionloader(cfg)
        model, sparse_autoencoder, activations_loader = loader.load_session()
        sparse_autoencoder_reg = None
        print(sparse_autoencoder.cfg)
    if cfg.log_to_wandb:
        wandb.init(project=cfg.wandb_project, config=cast(Any, cfg), name=cfg.run_name)

    # train SAE
    sparse_autoencoder = train_sae_on_language_model(
        model,
        sparse_autoencoder,
        activations_loader,
        n_checkpoints=cfg.n_checkpoints,
        batch_size=cfg.train_batch_size,
        feature_sampling_window=cfg.feature_sampling_window,
        dead_feature_threshold=cfg.dead_feature_threshold,
        use_wandb=cfg.log_to_wandb,
        wandb_log_frequency=cfg.wandb_log_frequency,
        sae_pretrained_reg=sparse_autoencoder_reg,
    )

    if cfg.log_to_wandb:
        wandb.finish()

    return sparse_autoencoder, cfg.checkpoint_path
