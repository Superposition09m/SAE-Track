# Codebase for "Tracking the Feature Dynamics in LLM Training: A Mechanistic Study"

### 🔹 **Utilities**
- **`feat_fns.py`**: Utility functions used throughout the codebase.
- **`generate_feat_seqs.py`**: Generates datapoints corresponding to given features.

### 🔹 **SAE-Track**
- **`pipeline.py`**: Implements SAE-Track by training a sequence of SAEs using `sparse_autoencoder_trainer.py`.
- **`sparse_autoencoder_trainer.py`**: Trains individual SAEs on model activations.

### 🔹 **Feature Semantics**
- **`vis_in_one.py`**: Feature panel visualization, including semantic information.

### 🔹 **Feature Formation**
- **`umap_vis.py`**: UMAP visualization.
- **`act_dynamics.py`**: Computes activation space progress measures.
- **`feat_dynamics.py`**: Computes feature space progress measures.
- **`w_no_jaccard.py`**: Uses Jaccard similarity for progress measure.
- **`wjaccard_dynamics.py`**: Uses weighted Jaccard similarity for progress measure.

### 🔹 **Feature Drift**
- **`cos_analysis_feature_centric.py`**: Cosine similarity analysis focusing on features.
- **`cos_plot_ckpt.py`**: Cosine similarity visualization across checkpoints.
- **`traj.py`**: Analyzes trajectories of decoder vectors (`W_dec`).


