# SPDX-License-Identifier: LicenseRef-Genentech-NonCommercial-1.0
# Copyright (c) 2025 Genentech, Inc.
#
# Licensed under the Genentech Non-Commercial Software License Version 1.0 (September 2022).
# You may not use this file except in compliance with the License.
# A copy of the License is included in this directory as "docker/presage/LICENSE" and in built images at "/LICENSE".
#
# NOTICE OF MODIFICATIONS:
# This file was created or modified by the CellSimBench team to integrate PRESAGE with CellSimBench.
# See docker/presage/MODIFICATIONS.md for a summary of changes.
"""
PRESAGE model wrapper for CellSimBench integration.
Handles training and prediction with pre-computed knowledge embeddings.
"""

import logging
import json
import sys
import os
from pathlib import Path
from typing import Dict, List
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import tempfile
import time
import shutil

# VENDORED (V1): locate the upstream PRESAGE checkout. `presage.def` bakes the
# pinned commit at /presage_src and exports PRESAGE_SRC; the env var is the
# override for a local checkout. The atheus original also listed
# `/capstor/store/cscs/.../PRESAGE/src` (a CSCS Alps path) and `ref/PRESAGE/src`
# (a gitignored dir that never existed here) — both dropped. Falling through a
# dead candidate list to a bare ImportError three frames deeper is worse than
# saying which paths were tried.
_PRESAGE_SRC_CANDIDATES = [
    os.environ.get('PRESAGE_SRC', ''),
    '/presage_src',
]
for _p in _PRESAGE_SRC_CANDIDATES:
    if _p and os.path.exists(_p):
        sys.path.insert(0, _p)
        break
else:
    raise ImportError(
        "PRESAGE source not found. Run inside presage.sif (which bakes the "
        "pinned upstream at /presage_src) or set PRESAGE_SRC to a checkout. "
        f"Tried: {[p for p in _PRESAGE_SRC_CANDIDATES if p]}")

from model_harness import ModelHarness
from train import set_seed
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning import seed_everything
from presage import GeneEmbeddingTransformation, PrepareInputs, ItemNet, Pool
from torch import nn

# VENDORED (V2): drop the `cellsimbench` dependency, exactly as gears_wrapper.py
# did. The two symbols the atheus wrapper imported from it — a Path-aware JSON
# encoder and a slim data loader — are inlined below, so there is no host package
# to install and no separate harness module to bind-mount.
from presage_datamodule_csb import CellSimBenchDataModule


class PathEncoder(json.JSONEncoder):
    """JSON encoder that serializes ``pathlib.Path`` as ``str``."""

    def default(self, obj):
        if isinstance(obj, Path):
            return str(obj)
        return json.JSONEncoder.default(self, obj)


class DataManager:
    """Minimal data loader: read one combined h5ad from ``config['data_path']``.

    The wrapper uses only ``DataManager(config)`` and ``.load_dataset()``. The
    combined h5ad carries the per-cell split column and every covariate inline;
    the wrapper slices it itself (see docker/CONTRACT.md).
    """

    def __init__(self, dataset_config: Dict):
        self.config = dataset_config
        self.adata = None

    def load_dataset(self):
        path = Path(self.config["data_path"])
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")
        print(f"[harness] Loading dataset from {path} ...", flush=True)
        self.adata = sc.read_h5ad(path)
        self.adata.var.index.name = None
        print(f"[harness] Loaded AnnData with shape: {self.adata.shape}", flush=True)
        return self.adata

log = logging.getLogger(__name__)


# Names that should be rewritten to PRESAGE's canonical control label.
# Matched as whole strings (after `+`→`_` normalization) — substring matching
# would corrupt gene names that happen to contain "ctrl" (e.g. RctrlSEL).
_CONTROL_ALIASES = ("ctrl", "ctrl_iegfp")


def _normalize_perturbation(name: str) -> str:
    """Convert a CellSimBench perturbation label to PRESAGE format.

    Combo separator `+` is rewritten to `_`. Whole-string control aliases
    are rewritten to "control"; non-control names are left untouched.
    """
    name = name.replace("+", "_")
    if name in _CONTROL_ALIASES:
        return "control"
    return name


class ComboPRESAGE(nn.Module):
    """PRESAGE model variant with support for combination perturbations.
    
    This class extends the standard PRESAGE architecture to handle combination
    perturbations by aggregating embeddings for multiple genes in combo conditions.
    Uses standard MSE loss without any weight modulation.
    """
    def __init__(self, config, datamodule, input_dimension, output_dimension):
        super(ComboPRESAGE, self).__init__()

        # prepare variables
        self.batch_size = datamodule.batch_size
        self.config = config
        self.validation_step_outputs = []
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.datamodule = datamodule
        self.genes = datamodule.train_dataset.adata.var.index.to_numpy().ravel()
        self.n_nmf_embeddings = config["n_nmf_embedding"]

        self.num_genes = output_dimension
        self.pca_dim = config["pca_dim"]

        # Covariate conditioning: if the dataset emits a non-empty covariate
        # tensor, project it into the latent space and add to emb_h before
        # decoding. Adding (vs concatenating) keeps item_net's input dim
        # unchanged so we don't have to special-case its construction.
        # Stays a no-op when n_covariates == 0 (legacy single-context runs).
        self.n_covariates = int(config.get("n_covariates", 0))
        if self.n_covariates > 0:
            self.cov_embed = nn.Linear(self.n_covariates, config["item_hidden_size"])
        else:
            self.cov_embed = None


        # get gene embeddings
        self.gene_embeddings = PrepareInputs(datamodule, config)._prep_inputs()

        norms = np.linalg.norm(self.gene_embeddings, axis=1, keepdims=True)

        norms = np.median(norms, axis=(0, 1), keepdims=True)

        # fixing areas with norm of 0
        norms[norms == 0] = 1

        self.gene_embeddings = self.gene_embeddings / (norms)

        self.gene_embeddings = (
            torch.tensor(self.gene_embeddings).type(torch.float32).to(self._device)
        )

        self.mask = torch.sum(self.gene_embeddings, dim=1, keepdims=True) != 0

        if config["learnable_gene_embedding"]:
            self.learnable_embedding = nn.Embedding(
                self.gene_embeddings.shape[0], config["item_hidden_size"]
            )

        ngenes, _, n_pathways = self.gene_embeddings.shape

        self.activation = nn.LeakyReLU()
        self.temperature = config["softmax_temperature"]

        # Map from raw gene embeddings to aligned gene embeddings
        # pathway encoder
        # VENDORED (V12): honour the configured value instead of overwriting it.
        # The original assigned 4 here unconditionally, silently discarding
        # whatever _build_model_config had just put in the config.
        config["num_heads"] = config.get("num_heads", 4)  # hidden_dim must be divisible by num_heads
        pathway_encoder_function = "MLP"  # MLP MOE MHA

        self.pathway_encoder = GeneEmbeddingTransformation(
            self.gene_embeddings, config, pathway_encoder_function
        )

        # Pool type to perturbation latent space
        config["num_genes"] = self.pca_dim or self.num_genes  # self.num_genes
        self.pool = Pool(n_pathways, config)
        # self.pool.KG_weights = None

        # map from latent to output (logFC)
        self.item_net = ItemNet("MLP", self.pca_dim or self.num_genes, config)

    def forward_to_emb(self, locs_gene, locs_combos):
        """Override to handle combo perturbations by summing embeddings."""
        # Get the gene embeddings for the perturbed genes
        emb = self.gene_embeddings[locs_gene, :, :]
        mask = self.mask[locs_gene, :, :]
        
        # Map KG specific embeddings to KG shared latent space
        emb_h = self.pathway_encoder(emb)
        emb_h = self.activation(emb_h)
        
        # Handle combo perturbations by aggregating embeddings for the same sample
        # Get unique sample indices
        unique_combos, inverse_indices = torch.unique(locs_combos, return_inverse=True)
        
        # Aggregate embeddings for each unique sample
        aggregated_emb_h = []
        aggregated_mask = []
        
        for i, combo_idx in enumerate(unique_combos):
            # Find all embeddings for this sample
            sample_mask = (locs_combos == combo_idx)
            sample_embeddings = emb_h[sample_mask]
            sample_masks = mask[sample_mask]
            
            # Sum embeddings for combo perturbations
            # This models the combined effect as additive in embedding space
            summed_embedding = sample_embeddings.sum(dim=0, keepdim=True)
            summed_mask = sample_masks.sum(dim=0, keepdim=True)
            
            aggregated_emb_h.append(summed_embedding)
            aggregated_mask.append(summed_mask)
        
        # Concatenate all aggregated embeddings
        emb_h_aggregated = torch.cat(aggregated_emb_h, dim=0)
        mask_aggregated = torch.cat(aggregated_mask, dim=0)
        
        # Normalize mask to 0 or 1
        mask_aggregated = (mask_aggregated > 0).float()
        
        # Store for potential visualization/debugging
        self.emb_h = emb_h_aggregated
        
        # Create new locs_combos that matches the aggregated embeddings
        # This ensures the pool function gets the right indices
        new_locs_combos = torch.arange(len(unique_combos), device=locs_combos.device)
        
        # Pool to latent space
        emb_h_final = self.pool(emb_h_aggregated, new_locs_combos, mask_aggregated)
        
        # Handle pool attributes
        if hasattr(self.pool.pool, "p_weight_vec"):
            self.pathway_weight_vector = self.pool.pool.p_weight_vec
        if hasattr(self.pool.pool, "attention_weights"):
            self.attention_weights = self.pool.pool.attention_weights
        
        return emb_h_aggregated, emb_h_final


    def emb_to_out(self, emb_h, locs_gene, locs_combos):

        # transformation to output dimensions uses ItemNet class
        out = self.item_net(emb_h)
        
        if self.pca_dim is not None:
            out = self.pca.inverse_transform(out)


        return out
    
    def forward(self, pert_inds, cov=None, update_node_embeddings=True):
        """Override forward to handle combo perturbations properly."""
        self.pert_inds = pert_inds
        locs = torch.nonzero(pert_inds)

        # indices of perturbed genes
        locs_gene = locs[:, 1]
        # indices of perturbations within a batch
        locs_combos = locs[:, 0]

        self.locs_gene = locs_gene
        self.locs_combos = locs_combos

        # Get embeddings with combo handling
        emb_h_temp, emb_h = self.forward_to_emb(locs_gene, locs_combos)

        # Inject covariate conditioning. cov is per-batch-row (one row = one
        # (covariate, perturbation) sample), but emb_h is aggregated to
        # per-unique-combo. torch.unique preserves first-occurrence order, so
        # cov[unique_combos] gives one cov vector per aggregated row.
        if self.cov_embed is not None and cov is not None and cov.numel() > 0 and cov.shape[-1] > 0:
            unique_combos = torch.unique(locs_combos)
            combo_cov = cov[unique_combos]
            emb_h = emb_h + self.cov_embed(combo_cov)

        # For emb_to_out, we need to pass the aggregated indices
        # Create dummy locs that match the aggregated embeddings
        unique_combos = torch.unique(locs_combos)
        dummy_locs_gene = torch.zeros_like(unique_combos)
        dummy_locs_combos = torch.arange(len(unique_combos), device=locs_combos.device)

        # Latent space to output dimension
        out = self.emb_to_out(emb_h, dummy_locs_gene, dummy_locs_combos)

        return out, emb_h_temp, "None"
    



    def compute_loss(self, pred, expr, tensor, pred_clust=None, expr_clust=None):
        """Compute standard MSE loss."""
        residual2 = (pred - expr) ** 2
        loss = residual2.mean()
        return loss

class CustomProgressBar(TQDMProgressBar):
    """Enhanced progress bar with better formatting."""
    
    def get_metrics(self, trainer, model):
        # Get the default metrics
        items = super().get_metrics(trainer, model)
        
        # Format losses with more decimal places and compact display
        if 'train_loss' in items:
            items['train_loss'] = f"{items['train_loss']:.6f}"
        if 'val_loss' in items:
            items['val_loss'] = f"{items['val_loss']:.6f}"
        
        return items

class CustomEarlyStopping(EarlyStopping):
    """Custom early stopping that displays losses with more precision."""
    
    def _improvement_message(self, current: float) -> str:
        """Generate an improvement message with more decimal places."""
        if self.best_score is None:
            return f"Metric {self.monitor} improved. New best score: {current:.6f}"
        else:
            improvement = abs(self.best_score - current)
            return (f"Metric {self.monitor} improved by {improvement:.6f} >= "
                   f"min_delta = {self.min_delta:.6f}. New best score: {current:.6f}")

class CellSimBenchModelHarness(ModelHarness):
    """Custom ModelHarness that skips evaluator initialization for CellSimBench."""
    
    def __init__(self, module, datamodule, config, encoder=None, decoder=None):
        # Initialize base Lightning module
        pl.LightningModule.__init__(self)
        
        # Copy necessary initialization from ModelHarness
        self.module = module
        self.module.current_batch = 0
        self.var_names = datamodule.var_names
        self.degs = datamodule.degs
        self.config = config
        
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.train_step_outputs = []
        self.test_set_keys = getattr(datamodule, "test_set_keys", [""])
        self.encoder = encoder
        self.decoder = decoder
        
        datamodule.encoder = encoder
        self.do_test_eval = False  # Disable test evaluation
        
        # Skip evaluator initialization - not needed for CellSimBench
        self.evaluator = None
        self.second_evaluator = None
        
        # Always set train_perturb_labels to None to avoid classification paths
        self.train_perturb_labels = None
        
        # Initialize model tracking attributes
        self.all_embh = []
        self.all_coef = []
        self.attention_weights = []
        self.transformed_embs = []
        self.all_locs_gene = []
        self.all_locs_ind = []
    
    def _step(self, batch, batch_idx):
        """Standard training step."""
        src, cov, tgt = self.unpack_batch(batch)
        
        pred, tensor, pred_clust = self(src, cov)
        loss = self.module.compute_loss(pred, tgt, tensor)
        return loss, None, None, src, pred
    
    def training_step(self, batch, batch_idx):
        """Custom training step with better logging."""
        self.module.current_batch += 1
        loss, ce_loss, accuracy, src, pred = self._step(batch, batch_idx)
        
        # Log with progress bar and on_step for real-time updates
        self.log("train_loss", loss, 
                prog_bar=True,   # Show in progress bar
                on_step=True,    # Log at each step
                on_epoch=True)   # Also log epoch average
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Custom validation step with better logging."""
        loss, ce_loss, accuracy, src, pred = self._step(batch, batch_idx)
        
        # Log validation loss with sync_dist for multi-GPU
        self.log("val_loss", loss,
                prog_bar=True,   # Show in progress bar
                on_step=False,   # Don't log each step
                on_epoch=True,   # Log epoch average
                sync_dist=True)  # Sync across devices if needed
        
        self.validation_step_outputs.append(
            {"src": src, "tgt": batch["expr"], "pred": pred}
        )
        return loss
    
    def on_validation_epoch_end(self):
        """Override to skip evaluator calls."""
        self.module.current_batch = 0
        self.validation_step_outputs.clear()
    
    def on_test_epoch_end(self):
        """Override to skip evaluator calls."""  
        self.test_step_outputs.clear()
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """Override to handle predictions."""
        src, cov, tgt = self.unpack_batch(batch)
        preds, tensor, pred_clust = self(src, cov)

        # Store embeddings and other info for visualization
        if hasattr(self.module, "emb_h"):
            self.all_embh.append(self.module.emb_h.detach().cpu().numpy())
        self.all_locs_gene.append(self.module.locs_gene.detach().cpu().numpy())
        self.all_locs_ind.append(self.module.locs_combos.detach().cpu().numpy())

        if hasattr(self.module, "pathway_weight_vector"):
            self.pathway_weight_vector = self.module.pathway_weight_vector
        if hasattr(self.module, "attention_weights"):
            self.attention_weights.append(self.module.attention_weights)

        self.transformed_embs.append(self.module.emb_h.cpu().numpy())

        if preds is not None:
            if self.decoder is not None:
                preds = self.decoder(preds.cpu())

            # Get keys from the batch - REQUIRED
            if 'pert_key' not in batch:
                raise KeyError("Batch MUST contain 'pert_key' with perturbation format")

            # Extract the perturbation keys from the batch
            keys = batch['pert_key']
            # Convert to numpy array if it's a list
            if isinstance(keys, list):
                keys = np.array(keys)

            # Covariate labels per row (empty strings in legacy single-context mode).
            cov_labels = batch.get('cov_label', None)
            if cov_labels is None:
                cov_labels = np.array([""] * len(keys))
            elif isinstance(cov_labels, list):
                cov_labels = np.array(cov_labels)

            return keys, preds, cov_labels

class PRESAGEWrapper:
    """Main wrapper class for PRESAGE model integration."""
    
    def __init__(self, config: Dict):
        self.config = config
        
        # Set up paths
        self.data_path = config['data_path']
        # TODO: Implement model path for continuing training in the future
        self.model_path = config.get('model_path', None)
        
        # Parse hyperparameters
        self.hyperparams = config['hyperparameters']
        
        # Create DataManager instance like GEARS does
        self.data_manager = DataManager(self.config)
        self.data_manager.load_dataset()

        # VENDORED (V3): read the seed from the TOP-LEVEL config, not from the
        # hyperparameters block. Our contract emits it at the top level for both
        # train and predict (docker/CONTRACT.md; _container/config.py), so the
        # original would have raised — or, worse, had a seed been left in the
        # recipe's hyperparameters, pinned every run to it regardless of what the
        # predictor asked for, turning a seed-stability check into a measurement of
        # CUDA nondeterminism. gears_wrapper.py fixes the same defect as F1.
        self.seed = int(self.config['seed'])
        
        # Set random seed for reproducibility using both methods
        set_seed(self.seed)  # PRESAGE's comprehensive seed setting
        seed_everything(self.seed, workers=True)  # PyTorch Lightning's seed setting
        
        log.info(f"Initialized PRESAGE wrapper with seed={self.seed}")
    
    def _seed_cache(self, work_dir: Path) -> Path:
        """Give PRESAGE a writable ``./cache`` in ``work_dir`` without copying 6.4 GB.

        VENDORED (V4): upstream PRESAGE hardcodes ``./cache/pathway_embeddings/``
        relative to the CWD and WRITES the embeddings it derives back into it. The
        atheus wrapper left training at whatever CWD the job was launched from, so
        concurrent folds raced on one directory (they hit it hard enough that
        `predict` grew a tempdir-chdir workaround for exactly this, symptom
        "pickle data was truncated"). Anchoring the cache in the run dir removes
        the race by construction: every fold has its own ``/model_output``.

        Built as a **symlink farm** — real directories, symlinked files — rather
        than a copy. The baked cache is 6.4 GB; copying it per fold would cost
        ~64 GB across a ten-fold sweep on a volume that is already full, to
        duplicate bytes that never change. PRESAGE only ever *adds* files here
        (its embeddings are keyed by source+dataset, and ours are named after our
        h5ad), so new files land as real files in the run dir while every
        pre-existing one costs nothing.

        If PRESAGE ever did try to overwrite a baked file, the write fails with
        EROFS against the read-only image rather than silently corrupting the
        shared cache — a loud failure, which is the one we want.

        This also removes the writable-overlay the atheus singularity runner
        needed: ``/model_output`` is already a read-write bind mount.
        """
        cache_dir = work_dir / "cache"
        if cache_dir.exists():
            log.info(f"PRESAGE cache already present at {cache_dir}")
            return cache_dir
        source = Path(self.config.get("presage_cache_path", "/opt/presage_cache"))
        if not source.exists():
            raise FileNotFoundError(
                f"PRESAGE pathway-embedding cache not found at {source}. It is baked "
                f"into presage.sif from the upstream Zenodo release; set "
                f"`presage_cache_path` in docker/presage/model.yaml if it moved.")

        n_dirs = n_links = 0
        for src_dir, _dirnames, filenames in os.walk(source):
            dst_dir = cache_dir / Path(src_dir).relative_to(source)
            dst_dir.mkdir(parents=True, exist_ok=True)
            n_dirs += 1
            for name in filenames:
                (dst_dir / name).symlink_to(Path(src_dir) / name)
                n_links += 1
        log.info(f"PRESAGE cache seeded at {cache_dir}: {n_dirs} dirs, "
                 f"{n_links} files symlinked from {source} (no bytes copied)")
        return cache_dir

    def train(self):
        """Train PRESAGE model using PyTorch Lightning."""
        log.info("Starting PRESAGE training...")
        
        # Re-set seeds at the start of training for reproducibility
        set_seed(self.seed)
        seed_everything(self.seed, workers=True)
        log.info(f"Training with seed={self.seed}")
        
        # Create output directory
        self.output_dir = Path(self.config['output_dir']).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        log.info(f"Training output directory: {self.output_dir}")

        # VENDORED (V4): run from the output dir so PRESAGE's hardcoded `./cache`
        # IS `<run_dir>/cache` — the same place _save_training_artifacts wants it
        # and the same place predict restores it from. Every other path in the
        # config is absolute, so the chdir affects nothing else.
        self._seed_cache(self.output_dir)
        original_cwd = os.getcwd()
        os.chdir(self.output_dir)
        log.info(f"Train chdir: {original_cwd} -> {self.output_dir}")
        try:
            # 1. Load and convert data
            adata = self._load_and_convert_data()

            # 2. Create PRESAGE data module
            datamodule = self._create_datamodule(adata)

            # 3. Build model configuration
            model_config = self._build_model_config(datamodule)

            # 4. Initialize model with combo support
            model = ComboPRESAGE(model_config, datamodule,
                                 input_dimension=len(datamodule.train_dataset.adata.var),
                                 output_dimension=len(datamodule.train_dataset.adata.var))

            log.info("Using ComboPRESAGE with combo support")

            # 5. Create model harness for training
            harness = CellSimBenchModelHarness(model, datamodule, model_config)

            # 6. Set up trainer
            trainer = self._create_trainer()

            # 7. Train model
            started = time.time()
            trainer.fit(harness, datamodule)

            # 8. Save model and metadata
            self._save_training_artifacts(harness, model_config, trainer,
                                          wall_seconds=time.time() - started)
        finally:
            os.chdir(original_cwd)

        log.info("Training completed successfully")
    
    def predict(self):
        """Generate predictions using trained PRESAGE model."""
        log.info("Starting PRESAGE prediction...")

        # PRESAGE hardcodes its cache path as `./cache/pathway_embeddings/`,
        # so multiple concurrent predict jobs sharing the same cwd race on
        # rmtree+copytree of that directory and corrupt each other's pickles
        # ("pickle data was truncated"). Isolate this process by chdir'ing to
        # a unique tempdir for the duration of the prediction; absolute paths
        # (output_path, model_path, data_path) are unaffected.
        original_cwd = os.getcwd()
        isolated_cwd = tempfile.mkdtemp(prefix="presage_predict_")
        os.chdir(isolated_cwd)
        log.info(f"Predict chdir: {original_cwd} -> {isolated_cwd}")
        try:
            # 1. Load trained model
            harness, datamodule, control_mean = self._load_trained_model()

            # 2. Get test conditions
            test_conditions = self.config['test_conditions']

            # 3. Generate predictions (Note: _generate_predictions loads data internally)
            predictions_adata = self._generate_predictions(harness, datamodule, control_mean,
                                                           test_conditions)

            if predictions_adata is None:
                raise RuntimeError("Failed to generate predictions")

            # 4. Save predictions
            output_path = self.config['output_path']
            log.info(f"Saving predictions to {output_path}")
            predictions_adata.write_h5ad(output_path)

            log.info("Prediction completed successfully")
        finally:
            os.chdir(original_cwd)
            try:
                shutil.rmtree(isolated_cwd, ignore_errors=True)
            except Exception as e:
                log.warning(f"Failed to clean up tempdir {isolated_cwd}: {e}")
    
    def _load_and_convert_data(self):
        """Load and convert CellSimBench data to PRESAGE format."""
        log.info(f"Loading data from {self.data_path}")
        adata = sc.read_h5ad(self.data_path)
        
        # Convert to PRESAGE format
        adata = self._convert_to_presage_format(adata)
        
        log.info(f"Loaded and converted data: {adata.shape}")
        return adata
    
    def _convert_to_presage_format(self, adata_raw):
        """Convert CellSimBench format to PRESAGE format."""
        adata = adata_raw.copy()

        # If 'perturbation' is already a column, drop it
        if 'perturbation' in adata.obs.columns and 'condition' in adata.obs.columns:
            adata.obs.drop(columns=['perturbation'], inplace=True)
        
        # Rename condition column if needed
        if 'condition' in adata.obs.columns:
            adata.obs['perturbation'] = adata.obs['condition'].copy()
        
        # Convert perturbation names: "GENE1+GENE2" -> "GENE1_GENE2",
        # and any whole-string control alias (e.g. "ctrl", "ctrl_iegfp") -> "control".
        adata.obs["perturbation"] = adata.obs["perturbation"].astype(str).map(_normalize_perturbation)
        
        # Add nperts column
        # Calculate number of perturbations per cell using vectorized operations
        perturbations = adata.obs["perturbation"].astype(str)
        
        # Create boolean mask for control conditions
        is_control = perturbations.isin(["control", "ctrl", "ctrl_iegfp", "nan"])
        
        # Count underscores for non-control conditions
        underscore_counts = perturbations.str.count("_")
        
        # Assign nperts: 0 for controls, 1 + underscore_count for others
        adata.obs["nperts"] = np.where(is_control, 0, 1 + underscore_counts)
        
        # VENDORED (V8): assert gene-name uniqueness instead of repairing it.
        # The original called `var_names_make_unique()`, which renames a collision
        # to `GENE-1`/`GENE-2`. Those renamed symbols match nothing in
        # `store.gene_names`, so the host's gene intersection
        # (_container/tensor_map.py) drops them — silently, because its duplicate
        # guard fires on duplicates, not on this rename. Checked across every
        # processed dataset (adamson16, frangieh21, jiang24, norman19, replogle20,
        # replogle22, sunshine23, wessels23, xatlas_orion, ecoli_synthetic): zero
        # duplicate gene names, and where a `gene_name` column exists it is
        # identical to the index. So the rename can only ever fire on a dataset
        # regression, which is exactly when we want to hear about it.
        if 'gene_name' not in adata.var.columns:
            adata.var['gene_name'] = adata.var.index.values
        adata.var.index.name = None
        adata.var = adata.var.reset_index().set_index("gene_name")

        dupes = adata.var.index[adata.var.index.duplicated()].unique().tolist()
        if dupes:
            raise ValueError(
                f"{len(dupes)} duplicate gene name(s) in var (e.g. {dupes[:5]}). "
                f"Renaming them would silently drop those genes from the benchmark "
                f"gene axis — fix the gene axis in the dataset build instead.")
        
        # Ensure X is dense array.
        # VENDORED (V18): `copy=False` on the astype. The original always copied,
        # so for an already-float32 input this allocated a SECOND full dense array
        # for no benefit. On replogle22 (504,932 x 7,226, 44.5% dense, 1.62e9 nnz)
        # one dense float32 copy is 14.6 GB, and the sparse CSR it is built from is
        # another 13 GB — job 9532031 was OOM-killed at 96 GB right here. The
        # stored dtype IS float32 (verified in the h5ad), so this copy was pure
        # waste; where the input is not float32 the conversion still happens.
        # This does NOT make the load cheap: `load_preprocessed` densifies the
        # whole input a second time on read, which is inherent to the datamodule
        # and is what sets --mem. See PRESAGE_NOTES.md section 4 (Memory).
        if hasattr(adata.X, 'toarray'):
            adata.X = adata.X.toarray()

        adata.X = adata.X.astype(np.float32, copy=False)
        
        log.info("Converted data to PRESAGE format")
        return adata
    
    def _create_datamodule(self, adata):
        """Create custom CellSimBench data module for PRESAGE."""
        # Create output directory structure
        data_dir = self.output_dir / "presage_data"
        data_dir.mkdir(exist_ok=True)

        # Save processed data.
        # VENDORED (V15): compressed. `_convert_to_presage_format` densifies X
        # (and `load_preprocessed` densifies again on read), so this file is a
        # dense float32 copy of the whole input — 2.7 GB for adamson16, and it is
        # written once PER FOLD into a run dir on a volume that is already full.
        # Uncompressed it scales to ~13.6 GB for replogle22 and ~32 GB for
        # jiang24. gzip costs a little CPU once and changes nothing else.
        processed_path = data_dir / "cellsimbench_processed.h5ad"
        print(f"Saving processed data to {processed_path}...")
        adata.write_h5ad(processed_path, compression="gzip")

        # Resolve which obs column (if any) to use for covariate conditioning.
        # `covariate_key` is the single canonical covariate the host sends
        # (_container/config.py); a column with <2 unique values auto-disables the
        # cov path, matching the single-context behaviour.
        #
        # VENDORED (V13): read `covariate_key` ONLY. The original fell back
        # through `self.config.get('covariate_field', ...)`, a key the host
        # deleted in F4. Nothing rejects it any more: it is not in
        # `_KNOWN_FIELDS`, so `_check_extras` waves it through as a scalar extra,
        # and a stray `covariate_field:` under `extra_config:` in model.yaml would
        # silently override the canonical covariate — on the one model whose whole
        # claim rests on that covariate. Rejecting it host-side was the obvious
        # alternative and is worse: it would put a dead model-specific key back
        # into the shared schema, which is precisely what `_KNOWN_FIELDS` exists
        # to prevent. Deleting the reader is the fix that leaves nothing to guard.
        cov_field = self.config.get('covariate_key')
        if cov_field is not None:
            if cov_field not in adata.obs.columns:
                log.warning(
                    f"covariate column {cov_field!r} not in adata.obs.columns; "
                    "disabling cov plumbing for this run."
                )
                cov_field = None
            elif adata.obs[cov_field].nunique() < 2:
                log.info(
                    f"covariate column {cov_field!r} has <2 unique values "
                    "({adata.obs[cov_field].nunique()}); disabling cov plumbing."
                )
                cov_field = None
        self._covariate_field_resolved = cov_field
        
        # Create splits from config
        splits = {'train': [], 'val': [], 'test': []}
        
        for split_name, conditions in [
            ('train', self.config['train_conditions']),
            ('val', self.config['val_conditions']),
            ('test', self.config['test_conditions'])
        ]:
            for cond in conditions:
                # Convert condition name to PRESAGE format (whole-string match;
                # see _normalize_perturbation re: substring-replace bug).
                splits[split_name].append(_normalize_perturbation(cond))
        
        # Save splits
        splits_path = data_dir / "splits.json"
        with open(splits_path, 'w') as f:
            json.dump(splits, f)
        

        
        # Create datamodule config
        datamodule_config = {
            'processed_adata_path': str(processed_path),
            'splits_path': str(splits_path),
            'dataset': 'cellsimbench',
            'data_dir': str(data_dir),
            'batch_size': self.hyperparams['batch_size'],
            'use_pseudobulk': True,
            'preprocessing_zscore': False,
            'noisy_pseudobulk': False,
            'perturb_field': 'perturbation',
            'control_key': 'control',
            'split_name': self.config['split_name'],
            'seed': self.seed,  # Pass seed for reproducibility
            # VENDORED (V7): pinned False, not a tunable — its inference half was
            # never implemented (see _load_trained_model), so training against the
            # perturbation mean would have predicted against the control mean.
            'perts_as_delta_ref': False,
            # Optional cov plumbing — resolved above; will be None for legacy
            # single-context datasets, in which case CellSimBenchDataModule
            # falls back to the original behaviour.
            'covariate_field': self._covariate_field_resolved,
        }

        # Create our custom datamodule
        datamodule = CellSimBenchDataModule.from_config(datamodule_config)

        # Prepare and setup data
        datamodule.prepare_data()
        datamodule.setup("fit")

        # VENDORED (V17): clamp the batch size to the training split.
        # Upstream's train_dataloader uses drop_last=True
        # (upstream datamodule.py:642), so a batch larger than the split yields
        # `len(loader) == 0` and Lightning exits with "No training batches" —
        # having trained on nothing. The authors' batch_size (256) was swept on
        # replogle_k562_essential_unfiltered, which has thousands of pseudobulk
        # rows; adamson16's training split has 63, so 63 // 256 == 0.
        #
        # Clamping means "full batch" where the configured size exceeds the data,
        # which is the only sensible reading of a batch larger than the dataset.
        # It binds ONLY where the authors' value cannot apply: replogle22
        # UnseenBoth has 3386 rows and is untouched. The effective value is logged
        # and recorded in the training report, so a run never hides which batch
        # size it actually used.
        n_train = len(datamodule.train_dataset)
        configured_bs = int(self.hyperparams['batch_size'])
        effective_bs = min(configured_bs, n_train)
        if effective_bs != configured_bs:
            log.warning(
                f"batch_size {configured_bs} exceeds the {n_train}-row training "
                f"split; clamping to {effective_bs} (full batch). Upstream's "
                f"drop_last=True would otherwise give ZERO training batches.")
            datamodule.batch_size = effective_bs
        self._effective_batch_size = effective_bs

        log.info(
            f"Created CellSimBench datamodule with batch_size={self.hyperparams['batch_size']}, "
            f"covariate_field={datamodule_config['covariate_field']!r}, "
            f"cov_categories={getattr(datamodule, 'cov_categories', None)}"
        )
        return datamodule
    
    def _build_model_config(self, datamodule):
        """Build model configuration from hyperparameters."""
        # PRESAGE hardcodes cache to ./cache/pathway_embeddings/
        # We need to ensure the cache exists there
        cache_path = "./cache"
        
        # VENDORED (V1): PRESAGE's prior-knowledge manifest ships in the upstream
        # checkout's sample_files/. Same candidate list as the import above (CSCS
        # and ref/ dropped), and a hard failure instead of the original's silent
        # fallthrough to a nonexistent ref/PRESAGE — which turned a missing
        # checkout into a FileNotFoundError on the pathway file two lines later.
        for _candidate in _PRESAGE_SRC_CANDIDATES:
            root = _candidate[:-4] if _candidate.endswith('/src') else _candidate
            if root and os.path.exists(os.path.join(root, 'sample_files')):
                sample_files_path = root
                break
        else:
            raise FileNotFoundError(
                "PRESAGE sample_files/ not found under "
                f"{[c for c in _PRESAGE_SRC_CANDIDATES if c]} — the image bakes it "
                "at /presage_src/sample_files (see presage.def).")
        
        # Create a temporary pathway file with corrected absolute paths
        original_pathway_file = f'{sample_files_path}/sample_files/prior_files/sample.knowledge_experimental.txt'
        temp_pathway_file = self.output_dir / 'pathway_files_absolute.txt'
        
        with open(original_pathway_file, 'r') as f_in:
            lines = f_in.readlines()
        
        # Replace relative paths with absolute paths to our cache
        # PRESAGE expects cache at ./cache/ relative to where it runs
        corrected_lines = []
        for line in lines:
            line = line.strip()
            if line and line.startswith('../cache/'):
                # Replace ../cache/ with ./cache/ since PRESAGE uses ./cache/
                corrected_path = line.replace('../cache/', './cache/')
                corrected_lines.append(corrected_path + '\n')
            else:
                corrected_lines.append(line + '\n' if line else '\n')
        
        with open(temp_pathway_file, 'w') as f_out:
            f_out.writelines(corrected_lines)
        
        log.info(f"Created temporary pathway file with corrected paths: {temp_pathway_file}")
        
        config = {
            # Architecture parameters
            'item_hidden_size': self.hyperparams['item_hidden_size'],
            'item_nlayers': self.hyperparams['item_nlayers'],
            'pathway_item_hidden_size': self.hyperparams['pathway_item_hidden_size'],
            'pathway_item_nlayers': self.hyperparams['pathway_item_nlayers'],
            
            # Pooling configuration
            'pathway_pool_type': self.hyperparams['pathway_pool_type'],
            'pathway_weight_type': self.hyperparams['pathway_weight_type'],
            'pool_nlayers': self.hyperparams['pool_nlayers'],
            'softmax_temperature': self.hyperparams['softmax_temperature'],
            'gat_weight': self.hyperparams['gat_weight'],
            
            # Embeddings
            'n_nmf_embedding': self.hyperparams['n_nmf_embedding'],
            
            # Knowledge source files (use temp file with absolute paths)
            'pathway_files': str(temp_pathway_file),
            'embedding_files': 'None',  # No embedding file provided
            
            # Pre-computed embeddings from cache - PRESAGE expects them at ./cache/pathway_embeddings/
            'pathway_embedding_dimension': self.hyperparams['pathway_embedding_dimension'],
            'pathway_embedding_files': './cache/pathway_embeddings/',
            
            # Node2Vec parameters
            'node2vec_walk_length': self.hyperparams['node2vec_walk_length'],
            'node2vec_context_size': self.hyperparams['node2vec_context_size'],
            'node2vec_walks_per_node': self.hyperparams['node2vec_walks_per_node'],
            'node2vec_num_negative_samples': self.hyperparams['node2vec_num_negative_samples'],
            'node2vec_p': self.hyperparams['node2vec_p'],
            'node2vec_q': self.hyperparams['node2vec_q'],
            # VENDORED (V16): its own hyperparameter, not an alias of the
            # training batch size. Upstream treats them as independent
            # (`--model.node2vec_batchsize`, default 32, and 32 again in the
            # authors' sweep); the atheus wrapper tied them together, so our
            # batch_size=32 only made this correct by coincidence and any change
            # to batch_size would have silently retuned the node2vec walk sampler
            # too.
            'node2vec_batchsize': self.hyperparams['node2vec_batchsize'],
            
            # Loss scales
            'mse_loss_scale': self.hyperparams['mse_loss_scale'],
            'cosine_loss_scale': self.hyperparams['cosine_loss_scale'], 
            'vector_norm_loss_scale': self.hyperparams['vector_norm_loss_scale'],
            
            # VENDORED (V12): these were hardcoded in the wrapper body, invisible
            # to any config, so `model.yaml` could not honestly claim to be the
            # single source of truth for how the model is trained. Lifted into
            # `hyperparameters` — the same defect as GEARS' silently-halved epochs.
            'batch_norm': self.hyperparams['batch_norm'],
            'n_neigh_prune': self.hyperparams['n_neigh_prune'],
            # Structural choices of this integration, not tunables: PRESAGE is fed
            # prior gene embeddings reduced by Node2Vec, un-PCA'd and frozen.
            # Changing any of these is a different model, not a different run.
            'learnable_gene_embedding': False,
            'pca_dim': None,
            'input_preparation': 'prep_gene_embeddings',
            'dim_red_alg': 'Node2Vec',
            # Make the per-dataset combined-embedding cache key unique per
            # dataset. PRESAGE builds the cache filename from source+dataset
            # (presage.py:read_and_embed); hardcoding both to "cellsimbench"
            # caused datasets with different gene sets to share a single
            # ./cache/pathway_embeddings/WeightedDeepset.*.embeddings.pkl,
            # leading to np.concatenate failures (e.g. 8247 vs 8321 genes).
            # The expensive node2vec-on-KG cache is keyed by the KG file
            # itself (presage.py:node_2_vec) and still shared, as intended.
            'dataset': Path(self.data_path).stem,
            'source': Path(self.data_path).stem,
            'num_heads': self.hyperparams['num_heads'],

            # Optimizer parameters (required by ModelHarness) — see V12.
            'lr': self.hyperparams['lr'],
            'weight_decay': self.hyperparams['weight_decay'],
            'momentum': self.hyperparams['momentum'],
            'optimizer': self.hyperparams['optimizer'],

            # Number of covariate categories the model conditions on. 0 disables
            # cov_embed (legacy single-context behaviour). Read from the
            # datamodule which derives it from the dataset's adata.obs[cov_field].
            'n_covariates': len(getattr(datamodule, 'cov_categories', None) or []),
        }

        log.info(
            f"Built model configuration with pre-computed knowledge embeddings; "
            f"n_covariates={config['n_covariates']}"
        )
        return config
    
    def _create_trainer(self):
        """Create PyTorch Lightning trainer."""
        # Set up callbacks
        checkpoint_callback = ModelCheckpoint(
            dirpath=self.output_dir / "checkpoints",
            filename='presage-{epoch:02d}-{val_loss:.6f}',  # More decimal places
            monitor='val_loss',
            save_top_k=1,
            mode='min'
        )
        
        # Patience is read from hyperparameters (default 10 — backwards-compatible).
        early_stop_callback = CustomEarlyStopping(
            monitor='val_loss',
            # VENDORED (V12): required, not defaulted. Early stopping decides
            # which epoch is scored, so a silent default is a training decision
            # nobody wrote down.
            patience=int(self.hyperparams['early_stopping_patience']),
            verbose=True,
            mode='min',
            min_delta=float(self.hyperparams['early_stopping_min_delta']),
        )
        
        progress_bar = CustomProgressBar(refresh_rate=10)  # Use custom progress bar

        # VENDORED (V14): wire W&B, which `logger=False` had hardcoded off.
        # docker/presage/model.yaml declares `wandb_project`, and
        # ContainerPredictor._train injects wandb/wandb_project/wandb_run into
        # config.json — all of which this wrapper silently ignored. A declared
        # setting that does nothing is worse than no setting: it made the training
        # card's Gate A ("per-run training dynamics from W&B") impossible while
        # looking configured. gears_wrapper.py wires the same three keys.
        #
        # The endpoint and credentials come from WANDB_BASE_URL / WANDB_API_KEY in
        # the job environment; singularity passes the host env through (we do not
        # use --cleanenv), so the sbatch script is where they are set.
        logger = False
        if self.config.get('wandb'):
            # Fail SAFE, not open. Wiring W&B (V14) turned a dead tracking server
            # from harmless into a hazard: an online logger with no reachable
            # endpoint or no credentials blocks on login in a batch job with no
            # tty, and would take a multi-hour training run down with it. So if we
            # cannot see credentials, log OFFLINE rather than gamble — the metrics
            # still land (train has chdir'd into the run dir, so they appear at
            # <run_dir>/wandb/offline-run-*, beside the checkpoint they describe)
            # and `wandb sync <dir>` uploads them whenever the server is back.
            # An explicit WANDB_MODE always wins.
            if not os.environ.get('WANDB_MODE'):
                if not os.environ.get('WANDB_API_KEY'):
                    os.environ['WANDB_MODE'] = 'offline'
                    log.warning("W&B: no WANDB_API_KEY in the environment — forcing "
                                "WANDB_MODE=offline so an unreachable server cannot "
                                "stall training. Sync later with `wandb sync`.")
            from pytorch_lightning.loggers import WandbLogger
            logger = WandbLogger(
                project=self.config.get('wandb_project', 'presage-ct'),
                name=self.config.get('wandb_run', 'presage_training'),
            )
            log.info(f"W&B logging ON: project={self.config.get('wandb_project')!r} "
                     f"run={self.config.get('wandb_run')!r} "
                     f"mode={os.environ.get('WANDB_MODE', 'online')} "
                     f"endpoint={os.environ.get('WANDB_BASE_URL', '(default cloud)')}")
        else:
            log.info("W&B logging OFF (config['wandb'] not set)")

        trainer = pl.Trainer(
            max_epochs=self.hyperparams['max_epochs'],
            callbacks=[checkpoint_callback, early_stop_callback, progress_bar],
            accelerator='gpu' if torch.cuda.is_available() else 'cpu',
            devices=1,
            logger=logger,
            enable_checkpointing=True,
            deterministic=True,  # Ensures reproducible results
            benchmark=False  # Disable cudnn.benchmark for reproducibility
        )
        
        log.info(f"Created trainer with max_epochs={self.hyperparams['max_epochs']}")
        return trainer
    
    def _save_training_artifacts(self, harness, model_config, trainer,
                                 wall_seconds: float = 0.0):
        """Save the trained model and the metadata predict needs to reload it."""
        # VENDORED (V5): save the BEST-VALIDATION weights, not the last epoch's.
        # The original saved `harness.state_dict()` straight after `trainer.fit`,
        # leaving the ModelCheckpoint(monitor='val_loss', save_top_k=1) file in
        # checkpoints/ unused — so with EarlyStopping(patience=N) the weights that
        # got scored were, by construction, N epochs past the validation optimum.
        # (GEARS already selects on best-val; this makes PRESAGE comparable.)
        best_epoch, best_val_loss = None, None
        ckpt_cb = getattr(trainer, "checkpoint_callback", None)
        best_path = getattr(ckpt_cb, "best_model_path", "") if ckpt_cb else ""
        if best_path and Path(best_path).exists():
            best = torch.load(best_path, map_location="cpu", weights_only=False)
            harness.load_state_dict(best["state_dict"])
            best_epoch = best.get("epoch")
            score = getattr(ckpt_cb, "best_model_score", None)
            best_val_loss = float(score) if score is not None else None
            log.info(f"Restored best-val weights from {best_path} "
                     f"(epoch={best_epoch}, val_loss={best_val_loss})")
        else:
            raise RuntimeError(
                "no best-validation checkpoint was written — refusing to save "
                "last-epoch weights silently. Check that val_loss was logged and "
                "that ModelCheckpoint ran (see _create_trainer).")

        model_path = self.output_dir / "trained_model.ckpt"
        torch.save(harness.state_dict(), model_path)

        # Save model config
        config_path = self.output_dir / "model_config.json"
        with open(config_path, 'w') as f:
            json.dump(model_config, f, indent=2)

        # The pathway manifest written by _build_model_config; predict re-reads it.
        pathway_file_path = self.output_dir / "pathway_files_absolute.txt"
        if not pathway_file_path.exists():
            raise FileNotFoundError(f"pathway file missing: {pathway_file_path}")

        # VENDORED (V4): no cache copytree. Training already ran with cwd set to
        # the output dir, so PRESAGE's `./cache` IS `<run_dir>/cache` — there is
        # nothing to move, and nothing to get half-copied on a crash.
        cache_dir = self.output_dir / "cache"
        if not cache_dir.exists():
            raise FileNotFoundError(
                f"PRESAGE cache missing at {cache_dir} after training — predict "
                f"cannot rebuild it")

        # Save control mean (needed to convert deltas back to absolute expression)
        train_dataset = harness.module.datamodule.train_dataset
        control_mean = getattr(train_dataset, 'control_mean', None)
        if control_mean is None:
            raise RuntimeError("Training dataset missing control_mean")

        control_mean_path = self.output_dir / "control_mean.csv"
        control_mean.to_csv(control_mean_path)
        log.info("Saved control mean")

        # VENDORED (V6): paths recorded RELATIVE to the run dir. The original
        # wrote absolute `/model_output/...` strings and predict recovered them
        # with `.replace("/model_output/", "/pretrained_model/")` — an atheus mount
        # convention. Our contract binds the same host dir at /model_output for
        # BOTH calls (docker/CONTRACT.md), so that rewrite would point every
        # artefact at a path that does not exist. Relative paths are correct under
        # either convention and survive the run dir being moved.
        checkpoint_data = {
            'model_config': model_config,
            'hyperparameters': self.hyperparams,
            'data_path': str(self.data_path),
            'split_name': self.config['split_name'],
            'seed': self.seed,
            'processed_data_path': 'presage_data/cellsimbench_processed.h5ad',
            'splits_path': 'presage_data/splits.json',
            'pathway_file': 'pathway_files_absolute.txt',
            'cache_dir': 'cache',
            'control_mean_path': 'control_mean.csv',
            'training_completed': True,
            # Covariate metadata (None / [] for single-context runs). Pinning the
            # categories at train time guarantees inference uses the same one-hot
            # encoding even if the inference adata has only a subset of them.
            'covariate_field': getattr(self, '_covariate_field_resolved', None),
            'cov_categories': list(getattr(harness.module.datamodule, 'cov_categories', None) or []),
            # Provenance the host lifts into fingerprint.json["report"].
            'report': {
                'best_epoch': best_epoch,
                'best_val_loss': best_val_loss,
                'max_epochs': self.hyperparams['max_epochs'],
                'wall_seconds': round(wall_seconds, 1),
                'n_train_rows': int(len(train_dataset)),
                # What the run ACTUALLY used, which is not always what the recipe
                # says — see V17. Recorded so `fingerprint.json` cannot imply a
                # batch size the run did not use.
                'batch_size_configured': int(self.hyperparams['batch_size']),
                'batch_size_effective': int(getattr(self, '_effective_batch_size',
                                                    self.hyperparams['batch_size'])),
                'n_val_rows': int(len(harness.module.datamodule.val_dataset)),
                'held_out': self._held_out_units(),
            },
        }
        checkpoint_path = self.output_dir / "presage_training_hparams.json"
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint_data, f, indent=2, cls=PathEncoder)

        log.info(f"Saved training artifacts to {self.output_dir}")

    def _held_out_units(self):
        """The (covariate, perturbation) units this run held out.

        The host fingerprint records the split *hash*; this records what the split
        actually withheld, which is what a reader needs to audit a cell-axis run.
        """
        adata = self.data_manager.adata
        split_name = self.config['split_name']
        cov_field = getattr(self, '_covariate_field_resolved', None)
        test = adata.obs[adata.obs[split_name].astype(str) == 'test']
        conds = test['condition'].astype(str)
        keep = ~conds.str.contains('ctrl', case=False, na=False, regex=False)
        if cov_field and cov_field in test.columns:
            units = sorted({(str(c), str(k)) for c, k in
                            zip(test.loc[keep, cov_field], conds[keep])})
            return {'covariate_field': cov_field,
                    'pairs': [list(u) for u in units], 'n': len(units)}
        perts = sorted(set(conds[keep]))
        return {'covariate_field': None, 'perturbations': perts, 'n': len(perts)}

    def _load_trained_model(self):
        """Load trained model for prediction.

        VENDORED (V6): every artefact is resolved RELATIVE to ``model_path``. The
        original recovered absolute paths recorded at train time by rewriting
        ``/model_output/`` -> ``/pretrained_model/``, which encoded the atheus
        runner's two-mount convention. Our contract binds one host dir at
        ``/model_output`` for both calls (docker/CONTRACT.md), so that rewrite
        produced paths that never exist.
        """
        if not self.model_path:
            raise ValueError("model_path must be provided for prediction mode")

        model_path = Path(self.model_path).resolve()

        def _artifact(key: str) -> Path:
            """Resolve a recorded artefact path against the run dir.

            Absolute values are rejected rather than silently honoured: a
            checkpoint written by the pre-V6 wrapper carries `/model_output/...`
            strings that happen to resolve inside a container and point at the
            wrong place outside one. Better to say so.
            """
            rel = checkpoint_data[key]
            if os.path.isabs(rel):
                raise ValueError(
                    f"{key}={rel!r} is absolute; this checkpoint predates the "
                    f"relative-path fix (V6) and cannot be trusted — retrain.")
            path = model_path / rel
            if not path.exists():
                raise FileNotFoundError(f"{key}: {path} not found under {model_path}")
            return path

        # Load training checkpoint with all metadata
        with open(model_path / "presage_training_hparams.json") as f:
            checkpoint_data = json.load(f)
        with open(model_path / "model_config.json") as f:
            model_config = json.load(f)

        # Restore the cache to ./cache/, where upstream PRESAGE hardcodes it.
        # `predict` has already chdir'd into a private tempdir, so this is a fresh
        # directory per process and cannot race another fold.
        cache_source = _artifact('cache_dir')
        cache_dest = Path("./cache")
        log.info(f"Restoring PRESAGE cache from {cache_source} to {cache_dest.resolve()}")
        if cache_dest.is_symlink():
            cache_dest.unlink()
        elif cache_dest.exists():
            shutil.rmtree(cache_dest)
        # `symlinks=True` is load-bearing, not a detail: the run dir's cache is a
        # symlink farm over the 6.4 GB baked cache (V4). Following those links here
        # would materialise all of it into a per-process tempdir on /tmp, which is
        # both wasteful and likely to run the node out of space. Preserving them
        # copies only what training actually generated.
        shutil.copytree(cache_source, cache_dest, symlinks=True)

        control_mean = pd.read_csv(_artifact('control_mean_path'), index_col=0)
        log.info("Loaded control mean")

        processed_data_path = _artifact('processed_data_path')
        splits_path = _artifact('splits_path')
        model_config['pathway_files'] = str(_artifact('pathway_file'))

        # Create datamodule config matching training
        datamodule_config = {
            'processed_adata_path': str(processed_data_path),
            'splits_path': str(splits_path),
            'dataset': 'cellsimbench',
            'data_dir': str(model_path / "presage_data"),
            'batch_size': self.hyperparams['batch_size'],
            'use_pseudobulk': True,
            'preprocessing_zscore': False,
            'noisy_pseudobulk': False,
            'perturb_field': 'perturbation',
            'control_key': 'control',
            'split_name': checkpoint_data['split_name'],
            'seed': self.seed,
            # VENDORED (V7): `perts_as_delta_ref` is pinned False, not exposed as a
            # hyperparameter. Its inference half was never implemented — the
            # original _generate_predictions logged "perturbation mean logic not
            # implemented yet" and fell back to the control mean anyway — so a
            # `true` here would have trained against one reference and predicted
            # against another.
            'perts_as_delta_ref': False,
            # Pin training-time covariate categories so inference one-hot
            # encoding matches train-time exactly.
            'covariate_field': checkpoint_data.get('covariate_field', None),
            'cov_categories': checkpoint_data.get('cov_categories', None) or None,
        }

        datamodule = CellSimBenchDataModule.from_config(datamodule_config)
        datamodule.prepare_data()
        datamodule.setup("fit")  # initialise the train dataset (shapes + encoders)

        model = ComboPRESAGE(
            model_config,
            datamodule,
            input_dimension=len(datamodule.train_dataset.adata.var),
            output_dimension=len(datamodule.train_dataset.adata.var)
        )
        harness = CellSimBenchModelHarness(model, datamodule, model_config)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log.info(f"Loading model weights from {model_path / 'trained_model.ckpt'} on {device}")
        state_dict = torch.load(model_path / "trained_model.ckpt", map_location=device)
        harness.load_state_dict(state_dict)
        harness.to(device)
        harness.eval()

        log.info("Model loaded successfully for prediction")
        return harness, datamodule, control_mean

    def _generate_predictions(self, harness, datamodule, control_mean,
                              test_conditions):
        """Generate predictions for test conditions."""
        import torch
        from torch.utils.data import DataLoader
        
        # Load the full test data (we need it for creating proper test dataset)
        test_data_path = self.config['data_path']
        full_adata = sc.read_h5ad(test_data_path)
        
        # Convert to PRESAGE format first
        full_adata = self._convert_to_presage_format(full_adata)
        
        # Convert test conditions to PRESAGE format (whole-string match;
        # see _normalize_perturbation re: substring-replace bug).
        presage_test_conditions = [_normalize_perturbation(c) for c in test_conditions]
        
        log.info(f"Generating predictions for {len(presage_test_conditions)} test conditions")
        
        # Update the splits to include only our test conditions
        with open(datamodule.splits_json_path, 'r') as f:
            original_splits = json.load(f)
        
        # Create new splits with our test conditions
        test_splits = {
            'train': original_splits['train'],  # Keep original train
            'val': original_splits['val'],      # Keep original val
            'test': presage_test_conditions     # Use our test conditions
        }
        
        # Save temporary test splits
        output_dir = Path(self.config['output_path']).parent
        temp_splits_path = output_dir / "test_splits.json"
        with open(temp_splits_path, 'w') as f:
            json.dump(test_splits, f)
        
        # Update datamodule's split path for test
        datamodule.split_path = str(temp_splits_path)
        datamodule.splits = test_splits
        
        # Setup datamodule for test stage
        datamodule._data_setup = False  # Force re-setup
        datamodule.setup("test")
        
        # Get the test dataloader
        test_loader = datamodule.test_dataloader()
        
        # Run predictions
        log.info("Running model inference...")
        harness.eval()
        
        all_keys = []
        all_cov_labels = []
        all_predictions = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                # Move batch to device if CUDA is available
                device = next(harness.parameters()).device
                batch = {k: v.to(device) if torch.is_tensor(v) else v
                        for k, v in batch.items()}

                # Run predict_step (following ModelHarness.predict_step logic)
                keys, preds, cov_labels = harness.predict_step(batch, batch_idx)

                # Collect results
                all_keys.extend(list(keys))
                all_cov_labels.extend(list(cov_labels))
                if preds is not None:
                    all_predictions.append(preds.cpu().numpy())

        # Concatenate all predictions
        if all_predictions:
            all_predictions = np.concatenate(all_predictions, axis=0)
        else:
            log.error("No predictions generated!")
            raise ValueError("No predictions generated!")

        # Detect cov-aware mode by inspecting whether any cov label is non-empty.
        cov_aware = any(c not in (None, "") for c in all_cov_labels)
        log.info(
            f"Generated {len(all_keys)} predictions"
            + (f" across {len(set(all_cov_labels))} covariates" if cov_aware else "")
        )

        # Build per-row identity (just perturbation in legacy mode; (pert, cov) in cov-aware mode).
        # Note duplicate perturbation indices in cov-aware mode: same KO appears for each cov.
        pred_df = pd.DataFrame(
            data=all_predictions,
            columns=datamodule.var_names,
        )
        pred_df["__pert__"] = list(all_keys)
        pred_df["__cov__"] = list(all_cov_labels)

        # Filter to only measured genes (remove the fake genes added for missing perturbations)
        if hasattr(datamodule.train_dataset.adata.var, 'measured_gene'):
            measured_genes = datamodule.train_dataset.adata.var.measured_gene
            keep_cols = [c for c in pred_df.columns if c in ("__pert__", "__cov__")] + \
                        [g for g in datamodule.var_names if g in measured_genes]
            pred_df = pred_df[keep_cols]

        # Convert deltas to absolute expression. In cov-aware mode, control_mean is a
        # per-covariate DataFrame indexed by cov label; pick the matching row per
        # prediction. In legacy mode (single row indexed "default"), use that row.
        # VENDORED (V7): the `perts_as_delta_ref` branch is gone — both arms of it
        # assigned `control_mean`, with the true arm logging that the perturbation
        # mean was "not implemented yet".
        reference_mean_df = control_mean

        gene_cols = [c for c in pred_df.columns if c not in ("__pert__", "__cov__")]
        common_genes = [g for g in gene_cols if g in reference_mean_df.columns]
        if len(common_genes) != len(gene_cols):
            raise ValueError(
                f"Gene mismatch: prediction has {len(gene_cols)} genes, "
                f"reference mean covers {len(common_genes)}"
            )
        # Apply per-covariate reference. Vectorize by covariate label so we do
        # at most n_cov iterations (cheap), keeping memory predictable.
        ref_index = [str(i) for i in reference_mean_df.index]
        ref_lookup = {str(i): i for i in reference_mean_df.index}
        for cov_label, group_idx in pred_df.groupby("__cov__").groups.items():
            cov_key = str(cov_label)
            if cov_key in ref_lookup:
                ref_row = reference_mean_df.loc[ref_lookup[cov_key], common_genes].values
            elif "default" in ref_lookup:
                ref_row = reference_mean_df.loc[ref_lookup["default"], common_genes].values
            elif len(reference_mean_df) == 1:
                # Legacy single-row checkpoint: the only row is the reference regardless of label.
                ref_row = reference_mean_df.iloc[0][common_genes].values
            else:
                raise KeyError(
                    f"Cov label {cov_label!r} not in reference mean index {ref_index}; "
                    "training and inference cov categories disagree."
                )
            ref_row = np.float32(ref_row)
            pred_df.loc[group_idx, common_genes] = (
                pred_df.loc[group_idx, common_genes].values + ref_row
            )

        # Build the output AnnData. Get a `var` template from the input adata.
        test_perturbations = list(set(presage_test_conditions + ['control']))
        test_mask = full_adata.obs['perturbation'].isin(test_perturbations)
        test_adata = full_adata[test_mask].copy()
        var_template = test_adata.var[test_adata.var.index.isin(pred_df.columns)].copy()

        # Reorder pred_df gene columns to match var_template index for safety
        var_template = var_template.reindex(common_genes)
        X_out = pred_df[common_genes].values.astype(np.float32)

        # Build obs with both perturbation and covariate columns. condition uses the
        # CellSimBench format ("+"-separated combos, "ctrl" for control).
        conditions = [k.replace("_", "+").replace("control", "ctrl") for k in pred_df["__pert__"]]
        cov_col = pred_df["__cov__"].tolist()
        # Use a synthetic but stable index for AnnData (must be string + unique).
        if cov_aware:
            obs_idx = [f"{c}__{p}" for c, p in zip(cov_col, conditions)]
        else:
            obs_idx = list(conditions)
        obs_df = pd.DataFrame(
            {
                "covariate": cov_col,
                "condition": conditions,
                "pair_key": [f"{c}_{cond}" for c, cond in zip(cov_col, conditions)],
                "perturbation": conditions,
                "model": ["presage"] * len(conditions),
                "is_control": [c == "ctrl" for c in conditions],
            },
            index=pd.Index(obs_idx, name=None),
        )

        output_adata = sc.AnnData(X=X_out, obs=obs_df, var=var_template)
        log.info(f"Created output AnnData with shape {output_adata.shape}")
        return output_adata
