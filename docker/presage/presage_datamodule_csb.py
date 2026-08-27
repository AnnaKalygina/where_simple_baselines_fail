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
Custom PRESAGE DataModule for CellSimBench data.
Inherits from PRESAGEDataModule but handles CellSimBench-specific data format.
"""

import os
import json
import re
from pathlib import Path
import pandas as pd
import numpy as np
import scanpy as sc
from typing import Dict, List
import logging
from tqdm import tqdm

import sys
# VENDORED (V1): same upstream-checkout resolution as presage_wrapper.py, with
# the CSCS and ref/ candidates dropped. Kept as a local copy rather than an
# import of the wrapper: this module is also imported BY the wrapper, so
# importing back would be circular.
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

from presage_datamodule import PRESAGEDataModule
import torch
from torch.utils.data import Dataset
from anndata import AnnData

log = logging.getLogger(__name__)


def compute_pseudobulk(
    adata: AnnData, condition_field: str = "perturbation"
) -> pd.DataFrame:

    return pd.DataFrame(
        adata.X, index=adata.obs[condition_field], columns=adata.var_names
    ).pipe(lambda df: df.groupby(df.index).mean())

class CellSimBenchscPerturbData(Dataset):
    """Interface for a preprocessed AnnData object derived from scPerturb.org

    Preprocessing defined in `prepare_data` and/or `setup` methods of `scPerturbDataModule`.

    Needs to identify and separate control cells.

    Needs to compute pseudobulk and implement option to generate samples from
    either pseudobulk or single cells

    Needs to implement an invertible mapping between
    perturbation keys and indicators over adata.var_names
    """

    def __init__(
        self,
        adata,
        pert_covariates=None,
        cov_categories=None,
        perturb_field="perturbation",
        control_key="control",
        use_pseudobulk=False,
        z_score=False,
    ):
        """Args:
            pert_covariates: name of the obs column carrying covariate (e.g. "cell_type"),
                or None to disable covariate conditioning (covmtx stays empty for backward compat).
            cov_categories: ordered list of all covariate categories the model will see across
                training/inference. Must be supplied (alongside pert_covariates) so that train
                and inference share the same one-hot encoding even when a per-split adata happens
                to be missing a category. None ⇒ inferred from adata at construction time.
        """
        self.adata = adata

        self.pert_covariates = pert_covariates

        self.perturb_field = perturb_field
        self.control_key = control_key

        # separate control cells
        self.perturbs = adata

        perturb_keys = self.perturbs.obs[self.perturb_field].to_numpy()
        self.X = self.perturbs.X.astype(np.float32)

        self.perturb_keys = perturb_keys
        self.var_names = adata.var_names

        self.indmtx = np.vstack(
            [self.pert_to_ind(key) for key in self.perturb_keys]
        ).astype(np.float32)

        # for now, drop perturbations of genes that aren't measured
        not_observed = self.indmtx.sum(1) == 0
        if not_observed.any():
            not_observed_keys = set(self.perturb_keys[not_observed])
            print(
                f"WARNING: Data contain perturbations for {len(not_observed_keys)} genes "
                f"for which there is no mRNA expression measurement: {not_observed_keys}.\n"
                "They will be removed because they have all-0 indicator variables."
            )
            observed = ~not_observed
            self.X = self.X[observed]
            self.perturb_keys = self.perturb_keys[observed]
            self.indmtx = self.indmtx[observed]

        # Build covariate matrix (one-hot encoded) when a covariate column is provided.
        # When pert_covariates is None or the column has only one unique value,
        # fall back to the legacy empty matrix so untouched callers keep working.
        if pert_covariates is not None and pert_covariates in adata.obs.columns:
            cov_values = adata.obs[pert_covariates].astype(str).to_numpy()
            if not_observed.any():
                cov_values = cov_values[observed]
            if cov_categories is None:
                cov_categories = sorted(np.unique(cov_values).tolist())
            self.cov_categories = list(cov_categories)
            self.pert_cov_field = pert_covariates
            cov_to_idx = {c: i for i, c in enumerate(self.cov_categories)}
            n_cov = len(self.cov_categories)
            covmtx = np.zeros((len(cov_values), n_cov), dtype=np.float32)
            for i, c in enumerate(cov_values):
                if c not in cov_to_idx:
                    raise KeyError(
                        f"Covariate value {c!r} not in cov_categories {self.cov_categories}; "
                        "training and inference must share the same category set."
                    )
                covmtx[i, cov_to_idx[c]] = 1.0
            self.covmtx = covmtx
            self.cov_values = cov_values  # keep raw labels for inference output
        else:
            self.cov_categories = []
            self.pert_cov_field = None
            self.cov_values = np.array([""] * self.indmtx.shape[0])
            self.covmtx = np.zeros((self.indmtx.shape[0], 0), dtype=np.float32)

    def pert_to_ind(self, pert_key):
        """Convert perturbation key to indicator vector."""
        ind = np.zeros(len(self.adata.var))
        gene_to_idx = {gene: i for i, gene in enumerate(self.adata.var.index)}
            
        # Handle single and combo perturbations
        if "_" in pert_key and pert_key != "control":
            # Combo perturbation
            genes = pert_key.split("_")
        else:
            # Single perturbation
            genes = [pert_key] if pert_key != "control" else []
            
        for gene in genes:
            if gene in gene_to_idx:
                ind[gene_to_idx[gene]] = 1
                
        return ind

    def ind_to_pert(self, ind) -> str:
        if hasattr(ind, "numpy"):
            ind = ind.numpy()
        ind = ind > 0
        key = "_".join(self.var_names[ind])
        if key not in self.perturb_keys:
            key = "_".join(reversed(self.var_names[ind]))
        assert key in self.perturb_keys, f"Could not find perturb key {key}"
        return key

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):

        batch_data = dict(
            inds=torch.tensor(self.indmtx[i]),
            expr=torch.tensor(self.X[i]),
            cov=torch.tensor(self.covmtx[i]),
        )

        # Include the perturbation key for identification
        pert_key = self.perturb_keys[i]
        batch_data['pert_key'] = pert_key
        # Also expose the raw covariate label so inference can write it back
        # into the output adata. Empty string when no covariate is in use.
        batch_data['cov_label'] = str(self.cov_values[i]) if i < len(self.cov_values) else ""

        return batch_data


class CellSimBenchDataModule(PRESAGEDataModule):
    """Custom PRESAGE DataModule for CellSimBench data."""
    
    def __init__(self, processed_adata_path: str, splits_path: str, 
                 data_dir: str, dataset: str = "cellsimbench", **kwargs):
        """
        Initialize CellSimBench DataModule.
        
        Args:
            processed_adata_path: Path to the processed AnnData file
            splits_path: Path to JSON file with train/val/test splits
            data_dir: Base data directory
            dataset: Dataset name (default: "cellsimbench")
            **kwargs: Additional arguments - ALL REQUIRED:
                - batch_size
                - use_pseudobulk
                - preprocessing_zscore
                - perturb_field
                - control_key
                - dataset_class
        """
        # Store our custom paths
        self.processed_adata_path = Path(processed_adata_path)
        self.splits_json_path = Path(splits_path)
        
        # Initialize LightningDataModule base class
        import pytorch_lightning as pl
        pl.LightningDataModule.__init__(self)
        
        # Skip PRESAGEDataModule's dataset validation
        # We need to set required attributes manually
        self.dataset = dataset
        self.data_dir = Path(data_dir)
        
        # REQUIRED parameters - will raise KeyError if missing
        self.batch_size = kwargs['batch_size']
        self.use_pseudobulk = kwargs['use_pseudobulk']
        
        self.perturb_field = kwargs['perturb_field']
        self.control_key = kwargs['control_key']
        self.dataset_class = kwargs['dataset_class']
        
        # Set up directories
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.dataset_dir, exist_ok=True)
        os.makedirs(self.deg_dir, exist_ok=True)
        
        # Initialize parent attributes
        self.n_genes = None
        self.degs = None
        self.var_names = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self._data_prepared = False
        self._data_setup = False
        
        # Additional parent class attributes
        self.nperturb_clusters = None  # Required by parent setup() method
        self.allow_list = None
        self.allow_list_out_genes = None
        self.X_train_pca = None
        
        # Set split path
        self.split_path = str(self.splits_json_path)
        
        log.info(f"Initialized CellSimBenchDataModule with dataset: {dataset}")
        
        # Store split name if provided (for proper control filtering)
        self.split_name = kwargs['split_name']  # Required
        # Store whether to use perturbation mean as delta reference (required)
        self.perts_as_delta_ref = kwargs['perts_as_delta_ref']

        # Optional covariate config (drives per-(cov,pert) pseudobulk + cov-aware model).
        # When unset, behaviour is identical to the legacy single-context path.
        self.covariate_field = kwargs.get('covariate_field', None)
        # cov_categories pinned at construction time so train and inference share the
        # same one-hot encoding even when a per-split adata is missing some categories.
        # When None, derived from the adata in load_preprocessed.
        self.cov_categories = kwargs.get('cov_categories', None)

    def compute_means(self, adata):
        """Compute control and perturbation means.

        With covariate_field set: returns DataFrames whose row index is the
        covariate value. Without a covariate field: returns single-row DataFrames
        (legacy behaviour).
        """
        cov_field = getattr(self, "covariate_field", None)
        cov_values = (
            list(self.cov_categories) if cov_field is not None and getattr(self, "cov_categories", None)
            else [None]
        )

        # Source of fallback controls when a per-split adata has none.
        full_adata_ctrl_source = (
            self._full_adata if getattr(self, "_full_adata", None) is not None else adata
        )

        control_means = {}
        perturbation_means = {}
        # VENDORED (V10): remember which covariates had NO control cells in this
        # split and borrowed them from the full adata. create_dataset() refuses to
        # build TRAINING rows against such a reference — see the check there.
        self._ctrl_fallback_covs = set()
        for cov_val in cov_values:
            # Subset to this covariate (or the whole adata in legacy mode).
            if cov_val is None:
                cov_adata = adata
                cov_full = full_adata_ctrl_source
                row_label = "default"
            else:
                cov_adata = adata[adata.obs[cov_field] == cov_val]
                cov_full = full_adata_ctrl_source[full_adata_ctrl_source.obs[cov_field] == cov_val]
                row_label = cov_val

            ctrl_cells = cov_adata[cov_adata.obs[self.perturb_field] == self.control_key]
            if len(ctrl_cells) == 0:
                # Fall back to controls from the full adata, restricted to this covariate.
                # Same rationale as before (mcfaline23 assigns controls to train only),
                # just done per covariate so each row of control_means is well defined.
                full_ctrl = cov_full[cov_full.obs[self.perturb_field] == self.control_key]
                if len(full_ctrl) > 0:
                    log.info(
                        f"[cov={row_label}] split has 0 controls; falling back to {len(full_ctrl)} "
                        "controls from the full adata for compute_means."
                    )
                    ctrl_cells = full_ctrl
                    self._ctrl_fallback_covs.add(row_label)
            if len(ctrl_cells) == 0:
                raise ValueError(
                    f"No control cells found with {self.perturb_field}=={self.control_key} "
                    f"for cov={row_label}"
                )

            ctrl_x = ctrl_cells.X
            if hasattr(ctrl_x, "toarray"):
                ctrl_x = ctrl_x.toarray()
            control_means[row_label] = np.mean(ctrl_x, axis=0)
            log.info(f"[cov={row_label}] computed control mean using {len(ctrl_cells)} cells")

            # Per-cov perturbation mean (balanced across perturbations within this cov).
            pert_means_list = []
            perturbations_included = []
            for pert in cov_adata.obs[self.perturb_field].unique():
                if pert == self.control_key:
                    continue
                pert_cells_subset = cov_adata[cov_adata.obs[self.perturb_field] == pert]
                if len(pert_cells_subset) == 0:
                    continue
                pert_x = pert_cells_subset.X
                if hasattr(pert_x, "toarray"):
                    pert_x = pert_x.toarray()
                pert_means_list.append(np.mean(pert_x, axis=0))
                perturbations_included.append(pert)

            if len(pert_means_list) == 0:
                raise ValueError(f"No perturbations found for cov={row_label}")
            perturbation_means[row_label] = np.mean(np.array(pert_means_list), axis=0)
            log.info(
                f"[cov={row_label}] computed perturbation mean using "
                f"{len(perturbations_included)} perturbations"
            )

        index = list(control_means.keys())  # preserves the order of cov_values
        var_index = adata.var.index
        control_mean_df = pd.DataFrame(
            np.stack([control_means[k] for k in index]),
            index=index,
            columns=var_index,
        )
        perturbation_mean_df = pd.DataFrame(
            np.stack([perturbation_means[k] for k in index]),
            index=index,
            columns=var_index,
        )
        return control_mean_df, perturbation_mean_df
    
    def create_dataset(self, adata, train: bool):
        """Create dataset with pseudobulk processing.

        With covariate_field set: produces one pseudobulk row per
        (covariate_value, perturbation) pair. control_mean / perturbation_mean
        are per-covariate. Predictions are centered against the per-covariate
        reference (so the same KO under different cell lines starts from
        different baselines).

        Without a covariate_field (legacy): one pseudobulk row per perturbation,
        single mean. Mathematically identical to the prior implementation.
        """
        cov_field = getattr(self, "covariate_field", None)
        cov_categories = getattr(self, "cov_categories", None)
        use_cov = cov_field is not None and cov_categories

        # Compute control and perturbation means (per-cov dicts, or single-row DFs in legacy mode)
        control_mean, perturbation_mean = self.compute_means(adata)

        if not self.use_pseudobulk:
            raise NotImplementedError("Non-pseudobulk not implemented")

        pseudobulk_data = []
        pseudobulk_keys = []
        pseudobulk_covs = []

        if use_cov:
            cov_values_to_process = cov_categories
        else:
            cov_values_to_process = [None]

        for cov_val in cov_values_to_process:
            if cov_val is None:
                cov_adata = adata
                row_label = "default"
            else:
                cov_adata = adata[adata.obs[cov_field] == cov_val]
                row_label = cov_val
                if len(cov_adata) == 0:
                    log.info(f"[cov={row_label}] no cells in this split — skipping pseudobulk")
                    continue

            # VENDORED (V10): a TRAINING row may not be centred on a control mean
            # borrowed from outside its split.
            #
            # compute_means falls back to controls from the full adata when a
            # covariate has none in this split. At inference that is legitimate and
            # necessary — every predictor, the baselines included, is given the
            # held-out cell type's basal state, and the benchmark subtracts control
            # itself. At TRAINING it is not: the target `pseudobulk - ref` would
            # carry information from val/test cells into the loss.
            #
            # In practice this only fires when a covariate has perturbed cells in
            # train but no control cells there (a covariate with no train cells at
            # all is skipped above). Raising rather than skipping: dropping those
            # rows silently would shrink the training set without saying so.
            if train and row_label in getattr(self, "_ctrl_fallback_covs", set()):
                raise ValueError(
                    f"[cov={row_label}] has perturbed cells in the training split but "
                    f"no control cells there, so its control mean was borrowed from "
                    f"outside the split. Training on that reference would leak "
                    f"held-out cells into the loss. Fix the split so every trained "
                    f"covariate carries its own controls.")

            # Per-covariate reference mean (control or perturbation, depending on flag).
            if self.perts_as_delta_ref:
                ref = perturbation_mean.loc[row_label].values
            else:
                ref = control_mean.loc[row_label].values

            for pert in tqdm(
                cov_adata.obs[self.perturb_field].unique(),
                desc=f"Pseudobulk [cov={row_label}]",
            ):
                if pert == self.control_key:
                    continue
                pert_cells = cov_adata[cov_adata.obs[self.perturb_field] == pert]
                if len(pert_cells) == 0:
                    continue
                expr = pert_cells.X
                if hasattr(expr, "toarray"):
                    expr = expr.toarray()
                pseudobulk_expr = np.mean(expr, axis=0)
                centered_expr = pseudobulk_expr - ref

                pseudobulk_data.append(centered_expr)
                pseudobulk_keys.append(pert)
                pseudobulk_covs.append(row_label if use_cov else "")

        if not pseudobulk_data:
            raise ValueError("No pseudobulk rows produced — adata may be empty after split filtering")

        X = np.array(pseudobulk_data, dtype=np.float32)
        perturb_keys = np.array(pseudobulk_keys)

        modified_obs = pd.DataFrame({self.perturb_field: perturb_keys})
        if use_cov:
            modified_obs[cov_field] = pseudobulk_covs
        modified_adata = sc.AnnData(X=X, obs=modified_obs, var=adata.var)

        dataset = CellSimBenchscPerturbData(
            modified_adata,
            use_pseudobulk=False,  # already computed
            pert_covariates=cov_field if use_cov else None,
            cov_categories=cov_categories if use_cov else None,
        )

        # Store control + perturbation means alongside the dataset (for inference centering).
        # Always per-cov-indexed DataFrame ("default" key in legacy mode); downstream code can
        # detect by checking dataset.cov_categories.
        dataset.control_mean = control_mean
        dataset.perturbation_mean = perturbation_mean
        dataset.perts_as_delta_ref = self.perts_as_delta_ref

        return dataset
    
    @property
    def dataset_dir(self) -> Path:
        """Dataset-specific directory."""
        return self.data_dir / self.dataset
    
    @property
    def deg_dir(self) -> Path:
        """Directory for DEG files."""
        return self.dataset_dir / "degs"
    
    @property
    def preprocessed_path(self) -> str:
        """Path to preprocessed data."""
        return str(self.processed_adata_path)
    
    @property
    def raw_path(self) -> str:
        """Path to raw data (same as preprocessed for us)."""
        return str(self.processed_adata_path)
    
    @property
    def merged_deg_file(self) -> str:
        """Path to merged DEG JSON file."""
        return str(self.deg_dir / "merged.degs.json")
    
    def prepare_data(self) -> None:
        """
        Prepare data for PRESAGE training.
        For CellSimBench, data is already prepared, we just need to extract DEGs.
        """
        if not self._data_prepared:
            log.info("Preparing CellSimBench data...")
            
            # Load the processed data
            adata = sc.read(self.processed_adata_path)
            
            # Extract and process DEGs if not already done
            if not os.path.exists(self.merged_deg_file):
                log.info("Processing DEGs from CellSimBench data...")
                
                if 'deg_gene_dict' in adata.uns:
                    # CellSimBench format: covariate_key_perturbation -> list of DEGs
                    deg_dict = adata.uns['deg_gene_dict']
                    
                    # Process keys to extract just perturbation names
                    # Format: "replogle22rpe1_UTP23" -> "UTP23"
                    processed_degs = {}
                    for key, genes in deg_dict.items():
                        # Extract perturbation name after first underscore
                        match = re.match(r'^[^_]+_(.+)$', key)
                        if match:
                            pert_name = match.group(1)
                            # Convert perturbation format to match PRESAGE expectations
                            pert_name = pert_name.replace("+", "_")
                            processed_degs[pert_name] = list(genes) if hasattr(genes, '__iter__') else [genes]
                    
                    # Save processed DEGs
                    with open(self.merged_deg_file, 'w') as f:
                        json.dump(processed_degs, f)
                    log.info(f"Saved {len(processed_degs)} perturbation DEGs to {self.merged_deg_file}")
                    
                else:
                    log.warning("No DEG information found in adata.uns. Creating empty DEG file.")
                    with open(self.merged_deg_file, 'w') as f:
                        json.dump({}, f)
            else:
                log.info(f"Found existing DEG file at {self.merged_deg_file}")
            
            self._data_prepared = True
    
    def setup(self, stage=None):
        """
        Override parent setup to properly handle control samples by split assignment.
        This fixes the data leakage issue where all control samples were included in all splits.
        """
        if not self._data_setup:
            log.info(f"Setting up data for stage: {stage}")
            
            # Load preprocessed data and splits
            adata = self.load_preprocessed()
            # Stash the full adata so compute_means can fall back to its
            # controls when a per-split adata happens to have none (e.g.
            # mcfaline23 P2 assigns all controls to the train split only).
            self._full_adata = adata

            # Load split assignments
            with open(self.split_path, "r") as f:
                splits = json.load(f)
            self.splits = splits
            
            # Get var names and degs from loaded data
            self.var_names = adata.var_names
            self.n_genes = len(self.var_names)
            
            # Load DEGs
            if os.path.exists(self.merged_deg_file):
                with open(self.merged_deg_file, "r") as f:
                    self.degs = json.load(f)
            else:
                self.degs = {}
            
            # Filter splits to only include perturbations that exist in adata
            for key in splits:
                valid_perturbations = []
                for pert in splits[key]:
                    # Check if this perturbation exists
                    exists = (adata.obs[self.perturb_field] == pert).any()
                    if exists:
                        valid_perturbations.append(pert)
                splits[key] = valid_perturbations

                        
            if stage == "fit":
                # Create train and validation datasets WITH PROPER CONTROL FILTERING
                subsets = {"train": splits["train"], "val": splits["val"]}
                for name, subset in subsets.items():
                    # Create mask for perturbations in this subset
                    condition_mask = pd.Series([False] * len(adata), index=adata.obs.index)
                    
                    for pert in subset:
                        # Add cells matching this perturbation
                        pert_mask = (adata.obs[self.perturb_field] == pert)
                        condition_mask |= pert_mask
                    
                    # Get control samples that belong to THIS SPECIFIC SPLIT
                    control_mask = (
                        (adata.obs[self.perturb_field] == self.control_key) &
                        (adata.obs[self.split_name] == name)  # Only controls from this split
                    )
                    
                    # Combine condition and control masks
                    combined_mask = condition_mask | control_mask
                    split_adata = adata[combined_mask]
                    # Remove any cells that are not in the split
                    split_adata = split_adata[split_adata.obs[self.split_name] == name]

                    # VENDORED (V11): post-condition on the per-cell filter above.
                    # This filter is the whole reason PRESAGE is leak-safe where
                    # GEARS/scGPT are not (they split by marginal condition lists);
                    # assert it rather than trust it, so a future edit that loosens
                    # it fails loudly instead of quietly training on held-out cells.
                    off_split = (split_adata.obs[self.split_name].astype(str) != name).sum()
                    if off_split:
                        raise RuntimeError(
                            f"{off_split} cells outside split {name!r} survived the "
                            f"per-cell filter — refusing to train on held-out cells")
                    if len(split_adata) == 0:
                        raise ValueError(f"split {name!r} is empty after filtering")

                    # Create dataset for this split
                    setattr(
                        self,
                        f"{name}_dataset",
                        self.create_dataset(
                            split_adata,
                            train=(name == "train"),
                        ),
                    )

                    log.info(f"Created {name} dataset with {combined_mask.sum()} samples "
                            f"({condition_mask.sum()} conditions + {control_mask.sum()} controls)")

                self.train_perturb_labels = None
            
            if stage == "test":
                self.nperturb_clusters = None
                self.train_perturb_labels = None

                # Create test dataset WITH PROPER CONTROL FILTERING
                condition_mask = pd.Series([False] * len(adata), index=adata.obs.index)
                for pert in splits["test"]:
                    pert_mask = (adata.obs[self.perturb_field] == pert)
                    condition_mask |= pert_mask

                # VENDORED (V9): the held-out split is `test`, stated, not guessed.
                #
                # The original inferred it from the FIRST row of the first matched
                # perturbation:
                #     actual_split = adata.obs.loc[obs.pert == matched, split_name].iloc[0]
                # That is only sound when a perturbation lives in exactly one split.
                # Under the cell-axis regimes (UnseenCell / UnseenBoth / UnseenPair)
                # a held-out perturbation legitimately ALSO appears in `train` cells
                # of other cell types, so `.iloc[0]` can return "train" — and the
                # "test" dataset would then be built from TRAINING cells and scored
                # as if it were held out. Silent, and it inflates the score.
                #
                # It existed to support predicting val conditions by stuffing them
                # into splits['test']. Our contract never does that: predict is
                # handed exactly the conditions derived from split == 'test'
                # (docker/CONTRACT.md), so the split is known.
                actual_split = "test"

                control_mask = (
                    (adata.obs[self.perturb_field] == self.control_key) &
                    (adata.obs[self.split_name] == actual_split)
                )
                combined_mask = condition_mask | control_mask
                split_adata = adata[combined_mask]
                split_adata = split_adata[split_adata.obs[self.split_name] == actual_split]

                # Post-condition: nothing outside the held-out split may reach the
                # test set. The filter above already guarantees it; assert so a
                # future edit that loosens it fails here instead of scoring.
                leaked = (split_adata.obs[self.split_name].astype(str) != actual_split).sum()
                if leaked:
                    raise RuntimeError(
                        f"{leaked} cells not in split {actual_split!r} reached the test "
                        f"dataset — refusing to predict from non-held-out cells")
                if len(split_adata) == 0:
                    raise ValueError(
                        f"no cells in split {actual_split!r} for the requested "
                        f"conditions — the model would predict nothing")

                self.test_dataset = self.create_dataset(
                    split_adata,
                    train=False,
                )

                log.info(f"Created test dataset (split = {actual_split!r}) "
                         f"with {combined_mask.sum()} samples "
                         f"({condition_mask.sum()} conditions + {control_mask.sum()} controls)")
            
            self._data_setup = True
    
    @classmethod
    def from_config(cls, config: Dict):
        """Create datamodule from config dict, compatible with PRESAGE."""
        from copy import deepcopy
        import random
        import numpy as np
        import torch
        
        config = deepcopy(config)
        
        # Set seed if provided
        if 'seed' in config:
            seed = config['seed']
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
        
        # Extract required paths - will raise KeyError if missing
        processed_adata_path = config.pop('processed_adata_path')
        splits_path = config.pop('splits_path')
        
        # Handle dataset_class like parent from_config
        config["dataset_class"] = CellSimBenchscPerturbData
        # Create instance with remaining config as kwargs
        return cls(
            processed_adata_path=processed_adata_path,
            splits_path=splits_path,
            **config
        ) 
    
    def load_preprocessed(self):
        log.info("Loading adata...")
        adata = sc.read(self.preprocessed_path)

        if hasattr(adata.X, "toarray"):
            adata.X = adata.X.toarray()
        self.n_genes = adata.shape[1]

        # Derive cov_categories from full adata if covariate_field is set but
        # categories weren't pinned by the caller. We sort for determinism so
        # the one-hot encoding is identical across train and inference runs
        # provided the adata's covariate column hasn't changed.
        if self.covariate_field is not None and self.cov_categories is None:
            if self.covariate_field not in adata.obs.columns:
                raise KeyError(
                    f"covariate_field={self.covariate_field!r} not in adata.obs.columns "
                    f"({list(adata.obs.columns)[:8]}...)"
                )
            self.cov_categories = sorted(adata.obs[self.covariate_field].astype(str).unique().tolist())
            log.info(
                f"Derived cov_categories from adata: {self.cov_categories} "
                f"(field={self.covariate_field})"
            )
        log.info("Loading DEGs...")
        with open(self.merged_deg_file) as fp:
            self.degs = json.load(fp)
        deg_dir = "/".join(self.merged_deg_file.split("/")[:-1])

        parent_data_dir = "/".join(deg_dir.split("/")[:-1]) + "/"

        # perturbation cluster file for eval
        self.pclust_file = parent_data_dir + "eval.stratification.clusters.json"
        # genesets for virtual screen
        self.gs_file = parent_data_dir + "virtual.screen.genesets.json"

        # Find the file matching f"{parent_data_dir}/ncells_per_perturbation*"
        import glob
        ncells_per_perturbation_files = glob.glob(f"{parent_data_dir}/ncells_per_perturbation*")
        if len(ncells_per_perturbation_files) == 0:
            self.ncells_per_perturbation_file = None
        else:
            self.ncells_per_perturbation_file = ncells_per_perturbation_files[0]

        if self.ncells_per_perturbation_file is not None:
            cells_per_perturbation = dict(adata.obs.value_counts("perturbation"))
            cells_per_perturbation_temp = {
                i: int(j) for i, j in cells_per_perturbation.items()
            }
            with open(self.ncells_per_perturbation_file, "w") as f:
                json.dump(cells_per_perturbation_temp, f)

        self.var_names = adata.var_names
        self.pseudobulk = compute_pseudobulk(adata, self.perturb_field)

        self.centered_pseudobulk = (
            self.pseudobulk - self.pseudobulk.loc[self.control_key]
        )

        return adata
