#!/usr/bin/env python
"""Download and preprocess the Jiang et al. 2024 perturbation dataset.

Source : HuggingFace altoslabs/perturbench
         (jiang24_processed.h5ad.gz).
Cell types: A549, BxPC3, HAP1, HT29, K562, MCF7 (six bins).
Scenarios applicable: UnseenPert (5 folds), UnseenCell (5 folds),
                      UnseenBoth (5 folds), UnseenPair (5 folds).

All treatment conditions are pooled together (they are part of the
perturbation design rather than a confounder to remove).

Before fold assignment, perturbations are restricted to those present across
ALL six cell types (>= 3 cells in each tech-dup half of each bin) via
restrict_to_full_coverage_perturbations, so the obs fold columns and the
pseudobulk/DEG arrays cover the same set. UnseenPert uses the multi-bin block
partition (assign_split_folds_unseen_pert_multibin, seed 0), matching how the
upstream DL pipeline derives UnseenPert from the same partition as UnseenBoth
(verified bit-identical to the DL folds).
"""
from __future__ import annotations

import gzip
import logging
import shutil
import sys
import urllib.request
from pathlib import Path

import anndata as ad
import numpy as np
import scanpy as sc
from scipy.sparse import csr_matrix, issparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import _utils as u  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).resolve().parent
DATASET_NAME = "jiang24"
OUTPUT_PATH = OUTPUT_DIR / f"{DATASET_NAME}_processed.h5ad"

# ---- explicit per-dataset declarations (never inferred) ---------------------
# Rep1/Rep2 biological replicates. `sample` (33) confounds cell type with treatment.
BATCH_COLUMN = "Batch_info"
# No per-cell guide-identity column in this source.
GUIDE_COLUMN = None
# The source ships a GENE PANEL: its matrix holds only 46.0% of each cell's counts
# (measured against obs['ncounts']). Its row sums are therefore NOT the sequencing
# depth, and every statistic with "total counts" in the denominator -- the
# mitochondrial fraction, CP10K -- must divide by the shipped depth instead.
# Computed on the panel the mitochondrial fraction reads 0.1446 against a true
# 0.0526 -- a 2.6x over-estimate that once deleted 26.8% of this dataset.
SOURCE_DEPTH_COLUMN = "ncounts"
N_HVG = 8192
DOWNLOAD_URL = (
    "https://huggingface.co/datasets/altoslabs/perturbench/"
    "resolve/main/jiang24_processed.h5ad.gz"
)


def download() -> Path:
    h5ad_path = OUTPUT_DIR / f"{DATASET_NAME}_downloaded.h5ad"
    if h5ad_path.exists():
        return h5ad_path
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    gz_path = OUTPUT_DIR / f"{DATASET_NAME}_downloaded.h5ad.gz"
    if not gz_path.exists():
        log.info("Downloading from HuggingFace: %s", DOWNLOAD_URL)
        urllib.request.urlretrieve(DOWNLOAD_URL, str(gz_path))
    log.info("Decompressing %s", gz_path.name)
    with gzip.open(str(gz_path), "rb") as fin, open(str(h5ad_path), "wb") as fout:
        shutil.copyfileobj(fin, fout)
    return h5ad_path


def load_and_harmonize() -> ad.AnnData:
    adata = ad.read_h5ad(str(download()))
    adata.obs["condition"] = adata.obs["condition"].astype(str)
    adata.obs["batch"] = adata.obs[BATCH_COLUMN].astype(str).values
    if not issparse(adata.X):
        adata.X = csr_matrix(adata.X)
    return adata


def _guide_target_counts(adata: ad.AnnData):
    """No guide-identity column for this dataset — the MOI gate is recorded
    as skipped rather than silently doing nothing."""
    return None


def build_through_pseudobulk() -> ad.AnnData:
    """The deterministic, resumable prefix: everything up to and including pseudobulk."""
    adata = load_and_harmonize()
    adata.uns["dataset_name"] = DATASET_NAME
    log.info("Loaded %d cells x %d genes", adata.n_obs, adata.n_vars)
    log.info("Cell types: %s", sorted(adata.obs["cell_type"].unique().tolist()))

    adata = u.apply_label_cleanup(
        adata, raw_col="condition", target_col="condition",
        audit_path=OUTPUT_DIR / "preprocessing_audit.csv",
    )

    # --- per-cell QC: min_genes + mitochondrial fraction + guide MOI --------
    # Verify the SOURCE_DEPTH_COLUMN declaration against the data before anything
    # depends on it. Runs on the source gene universe, before gene QC.
    u.count_retention(adata, depth_column=SOURCE_DEPTH_COLUMN,
                      dataset_name=DATASET_NAME)

    adata = u.apply_cell_qc(
        adata, dataset_name=DATASET_NAME, depth_column=SOURCE_DEPTH_COLUMN,
        guide_n_targets=_guide_target_counts(adata),
        audit_path=OUTPUT_DIR / "cell_qc_audit.csv",
    )
    adata = u.filter_genes_keeping_targets(adata, min_cells=3)

    # One normalisation for every dataset, computed from the counts layer rather
    # than from whatever X happens to hold. This source ships its own log-normalised
    # X; it is discarded and rebuilt here so all nine land on one scale instead of
    # the 4x spread they carried before. Its integral counts layer is required.
    if "counts" not in adata.layers:
        adata.layers["counts"] = adata.X.copy()
    u.normalize_from_counts(adata, target_sum=1e4, depth_column=SOURCE_DEPTH_COLUMN)
    sc.pp.log1p(adata)
    if adata.X.dtype != np.float32:
        adata.X = adata.X.astype(np.float32)

    # HVG on the FULL QC-passing cell set, before any subsampling, with the bin
    # axis as batch_key so a gene variable in only one cell type cannot dominate.
    u.select_hvg(adata, n_top_genes=N_HVG, batch_key="cell_type")
    u.annotate_perturbation_targets(adata)
    adata = adata[:, u.union_panel_mask(adata)].copy()
    log.info("After HVG subset: %d cells x %d genes", adata.n_obs, adata.n_vars)

    adata = u.downsample_per_condition(adata, bin_col="cell_type")
    log.info("After condition floor / control cap: %d cells", adata.n_obs)

    u.assign_tech_dup_split(adata, bin_col="cell_type")

    # Cross-cell-type intersection: keep only perturbations present in all 6 bins
    # (same predicate compute_pseudobulk uses for ko_names) so the obs fold
    # columns and the pseudobulk/DEG arrays cover the identical perturbation set.
    # Must run before the fold assignments so the CV shuffle splits only the
    # shared perturbations (otherwise partial-coverage perts shift every fold).
    adata = u.restrict_to_full_coverage_perturbations(adata, bin_col="cell_type")
    log.info("After full-coverage restriction: %d cells x %d genes",
             adata.n_obs, adata.n_vars)

    # UnseenPert: this multi-bin dataset's DL splits derive UnseenPert from the
    # SAME seed-0 block partition as UnseenBoth (perturbench convention), not the
    # single-bin PMOB round-robin. Use the multibin variant so the per-fold test
    # perturbation set matches the DL models (verified bit-identical to the
    # DL-aligned folds: sym_diff=0 across all 5 folds).
    u.assign_split_folds_unseen_pert_multibin(adata, n_folds=5, seed=0)
    u.assign_split_folds_unseen_cell(adata, bin_col="cell_type", n_folds=5, seed=0)
    u.assign_split_folds_unseen_both(adata, bin_col="cell_type", n_folds=5, seed=0)
    u.assign_split_folds_unseen_pair(adata, bin_col="cell_type", n_folds=5, seed=0)

    adata.uns["pseudobulk"] = u.compute_pseudobulk(adata, bin_col="cell_type")
    return adata


def main():
    # Resume point. The DEG stage below is the long pole and the stage we keep
    # retuning, so the prefix above is checkpointed and skipped when it is still
    # valid. The stamp hashes _utils.py + this file, so any code edit invalidates it.
    ckpt = u.predeg_path(OUTPUT_DIR, DATASET_NAME)
    stamp = u.recipe_stamp(
        __file__, depth_column=SOURCE_DEPTH_COLUMN, batch_column=BATCH_COLUMN,
        guide_column=GUIDE_COLUMN, n_hvg=N_HVG,
    )
    adata = u.load_predeg(ckpt, stamp)
    if adata is None:
        adata = build_through_pseudobulk()
        u.save_predeg(adata, ckpt, stamp)

    u.compute_degs_pdex(adata, half="first_half", bin_col="cell_type",
                   dataset_name=DATASET_NAME)
    u.compute_degs_pdex(adata, half="second_half", bin_col="cell_type",
                   dataset_name=DATASET_NAME)
    # All cells, not one half: these arrays weight and mask the scoring, and
    # deriving them from half the cells makes the weighting noisier than the data.
    u.compute_degs_pdex(adata, half="all", bin_col="cell_type",
                        dataset_name=DATASET_NAME)
    adata.uns["deg_arrays"] = u.compute_deg_arrays(
        adata, bin_col="cell_type", dataset_name=DATASET_NAME,
    )

    u.assert_split_invariants(adata)
    log.info("Writing %s", OUTPUT_PATH)
    u.write_h5ad_compressed(adata, OUTPUT_PATH)
    u.clear_predeg(ckpt)
    log.info("Done. Final shape: %d cells x %d genes", adata.n_obs, adata.n_vars)


if __name__ == "__main__":
    main()
