#!/usr/bin/env python
"""Download and preprocess the Sunshine et al. 2023 perturbation dataset.

Source : Zenodo 10044268 (SunshineHein2023.h5ad).
Cell type: Calu3 (single bin).
Scenarios applicable: UnseenCombo (2 folds).

Standardizes the combo separator from '_' to '+' at load time.
"""
from __future__ import annotations

import logging
import subprocess as sp
import sys
from pathlib import Path

import anndata as ad
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
DATASET_NAME = "sunshine23"
OUTPUT_PATH = OUTPUT_DIR / f"{DATASET_NAME}_processed.h5ad"

# ---- explicit per-dataset declarations (never inferred) ---------------------
BATCH_COLUMN = None   # no lane/gem-group covariate in this source
# `guide_id` tokens are `GENE_<strand>_<coord>...`, and non-targeting guides are
# literally named `non-targeting_*` (already in _utils' NT token set). Verified:
# every condition's label genes appear among its parsed guide genes (0% mismatch),
# and once NT guides are excluded the gate finds 0 violations — the dual constructs
# pair one targeting guide with one non-targeting guide, exactly as designed.
GUIDE_COLUMN = "guide_id"
# The matrix holds the WHOLE cell, so its own row sums ARE the sequencing depth and
# no depth column is needed. Not taken on trust: `u.count_retention` measures the
# matrix against the depth column the source ships and raises below 90%, which is
# what an earlier gene-count-based classification failed to do.
SOURCE_DEPTH_COLUMN = None
N_HVG = 8192
DOWNLOAD_URL = "https://zenodo.org/record/10044268/files/SunshineHein2023.h5ad"


def download() -> Path:
    path = OUTPUT_DIR / f"{DATASET_NAME}_downloaded.h5ad"
    if not path.exists():
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        log.info("Downloading from %s", DOWNLOAD_URL)
        sp.check_call(["wget", "-q", DOWNLOAD_URL, "-O", str(path)])
    return path


def load_and_harmonize() -> ad.AnnData:
    adata = sc.read_h5ad(str(download()))
    # Standardize combo separator '_' -> '+' on the raw column; universal
    # cleanup runs in main() via apply_label_cleanup.
    adata.obs["perturbation"] = (
        adata.obs["perturbation"].astype(str).str.replace("_", "+", regex=False).values
    )
    adata.obs["cell_type"] = "Calu3"
    adata.obs["batch"] = (adata.obs[BATCH_COLUMN].astype(str).values
                          if BATCH_COLUMN else DATASET_NAME)
    # Retain guide identity alongside the collapsed gene-level label.
    adata.obs["guide"] = adata.obs[GUIDE_COLUMN].astype(str).values

    dup_mask = adata.var_names.duplicated(keep=False)
    if dup_mask.any():
        adata = adata[:, ~dup_mask].copy()
    if "ensembl_id" in adata.var.columns:
        adata.var["gene_id"] = adata.var["ensembl_id"]
    adata.var.index.name = None

    if not issparse(adata.X):
        adata.X = csr_matrix(adata.X)
    return adata


def _guide_target_counts(adata: ad.AnnData):
    """Per-cell targeting-gene count parsed from this dataset's guide IDs."""
    return u.guide_target_sets(
        adata, GUIDE_COLUMN, token_to_gene=lambda t: t.split("_")[0],
    ).map(lambda g: None if g is None else len(g))


def build_through_pseudobulk() -> ad.AnnData:
    """The deterministic, resumable prefix: everything up to and including pseudobulk."""
    adata = load_and_harmonize()
    adata.uns["dataset_name"] = DATASET_NAME
    log.info("Loaded %d cells x %d genes", adata.n_obs, adata.n_vars)

    adata = u.apply_label_cleanup(
        adata, audit_path=OUTPUT_DIR / "preprocessing_audit.csv",
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
    # than from whatever X happens to hold. Sources that ship their own
    # log-normalised X (mcfaline23, jiang24) have it discarded and rebuilt here, so
    # all nine land on one scale instead of the 4x spread they carried before.
    if "counts" not in adata.layers:
        adata.layers["counts"] = adata.X.copy()
    u.normalize_from_counts(adata, target_sum=1e4, depth_column=SOURCE_DEPTH_COLUMN)
    sc.pp.log1p(adata)

    # HVG on the FULL QC-passing cell set, before any subsampling.
    u.select_hvg(adata, n_top_genes=N_HVG, batch_key=None)
    u.annotate_perturbation_targets(adata)
    adata = adata[:, u.union_panel_mask(adata)].copy()
    log.info("After HVG subset: %d cells x %d genes", adata.n_obs, adata.n_vars)

    adata = u.downsample_per_condition(adata, bin_col=None)
    log.info("After condition floor / control cap: %d cells", adata.n_obs)

    u.assign_tech_dup_split(adata, bin_col=None)
    u.assign_split_folds_unseen_combo(adata, n_folds=2, seed=42)

    adata.uns["pseudobulk"] = u.compute_pseudobulk(adata, bin_col=None)
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

    u.compute_degs_pdex(adata, half="first_half", bin_col=None, dataset_name=DATASET_NAME)
    u.compute_degs_pdex(adata, half="second_half", bin_col=None, dataset_name=DATASET_NAME)
    # All cells, not one half: these arrays weight and mask the scoring, and
    # deriving them from half the cells makes the weighting noisier than the data.
    u.compute_degs_pdex(adata, half="all", bin_col=None,
                        dataset_name=DATASET_NAME)
    adata.uns["deg_arrays"] = u.compute_deg_arrays(
        adata, bin_col=None, dataset_name=DATASET_NAME,
    )

    u.assert_split_invariants(adata)
    log.info("Writing %s", OUTPUT_PATH)
    u.write_h5ad_compressed(adata, OUTPUT_PATH)
    u.clear_predeg(ckpt)
    log.info("Done. Final shape: %d cells x %d genes", adata.n_obs, adata.n_vars)


if __name__ == "__main__":
    main()
