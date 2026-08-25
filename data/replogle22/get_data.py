#!/usr/bin/env python
"""Download and preprocess the Replogle et al. 2022 perturbation dataset.

Source : Zenodo 10044268, two h5ad files (K562_essential + RPE1).
Cell types: K562, RPE1 (two bins).
Scenarios applicable: UnseenPert (5 folds), UnseenBoth (5 folds).

Concatenates the two h5ads on the gene intersection, then runs the standard
pipeline with `bin_col="cell_type"`. Before fold assignment, perturbations are
restricted to those shared across BOTH cell types (>= 4 cells in each half of
each bin) via restrict_to_full_coverage_perturbations, so the obs fold columns
and the pseudobulk/DEG arrays cover the same set. UnseenPert uses the multi-bin
block partition (assign_split_folds_unseen_pert_multibin), matching how the
upstream DL pipeline derives UnseenPert from the same partition as UnseenBoth.
"""
from __future__ import annotations

import logging
import sys
import urllib.request
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
DATASET_NAME = "replogle22"
OUTPUT_PATH = OUTPUT_DIR / f"{DATASET_NAME}_processed.h5ad"

# ---- explicit per-dataset declarations (never inferred) ---------------------
# 56 gem groups carried through from the two source files — real batch structure,
# not confounded with cell_type (both lines span many gem groups).
BATCH_COLUMN = "batch"
# No per-cell guide-identity column in this source.
GUIDE_COLUMN = None
# The matrix holds the WHOLE cell, so its own row sums ARE the sequencing depth and
# no depth column is needed. Not taken on trust: `u.count_retention` measures the
# matrix against the depth column the source ships and raises below 90%, which is
# what an earlier gene-count-based classification failed to do.
SOURCE_DEPTH_COLUMN = None
N_HVG = 8192

URLS = {
    "K562": "https://zenodo.org/record/10044268/files/ReplogleWeissman2022_K562_essential.h5ad",
    "RPE1": "https://zenodo.org/record/10044268/files/ReplogleWeissman2022_rpe1.h5ad",
}


def download() -> dict[str, Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    paths = {}
    for cell_type, url in URLS.items():
        path = OUTPUT_DIR / f"{DATASET_NAME}_{cell_type.lower()}_downloaded.h5ad"
        if not path.exists():
            log.info("Downloading %s from %s", cell_type, url)
            urllib.request.urlretrieve(url, str(path))
        paths[cell_type] = path
    return paths


def load_and_harmonize() -> ad.AnnData:
    paths = download()
    adatas = []
    for cell_type, path in paths.items():
        a = ad.read_h5ad(str(path))
        dup = a.var_names.duplicated(keep=False)
        if dup.any():
            a = a[:, ~dup].copy()
        # Stash raw perturbation; universal cleanup writes obs['condition'] in main()
        a.obs["perturbation"] = a.obs["perturbation"].astype(str).values
        a.obs["cell_type"] = cell_type
        a.obs["batch"] = a.obs[BATCH_COLUMN].astype(str).values
        if not issparse(a.X):
            a.X = csr_matrix(a.X)
        adatas.append(a)

    adata = ad.concat(adatas, join="inner", index_unique="-")
    adata.var.index.name = None
    return adata


def _guide_target_counts(adata: ad.AnnData):
    """No guide-identity column for this dataset — the MOI gate is recorded
    as skipped rather than silently doing nothing."""
    return None


def build_through_pseudobulk() -> ad.AnnData:
    """The deterministic, resumable prefix: everything up to and including pseudobulk."""
    adata = load_and_harmonize()
    adata.uns["dataset_name"] = DATASET_NAME
    log.info("Loaded %d cells x %d genes (cell_types=%s)",
             adata.n_obs, adata.n_vars,
             sorted(adata.obs["cell_type"].unique().tolist()))

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

    # HVG on the FULL QC-passing cell set, before any subsampling, with the bin
    # axis as batch_key so a gene variable in only one cell type cannot dominate.
    u.select_hvg(adata, n_top_genes=N_HVG, batch_key="cell_type")
    u.annotate_perturbation_targets(adata)
    adata = adata[:, u.union_panel_mask(adata)].copy()
    log.info("After HVG subset: %d cells x %d genes", adata.n_obs, adata.n_vars)

    adata = u.downsample_per_condition(adata, bin_col="cell_type")
    log.info("After condition floor / control cap: %d cells", adata.n_obs)

    u.assign_tech_dup_split(adata, bin_col="cell_type")

    # Cross-cell-line intersection: keep only perturbations present in BOTH bins
    # (same predicate compute_pseudobulk uses for ko_names) so the obs fold columns
    # and the pseudobulk/DEG arrays cover the identical perturbation set. Must run
    # before the fold assignments (otherwise single-cell-line perts shift folds).
    adata = u.restrict_to_full_coverage_perturbations(adata, bin_col="cell_type")
    log.info("After full-coverage restriction: %d cells x %d genes",
             adata.n_obs, adata.n_vars)

    # UnseenPert: this multi-bin dataset's DL splits derive UnseenPert from the
    # SAME seed-0 block partition as UnseenBoth (perturbench convention), not the
    # single-bin PMOB round-robin. (Scoped to replogle22 — single-bin datasets
    # keep assign_split_folds_unseen_pert.)
    u.assign_split_folds_unseen_pert_multibin(adata, n_folds=5, seed=0)
    u.assign_split_folds_unseen_both(adata, bin_col="cell_type", n_folds=5, seed=0)

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
