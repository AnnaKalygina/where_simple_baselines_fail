#!/usr/bin/env python
"""Download and preprocess the Replogle et al. 2020 perturbation dataset.

Source : GEO GSE146194 (sample GSM4367984: 10x mtx + features + barcodes
         + cell_identities metadata).
Cell type: K562 (single bin).
Scenarios applicable: UnseenCombo (2 folds).

Conditions are reconstructed from the cell_identities metadata:
  - control flag set       -> "control"
  - double flag set        -> "{gene_A}+{gene_B}" (sorted)
  - single flag set        -> the non-NegCtrl gene
"""
from __future__ import annotations

import gzip
import logging
import shutil
import subprocess as sp
import sys
import urllib.request
from pathlib import Path
from typing import Dict

import anndata as ad
import pandas as pd
import scanpy as sc
from scipy.io import mmread
from scipy.sparse import csr_matrix, issparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import _utils as u  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).resolve().parent
DATASET_NAME = "replogle20"
OUTPUT_PATH = OUTPUT_DIR / f"{DATASET_NAME}_processed.h5ad"

# ---- explicit per-dataset declarations (never inferred) ---------------------
BATCH_COLUMN = "gemgroup"
# The condition label is DERIVED from this dataset's own guide calls (the
# control/single/double flags + gene_A/gene_B in cell_identities.csv), so a guide
# gate would just re-check the labelling that produced it.
GUIDE_COLUMN = None
# The matrix holds the WHOLE cell, so its own row sums ARE the sequencing depth and
# no depth column is needed. Not taken on trust: `u.count_retention` measures the
# matrix against the depth column the source ships and raises below 90%, which is
# what an earlier gene-count-based classification failed to do.
SOURCE_DEPTH_COLUMN = None
N_HVG = 8192

GEO_BASE = "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM4367nnn/GSM4367984/suppl"
GEO_FILES = {
    "matrix": "GSM4367984_exp6.matrix.mtx",
    "features": "GSM4367984_exp6.features.tsv",
    "barcodes": "GSM4367984_exp6.barcodes.tsv",
    "metadata": "GSM4367984_exp6.cell_identities.csv",
}


def download() -> Dict[str, Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, Path] = {}
    for key, raw_name in GEO_FILES.items():
        raw = OUTPUT_DIR / raw_name
        if raw.exists():
            paths[key] = raw
            continue
        gz = OUTPUT_DIR / f"{raw_name}.gz"
        if not gz.exists():
            url = f"{GEO_BASE}/{raw_name}.gz"
            log.info("Downloading %s", url)
            urllib.request.urlretrieve(url, str(gz))
        log.info("Decompressing %s", gz.name)
        with gzip.open(str(gz), "rb") as fin, open(str(raw), "wb") as fout:
            shutil.copyfileobj(fin, fout)
        paths[key] = raw
    return paths


def _is_true(value: object) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, bool):
        return value
    try:
        return float(value) == 1.0
    except (TypeError, ValueError):
        return False


def build_condition(row: pd.Series) -> str:
    if _is_true(row.get("control")):
        return "control"
    if _is_true(row.get("double")):
        genes = sorted([str(row["gene_A"]), str(row["gene_B"])])
        return f"{genes[0]}+{genes[1]}"
    if _is_true(row.get("single")):
        a, b = str(row["gene_A"]), str(row["gene_B"])
        a_is_neg, b_is_neg = "NegCtrl" in a, "NegCtrl" in b
        if a_is_neg and not b_is_neg:
            return b
        if b_is_neg and not a_is_neg:
            return a
        raise ValueError(f"Ambiguous single-pert row (gene_A={a}, gene_B={b}).")
    raise ValueError("Row is not labeled as control, single, or double.")


def load_and_harmonize() -> ad.AnnData:
    paths = download()
    with paths["barcodes"].open() as f:
        barcodes = [line.strip() for line in f if line.strip()]
    features = pd.read_csv(
        paths["features"], sep="\t", header=None,
        names=["gene_id", "gene_name", "feature_type"],
    )
    cell_metadata = pd.read_csv(paths["metadata"])
    raw_matrix = mmread(str(paths["matrix"])).tocsc()

    n_features, n_barcodes = features.shape[0], len(barcodes)
    if raw_matrix.shape == (n_features, n_barcodes):
        gene_by_cell = raw_matrix
    elif raw_matrix.shape == (n_barcodes, n_features):
        gene_by_cell = raw_matrix.T.tocsc()
    else:
        raise ValueError(f"Matrix shape {raw_matrix.shape} does not match data.")

    barcode_to_index = {bc: i for i, bc in enumerate(barcodes)}
    column_indices = [barcode_to_index[bc] for bc in cell_metadata["cell_barcode"]]
    aligned = gene_by_cell[:, column_indices].T.tocsr()

    var = features.copy()
    var.index = var["gene_name"].astype(str)
    var.index.name = None

    adata = ad.AnnData(X=aligned, obs=cell_metadata.copy(), var=var)
    # Stash the metadata-derived label into 'perturbation' so apply_label_cleanup
    # in main() can do the universal sentinel/guide-suffix cleanup.
    adata.obs["perturbation"] = adata.obs.apply(build_condition, axis=1).astype(str)
    adata.obs["cell_type"] = "K562"
    adata.obs["batch"] = (adata.obs[BATCH_COLUMN].astype(str).values
                          if BATCH_COLUMN else DATASET_NAME)

    dup_mask = adata.var_names.duplicated(keep=False)
    if dup_mask.any():
        adata = adata[:, ~dup_mask].copy()
    adata.var.index.name = None

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
