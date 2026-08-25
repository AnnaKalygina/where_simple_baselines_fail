"""Shared preprocessing helpers for every data/{dataset}/get_data.py.

Each get_data.py calls these inline — there is NO separate postprocessing step.
The h5ad written at the end of get_data.py has every slot a benchmark consumer
needs: pseudobulk arrays, per-half DEG matrices, derived deg_arrays (var_names-aligned),
and split fold columns.

Algorithms
==========

* Pseudobulk: literature (PMOB cellsimbench) formula `X[mask].mean(axis=0)`
  per (bin, condition, tech_dup_split). Stored as ABSOLUTE means in
  `adata.uns["pseudobulk"]`; consumers compute deltas as `bulk - ctrl_bulk`.

* DEGs: ArcInstitute `pdex` (the engine behind cell-eval), Mann-Whitney vs the
  CONTROL group, filtered by `pert_counts >= min_cells` (default 4).
  Computed twice per dataset:
  `half="first_half"` produces `*_df_dict_first_half`; `half="second_half"`
  produces `*_df_dict_second_half`. Both halves explicitly marked.

  For datasets with too many perturbations to run scanpy per-group
  (xatlas_orion has ~18k singles), `compute_degs_vectorized` computes the
  exact same overestim-var t-test in a single sparse matmul per half, and
  populates var_names-aligned score/pval matrices in adata.uns directly
  (the per-pert dicts are written as EMPTY to keep the schema consistent).

* deg_arrays: derived from the **first_half** dicts by reindexing each
  per-perturbation vector to var_names order. Stores three aligned tensors:
  `per_pert_weights`, `deg_mask`, `deg_directions`. Raw scores/pvals stay
  in the per-pert dicts (literature format). When the per-pert dicts are
  empty and the vectorized score/pval matrices are present, deg_arrays is
  derived directly from the matrices (no reindexing needed).

* Split folds: literature (PMOB) for UnseenPert and UnseenCombo (single-bin).
  Multi-bin scenarios (UnseenCell, UnseenBoth, UnseenPair) follow the
  existing VCR mcfaline23 pattern (`_partition_into_blocks` + `_fold_assignment`)
  since the literature does not cover multi-cell-type splits.

Scenario names everywhere are PascalCase: UnseenPert, UnseenCell, UnseenBoth,
UnseenPair, UnseenDose, UnseenCombo.

Public API (called from get_data.py)
====================================

Single-cell preprocessing:
    downsample_per_condition(adata, ...) -> AnnData
    force_include_perturbation_targets_in_hvg(adata, ...) -> None
    assign_tech_dup_split(adata, ...) -> None

Split fold assignment (one function per scenario):
    assign_split_folds_unseen_pert(adata, ...) -> None
    assign_split_folds_unseen_cell(adata, ...) -> None
    assign_split_folds_unseen_both(adata, ...) -> None
    assign_split_folds_unseen_pair(adata, ...) -> None
    assign_split_folds_unseen_dose(adata, ...) -> None
    assign_split_folds_unseen_combo(adata, ...) -> None

Split invariants (shared by get_data.py + benchmark/verify.py + pytest):
    check_no_leakage / check_no_empty_fold / check_pert_consistent_across_bins /
    check_full_coverage / check_unseenpert_subset_of_unseenboth   # pure predicates
    assert_split_invariants(adata, ...) -> None   # generation-time gate (raises)

DEG + pseudobulk + deg_arrays:
    compute_degs_pdex(adata, half=..., ...) -> None  # called twice per dataset
    compute_pseudobulk(adata, ...) -> dict           # assign to adata.uns["pseudobulk"]
    compute_deg_arrays(adata, ...) -> dict           # assign to adata.uns["deg_arrays"]

Pre-DEG checkpoint (the resume seam; DEGs are the long pole and the tunable stage):
    recipe_stamp(__file__, **declared_params) -> str   # invalidates on ANY code edit
    predeg_path(output_dir, dataset) -> Path
    load_predeg(path, stamp) -> AnnData | None         # None if absent/stale
    save_predeg(adata, path, stamp) -> None
    clear_predeg(path) -> None                         # after the final write

Parallelism:
    allocated_cpus() -> int                       # SLURM allocation, not the node
    pdex_parallelism(workers, threads) -> (w, t)  # num_threads has a HARD floor of 2
"""
from __future__ import annotations

import hashlib
import logging
import os
import re
import zlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import diags, issparse

log = logging.getLogger(__name__)

# NOTE: mcfaline23 carries a control-selection label "RcontrolSEL" that is NOT
# matched here, so it currently rides along as a pseudo-perturbation (n_kos counts
# it; it canonicalizes away in cross-model comparisons). The upstream DL pipeline
# has the same gap, so fold alignment is unaffected. To treat it as a control, add
# "rcontrolsel" to the set below — left out deliberately for now:
#   CONTROL_LABELS = {"control", "ctrl", "non-targeting", "rcontrolsel"}
CONTROL_LABELS = {"control", "ctrl", "non-targeting"}
MIN_CELLS_DEGS_DEFAULT = 4

_SENTINEL_TOKENS = {"", "nan", "*", "?", "none", "null"}
_GUIDE_SUFFIX_RE = re.compile(r"_\d+$")

# --- pdex parallelism -------------------------------------------------------
# `pdex` has two implementations of the SAME Mann-Whitney test and dispatches on
# `low_memory`. Unset + in-memory data lands in the STANDARD path, which densifies
# the matrix, enumerates n_targets*n_genes python-level combinations and ships them
# through mp.Pool.imap in batches of 100 dicts — and silently IGNORES gene_chunk_size
# (documented low-memory-only). The numba @njit(parallel=True) ranksum kernels, which
# are the fast part of pdex, are reachable only from the chunked path. Despite the
# name, `low_memory=True` is the modern path: measured 6.8x faster than standard at
# EQUAL core count, 27x faster than what this pipeline ran before, and bit-identical
# (492,240 comparisons, 51,647 significant calls under both, Jaccard 1.0000).
#
# PDEX_MIN_THREADS is a floor, not a default: num_threads=1 disables numba entirely
# and measured 1296.8s vs 179.9s for the old config — i.e. setting it to 1 makes this
# change a 7x REGRESSION. The numba threads do the work; the target-level workers only
# feed them.
PDEX_MIN_THREADS = 2
PDEX_GENE_CHUNK_DEFAULT = 4096

# pdex SUBSTITUTES this fold change when a pseudobulk mean is zero, because the ratio
# is undefined there: control mean 0 -> clip_value, perturbed mean 0 -> 1/clip_value,
# both zero -> 1. It is a declared placeholder meaning "undefined", not a measurement,
# and log2 of it lands at exactly +/-4.3219.
#
# We pin it rather than inherit pdex's default so the constant is visible to
# `compute_deg_arrays`, which must EXCLUDE these entries from the gene weights. Read
# as data they take 82-89% of all weight mass while 97-99% of them fail the
# significance test -- non-significant genes ended up carrying 1.7x the weight of
# significant ones. `deg_fc_substituted_*` marks them; see `compute_deg_arrays`.
#
# Not replaced by an epsilon: `log2((0+e)/(ref+e))` swings from -18.5 to -0.00 across
# plausible e, and in the range where it looks reasonable 87-98% of the values land
# inside the real log2FC range -- indistinguishable from measurements, where the fixed
# placeholder is self-labelling and therefore findable.
PDEX_CLIP_VALUE = 20.0


def allocated_cpus() -> int:
    """CPUs this process may actually use.

    Prefers SLURM's allocation over `os.cpu_count()`, which reports the whole node
    and would oversubscribe the cgroup (the pipeline previously asked pdex for 8
    workers inside a 4-CPU allocation).
    """
    for var in ("SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE"):
        val = os.environ.get(var)
        if val and val.isdigit() and int(val) > 0:
            return int(val)
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:
        return max(1, os.cpu_count() or 1)


def pdex_parallelism(num_workers: Optional[int] = None,
                     num_threads: Optional[int] = None) -> Tuple[int, int]:
    """(num_workers, num_threads) for pdex, derived from the CPU allocation.

    Measured optimum is workers ~= cpus/2 with 2 numba threads each (norman19,
    36 CPUs: w=16 t=2 -> 6.6s; w=8 t=4 -> 11.3s; w=4 t=8 -> 8.3s; w=32 t=1 -> 1296.8s).
    """
    cpus = allocated_cpus()
    workers = int(num_workers) if num_workers else max(1, cpus // 2)
    threads = int(num_threads) if num_threads else PDEX_MIN_THREADS
    if threads < PDEX_MIN_THREADS:
        raise ValueError(
            f"pdex num_threads={threads} disables numba parallelization, which is ~7x "
            f"SLOWER than the pre-fix configuration. Minimum is {PDEX_MIN_THREADS}.")
    return max(1, workers), threads


# ===================================================================
# Helpers
# ===================================================================

def _is_control_label(label: str) -> bool:
    return str(label).strip().lower() in CONTROL_LABELS


def _non_control_conditions(adata: ad.AnnData, pert_col: str = "condition") -> List[str]:
    return sorted([c for c in adata.obs[pert_col].unique() if not _is_control_label(c)])


def _control_conditions(adata: ad.AnnData, pert_col: str = "condition") -> List[str]:
    return [c for c in adata.obs[pert_col].unique() if _is_control_label(c)]


def apply_label_cleanup(
    adata: ad.AnnData,
    *,
    raw_col: str = "perturbation",
    target_col: str = "condition",
    audit_path=None,
) -> ad.AnnData:
    """Apply `clean_perturbation_label` to `adata.obs[raw_col]`.

    Drops cells whose labels parse to None (NaN / sentinel). Writes the
    cleaned label into `adata.obs[target_col]` as a categorical. Optionally
    writes an audit CSV at `audit_path` recording per-label sentinel drops
    and renames (e.g. `FDPS_2` → `FDPS`).

    Returns a new AnnData (sentinel rows dropped). adamson16's
    `(non-targeting)` → `control` rewrite and any other dataset-specific
    raw-column massaging are the responsibility of the caller and run
    BEFORE this function.
    """
    raw = adata.obs[raw_col].astype(str)
    cleaned = raw.apply(clean_perturbation_label)
    drop_mask = cleaned.isna()
    n_dropped = int(drop_mask.sum())

    if audit_path is not None:
        from pathlib import Path as _P
        audit_path = _P(audit_path)
        sentinel_counts = raw[drop_mask].value_counts()
        kept = pd.DataFrame({"raw": raw[~drop_mask].values,
                             "cleaned": cleaned[~drop_mask].values})
        renames = kept[kept["raw"] != kept["cleaned"]]
        rename_counts = renames.groupby(["raw", "cleaned"]).size()
        rows = [
            ("sentinel", str(label), int(count), "dropped: NaN/empty/sentinel token")
            for label, count in sentinel_counts.items()
        ] + [
            ("cleaned", str(raw_lbl), int(count), f"renamed to {new_lbl}")
            for (raw_lbl, new_lbl), count in rename_counts.items()
        ]
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows, columns=["category", "label", "count", "reason"]).to_csv(
            audit_path, index=False,
        )

    if n_dropped:
        log.info("apply_label_cleanup: dropped %d cells with sentinel/empty labels",
                 n_dropped)

    adata = adata[~drop_mask, :].copy()
    adata.obs[target_col] = cleaned[~drop_mask].astype("category")
    return adata


def clean_perturbation_label(raw) -> Optional[str]:
    """Clean one raw perturbation label; return None to drop the cell.

    Universal cleanup applied across every dataset:

      1. NaN/empty/sentinel ('nan', '*', '?', 'none', 'null') -> return None.
      2. Recognized control labels -> return 'control'.
      3. Optional '@dose' suffix is preserved verbatim.
      4. For each gene token (combos split on '+', complexes on ';'):
         strip a trailing '_<digits>' guide suffix (FDPS_2 -> FDPS).
         An empty token, or a sentinel sub-token ('GENE+*', 'nan+GENE'),
         after stripping -> return None.
      5. Reassemble: combo parts sorted alphabetically and joined by '+'.
         Complex parts join with ';'. Dose suffix re-attached.

    Dataset-specific raw-column normalization (e.g., legacy '_'-as-combo
    rewrite for norman19/sunshine23/wessels23, '(non-targeting)' -> 'control'
    for adamson16) is the responsibility of each get_data.py and runs
    BEFORE this function.
    """
    s = "" if raw is None else str(raw).strip()
    if s.lower() in _SENTINEL_TOKENS:
        return None
    if s.lower() in CONTROL_LABELS:
        return "control"

    dose = ""
    if "@" in s:
        s, dose_value = s.split("@", 1)
        dose = "@" + dose_value

    combo_parts: List[str] = []
    for combo_token in s.split("+"):
        complex_parts: List[str] = []
        for tok in combo_token.split(";"):
            tok = _GUIDE_SUFFIX_RE.sub("", tok.strip())
            if not tok or tok.lower() in _SENTINEL_TOKENS:
                # Empty after stripping, or a sentinel buried in a combo/complex
                # token (e.g. 'GENE+*', 'nan+GENE', 'GENE;nan') -> drop the cell.
                return None
            complex_parts.append(tok)
        combo_parts.append(";".join(complex_parts))

    combo_parts = sorted(combo_parts)
    return "+".join(combo_parts) + dose


# ===================================================================
# Sequencing depth and normalisation
# ===================================================================

# Several sources ship a matrix holding only PART of each cell's transcriptome: the
# producer selected a gene panel before publishing. This is NOT visible in the gene
# count -- replogle22 carries 8,563 genes and retains 100.0% of each cell's counts,
# while mcfaline23 carries 15,009 and retains 72.3%. Classifying by gene count got
# both of those backwards.
#
# It matters because every statistic with "total counts" in the denominator -- the
# mitochondrial fraction, CP10K -- is wrong on a truncated matrix. But every such
# source also ships the TRUE per-cell depth, so the statistic is computable rather
# than merely skippable. `SOURCE_DEPTH_COLUMN` in each get_data.py names that column;
# `count_retention` measures the declaration against the matrix so a wrong one cannot
# pass silently.
DEPTH_COLUMN_CANDIDATES = ("ncounts", "nCount_RNA", "n.umi", "UMI_count")

# Below this, a matrix declaring no depth column is not the whole transcriptome and
# the declaration is wrong. 0.90 leaves room for the producer's own gene filtering
# (measured: wessels23 100.3%, replogle22 100.0%) while catching mcfaline23 at 72.3%.
COUNT_RETENTION_COMPLETE_MIN = 0.90


def _row_sums(X) -> np.ndarray:
    return np.asarray(X.sum(axis=1)).ravel().astype(np.float64)


def _counts_matrix(adata: ad.AnnData, counts_layer: str = "counts"):
    """The raw-count matrix: the counts layer if present, else X. Never un-logged.

    A layer named `counts` holds counts by construction. Sources that ship log1p X
    (mcfaline23, jiang24) ship their own integral counts layer alongside it, so this
    always returns counts without any expm1 -- the operation that once overflowed
    integer counts to inf and silently deleted 91% of jiang24.
    """
    if counts_layer in adata.layers:
        return adata.layers[counts_layer], counts_layer
    return adata.X, "X"


def _assert_integral(X, *, where: str) -> None:
    """Refuse to treat a non-integral matrix as counts."""
    sample = X[:: max(1, X.shape[0] // 200)]
    vals = sample.data if hasattr(sample, "data") else np.asarray(sample).ravel()
    if vals.size and not np.allclose(vals, np.round(vals)):
        raise ValueError(
            f"{where}: the matrix is not integral, so its row sums are not counts. "
            f"Counts must be read from layers['counts'] or from a raw X -- never from "
            f"a normalised or log-transformed matrix.")


def resolve_depth(adata: ad.AnnData, depth_column: Optional[str]) -> Tuple[Optional[np.ndarray], str]:
    """Per-cell true total UMI count from the declared obs column.

    Returns (values, provenance). `depth_column=None` means the source ships none and
    the matrix is taken to hold the whole cell.
    """
    if depth_column is None:
        return None, "not declared"
    if depth_column not in adata.obs.columns:
        raise KeyError(
            f"resolve_depth: SOURCE_DEPTH_COLUMN='{depth_column}' is not in obs. "
            f"Available depth-like columns: "
            f"{[c for c in DEPTH_COLUMN_CANDIDATES if c in adata.obs.columns]}")
    v = pd.to_numeric(adata.obs[depth_column], errors="coerce").to_numpy(dtype=np.float64)
    n_bad = int((~np.isfinite(v)).sum() + (v <= 0).sum())
    if n_bad:
        raise ValueError(
            f"resolve_depth: obs['{depth_column}'] has {n_bad} non-positive or "
            f"non-finite entries; it cannot be a sequencing depth.")
    return v, f"obs['{depth_column}']"


def count_retention(
    adata: ad.AnnData,
    *,
    depth_column: Optional[str] = None,
    counts_layer: str = "counts",
    dataset_name: str = "",
) -> Optional[float]:
    """Median share of each cell's transcriptome that the matrix actually holds.

    Records the result in `uns['count_retention']` and `uns['depth_column']`, and
    RAISES when the declaration contradicts the data. Two contradictions are caught:

    * no depth column declared (i.e. "this matrix is the whole cell") while the
      matrix holds less than `COUNT_RETENTION_COMPLETE_MIN` of a depth column that
      the source does in fact ship;
    * a declared column the matrix exceeds, which means it is not this matrix's
      total and the declaration points at the wrong thing.

    Returns None when nothing is available to compare against, having logged that
    the declaration could not be verified.
    """
    X, where = _counts_matrix(adata, counts_layer)
    _assert_integral(X, where=f"count_retention[{dataset_name}] (layer={where})")
    rows = _row_sums(X)

    declared, prov = resolve_depth(adata, depth_column)
    check_col = depth_column
    if declared is None:
        # Auto-detect a column to check against. A candidate NAME is not proof: the
        # replogle20 source ships `UMI_count` as the constant 100 for every cell, which
        # is not a depth at all and produced a nonsense 12384% before this filter.
        #
        # Only the HIGH side disqualifies a candidate. A low ratio is exactly the
        # failure being hunted (mcfaline23 declared complete at 67.8%), so it must
        # reach the raise below rather than be filtered out here.
        for cand in DEPTH_COLUMN_CANDIDATES:
            if cand not in adata.obs.columns:
                continue
            try:
                v, p = resolve_depth(adata, cand)
            except ValueError:
                continue                       # non-positive / non-finite entries
            if np.ptp(v) == 0:
                log.info("count_retention[%s]: obs['%s'] is constant, so it is not a "
                         "per-cell depth; skipped", dataset_name, cand)
                continue
            if float(np.median(rows / v)) > 1.05:
                log.info("count_retention[%s]: the matrix exceeds obs['%s'], so that "
                         "column is not a whole-cell total; skipped", dataset_name, cand)
                continue
            check_col, declared, prov = cand, v, p
            break
        if declared is None:
            log.warning(
                "count_retention[%s]: no depth column declared and none of %s is usable "
                "as one -- the assumption that this matrix holds whole cells is "
                "UNVERIFIED.", dataset_name, list(DEPTH_COLUMN_CANDIDATES))
            adata.uns["count_retention"] = float("nan")
            adata.uns["depth_column"] = "none"
            return None

    ratio = float(np.median(rows / declared))
    adata.uns["count_retention"] = ratio
    adata.uns["depth_column"] = str(depth_column) if depth_column else "none"
    log.info("count_retention[%s]: matrix (%s) holds %.1f%% of %s per cell",
             dataset_name, where, 100 * ratio, prov)

    if depth_column is None and ratio < COUNT_RETENTION_COMPLETE_MIN:
        raise ValueError(
            f"count_retention[{dataset_name}]: no SOURCE_DEPTH_COLUMN is declared, so the "
            f"matrix is assumed to hold whole cells -- but it holds only {100 * ratio:.1f}% "
            f"of {prov}. Every total-counts denominator (mitochondrial fraction, CP10K) "
            f"would be inflated by {1 / max(ratio, 1e-9):.2f}x. Declare "
            f"SOURCE_DEPTH_COLUMN='{check_col}'.")
    if depth_column is not None and ratio > 1.05:
        raise ValueError(
            f"count_retention[{dataset_name}]: the matrix holds {100 * ratio:.1f}% of "
            f"{prov} -- more than all of it, so that column is not this matrix's total. "
            f"Check the SOURCE_DEPTH_COLUMN declaration.")
    return ratio


def normalize_from_counts(
    adata: ad.AnnData,
    *,
    target_sum: float = 1e4,
    depth_column: Optional[str] = None,
    counts_layer: str = "counts",
) -> None:
    """Write CP10K into `X`, computed from `layers[counts_layer]`.

    Replaces `sc.pp.normalize_total`, which can only divide by the row sum of the
    matrix in front of it. Where the matrix is a gene panel that row sum is not the
    cell's depth, so dividing by it rescales every cell to a total it does not have
    and puts the dataset on a different scale from the full-transcriptome ones.

    With `depth_column` declared the divisor is the producer's true UMI total, so a
    cell of which only 46% is present (jiang24) sums to ~0.46 * target_sum -- which
    is the honest number and is directly comparable to a complete dataset whose
    panel subset lands at a similar fraction.

    Called on the counts layer rather than on X, so a source that ships its own
    log-normalised X has that X discarded and rebuilt on our scale.
    """
    if counts_layer not in adata.layers:
        raise ValueError(
            f"normalize_from_counts: layer '{counts_layer}' is missing. Stash raw "
            f"counts before normalising.")
    counts = adata.layers[counts_layer]
    _assert_integral(counts, where=f"normalize_from_counts (layer={counts_layer})")

    declared, prov = resolve_depth(adata, depth_column)
    if declared is None:
        declared = _row_sums(counts)
        prov = f"row sums of layers['{counts_layer}']"
        if (declared <= 0).any():
            raise ValueError(
                f"normalize_from_counts: {int((declared <= 0).sum())} cells have zero "
                f"total counts; per-cell QC should have removed them.")

    scale = (target_sum / declared).astype(np.float64)
    adata.X = (diags(scale) @ counts.tocsr()).astype(np.float32) if issparse(counts) \
        else (np.asarray(counts, dtype=np.float64) * scale[:, None]).astype(np.float32)
    log.info("normalize_from_counts: X = counts / %s * %g (median row sum now %.1f)",
             prov, target_sum, float(np.median(_row_sums(adata.X))))


# ===================================================================
# Per-cell quality control
# ===================================================================

# Mitochondrial genes are identified by symbol prefix. Every dataset in this repo
# carries 12-13 `MT-` genes in its raw gene universe, so this works uniformly.
# Above this the mitochondrial gate is presumed broken rather than strict. The
# worst legitimate case measured is wessels23 monocytes at 2.2% under the MAD rule
# (25.1% under the binary rule this replaced, which was itself judged a bug).
MITO_MAX_REMOVED_FRACTION = 0.50

_MITO_PREFIXES = ("MT-", "MT_")

# obs columns that datasets ship claiming to be a mitochondrial fraction. These are
# NOT unit-consistent across our sources — scPerturb-derived files store a 0-100
# PERCENTAGE (adamson16 median 5.2, sunshine23 max 91.1) while replogle22 stores a
# 0-1 FRACTION (max 0.20). We therefore never read them as-is; see `mito_fraction`.
_MITO_OBS_CANDIDATES = ("percent_mito", "percent.mito", "pct_counts_mt")


def _shipped_mito_fraction(adata: ad.AnnData) -> Tuple[Optional[np.ndarray], str]:
    """Per-cell mito fraction from a shipped obs column -- CROSS-CHECK ONLY.

    Never feeds the gate. The gate is always computed from MT genes over the cell's
    true depth (`mito_fraction`), which means the same thing in every dataset; this
    is only used to corroborate that number against the producer's.

    The shipped columns are NOT unit-consistent (replogle22 ships a 0-1 fraction,
    everyone else a 0-100 percentage), so units are inferred. That inference is safe
    HERE and would not be safe in a gate: reading a percentage as a fraction produces
    a ~100x disagreement, which `mito_fraction` raises on rather than acts on.

    Cells with a missing value yield NaN, not 0.0 -- a missing measurement must not
    read as a clean cell. Callers compare medians with `np.nanmedian`.
    """
    found = [c for c in _MITO_OBS_CANDIDATES if c in adata.obs.columns]
    if not found:
        return None, "unavailable"
    if len(found) > 1:
        log.warning("mito cross-check: source ships %d candidate columns %s; using '%s'",
                    len(found), found, found[0])
    col = found[0]
    v = pd.to_numeric(adata.obs[col], errors="coerce").to_numpy(dtype=np.float64)
    if not np.isfinite(v).any():
        return None, "unavailable"
    n_missing = int((~np.isfinite(v)).sum())
    if n_missing:
        log.warning("mito cross-check: obs['%s'] has %d missing values; excluded from "
                    "the comparison", col, n_missing)
        v = np.where(np.isfinite(v), v, np.nan)
    vmax = float(np.nanmax(v))
    if vmax <= 1.0:
        return v, f"obs['{col}'] as fraction"
    return v / 100.0, f"obs['{col}'] as percent/100"


def mito_fraction(
    adata: ad.AnnData,
    *,
    counts_layer: str = "counts",
    depth_column: Optional[str] = None,
    dataset_name: str = "",
) -> Tuple[Optional[np.ndarray], str]:
    """Per-cell mitochondrial fraction in [0, 1], plus a provenance string.

    Always computed, never read from a shipped column::

        MT counts from the matrix  /  the cell's TRUE total counts

    The numerator comes from genes prefixed `MT-`; the denominator from
    `obs[depth_column]` when the source declares one, else the matrix row sum.

    Two things this deliberately does not do.

    It does not un-log anything. The matrix is `layers['counts']` when that exists
    and `X` otherwise, and both are asserted integral. An earlier version applied
    `expm1` whenever a dataset declared log-normalised X -- but mcfaline23 and
    jiang24 ship log1p X *and* their own integral counts layer, so `expm1` hit
    integer counts, overflowed to inf, and turned the gate into "is the highest-count
    gene mitochondrial". That deleted 91.4% of jiang24 and 51.8% of mcfaline23, and
    mcfaline23 then passed `verify` 113/0 on the wreckage.

    It does not fall back to the producer's column when the matrix is a gene panel.
    A panel truncates the DENOMINATOR, not the numerator: MT genes survive gene
    selection while abundant non-MT genes do not, so the ratio inflates by
    1/retention -- jiang24 reads 0.1446 against a true 0.0561. The fix is the right
    denominator, which every truncated source ships. Measured on jiang24:
    `MT / ncounts` = 0.0526 against the producer's 0.0566, agreeing to ~7%.

    Falls back to the shipped column only when the source carries no MT- gene at all
    (currently unreachable -- every source here has 12-14).
    """
    var_upper = np.asarray([str(g).upper() for g in adata.var_names])
    mt_mask = np.zeros(adata.n_vars, dtype=bool)
    for pref in _MITO_PREFIXES:
        mt_mask |= np.char.startswith(var_upper.astype(str), pref)

    if not mt_mask.any():
        frac, prov = _shipped_mito_fraction(adata)
        if frac is None:
            log.warning("mito_fraction[%s]: no MT- genes and no recognised obs column "
                        "-- the mitochondrial gate cannot be applied", dataset_name)
            return None, "unavailable"
        log.warning("mito_fraction[%s]: no MT- genes; falling back to %s",
                    dataset_name, prov)
        return np.nan_to_num(frac, nan=0.0), prov

    X, where = _counts_matrix(adata, counts_layer)
    _assert_integral(X, where=f"mito_fraction[{dataset_name}] (layer={where})")

    mito = np.asarray(X[:, mt_mask].sum(axis=1)).ravel().astype(np.float64)
    total, denom_prov = resolve_depth(adata, depth_column)
    if total is None:
        total = _row_sums(X)
        denom_prov = f"row sums of {where}"

    with np.errstate(invalid="ignore", divide="ignore"):
        frac = np.where(total > 0, mito / total, 0.0)
    n_bad = int((~np.isfinite(frac)).sum())
    if n_bad:
        raise ValueError(
            f"mito_fraction[{dataset_name}]: {n_bad}/{frac.size} cells have a non-finite "
            f"fraction (numerator={where}, denominator={denom_prov}) -- refusing to gate "
            f"on it.")
    if (frac > 1.0).any():
        raise ValueError(
            f"mito_fraction[{dataset_name}]: {int((frac > 1.0).sum())} cells have a "
            f"fraction above 1, so {denom_prov} is smaller than the MT counts it should "
            f"contain. The denominator is not this matrix's total.")

    med = float(np.median(frac))
    src = f"{int(mt_mask.sum())} MT- genes on {where} / {denom_prov}"

    # Corroborate against the producer's own column where one exists. This never
    # feeds the gate -- it only catches a wrong denominator, which is precisely the
    # failure that made jiang24 lose 26.8% of its cells after the expm1 fix.
    shipped, shipped_prov = _shipped_mito_fraction(adata)
    if shipped is not None:
        med_shipped = float(np.nanmedian(shipped))
        if med_shipped > 0 and med > 0:
            ratio = med / med_shipped
            if ratio > 1.5 or ratio < 0.67:
                raise ValueError(
                    f"mito_fraction[{dataset_name}]: computed median {med:.4f} disagrees "
                    f"with the shipped {shipped_prov} median {med_shipped:.4f} by "
                    f"{ratio:.2f}x. Either the denominator ({denom_prov}) is not the "
                    f"cell's true total, or SOURCE_DEPTH_COLUMN is wrong.")
            log.info("mito_fraction[%s]: computed median %.4f agrees with shipped %s "
                     "(median %.4f, ratio %.2f)",
                     dataset_name, med, shipped_prov, med_shipped, ratio)
        else:
            log.info("mito_fraction[%s]: cross-check skipped (computed median %.4f, "
                     "shipped median %.4f -- one of them is zero)",
                     dataset_name, med, med_shipped)

    log.info("mito_fraction[%s]: %s; median %.4f", dataset_name, src, med)
    return frac, src


def mito_threshold(frac: np.ndarray, *, floor: float = 0.10, cap: float = 0.20,
                   n_mads: float = 3.0) -> float:
    """Bounded MAD-based mitochondrial cut-off: `median + n_mads * MAD`, clipped
    to [floor, cap].

    Replaces an earlier binary switch ("0.10 normally, 0.15 if fewer than half the
    cells sit under 0.10"). That rule flipped on a data-dependent boundary — the
    same fragility as the mean-based downsample cap this pipeline removed — and it
    landed on the wrong side for wessels23, whose monocytes have a genuinely high
    mitochondrial baseline (median 8.6%): a fixed 0.10 discarded 25.1% of the
    dataset, while the MAD rule adapts to 0.144 and removes the 2.2% that are
    actually outliers. On the datasets whose baseline is low the two agree
    (adamson16 and sunshine23 land on the 0.10 floor either way).

    The floor stops a very clean dataset from cutting real cells; the cap stops a
    pathological one from disabling the gate entirely.
    """
    frac = np.asarray(frac, dtype=np.float64)
    frac = frac[np.isfinite(frac)]
    if frac.size == 0:
        return floor
    med = float(np.median(frac))
    mad = float(np.median(np.abs(frac - med))) * 1.4826   # -> sigma-equivalent
    return float(np.clip(med + n_mads * mad, floor, cap))


def condition_arity(label: str) -> int:
    """Number of targeting guides a condition label implies (control -> 0)."""
    if _is_control_label(label):
        return 0
    base = str(label).split("@")[0]
    return len([t for t in base.replace(";", "+").split("+") if t.strip()])


# --- Guide assignment / MOI -------------------------------------------------
#
# A per-cell guide COUNT is not a usable MOI signal on its own: the shipped count
# columns include non-targeting guides, so "n_guides > condition arity" flags a
# control legitimately carrying 3 NT guides. Two columns are outright wrong —
# adamson16's `nperts` is the constant 2 for every cell, including its 7,295
# arity-0 controls. We therefore gate on guide IDENTITY (which genes the guides
# target), declared explicitly per dataset, and never on a bare count.
#
# Measured on the current sources:
#   frangieh21  `guide_id`   ';'-separated `GENE_n` / `NO_SITE_n` -> 38.7% of
#               assigned cells carry MORE targeting genes than the label declares,
#               incl. 3,411 'control' cells carrying real targeting guides.
#   wessels23   `Guide.Class` {NT, Single, Dual} -> maps EXACTLY onto arity
#               (0/1/2), zero violations: a dual-guide array leaves no room for
#               contamination.
#   others      no guide-identity column -> gate skipped and recorded.

_NON_TARGETING_GUIDE_TOKENS = {
    "NO_SITE", "ONE_NON-GENE_SITE", "NON-GENE_SITE", "NO-TARGET",
    "NON-TARGETING", "NONTARGETING", "NT", "SAFE-HARBOR", "SCRAMBLE",
}


def guide_target_sets(
    adata: ad.AnnData, guide_col: str, *, sep: str = ";",
    token_to_gene=None,
) -> "pd.Series":
    """Per-cell SET of gene symbols the cell's guides target.

    Splits `obs[guide_col]` on `sep`, maps each token to a gene symbol, and drops
    non-targeting tokens (`NO_SITE`, `ONE_NON-GENE_SITE`, `non-targeting`, …).
    Cells with a missing/`nan` guide field yield `None` — "unassigned", which is
    distinct from "assigned, zero targeting guides".

    **`token_to_gene` is dataset-specific and must be verified, not assumed.**
    The default strips a trailing `_<n>` guide index (frangieh21: `HLA-B_2`), but
    guide-ID formats differ per source and a wrong tokenizer silently turns
    non-targeting guides into "genes":
      * sunshine23 `STAT2_+_56753865.23-P1P2_CR1-cs1` -> `t.split("_")[0]`
      * norman19   `SET_KLF1;SET_KLF1` encodes BOTH genes in one token — the
        default parser would invent a gene called `SET_KLF1`, so that dataset
        declares no guide column at all.
    Validate a candidate tokenizer by checking that every condition's label genes
    appear among its parsed guide genes before enabling the gate.
    """
    tok2gene = token_to_gene or (lambda t: _GUIDE_SUFFIX_RE.sub("", t))

    def _parse(raw) -> Optional[set]:
        s = str(raw).strip()
        if not s or s.lower() in ("nan", "none", "null"):
            return None
        toks = {tok2gene(t.strip()) for t in s.split(sep) if t.strip()}
        return {t for t in toks if t.upper() not in _NON_TARGETING_GUIDE_TOKENS}
    # .astype(str) first: mapping a CATEGORICAL to sets raises
    # "unhashable type: 'set'" when pandas tries to rebuild the categories.
    return adata.obs[guide_col].astype(str).map(_parse)


def guide_class_target_counts(
    adata: ad.AnnData, class_col: str, mapping: Dict[str, int],
) -> "pd.Series":
    """Per-cell targeting-guide COUNT from a categorical guide-class column.

    For designs that report a construct class rather than guide IDs, e.g.
    wessels23's ``Guide.Class`` with ``{"NT": 0, "Single": 1, "Dual": 2}``.
    Unmapped classes yield NaN (treated as unassigned).
    """
    return adata.obs[class_col].astype(str).map(mapping)


def guide_violation_mask(
    adata: ad.AnnData, *, n_targets: "pd.Series", pert_col: str = "condition",
) -> Tuple[np.ndarray, Dict[str, int]]:
    """Cells whose guides target MORE genes than their condition label declares.

    `n_targets` is a per-cell count (or NaN/None where the cell has no guide call)
    — build it with `guide_target_sets(...).map(len)` or `guide_class_target_counts`.
    The rule is arity-aware, so a 2-gene combo cell with 2 targeting guides passes.
    Unassigned cells are never flagged (no evidence against them); they are counted
    separately so the caller can report them.
    """
    arity = adata.obs[pert_col].astype(str).map(condition_arity)
    if len(n_targets) != adata.n_obs:
        raise ValueError(
            f"guide_violation_mask: n_targets has {len(n_targets)} entries but the "
            f"AnnData has {adata.n_obs} cells — they are aligned POSITIONALLY, so a "
            f"length mismatch would silently mislabel cells.")
    n = pd.to_numeric(pd.Series(n_targets).reset_index(drop=True), errors="coerce")
    a = arity.reset_index(drop=True)
    assigned = n.notna()
    mask = (assigned & (n > a)).to_numpy(dtype=bool)
    stats = {
        "n_cells": int(len(n)),
        "n_unassigned": int((~assigned).sum()),
        "n_violations": int(mask.sum()),
    }
    log.info("guide gate: %d/%d cells target more genes than declared (%.1f%%); "
             "%d unassigned (never flagged)", stats["n_violations"], stats["n_cells"],
             100.0 * stats["n_violations"] / max(stats["n_cells"], 1),
             stats["n_unassigned"])
    return mask, stats


def apply_cell_qc(
    adata: ad.AnnData,
    *,
    dataset_name: str = "",
    min_genes: int = 200,
    apply_mito: bool = True,
    mito_floor: float = 0.10,
    mito_cap: float = 0.20,
    mito_max: Optional[float] = None,
    depth_column: Optional[str] = None,
    guide_n_targets: Optional["pd.Series"] = None,
    pert_col: str = "condition",
    audit_path=None,
) -> ad.AnnData:
    """Per-cell QC gate: min_genes, mitochondrial fraction, guide-identity MOI.

    Runs BEFORE gene filtering / normalization / HVG so that (a) the mitochondrial
    fraction is computed on the full gene universe (the HVG panel may not retain
    MT- genes) and (b) every later statistic is derived from QC-passing cells only.

    Every gate is recorded in an audit CSV — **including gates that were skipped
    and why** — so "we did no MOI filtering here" is a visible fact rather than a
    silent absence. Returns a new AnnData; the input is not modified.
    """
    n0 = adata.n_obs
    rows: List[dict] = []
    keep = np.ones(n0, dtype=bool)

    # --- 1. minimum detected genes -----------------------------------------
    # getnnz avoids materialising a full boolean copy of X, which on the
    # largest datasets is a multi-GB temporary.
    X0 = adata.X
    counts_nnz = (X0.getnnz(axis=1) if hasattr(X0, "getnnz")
                  else np.asarray((X0 > 0).sum(axis=1)).ravel())
    m_genes = counts_nnz >= min_genes
    rows.append(dict(gate="min_genes", applied=True, detail=f"n_genes >= {min_genes}",
                     n_removed=int((~m_genes).sum())))
    keep &= m_genes

    # --- 2. mitochondrial fraction -----------------------------------------
    if apply_mito:
        frac, prov = mito_fraction(adata, depth_column=depth_column,
                                   dataset_name=dataset_name)
        if frac is None:
            rows.append(dict(gate="mito", applied=False, detail=prov, n_removed=0))
        else:
            # A FIXED threshold is required when this runs per-chunk (xatlas
            # streams batches): an adaptive cut-off computed inside each chunk
            # would differ chunk to chunk and make the gate non-uniform.
            thr = (mito_max if mito_max is not None else
                   mito_threshold(frac[keep] if keep.any() else frac,
                                  floor=mito_floor, cap=mito_cap))
            m_mito = frac < thr
            rows.append(dict(gate="mito", applied=True,
                             detail=f"{prov}; "
                                    f"{'fixed' if mito_max is not None else 'MAD'} "
                                    f"cut-off {thr:.4f} "
                                    f"(median {float(np.median(frac)):.4f})",
                             n_removed=int((keep & ~m_mito).sum())))
            # A mitochondrial gate is a quality filter, not a subsetting step. Any
            # dataset where it removes most of the cells is a bug in the fraction or
            # the threshold, not biology — jiang24 (91.4%) and mcfaline23 (51.8%) both
            # ran to completion under a silently-broken expm1 before this guard existed,
            # and mcfaline23 even passed `verify`. Fail loudly instead.
            mito_removed = int((keep & ~m_mito).sum())
            mito_frac_removed = mito_removed / max(int(keep.sum()), 1)
            if mito_frac_removed > MITO_MAX_REMOVED_FRACTION:
                raise ValueError(
                    f"apply_cell_qc[{dataset_name}]: the mitochondrial gate would remove "
                    f"{mito_removed} of {int(keep.sum())} cells ({100 * mito_frac_removed:.1f}%) "
                    f"at cut-off {thr:.4f} (median fraction {float(np.median(frac)):.4f}). "
                    f"That is a broken fraction or threshold, not biology — check "
                    f"SOURCE_DEPTH_COLUMN and the counts layer.")
            if mito_frac_removed > 0.20:
                log.warning("apply_cell_qc[%s]: mitochondrial gate removes %.1f%% of cells "
                            "— high; verify this is real", dataset_name, 100 * mito_frac_removed)
            keep &= m_mito
    else:
        rows.append(dict(gate="mito", applied=False, detail="disabled by caller", n_removed=0))

    # --- 3. guide-identity MOI ---------------------------------------------
    if guide_n_targets is None:
        rows.append(dict(gate="guide_moi", applied=False,
                         detail="no guide-identity column declared for this dataset",
                         n_removed=0))
    else:
        viol, stats = guide_violation_mask(adata, n_targets=guide_n_targets,
                                           pert_col=pert_col)
        rows.append(dict(gate="guide_moi", applied=True,
                         detail=f"targeting genes > condition arity; "
                                f"{stats['n_unassigned']} unassigned kept",
                         n_removed=int((keep & viol).sum())))
        keep &= ~viol

    n1 = int(keep.sum())
    log.info("cell QC [%s]: %d -> %d cells (%.1f%% removed)", dataset_name or "?",
             n0, n1, 100.0 * (n0 - n1) / max(n0, 1))
    for r in rows:
        log.info("  %-10s applied=%-5s removed=%-7d %s",
                 r["gate"], r["applied"], r["n_removed"], r["detail"])

    if audit_path is not None:
        df = pd.DataFrame(rows)
        df.insert(0, "dataset", dataset_name)
        df["n_before"] = n0
        df["n_after"] = n1
        pd.DataFrame(df).to_csv(audit_path, index=False)
        log.info("  cell-QC audit -> %s", audit_path)

    # Carry the audit INSIDE the artefact, not only in a sibling CSV. The
    # catastrophic-removal guard above runs at build time only, so a corrupted h5ad
    # already on disk verified clean -- which is exactly how mcfaline23 passed 113/0
    # having lost 52% of its cells.
    out = adata[keep].copy()
    out.uns["cell_qc_audit"] = {
        "n_before": int(n0), "n_after": int(n1),
        **{f"removed_{r['gate']}": int(r["n_removed"]) for r in rows},
        **{f"applied_{r['gate']}": bool(r["applied"]) for r in rows},
    }
    return out


def _dense_row_mean(X, row_mask: np.ndarray) -> np.ndarray:
    """Return the mean of X[row_mask, :] across rows as a dense (n_genes,) float32 array.

    Works for both sparse and dense X. Returns zeros when row_mask is empty.
    """
    if row_mask.sum() == 0:
        return np.zeros(X.shape[1], dtype=np.float32)
    sub = X[row_mask, :]
    if issparse(sub):
        m = np.asarray(sub.mean(axis=0)).ravel()
    else:
        m = np.asarray(sub.mean(axis=0)).ravel()
    return m.astype(np.float32)


# ===================================================================
# Single-cell preprocessing
# ===================================================================

def downsample_per_condition(
    adata: ad.AnnData,
    *,
    pert_col: str = "condition",
    bin_col: Optional[str] = None,
    min_cells_threshold: int = 12,
    control_cap: int = 8192,
    pert_cap: Optional[int] = None,
    seed: int = 42,
) -> ad.AnnData:
    """Drop under-powered conditions and cap the control population.

    * Drop conditions with fewer than `min_cells_threshold` cells (per bin when
      `bin_col` is set) — an under-powered condition yields an unusable pseudobulk.
    * Keep EVERY cell of every surviving perturbation. There is deliberately no
      per-perturbation cap by default.
    * Cap control at `control_cap`; cap perturbations only if `pert_cap` is given
      (xatlas_orion, whose 3.2M cells x 18k perturbations are otherwise intractable).

    **Why there is no mean-based cap any more.** The previous rule capped every
    perturbation at `round(mean cells per condition)`. That mean was computed over
    the surviving conditions *including control*, so the cap moved whenever the
    label partition moved — and because HVG selection ran on the resulting cells,
    the GENE PANEL itself became a function of a random draw. Measured on
    replogle20: collapsing two guide-level labels changed the cap 477 -> 448,
    changed 27% of the retained cells, and swapped 1,010 of 8,192 panel genes.
    Dropping the cap removes that coupling at the source and hands the evaluation
    ground truth every cell it is entitled to.

    Each condition is drawn from its OWN generator, seeded from `seed` and the
    condition label, so adding or removing one condition cannot reshuffle the cells
    kept for any other.

    Returns a new (subset) AnnData; the input is not modified in place.
    """
    def _condition_rng(bin_name: str, label: str) -> np.random.RandomState:
        # crc32 is stable across processes (unlike hash() for str) so the draw is
        # reproducible run-to-run and independent per condition.
        key = f"{seed}|{bin_name}|{label}".encode()
        return np.random.RandomState(zlib.crc32(key) & 0xFFFFFFFF)

    def _select_indices(adata_bin: ad.AnnData, bin_name: str) -> List[int]:
        idx_kept: List[int] = []
        pert_counts = adata_bin.obs[pert_col].value_counts()
        survivors = [p for p in pert_counts.index
                     if pert_counts[p] >= min_cells_threshold]  # control included
        if not survivors:
            log.warning("downsample: no condition has >= %d cells", min_cells_threshold)
            return idx_kept
        n_dropped = int(len(pert_counts) - len(survivors))
        log.info("  bin '%s': %d/%d conditions survive the >=%d-cell floor; "
                 "control cap %s, perturbation cap %s", bin_name, len(survivors),
                 len(pert_counts), min_cells_threshold, control_cap,
                 pert_cap if pert_cap is not None else "none (all cells kept)")
        if n_dropped:
            log.info("  bin '%s': dropped %d under-powered condition(s)", bin_name, n_dropped)
        values = adata_bin.obs[pert_col].values
        for p in survivors:
            cap = control_cap if _is_control_label(p) else pert_cap
            cell_pos = np.where(values == p)[0]
            if cap is not None and len(cell_pos) > cap:
                chosen = _condition_rng(bin_name, str(p)).choice(cell_pos, size=cap,
                                                                 replace=False)
            else:
                chosen = cell_pos
            idx_kept.extend(np.asarray(chosen).tolist())
        return sorted(idx_kept)

    if bin_col is None:
        return adata[_select_indices(adata, "all"), :].copy()

    kept_global: List[int] = []
    obs_pos = np.arange(adata.n_obs)
    for bn in sorted(adata.obs[bin_col].unique()):
        bin_pos = obs_pos[adata.obs[bin_col].values == bn]
        if len(bin_pos) == 0:
            continue
        local_keep = _select_indices(adata[bin_pos, :], str(bn))
        kept_global.extend(bin_pos[local_keep].tolist())
    return adata[sorted(set(kept_global)), :].copy()


def filter_genes_keeping_targets(
    adata: ad.AnnData,
    *,
    min_cells: int = 3,
    pert_col: str = "condition",
) -> ad.AnnData:
    """Like ``sc.pp.filter_genes(adata, min_cells=min_cells)`` but never drops a
    gene that is a perturbation target.

    QC by ``min_cells`` runs before HVG selection, so a real-but-sparsely-measured
    target gene would otherwise be removed before it can be force-included into the
    HVG panel. Exempting target genes here keeps measured targets in the panel.
    (Aliased / never-measured targets still won't appear — they are simply absent
    from ``var_names`` and are logged downstream by
    ``force_include_perturbation_targets_in_hvg``.)

    Returns a new gene-subset AnnData; the input is not modified in place.
    """
    X = adata.X
    if issparse(X):
        n_cells_per_gene = np.asarray(X.getnnz(axis=0)).ravel()
    else:
        n_cells_per_gene = np.asarray((X > 0).sum(axis=0)).ravel()
    keep = n_cells_per_gene >= min_cells

    # Parse perturbation target tokens (dose '@' stripped, combos '+', complexes ';').
    targets: set = set()
    for cond in adata.obs[pert_col].unique():
        if _is_control_label(cond):
            continue
        s = str(cond).split("@")[0]
        for token in s.split("+"):
            for sub in token.split(";"):
                sub = sub.strip()
                if sub:
                    targets.add(sub)

    target_mask = np.asarray(adata.var_names.isin(targets))
    rescued = int((target_mask & ~keep).sum())
    keep = keep | target_mask

    log.info(
        "filter_genes_keeping_targets: keeping %d/%d genes (min_cells=%d); "
        "rescued %d target gene(s) below threshold",
        int(keep.sum()), adata.n_vars, min_cells, rescued,
    )
    return adata[:, keep].copy()


def annotate_perturbation_targets(
    adata: ad.AnnData,
    *,
    pert_col: str = "condition",
) -> None:
    """Set the boolean `adata.var["is_perturbation_target"]` column.

    A gene is a perturbation target if it (or a sub-token of a combo/dose label)
    appears in `adata.obs[pert_col]`. Combo perturbations "GeneA+GeneB" split on
    '+'; complexes "GeneA;GeneB" split on ';'; dose "GeneA@0.25" strips after '@'.

    Unlike the old `force_include_perturbation_targets_in_hvg`, this does NOT touch
    `var["highly_variable"]` — it keeps the *pure* HVG flag intact so a downstream
    consumer can choose the HVG-only panel vs the HVG ∪ targets union via
    `union_panel_mask`. Idempotent. Raises if a non-control label parses to zero
    target tokens (the target-baseline contract).
    """
    targets: set = set()
    unparseable: List[str] = []
    for cond in adata.obs[pert_col].unique():
        if _is_control_label(cond):
            continue
        # Strip dose suffix, then split combos on '+', then split complexes on ';'
        s = str(cond)
        if "@" in s:
            s = s.split("@")[0]
        parsed: set = set()
        for token in s.split("+"):
            for sub in token.split(";"):
                sub = sub.strip()
                if sub:
                    parsed.add(sub)
        if not parsed:
            unparseable.append(str(cond))
        targets.update(parsed)
    if unparseable:
        raise ValueError(
            f"annotate_perturbation_targets: {len(unparseable)} non-control "
            f"perturbation label(s) parse to zero target tokens (first 5: "
            f"{unparseable[:5]}). Target-based baselines require every KO label "
            f"to yield at least one target gene name."
        )

    var_names = adata.var_names
    missing = sorted(targets - set(var_names))
    if missing:
        preview = missing[:10]
        suffix = f" ... and {len(missing) - 10} more" if len(missing) > 10 else ""
        log.info(
            "annotate_perturbation_targets: %d perturbation target gene(s) absent "
            "from var_names — kept in obs[%r] but not in the panel (matches AE 2025 "
            "behavior). First 10: %s%s", len(missing), pert_col, preview, suffix,
        )

    is_target = var_names.isin(targets)
    adata.var["is_perturbation_target"] = np.asarray(is_target, dtype=bool)
    log.info("annotate_perturbation_targets: %d/%d genes are perturbation targets "
             "in the panel", int(is_target.sum()), adata.n_vars)


def select_hvg(
    adata: ad.AnnData,
    *,
    n_top_genes: int = 8192,
    counts_layer: str = "counts",
    batch_key: Optional[str] = None,
) -> None:
    """Annotate `var['highly_variable']` — the ONE place HVG is chosen.

    `flavor="seurat_v3"` on **raw counts**, not the scanpy default `seurat` flavor
    on log-normalised X. Two reasons:

    * The default flavor ranks genes by a *binned* normalised dispersion. A gene's
      score depends on which mean-expression bin it lands in, so a small shift in
      the cell set can move a gene between bins and jump its rank discontinuously.
      `seurat_v3` fits a variance-mean trend on counts instead — no binning.
    * It accepts `batch_key`, selecting within each batch and merging ranks, which
      keeps a gene that is variable only because of one cell type out of the panel.

    MUST be called BEFORE any cell subsampling. HVG previously ran *after* the
    downsampler, which made the gene panel a function of a random draw (see
    `downsample_per_condition`). Running it on the full QC-passing cell set makes
    the panel a deterministic function of the data.
    """
    if counts_layer not in adata.layers:
        raise ValueError(
            f"select_hvg: layer '{counts_layer}' missing — seurat_v3 needs raw "
            f"counts. Store them before normalising (adata.layers['{counts_layer}'])."
        )
    n_top = min(n_top_genes, adata.n_vars)
    if n_top < n_top_genes:
        log.info("select_hvg: only %d genes available; capping n_top_genes %d -> %d",
                 adata.n_vars, n_top_genes, n_top)
    sc.pp.highly_variable_genes(
        adata, n_top_genes=n_top, flavor="seurat_v3", layer=counts_layer,
        batch_key=batch_key, subset=False,
    )
    log.info("select_hvg: %d HVGs (seurat_v3 on '%s'%s) from %d cells x %d genes",
             int(adata.var["highly_variable"].sum()), counts_layer,
             f", batch_key='{batch_key}'" if batch_key else "",
             adata.n_obs, adata.n_vars)


def union_panel_mask(adata: ad.AnnData) -> np.ndarray:
    """Boolean gene mask for the HVG ∪ perturbation-target panel.

    Requires `var["highly_variable"]` (pure HVG) and `var["is_perturbation_target"]`
    (from `annotate_perturbation_targets`).
    """
    assert "highly_variable" in adata.var.columns, \
        "highly_variable missing — call sc.pp.highly_variable_genes first"
    assert "is_perturbation_target" in adata.var.columns, \
        "is_perturbation_target missing — call annotate_perturbation_targets first"
    return (adata.var["highly_variable"].values
            | adata.var["is_perturbation_target"].values)


def force_include_perturbation_targets_in_hvg(
    adata: ad.AnnData,
    *,
    pert_col: str = "condition",
) -> None:
    """DEPRECATED back-compat shim. Prefer `annotate_perturbation_targets` +
    `union_panel_mask`. Sets `is_perturbation_target` and then folds targets into
    `var["highly_variable"]` (the old clobbering behavior) for any caller that
    still subsets on `highly_variable` alone.
    """
    annotate_perturbation_targets(adata, pert_col=pert_col)
    adata.var["highly_variable"] = union_panel_mask(adata)


def write_h5ad_compressed(adata: ad.AnnData, path, *, level: int = 1) -> None:
    """Write an h5ad with gzip compression (default level 1).

    The shared `/cluster/work/boeva` filesystem is CephFS and chronically near-full;
    sparse log1p data compresses ~2-3x with gzip-1, cutting both disk and network I/O
    at roughly neutral wall-time. (`hdf5plugin`/blosc-lz4 would be faster but isn't
    installed.) Use this for every dataset write instead of bare `adata.write_h5ad`.
    """
    adata.write_h5ad(str(path), compression="gzip", compression_opts=int(level))


# ===================================================================
# Pre-DEG checkpoint / resume
# ===================================================================
#
# get_data.py is one monolithic process from download to write, so a wall-time kill,
# an OOM or a node failure during the DEG stage discards EVERYTHING upstream: the
# source read (jiang24 is a 21 GB h5ad), per-cell QC, gene QC, normalize+log1p,
# select_hvg (the most expensive deterministic step), the panel subset, the condition
# floor, the splits and the pseudobulk. On jiang24 that is 30-60 minutes per retry.
#
# The seam is placed after compute_pseudobulk because it already exists and is already
# enforced: compute_degs_pdex REFUSES to run unless uns["pseudobulk"] is populated.
# Everything before it is deterministic and cheap to re-validate; everything after is
# the expensive, tunable part (pdex mode/workers/threads, and the still-reversible
# `score="signed_neglog10p"` weight basis).
#
# The staleness stamp is the part that must not be skipped. A checkpoint that silently
# survives a code change would produce an h5ad that does not correspond to the current
# source — precisely the "output is a function of something invisible" failure this
# rebuild exists to eliminate. The stamp therefore hashes the CONTENT of _utils.py and
# of the calling get_data.py, so ANY edit to either invalidates it. That is deliberately
# conservative: a needless rebuild is cheap, a silently-wrong h5ad is not.

_RECIPE_VERSION = "recipe-v1"


def recipe_stamp(script_path, **params) -> str:
    """Content hash of everything that determines the pre-DEG result."""
    h = hashlib.sha256()
    h.update(f"{_RECIPE_VERSION}\n".encode())
    for key in sorted(params):
        h.update(f"{key}={params[key]!r}\n".encode())
    for path in (Path(__file__).resolve(), Path(script_path).resolve()):
        h.update(path.read_bytes())
    return h.hexdigest()[:16]


def predeg_path(output_dir, dataset_name: str) -> Path:
    return Path(output_dir) / f"{dataset_name}_predeg.h5ad"


def load_predeg(path, stamp: str) -> Optional[ad.AnnData]:
    """Return the checkpointed AnnData, or None if absent/stale/unusable."""
    path = Path(path)
    if not path.exists():
        return None
    try:
        adata = ad.read_h5ad(str(path))
    except Exception as exc:                       # truncated write, bad HDF5, ...
        log.warning("pre-DEG checkpoint %s unreadable (%s) — rebuilding", path, exc)
        return None
    found = str(adata.uns.get("_recipe_stamp", ""))
    if found != stamp:
        log.warning("pre-DEG checkpoint %s is STALE (stamp %s, expected %s) — "
                    "rebuilding from scratch", path, found or "<missing>", stamp)
        return None
    if "pseudobulk" not in adata.uns:
        log.warning("pre-DEG checkpoint %s has no pseudobulk — rebuilding", path)
        return None
    log.info("Resuming from pre-DEG checkpoint %s (%d cells x %d genes)",
             path, adata.n_obs, adata.n_vars)
    return adata


def save_predeg(adata: ad.AnnData, path, stamp: str) -> None:
    """Write the checkpoint atomically, UNCOMPRESSED (transient; speed over size)."""
    path = Path(path)
    adata.uns["_recipe_stamp"] = stamp
    tmp = path.with_suffix(".h5ad.tmp")
    adata.write_h5ad(str(tmp))
    tmp.replace(path)
    log.info("Wrote pre-DEG checkpoint %s (%.1f GB) stamp=%s",
             path, path.stat().st_size / 1e9, stamp)


def clear_predeg(path) -> None:
    """Drop the checkpoint once the final h5ad is written."""
    path = Path(path)
    if path.exists():
        path.unlink()
        log.info("Removed pre-DEG checkpoint %s", path)


def assign_tech_dup_split(
    adata: ad.AnnData,
    *,
    pert_col: str = "condition",
    bin_col: Optional[str] = None,
    seed: int = 42,
) -> None:
    """Assign 'first_half' / 'second_half' to adata.obs['tech_dup_split'].

    Split is 50/50 per group: by (pert_col,) for single-bin, by (pert_col, bin_col)
    for multi-bin. Reproducible via seed. Cells with fewer than 2 group members
    get 'first_half'.
    """
    rng = np.random.RandomState(seed)
    adata.obs["tech_dup_split"] = "first_half"
    obs = adata.obs.copy()

    if bin_col is None:
        groupby_cols = [pert_col]
    else:
        groupby_cols = [pert_col, bin_col]

    for group_key, group_df in obs.groupby(groupby_cols, observed=True):
        cell_indices = group_df.index.values
        if len(cell_indices) < 2:
            continue
        perm = rng.permutation(len(cell_indices))
        half = len(cell_indices) // 2
        adata.obs.loc[cell_indices[perm[:half]], "tech_dup_split"] = "first_half"
        adata.obs.loc[cell_indices[perm[half:]], "tech_dup_split"] = "second_half"

    counts = adata.obs["tech_dup_split"].value_counts().to_dict()
    log.info("assign_tech_dup_split: %s", counts)


def restrict_to_full_coverage_perturbations(
    adata: ad.AnnData,
    *,
    bin_col: str,
    pert_col: str = "condition",
    min_cells: int = MIN_CELLS_DEGS_DEFAULT,
) -> ad.AnnData:
    """Keep controls + perturbations present with >= ``min_cells`` in BOTH
    tech-dup halves of EVERY bin (the cross-bin intersection).

    This is the SAME predicate ``compute_pseudobulk`` uses to build ``ko_names``.
    Calling it before ``assign_split_folds_*`` makes the obs fold columns and the
    pseudobulk / DEG arrays cover the IDENTICAL perturbation set. Without it the
    fold columns also enumerate perturbations seen in only some bins — which have
    no cross-bin pseudobulk ground truth, are silently NaN-skipped at eval, and
    (because they enter the fold shuffle) shift every CV fold relative to a
    pipeline that splits only the shared perturbations.

    Multi-bin only (``bin_col`` required). Requires ``assign_tech_dup_split`` to
    have run. Returns a subset AnnData; the input is not modified in place.
    """
    if "tech_dup_split" not in adata.obs.columns:
        raise ValueError("restrict_to_full_coverage_perturbations requires "
                         "tech_dup_split; call assign_tech_dup_split first")
    conds = adata.obs[pert_col].values
    halves = adata.obs["tech_dup_split"].values
    bins = adata.obs[bin_col].values
    bin_names = sorted(adata.obs[bin_col].unique())
    is_first = (halves == "first_half")
    is_second = (halves == "second_half")

    non_control = _non_control_conditions(adata, pert_col)
    valid_kos: List[str] = []
    for cond in non_control:
        cmask = (conds == cond)
        ok = True
        for bn in bin_names:
            inb = cmask & (bins == bn)
            if (int((inb & is_first).sum()) < min_cells
                    or int((inb & is_second).sum()) < min_cells):
                ok = False
                break
        if ok:
            valid_kos.append(cond)

    keep_labels = set(valid_kos) | set(_control_conditions(adata, pert_col))
    keep_mask = np.array([c in keep_labels for c in conds])
    log.info("restrict_to_full_coverage_perturbations: kept %d/%d perturbations "
             "(>=%d cells in both halves of all %d bins); dropped %d "
             "partial-coverage perturbations",
             len(valid_kos), len(non_control), min_cells, len(bin_names),
             len(non_control) - len(valid_kos))
    return adata[keep_mask].copy()


# ===================================================================
# Internal partition helpers (used by multi-bin split functions)
# ===================================================================

def _partition_into_blocks(n: int, n_blocks: int, seed: int) -> List[np.ndarray]:
    """Return n_blocks lists of indices [0, n), as a random partition."""
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    return [block.copy() for block in np.array_split(perm, n_blocks)]


def _fold_assignment(
    blocks: List[np.ndarray], fold: int, n_folds: int,
) -> Tuple[List[int], List[int], List[int]]:
    """For a given fold, split blocks into (train_idx, val_idx, test_idx).
    Test = current fold's block. Val = first half of next fold's block. Train = rest.
    """
    test_indices = set(blocks[fold].tolist())
    val_block = blocks[(fold + 1) % n_folds]
    val_indices = set(val_block[: len(val_block) // 2].tolist())
    all_indices: set = set()
    for b in blocks:
        all_indices.update(b.tolist())
    train_indices = all_indices - test_indices - val_indices
    return sorted(train_indices), sorted(val_indices), sorted(test_indices)


def _single_perturbations(non_ctrl: List[str]) -> List[str]:
    """Sorted single-gene, non-dose perturbations (drop combos '+' and doses '@').
    Shared by the multi-bin scenarios whose mask axes are over single perturbations.
    """
    return sorted(c for c in non_ctrl if "+" not in c and "@" not in c)


def _store_pair_masks(
    adata: ad.AnnData,
    scenario: str,
    fold: int,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    test_mask: np.ndarray,
    bin_names: List[str],
    singles: List[str],
) -> None:
    """Store per-fold (n_bins, n_singles) train/val/test masks + their axis-label
    vectors in ``adata.uns`` under the shared ``{scenario}_fold_{N}_*`` convention.

    Used by the non-rectangular multi-bin scenarios (UnseenPair, UnseenBoth) whose
    train region is NOT the outer product of bin/ko marginals and must therefore be
    carried explicitly. ``DatasetStore._load_pair_masks`` reindexes these to the
    canonical store order and verifies the names, so the key/axis convention MUST be
    identical across scenarios.
    """
    p = f"{scenario}_fold_{fold}"
    adata.uns[f"{p}_train_mask"] = train_mask
    adata.uns[f"{p}_val_mask"] = val_mask
    adata.uns[f"{p}_test_mask"] = test_mask
    adata.uns[f"{p}_single_conditions"] = np.array(list(singles))
    adata.uns[f"{p}_bin_names"] = np.array(list(bin_names))


# ===================================================================
# Split invariants — pure predicates + a generation-time assert
#
# These operate on (n_bins, n_ko) boolean mask grids — the SAME shape the
# benchmark verifier builds from SplitInfo.{train,test}_mask_2d — so ONE
# definition is shared by both the pytest unit tests and benchmark/verify.py
# (which imports from this module; the dependency arrow is scripts→core, never
# the reverse). Keep them pure and dependency-light (numpy only).
#
# Control columns are excluded by callers where relevant: single-bin UnseenPert
# round-robins control CELLS across folds, so the control ko-column legitimately
# appears in both train and test — that is not leakage.
# ===================================================================

def check_no_leakage(train_mask: np.ndarray, test_mask: np.ndarray) -> bool:
    """No (bin, ko) grid cell is both train and test."""
    return not bool(np.logical_and(np.asarray(train_mask, bool),
                                   np.asarray(test_mask, bool)).any())


def check_no_empty_fold(*masks: np.ndarray) -> bool:
    """Every provided mask has at least one True cell."""
    return all(bool(np.asarray(m, bool).any()) for m in masks)


def check_pert_consistent_across_bins(mask: np.ndarray) -> bool:
    """Every perturbation (column) is held the SAME way in all bins (rows): each
    column is all-True or all-False. This is the defining property of a multi-bin
    UnseenPert split (a perturbation is unseen in every cell type or in none) and
    is exactly what a per-bin UnseenPert mistake would violate. Single-bin grids
    (one row) trivially satisfy it.
    """
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError(f"expected a 2D (n_bins, n_ko) mask, got shape {mask.shape}")
    return bool((mask.any(axis=0) == mask.all(axis=0)).all())


def check_full_coverage(
    test_masks_by_fold: List[np.ndarray], non_control_cols: np.ndarray
) -> bool:
    """Across folds, every non-control perturbation is in the test set exactly
    once (the UnseenPert round-robin property). ``non_control_cols`` is a 1D bool
    over the ko axis (True = real perturbation, False = control).
    """
    non_control_cols = np.asarray(non_control_cols, dtype=bool)
    per_fold_tested = [np.asarray(m, bool).any(axis=0) for m in test_masks_by_fold]
    times_tested = np.sum(per_fold_tested, axis=0)  # (n_ko,)
    return bool((times_tested[non_control_cols] == 1).all())


def check_unseenpert_subset_of_unseenboth(up_test_kos, ub_test_kos) -> bool:
    """The per-fold UnseenPert test perturbations are a subset of the UnseenBoth
    held-out perturbation block for the same fold — the perturbench convention
    where both scenarios derive from the SAME seed-0 block partition.
    """
    return set(up_test_kos) <= set(ub_test_kos)


def check_unseenpair_pert_in_train_at_least_once(train_mask: np.ndarray) -> bool:
    """Every perturbation (column) appears in TRAIN in at least one bin (row).

    The defining property of UnseenPair (vs UnseenPert): a held-out
    (cell_type, pert) pair is unseen, but the perturbation itself must still be
    SEEN during training in at least one OTHER cell type — otherwise the pair task
    degenerates into UnseenPert for that perturbation. ``train_mask`` is the
    (n_bins, n_ko) train grid (non-control columns); returns True iff no column is
    all-False. Single-bin grids are nonsensical for UnseenPair (it needs ≥2 bins).
    """
    train_mask = np.asarray(train_mask, dtype=bool)
    if train_mask.ndim != 2:
        raise ValueError(f"expected a 2D (n_bins, n_ko) mask, got shape {train_mask.shape}")
    return bool(train_mask.any(axis=0).all())


_SPLIT_COL_RE = re.compile(r"^split_(?P<sc>.+)_fold_(?P<fold>\d+)$")


def _enumerate_split_columns(adata: ad.AnnData, scenarios=None):
    """Yield (scenario, fold, column_name) for every split_{scenario}_fold_{N}
    obs column, sorted by (scenario, fold). ``scenarios`` optionally restricts.
    """
    found: Dict[str, List[Tuple[int, str]]] = {}
    for col in adata.obs.columns:
        m = _SPLIT_COL_RE.match(str(col))
        if not m:
            continue
        sc = m.group("sc")
        if scenarios is not None and sc not in scenarios:
            continue
        found.setdefault(sc, []).append((int(m.group("fold")), col))
    for sc in sorted(found):
        for fold, col in sorted(found[sc]):
            yield sc, fold, col


def _grid_masks_from_obs(
    adata: ad.AnnData, col: str, *, pert_col: str, bin_col: str
):
    """Build (n_bins, n_ko) train/test boolean grids + a non-control column mask
    from a per-cell split column. A (bin, ko) cell is train/test if any cell with
    that (bin, ko) carries that label.
    """
    obs = adata.obs
    has_bin = bin_col in obs.columns
    bins = sorted(map(str, obs[bin_col].unique())) if has_bin else ["all"]
    kos = sorted(map(str, obs[pert_col].unique()))
    bidx = {b: i for i, b in enumerate(bins)}
    kidx = {k: i for i, k in enumerate(kos)}
    train = np.zeros((len(bins), len(kos)), dtype=bool)
    test = np.zeros((len(bins), len(kos)), dtype=bool)
    label = obs[col].astype(str)
    for lab, target in (("train", train), ("test", test)):
        sub = obs.loc[label.values == lab]
        if has_bin:
            pairs = sub[[bin_col, pert_col]].astype(str).drop_duplicates()
            for b, k in zip(pairs[bin_col], pairs[pert_col]):
                target[bidx[b], kidx[k]] = True
        else:
            for k in sub[pert_col].astype(str).unique():
                target[0, kidx[k]] = True
    non_control = np.array([not _is_control_label(k) for k in kos], dtype=bool)
    return train, test, non_control


def assert_split_invariants(
    adata: ad.AnnData,
    *,
    pert_col: str = "condition",
    bin_col: str = "cell_type",
    scenarios=None,
) -> None:
    """Generation-time gate — raise if any split fold column violates a universal
    invariant. Call at the tail of every get_data.py so a corrupt split is never
    written to disk.

    Checks (per scenario × fold, on NON-control columns): no train∩test leakage;
    train and test both non-empty. For UnseenPert additionally: each perturbation
    is held the same way across all bins. For UnseenPair additionally: every
    perturbation is seen in train in at least one bin (no degenerate-to-UnseenPert
    pert). Convention / cross-scenario checks (UnseenPert⊆UnseenBoth, full
    round-robin coverage) live in benchmark/verify.py.
    """
    n = 0
    for sc, fold, col in _enumerate_split_columns(adata, scenarios):
        train, test, non_control = _grid_masks_from_obs(
            adata, col, pert_col=pert_col, bin_col=bin_col)
        tr, te = train[:, non_control], test[:, non_control]
        if not check_no_leakage(tr, te):
            raise ValueError(f"split invariant FAILED: {col} has train∩test leakage")
        if not check_no_empty_fold(tr, te):
            raise ValueError(f"split invariant FAILED: {col} has an empty train or test set")
        if sc == "UnseenPert" and not check_pert_consistent_across_bins(te):
            raise ValueError(
                f"split invariant FAILED: {col} (UnseenPert) is not perturbation-"
                f"consistent across bins — a perturbation is test in some cell types "
                f"but not others (per-bin UnseenPert mistake?)")
        if sc == "UnseenPair" and not check_unseenpair_pert_in_train_at_least_once(tr):
            raise ValueError(
                f"split invariant FAILED: {col} (UnseenPair) has a perturbation with "
                f"no train bin — all its (cell_type, pert) versions are val/test, so it "
                f"degenerates into UnseenPert for that perturbation")
        n += 1
    log.info("assert_split_invariants: %d (scenario, fold) split column(s) OK", n)


# ===================================================================
# Split fold assignment
#
# Single-bin scenarios: UnseenPert (PMOB literature), UnseenCombo (combos→split),
#                        UnseenDose (dose-based).
# Multi-bin scenarios: UnseenCell, UnseenBoth, UnseenPair (VCR mcfaline23 pattern).
# ===================================================================


def assign_split_folds_unseen_pert(
    adata: ad.AnnData,
    *,
    pert_col: str = "condition",
    n_folds: int = 5,
    seed: int = 42,
) -> None:
    """UnseenPert (formerly S1): unseen perturbations across folds.

    Algorithm matches PMOB adamson16/get_data.py: round-robin assign each
    perturbation to a test fold, deterministic 70/10/20 split per fold,
    controls round-robined into the same folds for completeness.

    Writes adata.obs[f"split_UnseenPert_fold_{N}"] for N in range(n_folds).
    Values: 'train' | 'val' | 'test' | 'unassigned'.
    """
    all_conditions = adata.obs[pert_col].unique()
    non_control = sorted([c for c in all_conditions if not _is_control_label(c)])
    control_cells = sorted(adata.obs[adata.obs[pert_col].apply(_is_control_label)].index.tolist())

    # Two INDEPENDENT generators, and no touching of the global RNG.
    # Previously this seeded the global np.random and drew both permutations from
    # the same stream: `permutation` consumes state proportional to input length,
    # so the control-cell assignment silently depended on how many PERTURBATIONS
    # existed — add or drop one perturbation and every control cell moved fold.
    # Seeding a local RandomState with `seed` reproduces the perturbation draw
    # exactly; the control draw gets its own stream (`seed + 1`).
    shuffled_conditions = np.random.RandomState(seed).permutation(non_control)
    shuffled_control_cells = np.random.RandomState(seed + 1).permutation(control_cells)

    n_conditions = len(shuffled_conditions)
    conditions_per_fold = n_conditions // n_folds
    remainder_conditions = n_conditions % n_folds

    n_control_cells = len(shuffled_control_cells)
    control_cells_per_fold = n_control_cells // n_folds
    remainder_control = n_control_cells % n_folds

    # Pre-assign conditions and control cells to test folds
    condition_to_test_fold: Dict[str, int] = {}
    idx = 0
    for fold in range(n_folds):
        fold_size = conditions_per_fold + (1 if fold < remainder_conditions else 0)
        for cond in shuffled_conditions[idx:idx + fold_size]:
            condition_to_test_fold[cond] = fold
        idx += fold_size

    control_to_test_fold: Dict[str, int] = {}
    idx = 0
    for fold in range(n_folds):
        fold_size = control_cells_per_fold + (1 if fold < remainder_control else 0)
        for cell in shuffled_control_cells[idx:idx + fold_size]:
            control_to_test_fold[cell] = fold
        idx += fold_size

    for fold in range(n_folds):
        col = f"split_UnseenPert_fold_{fold}"
        test_conditions = [c for c, f in condition_to_test_fold.items() if f == fold]
        remaining_conditions = [c for c, f in condition_to_test_fold.items() if f != fold]
        # 10% of total = 13.75% of remaining (since test is 20%, remaining is 80%)
        n_val = int(len(remaining_conditions) * 0.1375)
        val_conditions = remaining_conditions[:n_val]
        train_conditions = remaining_conditions[n_val:]

        test_control = [c for c, f in control_to_test_fold.items() if f == fold]
        remaining_control = [c for c, f in control_to_test_fold.items() if f != fold]
        n_val_ctrl = int(len(remaining_control) * 0.1375)
        val_control = remaining_control[:n_val_ctrl]
        train_control = remaining_control[n_val_ctrl:]

        adata.obs[col] = "unassigned"
        adata.obs.loc[adata.obs[pert_col].isin(test_conditions), col] = "test"
        adata.obs.loc[adata.obs[pert_col].isin(val_conditions), col] = "val"
        adata.obs.loc[adata.obs[pert_col].isin(train_conditions), col] = "train"
        adata.obs.loc[test_control, col] = "test"
        adata.obs.loc[val_control, col] = "val"
        adata.obs.loc[train_control, col] = "train"

        counts = adata.obs[col].value_counts().to_dict()
        log.info("  %s: train=%d val=%d test=%d", col,
                 counts.get("train", 0), counts.get("val", 0), counts.get("test", 0))


def assign_split_folds_unseen_pert_multibin(
    adata: ad.AnnData,
    *,
    pert_col: str = "condition",
    n_folds: int = 5,
    seed: int = 0,
) -> None:
    """UnseenPert for a MULTI-BIN dataset, using the SAME seed-0 block partition
    as ``assign_split_folds_unseen_both`` so the per-fold test perturbation set
    matches the upstream (perturbench / DL-model) splits.

    The single-bin ``assign_split_folds_unseen_pert`` uses the adamson16
    round-robin (``RandomState(seed)``, alphabetical sort, contiguous slices). For multi-bin datasets the DL pipeline instead derives UnseenPert
    from the SAME perturbation block partition it uses for UnseenBoth, so this
    variant reuses ``_partition_into_blocks`` / ``_fold_assignment`` (seed 0) and
    holds out each perturbation block across ALL bins. Test = block[fold] perts;
    val = first half of block[(fold+1) % n_folds]; train = the rest; controls →
    train. Used by the multi-bin perturbench datasets whose DL splits follow this
    convention (mcfaline23, replogle22, jiang24); single-bin datasets keep the
    original ``assign_split_folds_unseen_pert``.

    Writes adata.obs[f"split_UnseenPert_fold_{N}"]; values train/val/test.
    """
    non_control = _non_control_conditions(adata, pert_col)  # sorted
    n = len(non_control)
    blocks = _partition_into_blocks(n, n_folds, seed)

    for fold in range(n_folds):
        col = f"split_UnseenPert_fold_{fold}"
        train_pos, val_pos, test_pos = _fold_assignment(blocks, fold, n_folds)
        ko_split: Dict[str, str] = {}
        for i in train_pos: ko_split[non_control[i]] = "train"
        for i in val_pos:   ko_split[non_control[i]] = "val"
        for i in test_pos:  ko_split[non_control[i]] = "test"

        labels = adata.obs[pert_col].map(ko_split)
        adata.obs[col] = "train"  # controls + train perts (+ any unmapped)
        adata.obs.loc[labels == "val", col] = "val"
        adata.obs.loc[labels == "test", col] = "test"

        counts = adata.obs[col].value_counts().to_dict()
        log.info("  %s (multibin block partition, seed=%d): train=%d val=%d test=%d",
                 col, seed, counts.get("train", 0), counts.get("val", 0),
                 counts.get("test", 0))


def assign_split_folds_unseen_cell(
    adata: ad.AnnData,
    *,
    bin_col: str = "cell_type",
    n_folds: int = 3,
    seed: int = 0,
) -> None:
    """UnseenCell (formerly S2): unseen cell types — bins rotate per fold.

    Multi-bin only. n_folds is clamped to min(n_folds, n_bins). For each fold,
    one bin is test, the next is val, the rest are train.

    Writes adata.obs[f"split_UnseenCell_fold_{N}"].
    """
    bin_names_sorted = sorted(adata.obs[bin_col].unique())
    n_bins = len(bin_names_sorted)
    if n_bins < 3:
        log.warning("assign_split_folds_unseen_cell: dataset has %d bins (< 3), "
                    "UnseenCell is not applicable; skipping", n_bins)
        return

    n_folds_eff = min(n_folds, n_bins)
    rng = np.random.RandomState(seed)
    bin_perm = rng.permutation(n_bins)
    for fold in range(n_folds_eff):
        col = f"split_UnseenCell_fold_{fold}"
        adata.obs[col] = "unassigned"
        test_bin = bin_names_sorted[bin_perm[fold]]
        val_bin = bin_names_sorted[bin_perm[(fold + 1) % n_bins]]
        train_bins = set(bin_names_sorted) - {test_bin, val_bin}
        for bn in train_bins:
            adata.obs.loc[adata.obs[bin_col] == bn, col] = "train"
        adata.obs.loc[adata.obs[bin_col] == val_bin, col] = "val"
        adata.obs.loc[adata.obs[bin_col] == test_bin, col] = "test"

        counts = adata.obs[col].value_counts().to_dict()
        log.info("  %s: train_bins=%s val_bin=%s test_bin=%s "
                 "(train=%d val=%d test=%d)", col,
                 sorted(train_bins), val_bin, test_bin,
                 counts.get("train", 0), counts.get("val", 0), counts.get("test", 0))


def assign_split_folds_unseen_both(
    adata: ad.AnnData,
    *,
    pert_col: str = "condition",
    bin_col: str = "cell_type",
    n_folds: int = 5,
    seed: int = 0,
) -> None:
    """UnseenBoth (formerly S3): unseen perturbations AND unseen cell types.

    Each fold has unseen perts AND a rotated bin. Tests cells that are both
    in a held-out perturbation set AND in a held-out bin. Cells that are
    neither in the held-out pert set nor held-out bin are 'train'; others
    are 'unassigned' (mixed-overlap cells).

    Multi-bin only. Writes adata.obs[f"split_UnseenBoth_fold_{N}"] and stores
    boolean masks (n_bins, n_singles) in
    adata.uns[f"UnseenBoth_fold_{N}_{train,val,test}_mask"] (same convention as
    UnseenPair), so predictors train on the exact labelled cells rather than the
    leaky train_bins × train_kos rectangle.
    """
    bin_names_sorted = sorted(adata.obs[bin_col].unique())
    n_bins = len(bin_names_sorted)
    if n_bins < 2:
        log.warning("assign_split_folds_unseen_both: dataset has %d bins (< 2), "
                    "UnseenBoth is not applicable; skipping", n_bins)
        return

    non_ctrl = _non_control_conditions(adata, pert_col)
    singles = _single_perturbations(non_ctrl)
    n_singles = len(singles)

    blocks = _partition_into_blocks(n_singles, n_folds, seed)
    rng = np.random.RandomState(seed)
    bin_perm = rng.permutation(n_bins)

    for fold in range(n_folds):
        col = f"split_UnseenBoth_fold_{fold}"
        train_pos, val_pos, test_pos = _fold_assignment(blocks, fold, n_folds)
        ko_split: Dict[str, str] = {}
        for i in train_pos: ko_split[singles[i]] = "train"
        for i in val_pos:   ko_split[singles[i]] = "val"
        for i in test_pos:  ko_split[singles[i]] = "test"

        fold_bin = fold % n_bins
        test_bin = bin_names_sorted[bin_perm[fold_bin]]
        val_bin = bin_names_sorted[bin_perm[(fold_bin + 1) % n_bins]]

        conds = adata.obs[pert_col]
        bins = adata.obs[bin_col]
        # `.apply` over a CATEGORICAL condition column returns a Categorical of bools,
        # and `~Categorical` raises TypeError. Force a plain bool Series so the mask
        # algebra below works regardless of the source dtype.
        is_ctrl = pd.Series(
            conds.astype(str).map(_is_control_label).to_numpy(dtype=bool),
            index=adata.obs.index)
        ko_labels = conds.map(ko_split)

        adata.obs[col] = "train"  # controls + anything not in ko_split
        test_mask = (ko_labels == "test") & (bins == test_bin) & ~is_ctrl
        adata.obs.loc[test_mask, col] = "test"
        val_mask = (ko_labels == "val") & (bins == val_bin) & ~is_ctrl
        adata.obs.loc[val_mask, col] = "val"
        unassigned_mask = ko_labels.isna() & ~is_ctrl
        adata.obs.loc[unassigned_mask, col] = "unassigned"

        # Store the (n_bins, n_singles) train/val/test masks (same machinery as
        # UnseenPair). The held-out test/val cells form a sub-rectangle (held-out
        # perts × one held-out bin), but the TRAIN region is the grid MINUS those
        # corners — NOT the outer product of train_bins × train_kos — so it cannot
        # be reconstructed from marginal index lists and must be carried explicitly,
        # otherwise predictors would re-include the held-out corner and leak. These
        # masks mirror the obs labels above exactly (singles only; controls/combos
        # are not part of the mask axes, as in UnseenPair).
        te2d = np.zeros((n_bins, n_singles), dtype=bool)
        va2d = np.zeros((n_bins, n_singles), dtype=bool)
        te2d[bin_perm[fold_bin], test_pos] = True
        va2d[bin_perm[(fold_bin + 1) % n_bins], val_pos] = True
        tr2d = ~te2d & ~va2d
        _store_pair_masks(adata, "UnseenBoth", fold, tr2d, va2d, te2d,
                          bin_names_sorted, singles)

        counts = adata.obs[col].value_counts().to_dict()
        log.info("  %s: train=%d val=%d test=%d", col,
                 counts.get("train", 0), counts.get("val", 0), counts.get("test", 0))


def assign_split_folds_unseen_pair(
    adata: ad.AnnData,
    *,
    pert_col: str = "condition",
    bin_col: str = "cell_type",
    n_folds: int = 5,
    seed: int = 0,
) -> None:
    """UnseenPair (formerly S4): unseen (cell_type, perturbation) pairs.

    Each (bin, ko) pair is its own unit; the n_bins*n_kos pairs are partitioned
    into n_folds blocks. For each fold, test cells are exactly the cells whose
    (bin, ko) pair is in the test block. Both bins AND kos appear in train
    and test, but the SPECIFIC combinations differ.

    Invariant: every perturbation is seen in train in at least one cell type — a
    repair pass moves a pair back to train for any pert whose bin-copies all landed
    in val/test (which would degenerate UnseenPair into UnseenPert). With enough
    bins this never fires (jiang24, 6 bins → 0 repairs); with 3 bins (mcfaline23) it
    fires for a handful of perts per fold.

    Multi-bin only. Writes adata.obs[f"split_UnseenPair_fold_{N}"] and stores
    boolean masks (n_bins, n_kos) in adata.uns[f"UnseenPair_fold_{N}_{train,val,test}_mask"].
    """
    bin_names_sorted = sorted(adata.obs[bin_col].unique())
    n_bins = len(bin_names_sorted)
    if n_bins < 2:
        log.warning("assign_split_folds_unseen_pair: dataset has %d bins (< 2), "
                    "UnseenPair is not applicable; skipping", n_bins)
        return
    bin_to_idx = {b: i for i, b in enumerate(bin_names_sorted)}

    non_ctrl = _non_control_conditions(adata, pert_col)
    singles = _single_perturbations(non_ctrl)
    n_singles = len(singles)
    cond_to_single_idx = {c: i for i, c in enumerate(singles)}

    n_pairs = n_bins * n_singles
    blocks = _partition_into_blocks(n_pairs, n_folds, seed)
    ctrl_conds = _control_conditions(adata, pert_col)

    for fold in range(n_folds):
        col = f"split_UnseenPair_fold_{fold}"
        adata.obs[col] = "unassigned"
        train_pos, val_pos, test_pos = _fold_assignment(blocks, fold, n_folds)

        train_mask = np.zeros((n_bins, n_singles), dtype=bool)
        val_mask = np.zeros((n_bins, n_singles), dtype=bool)
        test_mask = np.zeros((n_bins, n_singles), dtype=bool)
        for idx in train_pos: train_mask[idx // n_singles, idx % n_singles] = True
        for idx in val_pos:   val_mask[idx // n_singles, idx % n_singles] = True
        for idx in test_pos:  test_mask[idx // n_singles, idx % n_singles] = True

        # Guarantee every perturbation is SEEN in training in >=1 cell type: with few
        # bins the random pair partition can land all of a pert's bin-copies in
        # val/test, degenerating UnseenPair into UnseenPert for that pert. Repair by
        # moving one pair back to train — prefer flipping a val pair (keeps every test
        # pair); else flip the first test pair. Deterministic (lowest bin index); a
        # no-op when every pert already has a train bin (e.g. jiang24, 6 bins → 0
        # repairs → bit-identical output). Enforced by check_unseenpair_pert_in_train_
        # at_least_once in assert_split_invariants and benchmark/verify.py (L1d).
        for c in range(n_singles):
            if not train_mask[:, c].any():
                val_bins = np.where(val_mask[:, c])[0]
                if val_bins.size:
                    b = int(val_bins[0]); val_mask[b, c] = False
                else:
                    b = int(np.where(test_mask[:, c])[0][0]); test_mask[b, c] = False
                train_mask[b, c] = True

        for cond in singles:
            ci = cond_to_single_idx[cond]
            cond_mask = adata.obs[pert_col] == cond
            for bn in bin_names_sorted:
                bi = bin_to_idx[bn]
                cell_mask = cond_mask & (adata.obs[bin_col] == bn)
                if train_mask[bi, ci]:
                    adata.obs.loc[cell_mask, col] = "train"
                elif val_mask[bi, ci]:
                    adata.obs.loc[cell_mask, col] = "val"
                elif test_mask[bi, ci]:
                    adata.obs.loc[cell_mask, col] = "test"
        for c in ctrl_conds:
            adata.obs.loc[adata.obs[pert_col] == c, col] = "train"

        _store_pair_masks(adata, "UnseenPair", fold, train_mask, val_mask, test_mask,
                          bin_names_sorted, singles)

        counts = adata.obs[col].value_counts().to_dict()
        log.info("  %s: train=%d val=%d test=%d (pairs: %d/%d/%d)", col,
                 counts.get("train", 0), counts.get("val", 0), counts.get("test", 0),
                 train_mask.sum(), val_mask.sum(), test_mask.sum())


def assign_split_folds_unseen_dose(
    adata: ad.AnnData,
    *,
    pert_col: str = "condition",
    n_folds: int = 2,
    seed: int = 0,
) -> None:
    """UnseenDose (formerly S5): unseen dose levels.

    Identifies dose-suffix perturbations (e.g. "GeneA@0.25") and partitions
    the doses into n_folds blocks. Full-dose ("GeneA" without suffix) goes
    to train. Per fold: cells with a dose in the test block → test;
    cells with a dose in the val block → val; everything else → train.

    Single-bin or multi-bin. Writes adata.obs[f"split_UnseenDose_fold_{N}"].
    """
    all_conditions = adata.obs[pert_col].unique()
    dose_perts = [c for c in all_conditions
                  if not _is_control_label(c) and "@" in str(c)]
    if not dose_perts:
        log.warning("assign_split_folds_unseen_dose: no dose-suffix perturbations "
                    "found ('@' missing in all conditions); skipping")
        return

    rng = np.random.RandomState(seed)
    shuffled = list(rng.permutation(dose_perts))
    blocks = np.array_split(np.array(shuffled), n_folds)

    for fold in range(n_folds):
        col = f"split_UnseenDose_fold_{fold}"
        adata.obs[col] = "train"
        test_set = set(blocks[fold].tolist())
        val_block = blocks[(fold + 1) % n_folds]
        n_val = max(1, len(val_block) // 2)
        val_set = set(val_block[:n_val].tolist())
        adata.obs.loc[adata.obs[pert_col].isin(test_set), col] = "test"
        adata.obs.loc[adata.obs[pert_col].isin(val_set), col] = "val"
        counts = adata.obs[col].value_counts().to_dict()
        log.info("  %s: train=%d val=%d test=%d (%d test doses, %d val doses)", col,
                 counts.get("train", 0), counts.get("val", 0), counts.get("test", 0),
                 len(test_set), len(val_set))


def assign_split_folds_unseen_combo(
    adata: ad.AnnData,
    *,
    pert_col: str = "condition",
    n_folds: int = 2,
    seed: int = 42,
) -> None:
    """UnseenCombo (formerly S6): unseen combinatorial perturbations.

    Single-gene perturbations all go to train. Combo perturbations
    (containing '+') are partitioned into n_folds blocks; each fold's test
    set is one block, val is half of next block, train is the rest.

    Single-bin (only datasets with combos: norman19, wessels23, replogle20,
    sunshine23). Writes adata.obs[f"split_UnseenCombo_fold_{N}"].
    """
    all_conditions = list(adata.obs[pert_col].unique())
    single_perts = [c for c in all_conditions
                    if not _is_control_label(c) and "+" not in c]
    combo_perts = [c for c in all_conditions
                   if not _is_control_label(c) and "+" in c]

    if not combo_perts:
        log.warning("assign_split_folds_unseen_combo: no combo perturbations found "
                    "('+' missing in all conditions); skipping")
        return

    log.info("assign_split_folds_unseen_combo: %d singles, %d combos",
             len(single_perts), len(combo_perts))

    rng = np.random.RandomState(seed)
    shuffled_combos = list(rng.permutation(combo_perts))
    n_test = len(shuffled_combos) // n_folds
    blocks = [shuffled_combos[i * n_test:(i + 1) * n_test] for i in range(n_folds)]
    # Last block absorbs any remainder
    if len(shuffled_combos) % n_folds != 0:
        blocks[-1].extend(shuffled_combos[n_folds * n_test:])

    for fold in range(n_folds):
        col = f"split_UnseenCombo_fold_{fold}"
        adata.obs[col] = "unassigned"

        test_block = set(blocks[fold])
        val_block_raw = blocks[(fold + 1) % n_folds]
        n_val = max(1, len(val_block_raw) // 2)
        val_block = set(val_block_raw[:n_val])
        train_combos = set(shuffled_combos) - test_block - val_block

        # Singles → train
        for s in single_perts:
            adata.obs.loc[adata.obs[pert_col] == s, col] = "train"
        # Controls → train
        for c in _control_conditions(adata, pert_col):
            adata.obs.loc[adata.obs[pert_col] == c, col] = "train"
        # Combos partitioned
        for combo in train_combos:
            adata.obs.loc[adata.obs[pert_col] == combo, col] = "train"
        for combo in val_block:
            adata.obs.loc[adata.obs[pert_col] == combo, col] = "val"
        for combo in test_block:
            adata.obs.loc[adata.obs[pert_col] == combo, col] = "test"

        counts = adata.obs[col].value_counts().to_dict()
        log.info("  %s: train=%d val=%d test=%d "
                 "(combos: train=%d val=%d test=%d)", col,
                 counts.get("train", 0), counts.get("val", 0), counts.get("test", 0),
                 len(train_combos), len(val_block), len(test_block))


# ===================================================================
# DEGs (literature format, both halves)
# ===================================================================


def _group_indicator(codes, n_cells: int, n_groups: int):
    """One-hot ``(n_cells, n_groups)`` CSR from integer group codes.

    Cells whose code is < 0 are excluded (no row entry). Returns
    ``(indicator, counts)`` where ``counts`` is ``(n_groups,)`` float64 — the
    per-group cell count, i.e. ``indicator.sum(axis=0)``.
    """
    from scipy.sparse import csr_matrix as _csr
    codes = np.asarray(codes)
    valid = codes >= 0
    rows = np.arange(n_cells, dtype=np.int32)[valid]
    cols = codes[valid].astype(np.int32)
    indicator = _csr(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)),
        shape=(n_cells, n_groups),
    )
    counts = np.asarray(indicator.sum(axis=0), dtype=np.float64).ravel()
    return indicator, counts


def _grouped_sum(indicator, X, chunk_size: Optional[int] = None) -> np.ndarray:
    """Dense ``(n_groups, n_genes)`` float64 group sums = ``indicator.T @ X``.

    ``chunk_size=None`` (default): one sparse matmul over all groups at once
    (works for sparse or dense in-memory X) — unchanged behaviour.
    ``chunk_size=N``: stream ``X`` in row-chunks of N (supports a backed
    ``_CSRDataset``; loads only one chunk at a time). Numerically equivalent up
    to float reduction order.
    """
    if chunk_size is None:
        s = indicator.T @ X
        if issparse(s):
            s = s.todense()
        return np.asarray(s, dtype=np.float64)
    from scipy.sparse import csr_matrix as _csr
    n_cells = X.shape[0]
    out = np.zeros((indicator.shape[1], X.shape[1]), dtype=np.float64)
    for i in range(0, n_cells, chunk_size):
        xc = X[i:i + chunk_size]
        if not issparse(xc):
            xc = _csr(xc)
        s = indicator[i:i + chunk_size].T @ xc
        out += np.asarray(s.todense() if issparse(s) else s, dtype=np.float64)
    return out


def _grouped_sum_and_sq(indicator, X, chunk_size: Optional[int] = None):
    """Return ``(group_sums, group_sum_sq)`` = ``indicator.T @ X`` and
    ``indicator.T @ X**2`` in a SINGLE streamed pass (one read of each backed
    chunk). Used by the chunked t-test path so a backed X is read once.
    """
    from scipy.sparse import csr_matrix as _csr
    n_cells, n_genes = X.shape[0], X.shape[1]
    n_groups = indicator.shape[1]
    cs = chunk_size if chunk_size is not None else n_cells
    sums = np.zeros((n_groups, n_genes), dtype=np.float64)
    sumsq = np.zeros((n_groups, n_genes), dtype=np.float64)
    for i in range(0, n_cells, cs):
        xc = X[i:i + cs]
        if not issparse(xc):
            xc = _csr(xc)
        else:
            xc = xc.tocsr()
        ind_c = indicator[i:i + cs]
        s = ind_c.T @ xc
        sums += np.asarray(s.todense() if issparse(s) else s, dtype=np.float64)
        xsq = xc.copy()
        xsq.data **= 2
        sq = ind_c.T @ xsq
        sumsq += np.asarray(sq.todense() if issparse(sq) else sq, dtype=np.float64)
    return sums, sumsq


def _bh_adjust_rows(pvals: np.ndarray, valid_rows: Optional[np.ndarray] = None) -> np.ndarray:
    """Benjamini-Hochberg FDR per row of a (n_rows, n_genes) p-value matrix.

    Reusable so the HVG-only variant can recompute FDR over the HVG gene set from
    sliced *unadjusted* p-values (slicing already-adjusted p-values would keep the
    union-panel denominator and mis-threshold boundary genes). Rows not in
    `valid_rows` are left as 1.0.
    """
    pvals = np.asarray(pvals, dtype=np.float64)
    n_rows, n_genes = pvals.shape
    ranks = np.arange(1, n_genes + 1, dtype=np.float64)
    out = np.ones_like(pvals)
    rows = range(n_rows) if valid_rows is None else [int(r) for r in valid_rows]
    for gi in rows:
        sort_idx = np.argsort(pvals[gi])
        sorted_pv = pvals[gi, sort_idx]
        adjusted = sorted_pv * n_genes / ranks
        adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
        out[gi, sort_idx] = np.clip(adjusted, 0.0, 1.0)
    return out


def _ttest_overestim_var_matmul(
    X,
    labels: np.ndarray,
    group_names: List[str],
    chunk_size: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized overestim-var t-test against rest, BH-adjusted per group.

    Matches scanpy.tl.rank_genes_groups(method='t-test_overestim_var', reference='rest'):
      Welch's t-test where the rest sample size is hacked to equal the group
      size (the "overestim variance" trick that makes the test conservative
      for small groups). Welch-Satterthwaite degrees of freedom.

      n_g     = group cell count
      n_r     = n_g                              (overestim_var hack)
      score   = (mean_g - mean_rest) / sqrt(var_g/n_g + var_rest/n_g)
      df      = Welch-Satterthwaite from var_g/n_g, var_rest/n_g
      pvalue  = 2 * t.sf(|score|, df), BH-adjusted per group

    Implementation:
      indicator = (n_cells, n_groups) sparse one-hot of `labels` onto `group_names`
      group_sums  = indicator.T @ X            -- one matmul, all groups
      group_sumsq = indicator.T @ (X ** 2)     -- one matmul, all groups

    Returns:
      scores:      (n_groups_present, n_genes) float32
      pvals_unadj: (n_groups_present, n_genes) float32  (raw two-sided p-values)
      pvals_adj:   (n_groups_present, n_genes) float32  (BH per row)
      present:     (n_groups_present,) indices into `group_names` of groups
                   with >= 1 cell present in `labels`.
    """
    from scipy.stats import t as t_dist

    n_cells, n_genes = X.shape
    name_to_code = {n: i for i, n in enumerate(group_names)}
    codes = np.array([name_to_code.get(str(l), -1) for l in labels], dtype=np.int32)
    if not (codes >= 0).any():
        return (np.zeros((0, n_genes), dtype=np.float32),
                np.ones((0, n_genes), dtype=np.float32),
                np.ones((0, n_genes), dtype=np.float32),
                np.zeros(0, dtype=np.int64))

    n_groups = len(group_names)
    indicator, group_counts = _group_indicator(codes, n_cells, n_groups)
    if chunk_size is None:
        # In-memory path (unchanged): copy + square in place to avoid a 2nd big alloc.
        if not issparse(X):
            from scipy.sparse import csr_matrix
            Xc = csr_matrix(X)
        else:
            Xc = X.tocsr().copy()
        group_sums = _grouped_sum(indicator, Xc)     # one matmul, all groups
        Xc.data **= 2                                # square in place (Xc is a copy)
        group_sum_sq = _grouped_sum(indicator, Xc)   # one matmul, all groups
        del Xc
    else:
        # Streamed path: read each row-chunk of (possibly backed) X once,
        # accumulating both the sums and the sum-of-squares.
        group_sums, group_sum_sq = _grouped_sum_and_sq(indicator, X, chunk_size)

    # rest stats = total - group; total considers only cells assigned to a group.
    n_total = float(group_counts.sum())
    total_sum = group_sums.sum(axis=0)
    total_sum_sq = group_sum_sq.sum(axis=0)

    gc_ = np.maximum(group_counts[:, None], 1.0)
    group_means = group_sums / gc_
    group_vars = (group_sum_sq / gc_ - group_means ** 2) * gc_ / np.maximum(gc_ - 1, 1.0)

    rest_counts = n_total - group_counts
    rc = np.maximum(rest_counts[:, None], 1.0)
    rest_sums = total_sum[None, :] - group_sums
    rest_sum_sq = total_sum_sq[None, :] - group_sum_sq
    rest_means = rest_sums / rc
    rest_vars = (rest_sum_sq / rc - rest_means ** 2) * rc / np.maximum(rc - 1, 1.0)
    del group_sums, group_sum_sq, rest_sums, rest_sum_sq

    # Welch's t-test with scanpy's overestim_var hack: rest sample size := group size.
    # This makes the test conservative for small groups.
    n_g = gc_                       # (n_groups, 1)
    n_r = gc_                       # (n_groups, 1)  -- the hack
    se_sq_g = group_vars / n_g
    se_sq_r = rest_vars / n_r
    se_sq = se_sq_g + se_sq_r
    denom = np.sqrt(np.maximum(se_sq, 0.0))
    scores = np.where(denom > 0, (group_means - rest_means) / denom, 0.0)
    # Welch-Satterthwaite df
    n_g_m1 = np.maximum(n_g - 1.0, 1.0)
    n_r_m1 = np.maximum(n_r - 1.0, 1.0)
    df_num = se_sq ** 2
    df_den = (se_sq_g ** 2) / n_g_m1 + (se_sq_r ** 2) / n_r_m1
    df = np.where(df_den > 0, df_num / df_den, np.maximum(n_g + n_r - 2.0, 1.0))
    pvals = 2.0 * t_dist.sf(np.abs(scores), df)

    # BH FDR per row (only groups with cells present)
    present = np.where(group_counts > 0)[0]
    pvals_adj = _bh_adjust_rows(pvals, valid_rows=present)

    return (scores[present].astype(np.float32),
            pvals[present].astype(np.float32),
            pvals_adj[present].astype(np.float32),
            present.astype(np.int64))


def compute_degs_vectorized(
    adata: ad.AnnData,
    *,
    half: str,
    pert_col: str = "condition",
    bin_col: Optional[str] = None,
    min_cells: int = MIN_CELLS_DEGS_DEFAULT,
    dataset_name: Optional[str] = None,
    chunk_size: Optional[int] = None,
) -> None:
    """Compute DEGs for one half via vectorized sparse-matmul t-test.

    The legacy hand-rolled t-test path (superseded by `compute_degs_pdex`),
    ~100-1000x faster than a scanpy per-group loop
    on datasets with many perturbations (a single matmul does the work that
    scanpy does in a per-group loop). Required for xatlas_orion (18k singles
    -> scanpy would take ~6h per half; this finishes in ~5 min per bin).

    Writes to adata.uns:
      * scores_matrix_{half}:    (n_bins, n_kos, n_genes) float32
                                  var_names-aligned, ko_names-aligned
      * pvals_adj_matrix_{half}: (n_bins, n_kos, n_genes) float32
      * ko_names_matrix:         (n_kos,) string  — KOs the matrices cover
      * bin_names_matrix:        (n_bins,) string — bins the matrices cover

    ko_names_matrix / bin_names_matrix are read from adata.uns["pseudobulk"]
    so the matrices align with the pseudobulk arrays. **Run compute_pseudobulk
    BEFORE compute_degs_vectorized.**

    Also writes EMPTY per-pert dicts (names_df_dict_{half} = {}, etc.)
    so any downstream code that probes for those keys finds them (returns {}).
    The benchmark loaders prefer the matrix path when present.
    """
    assert half in ("first_half", "second_half"), f"invalid half: {half!r}"
    if "pseudobulk" not in adata.uns:
        raise RuntimeError(
            "compute_degs_vectorized requires compute_pseudobulk to have run "
            "first (ko_names / bin_names are read from adata.uns['pseudobulk'])."
        )

    if dataset_name is None:
        dataset_name = adata.uns.get("dataset_name", "ds")

    pb = adata.uns["pseudobulk"]
    ko_names_pb = [str(k) for k in pb["ko_names"]]
    bin_names_pb = [str(b) for b in pb["bin_names"]]
    n_kos = len(ko_names_pb)
    n_bins = len(bin_names_pb)
    n_genes = adata.n_vars

    log.info(
        "compute_degs_vectorized: half=%s, n_bins=%d, n_kos=%d, n_genes=%d, dataset=%s",
        half, n_bins, n_kos, n_genes, dataset_name,
    )

    halves_arr = adata.obs["tech_dup_split"].astype(str).values
    half_mask = (halves_arr == half)
    if not half_mask.any():
        log.warning("  no cells with tech_dup_split==%s; writing zero matrices", half)
        adata.uns[f"scores_matrix_{half}"] = np.zeros((n_bins, n_kos, n_genes), dtype=np.float32)
        adata.uns[f"pvals_adj_matrix_{half}"] = np.ones((n_bins, n_kos, n_genes), dtype=np.float32)
        adata.uns[f"pvals_unadj_matrix_{half}"] = np.ones((n_bins, n_kos, n_genes), dtype=np.float32)
        adata.uns[f"names_df_dict_{half}"] = {}
        adata.uns[f"scores_df_dict_{half}"] = {}
        adata.uns[f"pvals_adj_df_dict_{half}"] = {}
        adata.uns[f"pvals_unadj_df_dict_{half}"] = {}
        adata.uns[f"deg_gene_dict_{half}"] = {}
        return

    if bin_col is None:
        bins_arr = np.zeros(adata.n_obs, dtype=int)
        bin_name_per_idx = ["all"]
    else:
        bin_to_idx_pb = {b: i for i, b in enumerate(bin_names_pb)}
        bins_arr = np.array([bin_to_idx_pb.get(str(b), -1) for b in adata.obs[bin_col].values])
        bin_name_per_idx = bin_names_pb

    conds_arr = adata.obs[pert_col].astype(str).values

    scores_matrix = np.zeros((n_bins, n_kos, n_genes), dtype=np.float32)
    pvals_matrix = np.ones((n_bins, n_kos, n_genes), dtype=np.float32)
    pvals_unadj_matrix = np.ones((n_bins, n_kos, n_genes), dtype=np.float32)

    X = adata.X
    for bi, bn in enumerate(bin_name_per_idx):
        bin_mask = half_mask & (bins_arr == bi)
        if not bin_mask.any():
            log.warning("  bin '%s': no %s cells; skipping", bn, half)
            continue
        # Filter to perts with >= min_cells in this (bin, half)
        cond_counts = pd.Series(conds_arr[bin_mask]).value_counts()
        valid_perts = [p for p in cond_counts.index
                       if not _is_control_label(p)
                       and cond_counts[p] >= min_cells
                       and p in ko_names_pb]
        if not valid_perts:
            log.warning("  bin '%s': no perts with >= %d cells; skipping",
                        bn, min_cells)
            continue
        # Keep only cells belonging to valid_perts (rest cells = union of valid_perts
        # in this bin/half; rank_genes_groups would otherwise mix in non-valid perts
        # as a separate group, but we want "this pert vs all other valid perts +
        # control" — matching scanpy's behavior when groups parameter is given).
        # Legacy semantics: filter to valid_perts only.
        keep_mask = bin_mask & np.isin(conds_arr, valid_perts)
        row_idx = np.where(keep_mask)[0]
        log.info("  bin '%s': %d valid perts, %d %s cells",
                 bn, len(valid_perts), len(row_idx), half)
        if chunk_size is None:
            # In-memory: materialize just this bin/half's rows (unchanged).
            scores_bp, pvals_unadj_bp, pvals_bp, present = _ttest_overestim_var_matmul(
                X[row_idx], conds_arr[row_idx], valid_perts,
            )
        else:
            # Streamed: pass the full (backed) X with labels masked to "" outside
            # keep_mask (→ group code -1 → excluded), so the chunked matmul reads X
            # in row-chunks and never materializes the bin subset.
            labels_full = np.where(keep_mask, conds_arr.astype(object), "")
            scores_bp, pvals_unadj_bp, pvals_bp, present = _ttest_overestim_var_matmul(
                X, labels_full, valid_perts, chunk_size=chunk_size,
            )
        # Map valid_perts present back into ko_names_pb index space
        ko_to_idx_pb = {k: i for i, k in enumerate(ko_names_pb)}
        for local_i, present_idx in enumerate(present):
            pert = valid_perts[int(present_idx)]
            ki = ko_to_idx_pb[pert]
            scores_matrix[bi, ki, :] = scores_bp[local_i]
            pvals_matrix[bi, ki, :] = pvals_bp[local_i]
            pvals_unadj_matrix[bi, ki, :] = pvals_unadj_bp[local_i]
        del scores_bp, pvals_bp, pvals_unadj_bp

    adata.uns[f"scores_matrix_{half}"] = scores_matrix
    adata.uns[f"pvals_adj_matrix_{half}"] = pvals_matrix
    adata.uns[f"pvals_unadj_matrix_{half}"] = pvals_unadj_matrix
    adata.uns["ko_names_matrix"] = np.array(ko_names_pb, dtype=object)
    adata.uns["bin_names_matrix"] = np.array(bin_names_pb, dtype=object)
    # Empty per-pert dicts for schema consistency. Consumers check matrix path first.
    adata.uns[f"names_df_dict_{half}"] = {}
    adata.uns[f"scores_df_dict_{half}"] = {}
    adata.uns[f"pvals_adj_df_dict_{half}"] = {}
    adata.uns[f"pvals_unadj_df_dict_{half}"] = {}
    adata.uns[f"deg_gene_dict_{half}"] = {}
    n_sig = int((pvals_matrix < 0.05).sum())
    log.info("compute_degs_vectorized: stored (%d, %d, %d) matrices "
             "(%d significant calls at FDR<0.05) for half=%s",
             n_bins, n_kos, n_genes, n_sig, half)


# ===================================================================
# Pseudobulk (literature .mean(axis=0), reshaped to (n_bins, n_kos, n_genes))
# ===================================================================


def compute_pseudobulk(
    adata: ad.AnnData,
    *,
    pert_col: str = "condition",
    bin_col: Optional[str] = None,
    min_cells: int = MIN_CELLS_DEGS_DEFAULT,
    chunk_size: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """Compute literature-format pseudobulk reshaped to (n_bins, n_kos, n_genes).

    Returns dict (to be assigned to `adata.uns["pseudobulk"]`):
      * ctrl_bulk:        (n_bins, n_genes) float32   — mean of ALL control cells per bin
      * first_half_bulk:  (n_bins, n_kos, n_genes) float32 — mean of first_half cells per (bin, ko)
      * second_half_bulk: (n_bins, n_kos, n_genes) float32 — mean of second_half cells per (bin, ko)
      * n_cells_first:    (n_bins, n_kos) int32 — first_half cell count per (bin, ko)
      * n_cells_second:   (n_bins, n_kos) int32 — second_half cell count per (bin, ko)
      * ko_names:         (n_kos,) string — KOs with >= min_cells in BOTH halves of EVERY bin
      * bin_names:        (n_bins,) string — unique bins ('all' for single-bin)

    Stored as absolute means; consumers compute deltas as `bulk - ctrl_bulk`.
    The all-cell perturbation mean is the count-weighted average of the two
    halves: `all_bulk = (n1*first + n2*second) / (n1 + n2)`. We therefore store
    the per-half counts so consumers can reconstruct `all_bulk` exactly without a
    third heavy (n_bins, n_kos, n_genes) array (see DatasetStore.all_bulk).

    A KO is included in ko_names ONLY if it has >= min_cells in:
      - first_half of EVERY bin, AND
      - second_half of EVERY bin
    This ensures all entries of first_half_bulk and second_half_bulk are
    populated and not just zero-padded.

    Implementation: a single grouped sparse reduction per half
    (`indicator.T @ X`, see `_group_indicator`/`_grouped_sum`) instead of a
    per-(ko, bin) Python loop — the counts fall out of the same indicator.
    """
    X = adata.X
    n_genes = adata.n_vars
    conds_arr = adata.obs[pert_col].values

    if bin_col is None:
        bin_names_sorted = ["all"]
        n_bins = 1
        bin_assignment = np.zeros(adata.n_obs, dtype=np.int64)
    else:
        bin_names_sorted = sorted(adata.obs[bin_col].unique())
        n_bins = len(bin_names_sorted)
        bin_to_idx = {b: i for i, b in enumerate(bin_names_sorted)}
        bin_assignment = np.array(
            [bin_to_idx[b] for b in adata.obs[bin_col].values], dtype=np.int64
        )

    halves_arr = adata.obs["tech_dup_split"].values
    is_ctrl = np.array([_is_control_label(c) for c in conds_arr])
    is_first_half = (halves_arr == "first_half")
    is_second_half = (halves_arr == "second_half")
    n_cells = adata.n_obs

    # Determine ko_names: non-controls present with >= min_cells in BOTH halves of EVERY bin
    non_ctrl = _non_control_conditions(adata, pert_col)
    ko_candidates: List[str] = []
    for cond in non_ctrl:
        cond_mask = (conds_arr == cond)
        valid_in_all_bins = True
        for bi in range(n_bins):
            in_bin = (bin_assignment == bi)
            n_first = int((cond_mask & in_bin & is_first_half).sum())
            n_second = int((cond_mask & in_bin & is_second_half).sum())
            if n_first < min_cells or n_second < min_cells:
                valid_in_all_bins = False
                break
        if valid_in_all_bins:
            ko_candidates.append(cond)
    ko_names = sorted(ko_candidates)
    n_kos = len(ko_names)
    log.info("compute_pseudobulk: n_bins=%d, n_kos=%d, n_genes=%d (after >=%d-cell filter)",
             n_bins, n_kos, n_genes, min_cells)

    ko_to_idx = {k: i for i, k in enumerate(ko_names)}
    ko_code_cell = np.array(
        [ko_to_idx.get(c, -1) for c in conds_arr], dtype=np.int64
    )

    # --- Control baseline: group ALL control cells by bin (n_bins groups). ---
    ctrl_codes = np.where(is_ctrl, bin_assignment, -1)
    ctrl_ind, ctrl_counts = _group_indicator(ctrl_codes, n_cells, n_bins)
    ctrl_sums = _grouped_sum(ctrl_ind, X, chunk_size)           # (n_bins, n_genes)
    ctrl_bulk = (ctrl_sums / np.maximum(ctrl_counts[:, None], 1.0)).astype(np.float32)

    # --- Perturbation halves: group by flat (bin, ko) code = bin * n_kos + ko. ---
    def _half_bulk(in_half: np.ndarray):
        if n_kos == 0:
            return (np.zeros((n_bins, 0, n_genes), dtype=np.float32),
                    np.zeros((n_bins, 0), dtype=np.int32))
        valid = in_half & (ko_code_cell >= 0)
        codes = np.where(valid, bin_assignment * n_kos + ko_code_cell, -1)
        ind, counts = _group_indicator(codes, n_cells, n_bins * n_kos)
        sums = _grouped_sum(ind, X, chunk_size)               # (n_bins*n_kos, n_genes)
        means = sums / np.maximum(counts[:, None], 1.0)
        means = means.reshape(n_bins, n_kos, n_genes).astype(np.float32)
        counts = counts.reshape(n_bins, n_kos).astype(np.int32)
        return means, counts

    first_half_bulk, n_cells_first = _half_bulk(is_first_half)
    second_half_bulk, n_cells_second = _half_bulk(is_second_half)

    return {
        "ctrl_bulk": ctrl_bulk,
        "first_half_bulk": first_half_bulk,
        "second_half_bulk": second_half_bulk,
        "n_cells_first": n_cells_first,
        "n_cells_second": n_cells_second,
        "ko_names": np.array(ko_names, dtype=object),
        "bin_names": np.array(bin_names_sorted, dtype=object),
    }


# ===================================================================
# Derived deg_arrays (per_pert_weights, deg_mask, deg_directions)
# Reindex first_half DEG dicts to var_names order.
# ===================================================================


def compute_degs_pdex(
    adata: ad.AnnData,
    *,
    half: str,
    pert_col: str = "condition",
    bin_col: Optional[str] = None,
    min_cells: int = MIN_CELLS_DEGS_DEFAULT,
    dataset_name: Optional[str] = None,
    score: str = "log2fc",
    metric: str = "wilcoxon",
    num_workers: Optional[int] = None,
    num_threads: Optional[int] = None,
    gene_chunk_size: int = PDEX_GENE_CHUNK_DEFAULT,
    clip_value: float = PDEX_CLIP_VALUE,
) -> None:
    """Differential expression for one tech-dup half (or all cells), via `pdex`.

    Replaces the hand-rolled Welch/`overestim_var` t-test. Two things change:

    * **Reference is CONTROL, not "rest".** The previous implementation restricted
      to non-control perturbations and then asked scanpy for `reference="rest"`,
      so a gene's score measured how a perturbation differed from *the average
      perturbation* — with control cells excluded from the comparison entirely.
    * **Rank-based Mann-Whitney** (`metric="wilcoxon"`, pdex's default) instead of
      a t-test with the `overestim_var` heuristic.

    Note what this does NOT fix: pdex is still a per-cell test, so batch-driven
    pseudoreplication remains. Measured on norman19 controls split by `gemgroup`
    (no biological difference): 58 false positives for the old t-test, 69 for
    pdex/wilcoxon, out of 8,221 genes. Switching engines changes the reference and
    the test family, not the replicate unit.

    `score` selects what lands in `scores_matrix_{half}` (which drives
    `per_pert_weights` and `deg_directions` in `compute_deg_arrays`):
      * ``"log2fc"``          -> log2(fold_change); sign is the direction, and the
                                 magnitude is what cell-eval itself ranks on.
      * ``"signed_neglog10p"`` -> sign(log2FC) * -log10(p_value); the p-value-derived
                                 analogue of the old t-statistic.

    Writes the same `adata.uns` slots as the previous implementation, so
    `compute_deg_arrays` and every benchmark consumer are unaffected.

    **Execution mode.** `low_memory=True` selects pdex's numba-accelerated chunked
    implementation instead of its mp.Pool/densify path. This is an implementation
    switch with NO statistical content — same reference, same wilcoxon, same
    `tie_correct`, same `expm1(mean(log1p(X)))` aggregation, same pooled BH — and it
    is bit-identical on the call set (Jaccard 1.0000 over 492,240 comparisons; max
    |dp| 7.4e-4, max |dFC| 1.2e-4, no call crossing FDR 0.05). See `pdex_parallelism`
    for why `num_threads` has a hard floor.

    Note on BH: pdex runs `false_discovery_control` over EVERY (perturbation, gene)
    p-value in a call, not per perturbation — so `m = n_kos * n_genes` per (bin, half)
    and one common cutoff applies to the whole bin. Same in both modes.
    """
    from pdex import parallel_differential_expression as _pde

    assert half in ("first_half", "second_half", "all"), f"invalid half: {half!r}"
    assert score in ("log2fc", "signed_neglog10p"), f"invalid score: {score!r}"
    if "pseudobulk" not in adata.uns:
        raise RuntimeError("compute_degs_pdex requires compute_pseudobulk to have run first")
    if dataset_name is None:
        dataset_name = adata.uns.get("dataset_name", "ds")

    # Duplicate gene names would make two pdex rows map to ONE column below, the
    # second silently overwriting the first. Zero duplicates across all nine datasets
    # today; this is a guard, not a repair.
    dup = pd.Index(adata.var_names).duplicated()
    if dup.any():
        raise ValueError(
            f"compute_degs_pdex[{dataset_name}]: {int(dup.sum())} duplicate gene names "
            f"(e.g. {list(pd.Index(adata.var_names)[dup][:5])}). pdex results are keyed "
            f"by gene NAME, so duplicates would be silently collapsed.")

    pb = adata.uns["pseudobulk"]
    ko_names = [str(k) for k in pb["ko_names"]]
    bin_names = [str(b) for b in pb["bin_names"]]
    n_kos, n_bins, n_genes = len(ko_names), len(bin_names), adata.n_vars
    ko_ix = {k: i for i, k in enumerate(ko_names)}
    gene_ix = {g: i for i, g in enumerate(map(str, adata.var_names))}

    scores = np.zeros((n_bins, n_kos, n_genes), dtype=np.float32)
    pvals = np.ones((n_bins, n_kos, n_genes), dtype=np.float32)
    padj = np.ones((n_bins, n_kos, n_genes), dtype=np.float32)
    # True where pdex SUBSTITUTED the fold change instead of measuring it. Detected
    # by equality with the clip constant we pass in, so it follows automatically if
    # that constant ever changes. An entry never written stays False, which is right:
    # its score is 0 and it earns no weight either way.
    substituted = np.zeros((n_bins, n_kos, n_genes), dtype=bool)

    workers, threads = pdex_parallelism(num_workers, num_threads)
    log.info("compute_degs_pdex: half=%s bins=%d kos=%d genes=%d metric=%s score=%s",
             half, n_bins, n_kos, n_genes, metric, score)
    log.info("  pdex: low_memory=True workers=%d numba_threads=%d gene_chunk=%d "
             "(allocated cpus=%d)", workers, threads, gene_chunk_size, allocated_cpus())

    # half="all" scores every cell of the perturbation, which is what the DEG weights
    # are built from. The per-half calls stay: the half-vs-half ceiling run and the
    # alpha-blend control predictor both read them.
    half_mask = (np.ones(adata.n_obs, dtype=bool) if half == "all"
                 else (adata.obs["tech_dup_split"].astype(str).values == half))
    for bi, bn in enumerate(bin_names):
        sel = half_mask.copy()
        if bin_col is not None:
            sel &= (adata.obs[bin_col].astype(str).values == bn)
        if not sel.any():
            log.warning("  bin '%s': no %s cells — skipped", bn, half)
            continue
        sub = adata[sel].copy()
        cond = sub.obs[pert_col].astype(str)
        counts = cond.value_counts()

        ctrl_labels = [c for c in counts.index if _is_control_label(c)]
        if not ctrl_labels:
            log.warning("  bin '%s': no control cells — DEGs skipped", bn)
            continue
        # Only the most abundant control label becomes the reference; any others
        # would sit in `sub` contributing cells to no group at all. One label across
        # all nine datasets today, so this is a guard against a future source.
        if len(ctrl_labels) > 1:
            raise ValueError(
                f"compute_degs_pdex[{dataset_name}] bin '{bn}': {len(ctrl_labels)} control "
                f"labels {ctrl_labels}. Only one can be the reference; merge them in "
                f"apply_label_cleanup rather than letting the rest go untested.")
        ctrl = ctrl_labels[0]
        targets = [c for c in counts.index
                   if (not _is_control_label(c)) and counts[c] >= min_cells and c in ko_ix]
        if not targets:
            log.warning("  bin '%s': no perturbation clears >=%d cells", bn, min_cells)
            continue

        sub = sub[cond.isin(targets + ctrl_labels).values].copy()
        sub.obs[pert_col] = sub.obs[pert_col].astype(str)
        log.info("  bin '%s': %d cells, %d perturbations vs '%s' (%d cells)",
                 bn, sub.n_obs, len(targets), ctrl, int(counts[ctrl]))

        df = _pde(sub, groupby_key=pert_col, reference=ctrl, groups=list(targets),
                  metric=metric, is_log1p=True, exp_post_agg=True,
                  low_memory=True, num_workers=workers, num_threads=threads,
                  gene_chunk_size=gene_chunk_size, clip_value=clip_value,
                  show_progress=False)
        df = df[df["target"].isin(ko_ix)]
        rows = df["target"].map(ko_ix).to_numpy()
        cols = df["feature"].astype(str).map(gene_ix)
        ok = cols.notna().to_numpy()
        rows, cols = rows[ok], cols[ok].to_numpy(dtype=int)

        fc = pd.to_numeric(df.loc[ok, "fold_change"], errors="coerce").to_numpy(dtype=np.float64)
        pv = pd.to_numeric(df.loc[ok, "p_value"], errors="coerce").to_numpy(dtype=np.float64)
        fd = pd.to_numeric(df.loc[ok, "fdr"], errors="coerce").to_numpy(dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            l2 = np.log2(np.where(fc > 0, fc, np.nan))
        l2 = np.nan_to_num(l2, nan=0.0, posinf=0.0, neginf=0.0)
        if score == "log2fc":
            sc_vals = l2
        else:
            pv_safe = np.clip(np.nan_to_num(pv, nan=1.0), 1e-300, 1.0)
            sc_vals = np.sign(l2) * -np.log10(pv_safe)

        scores[bi, rows, cols] = sc_vals.astype(np.float32)
        pvals[bi, rows, cols] = np.nan_to_num(pv, nan=1.0).astype(np.float32)
        padj[bi, rows, cols] = np.nan_to_num(fd, nan=1.0).astype(np.float32)
        substituted[bi, rows, cols] = (np.isclose(fc, clip_value, rtol=1e-9)
                                       | np.isclose(fc, 1.0 / clip_value, rtol=1e-9))
        del sub, df

    adata.uns[f"scores_matrix_{half}"] = scores
    adata.uns[f"pvals_unadj_matrix_{half}"] = pvals
    adata.uns[f"pvals_adj_matrix_{half}"] = padj
    adata.uns[f"deg_fc_substituted_{half}"] = substituted
    adata.uns["ko_names_matrix"] = np.array(ko_names, dtype=object)
    adata.uns["bin_names_matrix"] = np.array(bin_names, dtype=object)
    # Legacy per-pert dict slots, kept empty (the matrix path supersedes them).
    for key in ("names", "scores", "pvals", "pvals_adj", "deg_gene"):
        adata.uns[f"{key}_df_dict_{half}"] = {}
    sig = padj < 0.05
    n_sig, n_sub = int(sig.sum()), int(substituted.sum())
    log.info("compute_degs_pdex: half=%s done — %d (bin,ko,gene) calls at FDR<0.05; "
             "%d entries (%.1f%%) had a SUBSTITUTED fold change, %d of them significant "
             "(these are excluded from the gene weights)",
             half, n_sig, n_sub, 100.0 * n_sub / max(substituted.size, 1),
             int((substituted & sig).sum()))


def compute_deg_arrays(
    adata: ad.AnnData,
    *,
    pert_col: str = "condition",
    bin_col: Optional[str] = None,
    dataset_name: Optional[str] = None,
    source_half: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    """Build the three var_names-aligned tensors from the DEG matrices.

    `source_half` picks which DEG call to read; by default the all-cell call when
    one exists, falling back to `first_half`. The all-cell call is preferred because
    these tensors weight and mask the scoring, and deriving them from half the cells
    while scoring the full set would make the weighting noisier than the data.

    Reads (must be already computed for half='first_half'):
      adata.uns["names_df_dict_first_half"]
      adata.uns["scores_df_dict_first_half"]
      adata.uns["pvals_adj_df_dict_first_half"]

    Reads ko_names and bin_names from adata.uns["pseudobulk"] so the output
    tensors are aligned with the pseudobulk arrays.

    Returns dict (to be assigned to `adata.uns["deg_arrays"]`):
      * per_pert_weights: (n_bins, n_kos, n_genes) float32
            Squared min-max normalized abs(scores), reindexed to var_names.
      * deg_mask:         (n_bins, n_kos, n_genes) bool
            pvals_adj < 0.05, reindexed to var_names.
      * deg_directions:   (n_bins, n_kos, n_genes) int8
            sign(scores), reindexed to var_names. 0 where missing.
      * ko_names:         (n_kos,)  matches pseudobulk['ko_names']
      * bin_names:        (n_bins,) matches pseudobulk['bin_names']
    """
    assert "pseudobulk" in adata.uns, "Run compute_pseudobulk first"

    if dataset_name is None:
        dataset_name = adata.uns.get("dataset_name", "ds")

    pb = adata.uns["pseudobulk"]
    ko_names = [str(k) for k in pb["ko_names"]]
    bin_names = [str(b) for b in pb["bin_names"]]
    n_kos = len(ko_names)
    n_bins = len(bin_names)
    n_genes = adata.n_vars
    var_names = list(adata.var_names)
    gene_to_idx = {g: i for i, g in enumerate(var_names)}

    # Matrix path: if compute_degs_vectorized was used, scores/pvals are
    # already var_names-aligned and ko/bin-aligned with the pseudobulk —
    # no reindexing needed.
    source = source_half or next(
        (h for h in ("all", "first_half") if f"scores_matrix_{h}" in adata.uns), None)
    if source is not None:
        if f"scores_matrix_{source}" not in adata.uns:
            raise KeyError(f"compute_deg_arrays: no scores_matrix_{source} in uns")
        log.info("compute_deg_arrays: reading score/pval matrices for half=%s", source)
        scores = np.asarray(adata.uns[f"scores_matrix_{source}"], dtype=np.float32)
        pvals = np.asarray(adata.uns[f"pvals_adj_matrix_{source}"], dtype=np.float32)
        assert scores.shape == (n_bins, n_kos, n_genes), \
            f"scores_matrix shape {scores.shape} != ({n_bins}, {n_kos}, {n_genes})"
        substituted = np.asarray(
            adata.uns.get(f"deg_fc_substituted_{source}",
                          np.zeros(scores.shape, dtype=bool)), dtype=bool)

        deg_mask = pvals < 0.05
        deg_directions = np.sign(scores).astype(np.int8)
        deg_directions[~deg_mask] = 0

        # A gene earns weight only if the test CALLED it and its effect size was
        # MEASURED. Both conditions are load-bearing:
        #
        #   significance — the p-value and the effect size are two independent
        #     readings, and weighting on effect size alone let non-significant genes
        #     carry 1.7x the weight of significant ones (adamson16: 0.0947 vs 0.0551).
        #   measured — pdex's placeholder sits at |log2FC| 4.32, eight times the
        #     median significant effect, so a significance gate ALONE still left
        #     47-49% of the weight mass on it. The survivors are genes switched on
        #     from a true zero baseline: significant (a clean rank separation) but
        #     with a perturbed mean of 0.0018 against 0.3312 for real hits.
        #
        # Linear in |log2FC|, not squared. The squaring existed to counteract the
        # flattening the placeholders caused; with them gone it over-concentrates,
        # cutting the genes effectively deciding a score from ~167 to ~43.
        usable = deg_mask & ~substituted
        a = np.where(usable, np.abs(scores), 0.0)
        rowmax = a.max(axis=-1, keepdims=True)
        per_pert_weights = np.where(
            rowmax > 0, a / np.where(rowmax > 0, rowmax, 1.0), 0.0).astype(np.float32)

        # A row with significant genes but no weight would silently drop out of every
        # weighted metric. Measured minimum across all nine datasets is 47 usable
        # genes per row, so this should never fire.
        starved = (deg_mask.any(axis=-1)) & (~usable.any(axis=-1))
        if starved.any():
            raise ValueError(
                f"compute_deg_arrays[{dataset_name}]: {int(starved.sum())} (bin, ko) rows "
                f"have significant genes but every one has a substituted fold change, so "
                f"they would carry no weight at all in the weighted metrics.")

        log.info("compute_deg_arrays: %d significant calls; %d substituted entries "
                 "excluded (%d of them significant); weights on %d (bin,ko,gene) entries",
                 int(deg_mask.sum()), int(substituted.sum()),
                 int((substituted & deg_mask).sum()), int(usable.sum()))
        return {
            "per_pert_weights": per_pert_weights,
            "deg_mask": deg_mask,
            "deg_directions": deg_directions,
            "ko_names": np.array(ko_names, dtype=object),
            "bin_names": np.array(bin_names, dtype=object),
        }

    # Dict path: existing behavior — reindex per-pert dicts to var_names order.
    assert "names_df_dict_first_half" in adata.uns, \
        "Run a compute_degs_* function for half='first_half' first"
    names_dict = adata.uns["names_df_dict_first_half"]
    scores_dict = adata.uns["scores_df_dict_first_half"]
    pvals_adj_dict = adata.uns["pvals_adj_df_dict_first_half"]

    per_pert_weights = np.zeros((n_bins, n_kos, n_genes), dtype=np.float32)
    deg_mask = np.zeros((n_bins, n_kos, n_genes), dtype=bool)
    deg_directions = np.zeros((n_bins, n_kos, n_genes), dtype=np.int8)

    missing_keys = 0
    populated = 0
    for bi, bn in enumerate(bin_names):
        for ki, ko in enumerate(ko_names):
            # Construct the dict key. Single-bin: "{dataset}_{ko}". Multi-bin: "{dataset}_{bin}_{ko}".
            if bin_col is None or bn == "all":
                key = f"{dataset_name}_{ko}"
            else:
                key = f"{dataset_name}_{bn}_{ko}"
            if key not in names_dict:
                missing_keys += 1
                continue

            gene_names = names_dict[key]
            scores = np.asarray(scores_dict[key], dtype=np.float32)
            pvals = np.asarray(pvals_adj_dict[key], dtype=np.float32)

            # Build inverse permutation: gene_idx_in_var = gene_to_idx[gene_name]
            # Some genes in dict may not be in var_names (e.g. dropped HVGs in subset filter).
            gene_idx = np.array([gene_to_idx.get(g, -1) for g in gene_names])
            valid = (gene_idx >= 0)
            gene_idx_valid = gene_idx[valid]
            scores_valid = scores[valid]
            pvals_valid = pvals[valid]

            # per_pert_weights: squared min-max normalized abs(scores)
            abs_scores = np.abs(scores_valid)
            if abs_scores.size > 0:
                lo, hi = float(abs_scores.min()), float(abs_scores.max())
                rng = (hi - lo) if hi > lo else 1.0
                weights = ((abs_scores - lo) / rng) ** 2
            else:
                weights = abs_scores

            per_pert_weights[bi, ki, gene_idx_valid] = weights
            deg_mask[bi, ki, gene_idx_valid] = pvals_valid < 0.05
            deg_directions[bi, ki, gene_idx_valid] = np.sign(scores_valid).astype(np.int8)
            populated += 1

    log.info("compute_deg_arrays: populated %d (bin, ko) cells, %d missing keys",
             populated, missing_keys)

    return {
        "per_pert_weights": per_pert_weights,
        "deg_mask": deg_mask,
        "deg_directions": deg_directions,
        "ko_names": np.array(ko_names, dtype=object),
        "bin_names": np.array(bin_names, dtype=object),
    }
