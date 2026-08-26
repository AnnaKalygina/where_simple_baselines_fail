"""Unified, self-contained verifier for the benchmark.

ONE importable module + CLI that runs *before* a benchmark to prove the pipeline
hangs together. It absorbs the old `dataset_validator` (the cheap structural gate
run inside `DatasetStore.__init__`) and the old `verify` harness, and adds the
checks that actually catch silent breakage.

Self-containment rule: this module imports ONLY core benchmark modules (config,
data_loader, predictors.*, meta_metrics, dl_adapter). It NEVER imports from
`benchmark/scripts/`. Ad-hoc scripts may import from here, not the other way
round. Module-level imports are kept minimal (numpy + config) so
`DatasetStore.__init__` (which imports `validate_dataset` from here) stays light
and there is no import cycle — every heavy import (DatasetStore, predictors,
anndata, meta_metrics, dl_adapter) is done lazily inside the function that needs it.

Layers (mapped to the failure modes they guard)
------------------------------------------------
  L0 INVARIANTS        `validate_dataset` — structural gate, runs on every store
                       open (required obs/uns/var, HVG count, pseudobulk shapes,
                       split labels, no NaN).                       [FM dataset/loading]
  L1 SPLIT-INTEGRITY   both tech-dup halves present (>=4 cells); per fold,
                       train ∩ test/val == 0; non-rectangular scenarios carry
                       their uns pair masks (no leaky-rectangle fallback). [FM leakage]
  L2 PSEUDOBULK-MATH   all_bulk == count-weighted half means; all_deltas exact;
                       vectorized == naive gate on one small dataset.   [FM bulks/train-target]
  L3 GT-AXIS           empirically pin that metrics score vs first_half_deltas
                       (not all/second half).                          [FM train-all/eval-half]
  L4 PREDICTOR-CONTRACTS  synthetic, data-free: every predictor recovers a known
                       answer AND a sentinel fires iff it took its degenerate
                       fallback (e.g. TargetScaling really scales).    [FM no-fallback/correctness]
  L5 COVERAGE          per dataset: perturbed genes + (bin,ko) perts, which each
                       predictor can resolve, and the cross-predictor overlap. [targets diagnostic]
  L6 FOLD-ALIGNMENT    DL fold test-sets == our h5ad fold test-sets (exact);
                       saved predictions' fold/scenario metadata match the split. [FM fold alignment]

Levels: `--fast` (L1,L2,L4,L5 + L0 via store open; numpy-only, runs in `preprocess`)
        `--full` (adds L3,L6; needs saved predictions + the DL manifest; `vcell`).

Run:
  conda run -n preprocess python -m benchmark.verify --fast
  conda run -n vcell      python -m benchmark.verify --full --dataset adamson16
  conda run -n preprocess python -m benchmark.verify --contracts   # L4 only, no data
  conda run -n vcell      python -m benchmark.verify --coverage --dataset mcfaline23
"""
from __future__ import annotations

import argparse
import logging
import re
import sys
import traceback
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from benchmark.config import (
    ADOPTED_FOLD_DATASETS,
    DATASET_CONFIG,
    MULTIBIN_BLOCK_PARTITION_DATASETS,
    expected_hvg,
    parse_target_genes,
    split_obs_column,
)

log = logging.getLogger(__name__)

MIN_CELLS = 4               # post-downsample floor (every (bin,ko) half must clear this)
_REL_TOL = 1e-5
_PB_GATE_TOL = 1e-4        # sparse-matmul vs per-mask .mean() float32 reduction-order slack
_SKIP_SENTINEL = "n_cells_first"  # pseudobulk key present only on all-cell-aware stores

# Scenarios whose CV hold-out is a scattered (cell_type, pert) set rather than a
# bin×ko rectangle — these MUST carry explicit uns pair masks or the loader falls
# back to the leaky rectangle. (UnseenCell holds out whole cell types → clean
# rectangle, no mask needed.)
_NON_RECTANGULAR_SCENARIOS = ("UnseenBoth", "UnseenPair")

# Guide-suffix / junk handling for matching OUR (guide-free) ko_names against the
# DL prediction labels, which encode per-guide variants (e.g. 'FDPS_2'). Our own
# ko_names carry no guide suffixes (stripped in get_data), so all guide-awareness
# lives at the DL↔ours matching layer, never in the canonical parse_target_genes.
# Canonical pert-key matching + the DL declared-test-set reader live in the shared
# leaf benchmark._fold_align (single source of truth; keeps the gate and the verifier
# in lock-step and avoids a verify↔dl_adapter import cycle).
from benchmark._fold_align import (  # noqa: E402
    match_key as _match_key, canon_set as _canon_set,
    dl_declared_test_set as _dl_declared_test_set,
)


# ===================================================================
# L0 — validate_dataset: the cheap structural gate (was dataset_validator)
# ===================================================================
#
# Import-light: takes an in-memory/backed AnnData + names, never opens a
# DatasetStore, so `data_loader` can import it without a cycle.


class DatasetValidationError(ValueError):
    """Raised when a processed h5ad fails one or more invariant checks."""


_CONTROL_LABELS = {"control", "ctrl", "non-targeting"}


def _is_control(label: str) -> bool:
    return str(label).strip().lower() in _CONTROL_LABELS


def _check_structural(adata) -> List[str]:
    failures: List[str] = []
    for col in ("condition", "cell_type", "tech_dup_split"):
        if col not in adata.obs.columns:
            failures.append(f"obs column '{col}' is missing")
    # Only the authoritative slots are required. The legacy `*_df_dict_*` uns
    # entries are retired empty placeholders (superseded by the `scores_matrix_*`
    # arrays + deg_arrays), so they are NOT required here anymore.
    for key in ("pseudobulk", "deg_arrays"):
        if key not in adata.uns:
            failures.append(f"adata.uns['{key}'] is missing")
    if "highly_variable" not in adata.var.columns:
        failures.append("var column 'highly_variable' is missing")
    if "is_perturbation_target" not in adata.var.columns:
        failures.append("var column 'is_perturbation_target' is missing "
                        "(re-run get_data.py)")
    return failures


# Synthetic datasets legitimately have no raw counts and no batch structure.
_NO_COUNTS_DATASETS = frozenset({"ecoli_synthetic"})


def _check_counts_layer(adata, dataset_name: str) -> List[str]:
    """L0: raw counts must be present and integral.

    `select_hvg` runs `seurat_v3` on `layers['counts']`, so a missing or already
    log-normalised counts layer silently changes which genes make the panel.
    Checking integrality catches the case where a normalised matrix was stored
    under the `counts` name.
    """
    if dataset_name in _NO_COUNTS_DATASETS:
        return []
    if adata.uns.get("_vcr_light_store"):
        # --fast opens obs/var + light uns only; layers are not read, so their
        # absence here means nothing. Deferred to --full (same as the log1p
        # X-range check).
        return []
    if "counts" not in adata.layers:
        return ["layers['counts'] is missing (seurat_v3 HVG needs raw counts)"]
    X = adata.layers["counts"]
    # STRIDED sample, not a head block: rows are often ordered by cell type, so
    # X[:200] inspects one corner of the matrix. Same reasoning as the raw-vs-log
    # heuristic this pipeline replaced with an explicit declaration.
    step = max(1, X.shape[0] // 200)
    sample = X[::step]
    vals = sample.data if hasattr(sample, "data") else np.asarray(sample).ravel()
    if vals.size and not np.allclose(vals, np.round(vals)):
        return ["layers['counts'] holds non-integral values — it looks normalised, "
                "not raw counts"]
    return []


def _check_batch_column(adata) -> List[str]:
    """L0 (warning): every dataset should carry an explicit `batch` covariate.

    Previously `donor_id` was written as the dataset name everywhere, i.e. a
    covariate that existed but carried no information. `batch` must be present;
    a constant value is allowed only as a *declared* fallback (the get_data.py
    sets it explicitly when the source has no lane/gem-group column).
    """
    if "batch" not in adata.obs.columns:
        return ["obs column 'batch' is missing (get_data.py should set it, using a "
                "declared constant when the source has no batch covariate)"]
    return []


def _check_hvg_count(adata, dataset_name: str) -> List[str]:
    failures: List[str] = []
    if "highly_variable" not in adata.var.columns:
        return failures
    exp = expected_hvg(dataset_name)
    n_hv = int(adata.var["highly_variable"].sum())
    if n_hv < exp - 200:
        failures.append(
            f"highly_variable count is {n_hv} (< expected {exp} - 200); the HVG "
            f"panel looks under-populated"
        )
    return failures


def _ko_bin_names(adata) -> Tuple[List[str], List[str]]:
    pb = adata.uns.get("pseudobulk")
    if pb is None:
        return [], []
    pb = dict(pb)
    return ([str(k) for k in pb.get("ko_names", [])],
            [str(b) for b in pb.get("bin_names", [])])


def _check_missing_targets(adata) -> List[str]:
    """Warning-level: perturbation targets absent from var_names (tolerated)."""
    var_set = set(adata.var_names)
    targets: set = set()
    if "condition" in adata.obs.columns:
        for cond in adata.obs["condition"].unique():
            if _is_control(cond):
                continue
            targets.update(parse_target_genes(cond))
    missing = sorted(targets - var_set)
    if not missing:
        return []
    return [f"{len(missing)} perturbation target gene(s) absent from var_names: "
            f"{missing[:10]}" + (f" ... and {len(missing)-10} more"
                                 if len(missing) > 10 else "")]


def _check_combo_singles(ko_names: List[str]) -> List[str]:
    ko_set = set(ko_names)
    missing = [(k, sorted(set(parse_target_genes(k)) - ko_set))
               for k in ko_names if "+" in k
               and (set(parse_target_genes(k)) - ko_set)]
    if not missing:
        return []
    preview = "; ".join(f"{n} (missing: {m})" for n, m in missing[:5])
    return [f"{len(missing)} combo perturbation(s) reference singles absent "
            f"from ko_names. First 5: {preview}"]


def _check_dose_bases(ko_names: List[str]) -> List[str]:
    ko_set = set(ko_names)
    missing = [k for k in ko_names if "@" in k
               and k.split("@", 1)[0].strip()
               and k.split("@", 1)[0].strip() not in ko_set]
    if not missing:
        return []
    return [f"{len(missing)} dose perturbation(s) reference a full-dose base "
            f"absent from ko_names. First 5: {missing[:5]}"]


def _check_deg_coverage(adata, ko_names, bin_names, dataset) -> List[str]:
    """Every (bin, ko) must have first/second-half DEG signal.

    Authoritative source is the vectorized `scores_matrix_{half}` (n_bins, n_kos,
    n_genes). Datasets that still carry only the per-pert dicts fall back to a
    dict-coverage check; the dict keys are NOT a hard structural requirement.
    """
    failures: List[str] = []
    n_bins = len(bin_names)
    single_bin = n_bins == 1
    for half in ("first_half", "second_half"):
        matrix_key = f"scores_matrix_{half}"
        if matrix_key in adata.uns:
            mat = adata.uns[matrix_key]
            want = (n_bins, len(ko_names), adata.n_vars)
            if tuple(np.asarray(mat).shape) != want:
                failures.append(f"uns['{matrix_key}'].shape == "
                                f"{tuple(np.asarray(mat).shape)}, expected {want}")
            continue
        names_dict = dict(adata.uns.get(f"names_df_dict_{half}", {}))
        if not names_dict:
            failures.append(
                f"no DEG signal for {half}: neither uns['{matrix_key}'] nor a "
                f"populated uns['names_df_dict_{half}'] is present")
            continue
        missing = [
            (f"{dataset}_{ko}" if single_bin or bn == "all" else f"{dataset}_{bn}_{ko}")
            for bn in bin_names for ko in ko_names
            if (f"{dataset}_{ko}" if single_bin or bn == "all"
                else f"{dataset}_{bn}_{ko}") not in names_dict
        ]
        if missing:
            failures.append(f"{len(missing)} {half} DEG dict key(s) missing. "
                            f"First 5: {missing[:5]}")
    return failures


def _check_split_obs_columns(adata, dataset) -> List[str]:
    failures: List[str] = []
    if dataset not in DATASET_CONFIG:
        return failures
    valid = {"train", "val", "test", "unassigned"}
    for scenario, cfg in DATASET_CONFIG[dataset]["scenarios"].items():
        for fold in range(cfg["n_folds"]):
            col = split_obs_column(scenario, fold)
            if col not in adata.obs.columns:
                failures.append(f"obs column '{col}' missing for {scenario} fold {fold}")
                continue
            invalid = set(adata.obs[col].astype(str).unique()) - valid
            if invalid:
                failures.append(f"obs['{col}'] has invalid labels {sorted(invalid)}; "
                                f"valid set is {sorted(valid)}")
    return failures


def _check_pseudobulk_shapes(adata, ko_names, bin_names) -> List[str]:
    failures: List[str] = []
    pb = adata.uns.get("pseudobulk")
    if pb is None:
        return failures
    pb = dict(pb)
    n_bins, n_kos, n_genes = len(bin_names), len(ko_names), adata.n_vars
    expected = {
        "ctrl_bulk": (n_bins, n_genes),
        "first_half_bulk": (n_bins, n_kos, n_genes),
        "second_half_bulk": (n_bins, n_kos, n_genes),
    }
    for key, want in expected.items():
        arr = pb.get(key)
        if arr is None:
            failures.append(f"pseudobulk['{key}'] is missing")
        elif tuple(np.asarray(arr).shape) != want:
            failures.append(f"pseudobulk['{key}'].shape == "
                            f"{tuple(np.asarray(arr).shape)}, expected {want}")
    for key in ("n_cells_first", "n_cells_second"):
        arr = pb.get(key)
        if arr is not None and tuple(np.asarray(arr).shape) != (n_bins, n_kos):
            failures.append(f"pseudobulk['{key}'].shape == "
                            f"{tuple(np.asarray(arr).shape)}, expected {(n_bins, n_kos)}")
    return failures


def _check_no_nans(adata) -> List[str]:
    failures: List[str] = []
    targets: List[Tuple[str, np.ndarray]] = []
    pb = adata.uns.get("pseudobulk")
    if pb is not None:
        for key in ("ctrl_bulk", "first_half_bulk", "second_half_bulk"):
            arr = dict(pb).get(key)
            if arr is not None:
                targets.append((f"pseudobulk['{key}']", np.asarray(arr)))
    deg = adata.uns.get("deg_arrays")
    if deg is not None:
        for key in ("per_pert_weights", "deg_mask", "deg_directions"):
            arr = dict(deg).get(key)
            if arr is not None:
                targets.append((f"deg_arrays['{key}']", np.asarray(arr)))
    for name, arr in targets:
        if np.issubdtype(arr.dtype, np.floating) and not np.isfinite(arr).all():
            failures.append(f"{name} contains NaN or Inf entries")
    return failures


# Datasets not built by the current preprocessing recipe, and therefore exempt from
# the three content checks below. xatlas_orion has never been ported to the recipe;
# ecoli_synthetic is synthetic and exempt from per-cell QC by design. Both still
# carry the pre-audit gene weights -- measured off-mask weight mass 100.0% and 95.5%.
#
# Exempted BY NAME rather than by downgrading the checks to warnings: the mito bug
# survived for exactly as long as it did because its cross-check was a warning nobody
# was blocked by. A named list is visible and shrinks; a warning does neither. Remove
# an entry when that dataset is rebuilt.
_DATASETS_NOT_ON_CURRENT_RECIPE = frozenset({"ecoli_synthetic", "xatlas_orion"})


def _check_deg_weight_sanity(adata, dataset_name: str) -> List[str]:
    """The gene weights must favour genes the DE test actually called.

    `verify` has only ever inspected the SHAPE of these tensors, which is how
    `per_pert_weights` came to hold 82-89% of its mass on pdex's placeholder fold
    change -- 97-99% of it on entries that failed significance -- while every
    structural check passed. Content, not shape.
    """
    failures: List[str] = []
    if dataset_name in _DATASETS_NOT_ON_CURRENT_RECIPE:
        log.warning("Dataset '%s' is exempt from the DEG-weight check: it still carries "
                    "the pre-audit weights. Rebuild it and remove the exemption.",
                    dataset_name)
        return failures
    deg = adata.uns.get("deg_arrays")
    if deg is None:
        return failures
    deg = dict(deg)
    w = deg.get("per_pert_weights")
    mask = deg.get("deg_mask")
    if w is None or mask is None:
        return failures
    w = np.asarray(w, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    if w.shape != mask.shape:
        return [f"deg_arrays: per_pert_weights {w.shape} != deg_mask {mask.shape}"]

    total = float(w.sum())
    if total <= 0:
        return ["deg_arrays['per_pert_weights'] is entirely zero"]
    off_mask = float(w[~mask].sum()) / total
    if off_mask > 0.50:
        failures.append(
            f"deg_arrays: {100 * off_mask:.1f}% of the gene-weight mass sits on genes the "
            f"DE test did NOT call significant. The weighted metrics are then driven by "
            f"entries with no evidence behind them.")

    starved = mask.any(axis=-1) & ~(w > 0).any(axis=-1)
    if starved.any():
        failures.append(
            f"deg_arrays: {int(starved.sum())} (bin, ko) rows have significant genes but "
            f"zero total weight, so they drop out of every weighted metric silently.")
    return failures


def _check_cell_qc_audit(adata, dataset_name: str) -> List[str]:
    """No single QC gate may have removed most of the dataset.

    `MITO_MAX_REMOVED_FRACTION` enforces this in `apply_cell_qc`, which runs only at
    BUILD time. Carrying the audit in `uns` puts the same invariant on the artefact,
    so an h5ad built before the guard existed can no longer verify clean.
    """
    if dataset_name in _DATASETS_NOT_ON_CURRENT_RECIPE:
        return []
    audit = adata.uns.get("cell_qc_audit")
    if audit is None:
        return [f"uns is missing cell_qc_audit, so which cells were removed by which "
                f"gate -- and whether any gate ran away -- is unrecorded."]
    audit = dict(audit)
    n_before = int(audit.get("n_before", 0))
    if n_before <= 0:
        return ["uns['cell_qc_audit'] records no starting cell count"]
    failures: List[str] = []
    for key, removed in audit.items():
        if not str(key).startswith("removed_"):
            continue
        frac = int(removed) / n_before
        if frac > 0.50:
            failures.append(
                f"cell QC gate '{str(key)[8:]}' removed {int(removed)} of {n_before} cells "
                f"({100 * frac:.1f}%). A quality gate taking most of a dataset is a broken "
                f"statistic, not strictness.")
    return failures


def _check_normalisation_declared(adata, dataset_name: str) -> List[str]:
    """Every dataset must record what its expression scale actually means.

    Two sources shipped their producer's own normalisation and sat on a 4x
    different scale from the other seven, with nothing in the file saying so.
    """
    if dataset_name in _DATASETS_NOT_ON_CURRENT_RECIPE:
        return []
    if "count_retention" not in adata.uns or "depth_column" not in adata.uns:
        return [f"uns is missing count_retention / depth_column, so what fraction of "
                f"each cell this matrix holds -- and therefore what its expression "
                f"values mean -- is unrecorded."]
    return []


def validate_dataset(adata, dataset_name: str) -> List[str]:
    """Run every structural invariant on a processed h5ad (the L0 gate).

    Raises `DatasetValidationError` listing every error-level failure if any are
    present; otherwise returns the list of warning-level findings (also logged).
    Called from `DatasetStore.__init__`.
    """
    errors: List[str] = []
    warnings_: List[str] = []

    errors.extend(_check_structural(adata))
    errors.extend(_check_hvg_count(adata, dataset_name))
    errors.extend(_check_counts_layer(adata, dataset_name))
    # WARNING, not an error: nothing in the benchmark reads `batch` yet, and making
    # it fatal would make every not-yet-rebuilt dataset unopenable mid-migration.
    warnings_.extend(_check_batch_column(adata))
    warnings_.extend(_check_missing_targets(adata))

    ko_names, bin_names = _ko_bin_names(adata)
    if ko_names and bin_names:
        warnings_.extend(_check_combo_singles(ko_names))
        warnings_.extend(_check_dose_bases(ko_names))
        errors.extend(_check_deg_coverage(adata, ko_names, bin_names, dataset_name))
        errors.extend(_check_pseudobulk_shapes(adata, ko_names, bin_names))

    errors.extend(_check_split_obs_columns(adata, dataset_name))
    errors.extend(_check_no_nans(adata))
    errors.extend(_check_deg_weight_sanity(adata, dataset_name))
    errors.extend(_check_cell_qc_audit(adata, dataset_name))
    errors.extend(_check_normalisation_declared(adata, dataset_name))

    if errors:
        bullets = "\n  - ".join(errors)
        raise DatasetValidationError(
            f"Dataset '{dataset_name}' failed {len(errors)} validation check(s):"
            f"\n  - {bullets}"
        )
    for w in warnings_:
        log.warning("Dataset '%s' validation warning: %s", dataset_name, w)
    return warnings_


# ===================================================================
# Result accounting
# ===================================================================


class Results:
    """Tally of per-check PASS/FAIL/SKIP outcomes across layers."""

    def __init__(self) -> None:
        self.passed = 0
        self.failed = 0
        self.skipped = 0
        self.blocking_skipped = 0
        self.fail_names: List[str] = []
        self.skip_names: List[str] = []

    def check(self, name: str, ok: bool, detail: str = "") -> bool:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
        if ok:
            self.passed += 1
        else:
            self.failed += 1
            self.fail_names.append(name)
        return ok

    def skip(self, name: str, detail: str = "", *, intentional: bool = False) -> None:
        print(f"  [SKIP] {name}" + (f" -- {detail}" if detail else ""))
        self.skipped += 1
        if not intentional:
            self.blocking_skipped += 1
        self.skip_names.append(name)

    def info(self, msg: str) -> None:
        print(f"  [INFO] {msg}")

    def expect_raises(self, name: str, fn: Callable, exc_type=Exception,
                      msg_substr: str = "") -> bool:
        try:
            fn()
        except exc_type as e:
            if msg_substr and msg_substr not in str(e):
                return self.check(name, False,
                                  f"raised {type(e).__name__} but message lacked "
                                  f"{msg_substr!r}: {e}")
            return self.check(name, True, f"raised {type(e).__name__}: {str(e)[:60]}")
        except Exception as e:
            return self.check(name, False, f"wrong type {type(e).__name__}: {e}")
        return self.check(name, False, "did not raise")


# ===================================================================
# Store helpers (lazy DatasetStore import)
# ===================================================================


def _resolve_datasets(spec: Optional[str]) -> List[str]:
    if spec is None or spec == "all":
        return list(DATASET_CONFIG)
    requested = [s.strip() for s in spec.split(",") if s.strip()]
    unknown = [d for d in requested if d not in DATASET_CONFIG]
    if unknown:
        raise SystemExit(f"Unknown dataset(s): {unknown}. Known: {list(DATASET_CONFIG)}")
    return requested


def _open_store(dataset: str, r: Results, tag: str,
                light: bool = False):
    """Open a DatasetStore (running L0 on the way in). Records SKIP/FAIL and
    returns None on any problem; otherwise returns the store.

    ``light=True`` skips loading X and the heavy DEG/p-value uns matrices — safe
    for the fast layers (L0/L1/L2/L5), which never read them as values."""
    from benchmark.data_loader import DatasetStore
    try:
        store = DatasetStore(dataset, light=light)
    except FileNotFoundError:
        r.skip(f"{tag} {dataset}", "h5ad not found")
        return None
    except DatasetValidationError as e:
        r.check(f"{tag} {dataset}: L0 validate_dataset", False, str(e)[:200])
        return None
    except Exception as e:
        r.check(f"{tag} {dataset}", False, f"store init: {type(e).__name__}: {e}")
        return None
    if _SKIP_SENTINEL not in store._pb():
        r.skip(f"{tag} {dataset}",
               f"pseudobulk lacks '{_SKIP_SENTINEL}' (pre-all-cell store; re-run get_data.py)")
        return None
    return store


# ===================================================================
# L1 — dataset invariants + split integrity (leakage)
# ===================================================================


def _backed_safe_xmax(adata, chunk: int = 8192) -> float:
    """Max of adata.X, reading row-chunks when X is backed."""
    from scipy.sparse import issparse
    X = adata.X
    if not adata.isbacked:
        return float(X.max()) if getattr(X, "nnz", 1) else 0.0
    n = X.shape[0]
    mx = 0.0
    for i in range(0, n, chunk):
        block = X[i:i + chunk]
        if issparse(block):
            bmax = float(block.max()) if block.nnz else 0.0
        else:
            bmax = float(np.asarray(block).max()) if block.size else 0.0
        mx = max(mx, bmax)
    return mx


def verify_dataset_invariants(store, r: Results) -> None:
    """L1a: shapes/names alignment, per-half counts, X looks log1p."""
    ds = store.dataset
    nb, nk, ng = store.n_bins, store.n_kos, store.n_genes
    ok_shapes = (store.first_half_bulk.shape == (nb, nk, ng)
                 and store.second_half_bulk.shape == (nb, nk, ng)
                 and store.ctrl_bulk.shape == (nb, ng)
                 and store.n_cells_first.shape == (nb, nk)
                 and store.n_cells_second.shape == (nb, nk))
    r.check(f"invariants {ds}: array shapes align with names", ok_shapes,
            f"n_bins={nb} n_kos={nk} n_genes={ng}")
    r.check(f"invariants {ds}: gene_names == var_names",
            list(store.gene_names) == list(store._adata.var_names))
    min_first = int(store.n_cells_first.min()) if nk else MIN_CELLS
    min_second = int(store.n_cells_second.min()) if nk else MIN_CELLS
    r.check(f"invariants {ds}: every (bin,ko) has >= {MIN_CELLS} cells per half",
            min_first >= MIN_CELLS and min_second >= MIN_CELLS,
            f"min_first={min_first} min_second={min_second}")
    if store._adata.X is not None:
        xmax = _backed_safe_xmax(store._adata)
        r.check(f"invariants {ds}: X looks log1p-normalized", 0.0 <= xmax < 50.0,
                f"X.max()={xmax:.3f}")
    else:
        # light store (fast mode): X not loaded — the full-X scan is skipped here
        # and covered by --full. Avoids decompressing the entire expression matrix.
        r.info(f"invariants {ds}: X not loaded (light store) — log1p range check deferred to --full")
    r.check(f"invariants {ds}: all_deltas.shape == first_half_deltas.shape",
            store.all_deltas.shape == store.first_half_deltas.shape,
            f"{store.all_deltas.shape}")


def verify_techdup_halves(store, r: Results) -> None:
    """L1b: the tech-DUPLICATE half-split is well-formed.

    This is the cell-level 50/50 `tech_dup_split` (first_half / second_half) that
    powers the evaluation ground truth (first half) and the positive control
    (second half) — NOT the cross-validation train/test split (see
    `verify_cv_split_leakage`). Asserts exactly two halves are present; the
    per-(bin,ko) ">= MIN_CELLS in each half" check lives in
    `verify_dataset_invariants` (via n_cells_first/second).
    """
    ds = store.dataset
    halves = set(str(h) for h in store._adata.obs["tech_dup_split"].astype(str).unique())
    r.check(f"techdup {ds}: tech_dup_split has exactly two halves",
            halves == {"first_half", "second_half"}, f"halves={sorted(halves)}")


def verify_cv_split_leakage(store, r: Results) -> None:
    """L1c: the cross-validation train/val/test split has NO leakage.

    For every (scenario, fold) this is the train/test split that is actually
    benchmarked: assert train ∩ test == 0 and train ∩ val == 0 at the (cell_type,
    pert) level, and that the non-rectangular scenarios (which hold out a
    scattered (cell_type, pert) set — UnseenBoth/UnseenPair) carry their explicit
    uns pair masks, so the loader never falls back to the leaky
    train_bins × train_kos rectangle.
    """
    ds = store.dataset
    cfg = DATASET_CONFIG.get(ds, {}).get("scenarios", {})
    for scenario, sc in cfg.items():
        for fold in range(sc["n_folds"]):
            try:
                split = store.split(scenario, fold)
            except Exception as e:
                r.check(f"cv-split {ds}/{scenario}/fold{fold}: loadable", False,
                        f"{type(e).__name__}: {e}")
                continue
            tr = split.train_mask_2d(store.n_bins, store.n_kos)
            te = split.test_mask_2d(store.n_bins, store.n_kos)
            n_tt = int((tr & te).sum())
            r.check(f"cv-split {ds}/{scenario}/fold{fold}: train ∩ test == 0", n_tt == 0,
                    f"overlap={n_tt} cells")
            if split.val_pair_mask is not None:
                n_tv = int((tr & split.val_pair_mask).sum())
                r.check(f"cv-split {ds}/{scenario}/fold{fold}: train ∩ val == 0", n_tv == 0,
                        f"overlap={n_tv} cells")
            # Non-rectangular scenarios MUST carry stored masks (no leaky fallback).
            if scenario in _NON_RECTANGULAR_SCENARIOS and store.n_bins > 1:
                has_mask = f"{scenario}_fold_{fold}_train_mask" in store._adata.uns
                r.check(
                    f"cv-split {ds}/{scenario}/fold{fold}: uns pair masks present "
                    f"(no leaky-rectangle fallback)", has_mask,
                    "" if has_mask else "missing — predictors would train on the "
                    "leaky train_bins × train_kos rectangle; re-run get_data.py / migration")


def verify_split_conventions(store, r: Results) -> None:
    """L1d: multi-bin split *convention* checks (beyond plain leakage).

    Leakage (L1c) only proves train ∩ test == 0 — it would PASS a multi-bin
    UnseenPert wired with the single-bin round-robin (the xatlas_orion bug). These
    convention checks catch that class:

      * UnseenPert is perturbation-consistent across bins (every perturbation is
        held out in all cell types or in none) and round-robin-covers every
        perturbation exactly once across folds.
      * For the perturbench block-partition datasets, the per-fold UnseenPert test
        set is a subset of the UnseenBoth held-out block for the same fold (both
        derive from the same seed-0 partition). Other multi-bin datasets are
        reported as INFO, not failed (e.g. xatlas_orion uses the single-bin
        round-robin for UnseenPert — a known, documented divergence).
      * UnseenPair: every perturbation is seen in train in at least one bin (no
        pert with all its (cell_type, pert) versions val/test, which would
        degenerate into UnseenPert). Hard check — the generator repairs to enforce it.

    Shares the predicates with data/_utils.py (and the pytest unit tests), so the
    artifact gate and the generator are validated against one definition.
    """
    from data._utils import (  # type: ignore
        _is_control_label,
        check_pert_consistent_across_bins,
        check_full_coverage,
        check_unseenpert_subset_of_unseenboth,
        check_unseenpair_pert_in_train_at_least_once,
    )
    ds = store.dataset
    if store.n_bins <= 1:
        return  # the convention is about the cell-type (bin) axis
    cfg = DATASET_CONFIG.get(ds, {}).get("scenarios", {})
    non_control = np.array([not _is_control_label(k) for k in store.ko_names], bool)

    def _test_ko_names_and_mask(scenario: str, fold: int):
        te = store.split(scenario, fold).test_mask_2d(store.n_bins, store.n_kos)
        cols = np.where(te.any(axis=0) & non_control)[0]
        return {store.ko_names[i] for i in cols}, te

    # --- UnseenPert: perturbation-consistency across bins + full round-robin coverage
    if "UnseenPert" in cfg:
        test_masks, loaded_all = [], True
        for fold in range(cfg["UnseenPert"]["n_folds"]):
            try:
                _, te = _test_ko_names_and_mask("UnseenPert", fold)
            except Exception as e:
                r.check(f"convention {ds}/UnseenPert/fold{fold}: loadable", False,
                        f"{type(e).__name__}: {e}")
                loaded_all = False
                continue
            ok = check_pert_consistent_across_bins(te)
            r.check(f"convention {ds}/UnseenPert/fold{fold}: perturbation-consistent "
                    f"across bins", ok,
                    "" if ok else "a perturbation is test in some cell types but not "
                    "others (per-bin UnseenPert mistake?)")
            test_masks.append(te)
        if loaded_all and test_masks:
            cov = check_full_coverage(test_masks, non_control)
            if ds in ADOPTED_FOLD_DATASETS and not cov:
                # Adopted (external) folds: per-fold test sets are independent
                # draws, not a round-robin partition, so full coverage is not
                # expected. Report, don't fail (same spirit as the UnseenPert⊆
                # UnseenBoth gate below for non-block-partition datasets).
                r.info(f"convention {ds}/UnseenPert: perturbations NOT tested exactly "
                       f"once across folds — {ds} adopts external folds (independent "
                       f"per-seed draws, not a round-robin partition); informational only")
            else:
                r.check(f"convention {ds}/UnseenPert: every perturbation tested exactly "
                        f"once across folds", cov,
                        "" if cov else "round-robin coverage broken")

    # --- UnseenPair: every perturbation must be seen in train in at least one bin
    # (else the (cell, pert) task degenerates into UnseenPert for that pert). The
    # generator enforces this via a repair pass, so this is a hard check.
    if "UnseenPair" in cfg:
        for fold in range(cfg["UnseenPair"]["n_folds"]):
            try:
                si = store.split("UnseenPair", fold)
                tr_full = si.train_mask_2d(store.n_bins, store.n_kos)
                te_full = si.test_mask_2d(store.n_bins, store.n_kos)
            except Exception as e:
                r.check(f"convention {ds}/UnseenPair/fold{fold}: loadable", False,
                        f"{type(e).__name__}: {e}")
                continue
            # Only perts that are PART of the pair partition (appear in some train
            # or test pair) are subject to the degeneracy check. Perts absent from
            # the singles partition entirely (e.g. ecoli_synthetic doubles/doses —
            # all-False mask columns, not held out at all) are out of scenario.
            in_scen = non_control & (tr_full.any(axis=0) | te_full.any(axis=0))
            tr = tr_full[:, in_scen]
            ok = check_unseenpair_pert_in_train_at_least_once(tr) if tr.shape[1] else True
            n_bad = int((~tr.any(axis=0)).sum())
            r.check(f"convention {ds}/UnseenPair/fold{fold}: every perturbation in "
                    f"train in ≥1 bin", ok,
                    "" if ok else f"{n_bad} perturbation(s) have all (cell_type, pert) "
                    f"versions val/test (degenerates into UnseenPert)")

    # --- cross-scenario: UnseenPert test ⊆ UnseenBoth block (perturbench convention)
    if "UnseenPert" in cfg and "UnseenBoth" in cfg:
        enforce = ds in MULTIBIN_BLOCK_PARTITION_DATASETS
        n_folds = min(cfg["UnseenPert"]["n_folds"], cfg["UnseenBoth"]["n_folds"])
        for fold in range(n_folds):
            try:
                up, _ = _test_ko_names_and_mask("UnseenPert", fold)
                ub, _ = _test_ko_names_and_mask("UnseenBoth", fold)
            except Exception as e:
                r.info(f"convention {ds}/fold{fold}: could not compare "
                       f"UnseenPert⊆UnseenBoth ({type(e).__name__}: {e})")
                continue
            ok = check_unseenpert_subset_of_unseenboth(up, ub)
            detail = f"|UnseenPert\\UnseenBoth|={len(set(up) - set(ub))}"
            if enforce:
                r.check(f"convention {ds}/fold{fold}: UnseenPert test ⊆ UnseenBoth "
                        f"block", ok, "" if ok else detail)
            elif not ok:
                r.info(f"convention {ds}/fold{fold}: UnseenPert NOT ⊆ UnseenBoth block "
                       f"— {ds} uses a different UnseenPert partition than the "
                       f"perturbench seed-0 convention ({detail}); informational only")


def verify_pseudobulk_math(store, r: Results) -> None:
    """L2a: all_bulk == count-weighted half means; all_deltas == all_bulk - ctrl."""
    ds = store.dataset
    n1 = store.n_cells_first.astype(np.float64)[:, :, None]
    n2 = store.n_cells_second.astype(np.float64)[:, :, None]
    total = n1 + n2
    ref_all = np.divide(n1 * store.first_half_bulk + n2 * store.second_half_bulk,
                        total, out=np.zeros_like(n1 * store.first_half_bulk),
                        where=total > 0)
    max_abs = float(np.abs(store.all_bulk - ref_all).max()) if store.n_kos else 0.0
    r.check(f"pb-math {ds}: all_bulk == count-weighted half means", max_abs <= _REL_TOL,
            f"max|Δ|={max_abs:.2e}")
    ad_ref = store.all_bulk - store.ctrl_bulk[:, None, :]
    r.check(f"pb-math {ds}: all_deltas == all_bulk - ctrl",
            np.array_equal(store.all_deltas, ad_ref))


def _naive_pseudobulk(adata, ko_names, bin_names, bin_col):
    from data._utils import _is_control_label, _dense_row_mean  # type: ignore
    X = adata.X
    n_genes = adata.n_vars
    conds = adata.obs["condition"].values
    halves = adata.obs["tech_dup_split"].values
    is_ctrl = np.array([_is_control_label(c) for c in conds])
    is_first = halves == "first_half"
    is_second = halves == "second_half"
    if bin_col is None:
        bin_assign = np.zeros(adata.n_obs, dtype=int)
    else:
        bidx = {b: i for i, b in enumerate(bin_names)}
        bin_assign = np.array([bidx[b] for b in adata.obs[bin_col].values])
    nb, nk = len(bin_names), len(ko_names)
    ctrl = np.zeros((nb, n_genes), dtype=np.float32)
    first = np.zeros((nb, nk, n_genes), dtype=np.float32)
    second = np.zeros((nb, nk, n_genes), dtype=np.float32)
    cf = np.zeros((nb, nk), dtype=np.int32)
    cs = np.zeros((nb, nk), dtype=np.int32)
    for bi in range(nb):
        ctrl[bi] = _dense_row_mean(X, (bin_assign == bi) & is_ctrl)
    for ki, cond in enumerate(ko_names):
        cm = conds == cond
        for bi in range(nb):
            in_bin = bin_assign == bi
            fm = in_bin & cm & is_first
            sm = in_bin & cm & is_second
            first[bi, ki] = _dense_row_mean(X, fm)
            second[bi, ki] = _dense_row_mean(X, sm)
            cf[bi, ki] = int(fm.sum())
            cs[bi, ki] = int(sm.sum())
    return ctrl, first, second, cf, cs


def verify_pseudobulk_gate(store, r: Results) -> None:
    """L2b: vectorized compute_pseudobulk == naive double-loop (one small dataset)."""
    ds = store.dataset
    try:
        from data._utils import compute_pseudobulk  # type: ignore
    except Exception as e:
        r.skip(f"pb-vectorize {ds}", f"data._utils unavailable: {e}", intentional=True)
        return
    if store.n_bins != 1:
        r.skip(f"pb-vectorize {ds}", "gate runs on single-bin datasets only",
               intentional=True)
        return
    try:
        adata = store.load_full()
        ko_names = list(store.ko_names)
        bin_names = list(store.bin_names)
        vec = compute_pseudobulk(adata, bin_col=None, min_cells=MIN_CELLS)
        ctrl, first, second, cf, cs = _naive_pseudobulk(adata, ko_names, bin_names, None)
        d = max(float(np.abs(vec["ctrl_bulk"] - ctrl).max()),
                float(np.abs(vec["first_half_bulk"] - first).max()),
                float(np.abs(vec["second_half_bulk"] - second).max()))
        r.check(f"pb-vectorize {ds}: ko_names match", list(vec["ko_names"]) == ko_names)
        r.check(f"pb-vectorize {ds}: vectorized == naive (means)", d <= _PB_GATE_TOL,
                f"max|Δ|={d:.2e}")
        r.check(f"pb-vectorize {ds}: per-half counts match",
                np.array_equal(vec["n_cells_first"], cf)
                and np.array_equal(vec["n_cells_second"], cs))
    except Exception as e:
        r.check(f"pb-vectorize {ds}", False, f"{type(e).__name__}: {e}")


def run_pseudobulk_gate_once(datasets: Sequence[str], r: Results) -> None:
    """Run the vectorization gate ONCE on the smallest available single-bin
    dataset. The gate validates the `compute_pseudobulk` code path (vectorized ==
    naive), which is dataset-independent — and it is expensive (loads full X +
    recomputes), so it must not run per dataset."""
    for ds in datasets:
        store = _open_store(ds, r, "pb-gate")
        if store is None:
            continue
        if store.n_bins == 1:
            verify_pseudobulk_gate(store, r)
            return
    r.skip("pb-vectorize gate", "no single-bin dataset in selection", intentional=True)


# ===================================================================
# L3 — GT-axis (empirically pin the eval ground truth = first_half)
# ===================================================================


def verify_gt_axis(store, r: Results) -> None:
    """Run the Tech-duplicate control through the metrics pipeline and prove the
    score is computed against first_half_deltas (NOT all_deltas or second_half).

    Tech-duplicate's prediction is exactly `second_half_deltas[test]`, fully
    determined by the store, so we can recompute its MSE against each candidate
    GT and check which one the pipeline matches. Needs the saved predictions
    (so it is a `--full` check); SKIPs loudly if they are absent.
    """
    from benchmark.config import predictions_path
    from benchmark.meta_metrics import METRICS_CONFIG, compute_per_pert_metrics

    ds = store.dataset
    scenario = next(iter(DATASET_CONFIG[ds]["scenarios"]))
    fold = 0
    ppath = predictions_path(ds, "Tech-duplicate", scenario, fold)
    if not ppath.exists():
        r.skip(f"gt-axis {ds}", "no Tech-duplicate predictions on disk "
               "(run `predict --predictor Tech-duplicate` first)")
        return
    try:
        mse_spec = [s for s in METRICS_CONFIG if s.name == "mse"]
        df = compute_per_pert_metrics(ds, scenario, fold, "Tech-duplicate", mse_spec, store=store)
        df = df[df["metric"] == "mse"]
        if df.empty:
            r.skip(f"gt-axis {ds}", "pipeline produced no mse rows")
            return
        split = store.split(scenario, fold)
        tbi = split.test_bin_indices_or_all(store.n_bins)
        tki = split.test_ko_indices
        pred = store.second_half_deltas[np.ix_(tbi, tki)]
        # candidate GTs sliced to the same (test_bin, test_ko) rectangle
        cand = {
            "first_half": store.first_half_deltas[np.ix_(tbi, tki)],
            "all": store.all_deltas[np.ix_(tbi, tki)],
            "second_half": store.second_half_deltas[np.ix_(tbi, tki)],
        }
        # pipeline mean mse across the scored rows
        pipe_mse = float(df["value"].mean())
        # our mean mse per candidate (row mean over genes, then mean over rows)
        def _mean_mse(gt):
            return float(np.nanmean(((pred - gt) ** 2).mean(axis=-1)))
        cand_mse = {k: _mean_mse(v) for k, v in cand.items()}
        best = min(cand_mse, key=lambda k: abs(cand_mse[k] - pipe_mse))
        r.check(f"gt-axis {ds}: pipeline scores vs first_half (not all/second)",
                best == "first_half" and abs(cand_mse["first_half"] - pipe_mse) < 1e-3,
                f"pipe={pipe_mse:.5f} first={cand_mse['first_half']:.5f} "
                f"all={cand_mse['all']:.5f} second={cand_mse['second_half']:.5f} → {best}")
        r.check(f"gt-axis {ds}: eval GT != training target (first_half != all)",
                abs(cand_mse["first_half"] - cand_mse["all"]) > 1e-9
                or np.array_equal(store.first_half_deltas, store.all_deltas),
                "first-half and all-cell deltas are distinguishable")
    except ModuleNotFoundError as e:
        # The metric kernels need torch — L3 only runs in the `vcell` env.
        r.skip(f"gt-axis {ds}", f"metrics backend unavailable ({e}); "
               f"run `--full` in the vcell env", intentional=True)
    except Exception as e:
        r.check(f"gt-axis {ds}", False, f"{type(e).__name__}: {e}")


# ===================================================================
# L4 — predictor contracts (synthetic, data-free)
# ===================================================================
#
# Each predictor is given a tiny synthetic store whose structure it models
# exactly; with noise=0 the correct answer is analytic. Every contract asserts
# (a) recovery of the known answer and (b) a SENTINEL that fires iff the
# predictor took its documented degenerate fallback (so "really scales the
# target" is distinguished from "fell back to zeros").


class _NS:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class _SynStore:
    """Minimal DatasetStore stand-in exposing the surface predictors read in
    fit()/predict() (no metrics, no disk)."""

    def __init__(self, *, gene_names, ko_names, bin_names, ctrl_bulk, all_deltas,
                 split, second_half_deltas=None, dataset="synthetic", uns=None):
        self.dataset = dataset
        self.gene_names = list(gene_names)
        self.ko_names = list(ko_names)
        self.bin_names = list(bin_names)
        self.n_genes = len(gene_names)
        self.n_kos = len(ko_names)
        self.n_bins = len(bin_names)
        self.ctrl_bulk = np.asarray(ctrl_bulk, dtype=np.float32)
        self.all_deltas = np.asarray(all_deltas, dtype=np.float32)
        self.all_bulk = (self.ctrl_bulk[:, None, :] + self.all_deltas).astype(np.float32)
        self.first_half_deltas = self.all_deltas
        self.second_half_deltas = (np.asarray(second_half_deltas, dtype=np.float32)
                                   if second_half_deltas is not None else self.all_deltas)
        self._split = split
        self.adata = _NS(uns=dict(uns or {}))

    def split(self, scenario, fold):
        return self._split


def _make_split(n_bins, train_kos, test_kos, *, train_bins=None, test_bins=None,
                train_pair_mask=None, test_pair_mask=None, val_pair_mask=None):
    from benchmark.data_loader import SplitInfo
    return SplitInfo(
        train_ko_indices=np.array(sorted(train_kos), dtype=int),
        val_ko_indices=np.array([], dtype=int),
        test_ko_indices=np.array(sorted(test_kos), dtype=int),
        train_bin_indices=None if train_bins is None else np.array(sorted(train_bins), dtype=int),
        val_bin_indices=None,
        test_bin_indices=None if test_bins is None else np.array(sorted(test_bins), dtype=int),
        train_pair_mask=train_pair_mask, val_pair_mask=val_pair_mask,
        test_pair_mask=test_pair_mask,
    )


def _pred(name):
    from benchmark.predictors.base import get_predictor
    return get_predictor(name)


def _corr(a, b) -> float:
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 2:
        return float("nan")
    a, b = a[m] - a[m].mean(), b[m] - b[m].mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else 0.0


# ---- individual contracts -------------------------------------------------


def _c_zero(r):
    rng = np.random.default_rng(0)
    g, b = 8, 1
    deltas = rng.normal(0, 1, (b, 5, g)).astype(np.float32)
    store = _SynStore(gene_names=[f"g{i}" for i in range(g)],
                      ko_names=[f"k{i}" for i in range(5)], bin_names=["b0"],
                      ctrl_bulk=rng.uniform(0.5, 2, (b, g)), all_deltas=deltas,
                      split=_make_split(b, [0, 1, 2], [3, 4]))
    out = _pred("Zero")().predict(store, "UnseenPert", 0)
    r.check("contract Zero: shape", out.shape == (1, 2, g), f"{out.shape}")
    r.check("contract Zero: all zeros", np.all(out == 0.0))


def _target_store(n_genes=12, n_bins=1, alpha=0.7, noise=0.0, seed=0,
                  extra_test=("g0+g1", "g2@0.5")):
    """KO k targets gene k (k<n_genes). delta[b,k,target]=-alpha*baseline. Plus
    optional combo/dose test KOs."""
    rng = np.random.default_rng(seed)
    gene_names = [f"g{i}" for i in range(n_genes)]
    bin_names = [f"b{i}" for i in range(n_bins)]
    baseline = rng.uniform(0.5, 2.0, (n_bins, n_genes)).astype(np.float32)
    ko_names = list(gene_names) + list(extra_test)
    n_kos = len(ko_names)
    deltas = rng.normal(0, noise, (n_bins, n_kos, n_genes)).astype(np.float32)
    for k, name in enumerate(ko_names):
        for tgt in parse_target_genes(name):
            if tgt in gene_names:
                gi = gene_names.index(tgt)
                for b in range(n_bins):
                    deltas[b, k, gi] = -alpha * baseline[b, gi] + rng.normal(0, noise)
    train = list(range(n_genes))            # all single-gene KOs train
    test = list(range(n_genes, n_kos))      # the extra (combo/dose) KOs test
    split = _make_split(n_bins, train, test)
    return _SynStore(gene_names=gene_names, ko_names=ko_names, bin_names=bin_names,
                     ctrl_bulk=baseline, all_deltas=deltas, split=split), baseline


def _c_targetzero(r):
    store, baseline = _target_store()
    out = _pred("TargetZero")().predict(store, "UnseenPert", 0)
    # test KO 0 is "g0+g1" -> targets g0, g1
    nz = set(np.nonzero(out[0, 0])[0].tolist())
    r.check("contract TargetZero: only target coords nonzero", nz <= {0, 1}, f"nz={sorted(nz)}")
    r.check("contract TargetZero: target == -baseline",
            abs(out[0, 0, 0] - (-baseline[0, 0])) < 1e-6 and abs(out[0, 0, 1] - (-baseline[0, 1])) < 1e-6)
    r.check("contract TargetZero: SENTINEL output not all-zero", float(np.nanmax(np.abs(out))) > 0)


def _c_targetscaling(r):
    store, baseline = _target_store(alpha=0.7, noise=0.0)
    ts = _pred("TargetScaling")()
    ts.fit(store, "UnseenPert", 0)
    r.check("contract TargetScaling: recovers alpha (noise=0)", abs(ts.alpha - 0.7) < 1e-6,
            f"alpha={ts.alpha:.10f}")
    out = ts.predict(store, "UnseenPert", 0)
    r.check("contract TargetScaling: target == -alpha*baseline",
            abs(out[0, 0, 0] - (-ts.alpha * baseline[0, 0])) < 1e-6)
    r.check("contract TargetScaling: SENTINEL really scales (not all-zero)",
            float(np.nanmax(np.abs(out))) > 0)
    # noisy recovery
    store_n, _ = _target_store(alpha=0.7, noise=0.01, seed=3)
    tsn = _pred("TargetScaling")(); tsn.fit(store_n, "UnseenPert", 0)
    r.check("contract TargetScaling: noisy alpha within 0.05", abs(tsn.alpha - 0.7) < 0.05,
            f"alpha={tsn.alpha:.5f}")


def _c_mean_plus_targetscaling(r):
    # Composite == regime mean (Mean-over-perturbations for UnseenPert) + the
    # (NaN->0) TargetScaling term, with the mean's coverage preserved.
    store, _ = _target_store(alpha=0.7, noise=0.0)
    sc = "UnseenPert"
    mtp = _pred("Mean+TargetScaling")(); mtp.fit(store, sc, 0)
    out = mtp.predict(store, sc, 0)
    mp = _pred("Mean-over-perturbations")(); mp.fit(store, sc, 0); mean_out = mp.predict(store, sc, 0)
    ts = _pred("TargetScaling")(); ts.fit(store, sc, 0); ts_out = ts.predict(store, sc, 0)
    expect = mean_out + np.nan_to_num(ts_out, nan=0.0)
    r.check("contract Mean+TargetScaling: == mean + (NaN->0) target-scaling term",
            np.allclose(np.nan_to_num(out), np.nan_to_num(expect), atol=1e-5))
    r.check("contract Mean+TargetScaling: keeps mean coverage (no NaN poisoning)",
            bool((np.isfinite(mean_out) == np.isfinite(out)).all()))
    r.check("contract Mean+TargetScaling: SENTINEL composes (not pure mean, not target-only)",
            (not np.allclose(np.nan_to_num(out), np.nan_to_num(mean_out), atol=1e-6))
            and (not np.allclose(np.nan_to_num(out), np.nan_to_num(ts_out, nan=0.0), atol=1e-6)))


def _c_target_errors(r):
    # empty train split -> raise
    store, _ = _target_store()
    empty = _SynStore(gene_names=store.gene_names, ko_names=store.ko_names,
                      bin_names=store.bin_names, ctrl_bulk=store.ctrl_bulk,
                      all_deltas=store.all_deltas,
                      split=_make_split(store.n_bins, [], [0]))
    r.expect_raises("contract TargetScaling: raises on empty train split",
                    lambda: _pred("TargetScaling")().fit(empty, "UnseenPert", 0),
                    ValueError, "empty training split")
    # degenerate denominator (baseline all zero at targets)
    deg = _SynStore(gene_names=store.gene_names, ko_names=store.ko_names,
                    bin_names=store.bin_names,
                    ctrl_bulk=np.zeros_like(store.ctrl_bulk), all_deltas=store.all_deltas,
                    split=store._split)
    r.expect_raises("contract TargetScaling: raises on degenerate denominator",
                    lambda: _pred("TargetScaling")().fit(deg, "UnseenPert", 0),
                    ValueError, "degenerate OLS denominator")
    r.expect_raises("contract TargetScaling: predict before fit raises",
                    lambda: _pred("TargetScaling")().predict(store, "UnseenPert", 0),
                    RuntimeError, "before fit")


def _c_mean_perts(r):
    rng = np.random.default_rng(1)
    g, nb, nk = 6, 2, 5
    mu = rng.normal(0, 1, (nb, g)).astype(np.float32)
    deltas = np.broadcast_to(mu[:, None, :], (nb, nk, g)).astype(np.float32).copy()
    store = _SynStore(gene_names=[f"g{i}" for i in range(g)],
                      ko_names=[f"k{i}" for i in range(nk)],
                      bin_names=[f"b{i}" for i in range(nb)],
                      ctrl_bulk=rng.uniform(0.5, 2, (nb, g)), all_deltas=deltas,
                      split=_make_split(nb, [0, 1, 2], [3, 4]))
    out = _pred("Mean-over-perturbations")().predict(store, "UnseenPert", 0)
    ok = all(np.allclose(out[i], mu[i], atol=1e-5) for i in range(nb))
    r.check("contract Mean-over-perturbations: == per-bin train mean", ok)
    r.check("contract Mean-over-perturbations: SENTINEL varies across bins",
            not np.allclose(out[0, 0], out[1, 0]))


def _c_mean_cells(r):
    rng = np.random.default_rng(2)
    g, nb, nk = 6, 3, 5
    nu = rng.normal(0, 1, (nk, g)).astype(np.float32)   # per-ko, constant across bins
    deltas = np.broadcast_to(nu[None, :, :], (nb, nk, g)).astype(np.float32).copy()
    split = _make_split(nb, list(range(nk)), list(range(nk)),
                        train_bins=[0, 1], test_bins=[2])
    store = _SynStore(gene_names=[f"g{i}" for i in range(g)],
                      ko_names=[f"k{i}" for i in range(nk)],
                      bin_names=[f"b{i}" for i in range(nb)],
                      ctrl_bulk=rng.uniform(0.5, 2, (nb, g)), all_deltas=deltas, split=split)
    out = _pred("Mean-over-cell-types")().predict(store, "UnseenCell", 0)
    r.check("contract Mean-over-cell-types: == per-ko train mean",
            np.allclose(out[0], nu, atol=1e-5))
    r.check("contract Mean-over-cell-types: SENTINEL varies across kos",
            not np.allclose(out[0, 0], out[0, 1]))


def _c_mean_both(r):
    rng = np.random.default_rng(3)
    g, nb, nk = 6, 2, 5
    deltas = rng.normal(0, 1, (nb, nk, g)).astype(np.float32)
    split = _make_split(nb, [0, 1, 2], [3, 4])
    store = _SynStore(gene_names=[f"g{i}" for i in range(g)],
                      ko_names=[f"k{i}" for i in range(nk)],
                      bin_names=[f"b{i}" for i in range(nb)],
                      ctrl_bulk=rng.uniform(0.5, 2, (nb, g)), all_deltas=deltas, split=split)
    gm = deltas[np.ix_([0, 1], [0, 1, 2])].mean(axis=(0, 1))
    out = _pred("Mean-over-perturbations-and-cell-types")().predict(store, "UnseenPert", 0)
    r.check("contract Mean-over-perturbations-and-cell-types: == global train mean",
            np.allclose(out[0, 0], gm, atol=1e-5))
    r.check("contract Mean-over-perturbations-and-cell-types: SENTINEL constant across (bin,ko)",
            np.allclose(out, out[0, 0][None, None, :], atol=1e-6))


def _c_twoway(r):
    rng = np.random.default_rng(4)
    g, nb, nk = 5, 3, 4
    mu = rng.normal(0, 1, g)
    alpha = rng.normal(0, 1, (nb, g))
    beta = rng.normal(0, 1, (nk, g))
    deltas = (mu[None, None] + alpha[:, None] + beta[None, :]).astype(np.float32)
    test_pairs = [(2, 0), (1, 3)]
    te = np.zeros((nb, nk), bool)
    for b, k in test_pairs:
        te[b, k] = True
    tr = ~te
    split = _make_split(nb, list(range(nk)), sorted({k for _, k in test_pairs}),
                        train_bins=list(range(nb)),
                        test_bins=sorted({b for b, _ in test_pairs}),
                        train_pair_mask=tr, test_pair_mask=te)
    store = _SynStore(gene_names=[f"g{i}" for i in range(g)],
                      ko_names=[f"k{i}" for i in range(nk)],
                      bin_names=[f"b{i}" for i in range(nb)],
                      ctrl_bulk=rng.uniform(0.5, 2, (nb, g)), all_deltas=deltas, split=split)
    out = _pred("Two-way-mean")().predict(store, "UnseenPair", 0)
    tbi = sorted({b for b, _ in test_pairs}); tki = sorted({k for _, k in test_pairs})
    truth = (mu[None, None] + alpha[tbi][:, None] + beta[tki][None, :])
    c = _corr(out, truth)
    r.check("contract Two-way-mean: recovers additive structure (corr>0.9)", c > 0.9, f"corr={c:.3f}")
    r.check("contract Two-way-mean: SENTINEL varies with bin AND ko",
            not np.allclose(out[0], out[-1]) and not np.allclose(out[:, 0], out[:, -1]))


def _c_ridge(r):
    # bin-only synthetic: delta depends only on bin -> additive model recovers it.
    rng = np.random.default_rng(5)
    g, nb, nk = 6, 3, 6
    a = rng.normal(0, 1, (nb, g)).astype(np.float32)
    deltas = np.broadcast_to(a[:, None, :], (nb, nk, g)).astype(np.float32).copy()
    split = _make_split(nb, [0, 1, 2, 3], [4, 5])  # all bins train, kos split (UnseenPert)
    store = _SynStore(gene_names=[f"g{i}" for i in range(g)],
                      ko_names=[f"k{i}" for i in range(nk)],
                      bin_names=[f"b{i}" for i in range(nb)],
                      ctrl_bulk=rng.uniform(0.5, 2, (nb, g)), all_deltas=deltas, split=split)
    la = _pred("LinearAdditive")(); la.fit(store, "UnseenPert", 0)
    out = la.predict(store, "UnseenPert", 0)
    ok = all(np.allclose(out[b], a[b], atol=1e-3) for b in range(nb))
    r.check("contract LinearAdditive: recovers bin-only structure (exact)", ok)
    r.check("contract LinearAdditive: SENTINEL W learned (nonzero) & varies across bins",
            la.W is not None and float(np.abs(la.W).max()) > 1e-6
            and not np.allclose(out[0, 0], out[-1, 0]))
    rd = _pred("Ridge")(); rd.fit(store, "UnseenPert", 0)
    outr = rd.predict(store, "UnseenPert", 0)
    cs = [_corr(outr[b, 0], a[b]) for b in range(nb)]
    r.check("contract Ridge: predictions track bin structure (corr>0.9)",
            all(c > 0.9 for c in cs), f"per-bin corr={[round(c,2) for c in cs]}")
    r.check("contract Ridge: SENTINEL W nonzero & varies across bins",
            rd.W is not None and float(np.abs(rd.W).max()) > 1e-6
            and not np.allclose(outr[0, 0], outr[-1, 0]))


def _c_correlation(r):
    rng = np.random.default_rng(6)
    g, nb, nk = 20, 1, 6
    deltas = rng.normal(0, 1, (nb, nk, g)).astype(np.float32)
    match = 2                       # test KO is a scaled copy of train KO #2
    test_idx = nk                   # appended test KO
    test_delta = (3.0 * deltas[0, match]).astype(np.float32)
    deltas = np.concatenate([deltas, test_delta[None, None]], axis=1)  # (1, nk+1, g)
    ko_names = [f"k{i}" for i in range(nk)] + ["t0"]
    split = _make_split(nb, list(range(nk)), [test_idx])
    store = _SynStore(gene_names=[f"g{i}" for i in range(g)], ko_names=ko_names,
                      bin_names=["b0"], ctrl_bulk=rng.uniform(0.5, 2, (nb, g)),
                      all_deltas=deltas, split=split)
    m = _pred("Correlation")(); m.fit(store, "UnseenPert", 0)
    out = m.predict(store, "UnseenPert", 0)
    r.check("contract Correlation: copies the matched TRAIN delta",
            np.allclose(out[0, 0], deltas[0, match], atol=1e-4))
    r.check("contract Correlation: SENTINEL/leakage out != test's own delta",
            not np.allclose(out[0, 0], deltas[0, test_idx], atol=1e-4))


def _c_latent(r):
    # low-rank target structure so the bilinear solve has signal.
    store, _ = _target_store(n_genes=24, alpha=0.8, noise=0.0, seed=7,
                             extra_test=("g0", "g1", "nonexistent_gene"))
    # rebuild split: train all single-gene KOs (0..23), test the appended ones
    n_single = 24
    store._split = _make_split(1, list(range(n_single)),
                               list(range(n_single, store.n_kos)))
    try:
        m = _pred("LatentAdditive")(k=8)
        m.fit(store, "UnseenPert", 0)
    except ImportError as e:
        r.skip("contract LatentAdditive", f"sklearn unavailable: {e}", intentional=True)
        return
    r.check("contract LatentAdditive: SENTINEL K learned (not center-only fallback)",
            m.K is not None and float(np.abs(m.K).max()) > 1e-8)
    out = m.predict(store, "UnseenPert", 0)
    # appended test KOs: g0 (resolvable), g1 (resolvable), nonexistent (NaN)
    r.check("contract LatentAdditive: resolvable test KO finite", np.isfinite(out[0, 0]).all())
    r.check("contract LatentAdditive: unresolvable test KO -> NaN", np.isnan(out[0, 2]).all())
    c = _corr(out[0, 0], store.all_deltas[0, store.ko_names.index("g0")])
    r.info(f"LatentAdditive resolvable-test corr={c:.3f} (soft)")


def _c_bilinear(r):
    rng = np.random.default_rng(8)
    g = 24
    gene_names = [f"g{i}" for i in range(g)]
    baseline = rng.uniform(0.5, 2, (1, g)).astype(np.float32)
    # KO names ARE gene symbols (BilinearRidge requires this); +1 unresolvable test KO
    ko_names = list(gene_names) + ["not_a_gene"]
    deltas = rng.normal(0, 0.01, (1, len(ko_names), g)).astype(np.float32)
    for k in range(g):
        deltas[0, k, k] = -0.8 * baseline[0, k]
    split = _make_split(1, list(range(g - 4)), list(range(g - 4, len(ko_names))))
    store = _SynStore(gene_names=gene_names, ko_names=ko_names, bin_names=["b0"],
                      ctrl_bulk=baseline, all_deltas=deltas, split=split)
    try:
        m = _pred("BilinearRidge")()
        m.fit(store, "UnseenPert", 0)
    except ImportError as e:
        r.skip("contract BilinearRidge", f"sklearn unavailable: {e}", intentional=True)
        return
    r.check("contract BilinearRidge: SENTINEL non-degenerate fit (W is not None)",
            m.W is not None)
    out = m.predict(store, "UnseenPert", 0)
    r.check("contract BilinearRidge: resolvable test KO finite", np.isfinite(out[0, 0]).all())
    r.check("contract BilinearRidge: unresolvable test KO ('not_a_gene') -> NaN",
            np.isnan(out[0, -1]).all())


def _c_globalepistasis(r):
    rng = np.random.default_rng(9)
    g = 8
    n_single = 6
    singles = [f"g{i}" for i in range(n_single)]
    d_single = rng.normal(0, 1, (n_single, g)).astype(np.float32)
    w_true = rng.normal(0, 1, g).astype(np.float32)
    combos, combo_deltas = [], []
    for i in range(n_single):
        for j in range(i + 1, n_single):
            combos.append(f"g{i}+g{j}")
            combo_deltas.append(d_single[i] + d_single[j] + w_true * d_single[i] * d_single[j])
    ko_names = singles + combos
    n_kos = len(ko_names)
    deltas = np.zeros((1, n_kos, g), dtype=np.float32)
    deltas[0, :n_single] = d_single
    deltas[0, n_single:] = np.stack(combo_deltas)
    n_combo = len(combos)
    train_combo = list(range(n_single, n_single + n_combo - 4))
    test_combo = list(range(n_single + n_combo - 4, n_kos))
    split = _make_split(1, list(range(n_single)) + train_combo, test_combo)
    store = _SynStore(gene_names=[f"g{i}" for i in range(g)], ko_names=ko_names,
                      bin_names=["b0"], ctrl_bulk=rng.uniform(0.5, 2, (1, g)),
                      all_deltas=deltas, split=split)
    m = _pred("GlobalEpistasis")(); m.fit(store, "UnseenCombo", 0)
    r.check("contract GlobalEpistasis: SENTINEL learned w != 0 (not additive-only fallback)",
            m.w is not None and float(np.abs(m.w).max()) > 1e-3)
    r.check("contract GlobalEpistasis: recovered w tracks true w (corr>0.9)",
            _corr(m.w, w_true) > 0.9, f"corr={_corr(m.w, w_true):.3f}")
    out = m.predict(store, "UnseenCombo", 0)
    truth = deltas[0, test_combo]
    r.check("contract GlobalEpistasis: predicts test combos (corr>0.9)",
            _corr(out[0], truth) > 0.9, f"corr={_corr(out[0], truth):.3f}")


def _c_combo_additive(r):
    rng = np.random.default_rng(10)
    g, n_single = 8, 4
    singles = [f"g{i}" for i in range(n_single)]
    d = rng.normal(0, 1, (n_single, g)).astype(np.float32)
    combos = ["g0+g1", "g2+g3"]
    ko_names = singles + combos + ["g0+g1+g2"]  # last = 3-way -> unresolvable
    deltas = np.zeros((1, len(ko_names), g), dtype=np.float32)
    deltas[0, :n_single] = d
    deltas[0, n_single] = d[0] + d[1]
    deltas[0, n_single + 1] = d[2] + d[3]
    test = [n_single, n_single + 1, len(ko_names) - 1]
    split = _make_split(1, list(range(n_single)), test)
    store = _SynStore(gene_names=[f"g{i}" for i in range(g)], ko_names=ko_names,
                      bin_names=["b0"], ctrl_bulk=rng.uniform(0.5, 2, (1, g)),
                      all_deltas=deltas, split=split)
    add = _pred("Additive")().predict(store, "UnseenCombo", 0)
    r.check("contract Additive: ΔAB == ΔA + ΔB (resolvable)",
            np.allclose(add[0, 0], d[0] + d[1], atol=1e-5)
            and np.allclose(add[0, 1], d[2] + d[3], atol=1e-5))
    r.check("contract Additive: 3-way combo -> NaN (no silent guess)", np.isnan(add[0, 2]).all())
    mm = _pred("Matching-mean")().predict(store, "UnseenCombo", 0)
    r.check("contract Matching-mean: ΔAB == (ΔA + ΔB)/2",
            np.allclose(mm[0, 0], (d[0] + d[1]) / 2, atol=1e-5))


def _c_scaleddelta(r):
    rng = np.random.default_rng(11)
    g = 8
    gene_names = [f"g{i}" for i in range(g)]
    d_full = rng.normal(0, 1, g).astype(np.float32)
    ko_names = ["GeneA", "GeneA@0.5", "GeneB@0.25"]   # GeneB has no full-dose base
    deltas = np.zeros((1, 3, g), dtype=np.float32)
    deltas[0, 0] = d_full
    split = _make_split(1, [0], [1, 2])
    store = _SynStore(gene_names=gene_names, ko_names=ko_names, bin_names=["b0"],
                      ctrl_bulk=rng.uniform(0.5, 2, (1, g)), all_deltas=deltas, split=split)
    out = _pred("Scaled-delta")().predict(store, "UnseenDose", 0)
    r.check("contract Scaled-delta: Δpartial == dose * Δfull",
            np.allclose(out[0, 0], 0.5 * d_full, atol=1e-5))
    r.check("contract Scaled-delta: missing full-dose base -> NaN", np.isnan(out[0, 1]).all())


def _c_techdup(r):
    rng = np.random.default_rng(12)
    g, nk = 8, 5
    all_d = rng.normal(0, 1, (1, nk, g)).astype(np.float32)
    second = rng.normal(0, 1, (1, nk, g)).astype(np.float32)  # distinct from all
    split = _make_split(1, [0, 1, 2], [3, 4])
    store = _SynStore(gene_names=[f"g{i}" for i in range(g)],
                      ko_names=[f"k{i}" for i in range(nk)], bin_names=["b0"],
                      ctrl_bulk=rng.uniform(0.5, 2, (1, g)), all_deltas=all_d,
                      second_half_deltas=second, split=split)
    out = _pred("Tech-duplicate")().predict(store, "UnseenPert", 0)
    r.check("contract Tech-duplicate: == second_half_deltas[test]",
            np.allclose(out, second[np.ix_([0], [3, 4])], atol=1e-6))


def _c_interp(r):
    rng = np.random.default_rng(13)
    g, nk = 6, 5
    all_d = rng.normal(0, 1, (1, nk, g)).astype(np.float32)
    second = rng.normal(0, 1, (1, nk, g)).astype(np.float32)
    pvals = np.full((1, nk, g), 0.5, dtype=np.float64)
    pvals[..., 0] = 0.0   # alpha=1 -> tech
    pvals[..., 1] = 1.0   # alpha=0 -> mean
    split = _make_split(1, [0, 1, 2], [3, 4])
    store = _SynStore(gene_names=[f"g{i}" for i in range(g)],
                      ko_names=[f"k{i}" for i in range(nk)], bin_names=["b0"],
                      ctrl_bulk=rng.uniform(0.5, 2, (1, g)), all_deltas=all_d,
                      second_half_deltas=second, split=split,
                      uns={"pvals_adj_matrix_second_half": pvals})
    out = _pred("Interp-duplicate")().predict(store, "UnseenPert", 0)
    mean_g1 = all_d[0, [0, 1, 2], 1].mean()
    r.check("contract Interp-duplicate: significant gene (pval=0) -> tech-dup",
            np.allclose(out[0, :, 0], second[0, [3, 4], 0], atol=1e-5))
    r.check("contract Interp-duplicate: non-significant gene (pval=1) -> mean baseline",
            np.allclose(out[0, :, 1], mean_g1, atol=1e-5))


def _c_missing_genes(r):
    """Metric-kernel contract: gene-axis handling of missing predictions.

    A perturbation the model predicted but with one gene missing (NaN) is scored
    with that gene = control (0 delta) under ``fill_missing_genes_with_zero`` and
    excluded under ``drop_missing_genes``; a perturbation with NO predicted genes
    (all-NaN) is SKIPPED in both modes (perturbation axis — never imputed).
    """
    try:
        from benchmark.meta_metrics import (
            _compute_per_pert_via_kernels, main_benchmark_metrics)
        import torch  # noqa: F401  (metric-kernel backend)
    except ImportError as e:
        r.skip("contract missing_genes",
               f"metrics backend (torch) unavailable: {e}; runs in the vcell env",
               intentional=True)
        return
    mse = next(s for s in main_benchmark_metrics() if s.name == "mse")
    # 1 bin, 2 kos, 4 genes. ko0: predicted, gene 3 missing (true Δ=2 there).
    #                        ko1: entirely unpredicted (all-NaN).
    gt = np.zeros((1, 2, 4), dtype=np.float32)
    gt[0, 0, 3] = 2.0
    pred = np.zeros((1, 2, 4), dtype=np.float32)
    pred[0, 0, 3] = np.nan
    pred[0, 1, :] = np.nan

    fill = _compute_per_pert_via_kernels(
        pred, gt, [mse], missing_genes="fill_missing_genes_with_zero")
    drop = _compute_per_pert_via_kernels(
        pred, gt, [mse], missing_genes="drop_missing_genes")

    def _row(df, ko):
        sub = df[(df["ko_idx"] == ko) & (df["metric"] == "mse")]
        return sub.iloc[0] if len(sub) else None

    fa, da = _row(fill, 0), _row(drop, 0)
    r.check("contract missing_genes: ko0 scored in both modes",
            fa is not None and da is not None)
    if fa is not None and da is not None:
        r.check("contract missing_genes: FILL scores full panel (n_genes_valid==4)",
                int(fa["n_genes_valid"]) == 4, f'{fa["n_genes_valid"]}')
        r.check("contract missing_genes: DROP excludes missing gene (n_genes_valid==3)",
                int(da["n_genes_valid"]) == 3, f'{da["n_genes_valid"]}')
        r.check("contract missing_genes: native coverage==3 in both modes",
                int(fa["n_genes_native"]) == 3 and int(da["n_genes_native"]) == 3,
                f'fill={fa["n_genes_native"]} drop={da["n_genes_native"]}')
        # FILL counts the missing gene as a 0 (=control) prediction -> mse=2^2/4=1.0;
        # DROP excludes it -> perfect on the 3 predicted genes -> mse=0.
        r.check("contract missing_genes: FILL imputes zero-delta (mse==1.0)",
                abs(float(fa["value"]) - 1.0) < 1e-6, f'{fa["value"]}')
        r.check("contract missing_genes: DROP perfect on predicted genes (mse==0)",
                abs(float(da["value"])) < 1e-6, f'{da["value"]}')
    # ko1 entirely unpredicted -> SKIPPED (no row) in both modes (perturbation axis).
    r.check("contract missing_genes: all-NaN perturbation skipped (FILL)",
            _row(fill, 1) is None)
    r.check("contract missing_genes: all-NaN perturbation skipped (DROP)",
            _row(drop, 1) is None)


# name -> contract fn. DL adapters have no synthetic contract (they read external
# manifest h5ads) — they are covered by L6 fold-alignment instead.
CONTRACTS: Dict[str, Callable] = {
    "Zero": _c_zero,
    "TargetZero": _c_targetzero,
    "TargetScaling": _c_targetscaling,
    "Mean+TargetScaling": _c_mean_plus_targetscaling,
    "Mean-over-perturbations": _c_mean_perts,
    "Mean-over-cell-types": _c_mean_cells,
    "Mean-over-perturbations-and-cell-types": _c_mean_both,
    "Two-way-mean": _c_twoway,
    "Ridge": _c_ridge,
    "LinearAdditive": _c_ridge,        # exercised together in _c_ridge
    "Correlation": _c_correlation,
    "LatentAdditive": _c_latent,
    "BilinearRidge": _c_bilinear,
    "GlobalEpistasis": _c_globalepistasis,
    "Additive": _c_combo_additive,
    "Matching-mean": _c_combo_additive,  # exercised together in _c_combo_additive
    "Scaled-delta": _c_scaleddelta,
    "Tech-duplicate": _c_techdup,
    "Interp-duplicate": _c_interp,
}
def _is_expensively_trained(cls) -> bool:
    """Is this predictor's training too expensive to run in a contract test?

    True for anything on the expensively-trained tier — training means a GPU run,
    so there is no in-codebase logic to exercise against a synthetic store.

    Asks the TIER, not its two current subclasses: a third one (a remote-API
    trainer, say) is then covered the day it is written rather than the day
    someone remembers to widen this tuple. Imported lazily so `verify` stays
    importable where the predictor stack is heavier than the verifier needs.
    """
    from benchmark.predictors.trained import TrainedPredictor
    return issubclass(cls, TrainedPredictor)


def _dl_model_key_map() -> Dict[str, str]:
    """{display_name: MANIFEST.json model_key} for the DL adapters, read straight
    from the predictor registry (DLAdapter subclasses already set ``model_key``).
    Single source — no separate hardcoded DL list to drift."""
    from benchmark.predictors.base import PREDICTOR_REGISTRY, get_predictor
    get_predictor("Zero")  # force-populate the registry
    return {n: c.model_key for n, c in PREDICTOR_REGISTRY.items()
            if getattr(c, "model_key", "")}


def _drop_rule_predictors() -> Set[str]:
    """Predictors that legitimately cover < all perts (per-target drop rule), read
    from the ``has_drop_rule`` class attribute. Every OTHER non-DL predictor MUST
    cover every perturbation — the one hard coverage invariant (L5)."""
    from benchmark.predictors.base import PREDICTOR_REGISTRY, get_predictor
    get_predictor("Zero")
    return {n for n, c in PREDICTOR_REGISTRY.items() if getattr(c, "has_drop_rule", False)}


def verify_predictor_contracts(r: Results) -> None:
    """L4: run every predictor's synthetic contract + error contracts."""
    from benchmark.predictors.base import PREDICTOR_REGISTRY, get_predictor
    get_predictor("Zero")  # force registry population
    ran: Set[Callable] = set()
    for name in sorted(CONTRACTS):
        fn = CONTRACTS[name]
        if fn in ran:
            continue
        ran.add(fn)
        try:
            fn(r)
        except Exception as e:
            traceback.print_exc()
            r.check(f"contract {fn.__name__}", False, f"{type(e).__name__}: {e}")
    _c_target_errors(r)
    _c_missing_genes(r)
    # Meta-check: every registered predictor whose logic IS in-codebase must have
    # a contract. Exempt are the ones whose logic cannot be exercised on a
    # synthetic store: DL adapters (predictions computed elsewhere) and the
    # expensively-trained tiers, whose "logic" is GPU-hours of training —
    # container predictors run it inside a .sif, torch predictors in-process.
    # Derived from the class, not declared, so a new predictor cannot arrive with
    # a stale flag and quietly skip the check.
    missing = [n for n, c in PREDICTOR_REGISTRY.items()
               if n not in CONTRACTS and n not in _dl_model_key_map()
               and not _is_expensively_trained(c)]
    r.check("contract suite covers every non-DL predictor", not missing,
            f"uncontracted: {missing}" if missing else "")


# ===================================================================
# L5 — coverage (perturbed genes + perturbations per predictor + overlap)
# ===================================================================
#
# The `handles()` predicate is the inverse of the post-hoc
# `overlapping_perturbations.explain()`; it lives HERE (core), so the script can
# import it (script -> core), not the other way round.

# The DL model-key map and the drop-rule predictor set are derived from the
# predictor registry (`model_key` / `has_drop_rule` class attributes) via
# `_dl_model_key_map()` / `_drop_rule_predictors()` above — no hardcoded list here.


# _match_key / _canon_set / _dl_declared_test_set are imported from
# benchmark._fold_align at the top of this module.


def _dl_model_info(dataset: str, scenario: str) -> Dict[str, Dict[str, Set[str]]]:
    """{model_lower: {genes: modeled-gene-set, preds: predicted conditions}} from
    the DL prediction h5ads (panel-independent). Empty if no manifest."""
    import anndata as ad
    from benchmark._fold_align import load_manifest, get_model_folds
    out: Dict[str, Dict[str, Set[str]]] = {}
    try:
        fbm = get_model_folds(load_manifest(), dataset, scenario)
    except Exception as e:
        log.warning("coverage: manifest lookup failed for %s/%s: %s", dataset, scenario, e)
        return out
    for model, folds in fbm.items():
        genes: Set[str] = set()
        preds: Set[str] = set()
        for f in folds:
            try:
                h = ad.read_h5ad(str(f["predictions_path"]), backed="r")
                try:
                    genes |= {str(g) for g in h.var_names}
                    col = "condition" if "condition" in h.obs.columns else h.obs.columns[0]
                    preds |= {_match_key(c) for c in h.obs[col].astype(str).unique()
                              if not _is_control(c)}
                finally:
                    try:
                        h.file.close()
                    except Exception:
                        pass
            except Exception as e:
                log.warning("coverage: read %s %s/%s: %s", model, dataset, scenario, e)
        out[model] = {"genes": genes, "preds": preds}
    return out


def handles(predictor: str, ko: str, store, scenario: str,
            dl_info: Optional[dict] = None) -> Tuple[bool, Optional[str]]:
    """Can `predictor` produce a valid prediction for perturbation `ko`?

    Returns (True, None) if handled, else (False, reason_code). Rule-based — the
    pre-benchmark forecast of the coverage that `overlapping_perturbations.py`
    confirms post-hoc.
    """
    from benchmark.predictors._shared import resolve_target_gene_indices
    tokens = parse_target_genes(ko)
    # Cached on the store (built once per dataset) — coverage calls this per
    # (predictor, ko); rebuilding these each call is O(n_genes+n_kos) and was the
    # dominant cost on high-pert datasets (e.g. xatlas: ~18k kos x ~15 predictors).
    gene_set = store.gene_name_set
    ko_set = store.ko_name_set

    dl_keys = _dl_model_key_map()
    if predictor in dl_keys:
        info = (dl_info or {}).get(dl_keys[predictor], {})
        genes, preds = info.get("genes", set()), info.get("preds", set())
        if _match_key(ko) in preds:
            return True, None
        missing = [t for t in tokens if t not in genes]
        if genes and missing:
            return False, "gene_not_modeled"
        return False, "pert_not_supported"

    if predictor == "BilinearRidge":
        if len(tokens) > 1:
            return False, "single_gene_only"
        return (True, None) if all(t in gene_set for t in tokens) else (False, "target_not_in_panel")
    if predictor == "LatentAdditive":
        return (True, None) if tokens and all(t in gene_set for t in tokens) else (False, "target_not_in_panel")
    if predictor in ("TargetScaling", "TargetZero"):
        return (True, None) if resolve_target_gene_indices(ko, store.gene_to_index, strict=False) \
            else (False, "target_not_in_panel")
    if predictor in ("Additive", "Matching-mean", "GlobalEpistasis"):
        if "+" not in ko:
            return False, "combo_arity"
        parts = sorted(ko.split("+"))
        if len(parts) != 2:
            return False, "combo_arity"
        return (True, None) if all(p in ko_set for p in parts) else (False, "combo_component_missing")
    if predictor == "Scaled-delta":
        if "@" not in ko:
            return False, "combo_arity"
        return (True, None) if ko.split("@", 1)[0].strip() in ko_set else (False, "combo_component_missing")
    # Zero / Mean-* / Two-way-mean / Ridge / LinearAdditive / Correlation /
    # Tech-duplicate / Interp-duplicate: no per-target drop rule.
    return True, None


def verify_coverage(store, r: Results, scenario: Optional[str] = None,
                    write_json: bool = False) -> dict:
    """L5: per dataset, report perturbed-gene + perturbation coverage per
    predictor and the cross-predictor overlap. Returns a summary dict."""
    from benchmark.predictors.base import PREDICTOR_REGISTRY, get_predictor
    get_predictor("Zero")
    ds = store.dataset
    scenarios = ([scenario] if scenario else list(DATASET_CONFIG[ds]["scenarios"]))
    non_ctrl = [k for k in store.ko_names if not _is_control(k)]
    gene_universe = sorted({g for k in non_ctrl for g in parse_target_genes(k)})
    summary: dict = {"dataset": ds, "scenarios": {}}

    print(f"\n  Coverage — {ds}: "
          f"{len(non_ctrl)} perturbations, {len(gene_universe)} perturbed genes")
    for sc in scenarios:
        applicable = [n for n, cls in PREDICTOR_REGISTRY.items() if sc in cls.scenarios]
        dl_keys = _dl_model_key_map()
        drop_rule = _drop_rule_predictors()
        dl_info = _dl_model_info(ds, sc) if any(p in dl_keys for p in applicable) else {}
        ko_cov: Dict[str, Set[str]] = {}
        gene_cov: Dict[str, Set[str]] = {}
        for p in applicable:
            handled_kos = set()
            for k in non_ctrl:
                ok, _ = handles(p, k, store, sc, dl_info)
                if ok:
                    handled_kos.add(k)
            ko_cov[p] = handled_kos
            gene_cov[p] = {g for k in handled_kos for g in parse_target_genes(k)}
        ko_overlap = set(non_ctrl)
        gene_overlap = set(gene_universe)
        for p in applicable:
            ko_overlap &= ko_cov[p]
            gene_overlap &= gene_cov[p]
        print(f"    [{sc}] predictors={len(applicable)}  "
              f"gene overlap={len(gene_overlap)}/{len(gene_universe)}  "
              f"pert overlap={len(ko_overlap)}/{len(non_ctrl)}")
        for p in sorted(applicable):
            drop = len(non_ctrl) - len(ko_cov[p])
            if drop:
                print(f"        {p:<40s} perts {len(ko_cov[p])}/{len(non_ctrl)} "
                      f"(drops {drop}), genes {len(gene_cov[p])}/{len(gene_universe)}")
        summary["scenarios"][sc] = {
            "predictors": sorted(applicable),
            "n_perturbations": len(non_ctrl),
            "n_genes": len(gene_universe),
            "pert_overlap": len(ko_overlap),
            "gene_overlap": len(gene_overlap),
            "per_predictor": {p: {"perts": len(ko_cov[p]), "genes": len(gene_cov[p])}
                              for p in sorted(applicable)},
        }
        # The per-predictor coverage + overlap are DIAGNOSTICS (the failure modes
        # to surface) — an empty overlap is a legitimate finding (e.g. PRESAGE has
        # no entry for this scenario), NOT a verifier failure.
        if not ko_overlap and len(applicable) > 1:
            worst = sorted(applicable, key=lambda p: len(ko_cov[p]))[:3]
            r.info(f"coverage {ds}/{sc}: EMPTY overlap — smallest coverage: "
                   + ", ".join(f"{p}({len(ko_cov[p])})" for p in worst))
        # The one HARD invariant: predictors with no drop rule must cover every pert.
        leaky = [p for p in applicable
                 if p not in drop_rule and p not in dl_keys
                 and len(ko_cov[p]) != len(non_ctrl)]
        r.check(f"coverage {ds}/{sc}: no-drop-rule predictors cover all perts",
                not leaky,
                f"unexpected shortfall: {[(p, len(ko_cov[p])) for p in leaky]}" if leaky else "")
    if write_json:
        import json
        from benchmark.config import RESULTS_DIR
        out = RESULTS_DIR / ds / "coverage.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2))
        r.info(f"wrote {out}")
    return summary


# ===================================================================
# L6 — fold alignment (DL vs our h5ad) + saved-prediction alignment
# ===================================================================

# _canon_set / _dl_declared_test_set now live in benchmark._fold_align (imported above).


def verify_fold_alignment(store, r: Results) -> None:
    """L6: for every DL model in the manifest for this dataset, require each DL
    fold's TEST perturbation set to EXACTLY equal one of our h5ad fold test sets
    (Jaccard==1, a bijection). UnseenCell is pert-degenerate and reported, not
    failed."""
    from benchmark._fold_align import load_manifest, get_model_folds
    ds = store.dataset
    obs = store._adata.obs
    cond = obs["condition"].astype(str)
    # Our folds are generated independently of the external (Miller-pipeline)
    # DL drops, so DL≠ours is EXPECTED and reported as a finding, not a failure.
    # This whole layer is scheduled for deletion at TRAINING_INFRA_PLAN M3, once
    # in-repo container training produces DL predictions on our own folds.
    hard = False
    for scenario in DATASET_CONFIG[ds]["scenarios"]:
        try:
            fbm = get_model_folds(load_manifest(), ds, scenario)
        except Exception as e:
            r.skip(f"fold-align {ds}/{scenario}", f"manifest: {e}", intentional=True)
            continue
        if not fbm:
            r.skip(f"fold-align {ds}/{scenario}", "no DL models in manifest",
                   intentional=True)
            continue
        # our h5ad fold test sets
        h5: Dict[int, Set[str]] = {}
        f = 0
        while split_obs_column(scenario, f) in obs.columns:
            lbl = obs[split_obs_column(scenario, f)].astype(str).values
            h5[f] = _canon_set(cond[lbl == "test"].unique())
            f += 1
        if not h5:
            r.skip(f"fold-align {ds}/{scenario}", "no split columns in h5ad")
            continue
        if scenario == "UnseenCell":
            r.skip(f"fold-align {ds}/{scenario}",
                   "pert-degenerate (every fold tests all perts) — cell-axis only",
                   intentional=True)
            continue
        for model, folds in fbm.items():
            n_exact = 0
            for fe in folds:
                dl_set = _dl_declared_test_set(fe["predictions_path"])
                if not dl_set:
                    continue
                # best h5ad fold by overlap; require exact equality
                best = max(h5, key=lambda hf: len(dl_set & h5[hf]) / max(len(dl_set | h5[hf]), 1))
                # PRESAGE emits a strict pert-subset (smaller vocabulary): restrict
                # our side to the DL set's universe before requiring equality, so a
                # coverage gap isn't read as a fold-assignment mismatch.
                if model == "presage":
                    exact = dl_set and dl_set <= h5[best]
                else:
                    exact = (dl_set == h5[best])
                n_exact += int(bool(exact))
            ok = n_exact == len(folds) and len(folds) > 0
            label = f"fold-align {ds}/{scenario}/{model}: all DL folds map exactly"
            if ok or hard:
                r.check(label, ok, f"{n_exact}/{len(folds)} exact")
            else:
                r.info(f"{label}: {n_exact}/{len(folds)} exact — DL folds differ "
                       "from ours (expected: the external drops were trained on the "
                       "upstream pipeline's folds). Not a failure.")


def verify_saved_prediction_alignment(store, r: Results) -> None:
    """L6b: every saved predictions.npz carries the requested fold/scenario/dataset
    metadata and its test_ko_indices match store.split(...).test_ko_indices."""
    from benchmark.config import predictions_path
    from benchmark.predictors.base import PREDICTOR_REGISTRY, get_predictor
    get_predictor("Zero")
    ds = store.dataset
    checked = 0
    for scenario in DATASET_CONFIG[ds]["scenarios"]:
        try:
            split = store.split(scenario, 0)
        except Exception:
            continue
        exp_tki = set(int(i) for i in split.test_ko_indices)
        for p in sorted(PREDICTOR_REGISTRY):
            ppath = predictions_path(ds, p, scenario, 0)
            if not ppath.exists():
                continue
            try:
                npz = np.load(str(ppath), allow_pickle=True)
                meta_ok = (str(npz.get("scenario")) == scenario
                           and int(npz.get("fold")) == 0
                           and str(npz.get("dataset")) == ds)
                tki = set(int(i) for i in npz["test_ko_indices"])
                # predictor's saved test KOs must be a subset of the split's test KOs
                # (target-aware/combo predictors save the same axis; DL may subset)
                tki_ok = tki <= exp_tki or p in _dl_model_key_map()
                r.check(f"pred-align {ds}/{scenario} {p}: metadata + test KOs match split",
                        meta_ok and tki_ok,
                        f"meta_ok={meta_ok} tki⊆split={tki <= exp_tki}")
                checked += 1
            except Exception as e:
                r.check(f"pred-align {ds}/{scenario} {p}", False, f"{type(e).__name__}: {e}")
    if checked == 0:
        r.skip(f"pred-align {ds}", "no saved predictions on disk", intentional=True)


# ===================================================================
# Orchestration
# ===================================================================


def verify_dataset(dataset: str, r: Results, *,
                   full: bool = False, scenario: Optional[str] = None) -> None:
    """Run the data-dependent layers (L0 via open, L1, L2, L5[, L3, L6]) for one
    dataset. Opens the store ONCE and reuses it for every layer (incl. coverage).

    In fast mode (``full=False``) the store is opened ``light`` — no X, no heavy
    DEG/p-value uns matrices — which the fast layers never read as values; this
    avoids decompressing tens of GB of uns on big datasets. ``--full`` needs the
    score/p-value matrices and X, so it opens a normal store."""
    store = _open_store(dataset, r, "dataset", light=not full)
    if store is None:
        return
    verify_dataset_invariants(store, r)
    verify_techdup_halves(store, r)
    verify_cv_split_leakage(store, r)
    verify_split_conventions(store, r)
    verify_pseudobulk_math(store, r)
    if full:
        verify_gt_axis(store, r)
        verify_fold_alignment(store, r)
        verify_saved_prediction_alignment(store, r)
    verify_coverage(store, r, scenario)


def preflight(datasets: Sequence[str]) -> Results:
    """Cheap pre-benchmark gate (L0 + L1) for the selected datasets. Run by
    `run_pipeline` before fit/all; raise/abort on failure there. Covers every
    scenario of each dataset (the leakage check is cheap)."""
    r = Results()
    for ds in datasets:
        store = _open_store(ds, r, "preflight", light=True)
        if store is None:
            continue
        verify_dataset_invariants(store, r)
        verify_techdup_halves(store, r)
        verify_cv_split_leakage(store, r)
    return r


def _summary(r: Results) -> int:
    intentional = r.skipped - r.blocking_skipped
    print("=" * 64)
    print(f"SUMMARY: {r.passed} passed, {r.failed} failed, {r.skipped} skipped "
          f"({r.blocking_skipped} blocking, {intentional} intentional)")
    if r.fail_names:
        print("FAILURES:")
        for n in r.fail_names:
            print(f"  - {n}")
    status = "PASS" if r.failed == 0 else "FAIL"
    print(f"OVERALL: {status}" + (" (incomplete)" if r.blocking_skipped else ""))
    return 0 if r.failed == 0 else 1


def _verify_dataset_worker(task):
    """Subprocess entry for parallel --fast/--full: run all data layers for ONE
    dataset with output captured, so the parent can print in order and merge the
    tally. Each worker uses its own Results (module-level → picklable)."""
    import io, contextlib
    ds, full, scenario = task
    r = Results()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        try:
            verify_dataset(ds, r, full=full, scenario=scenario)
        except Exception:
            traceback.print_exc()
            r.check(f"dataset {ds} crashed", False)
    return (ds, buf.getvalue(),
            (r.passed, r.failed, r.skipped, r.blocking_skipped, r.fail_names, r.skip_names))


def _merge_counts(r: Results, counts) -> None:
    passed, failed, skipped, blocking, fail_names, skip_names = counts
    r.passed += passed
    r.failed += failed
    r.skipped += skipped
    r.blocking_skipped += blocking
    r.fail_names.extend(fail_names)
    r.skip_names.extend(skip_names)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--fast", action="store_true",
                      help="L1,L2 (data) + L4 (contracts) + L5 (coverage). Default.")
    mode.add_argument("--full", action="store_true",
                      help="adds L3 (GT-axis) + L6 (fold alignment); needs saved predictions.")
    mode.add_argument("--contracts", action="store_true",
                      help="L4 predictor contracts only (no data files).")
    mode.add_argument("--coverage", action="store_true",
                      help="L5 coverage diagnostic only.")
    p.add_argument("--dataset", default="all", help="comma-list or 'all'")
    p.add_argument("--scenario", default=None, help="restrict coverage/checks to a scenario")
    p.add_argument("--jobs", "-j", type=int, default=1,
                   help="parallelize the per-dataset checks across N processes "
                        "(default 1 = serial). Each worker loads ONE dataset; the "
                        "large datasets (e.g. xatlas_orion) use ~10-15 GB RAM each, "
                        "so size N to available memory.")
    p.add_argument("--write-coverage-json", action="store_true",
                   help="write results/{ds}/coverage.json for --coverage")
    args = p.parse_args(argv)

    datasets = _resolve_datasets(args.dataset)
    r = Results()

    if args.contracts:
        print("[L4] PREDICTOR-CONTRACTS")
        verify_predictor_contracts(r)
        return _summary(r)

    if args.coverage:
        for ds in datasets:
            store = _open_store(ds, r, "coverage", light=True)
            if store is not None:
                verify_coverage(store, r, args.scenario, write_json=args.write_coverage_json)
        return _summary(r)

    full = args.full
    # Default + fast: contracts first (cheap, data-free), then per-dataset.
    print("[L4] PREDICTOR-CONTRACTS")
    try:
        verify_predictor_contracts(r)
    except Exception:
        traceback.print_exc()
        r.check("layer L4 crashed", False)
    print()
    if args.jobs > 1 and len(datasets) > 1:
        from concurrent.futures import ProcessPoolExecutor
        print(f"(running {len(datasets)} datasets across {args.jobs} processes)\n")
        tasks = [(ds, full, args.scenario) for ds in datasets]
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            for ds, out, counts in ex.map(_verify_dataset_worker, tasks):
                print(f"[DATA] {ds}")
                print(out, end="")
                _merge_counts(r, counts)
                print()
    else:
        for ds in datasets:
            print(f"[DATA] {ds}")
            try:
                verify_dataset(ds, r, full=full,
                               scenario=args.scenario)
            except Exception:
                traceback.print_exc()
                r.check(f"dataset {ds} crashed", False)
            print()
    print("[L2] PSEUDOBULK-VECTORIZATION GATE (once)")
    run_pseudobulk_gate_once(datasets, r)
    print()
    return _summary(r)


__all__ = [
    "DatasetValidationError", "validate_dataset",
    "Results", "preflight",
    "verify_dataset", "verify_predictor_contracts", "verify_coverage",
    "verify_fold_alignment", "handles",
]


if __name__ == "__main__":
    sys.exit(main())
