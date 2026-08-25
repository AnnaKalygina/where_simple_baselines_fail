"""Selective h5ad loader for the benchmark package.

`DatasetStore` is the single access layer over `data/{dataset}/{dataset}_processed.h5ad`.
Every consumer (predictors, metrics, BS/CF) goes through this class so that:

  * Gene/KO/bin name alignment is enforced at the API level.
  * Cheap accesses (pseudobulk, DEG arrays, splits) never touch
    the full single-cell `.X` matrix.
  * Expensive accesses (single-cell sampling, full DEG recomputation) are
    explicit: callers must invoke `load_full()` to pull `.X` into memory.

Layout it reads (written by `data/{dataset}/get_data.py`):

  adata.X                                       (sparse log1p)
  adata.layers["counts"]                        raw counts (optional)
  adata.obs["condition" | "cell_type" | "batch" | "tech_dup_split"]
  adata.obs["split_{Scenario}_fold_{N}"]        train/val/test/unassigned
  adata.var["highly_variable"]
  adata.var_names                               gene names (ordered)
  adata.uns["pseudobulk"]:
      ctrl_bulk          (n_bins, n_genes)
      first_half_bulk    (n_bins, n_kos, n_genes)
      second_half_bulk   (n_bins, n_kos, n_genes)
      n_cells_first      (n_bins, n_kos)          first_half cell counts
      n_cells_second     (n_bins, n_kos)          second_half cell counts
      ko_names           (n_kos,)
      bin_names          (n_bins,)
  adata.uns["deg_arrays"]:
      per_pert_weights   (n_bins, n_kos, n_genes)
      deg_mask           (n_bins, n_kos, n_genes)
      deg_directions     (n_bins, n_kos, n_genes)
      ko_names           (n_kos,)
      bin_names          (n_bins,)
  adata.uns["{names,scores,pvals_adj,pvals_unadj,deg_gene}_df_dict_first_half"]
  adata.uns["{names,scores,pvals_adj,pvals_unadj,deg_gene}_df_dict_second_half"]
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Dict, List, Optional

import anndata as ad
import numpy as np

from benchmark.config import BIN_AXIS_SCENARIOS, h5ad_path, split_obs_column
from benchmark.verify import validate_dataset

log = logging.getLogger(__name__)


@dataclass
class SplitInfo:
    """Concrete split info for one (scenario, fold).

    Indices index into `store.ko_names` and `store.bin_names`. Empty arrays
    are valid (e.g. UnseenPert has no bin split, so `train_bin_indices` and
    friends are None).
    """
    train_ko_indices: np.ndarray
    val_ko_indices: np.ndarray
    test_ko_indices: np.ndarray
    train_bin_indices: Optional[np.ndarray] = None
    val_bin_indices: Optional[np.ndarray] = None
    test_bin_indices: Optional[np.ndarray] = None
    # Per-(bin, ko) boolean masks, shape (n_bins, n_kos), in canonical
    # store.bin_names × store.ko_names order. Populated ONLY for UnseenPair, whose
    # held-out unit is a scattered set of (bin, ko) pairs that cannot be expressed
    # as the outer product of bin/ko marginals. None for rectangle scenarios
    # (UnseenPert/UnseenCell/UnseenBoth), where `*_mask_2d()` reconstructs the
    # rectangle from the index marginals instead.
    train_pair_mask: Optional[np.ndarray] = None
    val_pair_mask: Optional[np.ndarray] = None
    test_pair_mask: Optional[np.ndarray] = None

    def train_bin_indices_or_all(self, n_bins: int) -> np.ndarray:
        """Return `self.train_bin_indices` if non-empty; otherwise np.arange(n_bins).

        For scenarios with no bin axis (UnseenPert / UnseenCombo / UnseenDose),
        `train_bin_indices` is None or empty, and every bin is implicitly a
        training bin. Predictors call this instead of re-deriving the default.
        """
        if self.train_bin_indices is not None and len(self.train_bin_indices) > 0:
            return self.train_bin_indices
        return np.arange(n_bins, dtype=np.int64)

    def test_bin_indices_or_all(self, n_bins: int) -> np.ndarray:
        """Return `self.test_bin_indices` if non-empty; otherwise np.arange(n_bins).

        Symmetric with `train_bin_indices_or_all`: for scenarios with no bin axis
        (UnseenPert / UnseenCombo / UnseenDose) every bin is implicitly a test bin,
        so the prediction tensor spans all bins. Predictors call this to size/label
        the bin axis of their output rectangle.
        """
        if self.test_bin_indices is not None and len(self.test_bin_indices) > 0:
            return self.test_bin_indices
        return np.arange(n_bins, dtype=np.int64)

    def _rect_mask(self, bin_indices: Optional[np.ndarray],
                   ko_indices: np.ndarray, n_bins: int, n_kos: int) -> np.ndarray:
        m = np.zeros((n_bins, n_kos), dtype=bool)
        bins = (bin_indices if (bin_indices is not None and len(bin_indices) > 0)
                else np.arange(n_bins, dtype=np.int64))
        if len(ko_indices) > 0:
            m[np.ix_(bins, ko_indices)] = True
        return m

    def train_mask_2d(self, n_bins: int, n_kos: int) -> np.ndarray:
        """(n_bins, n_kos) bool mask of TRAIN (bin, ko) cells.

        Returns the stored pair mask for UnseenPair; otherwise the rectangle
        `train_bins × train_kos` (identical to the marginal slicing predictors
        used before), so rectangle scenarios are unchanged.
        """
        if self.train_pair_mask is not None:
            return self.train_pair_mask
        return self._rect_mask(self.train_bin_indices, self.train_ko_indices,
                               n_bins, n_kos)

    def test_mask_2d(self, n_bins: int, n_kos: int) -> np.ndarray:
        """(n_bins, n_kos) bool mask of TEST (bin, ko) cells (see `train_mask_2d`)."""
        if self.test_pair_mask is not None:
            return self.test_pair_mask
        return self._rect_mask(self.test_bin_indices, self.test_ko_indices,
                               n_bins, n_kos)

    def train_cells(self, n_bins: int, n_kos: int):
        """Return (bin_idx, ko_idx) int arrays of the TRAIN (bin, ko) cells.

        The blessed, leak-free enumeration of training cells — literally
        ``np.where(train_mask_2d(...))``. Use this (or ``train_mask_2d``) to source
        training cells; do NOT outer-product ``train_bin_indices × train_ko_indices``
        (see ``train_marginal_rect``), which over-includes the held-out test corner
        for the non-rectangular cell-holdout splits.
        """
        return np.where(self.train_mask_2d(n_bins, n_kos))

    def train_marginal_rect(self, n_bins: int):
        """Return ``(train_bin_indices_or_all, train_ko_indices)`` for use ONLY as a
        rectangular training-cell selector ``deltas[bins][:, kos]``.

        Guarded: raises if a ``train_pair_mask`` is present. For the non-rectangular
        splits (UnseenPair / UnseenBoth) the KO and bin marginals overlap with the
        test marginals, so their outer product manufactures the held-out test corner
        — see the ``split()`` docstring. There the rectangle is a leak; use
        ``train_cells`` / ``train_mask_2d`` instead. For the rectangle scenarios
        (UnseenPert / UnseenCell) the marginals are exact, so this is safe and cheap.
        """
        if self.train_pair_mask is not None:
            raise ValueError(
                "train_marginal_rect: non-rectangular split (train_pair_mask present); "
                "the train_bins × train_kos rectangle would include held-out test "
                "cells. Use train_cells(n_bins, n_kos) or train_mask_2d(...) instead."
            )
        return self.train_bin_indices_or_all(n_bins), self.train_ko_indices


def _membership_by_label(values, name_to_idx: Dict[str, int], labels):
    """Aggregate parallel (value, split-label) arrays into (train, val, test) index
    sets: an index is in a split iff ANY cell with that value carries that label.
    Shared by `split()` for both the KO (condition) and bin (cell_type) axes."""
    out = {"train": set(), "val": set(), "test": set()}
    for v, lbl in zip(values, labels):
        i = name_to_idx.get(v)
        if i is None:
            continue
        bucket = out.get(lbl)
        if bucket is not None:
            bucket.add(i)
    return out["train"], out["val"], out["test"]


# uns slots the --fast verifier never needs as VALUES. The per-(bin,ko,gene)
# p-value matrices are untouched in fast mode; the score matrices are only
# shape-checked. Skipping / short-circuiting them avoids decompressing tens of GB
# of uns at store-open time (e.g. xatlas_orion: ~17 GB of the ~27 GB total).
_LIGHT_UNS_SKIP = (
    "pvals_adj_matrix_first_half", "pvals_adj_matrix_second_half",
    "pvals_unadj_matrix_first_half", "pvals_unadj_matrix_second_half",
)
_LIGHT_UNS_SHAPE_ONLY = (
    "scores_matrix_first_half", "scores_matrix_second_half",
)


def _read_light_h5ad(path: str) -> ad.AnnData:
    """Read obs/var + only the light uns slots (no X, no heavy p-value matrices)
    for fast validation / leakage / coverage.

    pseudobulk + deg_arrays are read in full (L0 no-NaN scan, L2 pseudobulk math);
    the score matrices become zero-memory shape proxies so shape checks still pass;
    the p-value matrices (never read in --fast) are skipped entirely. On big
    datasets this turns a ~27 GB uns decompression into ~10 GB.
    """
    import h5py
    from anndata.io import read_elem
    with h5py.File(str(path), "r") as f:
        obs = read_elem(f["obs"])
        var = read_elem(f["var"])
        # Marker so validators can tell a light store from a real one: no X and
        # no layers are loaded here, so checks that need either must defer to
        # --full rather than report a false absence.
        uns: Dict = {"_vcr_light_store": True}
        if "uns" in f:
            for k in f["uns"].keys():
                if k in _LIGHT_UNS_SKIP:
                    continue
                if k in _LIGHT_UNS_SHAPE_ONLY:
                    uns[k] = np.broadcast_to(np.float32(0.0), f["uns"][k].shape)
                    continue
                uns[k] = read_elem(f["uns"][k])
    return ad.AnnData(obs=obs, var=var, uns=uns)


class DatasetStore:
    """Selective h5ad loader.

    By default lazy: opens the file in backed mode and only materializes the
    slots the caller asks for. Call `load_full()` to bring `.X` into memory
    (needed for single-cell access — synthetic tech-dup, e-distance with
    sampled cells).
    """

    def __init__(self, dataset: str, *, lazy: bool = True, validate: bool = True,
                 light: bool = False):
        self.dataset = dataset
        self.path = Path(h5ad_path(dataset))
        if not self.path.exists():
            raise FileNotFoundError(
                f"DatasetStore: h5ad not found at {self.path}. "
                f"Run `python data/{dataset}/get_data.py` to produce it."
            )
        self._lazy = lazy and not light
        self._adata: Optional[ad.AnnData] = None
        if light:
            # fast verifier path: obs/var + light uns only, no X (see _read_light_h5ad)
            self._adata = _read_light_h5ad(str(self.path))
        elif lazy:
            self._adata = ad.read_h5ad(str(self.path), backed="r")
        else:
            self._adata = ad.read_h5ad(str(self.path))
        self.validation_warnings: List[str] = []
        if validate:
            self.validation_warnings = validate_dataset(self._adata, dataset)

    # -----------------------------------------------------------------
    # AnnData passthrough
    # -----------------------------------------------------------------

    @property
    def adata(self) -> ad.AnnData:
        return self._adata

    def load_full(self) -> ad.AnnData:
        """Force-load `.X` into memory (no-op if already loaded)."""
        if self._adata.X is None:
            # light store (opened without X) — re-read the full file from disk
            log.info("DatasetStore[%s]: re-reading full file (light store had no .X)",
                     self.dataset)
            self._adata = ad.read_h5ad(str(self.path))
            self._lazy = False
            return self._adata
        if self._adata.isbacked:
            log.info("DatasetStore[%s]: loading full .X into memory", self.dataset)
            self._adata = self._adata.to_memory()
            self._lazy = False
        return self._adata

    # -----------------------------------------------------------------
    # Names (cheap — from var_names / pseudobulk uns slot)
    # -----------------------------------------------------------------

    @cached_property
    def gene_names(self) -> List[str]:
        return list(self._adata.var_names)

    @cached_property
    def gene_name_set(self) -> set:
        """Membership set over gene_names — built once, reused by per-ko coverage
        checks (verify.handles) that would otherwise rebuild it on every call."""
        return set(self.gene_names)

    @cached_property
    def gene_to_index(self) -> Dict[str, int]:
        """gene_name -> column index, built once (see gene_name_set)."""
        return {g: i for i, g in enumerate(self.gene_names)}

    @cached_property
    def n_genes(self) -> int:
        return self._adata.n_vars

    def _pb(self) -> Dict:
        if "pseudobulk" not in self._adata.uns:
            raise KeyError(
                f"DatasetStore[{self.dataset}]: adata.uns['pseudobulk'] is missing. "
                f"Re-run data/{self.dataset}/get_data.py to populate it."
            )
        return dict(self._adata.uns["pseudobulk"])

    @cached_property
    def ko_names(self) -> List[str]:
        return [str(k) for k in self._pb()["ko_names"]]

    @cached_property
    def ko_name_set(self) -> set:
        """Membership set over ko_names — built once, reused by per-ko coverage
        checks (verify.handles) that would otherwise rebuild it on every call."""
        return set(self.ko_names)

    @cached_property
    def bin_names(self) -> List[str]:
        return [str(b) for b in self._pb()["bin_names"]]

    @cached_property
    def n_kos(self) -> int:
        return len(self.ko_names)

    @cached_property
    def n_bins(self) -> int:
        return len(self.bin_names)

    # -----------------------------------------------------------------
    # Pseudobulk arrays
    # -----------------------------------------------------------------

    @cached_property
    def ctrl_bulk(self) -> np.ndarray:
        """(n_bins, n_genes) mean of ALL control cells per bin."""
        return np.asarray(self._pb()["ctrl_bulk"], dtype=np.float32)

    @cached_property
    def first_half_bulk(self) -> np.ndarray:
        """(n_bins, n_kos, n_genes) mean of first_half cells per (bin, ko)."""
        return np.asarray(self._pb()["first_half_bulk"], dtype=np.float32)

    @cached_property
    def second_half_bulk(self) -> np.ndarray:
        """(n_bins, n_kos, n_genes) mean of second_half cells per (bin, ko)."""
        return np.asarray(self._pb()["second_half_bulk"], dtype=np.float32)

    def _n_cells(self, half: str) -> np.ndarray:
        """(n_bins, n_kos) int — per-(bin, ko) cell count for a tech-dup half."""
        key = f"n_cells_{half}"
        pb = self._pb()
        if key not in pb:
            raise KeyError(
                f"DatasetStore[{self.dataset}]: adata.uns['pseudobulk']['{key}'] "
                f"is missing. This store predates the all-cell-target change — re-run "
                f"data/{self.dataset}/get_data.py to repopulate the pseudobulk."
            )
        return np.asarray(pb[key], dtype=np.int64)

    @cached_property
    def n_cells_first(self) -> np.ndarray:
        """(n_bins, n_kos) int — first_half cell count per (bin, ko)."""
        return self._n_cells("first")

    @cached_property
    def n_cells_second(self) -> np.ndarray:
        """(n_bins, n_kos) int — second_half cell count per (bin, ko)."""
        return self._n_cells("second")

    @cached_property
    def all_bulk(self) -> np.ndarray:
        """(n_bins, n_kos, n_genes) mean of ALL perturbed cells (both halves) per (bin, ko).

        Exact count-weighted average of the two half-means:
        ``all = (n1*first + n2*second) / (n1 + n2)``. Reconstructed lazily from
        the stored half-means + per-half counts, so no third heavy array is
        persisted. Computed in float64 then cast to float32.
        """
        n1 = self.n_cells_first.astype(np.float64)[:, :, None]
        n2 = self.n_cells_second.astype(np.float64)[:, :, None]
        num = (n1 * self.first_half_bulk.astype(np.float64)
               + n2 * self.second_half_bulk.astype(np.float64))
        total = n1 + n2
        out = np.divide(num, total, out=np.zeros_like(num), where=total > 0)
        return out.astype(np.float32)

    @cached_property
    def all_deltas(self) -> np.ndarray:
        """(n_bins, n_kos, n_genes) all-cell training target = all_bulk - ctrl_bulk.

        This is the target predictors should TRAIN on (matches Miller: all cells
        of the training perturbations). Distinct from the evaluation ground truth
        ``first_half_deltas``.
        """
        return self.all_bulk - self.ctrl_bulk[:, None, :]

    @cached_property
    def first_half_deltas(self) -> np.ndarray:
        """(n_bins, n_kos, n_genes) evaluation ground-truth deltas = first_half - ctrl_bulk.

        Computed lazily from the absolute pseudobulks. This is the noisy-half GT
        used to SCORE predictions (matches Miller's
        ``technical_duplicate_first_half_baseline``). Predictors train on
        ``all_deltas``, not this.
        """
        return self.first_half_bulk - self.ctrl_bulk[:, None, :]

    @cached_property
    def second_half_deltas(self) -> np.ndarray:
        """(n_bins, n_kos, n_genes) positive-control deltas = second_half - ctrl_bulk."""
        return self.second_half_bulk - self.ctrl_bulk[:, None, :]

    # -----------------------------------------------------------------
    # DEG arrays (derived, var_names-aligned)
    # -----------------------------------------------------------------

    def _deg_arr(self) -> Dict:
        if "deg_arrays" not in self._adata.uns:
            raise KeyError(
                f"DatasetStore[{self.dataset}]: adata.uns['deg_arrays'] is missing."
            )
        return dict(self._adata.uns["deg_arrays"])

    @cached_property
    def per_pert_weights(self) -> np.ndarray:
        return np.asarray(self._deg_arr()["per_pert_weights"], dtype=np.float32)

    @cached_property
    def deg_mask(self) -> np.ndarray:
        return np.asarray(self._deg_arr()["deg_mask"], dtype=bool)

    @cached_property
    def deg_directions(self) -> np.ndarray:
        return np.asarray(self._deg_arr()["deg_directions"], dtype=np.int8)

    # -----------------------------------------------------------------
    # Per-pert DEG dicts (literature format — both halves)
    # -----------------------------------------------------------------

    def deg_dict(self, half: str, kind: str = "names") -> Dict[str, List]:
        """Return one of the per-pert DEG dicts.

        half: 'first_half' | 'second_half'
        kind: 'names' | 'scores' | 'pvals_adj' | 'pvals_unadj' | 'deg_gene'
        """
        assert half in ("first_half", "second_half"), half
        if kind == "deg_gene":
            key = f"deg_gene_dict_{half}"
        else:
            key = f"{kind}_df_dict_{half}"
        return dict(self._adata.uns.get(key, {}))

    # -----------------------------------------------------------------
    # Splits (from obs columns)
    # -----------------------------------------------------------------

    def split(self, scenario: str, fold: int) -> SplitInfo:
        """Derive train/val/test KO + bin index arrays from obs columns.

        The h5ad obs column `split_{scenario}_fold_{fold}` carries one of
        'train'/'val'/'test'/'unassigned' per CELL. We aggregate to KO-level
        and bin-level memberships:

          * KO indices: a KO is in `train` (resp. val/test) if any cell with
            that condition is labeled that way. These marginals are NOT
            mutually exclusive for the non-rectangular scenarios — e.g. under
            UnseenBoth a held-out test perturbation is also `train` in the
            non-test cell types, so it appears in both `train_kos` and
            `test_kos`. The outer product `train_bins × train_kos` therefore
            OVER-includes the held-out corner; predictors must instead consume
            the explicit per-(bin, ko) masks via `train_mask_2d`/`test_mask_2d`
            (loaded from uns for UnseenPair/UnseenBoth). The marginals are kept
            only for the rectangle scenarios and for the predict-time/eval test
            rectangle.
          * Bin indices: same idea, derived from `cell_type` membership in
            train/val/test cells. For single-bin scenarios (UnseenPert,
            UnseenCombo, UnseenDose), all three are None.
        """
        col = split_obs_column(scenario, fold)
        if col not in self._adata.obs.columns:
            raise KeyError(
                f"DatasetStore[{self.dataset}]: obs column '{col}' missing. "
                f"Re-run data/{self.dataset}/get_data.py to assign splits."
            )

        # Build KO membership per split (an index is train/val/test if ANY cell
        # with that condition carries that label).
        ko_to_idx = {k: i for i, k in enumerate(self.ko_names)}
        conds = self._adata.obs["condition"].astype(str).values
        labels = self._adata.obs[col].astype(str).values
        train_kos, val_kos, test_kos = _membership_by_label(conds, ko_to_idx, labels)

        # Bin membership only for scenarios that rotate the cell-type axis.
        train_bins = val_bins = test_bins = None
        if scenario in BIN_AXIS_SCENARIOS \
                and self.n_bins > 1 and "cell_type" in self._adata.obs.columns:
            bin_to_idx = {b: i for i, b in enumerate(self.bin_names)}
            bins_arr = self._adata.obs["cell_type"].astype(str).values
            tr_b, va_b, te_b = _membership_by_label(bins_arr, bin_to_idx, labels)
            train_bins = np.array(sorted(tr_b), dtype=int)
            val_bins = np.array(sorted(va_b), dtype=int)
            test_bins = np.array(sorted(te_b), dtype=int)

        # Non-rectangular scenarios (UnseenPair, UnseenBoth) hold out a (bin, ko) set
        # that is NOT the outer product of the marginals above; when per-fold masks
        # were stored by get_data, load them so predictors train on the exact cells
        # rather than the leaky train_bins × train_kos rectangle. Gate on mask
        # presence (not a hardcoded scenario name) so any masked scenario works.
        train_pair_mask = val_pair_mask = test_pair_mask = None
        if f"{scenario}_fold_{fold}_train_mask" in self._adata.uns:
            train_pair_mask, val_pair_mask, test_pair_mask = \
                self._load_pair_masks(scenario, fold)

        return SplitInfo(
            train_ko_indices=np.array(sorted(train_kos), dtype=int),
            val_ko_indices=np.array(sorted(val_kos), dtype=int),
            test_ko_indices=np.array(sorted(test_kos), dtype=int),
            train_bin_indices=train_bins,
            val_bin_indices=val_bins,
            test_bin_indices=test_bins,
            train_pair_mask=train_pair_mask,
            val_pair_mask=val_pair_mask,
            test_pair_mask=test_pair_mask,
        )

    def _load_pair_masks(self, scenario: str, fold: int):
        """Return (train, val, test) (n_bins, n_kos) bool masks for a non-rectangular
        scenario fold, reindexed from the STORED (bin_names, single_conditions) order
        to the canonical `store.bin_names` × `store.ko_names` order.

        The masks are written by the `data/_utils.assign_split_folds_unseen_*`
        helpers (via `_store_pair_masks`) to
        `uns['{scenario}_fold_{N}_{train,val,test}_mask']` alongside their axis labels
        (`..._bin_names`, `..._single_conditions`); UnseenPair and UnseenBoth share
        this convention. Per the gene-alignment policy we verify names and RAISE if
        any store ko/bin is absent from the stored axes; stored entries not in the
        store (e.g. mcfaline23's mask carries single_conditions with no pseudobulk
        delta) are simply not referenced.
        """
        uns = self._adata.uns

        def _decode(arr):
            return [x.decode() if isinstance(x, (bytes, np.bytes_)) else str(x)
                    for x in np.asarray(arr).ravel()]

        bkey = f"{scenario}_fold_{fold}_bin_names"
        kkey = f"{scenario}_fold_{fold}_single_conditions"
        mkeys = {s: f"{scenario}_fold_{fold}_{s}_mask"
                 for s in ("train", "val", "test")}
        missing = [k for k in (bkey, kkey, *mkeys.values()) if k not in uns]
        if missing:
            raise KeyError(
                f"DatasetStore[{self.dataset}]: {scenario} fold {fold} masks missing "
                f"from uns: {missing}. Re-run data/{self.dataset}/get_data.py."
            )

        stored_bins = _decode(uns[bkey])
        stored_kos = _decode(uns[kkey])
        sb_idx = {b: i for i, b in enumerate(stored_bins)}
        sk_idx = {k: i for i, k in enumerate(stored_kos)}

        def _map(names, lookup, kind):
            out = []
            for nm in names:
                if nm not in lookup:
                    raise ValueError(
                        f"DatasetStore[{self.dataset}]: {scenario} fold {fold}: store "
                        f"{kind} {nm!r} absent from stored mask axes — mask/store "
                        f"misalignment, refusing to guess."
                    )
                out.append(lookup[nm])
            return np.asarray(out, dtype=int)

        bin_map = _map(self.bin_names, sb_idx, "bin")
        ko_map = _map(self.ko_names, sk_idx, "ko")

        masks = []
        for s in ("train", "val", "test"):
            stored = np.asarray(uns[mkeys[s]]).astype(bool)
            masks.append(stored[np.ix_(bin_map, ko_map)])
        return tuple(masks)
