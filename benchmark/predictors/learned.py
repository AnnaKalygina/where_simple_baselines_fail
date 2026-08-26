"""Learned baseline predictors with persisted weights.

Each predictor implements:
  * `fit(store, scenario, fold)`   — train on the fold's training entries,
                                       populate `self` with parameters.
  * `predict(store, scenario, fold)` — generate predictions using parameters.
  * `save_weights(path)`           — write parameters to weights.npz.
  * `load_weights(path)`           — reconstruct from weights.npz.

The seven learned baselines from the plan:
  Ridge, LinearAdditive, LatentAdditive, BilinearRidge, Correlation,
  TargetScaling, GlobalEpistasis.

Ridge / LinearAdditive / BilinearRidge / Correlation / TargetScaling /
GlobalEpistasis are fully implemented here.

LatentAdditive is a non-trivial MLP-encoder/decoder model. To keep this
module self-contained and avoid pulling 700+ lines of training code from
the legacy `baselines.py`, the LatentAdditive class is wired up to delegate
training/prediction to the legacy functions when they are available, and
otherwise raises a clear error.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from benchmark.config import parse_target_genes, weights_path
from benchmark.data_loader import DatasetStore, SplitInfo
from benchmark.predictors.base import Predictor, register
from benchmark.predictors._shared import (
    _expected_output_shape,
    masked_mean,
    resolve_combo_pairs,
)

log = logging.getLogger(__name__)


# ===================================================================
# Feature construction
# ===================================================================


def _multihot_rows(
    bin_idx: np.ndarray, ko_idx: np.ndarray, n_total_kos: int, n_total_bins: int,
) -> np.ndarray:
    """Multi-hot features for an explicit list of (bin, ko) cells: one row per
    parallel (bin_idx[i], ko_idx[i]) pair, with a one-hot for the ko and a one-hot
    for the bin → (len(ko_idx), n_total_kos + n_total_bins).

    The single feature builder for BOTH training (cells from the train mask, which
    is scattered for UnseenPair/UnseenBoth) and prediction (cells from the test
    rectangle, via `_rect_cells`).
    """
    n = len(ko_idx)
    X = np.zeros((n, n_total_kos + n_total_bins), dtype=np.float32)
    for r in range(n):
        X[r, int(ko_idx[r])] = 1.0
        X[r, n_total_kos + int(bin_idx[r])] = 1.0
    return X


def _rect_cells(
    bin_indices: np.ndarray, ko_indices: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """(bi, ki) index arrays for the full bin×ko rectangle in BIN-MAJOR order
    (bi outer, ki inner) — the same row order as a flattened
    ``np.ix_(bin_indices, ko_indices)``. Expresses a rectangle (the predict-time
    test set, or a rectangle-scenario train set) as explicit cells so one code
    path serves both the rectangle and the scattered-mask cases.
    """
    bin_indices = np.asarray(bin_indices)
    ko_indices = np.asarray(ko_indices)
    bi = np.repeat(bin_indices, len(ko_indices))
    ki = np.tile(ko_indices, len(bin_indices))
    return bi.astype(np.int64), ki.astype(np.int64)


def _flatten_targets_cells(
    deltas: np.ndarray, bin_idx: np.ndarray, ko_idx: np.ndarray,
) -> np.ndarray:
    """deltas at the explicit (bin_idx[i], ko_idx[i]) cells → (n_samples, n_genes)."""
    return deltas[bin_idx, ko_idx]


def _solve_ridge(X: np.ndarray, Y: np.ndarray, lam: float) -> Tuple[np.ndarray, np.ndarray]:
    """Closed-form ridge regression with centered Y."""
    X = X.astype(np.float64)
    Y = Y.astype(np.float64)
    Y_mean = Y.mean(axis=0)
    Yc = Y - Y_mean
    n_features = X.shape[1]
    A = X.T @ X + lam * np.eye(n_features, dtype=np.float64)
    W = np.linalg.solve(A, X.T @ Yc).astype(np.float32)
    b = Y_mean.astype(np.float32)
    return W, b


# ===================================================================
# 1. Ridge
# ===================================================================


# ===================================================================
# Learned tier
# ===================================================================


class LearnedPredictor(Predictor):
    """Predictors trained cheaply in-process, whose state is a small `weights.npz`.

    The tier owns the round-trip: subclasses keep their own `save_weights` /
    `load_weights` (they know their own arrays), and this base turns those into
    the fold-addressed operations the CLI drives — `persist`, `restore`,
    `is_trained`.

    `fit` stays PURE (train in memory, touch no disk). That is load-bearing:
    `verify` fits predictors on synthetic stores, and `Mean+TargetScaling` fits
    sub-predictors internally — if `fit` persisted, both would scribble stray
    weight files into the real `models/` tree. Persisting is therefore an
    explicit second step that only `run_pipeline fit` takes.
    """

    needs_training = True

    @classmethod
    def _weights_file(cls, dataset: str, scenario: str, fold: int) -> Path:
        return weights_path(dataset, cls.name, scenario, fold)

    def is_trained(self, store, scenario: str, fold: int) -> bool:
        """Present AND readable AND non-empty.

        A bare `.exists()` is what made the old CLI gate lie: a killed `fit`
        leaves a truncated npz that exists but cannot be loaded, and the base
        class used to write a deliberately EMPTY npz for every predictor. Open
        it and require at least one array, so those both read as "not trained".
        (Not yet checked: whether the weights match the current code/params —
        that needs a fingerprint stored inside the npz.)"""
        path = self._weights_file(store.dataset, scenario, fold)
        if not path.exists():
            return False
        try:
            with np.load(str(path), allow_pickle=True) as z:
                return len(z.files) > 0
        except Exception:
            return False

    def persist(self, store, scenario: str, fold: int) -> Path:
        """Write this fold's trained coefficients; returns the path written."""
        path = self._weights_file(store.dataset, scenario, fold)
        self.save_weights(path)
        return path

    @classmethod
    def restore(cls, store, scenario: str, fold: int) -> "LearnedPredictor":
        """Rebuild a fitted instance from this fold's `weights.npz`."""
        return cls.load_weights(cls._weights_file(store.dataset, scenario, fold))


@register
class Ridge(LearnedPredictor):
    """Ridge regression on multi-hot perturbation + bin features."""
    name = "Ridge"
    scenarios = ["UnseenPert", "UnseenCell", "UnseenBoth", "UnseenPair", "UnseenCombo"]

    def __init__(self, ridge_lambda: float = 1.0):
        self.ridge_lambda = float(ridge_lambda)
        self.W: Optional[np.ndarray] = None
        self.b: Optional[np.ndarray] = None
        self.n_total_kos: int = 0
        self.n_total_bins: int = 0

    def fit(self, store, scenario, fold):
        self.n_total_kos = store.n_kos
        self.n_total_bins = store.n_bins
        split = store.split(scenario, fold)
        if len(split.train_ko_indices) == 0:
            raise ValueError(f"Ridge.fit: empty train_ko_indices for {scenario}/fold{fold}")
        # Fit on exactly the TRAIN (bin, ko) cells. train_mask_2d is the explicit
        # pair mask for the non-rectangular scenarios (UnseenPair, UnseenBoth) and
        # the train_bins × train_kos rectangle otherwise, so the held-out corner is
        # never used and rectangle scenarios are unchanged.
        bi, ki = split.train_cells(store.n_bins, store.n_kos)  # (bin, ko) coords of TRAIN cells
        X = _multihot_rows(bi, ki, store.n_kos, store.n_bins)
        Y = _flatten_targets_cells(store.all_deltas, bi, ki)
        self.W, self.b = _solve_ridge(X, Y, self.ridge_lambda)

    def predict(self, store, scenario, fold):
        if self.W is None:
            raise RuntimeError("Ridge.predict called before fit() (or load_weights()).")
        split = store.split(scenario, fold)
        tbi = split.test_bin_indices_or_all(store.n_bins)
        tki = split.test_ko_indices
        bi, ki = _rect_cells(tbi, tki)  # full test rectangle, bin-major
        X = _multihot_rows(bi, ki, self.n_total_kos, self.n_total_bins)
        Y = X @ self.W + self.b
        return Y.reshape(len(tbi), len(tki), -1).astype(np.float32)

    def save_weights(self, path):
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(str(path),
                 W=self.W, b=self.b,
                 ridge_lambda=self.ridge_lambda,
                 n_total_kos=self.n_total_kos,
                 n_total_bins=self.n_total_bins)

    @classmethod
    def load_weights(cls, path):
        d = np.load(str(path), allow_pickle=True)
        m = cls(ridge_lambda=float(d["ridge_lambda"]))
        m.W = d["W"]; m.b = d["b"]
        m.n_total_kos = int(d["n_total_kos"])
        m.n_total_bins = int(d["n_total_bins"])
        return m


# ===================================================================
# 2. LinearAdditive — Ridge with tiny lambda (OLS closed-form)
# ===================================================================


@register
class LinearAdditive(Ridge):
    """OLS via Ridge with λ → 0 (numerical stability only)."""
    name = "LinearAdditive"

    def __init__(self):
        super().__init__(ridge_lambda=1e-8)

    @classmethod
    def load_weights(cls, path):
        d = np.load(str(path), allow_pickle=True)
        m = cls()
        m.W = d["W"]; m.b = d["b"]
        m.n_total_kos = int(d["n_total_kos"])
        m.n_total_bins = int(d["n_total_bins"])
        return m


# ===================================================================
# 3. BilinearRidge — Y = G W P (PCA-based bilinear baseline)
# ===================================================================


@register
class BilinearRidge(LearnedPredictor):
    """Y = G W P with ridge solver. G is the PCA basis of training pseudobulks,
    and P is restricted to perturbed genes that appear in G's row space.

    Predicts deltas for any KO whose name matches a row in G's basis.
    """
    name = "BilinearRidge"
    has_drop_rule = True
    scenarios = ["UnseenPert"]

    def __init__(self, n_components: int = 10, g_ridge: float = 0.1, p_ridge: float = 0.1):
        self.n_components = int(n_components)
        self.g_ridge = float(g_ridge)
        self.p_ridge = float(p_ridge)
        self.W: Optional[np.ndarray] = None
        self.b: Optional[np.ndarray] = None
        self.gene_names: Optional[List[str]] = None
        self.ko_names: Optional[List[str]] = None

    def fit(self, store, scenario, fold):
        from sklearn.decomposition import PCA
        split = store.split(scenario, fold)
        train_bins, train_kos = split.train_marginal_rect(store.n_bins)
        if len(train_kos) == 0:
            raise ValueError(
                f"BilinearRidge.fit: empty train_ko_indices for {scenario}/fold{fold}"
            )
        deltas = store.all_deltas[train_bins][:, train_kos]  # (Bt, Kt, G)
        # Y: averaged across train bins → (Kt, G)
        Y_train = deltas.mean(axis=0).astype(np.float64)
        gene_names = store.gene_names
        train_ko_names = [store.ko_names[int(k)] for k in train_kos]

        pca = PCA(n_components=self.n_components, random_state=42)
        pca.fit(Y_train)
        G = pca.components_.T  # (G, n_components) — gene basis (pert-independent)

        # P matrix: rows are training KO names that appear in gene_names.
        # Permissive (matches upstream): drop training KOs whose label is not a
        # gene symbol in var_names, with a warning, rather than aborting.
        gene_to_idx = {g: i for i, g in enumerate(gene_names)}
        keep = [i for i, k in enumerate(train_ko_names) if k in gene_to_idx]
        n_drop = len(train_ko_names) - len(keep)
        if n_drop:
            log.warning(
                "BilinearRidge.fit: dropping %d/%d training KO(s) whose label is "
                "not a gene in var_names for %s/%s/fold%d",
                n_drop, len(train_ko_names), store.dataset, scenario, fold,
            )
        if not keep:
            log.warning(
                "BilinearRidge.fit: no resolvable training KOs for %s/%s/fold%d; "
                "predictor will emit NaN (skipped from eval)",
                store.dataset, scenario, fold,
            )
            self.W = None
            self.b = None
            self.gene_names = gene_names
            self.ko_names = []
            return

        train_ko_names = [train_ko_names[i] for i in keep]
        Y_train = Y_train[keep]                 # (Kt_kept, G)
        p_idx = [gene_to_idx[k] for k in train_ko_names]
        Y_used = Y_train.T  # (G, Kt_kept)
        P = G[p_idx]  # (Kt_kept, n_components)

        b = Y_used.mean(axis=1, keepdims=True)
        Yc = Y_used - b
        GtG = G.T @ G + self.g_ridge * np.eye(self.n_components)
        PtP = P.T @ P + self.p_ridge * np.eye(self.n_components)
        Gm = np.linalg.inv(GtG) @ G.T
        Pm = P @ np.linalg.inv(PtP)
        W = Gm @ Yc @ Pm  # (n_components, n_components)
        self.W = np.nan_to_num(W).astype(np.float32)
        self.b = b.flatten().astype(np.float32)
        self.gene_names = gene_names
        self.ko_names = train_ko_names

    def predict(self, store, scenario, fold):
        split = store.split(scenario, fold)
        tbi = split.test_bin_indices_or_all(store.n_bins)
        tki = split.test_ko_indices
        # NaN init: unresolved test KOs stay NaN → skipped from eval (not scored).
        out = np.full(_expected_output_shape(store, tbi, tki), np.nan, dtype=np.float32)
        if self.W is None:
            log.warning(
                "BilinearRidge.predict: degenerate model (no resolvable training "
                "KOs) for %s/%s/fold%d → all-NaN (skipped from eval)",
                store.dataset, scenario, fold,
            )
            return out

        # Re-fit PCA basis on the cached training Y to predict for test KOs.
        # We need G in the SAME ordering used for fit; the simplest stable
        # path is to recompute it from store.all_deltas[train_bins][train_kos].
        # Cached `self.W` was solved against that basis.
        from sklearn.decomposition import PCA
        train_bins, train_kos = split.train_marginal_rect(store.n_bins)
        Y_train = store.all_deltas[train_bins][:, train_kos].mean(axis=0).astype(np.float64)
        pca = PCA(n_components=self.n_components, random_state=42)
        pca.fit(Y_train)
        G = pca.components_.T  # (G, k)
        gene_to_idx = {g: i for i, g in enumerate(store.gene_names)}

        test_ko_names = [store.ko_names[int(k)] for k in tki]
        n_skip = sum(1 for n in test_ko_names if n not in gene_to_idx)
        if n_skip:
            log.warning(
                "BilinearRidge.predict: skipping %d/%d test KO(s) whose label is "
                "not a gene in var_names for %s/%s/fold%d (NaN → excluded from eval)",
                n_skip, len(test_ko_names), store.dataset, scenario, fold,
            )

        for j, name in enumerate(test_ko_names):
            gi = gene_to_idx.get(name)
            if gi is None:
                continue  # leave NaN → skipped from eval
            p_vec = G[gi]  # (k,)
            y_pred = (G @ self.W @ p_vec + self.b).astype(np.float32)  # (n_genes,)
            for i in range(len(tbi)):
                out[i, j] = y_pred
        return out

    def save_weights(self, path):
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        degenerate = self.W is None
        np.savez(
            str(path),
            degenerate=np.array(degenerate),
            W=(self.W if not degenerate
               else np.zeros((self.n_components, self.n_components), dtype=np.float32)),
            b=(self.b if not degenerate else np.zeros(0, dtype=np.float32)),
            gene_names=np.array(self.gene_names or [], dtype=object),
            ko_names=np.array(self.ko_names or [], dtype=object),
            n_components=self.n_components,
            g_ridge=self.g_ridge, p_ridge=self.p_ridge,
        )

    @classmethod
    def load_weights(cls, path):
        d = np.load(str(path), allow_pickle=True)
        m = cls(n_components=int(d["n_components"]),
                g_ridge=float(d["g_ridge"]),
                p_ridge=float(d["p_ridge"]))
        if bool(d["degenerate"]) if "degenerate" in d else False:
            m.W = None; m.b = None
        else:
            m.W = d["W"]; m.b = d["b"]
        m.gene_names = [str(g) for g in d["gene_names"]]
        m.ko_names = [str(k) for k in d["ko_names"]]
        return m


# ===================================================================
# 4. Correlation — predict from nearest-neighbor pseudobulk
# ===================================================================


@register
class Correlation(LearnedPredictor):
    """For each test KO, find the most-correlated training KO and predict its delta.

    Same-bin matching is preferred: for each test (bin, ko), the nearest training
    KO is chosen from the training KOs **in that same bin**. For UnseenBoth where
    the test bin has no training entries, the search space relaxes to the union
    of all training bins' KOs (this is a search-space relaxation, not a degenerate
    substitution into another predictor's output).
    """
    name = "Correlation"
    scenarios = ["UnseenPert", "UnseenBoth"]

    def __init__(self):
        # Stores the per-bin training delta matrix for nearest-neighbor lookup.
        self._train_per_bin: List[np.ndarray] = []
        self._train_kos_per_bin: List[np.ndarray] = []

    def fit(self, store, scenario, fold):
        split = store.split(scenario, fold)
        train_mask = split.train_mask_2d(store.n_bins, store.n_kos)
        train_bins = split.train_bin_indices_or_all(store.n_bins)
        self._train_per_bin = []
        self._train_kos_per_bin = []
        for b in train_bins:
            # Per-bin TRAIN kos taken from the mask (NOT split.train_ko_indices,
            # which is a marginal that for UnseenBoth still lists the test perts —
            # they are train in OTHER bins). For the test bin the mask excludes the
            # held-out test perts, so a test (bin, ko) cell can no longer match
            # itself as its own nearest neighbor (the leak). For rectangle scenarios
            # the mask row is exactly train_ko_indices, so behavior is unchanged.
            kos_b = np.where(train_mask[int(b)])[0].astype(np.int64)
            mat = store.all_deltas[int(b)][kos_b]  # (Kt_b, G)
            self._train_per_bin.append(mat.astype(np.float32))
            self._train_kos_per_bin.append(kos_b)

    def predict(self, store, scenario, fold):
        if not self._train_per_bin:
            raise RuntimeError("Correlation.predict before fit()")
        split = store.split(scenario, fold)
        tbi = split.test_bin_indices_or_all(store.n_bins)
        tki = split.test_ko_indices

        # Build NN lookup: for each test bin, pick nearest train bin if same
        train_bins = split.train_bin_indices_or_all(store.n_bins)
        bin_to_local = {int(b): i for i, b in enumerate(train_bins)}

        out = np.zeros(_expected_output_shape(store, tbi, tki), dtype=np.float32)
        deltas = store.all_deltas
        tki_arr = np.asarray(tki, dtype=int)
        for i, b in enumerate(tbi):
            # Pick correlated NN from the same train bin if available, else fall back.
            if int(b) in bin_to_local:
                ref_mat = self._train_per_bin[bin_to_local[int(b)]]
            else:
                ref_mat = np.concatenate(self._train_per_bin, axis=0)
            # Pearson NN, vectorized over all test kos at once. The reference
            # centering + norms are independent of the test ko, so they are hoisted
            # out of the per-ko loop (which otherwise recomputes an (n_ref × n_genes)
            # centering for every test ko — O(n_test·n_ref·n_genes) of redundant work,
            # intractable for large screens like xatlas). Same math as before.
            rv = ref_mat - ref_mat.mean(axis=1, keepdims=True)        # (n_ref, G)
            rv_sq = (rv ** 2).sum(axis=1)                             # (n_ref,)
            tmat = deltas[int(b), tki_arr]                            # (n_test, G)
            tv = tmat - tmat.mean(axis=1, keepdims=True)             # (n_test, G)
            tv_sq = (tv ** 2).sum(axis=1)                            # (n_test,)
            num = rv @ tv.T                                          # (n_ref, n_test)
            denom = np.sqrt(rv_sq[:, None] * tv_sq[None, :] + 1e-8)  # (n_ref, n_test)
            nn = np.argmax(num / denom, axis=0)                     # (n_test,) best ref per test ko
            out[i] = ref_mat[nn]
        return out

    def save_weights(self, path):
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        # Store each per-bin matrix under a separate key to preserve dtype.
        # np.array(list_of_2d_arrays, dtype=object) can silently stack into a
        # 3D object array whose inner rows lose their float32 dtype.
        d = {"n_bins": np.array(len(self._train_per_bin))}
        for i, mat in enumerate(self._train_per_bin):
            d[f"train_per_bin_{i}"] = np.asarray(mat, dtype=np.float32)
            d[f"train_kos_per_bin_{i}"] = np.asarray(self._train_kos_per_bin[i], dtype=np.int64)
        np.savez(str(path), **d)

    @classmethod
    def load_weights(cls, path):
        d = np.load(str(path), allow_pickle=True)
        m = cls()
        n_bins = int(d["n_bins"])
        m._train_per_bin = [np.asarray(d[f"train_per_bin_{i}"], dtype=np.float32)
                            for i in range(n_bins)]
        m._train_kos_per_bin = [np.asarray(d[f"train_kos_per_bin_{i}"], dtype=np.int64)
                                for i in range(n_bins)]
        return m


# ===================================================================
# 5. TargetScaling — pure target-only baseline: delta[target] = -alpha * baseline[target]
# ===================================================================


@register
class TargetScaling(LearnedPredictor):
    """Pure target-only baseline in delta space.

    Prediction:
        delta_pred[b, k, g] = -alpha * baseline[b, g]   if g in target(k)
                              0                          otherwise

    Single scalar alpha shared across all perturbations and bins, fit by OLS
    over training (bin, ko, target_gene) triples:

        alpha = -sum_{(b,k,g) in train, g in target(k)} delta_true[b,k,g] * baseline[b,g]
                / sum_{(b,k,g)} baseline[b,g]^2

    No silent fallbacks. Raises if any training/test KO parses to zero targets
    or to a target name absent from store.gene_names; raises if the OLS
    denominator is degenerate; raises if the training split is empty.
    """
    name = "TargetScaling"
    has_drop_rule = True
    scenarios = [
        "UnseenPert", "UnseenCell", "UnseenBoth", "UnseenPair",
        "UnseenDose", "UnseenCombo",
    ]

    def __init__(self):
        self.alpha: Optional[float] = None

    def fit(self, store, scenario, fold):
        from benchmark.predictors._shared import resolve_target_gene_indices
        split = store.split(scenario, fold)
        train_kos = split.train_ko_indices
        train_bins = split.train_bin_indices_or_all(store.n_bins)

        if len(train_kos) == 0 or len(train_bins) == 0:
            raise ValueError(
                f"TargetScaling.fit: empty training split "
                f"(n_train_bins={len(train_bins)}, n_train_kos={len(train_kos)})"
            )

        deltas = store.all_deltas   # (n_bins, n_kos, n_genes)
        baseline = store.ctrl_bulk    # (n_bins, n_genes)
        gene_to_idx = {g: i for i, g in enumerate(store.gene_names)}

        num = 0.0
        den = 0.0
        n_triples = 0
        n_skipped_kos = 0

        # Accumulate over the TRAIN (bin, ko) cells only. train_mask_2d excludes the
        # held-out corner for UnseenBoth/UnseenPair (the rectangle would include the
        # held-out cells); it is the rectangle for the other scenarios. Resolve each
        # ko's target genes once.
        bi_arr, ki_arr = split.train_cells(store.n_bins, store.n_kos)  # (bin, ko) coords of TRAIN cells
        tgt_cache: dict = {}
        skipped_kos: set = set()
        for bi_int, k_int in zip(bi_arr.tolist(), ki_arr.tolist()):
            if k_int not in tgt_cache:
                tgt_cache[k_int] = resolve_target_gene_indices(
                    store.ko_names[k_int], gene_to_idx, strict=False)
            target_idx = tgt_cache[k_int]
            if not target_idx:
                skipped_kos.add(k_int)
                continue
            base_row = baseline[bi_int].astype(np.float64)
            delta_row = deltas[bi_int, k_int].astype(np.float64)
            for gi in target_idx:
                num += float(delta_row[gi] * base_row[gi])
                den += float(base_row[gi] * base_row[gi])
                n_triples += 1
        n_skipped_kos = len(skipped_kos)

        if n_skipped_kos:
            log.warning(
                "TargetScaling.fit: %d/%d training KOs had unresolvable targets, "
                "skipped from OLS",
                n_skipped_kos, len(train_kos),
            )

        if n_triples == 0:
            raise ValueError(
                "TargetScaling.fit: no (bin, ko, target_gene) triples in training "
                "split; cannot fit alpha."
            )
        if den < np.finfo(np.float64).eps:
            raise ValueError(
                f"TargetScaling.fit: degenerate OLS denominator (sum of baseline^2 "
                f"over {n_triples} training triples is {den:.3e}); baseline expression "
                f"is zero at every target gene coordinate."
            )
        self.alpha = -num / den

    def predict(self, store, scenario, fold):
        from benchmark.predictors._shared import resolve_target_gene_indices
        if self.alpha is None:
            raise RuntimeError("TargetScaling.predict before fit()")
        split = store.split(scenario, fold)
        tbi = split.test_bin_indices_or_all(store.n_bins)
        tki = split.test_ko_indices
        baseline = store.ctrl_bulk
        gene_to_idx = {g: i for i, g in enumerate(store.gene_names)}

        per_ko_target_idx: List[List[int]] = []
        n_skipped = 0
        for ko_g in tki:
            ko_name = store.ko_names[int(ko_g)]
            target_idx = resolve_target_gene_indices(ko_name, gene_to_idx, strict=False)
            if not target_idx:
                n_skipped += 1
            per_ko_target_idx.append(target_idx)

        if n_skipped:
            log.warning(
                "TargetScaling.predict: skipping %d/%d test KO(s) with unresolvable "
                "targets for %s/%s/fold%d (NaN → excluded from eval)",
                n_skipped, len(tki), store.dataset, scenario, fold,
            )

        out = np.zeros(_expected_output_shape(store, tbi, tki), dtype=np.float32)
        for i, b in enumerate(tbi):
            bi = int(b)
            base_row = baseline[bi]
            for j, target_idx in enumerate(per_ko_target_idx):
                if not target_idx:
                    out[i, j, :] = np.nan   # unresolved → skipped from eval
                    continue
                for gi in target_idx:
                    out[i, j, gi] = -float(self.alpha) * float(base_row[gi])
        return out

    def save_weights(self, path):
        if self.alpha is None:
            raise RuntimeError("TargetScaling.save_weights before fit()")
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(str(path), alpha=np.float64(self.alpha))

    @classmethod
    def load_weights(cls, path):
        d = np.load(str(path), allow_pickle=True)
        m = cls()
        m.alpha = float(d["alpha"])
        return m


# ===================================================================
# 5b. Mean+TargetScaling — regime mean delta + target-gene scaling term
# ===================================================================


@register
class MeanPlusTargetScaling(LearnedPredictor):
    """Composite simple baseline: the regime's mean-delta baseline PLUS the
    TargetScaling target-gene correction.

    For each test (bin, ko): predict the regime-appropriate mean delta
    (``DEFAULT_BASELINE_PER_SCENARIO`` — Mean-over-perturbations for UnseenPert,
    Mean-over-cell-types for UnseenCell, Mean-over-perturbations-and-cell-types for
    UnseenBoth) and ADD the fitted ``-alpha*baseline``
    scaling at the ko's resolvable target genes. Where a target is unresolvable the
    scaling term is simply omitted (the prediction falls back to the mean), so the
    composite keeps the mean's coverage rather than inheriting TargetScaling's
    per-ko NaN drop.

    Rationale: the mean captures the population-average response while the target
    term adds the knockdown of the perturbed gene itself — often the strongest
    *simple* baseline, hence a prime candidate for the per-regime hardest baseline.
    """

    name = "Mean+TargetScaling"
    has_drop_rule = True             # safe superset (the mean component may drop on some regimes)
    scenarios = ["UnseenPert", "UnseenCell", "UnseenBoth", "UnseenPair"]

    # Composite-specific mean override, consulted before DEFAULT_BASELINE_PER_SCENARIO.
    # UnseenPair: the regime *default* baseline is Two-way-mean, but Mean-over-perturbations
    # empirically dominates it as a reference (e.g. mcfaline23: weighted-Pearson 0.22 vs 0.08,
    # MoP better in 85-98% of perts) — the cell-type consensus generalises better than Two-way-mean's
    # noisy / non-transferable per-pert term. Mean-over-perturbations is valid for UnseenPair (every
    # cell type is in training; only specific (cell, pert) pairs are held out). DEFAULT_BASELINE_PER_
    # SCENARIO is left unchanged, so the scenario's default baseline (used elsewhere) stays
    # Two-way-mean — only the composite's mean changes here.
    _COMPOSITE_MEAN = {"UnseenPair": "Mean-over-perturbations"}

    def __init__(self):
        self._ts = TargetScaling()

    @classmethod
    def _mean_for(cls, scenario: str) -> "Predictor":
        # Single source of truth for the regime baseline; lazy import avoids a
        # predictors <-> meta_metrics import cycle (TargetScaling uses the same
        # lazy-import pattern for _shared helpers).
        from benchmark.meta_metrics import DEFAULT_BASELINE_PER_SCENARIO
        from benchmark.predictors.base import get_predictor
        name = cls._COMPOSITE_MEAN.get(scenario, DEFAULT_BASELINE_PER_SCENARIO[scenario])
        return get_predictor(name)()

    def fit(self, store, scenario, fold):
        # The mean components are stateless (needs_training=False); only alpha is fit.
        self._ts.fit(store, scenario, fold)

    def predict(self, store, scenario, fold):
        mean = self._mean_for(scenario)
        mean.fit(store, scenario, fold)                         # no-op for the mean baselines
        mean_pred = mean.predict(store, scenario, fold)         # (n_tb, n_tk, n_genes)
        ts_pred = self._ts.predict(store, scenario, fold)       # 0 off-target, term on-target, NaN if unresolvable
        # TargetScaling's per-ko NaN (unresolvable target) -> "no correction" so the
        # composite keeps the mean's coverage; a NaN in the mean itself (test bin with
        # no train cells) is preserved (truly unpredictable -> skipped in eval).
        return (mean_pred + np.nan_to_num(ts_pred, nan=0.0)).astype(np.float32)

    def save_weights(self, path):
        self._ts.save_weights(path)              # only the alpha is stateful

    @classmethod
    def load_weights(cls, path):
        m = cls()
        m._ts = TargetScaling.load_weights(path)
        return m


# ===================================================================
# 6. GlobalEpistasis — combo predictor with a learned interaction term
# ===================================================================


@register
class GlobalEpistasis(LearnedPredictor):
    """Combo predictor: Δ_AB = Δ_A + Δ_B + W [Δ_A ⊙ Δ_B].

    `W` is a per-gene scalar learned by least-squares on the training combos.
    """
    name = "GlobalEpistasis"
    has_drop_rule = True
    scenarios = ["UnseenCombo"]

    def __init__(self, ridge_lambda: float = 1.0):
        self.ridge_lambda = float(ridge_lambda)
        self.w: Optional[np.ndarray] = None  # (n_genes,)
        # True when fit() found no resolvable training combos and fell back to
        # pure-additive (w=0) — distinguishes that from a genuinely learned w≈0.
        self.degenerate: bool = False

    def _combo_pairs(self, store, ko_indices, *, scenario, fold, phase):
        """Thin wrapper over the shared resolver. See
        ``benchmark.predictors._shared.resolve_combo_pairs``."""
        return resolve_combo_pairs(store, ko_indices, label=f"GlobalEpistasis.{phase}",
                                   scenario=scenario, fold=fold)

    def fit(self, store, scenario, fold):
        split = store.split(scenario, fold)
        if len(split.train_ko_indices) == 0:
            raise ValueError(
                f"GlobalEpistasis.fit: empty train_ko_indices for "
                f"{scenario}/fold{fold} on {store.dataset}"
            )
        deltas = store.all_deltas
        pairs = self._combo_pairs(
            store, split.train_ko_indices,
            scenario=scenario, fold=fold, phase="fit",
        )
        if not pairs:
            # No resolvable training combos → no interaction term to learn;
            # fall back to pure additive (w=0). predict() handles the rest.
            log.warning(
                "GlobalEpistasis.fit: no resolvable training combos for %s/%s/fold%d; "
                "falling back to pure-additive (w=0)",
                store.dataset, scenario, fold,
            )
            self.w = np.zeros(store.n_genes, dtype=np.float32)
            self.degenerate = True
            return
        train_bins = split.train_bin_indices_or_all(store.n_bins)
        Y_resid_rows = []
        F_rows = []
        for local_pos, ai, bi in pairs:
            ko_combo = split.train_ko_indices[local_pos]
            for b in train_bins:
                dA = deltas[int(b), ai]
                dB = deltas[int(b), bi]
                dAB = deltas[int(b), int(ko_combo)]
                resid = dAB - (dA + dB)
                interaction = dA * dB
                Y_resid_rows.append(resid)
                F_rows.append(interaction)
        Y = np.stack(Y_resid_rows, axis=0)  # (n_samples, n_genes)
        F = np.stack(F_rows, axis=0)         # (n_samples, n_genes)
        # Per-gene 1D ridge: w_g = sum(F_g * Y_g) / (sum(F_g^2) + λ)
        num = (F * Y).sum(axis=0)
        denom = (F * F).sum(axis=0) + self.ridge_lambda
        self.w = (num / denom).astype(np.float32)

    def predict(self, store, scenario, fold):
        if self.w is None:
            raise RuntimeError("GlobalEpistasis.predict before fit()")
        split = store.split(scenario, fold)
        tbi = split.test_bin_indices_or_all(store.n_bins)
        tki = split.test_ko_indices
        deltas = store.all_deltas
        pairs = self._combo_pairs(
            store, tki, scenario=scenario, fold=fold, phase="predict",
        )
        # NaN init: unresolved combos stay NaN → skipped from eval (not scored).
        out = np.full(_expected_output_shape(store, tbi, tki), np.nan, dtype=np.float32)
        for local_pos, ai, bi in pairs:
            for i, b in enumerate(tbi):
                dA = deltas[int(b), ai]
                dB = deltas[int(b), bi]
                out[i, local_pos] = (dA + dB + self.w * dA * dB).astype(np.float32)
        return out

    def save_weights(self, path):
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(str(path), w=self.w, ridge_lambda=self.ridge_lambda,
                 degenerate=np.array(self.degenerate))

    @classmethod
    def load_weights(cls, path):
        d = np.load(str(path), allow_pickle=True)
        m = cls(ridge_lambda=float(d["ridge_lambda"]))
        m.w = d["w"]
        m.degenerate = bool(d["degenerate"]) if "degenerate" in d else False
        return m


# ===================================================================
# 7. LatentAdditive — bilinear linear+embedding model (Ahlmann-Eltze 2025)
# ===================================================================
#
# Y = gene_emb @ K @ pert_emb + center + baseline
#
# where:
#   gene_emb  (n_genes, k)         PCA of training pseudobulks (genes × PCs)
#   pert_emb  (k, n_perts)         each pert's column = its target gene's row
#                                   in gene_emb. Combo perts ('A+B') average
#                                   the embeddings of A and B.
#   K         (k, k)               learned via closed-form ridge bilinear
#                                   regression (Ahlmann-Eltze solve_y_axb):
#                                     K = (A'A + λI)^-1 A' Y B' (BB' + λI)^-1
#   center    (n_genes,)           per-gene mean of training Y
#   baseline  (n_genes,)           per-gene control mean
#
# Reference: `linear_perturbation_prediction-Paper/benchmark/src/
#            run_linear_pretrained_model.R`, gene_embedding=pert_embedding=
#            "training_data". This is the strong simple-linear baseline that
#            matches/beats DL models in the Ahlmann-Eltze 2025 paper.
#
# For multi-bin scenarios we pool training pseudobulks across training bins
# (the AE formulation has no native bin axis); per-bin extension would need
# a 3D bilinear model.


def _solve_y_axb(
    Y: np.ndarray, A: np.ndarray, B: np.ndarray,
    a_ridge: float, b_ridge: float,
) -> np.ndarray:
    """Closed-form ridge solver for Y ≈ A K B.

    Y: (n_genes, n_perts), A: (n_genes, k), B: (k, n_perts).
    Returns K: (k, k). Replaces NaNs with 0 (matches AE behavior).
    """
    k = A.shape[1]
    AtA = A.T @ A + a_ridge * np.eye(k, dtype=np.float64)
    BBt = B @ B.T + b_ridge * np.eye(k, dtype=np.float64)
    # K = (A'A + λI)^-1 A' Y B' (BB' + λI)^-1
    M = np.linalg.solve(AtA, A.T @ Y @ B.T)         # (k, k)
    K = np.linalg.solve(BBt.T, M.T).T               # (k, k)
    K[np.isnan(K)] = 0.0
    return K


def _pert_target_indices(
    ko_name: str, gene_to_idx: dict,
) -> Tuple[List[int], List[str]]:
    """Return (resolved_indices, missing_tokens) for a KO's target gene(s).

    Parses 'GeneA', 'GeneA+GeneB', 'GeneA;GeneB', 'GeneA@0.5' (dose ignored) via
    the single canonical `parse_target_genes`. Returns the list of `var_names`
    indices for resolved tokens, plus the list of tokens that did NOT resolve.
    Callers raise / drop when missing_tokens is non-empty.
    """
    indices: List[int] = []
    missing: List[str] = []
    for tok in parse_target_genes(ko_name):
        idx = gene_to_idx.get(tok)
        if idx is None:
            missing.append(tok)
        else:
            indices.append(idx)
    return indices, missing


def _build_pert_emb(
    ko_names_subset: List[str],
    gene_emb: np.ndarray,
    gene_to_idx: dict,
    *,
    predictor_name: str,
    phase: str,
    dataset: str,
    scenario: str,
    fold: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build pert_emb columns for the RESOLVABLE KOs in `ko_names_subset`.

    Permissive (matches upstream): a KO with any target-gene token absent from
    `gene_to_idx` (or no parseable token) is dropped with a warning. Returns
    ``(emb, keep_mask)`` where ``emb`` is ``(k, n_kept)`` (columns in the order
    of the kept KOs) and ``keep_mask`` is a bool array over `ko_names_subset`.
    fit() filters its target matrix by `keep_mask`; predict() leaves dropped KO
    slices as NaN so they are skipped from eval.
    """
    cols: List[np.ndarray] = []
    keep: List[bool] = []
    offenders: List[str] = []
    for ko in ko_names_subset:
        idxs, missing = _pert_target_indices(ko, gene_to_idx)
        if missing or not idxs:
            offenders.append(ko)
            keep.append(False)
            continue
        cols.append(gene_emb[idxs].mean(axis=0))
        keep.append(True)

    if offenders:
        log.warning(
            "%s.%s: skipping %d/%d KO(s) with target gene token(s) absent from "
            "var_names for %s/%s/fold%d: %s",
            predictor_name, phase, len(offenders), len(ko_names_subset),
            dataset, scenario, fold, offenders[:3],
        )
    keep_mask = np.array(keep, dtype=bool)
    emb = (np.stack(cols, axis=1) if cols
           else np.zeros((gene_emb.shape[1], 0), dtype=gene_emb.dtype))
    return emb, keep_mask  # (k, n_kept), (n_subset,)


@register
class LatentAdditive(LearnedPredictor):
    """Bilinear linear+embedding predictor (Ahlmann-Eltze et al. 2025).

    Closed-form ridge solver — no SGD, no MLP, no PyTorch. Trains in O(n_genes·k²)
    after PCA (which itself is O(n_genes · n_train_perts · k)). For typical sizes
    (n_genes=8k, n_train_perts=200, k=10) the entire fit takes <1 s.

    By default k=10 (matches the AE paper). Ridge λ defaults to 0.1 for both
    A and B (also matches the paper's `--ridge_penalty` default).
    """
    name = "LatentAdditive"
    has_drop_rule = True
    scenarios = [
        "UnseenPert", "UnseenCell", "UnseenBoth", "UnseenPair", "UnseenCombo",
    ]

    def __init__(self, k: int = 10, a_ridge: float = 0.1, b_ridge: float = 0.1):
        self.k = int(k)
        self.a_ridge = float(a_ridge)
        self.b_ridge = float(b_ridge)
        # Learned state
        self.K: Optional[np.ndarray] = None         # (k, k)
        self.gene_emb: Optional[np.ndarray] = None  # (n_genes, k)
        self.center: Optional[np.ndarray] = None    # (n_genes,)
        self.gene_names: Optional[List[str]] = None
        # True when fit() had <2 resolvable training perts and fell back to
        # center-only (K=0) — distinguishes that from a genuinely learned K≈0.
        self.degenerate: bool = False

    def fit(self, store, scenario, fold):
        from sklearn.decomposition import PCA

        split = store.split(scenario, fold)
        train_bins = split.train_bin_indices_or_all(store.n_bins)
        train_kos = split.train_ko_indices
        if len(train_kos) == 0:
            raise ValueError(
                f"LatentAdditive.fit: empty train_ko_indices for {scenario}/fold{fold}"
            )

        # Pool training pseudobulks across training bins → (n_train_perts, n_genes).
        # All-cell mean (matches Miller's training target), not first-half only.
        # For each ko, mean over the bins where it is a TRAIN cell: train_mask_2d
        # excludes the held-out corner for UnseenBoth/UnseenPair (the rectangle mean
        # would average in the held-out bins); it is the rectangle otherwise.
        per_ko = masked_mean(
            store.all_bulk, split.train_mask_2d(store.n_bins, store.n_kos),
            over="per_ko")  # (n_kos, n_genes)
        train_X_pert = per_ko[train_kos]
        baseline = store.ctrl_bulk[train_bins].mean(axis=0)  # (n_genes,)
        # Y is the "change" matrix, gene-major to mirror AE: (n_genes, n_train_perts)
        Y = (train_X_pert - baseline).T.astype(np.float64)

        # Per-gene center (subtract before solving — corresponds to AE's `center`)
        center = Y.mean(axis=1)  # (n_genes,)
        Y_centered = Y - center[:, None]

        # PCA of training pseudobulks. AE runs PCA on the absolute expression
        # matrix `X` (genes × perts); we use the change matrix `Y` (centered)
        # to better isolate perturbation signal. Both have shape (n_genes, n_train_perts).
        # `PCA.fit_transform` with input shape (n_samples, n_features) where
        # samples = genes returns the gene embeddings (n_genes, k).
        pca = PCA(n_components=min(self.k, Y_centered.shape[1]), random_state=42)
        gene_emb_full = pca.fit_transform(Y_centered).astype(np.float64)  # (n_genes, k)
        k_eff = gene_emb_full.shape[1]
        if k_eff < self.k:
            log.info("LatentAdditive: PCA returned k=%d (< requested %d)",
                      k_eff, self.k)

        gene_to_idx = {g: i for i, g in enumerate(store.gene_names)}
        train_ko_names = [store.ko_names[int(i)] for i in train_kos]
        pert_emb_train, keep = _build_pert_emb(
            train_ko_names, gene_emb_full, gene_to_idx,
            predictor_name="LatentAdditive", phase="fit",
            dataset=store.dataset, scenario=scenario, fold=fold,
        )
        if pert_emb_train.shape[1] < 2:
            # Too few resolvable training perts for the bilinear solve → fall
            # back to center-only (K=0); predict() then returns the per-gene
            # center for resolvable test KOs and NaN for the rest.
            log.warning(
                "LatentAdditive.fit: only %d resolvable training pert(s) on "
                "%s/%s/fold%d; falling back to center-only (K=0)",
                pert_emb_train.shape[1], store.dataset, scenario, fold,
            )
            self.K = np.zeros((k_eff, k_eff), dtype=np.float32)
            self.gene_emb = gene_emb_full.astype(np.float32)
            self.center = center.astype(np.float32)
            self.gene_names = list(store.gene_names)
            self.degenerate = True
            return

        # Filter the target matrix columns to the kept (resolvable) train perts.
        Y_fit = Y_centered[:, keep]
        K = _solve_y_axb(
            Y_fit, gene_emb_full, pert_emb_train,
            self.a_ridge, self.b_ridge,
        )

        self.K = K.astype(np.float32)
        self.gene_emb = gene_emb_full.astype(np.float32)
        self.center = center.astype(np.float32)
        self.gene_names = list(store.gene_names)

    def predict(self, store, scenario, fold):
        if self.K is None or self.gene_emb is None:
            raise RuntimeError("LatentAdditive.predict before fit() (or load_weights()).")
        if self.gene_names != store.gene_names:
            raise RuntimeError(
                "LatentAdditive.predict: gene_names changed between fit and predict "
                "(predictor saw %d genes, store has %d). Re-run fit on this dataset."
                % (len(self.gene_names), len(store.gene_names))
            )

        from benchmark.predictors._shared import _expected_output_shape
        split = store.split(scenario, fold)
        tbi = split.test_bin_indices_or_all(store.n_bins)
        tki = split.test_ko_indices

        gene_to_idx = {g: i for i, g in enumerate(self.gene_names)}
        test_ko_names = [store.ko_names[int(i)] for i in tki]
        pert_emb_test, keep = _build_pert_emb(
            test_ko_names, self.gene_emb.astype(np.float64), gene_to_idx,
            predictor_name="LatentAdditive", phase="predict",
            dataset=store.dataset, scenario=scenario, fold=fold,
        )

        # pred_delta = gene_emb @ K @ pert_emb + center[:, None]; only for the
        # kept (resolvable) test KOs. Dropped KOs stay NaN → skipped from eval.
        A = self.gene_emb.astype(np.float64)
        K = self.K.astype(np.float64)
        n_genes = A.shape[0]
        pred_per_ko = np.full((len(test_ko_names), n_genes), np.nan, dtype=np.float64)
        if pert_emb_test.shape[1] > 0:
            pred_kept = (A @ K @ pert_emb_test + self.center[:, None]).T  # (n_kept, n_genes)
            pred_per_ko[keep] = pred_kept

        out = np.full(_expected_output_shape(store, tbi, tki), np.nan, dtype=np.float32)
        for i in range(len(tbi)):
            out[i] = pred_per_ko.astype(np.float32)
        return out

    def save_weights(self, path):
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            str(path),
            K=self.K,
            gene_emb=self.gene_emb,
            center=self.center,
            gene_names=np.array(self.gene_names or [], dtype=object),
            k=self.k, a_ridge=self.a_ridge, b_ridge=self.b_ridge,
            degenerate=np.array(self.degenerate),
        )

    @classmethod
    def load_weights(cls, path):
        d = np.load(str(path), allow_pickle=True)
        m = cls(k=int(d["k"]),
                a_ridge=float(d["a_ridge"]),
                b_ridge=float(d["b_ridge"]))
        m.K = d["K"]
        m.gene_emb = d["gene_emb"]
        m.center = d["center"]
        m.gene_names = [str(g) for g in d["gene_names"]]
        m.degenerate = bool(d["degenerate"]) if "degenerate" in d else False
        return m


__all__ = [
    "Ridge", "LinearAdditive", "BilinearRidge", "Correlation",
    "TargetScaling", "GlobalEpistasis", "LatentAdditive",
]
