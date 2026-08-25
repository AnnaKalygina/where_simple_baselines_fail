"""Positive-control predictors.

These predictors define the "ceiling" of the benchmark — what a perfect
predictor with access to test-time information could achieve. They are
used by BS/CF as the `positive_predictor` reference.

  * Tech-duplicate   — predicts second_half pseudobulk delta for each test KO.
                        Read off `store.second_half_deltas` directly.
                        Read-only (no `fit`), but `needs_training=False`.
  * Interp-duplicate — predicts the second-half DEG vector reindexed to var_names,
                        scaled by the predicted KO's first-half delta sign.
                        Useful as a "weak positive control" — captures DEG-level
                        structure without the per-gene magnitude of Tech-dup.
"""
from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np

from benchmark.data_loader import DatasetStore
from benchmark.predictors._shared import _expected_output_shape, masked_mean
from benchmark.predictors.base import Predictor, register

log = logging.getLogger(__name__)


@register
class TechDuplicate(Predictor):
    """Positive control: second_half - ctrl_bulk for each test KO.

    The second half of the tech-duplicate split is independent from the
    first half (which provides the ground-truth deltas), so this gives a
    realistic upper bound on what any predictor could achieve.
    """
    name = "Tech-duplicate"
    needs_training = False
    scenarios = [
        "UnseenPert", "UnseenCell", "UnseenBoth", "UnseenPair",
        "UnseenDose", "UnseenCombo",
    ]

    def fit(self, store, scenario, fold):
        return None

    def predict(self, store, scenario, fold):
        split = store.split(scenario, fold)
        tbi = split.test_bin_indices_or_all(store.n_bins)
        tki = split.test_ko_indices
        deltas = store.second_half_deltas  # (n_bins, n_kos, n_genes)
        return deltas[np.ix_(tbi, tki)].astype(np.float32)


@register
class InterpDuplicate(Predictor):
    """Positive control: per-gene interpolation between the tech-duplicate and
    the mean baseline, weighted by second-half DEG significance.

    Matches the upstream `add_interpolated_baseline.py` (Perturbation-Models-
    Outperform-Baselines): for each test KO and gene,

        alpha[g]  = 1 - pval_adj_second_half[g]          (missing/NaN -> 0)
        interp[g] = alpha[g] * tech_dup[g] + (1 - alpha[g]) * mean[g]

    where `tech_dup` is the second-half delta and `mean` is the training-mean
    delta (the Mean-over-perturbations baseline). Significant genes lean toward
    the tech-duplicate; non-significant genes fall back to the mean — so this
    control is, by design, never beaten by the mean baseline (unlike the raw
    Tech-duplicate, which saturates on weak-effect datasets). This is the proper
    benchmark ceiling.
    """
    name = "Interp-duplicate"
    needs_training = False
    scenarios = [
        "UnseenPert", "UnseenCell", "UnseenBoth", "UnseenPair",
        "UnseenDose", "UnseenCombo",
    ]

    def fit(self, store, scenario, fold):
        return None

    def _mean_baseline_per_test_bin(self, store, tbi, split):
        """Training-mean delta per test bin (mirrors Mean-over-perturbations):
        per-bin mean over each bin's TRAIN cells, with the global train mean as the
        fallback for fully-held-out test bins.

        Sourced from the per-(bin, ko) train mask (``train_mask_2d``) — NOT the
        ``train_bins × train_ko_indices`` rectangle, whose outer product
        over-includes the held-out test corner for the cell-holdout scenarios
        (UnseenCell / UnseenBoth / UnseenPair). For single-cell-type datasets and
        UnseenPert the mask IS that rectangle, so this is unchanged and stays
        faithful to Miller's train-only interpolated duplicate; for the multi-bin
        cell-holdout scenarios it is the leak-free, cell-type-specific train mean.
        (See memory ``project_interpdup_multibin_mean_mask``.)
        """
        mask = split.train_mask_2d(store.n_bins, store.n_kos)
        if not mask.any():
            raise ValueError(f"InterpDuplicate: empty training split for {store.dataset}")
        per_bin = masked_mean(store.all_deltas, mask, over="per_bin")  # (n_bins, n_genes); NaN where a bin has no train cell
        global_mean = masked_mean(store.all_deltas, mask, over="all")  # (n_genes,)
        out = np.empty((len(tbi), store.n_genes), dtype=np.float32)
        for i, b in enumerate(tbi):
            row = per_bin[int(b)]
            out[i] = row if not np.isnan(row).all() else global_mean
        return out

    def _alpha(self, store, tbi, tki, scenario, fold):
        """Per-(test_bin, test_ko, gene) alpha = 1 - pval_adj_second_half, clipped
        to [0, 1]; missing genes/keys -> pval 1 -> alpha 0 (use the mean)."""
        n_tb, n_tk, n_genes = len(tbi), len(tki), store.n_genes
        # Matrix path (e.g. xatlas): var-aligned tensor, slice directly.
        pmat = store.adata.uns.get("pvals_adj_matrix_second_half")
        if pmat is not None:
            pv = np.asarray(pmat, dtype=np.float64)[np.ix_(tbi, tki)]
        else:
            # Dict path: per-pert adjusted p-values in DEG-name order; reindex to
            # var_names with missing genes filled at pval 1.0.
            names_dict = store.deg_dict("second_half", "names")
            pvals_dict = store.deg_dict("second_half", "pvals_adj")
            gene_to_idx = {g: i for i, g in enumerate(store.gene_names)}
            n_bins = store.n_bins
            pv = np.ones((n_tb, n_tk, n_genes), dtype=np.float64)
            missing: List[str] = []
            for i, bg in enumerate(tbi):
                bn = store.bin_names[int(bg)]
                for j, kg in enumerate(tki):
                    kn = store.ko_names[int(kg)]
                    key = (f"{store.dataset}_{kn}" if n_bins == 1 or bn == "all"
                           else f"{store.dataset}_{bn}_{kn}")
                    if key not in pvals_dict or key not in names_dict:
                        missing.append(key)
                        continue
                    names = names_dict[key]
                    pvals = np.asarray(pvals_dict[key], dtype=np.float64)
                    for gn, pval in zip(names, pvals):
                        gi = gene_to_idx.get(str(gn))
                        if gi is not None:
                            pv[i, j, gi] = pval
            if missing:
                raise ValueError(
                    f"InterpDuplicate: {len(missing)} second-half DEG dict key(s) "
                    f"missing for {store.dataset}/{scenario}/fold{fold}: {missing[:5]}"
                    + (f" ... +{len(missing)-5} more" if len(missing) > 5 else "")
                    + ". Re-run preprocessing or fix the DEG dicts."
                )
        alpha = 1.0 - np.nan_to_num(pv, nan=1.0)
        return np.clip(alpha, 0.0, 1.0).astype(np.float32)

    def predict(self, store, scenario, fold):
        split = store.split(scenario, fold)
        tbi = split.test_bin_indices_or_all(store.n_bins)
        tki = split.test_ko_indices
        tech = store.second_half_deltas[np.ix_(tbi, tki)].astype(np.float32)  # (n_tb,n_tk,ng)
        mean_per_bin = self._mean_baseline_per_test_bin(store, tbi, split).astype(np.float32)
        alpha = self._alpha(store, tbi, tki, scenario, fold)  # (n_tb,n_tk,ng)
        # interp = alpha * tech_dup + (1 - alpha) * mean  (mean broadcast over KOs)
        return alpha * tech + (1.0 - alpha) * mean_per_bin[:, None, :]


__all__ = ["TechDuplicate", "InterpDuplicate"]
