"""Analytical (non-learned) baseline predictors.

Each predictor reads pseudobulk deltas + a split spec from `DatasetStore`
and emits a `(n_test_bins, n_test_kos, n_genes)` prediction tensor. None of
them have trainable parameters, so `fit` is a no-op and `save_weights`
writes an empty NPZ.

Defined here (8 total):

  * Zero                                       — predict no perturbation
  * Mean-over-perturbations                    — average effect across training perts
  * Mean-over-cell-types                       — average across training bins per pert
  * Mean-over-perturbations-and-cell-types     — global mean
  * Two-way-mean                               — additive row+column (μ + α[bin] + β[ko])
  * Additive                                   — Δ_AB = Δ_A + Δ_B (for combos)
  * Matching-mean                              — Δ̂ = (Δ_A + Δ_B) / 2 (for combos)
  * Scaled-delta                               — Δ_partial = dose * Δ_full (for doses)
"""
from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np

from benchmark.config import parse_target_genes
from benchmark.data_loader import DatasetStore
from benchmark.predictors.base import Predictor, register
from benchmark.predictors._shared import (
    _expected_output_shape,
    masked_mean,
    resolve_combo_pairs,
    resolve_target_gene_indices,
)

log = logging.getLogger(__name__)


# The predictor-agnostic helpers live in `benchmark.predictors._shared`; this
# module imports the ones it uses. `resolve_target_gene_indices` and
# `parse_target_genes` (the canonical parser from the dependency-free
# `benchmark.config` leaf) are additionally re-exported via __all__ so existing
# `from benchmark.predictors.analytical import <helper>` call sites keep working.


# ===================================================================
# 1. Zero — predict no effect
# ===================================================================


@register
class Zero(Predictor):
    name = "Zero"
    needs_training = False
    scenarios = [
        "UnseenPert", "UnseenCell", "UnseenBoth", "UnseenPair",
        "UnseenDose", "UnseenCombo",
    ]

    def fit(self, store, scenario, fold):  # noqa: D401
        return None

    def predict(self, store, scenario, fold):
        split = store.split(scenario, fold)
        tbi = split.test_bin_indices_or_all(store.n_bins)
        tki = split.test_ko_indices
        return np.zeros(_expected_output_shape(store, tbi, tki), dtype=np.float32)


# ===================================================================
# 2. Mean-over-perturbations
# ===================================================================


@register
class MeanOverPerturbations(Predictor):
    """For each test bin, predict the mean delta across training perts in that bin.

    Only valid for scenarios where every test bin is also in the training set
    (UnseenPert: no bin split, all bins are training; UnseenPair: only specific
    (cell, pert) pairs are held out, every bin is still in training).
    """
    name = "Mean-over-perturbations"
    needs_training = False
    scenarios = ["UnseenPert", "UnseenPair"]

    def fit(self, store, scenario, fold):
        return None

    def predict(self, store, scenario, fold):
        split = store.split(scenario, fold)
        tbi = split.test_bin_indices_or_all(store.n_bins)
        tki = split.test_ko_indices
        deltas = store.all_deltas  # (n_bins, n_kos, n_genes)

        # Per test bin, average over that bin's TRAIN (bin, ko) cells. train_mask_2d
        # is the explicit pair mask (UnseenPair) or the train_bins × train_kos
        # rectangle (UnseenPert); either way the held-out cells are excluded. A test
        # bin with no training cells yields an all-NaN row (raised below) — this
        # predictor requires every test bin to also appear in training.
        per_bin = masked_mean(deltas, split.train_mask_2d(store.n_bins, store.n_kos),
                              over="per_bin")  # (n_bins, n_genes)
        out = np.empty(_expected_output_shape(store, tbi, tki), dtype=np.float32)
        for i, b in enumerate(tbi):
            row = per_bin[int(b)]
            if np.isnan(row).all():
                # Permissive policy (consistent with every other predictor): a test
                # bin with no training cells yields NaN (skipped in eval) + a warning,
                # rather than raising. This predictor is only meaningful when every
                # test bin is also a training bin (UnseenPert / UnseenPair); use
                # Mean-over-perturbations-and-cell-types when the bin axis is unseen.
                log.warning(
                    "%s.predict: test bin %d has no training cells for %s/fold%d on "
                    "%s — emitting NaN (skipped in eval)",
                    self.name, int(b), scenario, fold, store.dataset)
                out[i] = np.nan
                continue
            out[i] = np.broadcast_to(row[None, :], (len(tki), store.n_genes))
        return out


# ===================================================================
# 3. Mean-over-cell-types
# ===================================================================


@register
class MeanOverCellTypes(Predictor):
    """For each test pert (seen during training), predict its mean delta across training bins."""

    name = "Mean-over-cell-types"
    needs_training = False
    scenarios = ["UnseenCell", "UnseenPair"]

    def fit(self, store, scenario, fold):
        return None

    def predict(self, store, scenario, fold):
        split = store.split(scenario, fold)
        tbi = split.test_bin_indices_or_all(store.n_bins)
        tki = split.test_ko_indices
        deltas = store.all_deltas

        # For each ko, average over the bins where it is a TRAIN cell: the pair mask
        # for UnseenPair, the train_bins × all-kos rectangle for UnseenCell. A test
        # ko with no training bins is NaN (skipped in eval).
        mask = split.train_mask_2d(store.n_bins, store.n_kos)
        if not mask.any():
            raise ValueError(
                f"{self.name}.predict: empty training split for "
                f"{scenario}/fold{fold} on {store.dataset}"
            )
        per_ko = masked_mean(deltas, mask, over="per_ko")  # (n_kos, n_genes)
        out = np.empty(_expected_output_shape(store, tbi, tki), dtype=np.float32)
        for i in range(len(tbi)):
            out[i] = per_ko[tki]
        return out


# ===================================================================
# 4. Mean-over-perturbations-and-cell-types
# ===================================================================


@register
class MeanOverPerturbationsAndCellTypes(Predictor):
    """Global mean delta across (training bins, training perts). One vector predicted
    for every (bin, ko) in the test set."""

    name = "Mean-over-perturbations-and-cell-types"
    needs_training = False
    scenarios = ["UnseenPert", "UnseenCell", "UnseenBoth", "UnseenPair"]

    def fit(self, store, scenario, fold):
        return None

    def predict(self, store, scenario, fold):
        split = store.split(scenario, fold)
        tbi = split.test_bin_indices_or_all(store.n_bins)
        tki = split.test_ko_indices
        deltas = store.all_deltas

        # Global mean over the TRAIN (bin, ko) cells. train_mask_2d excludes the
        # held-out corner for UnseenBoth (where the rectangle train_bins × train_kos
        # would re-include it); for the rectangle scenarios the mask IS that
        # rectangle, so the result is unchanged.
        mask = split.train_mask_2d(store.n_bins, store.n_kos)
        if not mask.any():
            raise ValueError(
                f"{self.name}.predict: empty training split for "
                f"{scenario}/fold{fold} on {store.dataset}"
            )
        gm = masked_mean(deltas, mask, over="all")  # (n_genes,)
        return np.broadcast_to(
            gm[None, None, :], _expected_output_shape(store, tbi, tki),
        ).astype(np.float32).copy()


# ===================================================================
# 5. Two-way-mean (μ + α[bin] + β[ko])
# ===================================================================


@register
class TwoWayMean(Predictor):
    """Additive row + column model on the training entries only.

    Estimates μ, α[bin], β[ko] from the Cartesian product of (train_bins ×
    train_kos), then predicts test entries as μ + α[bin] + β[ko]. Valid only
    when every test bin and every test KO have at least one training entry
    in their row / column (i.e. UnseenPair: both axes are seen individually
    in training, only specific pairs are held out).
    """
    name = "Two-way-mean"
    needs_training = False
    scenarios = ["UnseenPair"]

    def fit(self, store, scenario, fold):
        return None

    def predict(self, store, scenario, fold):
        split = store.split(scenario, fold)
        tbi = split.test_bin_indices_or_all(store.n_bins)
        tki = split.test_ko_indices
        deltas = store.all_deltas  # (n_bins, n_kos, n_genes)

        train_kos = split.train_ko_indices
        if len(train_kos) == 0:
            raise ValueError(
                f"{self.name}.predict: empty train_ko_indices for "
                f"{scenario}/fold{fold} on {store.dataset}"
            )

        # Train on the actual TRAIN (bin, ko) cells. UnseenPair holds out scattered
        # pairs, so the bin×ko rectangle would leak the held-out test pairs into
        # μ/α/β; train_mask_2d returns the explicit pair mask (or the rectangle for
        # rectangle scenarios).
        mask = split.train_mask_2d(store.n_bins, store.n_kos)

        row_count = mask.sum(axis=1)
        col_count = mask.sum(axis=0)
        # Permissive (matches upstream): a test bin/KO with zero training entries
        # has no estimable α/β → predict NaN for those entries (skipped from
        # eval) rather than aborting the whole predictor.
        unseen_test_bins = {int(b) for b in tbi if row_count[int(b)] == 0}
        unseen_test_kos = {int(k) for k in tki if col_count[int(k)] == 0}
        if unseen_test_bins or unseen_test_kos:
            log.warning(
                "%s.predict: %d test bin(s) and %d test KO(s) have zero training "
                "entries on %s/%s/fold%d → NaN (skipped from eval)",
                self.name, len(unseen_test_bins), len(unseen_test_kos),
                store.dataset, scenario, fold,
            )

        m3 = mask[:, :, None].astype(np.float64)
        mu = (deltas.astype(np.float64) * m3).sum(axis=(0, 1)) / int(mask.sum())

        # Rows/columns with zero training entries get a safe denom of 1 so the
        # arithmetic doesn't divide by zero; their α/β are undefined but the
        # resulting (bin, ko) entries are NaN-masked below before returning.
        row_count_safe = np.where(row_count > 0, row_count, 1).astype(np.float64)
        alpha = (deltas.astype(np.float64) * m3).sum(axis=1) / row_count_safe[:, None] - mu
        col_count_safe = np.where(col_count > 0, col_count, 1).astype(np.float64)
        beta = (deltas.astype(np.float64) * m3).sum(axis=0) / col_count_safe[:, None] - mu

        pred = (
            mu[None, None, :]
            + alpha[tbi][:, None, :]
            + beta[tki][None, :, :]
        ).astype(np.float32)

        # Mask entries whose test bin or test KO had no training entry → NaN.
        if unseen_test_bins:
            for i, b in enumerate(tbi):
                if int(b) in unseen_test_bins:
                    pred[i, :, :] = np.nan
        if unseen_test_kos:
            for j, k in enumerate(tki):
                if int(k) in unseen_test_kos:
                    pred[:, j, :] = np.nan
        return pred


# ===================================================================
# 6/7. Additive + Matching-mean (combo baselines)
# ===================================================================


def _resolve_combo_pairs(
    store: DatasetStore,
    test_ko_indices: np.ndarray,
    *,
    predictor_name: str,
    scenario: str,
    fold: int,
) -> List[Tuple[int, int, int]]:
    """Thin wrapper over the shared resolver (kept for its existing call sites).
    See ``benchmark.predictors._shared.resolve_combo_pairs``."""
    return resolve_combo_pairs(store, test_ko_indices, label=predictor_name,
                               scenario=scenario, fold=fold)


@register
class Additive(Predictor):
    """Δ_AB = Δ_A + Δ_B for 2-way combo perturbations."""
    name = "Additive"
    has_drop_rule = True
    needs_training = False
    scenarios = ["UnseenCombo"]

    def fit(self, store, scenario, fold):
        return None

    def predict(self, store, scenario, fold):
        split = store.split(scenario, fold)
        tbi = split.test_bin_indices_or_all(store.n_bins)
        tki = split.test_ko_indices
        deltas = store.all_deltas
        pairs = _resolve_combo_pairs(
            store, tki, predictor_name=self.name, scenario=scenario, fold=fold,
        )
        # NaN init: unresolved combos stay NaN → skipped from eval (not scored).
        out = np.full(_expected_output_shape(store, tbi, tki), np.nan, dtype=np.float32)
        for local_pos, ai, bi in pairs:
            out[:, local_pos, :] = (deltas[tbi, ai, :] + deltas[tbi, bi, :]).astype(np.float32)
        return out


@register
class MatchingMean(Predictor):
    """Δ̂_AB = (Δ_A + Δ_B) / 2 — averaging variant of Additive (2-way combos)."""
    name = "Matching-mean"
    has_drop_rule = True
    needs_training = False
    scenarios = ["UnseenCombo"]

    def fit(self, store, scenario, fold):
        return None

    def predict(self, store, scenario, fold):
        split = store.split(scenario, fold)
        tbi = split.test_bin_indices_or_all(store.n_bins)
        tki = split.test_ko_indices
        deltas = store.all_deltas
        pairs = _resolve_combo_pairs(
            store, tki, predictor_name=self.name, scenario=scenario, fold=fold,
        )
        # NaN init: unresolved combos stay NaN → skipped from eval (not scored).
        out = np.full(_expected_output_shape(store, tbi, tki), np.nan, dtype=np.float32)
        for local_pos, ai, bi in pairs:
            out[:, local_pos, :] = (
                (deltas[tbi, ai, :] + deltas[tbi, bi, :]) / 2.0
            ).astype(np.float32)
        return out


# ===================================================================
# 8. Scaled-delta (dose baseline)
# ===================================================================


def _parse_dose_ko(name: str) -> Optional[Tuple[str, float]]:
    if "@" not in name:
        return None
    base, dose_str = name.split("@", 1)
    try:
        return base, float(dose_str)
    except ValueError:
        return None


@register
class ScaledDelta(Predictor):
    """Δ_partial = dose * Δ_full for dose-suffixed perturbations.

    For each test KO of the form 'GeneA@0.25', looks up the full-dose KO
    'GeneA' in ko_names and scales by 0.25. Strict: raises if any test KO
    is missing the dose suffix or if the full-dose base KO is absent from
    ko_names.
    """
    name = "Scaled-delta"
    has_drop_rule = True
    needs_training = False
    scenarios = ["UnseenDose"]

    def fit(self, store, scenario, fold):
        return None

    def predict(self, store, scenario, fold):
        split = store.split(scenario, fold)
        tbi = split.test_bin_indices_or_all(store.n_bins)
        tki = split.test_ko_indices
        deltas = store.all_deltas
        ko_name_to_idx = {k: i for i, k in enumerate(store.ko_names)}

        not_dose: List[str] = []
        missing_base: List[str] = []
        resolved: List[Tuple[int, int, float]] = []  # (local_pos, full_idx, dose)
        for local_pos, ko_global_idx in enumerate(tki):
            name = store.ko_names[int(ko_global_idx)]
            parsed = _parse_dose_ko(name)
            if parsed is None:
                not_dose.append(name)
                continue
            base, dose = parsed
            full_idx = ko_name_to_idx.get(base)
            if full_idx is None:
                missing_base.append(name)
                continue
            resolved.append((local_pos, full_idx, dose))

        if not_dose or missing_base:
            # Permissive (matches upstream): warn + skip unresolvable dose KOs.
            log.warning(
                "%s: skipping %d/%d unresolvable test KO(s) for %s/%s/fold%d "
                "(no-dose-suffix=%s, missing-base=%s)",
                self.name, len(not_dose) + len(missing_base), len(tki),
                store.dataset, scenario, fold, not_dose[:3], missing_base[:3],
            )

        # NaN init: unresolved dose KOs stay NaN → skipped from eval (not scored).
        out = np.full(_expected_output_shape(store, tbi, tki), np.nan, dtype=np.float32)
        for local_pos, full_idx, dose in resolved:
            out[:, local_pos, :] = (dose * deltas[tbi, full_idx, :]).astype(np.float32)
        return out


# ===================================================================
# 9. TargetZero — full-knockout baseline: delta[target] = -baseline[target]
# ===================================================================


@register
class TargetZero(Predictor):
    """Full-knockout baseline in delta space (TargetScaling with alpha=1 fixed).

    Predicts that perturbation drives the target gene's expression to zero:

        delta_pred[b, k, g] = -baseline[b, g]   if g in target(k)
                              0                  otherwise

    No fit. Companion to TargetScaling: where TargetScaling learns one global
    alpha by OLS, TargetZero asserts alpha=1 a priori. Useful as the parameter-
    free reference for "what if every KO were a complete knockout?".

    Strict: raises if any test KO parses to zero target genes or to a target
    name absent from store.gene_names.
    """
    name = "TargetZero"
    has_drop_rule = True
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
                "%s: skipping %d/%d test KO(s) with unresolvable targets for "
                "%s/%s/fold%d (NaN → excluded from eval)",
                self.name, n_skipped, len(tki), store.dataset, scenario, fold,
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
                    out[i, j, gi] = -float(base_row[gi])
        return out


__all__ = [
    "Zero",
    "MeanOverPerturbations",
    "MeanOverCellTypes",
    "MeanOverPerturbationsAndCellTypes",
    "TwoWayMean",
    "Additive",
    "MatchingMean",
    "ScaledDelta",
    "TargetZero",
    "parse_target_genes",
    "resolve_target_gene_indices",
]
