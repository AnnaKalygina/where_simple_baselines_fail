"""Adapter classes for externally-trained DL models (scGPT, GEARS, PRESAGE).

These predictors do not train inside this codebase — `fit()` is a no-op.
At `predict()` time we look up the model's predictions for the (dataset,
scenario, fold) cell via `models/atheus_csb/MANIFEST.json`, load the
absolute-expression h5ad the model produced, subtract `store.ctrl_bulk`
to get deltas, and return them aligned to `store.gene_names`.

Genes the model does not predict become NaN. Downstream metric loaders use
`load_predictions(..., require_exact_genes=False)` to handle this case.

All MANIFEST access + fold-alignment machinery now lives in the shared leaf
`benchmark._fold_align` (so the verifier and this adapter share one
implementation, no import cycle); the names are re-exported here for the
backward-compat callers `run_pipeline` and `scripts/overlapping_perturbations`
that still import them from this module.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from benchmark.data_loader import DatasetStore
from benchmark.predictors._shared import _expected_output_shape
from benchmark.predictors.base import Predictor, register
from benchmark._fold_align import (  # noqa: F401  (re-exported for old callers)
    LEGACY_TO_PASCAL, MANIFEST_PATH, PREDICTIONS_DIR,
    load_manifest, parse_manifest_dataset, get_model_folds, resolve_canonical_folds,
    normalize_combo_label, dl_declared_test_set, dl_predicted_bins,
    canon_set, load_model_vocab, fold_alignment_verdict,
)

log = logging.getLogger(__name__)


# ===================================================================
# Adapter base class
# ===================================================================


class DLAdapter(Predictor):
    """Common machinery for scGPT/GEARS/PRESAGE adapters.

    Subclasses set `model_key` (the lowercase token used in MANIFEST.json,
    e.g. 'scgpt') and `name` (the display name, e.g. 'scGPT').
    """
    needs_training = True  # but `fit` is a no-op — training is external
    is_external = True     # marks predictions adopted from atheus_csb (gated at adoption)
    model_key: str = ""

    def __init__(self):
        self._predictions_cache: Dict[Tuple[str, str, int], Path] = {}

    def fit(self, store, scenario, fold):
        # DL models are trained outside this codebase. `fit` just resolves which
        # source predictions h5ad backs the requested CANONICAL fold.
        manifest = load_manifest()
        folds = get_model_folds(manifest, store.dataset, scenario,
                                model_filter=self.model_key)
        if self.model_key not in folds:
            raise FileNotFoundError(
                f"{self.name}: no MANIFEST entry for {store.dataset}/{scenario}"
            )
        # Resolve the source fold by EXACT containment (handles scGPT's swapped
        # fold labels on replogle20/wessels23) instead of trusting the manifest
        # fold index. `remap[(model, source_fold)] = canonical_fold`.
        remap = resolve_canonical_folds(store.dataset, scenario, manifest)["remap"]
        source_folds = [sf for (m, sf), cf in remap.items()
                        if m == self.model_key and cf == fold]
        if not source_folds:
            raise FileNotFoundError(
                f"{self.name}: no source fold maps to canonical "
                f"{store.dataset}/{scenario}/fold{fold}"
            )
        match = [f for f in folds[self.model_key] if f["fold"] == source_folds[0]]
        if not match:
            raise FileNotFoundError(
                f"{self.name}: source fold {source_folds[0]} for canonical "
                f"{store.dataset}/{scenario}/fold{fold} not found on disk"
            )
        self._predictions_cache[(store.dataset, scenario, fold)] = match[0]["predictions_path"]

    def _load_predictions_h5ad(
        self, store: DatasetStore, scenario: str, fold: int,
    ) -> Path:
        key = (store.dataset, scenario, fold)
        if key in self._predictions_cache:
            return self._predictions_cache[key]
        # Not fit yet — try to resolve now.
        self.fit(store, scenario, fold)
        return self._predictions_cache[key]

    def predict(self, store, scenario, fold):
        import anndata as ad
        path = self._load_predictions_h5ad(store, scenario, fold)
        adata = ad.read_h5ad(str(path))

        split = store.split(scenario, fold)
        tbi = split.test_bin_indices_or_all(store.n_bins)
        tki = split.test_ko_indices
        out_shape = _expected_output_shape(store, tbi, tki)
        out = np.full(out_shape, np.nan, dtype=np.float32)

        model_genes = list(adata.var_names)
        model_to_idx = {g: i for i, g in enumerate(model_genes)}
        # Indices in store.gene_names for every gene the model predicts.
        store_gene_to_idx = {g: i for i, g in enumerate(store.gene_names)}
        common = [(model_to_idx[g], store_gene_to_idx[g])
                  for g in model_genes if g in store_gene_to_idx]
        if not common:
            log.warning("%s: zero gene overlap with %s", self.name, store.dataset)
            return out
        model_idx = np.array([m for m, _ in common], dtype=np.int64)
        store_idx = np.array([s for _, s in common], dtype=np.int64)

        # Build per-(condition, covariate) → row index in adata
        cond_arr = adata.obs["condition"].astype(str).values
        # Match conditions on a guide-resolved key so the DL's '+N' guide encoding
        # (e.g. 'FDPS+HUS1+2') matches our '_N' labels ('FDPS+HUS1_2'); otherwise the
        # guide-suffixed combos get all-NaN predictions (replogle20). No-op for
        # single genes / guide-free combos.
        cond_norm = np.array([normalize_combo_label(c) for c in cond_arr])
        cov_arr = (adata.obs["covariate"].astype(str).values
                   if "covariate" in adata.obs.columns else None)
        # Match the covariate (cell line / cell type) case-INSENSITIVELY: some DL
        # prediction h5ads store it lowercase (e.g. 'k562') while a dataset's
        # cell_type/bin names may be uppercase ('K562'). An exact compare would
        # silently match zero rows -> all-NaN predictions (replogle22).
        cov_arr_lc = (np.array([str(c).lower() for c in cov_arr])
                      if cov_arr is not None else None)
        # Get predictions matrix
        X = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
        X = X.astype(np.float32)

        n_missing = 0
        for j, ko_g in enumerate(tki):
            ko_name = store.ko_names[int(ko_g)]
            ko_norm = normalize_combo_label(ko_name)
            for i, b in enumerate(tbi):
                bin_name = store.bin_names[int(b)]
                mask = (cond_norm == ko_norm)
                if cov_arr is not None and any(c not in {"", "none", "None", "nan", "NaN"}
                                                for c in set(cov_arr)):
                    mask &= (cov_arr_lc == str(bin_name).lower())
                rows = np.where(mask)[0]
                if len(rows) == 0:
                    n_missing += 1   # no matching DL row → this (bin,ko) stays NaN
                    continue
                pred_abs = X[rows].mean(axis=0)
                # Convert to delta vs control. Use only common genes; others stay NaN.
                ctrl = store.ctrl_bulk[int(b), store_idx]
                out[i, j, store_idx] = pred_abs[model_idx] - ctrl
        if n_missing:
            log.warning(
                "%s: %d/%d (cell_type, perturbation) cells had no matching prediction "
                "row in the DL output → left NaN (skipped from eval) for %s/%s/fold%d",
                self.name, n_missing, len(tbi) * len(tki), store.dataset, scenario, fold,
            )
        return out

    def check_alignment(self, store, scenario, fold, *, preds) -> Tuple[bool, str]:
        """Adoption gate (uniform across regimes): verify this DL fold's predictions
        actually correspond to OUR (scenario, fold) test set before they are written
        as a benchmark `predictions.npz`. Delegates the decision to
        `_fold_align.fold_alignment_verdict`. Returns (ok, reason)."""
        path = self._predictions_cache.get((store.dataset, scenario, fold))
        if path is None:
            return False, f"{self.name}: source predictions not resolved (fit not run)"
        split = store.split(scenario, fold)
        tki = split.test_ko_indices
        tbi = split.test_bin_indices_or_all(store.n_bins)
        if scenario == "UnseenCell":
            # pert-degenerate → judge by scoreability of the held-out test cell
            held = {str(store.bin_names[int(b)]).lower() for b in tbi}
            present = dl_predicted_bins(path)
            return fold_alignment_verdict(
                scenario=scenario, model=self.model_key,
                held_out_bins=held, present_bins=present)
        h5ad_fold = canon_set(store.ko_names[int(i)] for i in tki)
        declared = dl_declared_test_set(path)
        # covered = perts the adapter actually populated (a row is covered if any
        # bin is non-all-NaN), as canonical keys.
        covered = canon_set(
            store.ko_names[int(ko_g)] for j, ko_g in enumerate(tki)
            if not np.isnan(preds[:, j, :]).all())
        vocab = load_model_vocab(self.model_key, store.dataset)
        return fold_alignment_verdict(
            scenario=scenario, model=self.model_key,
            declared=declared, h5ad_fold=h5ad_fold, covered=covered, vocab=vocab)

    def save_weights(self, path):
        """Save a pointer file — actual weights live elsewhere."""
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        # Tuple keys (ds, sc, fold) are not JSON-serializable; encode as "ds|sc|fold".
        ptr = {"|".join(str(p) for p in k): str(v)
               for k, v in self._predictions_cache.items()}
        np.savez(str(path), pointer=np.array(json.dumps(ptr)))

    @classmethod
    def load_weights(cls, path):
        # DL "weights" are only a pointer to the external predictions h5ad; the
        # prediction-path cache is re-resolved from the manifest on demand in
        # fit()/predict(), so a fresh instance is all that's needed.
        return cls()


# ===================================================================
# Concrete adapters
# ===================================================================


@register
class ScGPT(DLAdapter):
    name = "scGPT"
    model_key = "scgpt"
    scenarios = ["UnseenPert", "UnseenCell", "UnseenBoth", "UnseenPair", "UnseenCombo"]


@register
class GEARS(DLAdapter):
    name = "GEARS"
    model_key = "gears"
    scenarios = ["UnseenPert", "UnseenCell", "UnseenBoth", "UnseenPair", "UnseenCombo"]


@register
class PRESAGE(DLAdapter):
    name = "PRESAGE"
    model_key = "presage"
    scenarios = ["UnseenPert", "UnseenCell", "UnseenBoth", "UnseenPair", "UnseenCombo"]


__all__ = [
    "ScGPT", "GEARS", "PRESAGE", "DLAdapter",
    "load_manifest", "parse_manifest_dataset", "get_model_folds",
    "resolve_canonical_folds", "normalize_combo_label",
]
