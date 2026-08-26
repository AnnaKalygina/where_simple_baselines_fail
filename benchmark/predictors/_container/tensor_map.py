"""Map a container's ``predictions.h5ad`` onto the benchmark's delta tensor.

This is the host-side boundary between a training container's native output and
the benchmark's evaluation grid. A container emits ``predictions.h5ad`` of
**absolute log1p expression**; this module reduces it to the
``(n_test_bins, n_test_kos, n_genes)`` **delta** tensor the benchmark scores
against ``store.first_half_deltas``.

Single-cell vs pseudobulk is a NON-ISSUE here, by construction: for each test
``(cell_type, perturbation)`` we take **all** matching rows and average them
(``X[rows].mean(axis=0)``). If the model emitted one row per condition
(pseudobulk models: GEARS, PRESAGE) the mean is over 1 row = that row; if it
emitted many cells (single-cell models: scGPT, cellflow, STATE) the mean is the
pseudobulk of those cells. Either way the benchmark sees the same thing — the
mean absolute profile per cell, minus control. The container's prediction
*space* only changes how many rows arrive; the reduction is identical.

The three mapping primitives are **preserved verbatim** from the (to-be-removed)
``benchmark.predictors.dl_adapter``/``_fold_align`` path so this keeps working
after that machinery is deleted (see ``TRAINING_INFRA_PLAN.md`` §Decommissioning
& §Engineering-quality commitments #1):
  * ``normalize_combo_label`` — collapses the DL '+N'/'_N' guide encoding so a
    model's ``FDPS+HUS1+2`` matches our ``FDPS+HUS1_2`` (no-op for single genes).
  * gene-set **intersection** by symbol — genes the model doesn't predict stay
    NaN (the metric layer skips them), never zero-filled.
  * **case-insensitive covariate** match — a model's ``k562`` matches our ``K562``.
"""
from __future__ import annotations

import argparse
import logging

import numpy as np

log = logging.getLogger(__name__)

# Covariate values that mean "no cell axis" (GEARS/scGPT write a dummy 'none').
_DEGENERATE_COV = {"", "none", "None", "nan", "NaN", "NOTHING", "nothing"}


def normalize_combo_label(c: str) -> str:
    """Guide-resolved canonical key for a (combo) condition — PRESERVED verbatim
    from ``benchmark._fold_align.normalize_combo_label``.

    The DL prediction h5ads write per-guide suffixes with '+' (the same char as
    the combo separator): our 'FDPS+HUS1_2' / 'FDPS_2+HUS1_2' appear as
    'FDPS+HUS1+2' / 'FDPS+2+HUS1+2'. Split on '+', reattach any pure-numeric token
    as a '_N' guide suffix of the preceding gene, then sort — collapsing BOTH
    encodings to one key while keeping guide variants distinct. No-op for
    guide-free combos and single genes (gene symbols are never purely numeric).
    """
    genes: list = []
    for t in str(c).split("+"):
        if t == "":
            continue
        if t.isdigit() and genes:
            genes[-1] = f"{genes[-1]}_{t}"
        else:
            genes.append(t)
    return "+".join(sorted(genes))


def _assert_unique(names, what: str) -> None:
    """Gene axes are used as dict keys; duplicates must not pass silently."""
    seen, dupes = set(), []
    for n in names:
        if n in seen and len(dupes) < 5:
            dupes.append(str(n))
        seen.add(n)
    if dupes:
        raise ValueError(
            f"{what}: duplicate gene name(s) {dupes} — a name-keyed axis cannot "
            f"be built from a non-unique index without silently losing genes")


def map_predictions_to_delta_tensor(adata, store, scenario: str, fold: int, *,
                                    model_name: str = "container") -> np.ndarray:
    """Reduce a container's ``predictions.h5ad`` (``adata``) to the benchmark's
    ``(n_test_bins, n_test_kos, n_genes)`` delta tensor for ``(scenario, fold)``.

    Faithful port of the proven ``DLAdapter.predict`` mapping. Unmatched
    ``(bin, ko)`` cells and un-predicted genes are left NaN (skipped by metrics).
    """
    split = store.split(scenario, fold)
    tbi = split.test_bin_indices_or_all(store.n_bins)   # test cell-type indices (or all)
    tki = split.test_ko_indices                          # test perturbation indices
    out = np.full((len(tbi), len(tki), store.n_genes), np.nan, dtype=np.float32)

    # --- gene intersection (by symbol); model genes not in the store are dropped,
    #     store genes the model doesn't predict stay NaN. ---
    # Duplicate names would collapse to the LAST index here, silently dropping a
    # gene from the mapping and mis-attributing its prediction. Cheap to assert,
    # impossible to notice downstream.
    _assert_unique(adata.var_names, f"{model_name} predictions.h5ad var_names")
    _assert_unique(store.gene_names, f"{store.dataset} store.gene_names")
    model_to_idx = {g: i for i, g in enumerate(adata.var_names)}
    store_gene_to_idx = {g: i for i, g in enumerate(store.gene_names)}
    common = [(m, store_gene_to_idx[g]) for g, m in model_to_idx.items()
              if g in store_gene_to_idx]
    if not common:
        log.warning("%s: zero gene overlap with %s — all-NaN", model_name, store.dataset)
        return out
    model_idx = np.array([m for m, _ in common], dtype=np.int64)
    store_idx = np.array([s for _, s in common], dtype=np.int64)

    # --- condition + covariate keys ---
    cond_norm = np.array([normalize_combo_label(c)
                          for c in adata.obs["condition"].astype(str).values])
    cov_col = ("covariate" if "covariate" in adata.obs.columns
               else "cell_type" if "cell_type" in adata.obs.columns else None)
    cov_arr = adata.obs[cov_col].astype(str).values if cov_col else None
    cov_arr_lc = np.array([c.lower() for c in cov_arr]) if cov_arr is not None else None
    # Filter on covariate ONLY when the dataset has a real cell-type axis (n_bins>1)
    # AND the model actually wrote non-degenerate covariates. For single-bin
    # datasets (bin_names == ['all']) or a dummy 'none' covariate, match by
    # condition only and broadcast to the (single / all) test bin(s).
    use_cov = (store.n_bins > 1 and cov_arr is not None
               and any(c not in _DEGENERATE_COV for c in set(cov_arr)))

    X = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
    X = X.astype(np.float32)

    n_missing = 0
    for j, ko_g in enumerate(tki):
        ko_norm = normalize_combo_label(store.ko_names[int(ko_g)])
        for i, b in enumerate(tbi):
            mask = cond_norm == ko_norm
            if use_cov:
                mask = mask & (cov_arr_lc == str(store.bin_names[int(b)]).lower())
            rows = np.where(mask)[0]
            if len(rows) == 0:
                n_missing += 1          # no matching row → this (bin, ko) stays NaN
                continue
            pred_abs = X[rows].mean(axis=0)                 # 1 row (pseudobulk) or N (single-cell)
            ctrl = store.ctrl_bulk[int(b), store_idx]
            out[i, j, store_idx] = pred_abs[model_idx] - ctrl
    if n_missing:
        log.warning("%s: %d/%d (cell_type, perturbation) cells had no matching row "
                    "→ left NaN for %s/%s/fold%d",
                    model_name, n_missing, len(tbi) * len(tki),
                    store.dataset, scenario, fold)
    return out


def load_and_map(predictions_h5ad, store, scenario: str, fold: int, *,
                 model_name: str = "container") -> np.ndarray:
    """Read a container's ``predictions.h5ad`` from disk and map it to the tensor."""
    import anndata as ad
    adata = ad.read_h5ad(str(predictions_h5ad))
    return map_predictions_to_delta_tensor(adata, store, scenario, fold,
                                           model_name=model_name)


# ===================================================================
# Self-test: prove single-cell and pseudobulk row layouts give an IDENTICAL
# DIAGNOSTIC, not a test suite: this runs against a REAL dataset on disk, which
# CI does not have. The same properties are covered on a fabricated store in
# tests/test_container_predictor_e2e.py and run automatically; this exists to
# check the mapping against real gene names, condition labels and covariates
# before trusting a scored run. Keep both — they answer different questions.
#
# tensor, that the recovered delta matches a fabricated one, and that
# un-predicted genes stay NaN. Run:
#   python -m benchmark.predictors._container.tensor_map --selftest \
#          --dataset adamson16 --scenario UnseenPert --fold 0
# ===================================================================
def _selftest(dataset: str, scenario: str, fold: int) -> None:
    import anndata as ad
    import pandas as pd
    from benchmark.data_loader import DatasetStore

    store = DatasetStore(dataset)
    split = store.split(scenario, fold)
    tki = split.test_ko_indices
    assert len(tki) > 0, "no test perturbations"
    genes = list(store.gene_names)
    ng = len(genes)
    rng = np.random.default_rng(0)

    # Fabricate a per-ko absolute profile = control + a known delta.
    ko_names = [store.ko_names[int(k)] for k in tki]
    deltas = {k: (rng.standard_normal(ng).astype(np.float32) * 0.1) for k in ko_names}
    ctrl0 = store.ctrl_bulk[0].astype(np.float32)

    def _adata(rows_per_ko: int, gene_subset=None):
        gcols = gene_subset if gene_subset is not None else genes
        gidx = [genes.index(g) for g in gcols]
        rows, conds = [], []
        for k in ko_names:
            abs_prof = (ctrl0 + deltas[k])[gidx]
            for _ in range(rows_per_ko):
                rows.append(abs_prof)
                conds.append(k)
        X = np.vstack(rows).astype(np.float32)
        obs = pd.DataFrame({"condition": conds,
                            "covariate": ["K562"] * len(conds)},  # realistic, not 'all'
                           index=[f"c{i}" for i in range(len(conds))])
        return ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=gcols))

    t_pseudobulk = map_predictions_to_delta_tensor(_adata(1), store, scenario, fold)
    t_singlecell = map_predictions_to_delta_tensor(_adata(5), store, scenario, fold)

    # 1. per-cell (5 rows) and pseudobulk (1 row) give the SAME tensor, up to
    #    float32 rounding of the averaging (a real pseudobulk IS the cell mean).
    assert np.allclose(t_pseudobulk, t_singlecell, equal_nan=True, atol=1e-4, rtol=1e-4), \
        "single-cell vs pseudobulk tensors differ beyond float32 tolerance"
    # 2. recovered delta matches the fabricated delta (full gene overlap → all finite).
    for j, k in enumerate(ko_names):
        assert np.allclose(t_pseudobulk[0, j, :], deltas[k], atol=1e-4), \
            f"recovered delta != fabricated for {k}"
    assert np.isfinite(t_pseudobulk).all(), "unexpected NaN with full gene overlap"
    # 3. un-predicted genes stay NaN (partial overlap).
    half = genes[: ng // 2]
    t_partial = map_predictions_to_delta_tensor(_adata(1, gene_subset=half), store,
                                                scenario, fold)
    assert np.isfinite(t_partial[0, 0, : ng // 2]).all(), "predicted genes should be finite"
    assert np.isnan(t_partial[0, 0, ng // 2:]).all(), "un-predicted genes should be NaN"

    print(f"SELFTEST PASSED  ({dataset}/{scenario}/fold{fold}): "
          f"n_test_ko={len(tki)}, n_bins={store.n_bins}, n_genes={ng}; "
          f"single-cell==pseudobulk, delta recovered, missing-genes NaN")


def main() -> None:
    ap = argparse.ArgumentParser(description="Map predictions.h5ad → benchmark delta tensor.")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--predictions", help="path to a container's predictions.h5ad")
    ap.add_argument("--dataset")
    ap.add_argument("--scenario")
    ap.add_argument("--fold", type=int)
    a = ap.parse_args()

    if a.selftest:
        _selftest(a.dataset, a.scenario, a.fold)
        return
    if not (a.predictions and a.dataset and a.scenario is not None and a.fold is not None):
        ap.error("need --predictions --dataset --scenario --fold (or --selftest)")
    from benchmark.data_loader import DatasetStore
    store = DatasetStore(a.dataset)
    t = load_and_map(a.predictions, store, a.scenario, a.fold)
    finite = np.isfinite(t)
    print(f"tensor shape {t.shape}; finite {finite.sum()}/{t.size} "
          f"({100*finite.mean():.1f}%); NaN (bin,ko) cells "
          f"{int((~finite.reshape(t.shape[0]*t.shape[1], -1).any(axis=1)).sum())}")


if __name__ == "__main__":
    main()
