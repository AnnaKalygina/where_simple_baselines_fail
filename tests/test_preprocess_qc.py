"""Preprocessing QC + panel-determinism unit tier — the fast counterpart to the
new verify.py L0 checks. Covers the parts of `data/_utils.py` that had no test
coverage at all: label cleanup, per-cell QC, guide parsing, HVG selection and the
condition floor / control cap. No data files, no torch.
"""
import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

import data._utils as u


def _synth_counts(*, n_perts=4, cells_per=30, n_genes=40, n_mito=3,
                  mito_frac=0.02, seed=0):
    """Synthetic raw-count AnnData with MT- genes carrying a known fraction."""
    rng = np.random.RandomState(seed)
    conds = ["control"] + [f"g{i}" for i in range(n_perts)]
    cond = [c for c in conds for _ in range(cells_per)]
    n_cells = len(cond)
    genes = [f"GENE{i}" for i in range(n_genes - n_mito)] + \
            [f"MT-X{i}" for i in range(n_mito)]
    X = rng.poisson(5, size=(n_cells, n_genes)).astype(np.float32)
    # force the MT block to a known share of each cell's total
    non_mt_tot = X[:, :-n_mito].sum(axis=1)
    X[:, -n_mito:] = 0
    X[:, -1] = np.round(non_mt_tot * mito_frac / (1 - mito_frac))
    obs = pd.DataFrame({"condition": cond, "cell_type": ["ct0"] * n_cells,
                        "batch": ["b0"] * n_cells})
    A = ad.AnnData(X=csr_matrix(X), obs=obs.reset_index(drop=True))
    A.var_names = genes
    return A


# --------------------------------------------------------------------------
# mitochondrial fraction
# --------------------------------------------------------------------------

def test_mito_fraction_computed_from_mt_genes():
    A = _synth_counts(mito_frac=0.05)
    frac, prov = u.mito_fraction(A)
    assert "MT- genes" in prov
    assert np.allclose(frac, 0.05, atol=0.01)


def test_mito_fraction_falls_back_to_obs_and_infers_units():
    """Shipped columns are NOT unit-consistent across our sources: replogle22
    ships a 0-1 fraction, everyone else a 0-100 percentage. The fallback must
    infer, and must return a fraction either way."""
    A = _synth_counts()
    A = A[:, [g for g in A.var_names if not g.startswith("MT-")]].copy()  # no MT genes
    A.obs["percent_mito"] = 7.5                                          # percentage
    frac, prov = u.mito_fraction(A)
    assert "percent/100" in prov and np.allclose(frac, 0.075)

    A.obs["percent_mito"] = 0.075                                        # fraction
    frac, prov = u.mito_fraction(A)
    assert "as fraction" in prov and np.allclose(frac, 0.075)


def test_mito_threshold_is_a_bounded_mad_rule():
    rng = np.random.RandomState(0)
    # clean dataset: MAD is tiny, so the floor binds
    assert u.mito_threshold(rng.normal(0.02, 0.005, 5000)) == 0.10
    # high-baseline dataset (wessels23-like monocytes): adapts ABOVE the floor
    # instead of deleting a quarter of the cells at a fixed 0.10
    thr = u.mito_threshold(rng.normal(0.086, 0.02, 5000))
    assert 0.10 < thr < 0.20
    # pathological spread is capped, so the gate never switches itself off
    assert u.mito_threshold(rng.normal(0.30, 0.20, 5000)) == 0.20
    assert u.mito_threshold(np.array([])) == 0.10


def test_apply_cell_qc_removes_high_mito_and_records_skips():
    A = _synth_counts(mito_frac=0.02, cells_per=40)
    A.X = A.X.tolil()
    A.X[:20, -1] = 10_000            # first 20 cells -> overwhelmingly mitochondrial
    A.X = A.X.tocsr()
    out = u.apply_cell_qc(A, dataset_name="synth", min_genes=0)
    assert out.n_obs == A.n_obs - 20


# --------------------------------------------------------------------------
# guide identity / MOI
# --------------------------------------------------------------------------

def test_condition_arity():
    assert u.condition_arity("control") == 0
    assert u.condition_arity("GENEA") == 1
    assert u.condition_arity("GENEA+GENEB") == 2
    assert u.condition_arity("GENEA@0.5") == 1


def test_guide_target_sets_drops_non_targeting_tokens():
    A = _synth_counts(n_perts=2, cells_per=2)          # 3 conditions x 2 cells
    A.obs["guide_id"] = ["STAT1_3;NO_SITE_411", "NO_SITE_36;ONE_NON-GENE_SITE_9",
                         "A_1;B_2", "nan", "C_1", "D_1"]
    ts = u.guide_target_sets(A, "guide_id")
    assert ts.iloc[0] == {"STAT1"}          # NT token dropped
    assert ts.iloc[1] == set()              # assigned, but purely non-targeting
    assert ts.iloc[2] == {"A", "B"}
    assert ts.iloc[3] is None               # unassigned is distinct from empty


def test_guide_violation_mask_is_arity_aware_and_spares_unassigned():
    A = _synth_counts(n_perts=1, cells_per=3)          # control + g0
    A.obs["condition"] = ["control", "control", "control", "g0", "g0", "g0"]
    n_targets = pd.Series([0, 1, np.nan, 1, 2, np.nan])
    mask, stats = u.guide_violation_mask(A, n_targets=n_targets)
    # control with 1 targeting guide -> violation; combo-consistent cells -> not;
    # NaN (unassigned) -> never flagged
    assert list(mask) == [False, True, False, False, True, False]
    assert stats["n_unassigned"] == 2 and stats["n_violations"] == 2


def test_guide_class_target_counts():
    A = _synth_counts(n_perts=1, cells_per=3)
    A.obs["Guide.Class"] = ["NT", "Single", "Dual"] * 2
    n = u.guide_class_target_counts(A, "Guide.Class", {"NT": 0, "Single": 1, "Dual": 2})
    assert list(n) == [0, 1, 2, 0, 1, 2]


# --------------------------------------------------------------------------
# HVG panel determinism — the invariant this whole rebuild exists to establish
# --------------------------------------------------------------------------

def test_select_hvg_requires_a_counts_layer():
    A = _synth_counts()
    with pytest.raises(ValueError, match="raw counts"):
        u.select_hvg(A, n_top_genes=10)


def test_select_hvg_is_deterministic():
    A = _synth_counts(n_genes=60)
    A.layers["counts"] = A.X.copy()
    B = A.copy()
    u.select_hvg(A, n_top_genes=20)
    u.select_hvg(B, n_top_genes=20)
    assert list(A.var_names[A.var["highly_variable"]]) == \
           list(B.var_names[B.var["highly_variable"]])


def test_select_hvg_caps_to_available_genes():
    A = _synth_counts(n_genes=30)
    A.layers["counts"] = A.X.copy()
    u.select_hvg(A, n_top_genes=8192)
    assert int(A.var["highly_variable"].sum()) == 30


# --------------------------------------------------------------------------
# condition floor / control cap (the former mean-based downsampler)
# --------------------------------------------------------------------------

def test_no_per_perturbation_cap_by_default():
    """Every cell of every surviving perturbation is kept — the mean-based cap
    (which made the gene panel a function of the label partition) is gone."""
    A = _synth_counts(n_perts=3, cells_per=50)
    out = u.downsample_per_condition(A, bin_col=None, control_cap=10_000)
    assert out.n_obs == A.n_obs


def test_condition_floor_drops_underpowered_conditions():
    A = _synth_counts(n_perts=2, cells_per=30)
    keep = np.ones(A.n_obs, dtype=bool)
    keep[np.where(A.obs["condition"].values == "g1")[0][5:]] = False  # g1 -> 5 cells
    A = A[keep].copy()
    out = u.downsample_per_condition(A, bin_col=None, min_cells_threshold=12)
    assert "g1" not in set(out.obs["condition"])
    assert "g0" in set(out.obs["condition"])


def test_control_cap_applies_and_is_reproducible():
    A = _synth_counts(n_perts=1, cells_per=100)
    a = u.downsample_per_condition(A, bin_col=None, control_cap=40)
    b = u.downsample_per_condition(A, bin_col=None, control_cap=40)
    n_ctrl = int((a.obs["condition"] == "control").sum())
    assert n_ctrl == 40
    assert list(a.obs_names) == list(b.obs_names)      # deterministic


def test_condition_draws_are_independent_of_each_other():
    """Adding a perturbation must not change which cells another one keeps.

    The old implementation drew every condition from one shared stream, so the
    kept set moved whenever the condition list changed.
    """
    A = _synth_counts(n_perts=2, cells_per=60)
    kept_a = set(u.downsample_per_condition(A, bin_col=None, control_cap=20,
                                            pert_cap=20).obs_names)
    B = _synth_counts(n_perts=3, cells_per=60)         # one EXTRA perturbation
    kept_b = set(u.downsample_per_condition(B, bin_col=None, control_cap=20,
                                            pert_cap=20).obs_names)
    g0_a = {c for c in kept_a if A.obs.loc[c, "condition"] == "g0"}
    g0_b = {c for c in kept_b if B.obs.loc[c, "condition"] == "g0"}
    assert g0_a == g0_b


# --------------------------------------------------------------------------
# counts, depth and normalisation — the jiang24/mcfaline23 regressions
# --------------------------------------------------------------------------

def _lognorm_with_counts(n_cells=200, n_genes=20, seed=0):
    """mcfaline23/jiang24 shape: log1p X *and* the source's own integral counts."""
    rng = np.random.RandomState(seed)
    X = rng.poisson(5, size=(n_cells, n_genes)).astype(np.float32)
    genes = [f"G{i}" for i in range(n_genes - 2)] + ["MT-A", "MT-B"]
    A = ad.AnnData(X=csr_matrix(np.log1p(X)),
                   obs=pd.DataFrame({"condition": ["control"] * n_cells,
                                     "cell_type": ["ct0"] * n_cells}))
    A.var_names = genes
    A.layers["counts"] = csr_matrix(X)
    return A, X, genes


def test_mito_reads_the_counts_layer_never_un_logs_it():
    """A source can ship log1p X *and* its own integral counts layer.

    The old code un-logged whenever a dataset declared log-normalised X, so expm1
    hit integer counts, overflowed to inf, and turned the gate into "is the
    largest-count gene mitochondrial" — deleting 91.4% of jiang24 and 51.8% of
    mcfaline23, one of them silently passing `verify`.
    """
    A, X, genes = _lognorm_with_counts()
    frac, prov = u.mito_fraction(A, dataset_name="synth")
    assert "counts" in prov and "expm1" not in prov
    assert np.isfinite(frac).all()
    expected = X[:, -2:].sum(axis=1) / X.sum(axis=1)
    assert np.allclose(frac, expected, atol=1e-6)


def test_mito_refuses_a_matrix_that_is_not_counts():
    """No counts layer and a log-transformed X is not something to guess about."""
    A, X, genes = _lognorm_with_counts()
    B = ad.AnnData(X=A.X.copy(), obs=A.obs.copy())   # log1p X, NO counts layer
    B.var_names = genes
    with pytest.raises(ValueError, match="not integral"):
        u.mito_fraction(B, dataset_name="synth")


def test_mito_uses_the_declared_depth_when_the_matrix_is_a_panel():
    """A gene panel truncates the DENOMINATOR, not the MT numerator.

    jiang24 holds 46% of each cell, so MT/rowsum reads 0.1446 against a true
    0.0526. The fix is the shipped depth, not the producer's mito column.
    """
    A, X, genes = _lognorm_with_counts()
    true_depth = X.sum(axis=1) * 2.0                  # matrix holds half the cell
    A.obs["ncounts"] = true_depth

    panel, _ = u.mito_fraction(A, dataset_name="synth")
    true, _ = u.mito_fraction(A, depth_column="ncounts", dataset_name="synth")
    assert np.allclose(panel, 2.0 * true, atol=1e-6)
    assert np.median(true) < np.median(panel)


def test_mito_cross_check_survives_a_zero_median():
    """A clean dataset must not fail on `computed == shipped == 0`."""
    A, X, genes = _lognorm_with_counts()
    A.layers["counts"] = csr_matrix(np.asarray(
        A.layers["counts"].todense()) * np.array([1] * 18 + [0, 0]))
    A.obs["percent_mito"] = 0.0
    frac, _ = u.mito_fraction(A, dataset_name="synth")
    assert np.allclose(frac, 0.0)


def test_shipped_mito_column_keeps_missing_values_missing():
    """A missing measurement must not read as a clean cell."""
    A, X, genes = _lognorm_with_counts()
    A.obs["percent_mito"] = 5.0
    A.obs.loc[A.obs.index[:10], "percent_mito"] = np.nan
    v, prov = u._shipped_mito_fraction(A)
    assert np.isnan(v[:10]).all() and "percent" in prov


# --------------------------------------------------------------------------
# count retention — the declaration that was set from gene count and got two
# datasets backwards
# --------------------------------------------------------------------------

def test_count_retention_raises_when_a_panel_claims_to_be_complete():
    A = _synth_counts(cells_per=20)
    A.obs["ncounts"] = np.asarray(A.X.sum(axis=1)).ravel() * 2.0   # half is missing
    with pytest.raises(ValueError, match="SOURCE_DEPTH_COLUMN"):
        u.count_retention(A, depth_column=None, dataset_name="synth")


def test_count_retention_accepts_a_complete_matrix_and_records_it():
    A = _synth_counts(cells_per=20)
    A.obs["ncounts"] = np.asarray(A.X.sum(axis=1)).ravel()
    r = u.count_retention(A, depth_column=None, dataset_name="synth")
    assert r == pytest.approx(1.0, abs=1e-6)
    assert A.uns["count_retention"] == pytest.approx(1.0, abs=1e-6)


def test_count_retention_raises_when_the_column_is_smaller_than_the_matrix():
    A = _synth_counts(cells_per=20)
    A.obs["ncounts"] = np.asarray(A.X.sum(axis=1)).ravel() * 0.5
    with pytest.raises(ValueError, match="more than all of it"):
        u.count_retention(A, depth_column="ncounts", dataset_name="synth")


# --------------------------------------------------------------------------
# normalisation
# --------------------------------------------------------------------------

def test_normalize_from_counts_matches_cp10k_without_a_depth_column():
    """The no-depth-column path must reproduce scanpy exactly."""
    import scanpy as sc
    A = _synth_counts(cells_per=20)
    A.layers["counts"] = A.X.copy()
    B = A.copy()
    u.normalize_from_counts(A, target_sum=1e4, depth_column=None)
    sc.pp.normalize_total(B, target_sum=1e4)
    assert np.allclose(np.asarray(A.X.todense()), np.asarray(B.X.todense()), rtol=1e-5)


def test_normalize_from_counts_divides_by_the_true_depth():
    """A cell of which only half is present must sum to half the target."""
    A = _synth_counts(cells_per=20)
    A.layers["counts"] = A.X.copy()
    A.obs["ncounts"] = np.asarray(A.X.sum(axis=1)).ravel() * 2.0
    u.normalize_from_counts(A, target_sum=1e4, depth_column="ncounts")
    assert np.allclose(np.asarray(A.X.sum(axis=1)).ravel(), 5e3, rtol=1e-5)


def test_normalize_from_counts_requires_a_counts_layer():
    A = _synth_counts(cells_per=20)
    with pytest.raises(ValueError, match="is missing"):
        u.normalize_from_counts(A, target_sum=1e4)


# --------------------------------------------------------------------------
# gene weights — the placeholder that took 82-89% of the weight mass
# --------------------------------------------------------------------------

def _deg_store(scores, padj, substituted):
    """Minimal AnnData carrying the tensors compute_deg_arrays reads."""
    n_bins, n_kos, n_genes = scores.shape
    A = ad.AnnData(X=csr_matrix(np.zeros((n_kos, n_genes), dtype=np.float32)),
                   obs=pd.DataFrame({"condition": [f"k{i}" for i in range(n_kos)]}))
    A.var_names = [f"G{i}" for i in range(n_genes)]
    A.uns["pseudobulk"] = {"ko_names": np.array([f"k{i}" for i in range(n_kos)],
                                                dtype=object),
                           "bin_names": np.array(["all"] * n_bins, dtype=object)}
    A.uns["scores_matrix_all"] = scores.astype(np.float32)
    A.uns["pvals_adj_matrix_all"] = padj.astype(np.float32)
    A.uns["deg_fc_substituted_all"] = substituted
    return A


def test_weights_ignore_genes_the_test_did_not_call():
    scores = np.array([[[4.0, 2.0, 1.0, 8.0]]], dtype=np.float32)
    padj = np.array([[[0.01, 0.01, 0.01, 0.90]]], dtype=np.float32)   # last not called
    sub = np.zeros(scores.shape, dtype=bool)
    w = u.compute_deg_arrays(_deg_store(scores, padj, sub))["per_pert_weights"]
    assert w[0, 0, 3] == 0.0, "a non-significant gene must carry no weight"
    assert w[0, 0, 0] == pytest.approx(1.0)      # 4.0 is the largest USABLE score
    assert w[0, 0, 1] == pytest.approx(0.5)      # linear, not squared
    assert w[0, 0, 2] == pytest.approx(0.25)


def test_weights_exclude_substituted_fold_changes_even_when_significant():
    """Significance alone still left 47-49% of the weight on the placeholder."""
    scores = np.array([[[4.3219, 2.0, 1.0]]], dtype=np.float32)
    padj = np.array([[[0.001, 0.01, 0.01]]], dtype=np.float32)   # ALL significant
    sub = np.array([[[True, False, False]]])
    w = u.compute_deg_arrays(_deg_store(scores, padj, sub))["per_pert_weights"]
    assert w[0, 0, 0] == 0.0, "a substituted fold change is not a measurement"
    assert w[0, 0, 1] == pytest.approx(1.0)      # 2.0 is now the row maximum
    assert w[0, 0, 2] == pytest.approx(0.5)


def test_weights_are_linear_in_effect_size():
    scores = np.array([[[10.0, 5.0, 2.5, 1.0]]], dtype=np.float32)
    padj = np.full(scores.shape, 0.001, dtype=np.float32)
    w = u.compute_deg_arrays(
        _deg_store(scores, padj, np.zeros(scores.shape, dtype=bool))
    )["per_pert_weights"]
    assert np.allclose(w[0, 0], [1.0, 0.5, 0.25, 0.1], atol=1e-6)


def test_a_row_with_only_substituted_hits_is_an_error_not_a_silent_zero():
    scores = np.array([[[4.3219, 4.3219]]], dtype=np.float32)
    padj = np.array([[[0.001, 0.001]]], dtype=np.float32)
    sub = np.ones(scores.shape, dtype=bool)
    with pytest.raises(ValueError, match="no weight at all"):
        u.compute_deg_arrays(_deg_store(scores, padj, sub))


def test_deg_mask_and_directions_are_unchanged_by_the_weight_fix():
    """gsea_* and frac_correct_direction must not move."""
    scores = np.array([[[4.0, -2.0, 1.0]]], dtype=np.float32)
    padj = np.array([[[0.01, 0.01, 0.90]]], dtype=np.float32)
    out = u.compute_deg_arrays(
        _deg_store(scores, padj, np.array([[[True, False, False]]])))
    assert out["deg_mask"][0, 0].tolist() == [True, True, False]
    assert out["deg_directions"][0, 0].tolist() == [1, -1, 0]


def test_deg_arrays_prefers_the_all_cell_call_over_a_half():
    scores = np.array([[[4.0, 2.0]]], dtype=np.float32)
    padj = np.full(scores.shape, 0.001, dtype=np.float32)
    A = _deg_store(scores, padj, np.zeros(scores.shape, dtype=bool))
    A.uns["scores_matrix_first_half"] = np.array([[[1.0, 99.0]]], dtype=np.float32)
    A.uns["pvals_adj_matrix_first_half"] = padj.copy()
    w = u.compute_deg_arrays(A)["per_pert_weights"]
    assert w[0, 0, 0] == pytest.approx(1.0), "must read the all-cell scores, not the half"


def test_cell_qc_refuses_a_catastrophic_mito_removal():
    """A gate that deletes most of the dataset is a bug, not strictness."""
    A = _synth_counts(mito_frac=0.02, cells_per=40)
    A.X = A.X.tolil()
    A.X[:, -1] = 10_000              # every cell overwhelmingly mitochondrial
    A.X = A.X.tocsr()
    with pytest.raises(ValueError, match="not biology"):
        u.apply_cell_qc(A, dataset_name="synth", min_genes=0)


def test_count_retention_ignores_a_constant_column_that_is_not_a_depth():
    """replogle20 ships `UMI_count` as the constant 100 for every cell.

    Matched by name, it produced a 12384% "retention" that was recorded in the
    artefact and raised nothing, because the too-high guard only ran for an
    explicitly declared column.
    """
    A = _synth_counts(cells_per=20)
    A.obs["UMI_count"] = 100.0                  # constant: not a per-cell depth
    r = u.count_retention(A, depth_column=None, dataset_name="synth")
    assert r is None, "a constant column must not be treated as a depth"
    assert np.isnan(A.uns["count_retention"])


def test_count_retention_still_catches_a_truncated_matrix_via_autodetect():
    """Skipping bad candidates must not skip the case being hunted."""
    A = _synth_counts(cells_per=20)
    A.obs["UMI_count"] = 100.0                                     # junk candidate
    A.obs["ncounts"] = np.asarray(A.X.sum(axis=1)).ravel() * 2.0    # real one, 50%
    with pytest.raises(ValueError, match="SOURCE_DEPTH_COLUMN"):
        u.count_retention(A, depth_column=None, dataset_name="synth")
