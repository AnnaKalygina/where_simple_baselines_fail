"""Synthetic pseudobulk sanity — the fast counterpart to verify.py L2 (which runs
the same math on the real h5ads). No torch, no data files."""
import anndata as ad
import numpy as np
import pandas as pd

import data._utils as u


def _synth_expr(*, n_singles=3, cells_per=12, n_genes=6, seed=0):
    rng = np.random.RandomState(seed)
    conds = ["control"] + [f"g{i}" for i in range(n_singles)]
    cond = []
    for c in conds:
        cond += [c] * cells_per
    obs = pd.DataFrame({"cell_type": ["ct0"] * len(cond), "condition": cond})
    X = rng.rand(len(cond), n_genes).astype(np.float32)
    A = ad.AnnData(X=X, obs=obs.reset_index(drop=True))
    u.assign_tech_dup_split(A, bin_col=None)   # 50/50 first/second per condition
    return A


def test_pseudobulk_matches_manual_means():
    A = _synth_expr()
    pb = u.compute_pseudobulk(A, pert_col="condition", bin_col=None)
    X = np.asarray(A.X)
    cond = A.obs["condition"].values
    half = A.obs["tech_dup_split"].values

    # ctrl_bulk = mean of ALL control cells
    ctrl_ref = X[cond == "control"].mean(axis=0)
    assert np.allclose(pb["ctrl_bulk"][0], ctrl_ref, atol=1e-5)

    # first/second_half_bulk per ko == mean of that ko's cells in each half
    ko_names = list(pb["ko_names"])
    for ko in ("g0", "g1"):
        ki = ko_names.index(ko)
        for key, h in (("first_half_bulk", "first_half"),
                       ("second_half_bulk", "second_half")):
            ref = X[(cond == ko) & (half == h)].mean(axis=0)
            assert np.allclose(pb[key][0, ki], ref, atol=1e-5), f"{key}/{ko}"


def test_pseudobulk_all_bulk_is_count_weighted_mean():
    """all_bulk (reconstructed by DatasetStore) == count-weighted half means."""
    A = _synth_expr()
    pb = u.compute_pseudobulk(A, pert_col="condition", bin_col=None)
    n1 = pb["n_cells_first"].astype(np.float64)[:, :, None]
    n2 = pb["n_cells_second"].astype(np.float64)[:, :, None]
    total = n1 + n2
    all_bulk = np.divide(n1 * pb["first_half_bulk"] + n2 * pb["second_half_bulk"],
                         total, out=np.zeros_like(n1 * pb["first_half_bulk"]),
                         where=total > 0)
    X = np.asarray(A.X)
    cond = A.obs["condition"].values
    ko_names = list(pb["ko_names"])
    ki = ko_names.index("g0")
    ref = X[cond == "g0"].mean(axis=0)   # all g0 cells, both halves
    assert np.allclose(all_bulk[0, ki], ref, atol=1e-5)
