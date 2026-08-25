"""Unit tests for split-fold generation logic and the shared invariant predicates.

These run on tiny synthetic AnnData (no real h5ads, no torch) and are the fast
counterpart to benchmark/verify.py's artifact-level checks. The headline test is
``test_single_bin_unseen_pert_diverges_from_block`` + the convention assertions:
together they pin down the multi-bin UnseenPert convention so the xatlas-class
"wrong fold variant" bug is caught at the function level, not just on a 50 GB
artifact.
"""
import anndata as ad
import numpy as np
import pandas as pd
import pytest

import data._utils as u


def _synth(*, n_bins=3, n_singles=12, combos=(), cells_per=4):
    """Build a tiny AnnData with `cell_type` + `condition` columns."""
    bins = [f"ct{i}" for i in range(n_bins)]
    conds = ["control"] + [f"g{i}" for i in range(n_singles)] + list(combos)
    ct, cond = [], []
    for b in bins:
        for c in conds:
            ct += [b] * cells_per
            cond += [c] * cells_per
    obs = pd.DataFrame({"cell_type": ct, "condition": cond})
    X = np.zeros((len(obs), 4), dtype=np.float32)
    return ad.AnnData(X=X, obs=obs.reset_index(drop=True))


def _fold_masks(A, scenario, fold):
    return u._grid_masks_from_obs(
        A, f"split_{scenario}_fold_{fold}", pert_col="condition", bin_col="cell_type")


def _test_ko_names(A, scenario, fold):
    tr, te, nc = _fold_masks(A, scenario, fold)
    kos = sorted(map(str, A.obs["condition"].unique()))
    cols = np.where(te.any(axis=0) & nc)[0]
    return {kos[i] for i in cols}


# --------------------------------------------------------------------------- #
# Pure predicates
# --------------------------------------------------------------------------- #
def test_predicates_basic():
    tr = np.array([[1, 1, 0, 0], [1, 1, 0, 0]], bool)
    te = np.array([[0, 0, 1, 1], [0, 0, 1, 1]], bool)
    assert u.check_no_leakage(tr, te)
    assert not u.check_no_leakage(tr, tr)
    assert u.check_no_empty_fold(tr, te)
    assert not u.check_no_empty_fold(np.zeros((2, 4), bool))
    assert u.check_pert_consistent_across_bins(te)
    assert not u.check_pert_consistent_across_bins(np.array([[1, 0], [0, 0]], bool))
    nc = np.array([False, True, True, True])
    assert u.check_full_coverage([np.array([[0, 1, 0, 0]], bool),
                                  np.array([[0, 0, 1, 1]], bool)], nc)
    assert not u.check_full_coverage([np.array([[0, 1, 0, 0]], bool),
                                      np.array([[0, 1, 1, 1]], bool)], nc)
    assert u.check_unseenpert_subset_of_unseenboth({"A", "B"}, {"A", "B", "C"})
    assert not u.check_unseenpert_subset_of_unseenboth({"A", "Z"}, {"A", "B"})
    # UnseenPair: every column (pert) must be train in ≥1 row (bin)
    assert u.check_unseenpair_pert_in_train_at_least_once(
        np.array([[1, 0, 1], [0, 1, 0], [0, 0, 1]], bool))
    assert not u.check_unseenpair_pert_in_train_at_least_once(
        np.array([[1, 0, 0], [0, 0, 0], [0, 0, 1]], bool))  # column 1 all-False


# --------------------------------------------------------------------------- #
# Multi-bin UnseenPert (the perturbench block partition)
# --------------------------------------------------------------------------- #
def test_multibin_unseen_pert_consistent_and_covers():
    A = _synth(n_bins=3, n_singles=12)
    u.assign_split_folds_unseen_pert_multibin(A, n_folds=5, seed=0)
    masks = []
    for f in range(5):
        tr, te, nc = _fold_masks(A, "UnseenPert", f)
        assert u.check_no_leakage(tr[:, nc], te[:, nc])
        assert u.check_pert_consistent_across_bins(te[:, nc])
        assert te[:, nc].any()
        masks.append(te[:, nc])
    # every non-control perturbation is tested in exactly one fold
    assert u.check_full_coverage(masks, np.ones(masks[0].shape[1], bool))


def test_multibin_unseen_pert_is_deterministic():
    A1 = _synth(n_bins=3, n_singles=12)
    A2 = _synth(n_bins=3, n_singles=12)
    u.assign_split_folds_unseen_pert_multibin(A1, n_folds=5, seed=0)
    u.assign_split_folds_unseen_pert_multibin(A2, n_folds=5, seed=0)
    for f in range(5):
        col = f"split_UnseenPert_fold_{f}"
        assert (A1.obs[col].values == A2.obs[col].values).all()


def test_unseen_pert_subset_of_unseen_both_block():
    """The perturbench convention the verifier enforces (L1d): per fold, the
    multibin UnseenPert test set ⊆ the UnseenBoth held-out block."""
    A = _synth(n_bins=3, n_singles=15)
    u.assign_split_folds_unseen_pert_multibin(A, n_folds=5, seed=0)
    u.assign_split_folds_unseen_both(A, bin_col="cell_type", n_folds=5, seed=0)
    for f in range(5):
        up = _test_ko_names(A, "UnseenPert", f)
        ub = _test_ko_names(A, "UnseenBoth", f)
        assert up and ub
        assert u.check_unseenpert_subset_of_unseenboth(up, ub)


def test_single_bin_unseen_pert_diverges_from_block():
    """The xatlas situation: the single-bin round-robin still produces a VALID
    (leakage-free, perturbation-consistent) UnseenPert — so assert_split_invariants
    PASSES — but it does NOT follow the seed-0 block partition, so it is not a
    subset of UnseenBoth. This is exactly why the artifact-level convention check
    is INFO (not a hard failure) for non-perturbench multi-bin datasets."""
    A = _synth(n_bins=3, n_singles=15)
    u.assign_split_folds_unseen_pert(A, n_folds=5, seed=42)        # single-bin variant
    u.assign_split_folds_unseen_both(A, bin_col="cell_type", n_folds=5, seed=0)
    u.assert_split_invariants(A, scenarios={"UnseenPert"})         # still valid
    diverged = any(
        not u.check_unseenpert_subset_of_unseenboth(
            _test_ko_names(A, "UnseenPert", f), _test_ko_names(A, "UnseenBoth", f))
        for f in range(5))
    assert diverged


# --------------------------------------------------------------------------- #
# assert_split_invariants — the generation-time gate
# --------------------------------------------------------------------------- #
def test_assert_split_invariants_passes_all_multibin_scenarios():
    A = _synth(n_bins=3, n_singles=15)
    u.assign_split_folds_unseen_pert_multibin(A, n_folds=5, seed=0)
    u.assign_split_folds_unseen_cell(A, bin_col="cell_type", n_folds=3, seed=0)
    u.assign_split_folds_unseen_both(A, bin_col="cell_type", n_folds=5, seed=0)
    u.assign_split_folds_unseen_pair(A, bin_col="cell_type", n_folds=5, seed=0)
    u.assert_split_invariants(A)  # must not raise


def test_assert_split_invariants_catches_per_bin_unseen_pert():
    """A per-bin UnseenPert mistake (a perturbation test in some cell types but
    not others) must be rejected — the xatlas-class bug at the function level."""
    A = _synth(n_bins=3, n_singles=12)
    u.assign_split_folds_unseen_pert_multibin(A, n_folds=5, seed=0)
    col = "split_UnseenPert_fold_0"
    # force g3 to be test in ct0 only
    m = (A.obs["cell_type"] == "ct0") & (A.obs["condition"] == "g3")
    A.obs.loc[m, col] = "test"
    with pytest.raises(ValueError, match="perturbation-consistent"):
        u.assert_split_invariants(A, scenarios={"UnseenPert"})


def test_assert_split_invariants_catches_leakage():
    A = _synth(n_bins=1, n_singles=10)
    u.assign_split_folds_unseen_pert(A, n_folds=5, seed=42)
    col = "split_UnseenPert_fold_0"
    # mark a NON-control test perturbation's cells ALSO as train -> leakage.
    # (controls are deliberately round-robined into both train and test folds, so
    # they are excluded from the leakage check — pick a real perturbation.)
    test_pert = next(c for c in A.obs["condition"].unique()
                     if not u._is_control_label(c)
                     and (A.obs.loc[A.obs["condition"] == c, col] == "test").any())
    idx = A.obs.index[A.obs["condition"] == test_pert]
    A.obs.loc[idx[: len(idx) // 2], col] = "train"   # same (bin, ko) now train AND test
    with pytest.raises(ValueError, match="leakage"):
        u.assert_split_invariants(A, scenarios={"UnseenPert"})


# --------------------------------------------------------------------------- #
# UnseenPair: every perturbation must be seen in train in ≥1 cell type
# --------------------------------------------------------------------------- #
def test_unseenpair_every_pert_in_train_across_seeds():
    """Contract: after the generator's repair pass, every fold satisfies the
    'pert seen in ≥1 train bin' invariant for every seed (and assert_split_invariants
    does not raise)."""
    for seed in range(10):
        A = _synth(n_bins=3, n_singles=12)
        u.assign_split_folds_unseen_pair(A, bin_col="cell_type", n_folds=5, seed=seed)
        for f in range(5):
            tr, _, nc = _fold_masks(A, "UnseenPair", f)
            assert u.check_unseenpair_pert_in_train_at_least_once(tr[:, nc]), \
                f"seed {seed} fold {f}: a perturbation has no train bin"
        u.assert_split_invariants(A, scenarios={"UnseenPair"})


def test_unseenpair_repair_actually_fires():
    """Prove the repair has teeth: reconstruct the *pre-repair* partition with the
    same helpers the generator uses, find a (seed, fold) where it would strand a
    pert (all bin-copies non-train), then confirm the generated split repaired it."""
    n_bins, n_singles, n_folds = 3, 12, 5

    def pre_repair_strands_a_pert(seed, fold):
        blocks = u._partition_into_blocks(n_bins * n_singles, n_folds, seed)
        train_pos, _, _ = u._fold_assignment(blocks, fold, n_folds)
        train = np.zeros((n_bins, n_singles), bool)
        for idx in train_pos:
            train[idx // n_singles, idx % n_singles] = True
        return not train.any(axis=0).all()

    hit = next(((s, f) for s in range(20) for f in range(n_folds)
                if pre_repair_strands_a_pert(s, f)), None)
    assert hit is not None, "expected some seed/fold to strand a pert pre-repair"
    seed, fold = hit
    A = _synth(n_bins=n_bins, n_singles=n_singles)
    u.assign_split_folds_unseen_pair(A, bin_col="cell_type", n_folds=n_folds, seed=seed)
    tr, _, nc = _fold_masks(A, "UnseenPair", fold)
    assert u.check_unseenpair_pert_in_train_at_least_once(tr[:, nc])  # repaired


def test_assert_split_invariants_catches_unseenpair_stranded_pert():
    """A perturbation with all (cell_type, pert) versions val/test must be rejected."""
    A = _synth(n_bins=3, n_singles=12)
    u.assign_split_folds_unseen_pair(A, bin_col="cell_type", n_folds=5, seed=0)
    col = "split_UnseenPair_fold_0"
    A.obs.loc[A.obs["condition"] == "g1", col] = "test"   # strand g1 (no train bin)
    with pytest.raises(ValueError, match="no train bin"):
        u.assert_split_invariants(A, scenarios={"UnseenPair"})


# --------------------------------------------------------------------------- #
# Combo scenario
# --------------------------------------------------------------------------- #
def test_unseen_combo_invariants():
    A = _synth(n_bins=1, n_singles=6,
               combos=("g0+g1", "g1+g2", "g2+g3", "g0+g3", "g4+g5", "g3+g4"))
    u.assign_split_folds_unseen_combo(A, n_folds=2, seed=42)
    u.assert_split_invariants(A)  # singles->train, combos split; no leakage
