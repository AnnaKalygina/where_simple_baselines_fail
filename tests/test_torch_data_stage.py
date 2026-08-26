"""Leak-safety and correctness of the torch data stage (`_torch/data.py`).

The property under test is that nothing held out can reach training: not through
the sampled pairs, and not through the normalisation constants. The second route
is the quiet one — standardising with statistics computed over all cells would
leak the held-out scale into every batch without changing any downstream metric.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from benchmark.data_loader import SplitInfo
from benchmark.predictors._torch import data as td


# ---------------------------------------------------------------------------
# Fakes: a store small enough to reason about exactly
# ---------------------------------------------------------------------------

class FakeStore:
    """Minimal stand-in exposing only what the data stage consumes."""

    def __init__(self, n_bins=3, n_kos=6, n_genes=5, split=None):
        self.dataset = "fake"
        self.n_bins, self.n_kos, self.n_genes = n_bins, n_kos, n_genes
        self.gene_names = [f"g{i}" for i in range(n_genes)]
        # ko 0..n-2 are singles on g0..; the last is a combo, so combo parsing is
        # exercised by the default fixture rather than only in its own test.
        self.ko_names = [f"g{i % n_genes}" for i in range(n_kos - 1)] + ["g0+g1"]
        self.bin_names = [f"bin_{i}" for i in range(n_bins)]
        rng = np.random.default_rng(0)
        self.ctrl_bulk = rng.normal(size=(n_bins, n_genes)).astype(np.float32)
        self.all_bulk = rng.normal(size=(n_bins, n_kos, n_genes)).astype(np.float32)
        self._split = split

    @property
    def gene_to_index(self):
        return {g: i for i, g in enumerate(self.gene_names)}

    @property
    def all_deltas(self):
        return self.all_bulk - self.ctrl_bulk[:, None, :]

    def split(self, scenario, fold):
        return self._split


def _unseen_pert_split(n_kos=6):
    """Rectangle regime: kos split, all bins train."""
    return SplitInfo(
        train_ko_indices=np.array([0, 1, 2], dtype=np.int64),
        val_ko_indices=np.array([3], dtype=np.int64),
        test_ko_indices=np.array([4, 5], dtype=np.int64),
    )


def _unseen_pair_split(n_bins=3, n_kos=6):
    """Scattered-pair regime: the held-out unit is not a rectangle."""
    train = np.zeros((n_bins, n_kos), dtype=bool)
    test = np.zeros((n_bins, n_kos), dtype=bool)
    train[0, :4] = True
    train[1, 2:] = True
    test[0, 4:] = True
    test[2, :2] = True
    s = SplitInfo(
        train_ko_indices=np.array([0, 1, 2, 3], dtype=np.int64),
        val_ko_indices=np.array([3], dtype=np.int64),
        test_ko_indices=np.array([4, 5], dtype=np.int64),
    )
    s.train_pair_mask = train
    s.test_pair_mask = test
    return s


# ---------------------------------------------------------------------------
# Label parsing
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("label,genes,dose", [
    ("aaea", ["aaea"], 1.0),                 # unqualified == full dose
    ("aaea@0.25", ["aaea"], 0.25),
    ("aaea@0.50", ["aaea"], 0.5),
    ("aaer+cra", ["aaer", "cra"], 1.0),
    ("h-+ns@0.50", ["h-", "ns"], 0.5),       # dose applies to the whole combo
    ("a;b", ["a", "b"], 1.0),                # complexes split like combos
])
def test_parse_ko_dose(label, genes, dose):
    assert td.parse_ko_dose(label) == (genes, dose)


def test_perturbation_matrix_marks_targets_with_dose():
    store = FakeStore()
    store.ko_names = ["g0", "g1@0.25", "g0+g1", "nosuchgene"]
    store.n_kos = 4
    p, unrepresentable = td.build_perturbation_matrix(store)

    assert p[0, 0] == 1.0 and p[0, 1:].sum() == 0
    assert p[1, 1] == 0.25
    assert p[2, 0] == 1.0 and p[2, 1] == 1.0
    # A ko whose target is not in the panel is reported, not silently dropped.
    assert p[3].sum() == 0
    assert unrepresentable == 1


# ---------------------------------------------------------------------------
# Leak-safety
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("make_split", [_unseen_pert_split, _unseen_pair_split],
                         ids=["rectangle", "scattered_pairs"])
def test_train_pairs_never_include_a_test_pair(make_split):
    split = make_split()
    store = FakeStore(split=split)
    fd = td.build_fold_data(store, "UnseenPert", 0)

    test = split.test_mask_2d(store.n_bins, store.n_kos)
    for b, k in fd.train.pairs:
        assert not test[b, k], f"train pair ({b},{k}) is a TEST cell"


@pytest.mark.parametrize("make_split", [_unseen_pert_split, _unseen_pair_split],
                         ids=["rectangle", "scattered_pairs"])
def test_val_pairs_never_include_a_test_pair(make_split):
    split = make_split()
    store = FakeStore(split=split)
    fd = td.build_fold_data(store, "UnseenPert", 0)
    if fd.val is None:
        pytest.skip("no validation pairs for this split")

    test = split.test_mask_2d(store.n_bins, store.n_kos)
    for b, k in fd.val.pairs:
        assert not test[b, k], f"val pair ({b},{k}) is a TEST cell"


def test_sampler_only_ever_yields_train_pairs():
    """The end-to-end guarantee: what the loader hands the model is train-only."""
    split = _unseen_pert_split()
    store = FakeStore(split=split)
    fd = td.build_fold_data(store, "UnseenPert", 0)
    loader, _ = td.make_loaders(fd, batch_size=4, num_workers=0, seed=42)

    test = split.test_mask_2d(store.n_bins, store.n_kos)
    seen = 0
    for _ in range(5):                       # several passes: sampling is with replacement
        for batch in loader:
            for b, k in zip(batch["bin_idx"].tolist(), batch["ko_idx"].tolist()):
                assert not test[b, k], f"sampler yielded TEST cell ({b},{k})"
                seen += 1
    assert seen > 0


def test_norm_stats_ignore_held_out_cells():
    """Statistics must not shift when held-out expression changes."""
    split = _unseen_pert_split()
    store = FakeStore(split=split)
    before = td.compute_norm_stats(store, split)

    # Corrupt the TEST cells only, wildly.
    test = split.test_mask_2d(store.n_bins, store.n_kos)
    store.all_bulk[test] += 1000.0
    after = td.compute_norm_stats(store, split)

    np.testing.assert_allclose(before.mu, after.mu, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(before.sigma, after.sigma, rtol=1e-6, atol=1e-6)


def test_norm_stats_do_change_with_training_cells():
    """Guard against the above passing because the stats ignore everything."""
    split = _unseen_pert_split()
    store = FakeStore(split=split)
    before = td.compute_norm_stats(store, split)

    train = split.train_mask_2d(store.n_bins, store.n_kos)
    store.all_bulk[train] += 1000.0
    after = td.compute_norm_stats(store, split)

    assert not np.allclose(before.mu, after.mu)


# ---------------------------------------------------------------------------
# Shapes / values
# ---------------------------------------------------------------------------

def test_batch_shapes_and_target_scaling():
    split = _unseen_pert_split()
    store = FakeStore(split=split)
    fd = td.build_fold_data(store, "UnseenPert", 0)
    sample = fd.train[0]

    n_genes = store.n_genes
    for key in ("x_wt", "p", "gene_idx", "delta"):
        assert sample[key].shape == (n_genes,), key
    assert sample["gene_idx"].tolist() == list(range(n_genes))

    b, k = fd.train.pairs[0]
    np.testing.assert_allclose(
        sample["delta"].numpy(),
        (store.all_deltas[b, k] / fd.stats.sigma).astype(np.float32), rtol=1e-5)
    np.testing.assert_allclose(
        sample["x_wt"].numpy(),
        ((store.ctrl_bulk[b] - fd.stats.mu) / fd.stats.sigma).astype(np.float32),
        rtol=1e-5)


def test_sigma_floor_survives_a_constant_gene():
    """A gene with no spread must not produce inf/nan after standardising."""
    split = _unseen_pert_split()
    store = FakeStore(split=split)
    store.all_bulk[:, :, 0] = 3.0
    store.ctrl_bulk[:, 0] = 3.0

    fd = td.build_fold_data(store, "UnseenPert", 0)
    assert fd.stats.sigma[0] >= td._SIGMA_FLOOR
    sample = fd.train[0]
    assert np.isfinite(sample["x_wt"].numpy()).all()
    assert np.isfinite(sample["delta"].numpy()).all()


def test_sampling_weights_track_effect_size():
    """A high-effect ko must be weighted above a near-silent one."""
    split = _unseen_pert_split()
    store = FakeStore(split=split)
    store.all_bulk[:, 0, :] = store.ctrl_bulk            # ko 0: no effect
    store.all_bulk[:, 1, :] = store.ctrl_bulk + 50.0     # ko 1: large effect

    fd = td.build_fold_data(store, "UnseenPert", 0)
    by_ko = {}
    for w, (_, k) in zip(fd.train_weights, fd.train.pairs):
        by_ko.setdefault(int(k), []).append(w)
    assert np.mean(by_ko[1]) > np.mean(by_ko[0])


@pytest.mark.parametrize("make_split", [_unseen_pert_split, _unseen_pair_split],
                         ids=["rectangle", "scattered_pairs"])
def test_val_pairs_never_overlap_train(make_split):
    """Validation must be held out from TRAINING too, not just from test.

    In rectangle regimes val kos are disjoint from train kos, so this holds for
    free. Under UnseenPair the train mask is a scattered pair set that can
    contain (bin, val_ko) cells, and validating on cells the model trained on
    silently disables early stopping and best-checkpoint selection.
    """
    split = make_split()
    store = FakeStore(split=split)
    fd = td.build_fold_data(store, "UnseenPert", 0)
    if fd.val is None:
        pytest.skip("no validation pairs for this split")

    train = split.train_mask_2d(store.n_bins, store.n_kos)
    for b, k in fd.val.pairs:
        assert not train[b, k], f"val pair ({b},{k}) is also a TRAIN cell"
