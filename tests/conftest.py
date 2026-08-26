"""Shared fixtures for the predictor-tier tests.

`FakeStore` is a hand-built stand-in for `DatasetStore`: small enough to reason
about exactly, and — the point — buildable in a unit test, so the expensive
tiers can be driven end-to-end without a 7.9 GB h5ad or a GPU.

It implements only what the predictors actually consume. That is deliberate: if
a predictor starts reaching for something new, the fake fails loudly rather than
silently returning a plausible zero.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pytest

from benchmark.data_loader import SplitInfo


class FakeStore:
    """A tiny in-memory `DatasetStore` look-alike."""

    def __init__(self, n_bins: int = 2, n_kos: int = 8, n_genes: int = 10,
                 split: Optional[SplitInfo] = None, dataset: str = "fake",
                 seed: int = 0, unrepresentable: int = 0):
        self.dataset = dataset
        self.n_bins, self.n_kos, self.n_genes = n_bins, n_kos, n_genes
        self.gene_names = [f"g{i}" for i in range(n_genes)]
        self.bin_names = [f"bin_{i:02d}" for i in range(n_bins)]

        # Perturbation labels must be UNIQUE: recycling them (g0, g1, ... g0)
        # puts the same perturbation in train and test, which is real leakage and
        # is correctly refused by the leakage gate. Each ko therefore targets a
        # distinct gene, so a store needs at least as many genes as kos.
        n_real = n_kos - unrepresentable
        if n_real > n_genes:
            raise ValueError(
                f"FakeStore: {n_real} representable kos need at least that many "
                f"genes to stay uniquely labelled (got n_genes={n_genes})")
        names = [f"g{i}" for i in range(n_real)]
        # `unrepresentable` kos name a gene absent from the panel — the real
        # ecoli case (`h-`, `ns`), and what the drop rule exists for.
        names += [f"absent{i}" for i in range(unrepresentable)]
        self.ko_names = names

        rng = np.random.default_rng(seed)
        self.ctrl_bulk = rng.normal(size=(n_bins, n_genes)).astype(np.float32)
        self.all_bulk = rng.normal(size=(n_bins, n_kos, n_genes)).astype(np.float32)
        self._split = split or unseen_pert_split(n_kos)

    # -- axes ---------------------------------------------------------
    @property
    def gene_to_index(self):
        return {g: i for i, g in enumerate(self.gene_names)}

    # -- targets ------------------------------------------------------
    @property
    def all_deltas(self):
        return self.all_bulk - self.ctrl_bulk[:, None, :]

    @property
    def first_half_deltas(self):
        return self.all_deltas

    def split(self, scenario: str, fold: int) -> SplitInfo:
        return self._split


def unseen_pert_split(n_kos: int = 8) -> SplitInfo:
    """Rectangle regime: kos are split, every bin trains."""
    per = max(1, n_kos // 4)
    idx = np.arange(n_kos, dtype=np.int64)
    return SplitInfo(
        train_ko_indices=idx[: n_kos - 2 * per],
        val_ko_indices=idx[n_kos - 2 * per: n_kos - per],
        test_ko_indices=idx[n_kos - per:],
    )


def unseen_pair_split(n_bins: int = 2, n_kos: int = 8) -> SplitInfo:
    """Scattered-pair regime: the held-out unit is not a rectangle, so the
    marginal indices do not describe it and only the masks do."""
    train = np.zeros((n_bins, n_kos), dtype=bool)
    test = np.zeros((n_bins, n_kos), dtype=bool)
    train[0, : n_kos - 2] = True
    train[1, 2:] = True
    test[0, n_kos - 2:] = True
    test[1, :2] = True
    s = SplitInfo(
        train_ko_indices=np.arange(n_kos - 2, dtype=np.int64),
        val_ko_indices=np.array([n_kos - 3], dtype=np.int64),
        test_ko_indices=np.arange(n_kos - 2, n_kos, dtype=np.int64),
    )
    s.train_pair_mask = train
    s.test_pair_mask = test
    return s


@pytest.fixture
def fake_store():
    return FakeStore()


@pytest.fixture
def isolated_checkpoints(tmp_path, monkeypatch):
    """Send every write under `models/` to tmp_path.

    TWO levers, because one does not cover the other:

    * `VCR_TRAINED_MODEL_ROOT` (read at call time by `config.checkpoint_dir`)
      redirects the expensively-trained tier's checkpoints.
    * `config.MODELS_DIR` is a module global that `predictor_dir` reads at call
      time, so rebinding it redirects `predictor_dir` / `weights_path` /
      `predictions_path` — i.e. the learned and analytical tiers.

    Setting only the env var is what let a test delete a real
    `models/<dataset>/Zero/` directory: the trained tier was safely redirected
    and every other tier was still pointed at the repo. Same pattern as the
    context manager in `perturb_dataset_analysis/cleaning_comparison.py`.
    """
    import benchmark.config as bconfig
    monkeypatch.setenv("VCR_TRAINED_MODEL_ROOT", str(tmp_path))
    monkeypatch.setattr(bconfig, "MODELS_DIR", tmp_path)
    return tmp_path
