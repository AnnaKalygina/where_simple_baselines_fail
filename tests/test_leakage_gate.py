"""Direct tests of the host-side leakage gate.

This is the highest-stakes check in the codebase: a leak that reaches training
produces no crash and no warning — every metric simply improves, and the result
looks like a better model. Until now the gate was only exercised *incidentally*
(it caught a duplicate-label bug in a test fixture, which is how we knew it
worked at all).

Each test builds a small h5ad that is leaky in exactly one way and asserts the
gate says so, plus a clean one it must accept. Precision matters as much as
recall here: a gate that flags everything gets bypassed.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("anndata")
import anndata as ad
import pandas as pd

from benchmark.predictors._container.leakage import assert_leak_safe_columns


def _write(tmp_path, rows, name="d.h5ad"):
    """rows: (condition, cell_type, split) triples -> a minimal h5ad on disk."""
    obs = pd.DataFrame(
        {"condition": [r[0] for r in rows],
         "cell_type": [r[1] for r in rows],
         "split_UnseenPert_fold_0": [r[2] for r in rows]},
        index=[f"c{i}" for i in range(len(rows))])
    adata = ad.AnnData(X=np.zeros((len(rows), 3), dtype=np.float32), obs=obs,
                       var=pd.DataFrame(index=["g0", "g1", "g2"]))
    path = tmp_path / name
    adata.write_h5ad(path)
    return path


def _check(path, regime="UnseenPert"):
    return assert_leak_safe_columns(
        path, "split_UnseenPert_fold_0", regime=regime, covariate_col="cell_type")


# ---------------------------------------------------------------------------
# The clean case must PASS — a gate that rejects everything gets switched off
# ---------------------------------------------------------------------------

def test_a_clean_split_is_accepted(tmp_path):
    path = _write(tmp_path, [
        ("geneA", "K562", "train"), ("geneB", "K562", "train"),
        ("geneC", "K562", "val"),
        ("geneD", "K562", "test"), ("geneE", "K562", "test"),
        ("ctrl", "K562", "train"), ("ctrl", "K562", "test"),
    ])
    report = _check(path)
    assert report["ok"], report["problems"]


def test_controls_may_appear_on_both_sides(tmp_path):
    """Control is the reference every delta is measured against; it is present
    everywhere by design and must not be mistaken for a leak."""
    path = _write(tmp_path, [
        ("geneA", "K562", "train"),
        ("geneD", "K562", "test"),
        ("ctrl", "K562", "train"), ("ctrl", "K562", "val"), ("ctrl", "K562", "test"),
    ])
    assert _check(path)["ok"]


# ---------------------------------------------------------------------------
# Each way a split can leak
# ---------------------------------------------------------------------------

def test_a_perturbation_in_both_train_and_test_is_caught(tmp_path):
    """The classic UnseenPert leak: the model has seen the held-out perturbation."""
    path = _write(tmp_path, [
        ("geneA", "K562", "train"),
        ("geneD", "K562", "train"),
        ("geneD", "K562", "test"),        # <- same perturbation on both sides
        ("ctrl", "K562", "train"),
    ])
    report = _check(path)
    assert not report["ok"]
    assert any("held-out" in p for p in report["problems"]), report["problems"]


def test_a_perturbation_in_both_val_and_test_is_caught(tmp_path):
    """Val is training signal too — early stopping selects on it."""
    path = _write(tmp_path, [
        ("geneA", "K562", "train"),
        ("geneD", "K562", "val"),
        ("geneD", "K562", "test"),
        ("ctrl", "K562", "train"),
    ])
    report = _check(path)
    assert not report["ok"]
    assert any("held-out" in p for p in report["problems"]), report["problems"]


def test_a_missing_split_column_is_reported_not_ignored(tmp_path):
    path = _write(tmp_path, [("geneA", "K562", "train")])
    report = assert_leak_safe_columns(
        path, "split_DoesNotExist_fold_9", regime="UnseenPert",
        covariate_col="cell_type")
    assert not report["ok"]
    assert any("not in obs" in p for p in report["problems"])


def test_an_empty_test_set_is_reported(tmp_path):
    """Nothing held out means nothing was measured — silently 'leak-free'."""
    path = _write(tmp_path, [
        ("geneA", "K562", "train"), ("geneB", "K562", "train"),
        ("ctrl", "K562", "train"),
    ])
    report = _check(path)
    assert not report["ok"], "a split with no test cells must not pass as clean"


# ---------------------------------------------------------------------------
# The cell-axis regimes, where the held-out unit is a (cell_type, pert) PAIR
# ---------------------------------------------------------------------------

def test_pair_regime_allows_a_perturbation_seen_in_another_cell_type(tmp_path):
    """Under UnseenBoth/UnseenPair the unit is the PAIR: the same perturbation
    in a different cell type is legitimately training data, and a gate that
    flagged it would make those regimes untrainable."""
    path = _write(tmp_path, [
        ("geneD", "K562", "train"),      # same pert...
        ("geneD", "RPE1", "test"),       # ...different cell type -> not a leak
        ("geneA", "K562", "train"),
        ("ctrl", "K562", "train"), ("ctrl", "RPE1", "train"),
    ])
    report = _check(path, regime="UnseenBoth")
    assert report["ok"], report["problems"]


def test_pair_regime_still_catches_the_same_pair_on_both_sides(tmp_path):
    path = _write(tmp_path, [
        ("geneD", "RPE1", "train"),
        ("geneD", "RPE1", "test"),       # <- the SAME (cell_type, pert) pair
        ("geneA", "K562", "train"),
        ("ctrl", "K562", "train"), ("ctrl", "RPE1", "train"),
    ])
    report = _check(path, regime="UnseenBoth")
    assert not report["ok"], "the same pair on both sides is a leak"
