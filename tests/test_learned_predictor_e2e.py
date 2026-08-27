"""The learned tier's persistence contract.

The cheap tier persists a small `weights.npz` instead of a checkpoint dir, but
it answers the same question about it as the expensive tier: are these
coefficients still valid for the store, scenario and fold being asked about?
For a long time it did not — `is_trained` checked only that the npz opened and
held at least one array — so a rebuilt gene panel or a regenerated fold left
`Ridge` reporting itself trained and predicting from coefficients fitted on
cells that had since become test. That is the failure these tests pin: silent,
score-inflating, and invisible to every other check.

Mirrors `test_torch_predictor_e2e.py`, deliberately: the guarantee is supposed
to be the same one, so it is worth being able to read the two side by side.
"""

from __future__ import annotations

import json

import pytest

from benchmark.predictors._shared import sha_split, sha_strings
from benchmark.predictors.base import PREDICTOR_REGISTRY, _ensure_predictors_loaded
from benchmark.predictors.learned import LearnedPredictor

from conftest import FakeStore, unseen_pair_split


def _learned_names():
    """Every learned predictor that can handle the scenario these tests use."""
    _ensure_predictors_loaded()
    return sorted(
        name for name, cls in PREDICTOR_REGISTRY.items()
        if issubclass(cls, LearnedPredictor) and "UnseenPert" in cls.scenarios)


@pytest.fixture
def fitted(isolated_checkpoints):
    """A fitted-and-persisted Ridge plus its store — the shared setup."""
    store = FakeStore()
    p = PREDICTOR_REGISTRY["Ridge"]()
    p.fit(store, "UnseenPert", 0)
    p.persist(store, "UnseenPert", 0)
    return p, store


def _fp_path(p, store):
    return p._fingerprint_file(store.dataset, "UnseenPert", 0)


# ---------------------------------------------------------------------------
# The cycle
# ---------------------------------------------------------------------------

def test_fit_alone_writes_nothing(isolated_checkpoints):
    """`fit` is PURE by design and the tier docstring says so: `verify` fits
    predictors on synthetic stores and `Mean+TargetScaling` fits sub-predictors
    internally, so a persisting `fit` would scribble into the real models tree."""
    store = FakeStore()
    p = PREDICTOR_REGISTRY["Ridge"]()
    p.fit(store, "UnseenPert", 0)
    assert not p._weights_file(store.dataset, "UnseenPert", 0).exists()
    assert not p.is_trained(store, "UnseenPert", 0)


def test_persist_then_restore_round_trips(fitted):
    p, store = fitted
    assert p.is_trained(store, "UnseenPert", 0)
    restored = type(p).restore(store, "UnseenPert", 0)
    expected = p.predict(store, "UnseenPert", 0)
    assert restored.predict(store, "UnseenPert", 0).shape == expected.shape


@pytest.mark.parametrize("name", _learned_names())
def test_persist_creates_its_own_directory(isolated_checkpoints, name):
    """The tier makes the directory, not the fifteen `save_weights` bodies.

    Six of the seven remembered to; `Mean+TargetScaling` did not, and worked
    only because it delegates to `TargetScaling`, which does. Relying on that is
    relying on an implementation detail of a different class.
    """
    # Bigger than the default store: `BilinearRidge` runs a 10-component PCA
    # over the TRAIN kos, and `unseen_pert_split` gives `n_kos - 2*(n_kos//4)`
    # of them — so anything under 20 kos fails inside sklearn before it ever
    # reaches the persistence this test is about.
    store = FakeStore(n_bins=2, n_kos=24, n_genes=24)
    p = PREDICTOR_REGISTRY[name]()
    p.fit(store, "UnseenPert", 0)
    wpath = p._weights_file(store.dataset, "UnseenPert", 0)
    assert not wpath.parent.exists(), "fixture should start from an empty tree"
    p.persist(store, "UnseenPert", 0)
    assert wpath.exists() and _fp_path(p, store).exists()


# ---------------------------------------------------------------------------
# Staleness — the point of the fingerprint
# ---------------------------------------------------------------------------

def test_persist_writes_a_fingerprint_beside_the_weights(fitted):
    p, store = fitted
    fp = json.loads(_fp_path(p, store).read_text())
    assert fp["gene_axis_sha"] == sha_strings(store.gene_names)
    assert fp["split_sha"] == sha_split(store, "UnseenPert", 0)


def test_a_regenerated_split_makes_the_weights_unusable(fitted):
    """The one with teeth, and the reason cheap-to-refit is not a reason to
    check less: the fit used the TRAIN cells, so re-cutting the folds turns
    valid coefficients into coefficients fitted on cells that are now test.
    Nothing crashes; the score just quietly improves."""
    p, store = fitted
    store._split = unseen_pair_split(n_bins=store.n_bins, n_kos=store.n_kos)
    assert sha_split(store, "UnseenPert", 0) != json.loads(
        _fp_path(p, store).read_text())["split_sha"], "the split must really differ"
    assert not p.is_trained(store, "UnseenPert", 0)


@pytest.mark.parametrize("field,label", [
    ("gene_axis_sha", "the gene panel"),
    ("split_sha", "the train/test split"),
])
def test_learned_tier_detects_every_enforced_field(fitted, field, label, caplog):
    p, store = fitted
    path = _fp_path(p, store)
    fp = json.loads(path.read_text())
    fp[field] = "SOMETHING-ELSE"
    path.write_text(json.dumps(fp))

    with caplog.at_level("INFO"):
        assert not p.is_trained(store, "UnseenPert", 0)
    assert label in caplog.text, "the cause must reach the user, not just a False"


def test_code_sha_is_advisory_here_too(fitted):
    """Otherwise every commit would invalidate every weights file on the volume
    — the same reason the expensive tier leaves it out of `_ENFORCED`."""
    p, store = fitted
    path = _fp_path(p, store)
    fp = json.loads(path.read_text())
    fp["code_sha"] = "0000000"
    path.write_text(json.dumps(fp))
    assert p.is_trained(store, "UnseenPert", 0)


def test_weights_without_a_fingerprint_read_as_untrained(fitted):
    """The state of every `weights.npz` written before this check existed.

    Grandfathering them would permanently exempt exactly the condition the
    fingerprint exists to catch, so they refit instead — which for this tier
    costs seconds.
    """
    p, store = fitted
    _fp_path(p, store).unlink()
    assert p._weights_file(store.dataset, "UnseenPert", 0).exists()
    assert not p.is_trained(store, "UnseenPert", 0)


def test_an_empty_npz_reads_as_untrained(fitted):
    """A killed `fit` used to leave one, and the old base class wrote one for
    every predictor on purpose. Nothing writes one now — `persist` is the only
    writer and the stateless tiers have no `persist` — but the guard is what
    makes that structural rather than assumed."""
    import numpy as np
    p, store = fitted
    np.savez(str(p._weights_file(store.dataset, "UnseenPert", 0)))
    assert not p.is_trained(store, "UnseenPert", 0)
