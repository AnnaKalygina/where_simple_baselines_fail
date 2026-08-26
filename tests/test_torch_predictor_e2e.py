"""`TorchPredictor` driven end-to-end: fit -> is_trained -> predict.

This is the test whose absence let a `NameError` ship: `_infer` referenced a
symbol whose import had been dropped, and every static check passed — the module
imported, the registry loaded, `py_compile` was happy — because nothing ever RAN
the predict path. Anything that only executes inside `_infer` is invisible until
something calls it.

It needs no GPU (the trainer falls back to CPU) and no real dataset: a fabricated
6-gene store and a 1-layer model make a full train+predict cycle take seconds.
Checkpoints go to tmp_path via `VCR_TRAINED_MODEL_ROOT`.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from benchmark.predictors._torch import data as td
from benchmark.predictors._torch import recipe
from benchmark.predictors.torch_predictor import TorchPredictor
from benchmark.predictors.trained import CLAIM_FILE, FINGERPRINT_FILE
from tests.conftest import FakeStore, unseen_pair_split


class TinyTransformer(TorchPredictor):
    """A real TorchPredictor, sized so a full cycle runs in a unit test."""

    name = "test-tiny-transformer"
    scenarios = ["UnseenPert"]
    wandb_project = None          # never call out to a logging server from a test

    def architecture(self, n_genes: int):
        return dict(d_model=16, nhead=2, ff_mult=1, dropout=0.0,
                    layers=1, num_steps=1)


@pytest.fixture
def fast_recipe(monkeypatch):
    """Two epochs, single-process loading — enough to exercise every path."""
    monkeypatch.setitem(recipe.TRAINING, "epochs", 2)
    monkeypatch.setitem(recipe.TRAINING, "warmup_epochs", 0)
    monkeypatch.setitem(recipe.TRAINING, "num_workers", 0)
    monkeypatch.setitem(recipe.TRAINING, "batch_size", 4)


@pytest.fixture
def trained(isolated_checkpoints, fast_recipe):
    """A fitted predictor plus its store — the shared setup for most tests."""
    store = FakeStore()
    p = TinyTransformer()
    p.fit(store, "UnseenPert", 0)
    return p, store


# ---------------------------------------------------------------------------
# The cycle
# ---------------------------------------------------------------------------

def test_untrained_predict_refuses_and_does_not_train(isolated_checkpoints, fast_recipe):
    store = FakeStore()
    p = TinyTransformer()
    assert not p.is_trained(store, "UnseenPert", 0)
    with pytest.raises(RuntimeError, match="no usable trained model"):
        p.predict(store, "UnseenPert", 0)
    # predict must not have quietly produced a checkpoint on its way past
    assert not p.is_trained(store, "UnseenPert", 0)


def test_fit_then_predict(trained):
    p, store = trained
    assert p.is_trained(store, "UnseenPert", 0)

    out = p.predict(store, "UnseenPert", 0)

    split = store.split("UnseenPert", 0)
    n_bins = len(split.test_bin_indices_or_all(store.n_bins))
    n_kos = len(split.test_ko_indices)
    assert out.shape == (n_bins, n_kos, store.n_genes)
    assert out.dtype == np.float32
    assert np.isfinite(out).all(), "every test cell is representable here"


def test_predict_is_repeatable(trained):
    """Two predicts from one checkpoint must agree — inference has no RNG."""
    p, store = trained
    a = p.predict(store, "UnseenPert", 0)
    b = p.predict(store, "UnseenPert", 0)
    np.testing.assert_allclose(a, b, rtol=1e-6)


def test_checkpoint_survives_a_fresh_instance(trained):
    """State lives on disk, not in the object that trained it."""
    p, store = trained
    fresh = TinyTransformer()
    assert fresh.is_trained(store, "UnseenPert", 0)
    np.testing.assert_allclose(fresh.predict(store, "UnseenPert", 0),
                               p.predict(store, "UnseenPert", 0), rtol=1e-6)


# ---------------------------------------------------------------------------
# Honesty of the gate
# ---------------------------------------------------------------------------

def test_deleting_an_artefact_makes_it_untrained(trained):
    p, store = trained
    run_dir = p._run_dir(store, "UnseenPert", 0)
    (run_dir / p._expected_artifacts()[0]).unlink()

    assert not p.is_trained(store, "UnseenPert", 0)
    assert "missing artefacts" in p.unusable_reason(store, "UnseenPert", 0)
    with pytest.raises(RuntimeError, match="no usable trained model"):
        p.predict(store, "UnseenPert", 0)


@pytest.mark.parametrize("field,label", [
    ("gene_axis_sha", "gene panel"),
    ("split_sha", "train/test split"),
    ("recipe_sha", "training recipe"),
])
def test_fingerprint_detects_staleness(trained, field, label):
    """The checkpoint must notice the world changed under it.

    `split_sha` is the one with teeth: regenerate the folds and a reused
    checkpoint would be scored on cells it trained on, with nothing to show for
    it but a better number.
    """
    p, store = trained
    fp_path = p._run_dir(store, "UnseenPert", 0) / FINGERPRINT_FILE
    fp = json.loads(fp_path.read_text())
    fp[field] = "SOMETHING-ELSE"
    fp_path.write_text(json.dumps(fp))

    assert not p.is_trained(store, "UnseenPert", 0)
    assert label in p.unusable_reason(store, "UnseenPert", 0)


def test_code_sha_is_advisory_not_a_gate(trained):
    """Otherwise every commit would invalidate every checkpoint on the volume."""
    p, store = trained
    fp_path = p._run_dir(store, "UnseenPert", 0) / FINGERPRINT_FILE
    fp = json.loads(fp_path.read_text())
    fp["code_sha"] = "0000000"
    fp_path.write_text(json.dumps(fp))
    assert p.is_trained(store, "UnseenPert", 0)


def test_fit_leaves_no_claim_behind(trained):
    p, store = trained
    assert not (p._run_dir(store, "UnseenPert", 0) / CLAIM_FILE).exists()


def test_a_crashed_retrain_does_not_leave_a_dir_that_reports_itself_trained(
        trained, monkeypatch):
    """The bug this tier had, and the reason the fix belongs in `fit`.

    This tier does NOT wipe its run dir, so a crashed retrain leaves the previous
    fingerprint standing over artefacts the crash touched — `is_trained` said
    yes, `fit` skipped it next time, and `predict` scored it. Worse than stale:
    the trainer checkpoints DURING training while `norm_stats.npz` is written
    only after it returns, so the surviving weights and the surviving sigma come
    from different runs. Nothing crashes; the number is just wrong.
    """
    p, store = trained
    run_dir = p._run_dir(store, "UnseenPert", 0)
    sigma_before = np.load(str(run_dir / "norm_stats.npz"))["sigma"].copy()

    def boom(*a, **k):
        # Leave a plausible half-written checkpoint behind, as a real OOM would.
        torch.save({"state_dict": {}, "sabotaged": True},
                   str(run_dir / "checkpoint_best.pt"))
        raise RuntimeError("simulated OOM mid-training")
    monkeypatch.setattr(p, "_train", boom)

    with pytest.raises(RuntimeError, match="simulated OOM"):
        p.fit(store, "UnseenPert", 0, force=True)

    # The artefacts survive (this tier does not wipe) and the stats are still the
    # PREVIOUS run's — exactly the mismatch. What must not survive is the claim
    # that they belong together.
    assert (run_dir / "checkpoint_best.pt").exists()
    assert np.array_equal(np.load(str(run_dir / "norm_stats.npz"))["sigma"], sigma_before)
    assert not (run_dir / FINGERPRINT_FILE).exists()
    assert not p.is_trained(store, "UnseenPert", 0)
    assert FINGERPRINT_FILE in p.unusable_reason(store, "UnseenPert", 0)

    with pytest.raises(RuntimeError, match="no usable trained model"):
        p.predict(store, "UnseenPert", 0)


def test_the_torch_tier_sets_its_own_claim_ttl(trained):
    """Nothing kills an in-process run, so there is no `train_timeout` to borrow
    for the abandoned-claim horizon. Inheriting the base 24 h would be a number
    nobody chose."""
    p, _ = trained
    assert p._claim_ttl() == p.claim_ttl_seconds
    assert p._claim_ttl() != 24 * 3600


# ---------------------------------------------------------------------------
# Coverage honesty
# ---------------------------------------------------------------------------

def test_unrepresentable_perturbations_are_declined_not_guessed(
        isolated_checkpoints, fast_recipe):
    """A ko whose target is absent from the panel has an all-zero marker, i.e.
    it is indistinguishable from control. Emitting a number there would be
    scored as if it meant something."""
    # 12 kos -> the split holds out 9,10,11; one of those is unrepresentable, so
    # the fold contains BOTH declined and predicted cells. An all-declined fold
    # is a different situation and is rejected by the no-finite-prediction guard.
    store = FakeStore(n_kos=12, n_genes=12, unrepresentable=1)
    p = TinyTransformer()
    p.fit(store, "UnseenPert", 0)
    out = p.predict(store, "UnseenPert", 0)

    split = store.split("UnseenPert", 0)
    blind = set(td.unrepresentable_kos(store).tolist())
    assert blind, "fixture should contain an unrepresentable ko"

    declined = predicted = 0
    for j, k in enumerate(split.test_ko_indices):
        if int(k) in blind:
            assert np.isnan(out[:, j, :]).all(), f"ko {k} should be declined"
            declined += 1
        else:
            assert np.isfinite(out[:, j, :]).all(), f"ko {k} should be predicted"
            predicted += 1
    assert declined and predicted, "the fold must exercise both branches"
    assert p.has_drop_rule, "declining cells is a drop rule and must be declared"


# ---------------------------------------------------------------------------
# Regimes
# ---------------------------------------------------------------------------

def test_scattered_pair_regime(isolated_checkpoints, fast_recipe):
    """UnseenPair holds out a scattered pair set that no marginal describes."""
    store = FakeStore(n_bins=2, n_kos=8, split=unseen_pair_split(2, 8))

    class PairTiny(TinyTransformer):
        name = "test-tiny-pair"
        scenarios = ["UnseenPair"]

    p = PairTiny()
    p.fit(store, "UnseenPair", 0)
    out = p.predict(store, "UnseenPair", 0)
    split = store.split("UnseenPair", 0)
    assert out.shape == (len(split.test_bin_indices_or_all(store.n_bins)),
                         len(split.test_ko_indices), store.n_genes)


def test_undeclared_scenario_is_refused(isolated_checkpoints, fast_recipe):
    p = TinyTransformer()
    with pytest.raises(ValueError, match="does not declare scenario"):
        p.fit(FakeStore(), "UnseenCombo", 0)
