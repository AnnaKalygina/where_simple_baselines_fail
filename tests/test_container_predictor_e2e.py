"""`ContainerPredictor` driven end-to-end with the container itself stubbed.

`runner.run_container` is the single seam between the host and the `.sif`, so
replacing it exercises *everything host-side* — recipe validation, config
generation, the artefact contract, the claim file, the fingerprint, and the
mapping of a container's absolute-expression output onto the benchmark's delta
axes — without a GPU, an image, or a real dataset.

The one thing it cannot prove is that a real container honours the contract;
that stays a GPU job. What it does prove is that if a container honours the
contract, the host handles it correctly — which is the half that changes often.

It also covers the covariate branch of `tensor_map`, which had only ever run on
adamson16 (single cell line, so the branch never executed).
"""

from __future__ import annotations

import json
import pickle

import numpy as np
import pytest

pytest.importorskip("anndata")
import anndata as ad
import pandas as pd

from benchmark.predictors import container_predictor as cp
from benchmark.predictors._container import runner
from benchmark.predictors.container_predictor import (
    ContainerPredictor, _validate_recipe)
from benchmark.predictors.trained import CLAIM_FILE, FINGERPRINT_FILE
from tests.conftest import FakeStore


GOOD_RECIPE = {
    "model_key": "fake",
    "sif_path": "docker/fake/fake.sif",
    "entry": ["python", "/app_code/fake/run.py"],
    "code_binds": {"docker/fake": "/app_code/fake"},
    "hyperparameters": {"epochs": 3, "lr": 0.01},
    "expected_artifacts": ["model.pt", "state.pkl"],
    "train_timeout": 600,
    "predict_timeout": 120,
    "wandb_project": None,
}


class FakeContainer(ContainerPredictor):
    """A container predictor whose recipe is in-memory and whose leakage gate
    works off the split masks (a fabricated store has no h5ad on disk)."""

    name = "test-fake-container"
    model_dir = "docker/fake"
    scenarios = ["UnseenPert"]
    has_drop_rule = True

    @classmethod
    def _recipe(cls):
        return dict(GOOD_RECIPE)

    def _sif(self):
        return __import__("pathlib").Path("/nonexistent/fake.sif")

    def _data_dir(self, store):
        return __import__("pathlib").Path("/nonexistent/data")

    def _preflight(self, store, scenario, fold) -> None:
        pass


def _stub_container(monkeypatch, store, *, drop_kos=(), gene_subset=None,
                    write_artifacts=True, covariate=True, rows_per_ko=1):
    """Replace the `.sif` call with a container that honours the contract.

    Train mode leaves the declared artefacts; predict mode writes an
    `predictions.h5ad` of ABSOLUTE expression (control + a known delta), which is
    what the contract says a container emits and what `tensor_map` reduces.
    """
    truth = {}

    def fake_run(sif, mode, cfg, *, data_dir, output_dir, code_binds, entry,
                 timeout=None):
        output_dir = __import__("pathlib").Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / f"{mode}_config.json").write_text(json.dumps(cfg, indent=2))
        if mode == "train":
            if write_artifacts:
                (output_dir / "model.pt").write_bytes(b"weights")
                with open(output_dir / "state.pkl", "wb") as fh:
                    pickle.dump({"trained": True}, fh)
            return
        # predict
        split = store.split("UnseenPert", 0)
        tki = [int(k) for k in split.test_ko_indices if int(k) not in set(drop_kos)]
        genes = list(gene_subset if gene_subset is not None else store.gene_names)
        gidx = [store.gene_names.index(g) for g in genes]
        # Two independent streams: the per-ko delta must be IDENTICAL across
        # runs, so it cannot share an RNG with a jitter draw whose size varies
        # with rows_per_ko.
        rng = np.random.default_rng(1)
        rng_jitter = np.random.default_rng(2)
        rows, conds, covs = [], [], []
        for b in range(store.n_bins):
            for k in tki:
                d = rng.standard_normal(len(genes)).astype(np.float32) * 0.1
                truth[(b, k)] = d
                # `rows_per_ko` > 1 emits single cells that AVERAGE to the same
                # profile, so the mapping's reduction can be checked.
                jitter = rng_jitter.standard_normal((rows_per_ko, len(genes))).astype(np.float32)
                jitter -= jitter.mean(axis=0, keepdims=True)
                for r in range(rows_per_ko):
                    rows.append((store.ctrl_bulk[b][gidx] + d + jitter[r]).astype(np.float32))
                    conds.append(store.ko_names[k])
                    covs.append(store.bin_names[b])
        obs = {"condition": conds}
        if covariate:
            obs["covariate"] = covs
        adata = ad.AnnData(
            X=np.vstack(rows).astype(np.float32),
            obs=pd.DataFrame(obs, index=[f"c{i}" for i in range(len(conds))]),
            var=pd.DataFrame(index=genes))
        adata.write_h5ad(output_dir / "predictions.h5ad")

    monkeypatch.setattr(runner, "run_container", fake_run)
    monkeypatch.setattr(cp.runner, "run_container", fake_run)
    return truth



@pytest.fixture
def store_on_disk(tmp_path, monkeypatch):
    """A FakeStore plus the minimal real h5ad the host code reads.

    `build_config` derives the container's train/val/test condition lists from
    `obs`, and the leakage gate reads the same file — so stubbing them would skip
    the two host-side steps most worth testing. Writing a genuine 20-cell h5ad
    into a redirected DATA_DIR costs milliseconds and exercises both.
    """
    from benchmark import config as bconfig

    def _make(n_bins=1, n_kos=8, n_genes=10, **kw):
        store = FakeStore(n_bins=n_bins, n_kos=n_kos, n_genes=n_genes, **kw)
        split = store.split("UnseenPert", 0)
        role = {}
        for k in split.train_ko_indices: role[int(k)] = "train"
        for k in split.val_ko_indices:   role[int(k)] = "val"
        for k in split.test_ko_indices:  role[int(k)] = "test"

        conds, cells, splits, rows = [], [], [], []
        rng = np.random.default_rng(0)
        for b in range(n_bins):
            for k in range(n_kos):
                conds.append(store.ko_names[k])
                cells.append(store.bin_names[b])
                splits.append(role.get(k, "train"))
                rows.append(rng.normal(size=n_genes))
            # controls are never held out; they are the reference for every delta
            conds.append("control"); cells.append(store.bin_names[b])
            splits.append("train"); rows.append(rng.normal(size=n_genes))

        obs = pd.DataFrame(
            {"condition": conds, "cell_type": cells,
             "split_UnseenPert_fold_0": splits},
            index=[f"cell{i}" for i in range(len(conds))])
        adata = ad.AnnData(X=np.asarray(rows, dtype=np.float32), obs=obs,
                           var=pd.DataFrame(index=store.gene_names))

        monkeypatch.setattr(bconfig, "DATA_DIR", tmp_path)
        d = tmp_path / store.dataset
        d.mkdir(parents=True, exist_ok=True)
        adata.write_h5ad(d / f"{store.dataset}_processed.h5ad")
        return store

    return _make


# ---------------------------------------------------------------------------
# Recipe schema (§4.2, §4.5)
# ---------------------------------------------------------------------------

def test_valid_recipe_passes():
    _validate_recipe(dict(GOOD_RECIPE), "test")


@pytest.mark.parametrize("missing", sorted(
    set(GOOD_RECIPE) - {"wandb_project"}))
def test_missing_required_key_is_rejected(missing):
    bad = dict(GOOD_RECIPE); bad.pop(missing)
    with pytest.raises(ValueError, match="missing required key"):
        _validate_recipe(bad, "test")


def test_unknown_key_is_rejected():
    """A typo must not be silently ignored, or 'the recipe is the single source
    of truth' stops being true."""
    bad = dict(GOOD_RECIPE); bad["epocs"] = 40
    with pytest.raises(ValueError, match="unrecognised key"):
        _validate_recipe(bad, "test")


def test_empty_artifact_list_is_refused(monkeypatch):
    """Otherwise an EMPTY run dir reports itself trained."""
    p = FakeContainer()
    monkeypatch.setattr(FakeContainer, "_recipe",
                        classmethod(lambda cls: {**GOOD_RECIPE,
                                                 "expected_artifacts": []}))
    with pytest.raises(ValueError, match="empty `expected_artifacts`"):
        p._expected_artifacts()


# ---------------------------------------------------------------------------
# The cycle
# ---------------------------------------------------------------------------

def test_fit_then_predict(isolated_checkpoints, monkeypatch, store_on_disk):
    store = store_on_disk(n_bins=1, n_kos=8)
    truth = _stub_container(monkeypatch, store)
    p = FakeContainer()

    assert not p.is_trained(store, "UnseenPert", 0)
    p.fit(store, "UnseenPert", 0)
    assert p.is_trained(store, "UnseenPert", 0)

    out = p.predict(store, "UnseenPert", 0)
    split = store.split("UnseenPert", 0)
    assert out.shape == (1, len(split.test_ko_indices), store.n_genes)

    # the delta the container implied must be what the host recovered
    for j, k in enumerate(split.test_ko_indices):
        np.testing.assert_allclose(out[0, j], truth[(0, int(k))], atol=1e-5)


def test_train_and_predict_configs_both_survive(isolated_checkpoints, monkeypatch, store_on_disk):
    """Predict used to overwrite the record of how training was configured."""
    store = store_on_disk(n_bins=1)
    _stub_container(monkeypatch, store)
    p = FakeContainer()
    p.fit(store, "UnseenPert", 0)
    p.predict(store, "UnseenPert", 0)

    run_dir = p._run_dir(store, "UnseenPert", 0)
    assert (run_dir / "train_config.json").exists()
    assert (run_dir / "predict_config.json").exists()
    train_cfg = json.loads((run_dir / "train_config.json").read_text())
    assert train_cfg["mode"] == "train"
    assert train_cfg["hyperparameters"]["epochs"] == 3


def test_missing_artifacts_fail_loudly_in_fit(isolated_checkpoints, monkeypatch, store_on_disk):
    """A crashed container must fail HERE, not later at predict."""
    store = store_on_disk(n_bins=1)
    _stub_container(monkeypatch, store, write_artifacts=False)
    p = FakeContainer()
    with pytest.raises(RuntimeError, match="unusable run dir"):
        p.fit(store, "UnseenPert", 0)
    assert not p.is_trained(store, "UnseenPert", 0)


def test_predict_refuses_without_a_checkpoint(isolated_checkpoints, monkeypatch, store_on_disk):
    store = store_on_disk(n_bins=1)
    _stub_container(monkeypatch, store)
    p = FakeContainer()
    with pytest.raises(RuntimeError, match="no usable trained model"):
        p.predict(store, "UnseenPert", 0)


# ---------------------------------------------------------------------------
# Staleness + concurrency, on the container tier
# ---------------------------------------------------------------------------

def test_recipe_change_invalidates_the_checkpoint(isolated_checkpoints, monkeypatch, store_on_disk):
    """S2: a hyperparameter edit must not be served from an old checkpoint."""
    store = store_on_disk(n_bins=1)
    _stub_container(monkeypatch, store)
    p = FakeContainer()
    p.fit(store, "UnseenPert", 0)
    assert p.is_trained(store, "UnseenPert", 0)

    monkeypatch.setattr(FakeContainer, "_recipe",
                        classmethod(lambda cls: {**GOOD_RECIPE,
                                                 "hyperparameters": {"epochs": 99, "lr": 0.01}}))
    assert not p.is_trained(store, "UnseenPert", 0)
    assert "training recipe" in p.unusable_reason(store, "UnseenPert", 0)


@pytest.mark.parametrize("field,label", [
    ("gene_axis_sha", "gene panel"),
    ("split_sha", "train/test split"),
    ("recipe_sha", "training recipe"),
])
def test_container_detects_every_staleness_field(isolated_checkpoints, monkeypatch,
                                                 store_on_disk, field, label):
    """The mechanism is shared tier code, but a tier CAN override
    `_staleness_reason` (the container did, until it was unified) — so both tiers
    are pinned rather than one being assumed to imply the other."""
    store = store_on_disk(n_bins=1)
    _stub_container(monkeypatch, store)
    p = FakeContainer()
    p.fit(store, "UnseenPert", 0)

    fp_path = p._run_dir(store, "UnseenPert", 0) / FINGERPRINT_FILE
    fp = json.loads(fp_path.read_text())
    fp[field] = "SOMETHING-ELSE"
    fp_path.write_text(json.dumps(fp))

    assert not p.is_trained(store, "UnseenPert", 0)
    assert label in p.unusable_reason(store, "UnseenPert", 0)


def test_a_live_claim_blocks_a_second_fit(isolated_checkpoints, monkeypatch, store_on_disk):
    """S5: two array tasks must not train into one run dir."""
    import time
    store = store_on_disk(n_bins=1)
    _stub_container(monkeypatch, store)
    p = FakeContainer()
    run_dir = p._run_dir(store, "UnseenPert", 0)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / CLAIM_FILE).write_text(json.dumps(
        {"host": "other", "pid": 4242, "started_at": time.time()}))

    with pytest.raises(RuntimeError, match="already being trained"):
        p.fit(store, "UnseenPert", 0)
    p.fit(store, "UnseenPert", 0, force=True)        # --force overrides
    assert p.is_trained(store, "UnseenPert", 0)


# ---------------------------------------------------------------------------
# tensor_map branches that adamson16 could never reach
# ---------------------------------------------------------------------------

def test_covariate_matching_across_multiple_bins(isolated_checkpoints, monkeypatch, store_on_disk):
    """The cell-axis branch: with >1 bin the mapping must match on covariate as
    well as condition, or every bin gets bin 0's prediction."""
    store = store_on_disk(n_bins=3, n_kos=8)
    truth = _stub_container(monkeypatch, store, covariate=True)
    p = FakeContainer()
    p.fit(store, "UnseenPert", 0)
    out = p.predict(store, "UnseenPert", 0)

    split = store.split("UnseenPert", 0)
    assert out.shape == (3, len(split.test_ko_indices), store.n_genes)
    for i in range(3):
        for j, k in enumerate(split.test_ko_indices):
            np.testing.assert_allclose(out[i, j], truth[(i, int(k))], atol=1e-5)


def test_a_perturbation_the_container_skipped_stays_nan(isolated_checkpoints, monkeypatch, store_on_disk):
    """A container that drops a condition must leave NaN, not a wrong number."""
    store = store_on_disk(n_bins=1, n_kos=8)
    split = store.split("UnseenPert", 0)
    dropped = int(split.test_ko_indices[0])
    _stub_container(monkeypatch, store, drop_kos=(dropped,))

    p = FakeContainer()
    p.fit(store, "UnseenPert", 0)
    out = p.predict(store, "UnseenPert", 0)
    assert np.isnan(out[0, 0]).all(), "the skipped perturbation must be NaN"
    assert np.isfinite(out[0, 1:]).all(), "the rest must still be predicted"


def test_partial_gene_overlap_leaves_missing_genes_nan(isolated_checkpoints, monkeypatch, store_on_disk):
    """A container with a narrower panel must not shift genes into wrong slots."""
    store = store_on_disk(n_bins=1, n_genes=10)
    subset = store.gene_names[:4]
    _stub_container(monkeypatch, store, gene_subset=subset)

    p = FakeContainer()
    p.fit(store, "UnseenPert", 0)
    out = p.predict(store, "UnseenPert", 0)
    assert np.isfinite(out[..., :4]).all()
    assert np.isnan(out[..., 4:]).all()


def test_single_cell_and_pseudobulk_agree(isolated_checkpoints, monkeypatch,
                                          store_on_disk):
    """A container may emit one pseudobulk row per condition or many single
    cells; the mapping averages, so both must reduce to the same delta.

    (Ported from `tensor_map --selftest`, which only ever ran by hand.)
    """
    store = store_on_disk(n_bins=1, n_kos=8)

    outs = []
    for rows_per_cond in (1, 5):
        _stub_container(monkeypatch, store, rows_per_ko=rows_per_cond)
        p = FakeContainer()
        p.fit(store, "UnseenPert", 0, force=True)
        outs.append(p.predict(store, "UnseenPert", 0))

    np.testing.assert_allclose(outs[0], outs[1], atol=1e-5)
