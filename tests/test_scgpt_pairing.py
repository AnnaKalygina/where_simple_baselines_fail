"""Unit tests for the two structural rebuilds in docker/scgpt/scgpt_wrapper.py.

V6 (bucket dataloaders on the per-cell ``obs[split_name]`` tag rather than a
condition list) and V7 (draw control donors from a covariate-, batch- and
split-matched pool) are the fixes that make scGPT-ct leak-safe and cell-aware.
Both fail SILENTLY when broken — a condition-level split inflates the score
without raising, and a cross-covariate control pool makes the model cell-blind
while ``obs['covariate']`` still claims otherwise. Nothing else in the suite
would catch either, so they are pinned here.

The wrapper runs inside the container, so ``gears`` / ``scgpt`` /
``torch_geometric`` are not importable on the host. They are stubbed: the code
under test is the real wrapper, only its container-only dependencies are fake.
"""
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
WRAPPER_DIR = REPO / "docker" / "scgpt"


def _stub(name, **attrs):
    m = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    sys.modules[name] = m
    return m


@pytest.fixture(scope="module")
def W():
    """Import scgpt_wrapper with its container-only dependencies stubbed."""
    pytest.importorskip("torch", reason="scgpt_wrapper imports torch")

    class _PertData:
        def __init__(self, data_path):
            self.data_path = data_path

    class _Data:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    class _TransformerGenerator:
        def __init__(self, *a, **k):
            pass

    _stub("gears", PertData=_PertData)
    _stub("torch_geometric")
    _stub("torch_geometric.data", Data=_Data)
    _stub("torch_geometric.loader", DataLoader=object)
    _stub("scgpt")
    _stub("scgpt.model", TransformerGenerator=_TransformerGenerator)
    _stub("scgpt.tokenizer")
    _stub("scgpt.tokenizer.gene_tokenizer", GeneVocab=object)
    _stub("scgpt.utils", map_raw_id_to_vocab_id=lambda *a: None)
    _stub("scgpt.utils.util", load_pretrained=lambda **k: None)

    sys.path.insert(0, str(WRAPPER_DIR))
    try:
        import scgpt_wrapper
    finally:
        sys.path.remove(str(WRAPPER_DIR))
    return scgpt_wrapper


class _FakeCtrl:
    """Stands in for ``pert_data.ctrl_adata`` (only ``.obs``/``.n_obs`` are read)."""

    def __init__(self, df):
        self.obs = df
        self.n_obs = len(df)


def _pert_data(W, cov, batch, split, *, use_cov=True, use_batch=True):
    p = W.SCGPTPertData.__new__(W.SCGPTPertData)
    p.split_col = "split"
    p.cov_col = "cov" if use_cov else None
    p.batch_col = "batch" if use_batch else None
    p._pool_cache = {}
    p._ctrl_meta = None
    p._ctrl_X_cache = None
    p.pairing_tier_counts = {"cov+batch+split": 0, "cov+split": 0, "split_only": 0}
    p.ctrl_adata = _FakeCtrl(pd.DataFrame({"cov": cov, "batch": batch, "split": split}))
    return p


# --------------------------------------------------------------- V7: pairing

def test_exact_covariate_and_batch_match_wins(W):
    p = _pert_data(W, cov=["A", "A", "B", "B"], batch=["b1", "b2", "b1", "b2"],
                   split=["train"] * 4)
    assert list(p._ctrl_pool("A", "b1", ("train",))) == [0]
    assert p.pairing_tier_counts["cov+batch+split"] == 1


def test_missing_batch_falls_back_to_covariate_and_records_the_tier(W):
    p = _pert_data(W, cov=["A", "A", "B"], batch=["b1", "b1", "b2"],
                   split=["train"] * 3)
    assert list(p._ctrl_pool("A", "b9", ("train",))) == [0, 1]
    assert p.pairing_tier_counts["cov+split"] == 1
    assert p.pairing_tier_counts["cov+batch+split"] == 0


def test_train_donors_never_come_from_held_out_cells(W):
    """The leak guard: a held-out control cell must not reach a training row."""
    p = _pert_data(W, cov=["A"] * 4, batch=["b1"] * 4,
                   split=["train", "test", "test", "val"])
    assert list(p._ctrl_pool("A", "b1", ("train",))) == [0]


def test_val_degrades_to_train_controls_when_a_split_has_none(W):
    """mcfaline23 has zero val/test control cells; the fallback is load-bearing."""
    p = _pert_data(W, cov=["A"] * 3, batch=["b1"] * 3, split=["train"] * 3)
    assert list(p._ctrl_pool("A", "b1", ("val", "train"))) == [0, 1, 2]


def test_refuses_to_borrow_another_covariates_controls(W):
    """Relaxing the covariate is the silent failure V7 exists to prevent."""
    p = _pert_data(W, cov=["A", "A"], batch=["b1", "b1"], split=["train", "train"])
    with pytest.raises(RuntimeError, match="Refusing to borrow controls"):
        p._ctrl_pool("B", "b1", ("train",))


def test_single_cell_line_degrades_to_split_only_but_keeps_the_split_guard(W):
    p = _pert_data(W, cov=[""] * 3, batch=[""] * 3,
                   split=["train", "test", "train"], use_cov=False, use_batch=False)
    assert list(p._ctrl_pool("", "", ("train",))) == [0, 2]
    assert p.pairing_tier_counts["split_only"] == 1


# ------------------------------------------------------------- V6: bucketing

class _Graph:
    def __init__(self, split):
        self.split = split


def test_dataloaders_bucket_on_the_per_cell_tag_not_the_condition(W, monkeypatch):
    """The cell-axis case: one condition legitimately spans two splits.

    A condition-level split cannot represent this — it would put every cell of
    ``ctrl+KO1`` in train, including the val-labelled one.
    """
    monkeypatch.setattr(W, "DataLoader", lambda graphs, **kw: graphs)
    p = W.SCGPTPertData.__new__(W.SCGPTPertData)
    p.split_sizes = {}
    p.dataset_processed = {
        "ctrl+KO1": [_Graph("train"), _Graph("val")],
        "ctrl+KO2": [_Graph("train")],
    }
    p.get_dataloader(batch_size=2)
    assert p.split_sizes == {"train": 2, "val": 1, "test": 0}


def test_untagged_graphs_are_refused(W, monkeypatch):
    """An untagged graph means a stale cache (V15) — never bucket it silently."""
    monkeypatch.setattr(W, "DataLoader", lambda graphs, **kw: graphs)
    p = W.SCGPTPertData.__new__(W.SCGPTPertData)
    p.split_sizes = {}
    p.dataset_processed = {"ctrl+KO1": [_Graph("train"), _Graph("val"), _Graph("")]}
    with pytest.raises(RuntimeError, match="no usable per-cell split tag"):
        p.get_dataloader(batch_size=2)


def test_empty_validation_split_is_refused(W, monkeypatch):
    """Best-val selection (V10) would have nothing to select on."""
    monkeypatch.setattr(W, "DataLoader", lambda graphs, **kw: graphs)
    p = W.SCGPTPertData.__new__(W.SCGPTPertData)
    p.split_sizes = {}
    p.dataset_processed = {"ctrl+KO1": [_Graph("train")]}
    with pytest.raises(RuntimeError, match="empty validation split"):
        p.get_dataloader(batch_size=2)


# ------------------------------------------------------------------ V13

def test_control_labels_match_whole_strings_not_substrings(W):
    """Substring matching would rewrite gene names that merely contain 'ctrl'."""
    assert W._is_control_label("ctrl")
    assert W._is_control_label("control")
    assert not W._is_control_label("RctrlSEL")
