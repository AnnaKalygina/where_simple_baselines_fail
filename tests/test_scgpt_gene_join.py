"""Unit tests for V26 — resolving our genes to scGPT's vocabulary by Ensembl ID.

scGPT tokenises against a fixed vocabulary. Matching our genes to it by HGNC
SYMBOL silently discarded every gene HGNC has since renamed (`AARS`/`AARS1`,
`ATP5B`/`ATP5F1B`, `SRPR`/`SRPRA`) — on adamson16, 965 genes and 11 of 89
perturbation targets that scGPT knows perfectly well. Resolving by Ensembl
accession recovers them.

The failure modes are all silent: a renamed gene simply vanishes from the panel,
and a collision (two of our genes claiming one vocabulary token) would emit a
duplicate token id that `np.where(...)[0][0]` resolves to the first match. So the
resolver's post-conditions are pinned here.

Same stub harness as ``test_scgpt_pairing.py``: the code under test is the real
wrapper, only its container-only dependencies are fake.
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


# The vocabulary side: `feature_id` -> `feature_name`, a bijection, as the real
# gene_info_scgpt.csv is (60 664 unique on both axes).
VOCAB = [
    ("ENSG00000090861", "AARS1"),        # our data still says AARS
    ("ENSG00000110955", "ATP5F1B"),      # our data still says ATP5B
    ("ENSG00000102145", "GATA1"),        # unchanged
    ("ENSG00000141510", "TP53"),         # unchanged
    ("ENSG00000230806", "SRGAP2-AS1"),   # the adamson16 collision, see below
]


@pytest.fixture
def gene_info(tmp_path):
    p = tmp_path / "gene_info_scgpt.csv"
    pd.DataFrame(VOCAB, columns=["feature_id", "feature_name"]).to_csv(p, index=False)
    return p


class _FakeAdata:
    """Stands in for AnnData — the resolver reads var/var_names/n_vars only."""

    def __init__(self, symbols, **cols):
        self.var_names = pd.Index(symbols)
        self.var = pd.DataFrame(cols, index=self.var_names)

    @property
    def n_vars(self):
        return len(self.var_names)


def _wrapper(W, gene_info, mode="ensembl_first"):
    w = W.SCGPTWrapper.__new__(W.SCGPTWrapper)
    w.config = {"gene_id_join": mode, "scgpt_gene_info_path": str(gene_info)}
    w._report = {}
    w._oov_masked = False
    return w


def _resolve(W, gene_info, symbols, mode="ensembl_first", **cols):
    w = _wrapper(W, gene_info, mode)
    a = _FakeAdata(symbols, **cols)
    w._ensure_symbol_scgpt(a)
    return a, w._report


# ------------------------------------------------------- the point of the change

def test_ensembl_recovers_a_renamed_symbol(W, gene_info):
    """AARS is absent from the vocabulary; its accession is not."""
    a, rep = _resolve(W, gene_info, ["AARS", "GATA1"],
                      ensembl_id=["ENSG00000090861", "ENSG00000102145"])
    assert list(a.var["symbol_scgpt"]) == ["AARS1", "GATA1"]
    assert rep["n_genes_matched_by_symbol"] == 1      # GATA1
    assert rep["n_genes_matched_by_ensembl"] == 1     # AARS -> AARS1
    assert rep["n_genes_unresolved"] == 0


def test_symbol_only_mode_reproduces_the_old_join(W, gene_info):
    a, rep = _resolve(W, gene_info, ["AARS", "GATA1"], mode="symbol_only",
                      ensembl_id=["ENSG00000090861", "ENSG00000102145"])
    assert list(a.var["symbol_scgpt"]) == [None, "GATA1"]
    assert rep["n_genes_matched_by_ensembl"] == 0
    assert rep["n_genes_unresolved"] == 1
    assert rep["ensembl_column"] is None


def test_dataset_without_any_accession_column_degrades_to_symbols(W, gene_info):
    """replogle22 / jiang24 / wessels23 / xatlas_orion carry no accessions."""
    a, rep = _resolve(W, gene_info, ["AARS", "GATA1"], highly_variable=[True, False])
    assert list(a.var["symbol_scgpt"]) == [None, "GATA1"]
    assert rep["ensembl_column"] is None
    assert rep["n_genes_unresolved"] == 1


# --------------------------------------------------------------- robustness

def test_accession_column_is_found_by_value_not_by_name(W, gene_info):
    """norman19 spells it `ensemble_id`; replogle20 spells it `gene_id`."""
    for col in ("ensemble_id", "gene_id", "something_nobody_would_guess"):
        a, rep = _resolve(W, gene_info, ["AARS", "GATA1"],
                          **{col: ["ENSG00000090861", "ENSG00000102145"]})
        assert rep["ensembl_column"] == col
        assert list(a.var["symbol_scgpt"]) == ["AARS1", "GATA1"]


def test_partly_populated_accession_column_falls_back_per_gene(W, gene_info):
    """frangieh21's column is only ~78 % accessions — the rest must use symbols."""
    a, rep = _resolve(
        W, gene_info,
        ["AARS", "GATA1", "TP53", "ATP5B", "NOTAGENE"],
        ensembl_id=["ENSG00000090861", "", "not-an-accession",
                    "ENSG00000110955", ""])
    assert list(a.var["symbol_scgpt"]) == ["AARS1", "GATA1", "TP53", "ATP5F1B", None]
    assert rep["n_genes_matched_by_symbol"] == 2       # GATA1, TP53 (by name)
    assert rep["n_genes_matched_by_ensembl"] == 2      # AARS, ATP5B
    assert rep["n_genes_unresolved"] == 1


def test_a_column_with_no_accessions_is_never_selected(W, gene_info):
    a, rep = _resolve(W, gene_info, ["GATA1", "TP53"],
                      some_ids=["junk", "also-junk"], ncounts=[1, 2])
    assert rep["ensembl_column"] is None
    assert list(a.var["symbol_scgpt"]) == ["GATA1", "TP53"]


def test_the_best_accession_column_wins_when_several_look_plausible(W, gene_info):
    a, rep = _resolve(W, gene_info, ["AARS", "ATP5B"],
                      partial=["ENSG00000090861", "junk"],
                      complete=["ENSG00000090861", "ENSG00000110955"])
    assert rep["ensembl_column"] == "complete"
    assert list(a.var["symbol_scgpt"]) == ["AARS1", "ATP5F1B"]


def test_unresolved_genes_become_na_and_are_counted(W, gene_info):
    a, rep = _resolve(W, gene_info, ["NOTAGENE", "ALSONOT"],
                      ensembl_id=["ENSG99999999999", "ENSG88888888888"])
    assert a.var["symbol_scgpt"].isna().all()
    assert rep["n_genes_unresolved"] == 2
    assert rep["n_genes_oov_dropped"] == 2


# ----------------------------------------------------------------- collisions

def test_symbol_identity_wins_a_collision(W, gene_info):
    """The live adamson16 case.

    `RP11-343N15.1` (ENSG00000230806) resolves by accession to `SRGAP2-AS1`,
    which our own gene named `SRGAP2-AS1` already holds by symbol. The symbol
    match must win, and the accession claimant must be dropped rather than
    emitting a second copy of the same vocabulary token.
    """
    a, rep = _resolve(W, gene_info, ["RP11-343N15.1", "SRGAP2-AS1"],
                      ensembl_id=["ENSG00000230806", "ENSG00000233501"])
    assert list(a.var["symbol_scgpt"]) == [None, "SRGAP2-AS1"]
    assert rep["n_gene_id_collisions"] == 1


def test_collision_order_does_not_matter(W, gene_info):
    """Symbol identity is resolved in a first pass, so var order is irrelevant."""
    a, _ = _resolve(W, gene_info, ["SRGAP2-AS1", "RP11-343N15.1"],
                    ensembl_id=["ENSG00000233501", "ENSG00000230806"])
    assert list(a.var["symbol_scgpt"]) == ["SRGAP2-AS1", None]


def test_two_genes_sharing_one_accession_do_not_both_resolve(W, gene_info):
    a, rep = _resolve(W, gene_info, ["AARS", "AARS_DUP"],
                      ensembl_id=["ENSG00000090861", "ENSG00000090861"])
    kept = [v for v in a.var["symbol_scgpt"] if v is not None]
    assert kept == ["AARS1"]
    assert rep["n_gene_id_collisions"] == 1


def test_resolution_is_injective(W, gene_info):
    """The post-condition that makes a duplicate token id impossible."""
    a, _ = _resolve(W, gene_info,
                    ["AARS", "AARS1", "GATA1", "RP11-343N15.1", "SRGAP2-AS1"],
                    ensembl_id=["ENSG00000090861", "", "ENSG00000102145",
                                "ENSG00000230806", "ENSG00000233501"])
    kept = [v for v in a.var["symbol_scgpt"] if v is not None]
    assert len(kept) == len(set(kept))


# ------------------------------------------------------------- the invariant

@pytest.mark.parametrize("symbols,accessions", [
    (["AARS", "GATA1", "NOTAGENE"],
     ["ENSG00000090861", "ENSG00000102145", ""]),
    (["RP11-343N15.1", "SRGAP2-AS1", "TP53"],
     ["ENSG00000230806", "ENSG00000233501", "ENSG00000141510"]),
    (["ATP5B", "ATP5F1B"], ["ENSG00000110955", ""]),
])
def test_ensembl_first_is_a_superset_of_symbol_only(W, gene_info, symbols, accessions):
    """The invariant that makes this change safe: it may only ADD genes.

    No gene the old join kept may be lost, so no scGPT result can get worse for a
    gene that already worked.
    """
    a_new, _ = _resolve(W, gene_info, symbols, ensembl_id=list(accessions))
    a_old, _ = _resolve(W, gene_info, symbols, mode="symbol_only",
                        ensembl_id=list(accessions))
    kept_new = {g for g, v in zip(symbols, a_new.var["symbol_scgpt"]) if v is not None}
    kept_old = {g for g, v in zip(symbols, a_old.var["symbol_scgpt"]) if v is not None}
    assert kept_old <= kept_new


def test_an_unknown_join_mode_is_refused(W, gene_info):
    with pytest.raises(ValueError, match="gene_id_join"):
        _resolve(W, gene_info, ["GATA1"], mode="whatever_looks_plausible")
