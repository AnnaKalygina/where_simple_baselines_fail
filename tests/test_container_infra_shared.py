"""Unit tests for the helpers the three container predictors share.

These exist because each helper replaced code that had been COPIED between
`GEARSContainer`, `PRESAGEContainer` and `SCGPTContainer`. Copies drift silently:
the control-label rule was spelled four different ways in one file, and the
perturbation-flag vector was built twice — which is exactly how V25 (predict used
`np.sign(p)`, train used `1`) came about. One implementation only helps if it
stays correct, so the contract each caller depends on is pinned here.
"""
import json

import pandas as pd
import pytest

from benchmark.predictors._container import leakage


# ===================================================================
# The control-label predicate
# ===================================================================
#
# `_preflight`'s B1/B2 checks use the SERIES form; the coverage checks use the
# SCALAR form. If the two ever disagree, one gate's idea of "this is a control"
# stops matching another's — so their equivalence is the thing to pin, not either
# one alone.

CONTROL_CASES = ["control", "ctrl", "ctrl_iegfp", "ctrl_something", "CTRL",
                 "Ctrl", "non-ctrl-ish", "ctrl+AARS", "AARS+ctrl"]
PERTURBATION_CASES = ["AARS", "TP53", "GATA1", "AARS+TP53", "non-targeting",
                      "CTRB1", "CTLA4", ""]


@pytest.mark.parametrize("label", CONTROL_CASES + PERTURBATION_CASES)
def test_scalar_and_series_control_predicates_agree(label):
    series = leakage.control_mask(pd.Series([label]),
                                  leakage.DEFAULT_CONTROL_LABELS)
    assert bool(series.iloc[0]) is leakage.is_control_label(label), label


def test_control_labels_are_recognised():
    assert all(leakage.is_control_label(c) for c in CONTROL_CASES)


def test_real_gene_symbols_are_not_mistaken_for_controls():
    # CTRB1 and CTLA4 start with "CT" but contain no "ctrl" — the substring rule
    # is what makes 'ctrl_iegfp'-style labels work, and it must not overreach.
    assert not any(leakage.is_control_label(g) for g in PERTURBATION_CASES)


def test_control_mask_is_the_leakage_gates_own_function():
    """Not a reimplementation — the same object, so it cannot drift."""
    assert leakage.control_mask is leakage._is_control


# ===================================================================
# ContainerPredictor._check_target_coverage / _read_json_report
# ===================================================================

@pytest.fixture(scope="module")
def P():
    from benchmark.predictors.container_predictor import ContainerPredictor
    return ContainerPredictor


class _Stub:
    """Minimal stand-in: the helpers touch only `name` and the floor."""

    name = "stub-ct"

    def __init__(self, floor=0.5):
        self._MIN_TARGET_COVERAGE = floor

    def __getattr__(self, item):
        from benchmark.predictors.container_predictor import ContainerPredictor
        return getattr(ContainerPredictor, item).__get__(self)


def _conds(train=(), val=(), test=()):
    return {"train": list(train), "val": list(val), "test": list(test)}


def test_full_coverage_passes(P):
    P._check_target_coverage(
        _Stub(), _conds(train=["AARS", "TP53"], test=["GATA1"]),
        is_covered={"AARS", "TP53", "GATA1"}.__contains__, source="ref")


def test_partial_coverage_above_the_floor_passes(P):
    # 2/3 = 0.67 > 0.5
    P._check_target_coverage(
        _Stub(), _conds(train=["AARS", "TP53"], test=["GATA1"]),
        is_covered={"AARS", "TP53"}.__contains__, source="ref")


def test_coverage_below_the_floor_raises(P):
    with pytest.raises(RuntimeError, match="gene-ID space mismatch"):
        P._check_target_coverage(
            _Stub(), _conds(train=["AARS", "TP53"], test=["GATA1", "MYC"]),
            is_covered={"AARS"}.__contains__, source="ref")


def test_the_floor_is_inclusive_at_exactly_the_boundary(P):
    """frac == floor must PASS; only strictly below stops the run."""
    P._check_target_coverage(
        _Stub(), _conds(train=["AARS", "TP53"]),
        is_covered={"AARS"}.__contains__, source="ref")     # 1/2 == 0.5


def test_controls_are_not_counted_as_targets(P):
    """A control label must not be parsed as a perturbation and then 'dropped'."""
    P._check_target_coverage(
        _Stub(), _conds(train=["ctrl", "control", "ctrl_iegfp"], test=["AARS"]),
        is_covered={"AARS"}.__contains__, source="ref")     # 1/1, not 1/4


def test_no_targets_at_all_is_skipped_not_a_crash(P):
    P._check_target_coverage(_Stub(), _conds(train=["ctrl"]),
                             is_covered=lambda g: False, source="ref")


def test_read_json_report_round_trips(P, tmp_path):
    (tmp_path / "r.json").write_text(json.dumps({"best_epoch": 7}))
    assert P._read_json_report(_Stub(), tmp_path, "r.json") == {"best_epoch": 7}


def test_read_json_report_honours_a_key(P, tmp_path):
    (tmp_path / "r.json").write_text(json.dumps({"report": {"a": 1}, "other": 2}))
    assert P._read_json_report(_Stub(), tmp_path, "r.json", key="report") == {"a": 1}


def test_read_json_report_missing_file_is_none_not_an_error(P, tmp_path):
    assert P._read_json_report(_Stub(), tmp_path, "absent.json") is None


def test_read_json_report_corrupt_file_warns_and_returns_none(P, tmp_path):
    """A torn provenance file must not destroy an otherwise valid run."""
    (tmp_path / "r.json").write_text("{not json")
    assert P._read_json_report(_Stub(), tmp_path, "r.json") is None


def test_read_json_report_missing_key_is_none(P, tmp_path):
    (tmp_path / "r.json").write_text(json.dumps({"other": 2}))
    assert P._read_json_report(_Stub(), tmp_path, "r.json", key="report") is None
