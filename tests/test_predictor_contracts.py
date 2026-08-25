"""Run the predictor contracts (verify.py L4) under pytest.

The synthetic-fixture contract suite already lives in benchmark/verify.py (a core
module — tests importing it respects the scripts->core rule). Rather than
duplicate the ~600 lines of fixtures, we import CONTRACTS and execute each one
with a fresh Results, asserting it records no failure. This gives the contracts a
fast, CI-friendly pytest entry point without a risky physical extraction.
"""
import pytest

from benchmark.verify import (
    CONTRACTS,
    Results,
    _c_target_errors,
    _c_missing_genes,
)


@pytest.mark.parametrize("name", sorted(CONTRACTS))
def test_predictor_contract(name):
    r = Results()
    CONTRACTS[name](r)
    assert r.failed == 0, f"{name} contract failed: {r.fail_names}"


def test_target_error_contracts():
    r = Results()
    _c_target_errors(r)
    assert r.failed == 0, r.fail_names


@pytest.mark.torch
def test_missing_genes_contract():
    r = Results()
    _c_missing_genes(r)
    assert r.failed == 0, r.fail_names
