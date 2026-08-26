"""Invariants of the predictor infrastructure that nothing else checks.

These have been verified by hand several times — which is exactly the problem.
Each one is silent when it breaks: the registry simply comes back smaller, or a
predictor quietly disappears from a category, and the first symptom is an
"unknown predictor" much later.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

PY = sys.executable
REPO = "/cluster/work/boeva/virtual_cell_reasoning"


def _run_isolated(*parts: str) -> subprocess.CompletedProcess:
    """Run snippets in a FRESH interpreter.

    A subprocess rather than sys.modules surgery: blocking an import and
    re-importing half a package in-process leaves the test session in a state
    later tests inherit.

    Each part is dedented SEPARATELY — they come from string literals at
    different indentation levels, and dedenting the concatenation would only
    strip their common prefix and leave one half over-indented.
    """
    src = "\n".join(textwrap.dedent(p) for p in parts)
    return subprocess.run(
        [PY, "-c", src],
        capture_output=True, text=True, timeout=300,
        env={"PYTHONPATH": REPO, "PATH": "/usr/bin:/bin", "HOME": "/tmp"},
        cwd=REPO)


# Plain substitution, not str.format: the snippet contains braces of its own.
#
# `sys.modules[name] = None` is how you simulate a module being ABSENT: any
# `import name` then raises ImportError, which is what a library's
# `try: import torch / except ImportError` guard expects. Raising from a
# meta-path finder instead aborts the import machinery in a way those guards
# never see — and anndata, which optionally imports torch, then fails to load
# at all. That would test our blocker, not our code.
BLOCKER = """
    import sys
    for _n in __BLOCKED__:
        sys.modules[_n] = None
"""


@pytest.mark.parametrize("blocked,why", [
    ("{'torch'}", "the preprocess env has no torch"),
    ("{'yaml'}", "the preprocess env has no pyyaml"),
    ("{'torch', 'yaml'}", "both absent, as in the real preprocess env"),
])
def test_registry_loads_without_optional_heavy_deps(blocked, why):
    """Every predictor module must import in an env missing torch/pyyaml.

    This is what the deferred-import discipline buys: `ContainerPredictor`
    imports yaml inside `_recipe()`, `TorchPredictor` imports torch inside
    `_train`/`_infer`. Move either to module level and the registry silently
    loses predictors in `preprocess` — or, since C5, fails loudly. Either way
    this test is the tripwire.
    """
    r = _run_isolated(BLOCKER.replace("__BLOCKED__", blocked), """
        from benchmark.predictors.base import (
            PREDICTOR_REGISTRY, _ensure_predictors_loaded)
        _ensure_predictors_loaded()
        print("COUNT", len(PREDICTOR_REGISTRY))
    """)
    assert r.returncode == 0, f"registry failed to load when {why}:\n{r.stderr[-1500:]}"
    count = int(r.stdout.split("COUNT")[1].split()[0])
    assert count >= 30, f"registry came back short ({count}) when {why}"


def test_torch_internals_are_not_imported_at_registry_time():
    """`_torch/*` must stay unimported until a transformer actually trains.

    It is what makes the module-level `import torch` in that package safe.
    """
    r = _run_isolated("""
        import sys
        from benchmark.predictors.base import _ensure_predictors_loaded
        _ensure_predictors_loaded()
        leaked = [m for m in sys.modules
                  if m.startswith('benchmark.predictors._torch.')
                  and not m.endswith('.recipe')]
        print("LEAKED", leaked)
    """)
    assert r.returncode == 0, r.stderr[-1500:]
    leaked = r.stdout.split("LEAKED")[1].strip()
    assert leaked == "[]", f"eagerly imported torch internals: {leaked}"


def test_every_registered_predictor_lands_in_a_known_category():
    """Categories are derived, so a class built in an unusual way (the nine
    transformers are made with `type()`) can land somewhere unexpected —
    `abc`, in one real case, until `__module__` was set explicitly."""
    r = _run_isolated("""
        from benchmark.predictors.base import (
            PREDICTOR_REGISTRY, _ensure_predictors_loaded, predictor_category)
        _ensure_predictors_loaded()
        cats = {predictor_category(c) for c in PREDICTOR_REGISTRY.values()}
        print("CATS", sorted(cats))
    """)
    assert r.returncode == 0, r.stderr[-1500:]
    cats = r.stdout.split("CATS")[1].strip()
    known = {"analytical", "learned", "controls", "dl", "transformers"}
    got = set(eval(cats))
    assert got <= known, f"unexpected category (a class built oddly?): {got - known}"
