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
from pathlib import Path

import pytest

PY = sys.executable
#: Derived, not hardcoded — this file used to name the cluster path here while
#: deriving it from `__file__` further down, so the suite only ran in one place.
REPO = str(Path(__file__).resolve().parents[1])


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
    # No "transformers": the nine variants are `TorchPredictor`s and land in
    # `dl`. Leaving it here would let one that regressed to a plain `Predictor`
    # fall back to its module name and still satisfy `got <= known` — passing on
    # exactly the bug this test exists to catch.
    known = {"analytical", "learned", "controls", "dl"}
    got = set(eval(cats))
    assert got <= known, f"unexpected category (a class built oddly?): {got - known}"


# ---------------------------------------------------------------------------
# The recipes we actually ship
# ---------------------------------------------------------------------------
# The container e2e tests exercise a FakeContainer against a synthetic recipe, so
# nothing checks the `docker/<model>/model.yaml` files that real runs read. A
# malformed one is otherwise found at the top of a GPU job.


def _shipped_container_predictors():
    from benchmark.predictors.base import PREDICTOR_REGISTRY, _ensure_predictors_loaded
    from benchmark.predictors.container_predictor import ContainerPredictor
    _ensure_predictors_loaded()
    return sorted(
        (name, cls) for name, cls in PREDICTOR_REGISTRY.items()
        if isinstance(cls, type) and issubclass(cls, ContainerPredictor))


def test_there_is_at_least_one_shipped_container_model():
    """Guards the parametrisation below: an empty registry would make every
    recipe test vacuously pass."""
    assert _shipped_container_predictors(), "no ContainerPredictor is registered"


@pytest.mark.parametrize("name,cls", _shipped_container_predictors())
def test_shipped_recipe_is_valid_and_usable(name, cls):
    """Load each real `model.yaml` through the same validator `_recipe()` uses.

    The .sif is deliberately NOT required to exist: a recipe is authored before
    its image is built (PRESAGE's is), and tying this to the image would make the
    check unrunnable exactly when it is most useful.
    """
    p = cls()
    recipe = cls._recipe()                      # raises on missing/unknown keys

    assert recipe["model_key"], f"{name}: empty model_key"
    assert p.sif_path.startswith("docker/"), f"{name}: sif_path not repo-relative"
    assert p.entry and p.entry[0] == "python", f"{name}: unexpected entry {p.entry}"
    assert p.code_binds, f"{name}: no code_binds — the wrapper would not be mounted"
    assert p.default_hyperparameters, f"{name}: no pinned hyperparameters"
    assert p._expected_artifacts(), f"{name}: nothing distinguishes trained from empty"
    assert p.train_timeout > 0 and p.predict_timeout > 0, f"{name}: non-positive timeout"

    # extra_config validates its own values; calling it is the check.
    assert isinstance(p.extra_config, dict)

    # Artefact paths are run-dir-relative: an absolute one would escape the run
    # dir and make two folds share state.
    import os
    bad = [a for a in p._expected_artifacts() if os.path.isabs(a)]
    assert not bad, f"{name}: absolute expected_artifacts {bad}"


@pytest.mark.parametrize("name,cls", _shipped_container_predictors())
def test_shipped_code_binds_point_at_real_directories(name, cls):
    """The wrapper is bind-mounted, not baked, so a typo'd path yields an empty
    mount and a `ModuleNotFoundError` inside the container."""
    repo = Path(REPO)
    for src in cls._recipe()["code_binds"]:
        assert (repo / src).is_dir(), f"{name}: code_bind source {src} is not a directory"
        assert list((repo / src).glob("run_model.py")), \
            f"{name}: {src} has no run_model.py — `entry` would have nothing to run"


@pytest.mark.parametrize("name,cls", _shipped_container_predictors())
def test_cell_aware_models_declare_at_least_one_cell_axis_regime(name, cls):
    """And cell-blind ones do not: `cell_aware` and `scenarios` are two halves of
    the same claim, and they drifting apart is how a cell-blind model ends up
    scored on a cell-axis regime."""
    from benchmark.predictors._container import leakage
    cell_axis = set(cls.scenarios) & set(leakage.BIN_AXIS_REGIMES)
    if cls.cell_aware:
        assert cell_axis, f"{name}: cell_aware but declares no cell-axis regime"
    else:
        assert not cell_axis, (
            f"{name}: declares cell-axis regime(s) {sorted(cell_axis)} while "
            f"cell_aware=False — it cannot condition on the covariate")
