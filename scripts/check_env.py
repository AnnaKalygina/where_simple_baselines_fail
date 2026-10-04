#!/usr/bin/env python
"""Validate that the environment can actually run every host-side script.

Run it with the interpreter you want to check::

    <prefix>/bin/python scripts/check_env.py            # all tiers
    <prefix>/bin/python scripts/check_env.py --static   # imports only, fast

This exists because the repo lost five months to a failure that no test caught:
a `pip install` ran out of disk mid-transaction and left `site-packages/scanpy/`
as a single 0-byte `__init__.py` with no metadata. An empty `__init__.py` SHADOWS
the package name, so `import scanpy` succeeded and exported nothing, surfacing as
`AttributeError: module 'scanpy' has no attribute 'pp'` rather than a clean
ImportError. The same transaction left four packages with two `.dist-info`
directories apiece, so `conda list` and `importlib.metadata` disagreed about which
version was installed. Tier 0 finds all of that in about a second.

Tiers:
  0  environment integrity   — stubs, duplicate metadata, mixed provenance, ABI
  1  import coverage, static — AST-parse EVERY host .py, resolve each import
  2  import coverage, live   — actually import the modules and entry points

`docker/` is excluded on purpose: those wrappers import scgpt / gears /
torch_geometric, which exist only inside the .sif images and must NOT be present
here. Container code is validated by the container smokes instead.

Tier 2 IMPORTS the entry-point scripts; it never executes them. That is not
fastidiousness: `data/*/get_data.py` take no arguments, so running one with
`--help` does not print help — it ignores the flag and regenerates the dataset,
overwriting `data/<ds>/<ds>_processed.h5ad`. Importing is safe because every one
of them carries a `__main__` guard, and it catches a missing dependency just as
well. Do not "improve" this tier by shelling out to the scripts.
"""
from __future__ import annotations

import argparse
import ast
import importlib
import importlib.metadata as md
import importlib.util
import os
import re
import sys
import sysconfig
import warnings
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

#: Scanned for imports. `docker/` is absent deliberately — see the module docstring.
HOST_TREES = ("benchmark", "data", "tests", "scripts")

#: Compiled extensions: a numpy ABI mismatch shows up as a RuntimeWarning on
#: import ("numpy.dtype size changed"), which is why tier 0 turns it into an error.
COMPILED = ("numpy", "scipy", "sklearn", "h5py", "numba", "pyarrow", "torch",
            "skmisc", "polars", "pydeseq2")

#: import name -> distribution name, where they differ.
IMPORT_TO_DIST = {
    "sklearn": "scikit-learn", "yaml": "PyYAML", "skmisc": "scikit-misc",
    "PIL": "pillow", "cv2": "opencv-python", "igraph": "igraph",
}

#: Modules that resolve from the standard library or the interpreter itself.
def _stdlib_names() -> set:
    names = set(getattr(sys, "stdlib_module_names", ()))
    names |= {"__future__", "__main__"}
    return names


STDLIB = _stdlib_names()


class Result:
    def __init__(self) -> None:
        self.failures: list[str] = []
        self.notes: list[str] = []

    def check(self, ok: bool, label: str, detail: str = "") -> None:
        print(("  ok   " if ok else "  FAIL ") + label + ((" — " + detail) if detail and not ok else ""))
        if not ok:
            self.failures.append(f"{label}: {detail}" if detail else label)

    def note(self, text: str) -> None:
        print("       " + text)
        self.notes.append(text)


# ===================================================================
# Tier 0 — environment integrity
# ===================================================================

def tier0(r: Result) -> None:
    print("\n[0] environment integrity")
    sp = Path(sysconfig.get_paths()["purelib"])

    # -- metadata index -------------------------------------------------------
    dist_versions: dict[str, list[str]] = {}
    for entry in os.listdir(sp):
        m = re.match(r"^([A-Za-z0-9_.\-]+?)-([0-9][^-]*)\.(dist-info|egg-info)$", entry)
        if m:
            key = m.group(1).lower().replace("_", "-")
            dist_versions.setdefault(key, []).append(m.group(2))

    # -- duplicate metadata (the doubled numpy/matplotlib case) ---------------
    dupes = {k: sorted(v) for k, v in dist_versions.items() if len(v) > 1}
    r.check(not dupes, "no package has two .dist-info directories", str(dupes))

    # -- stub packages (the scanpy case) --------------------------------------
    # An empty __init__.py is only suspicious when NO distribution claims the
    # directory. packages_distributions() maps top-level import names to the
    # distributions that own them, which is what makes namespace packages work:
    # `nvidia/` is legitimately empty and owned by nvidia-*-cu12, and matching on
    # the directory name alone reports it as a stub.
    try:
        owned = set(md.packages_distributions())
    except Exception:
        owned = set()
    stubs = []
    for entry in sorted(os.listdir(sp)):
        init = sp / entry / "__init__.py"
        if init.is_file() and init.stat().st_size == 0:
            if entry in owned:
                continue
            if entry.lower().replace("_", "-") not in dist_versions:
                stubs.append(entry)
    r.check(not stubs, "no 0-byte __init__.py without metadata (stub packages)", str(stubs))

    # -- mixed provenance ------------------------------------------------------
    # A conda-installed package records itself in conda-meta; a pip one does not.
    # The same name in both is the classic 'imports but misbehaves' setup.
    conda_meta = Path(sys.prefix) / "conda-meta"
    if conda_meta.is_dir():
        conda_pkgs = set()
        for f in conda_meta.glob("*.json"):
            name = f.stem.rsplit("-", 2)[0].lower().replace("_", "-")
            conda_pkgs.add(name)
        both = sorted((conda_pkgs & set(dist_versions)) - {"python", "pip", "setuptools", "wheel"})
        # Informational, not fatal: conda writes dist-info for its python packages
        # too, so overlap is expected unless the env is single-provenance by design.
        r.note("packages present in both conda-meta and site-packages metadata: "
               + (", ".join(both[:12]) + (" …" if len(both) > 12 else "") if both else "none"))

    # -- ABI: a numpy mismatch is a RuntimeWarning, so make it fatal ----------
    bad_abi = []
    for mod in COMPILED:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", RuntimeWarning)
                importlib.import_module(mod)
        except RuntimeWarning as e:
            bad_abi.append(f"{mod}: {e}")
        except ImportError:
            pass  # absence is tier 1/2's problem, not an ABI problem
        except Exception as e:
            bad_abi.append(f"{mod}: {type(e).__name__}: {e}")
    r.check(not bad_abi, "compiled extensions import with no ABI warning", "; ".join(bad_abi))

    # -- the one version the container contract depends on --------------------
    try:
        import anndata
        r.check(anndata.__version__ == "0.11.4",
                "anndata is exactly 0.11.4 (host reads back container output)",
                f"found {anndata.__version__}")
    except ImportError:
        r.check(False, "anndata importable", "not installed")


# ===================================================================
# Tier 1 — static import coverage over every host-side file
# ===================================================================

def _top_level_imports(path: Path) -> set:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
    except SyntaxError:
        return set()
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                out.add(a.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:          # skip relative imports
                out.add(node.module.split(".")[0])
    return out


def _is_local(name: str) -> bool:
    """A module provided by the repo itself rather than the environment."""
    if (REPO / name).is_dir() or (REPO / f"{name}.py").is_file():
        return True
    # test helpers and container wrappers imported by path-manipulating tests
    return name in {"conftest", "scgpt_wrapper", "gears_wrapper", "presage_wrapper",
                    "presage_datamodule", "presage_datamodule_csb", "model_harness",
                    "_utils", "train"}


def tier1(r: Result) -> None:
    print("\n[1] static import coverage (every host-side .py, no execution)")
    files, wanted = [], {}
    for tree in HOST_TREES:
        root = REPO / tree
        if not root.is_dir():
            continue
        for p in sorted(root.rglob("*.py")):
            if "__pycache__" in p.parts:
                continue
            files.append(p)
            for name in _top_level_imports(p):
                if name in STDLIB or _is_local(name):
                    continue
                wanted.setdefault(name, []).append(p.relative_to(REPO))

    missing = {}
    for name in sorted(wanted):
        try:
            found = importlib.util.find_spec(name) is not None
        except (ImportError, ValueError):
            found = False
        if not found:
            missing[name] = [str(x) for x in wanted[name][:3]]

    r.note(f"scanned {len(files)} files, {len(wanted)} distinct third-party imports")
    r.check(not missing, "every third-party import in every host file resolves",
            "; ".join(f"{k} (needed by {', '.join(v)})" for k, v in missing.items()))


# ===================================================================
# Tier 2 — live imports of the real modules and entry points
# ===================================================================

def tier2(r: Result) -> None:
    print("\n[2] live imports (modules actually execute)")
    sys.path.insert(0, str(REPO))
    targets: list[str] = []
    for tree in ("benchmark",):
        for p in sorted((REPO / tree).rglob("*.py")):
            if "__pycache__" in p.parts or p.name == "__init__.py":
                continue
            targets.append(".".join(p.relative_to(REPO).with_suffix("").parts))
    targets.append("data._utils")

    failed = []
    for mod in targets:
        try:
            importlib.import_module(mod)
        except Exception as e:                       # noqa: BLE001 — report, don't mask
            failed.append(f"{mod}: {type(e).__name__}: {e}")
    r.note(f"imported {len(targets)} benchmark/data modules")
    r.check(not failed, "every benchmark/ module imports", "; ".join(failed[:5]))

    # Entry-point scripts: all carry __main__ guards, so importing runs no work.
    scripts = sorted((REPO / "data").glob("*/get_data.py"))
    scripts += [REPO / "data" / "gen_dl_vocab.py"]
    scripts += [p for p in (REPO / "benchmark" / "scripts").glob("*.py")
                if p.name != "__init__.py"]
    failed = []
    for p in scripts:
        spec = importlib.util.spec_from_file_location(f"_probe_{p.stem}_{p.parent.name}", p)
        try:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        except Exception as e:                       # noqa: BLE001
            failed.append(f"{p.relative_to(REPO)}: {type(e).__name__}: {e}")
    r.note(f"imported {len(scripts)} entry-point scripts "
           f"({len(list((REPO / 'data').glob('*/get_data.py')))} get_data.py)")
    r.check(not failed, "every entry-point script imports", "; ".join(failed[:5]))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--static", action="store_true",
                    help="tiers 0-1 only; skip the live-import tier")
    args = ap.parse_args()

    print(f"interpreter : {sys.executable}")
    print(f"prefix      : {sys.prefix}")
    print(f"repo        : {REPO}")

    r = Result()
    tier0(r)
    tier1(r)
    if not args.static:
        tier2(r)

    print()
    if r.failures:
        print(f"FAILED ({len(r.failures)}):")
        for f in r.failures:
            print("  - " + f)
        return 1
    print("environment OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
