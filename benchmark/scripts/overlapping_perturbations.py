"""Common-set metrics + per-gene coverage/why for pooled benchmark results.

For each pooled results dir (``results/{ds}/{sc}/pooled/``) this
writes two derived artifacts into that same dir, idempotently:

1. ``metrics_on_overlapping_perturbations.csv`` — the pooled ``metrics.csv``
   row-filtered to the ``(bin_name, ko_name)`` perturbation-cells that are
   covered by EVERY predictor present (the intersection of per-predictor
   coverage). Same 9-column schema. ``_global_`` aggregate rows are excluded
   (they summarize each predictor's OWN full coverage set, which is inconsistent
   with the common subset — recompute any aggregate downstream from the
   per-perturbation rows in this file).

2. ``perturbations.json`` — every ``(bin_name, ko_name)`` cell in the universe
   (union of all predictors' coverage), annotated with ``handled_by`` and, for
   each predictor that does NOT handle it, ``not_handled_by[predictor] =
   {reason, explanation}``.

The unit is a **perturbation** — a ``(bin_name, ko_name)`` pair (a perturbation
evaluated in one cell_type), which is the unit a metric row is computed on. For
single-bin datasets this is just the perturbation (``bin_name`` is the sole bin
label, usually ``"all"``).

Coverage truth is the pooled CSV itself: a predictor "handles" a perturbation iff
it has at least one row there (``meta_metrics`` skips rows with
``n_genes_valid == 0``, so rows exist only for valid predictions).

Why-not reasons:
  * DL (GEARS/PRESAGE/scGPT): read the model's predicted set + modeled gene set
    (predictions ``var``) across ALL folds from MANIFEST.json. Target gene not in
    the model's modeled gene set -> ``gene_not_modeled``; gene IS modeled but not
    predicted as a perturbation -> ``pert_not_supported``; gene predicted but in a
    different fold than our split assigned -> ``fold_mismatch``.
  * Target-aware (BilinearRidge/LatentAdditive/TargetScaling/TargetZero):
    reproduce each predictor's own drop rule against the panel's gene_names
    -> ``target_not_in_panel`` / ``single_gene_only``.
  * Combos dropped by other predictors: ``combo_arity`` /
    ``combo_component_missing`` via store.ko_names.
  * Anything else: ``no_valid_prediction`` (honest catch-all).

Run (in the ``vcell`` conda env):
    python -m benchmark.scripts.overlapping_perturbations \
        [--dataset D ...] [--scenario S ...] [--results-root DIR]

With no filters it processes every pooled dir under RESULTS_DIR.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd

from benchmark.config import RESULTS_DIR
from benchmark.data_loader import DatasetStore
from benchmark.config import parse_target_genes
from benchmark.meta_metrics import overlap_perturbations

REASON_CODES = {
    "target_not_in_panel": "Target gene(s) are absent from this dataset's gene panel; "
                           "the target-aware predictor skips perturbations it cannot resolve.",
    "single_gene_only": "Predictor handles single-gene perturbations only; this label "
                        "is a combo/complex.",
    "combo_arity": "Predictor handles 2-way (A+B) combos only; this perturbation is not "
                   "a 2-way combo.",
    "combo_component_missing": "A single-gene component of this combo is not among the "
                               "dataset's available perturbations.",
    "no_valid_prediction": "Predictor produced no valid prediction for this "
                           "(cell_type, gene) (all-NaN row / n_genes_valid == 0); deeper "
                           "cause not auto-classified.",
}

_GLOBAL = "_global_"


# ---- cached, reused across scenario iterations ------------------------------
_STORE_CACHE: Dict[Tuple[str, str], Optional[DatasetStore]] = {}


def get_store(dataset: str) -> Optional[DatasetStore]:
    key = dataset
    if key not in _STORE_CACHE:
        try:
            _STORE_CACHE[key] = DatasetStore(dataset)
        except Exception as e:  # missing h5ad etc. — degrade gracefully
            print(f"  WARN: could not open store {dataset}: {e}")
            _STORE_CACHE[key] = None
    return _STORE_CACHE[key]


# ---- reason engine ----------------------------------------------------------
def explain(predictor: str, ko: str, dataset: str, scenario: str,
            ) -> Dict[str, str]:
    """Why `predictor` did not produce a valid prediction for `ko`."""
    # 1) The DROP DECISION is single-sourced in `verify.handles()` (the inverse
    #    of this function). Here we only FORMAT the human-readable reason for the
    #    code it returns, so the rule logic lives in exactly one place (core),
    #    not duplicated in this script.
    store = get_store(dataset)
    if store is not None:
        from benchmark.verify import handles
        ok, code = handles(predictor, ko, store, scenario)
        if not ok and code:
            return {"reason": code,
                    "explanation": _reason_text(code, predictor, ko, store)}

    # 2) handles() says it should resolve (or the store is unavailable) yet the
    #    perturbation is absent from coverage → honest catch-all.
    return {"reason": "no_valid_prediction",
            "explanation": f"{predictor} produced no valid prediction for '{ko}' "
                           f"(all-NaN row / n_genes_valid == 0)."}


def _reason_text(code: str, predictor: str, ko: str, store) -> str:
    """Human-readable explanation for a drop reason-code from `verify.handles()`.
    Formatting only — the decision of WHICH code applies lives in handles()."""
    tokens = parse_target_genes(ko)
    if code == "single_gene_only":
        return (f"{predictor} handles single-gene perturbations only; "
                f"'{ko}' is a combo/complex.")
    if code == "target_not_in_panel":
        miss = [t for t in tokens if t not in set(store.gene_names)]
        return (f"Target gene(s) {miss or tokens} of '{ko}' are absent from the "
                f"gene panel; {predictor} skips perturbations it cannot resolve.")
    if code == "combo_arity":
        parts = sorted(ko.split("+"))
        return (f"'{ko}' is not a 2-way A+B combo ({len(parts)} component(s)); "
                f"{predictor} handles 2-way combos only.")
    if code == "combo_component_missing":
        parts = sorted(ko.split("+"))
        miss = [p for p in parts if p not in set(store.ko_names)]
        return (f"Combo component(s) {miss} of '{ko}' are not among this dataset's "
                f"perturbations, so {predictor} cannot build the additive prediction.")
    return REASON_CODES.get(code, code)


# ---- per pooled dir ---------------------------------------------------------
def _restrict_meta_to_overlap(
    pooled_dir: Path, overlap: Set[Tuple[str, str]], src_name: str, out_name: str,
) -> None:
    """Row-filter a per-pert BS/CF table (``bs.csv`` / ``cf.csv``) to the SAME
    ``(bin_name, ko_name)`` overlap set used for the metrics file, and write it as
    ``{bs,cf}_on_overlapping_perturbations.csv`` in the same pooled dir.

    Keeps the source schema verbatim (the figure scripts read these columns).
    No-op if the source file is absent (e.g. BS/CF not yet computed for this
    dataset). ``_global_`` rows are naturally excluded — the overlap set,
    built from per-perturbation coverage, never contains them.
    """
    src = pooled_dir / src_name
    if not src.exists():
        return
    df = pd.read_csv(src)
    if "bin_name" not in df.columns or "ko_name" not in df.columns:
        print(f"  WARN: {src_name} lacks bin_name/ko_name; skipping {out_name}")
        return
    pert = pd.Series(
        list(zip(df["bin_name"].astype(str), df["ko_name"].astype(str))),
        index=df.index,
    )
    keep = df[pert.isin(overlap).values]
    out = pooled_dir / out_name
    keep.to_csv(out, index=False)
    print(f"  -> {out_name} ({len(keep)} rows from {src_name})")


def process_pooled(metrics_csv: Path, dataset: str, scenario: str) -> None:
    print(f"== {dataset} / {scenario} ==")
    df = pd.read_csv(metrics_csv)
    real = df[df["ko_name"] != _GLOBAL].copy()
    if real.empty:
        print("  (no per-perturbation rows; skipping)")
        return

    predictors = sorted(real["predictor"].unique())
    real["_pert"] = list(zip(real["bin_name"].astype(str), real["ko_name"].astype(str)))
    covered: Dict[str, Set[Tuple[str, str]]] = {
        p: set(real.loc[real["predictor"] == p, "_pert"]) for p in predictors
    }
    universe: Set[Tuple[str, str]] = set().union(*covered.values())
    overlap: Set[Tuple[str, str]] = overlap_perturbations(covered)

    # our-split fold per perturbation (each is in exactly one test fold)
    fold_of: Dict[Tuple[str, str], int] = (
        real.drop_duplicates("_pert").set_index("_pert")["fold"].to_dict()
    )

    # --- 1) overlapping-perturbations metrics CSV ---
    out_csv = metrics_csv.parent / "metrics_on_overlapping_perturbations.csv"
    keep = real[real["_pert"].isin(overlap)].drop(columns="_pert")
    keep.to_csv(out_csv, index=False)
    print(f"  predictors={len(predictors)}  universe={len(universe)}  "
          f"overlap={len(overlap)}  -> {out_csv.name} ({len(keep)} rows)")
    if len(universe) and not overlap:
        # name the predictor(s) that shrank it most
        worst = sorted(predictors, key=lambda p: len(covered[p]))[:3]
        print("  WARN: empty overlap; smallest-coverage predictors: "
              + ", ".join(f"{p}({len(covered[p])})" for p in worst))

    # --- 1b) overlapping-perturbations BS / CF CSVs (same overlap set) ---
    # The figure scripts (plotting/plot_bs_vs_edist, plot_cf_vs_bs[_binned]) read
    # these; restrict the raw per-pert bs.csv / cf.csv written by run_pipeline.
    _restrict_meta_to_overlap(metrics_csv.parent, overlap, "bs.csv",
                              "bs_on_overlapping_perturbations.csv")
    _restrict_meta_to_overlap(metrics_csv.parent, overlap, "cf.csv",
                              "cf_on_overlapping_perturbations.csv")

    # --- 2) per-gene coverage / why JSON ---
    perts = []
    for bin_name, ko in sorted(universe):
        handled = [p for p in predictors if (bin_name, ko) in covered[p]]
        not_handled = {
            p: explain(p, ko, dataset, scenario)
            for p in predictors if (bin_name, ko) not in covered[p]
        }
        entry = {
            "ko_name": ko,
            "bin_name": bin_name,
            "fold": int(fold_of[(bin_name, ko)]),
            "in_overlap": (bin_name, ko) in overlap,
            "handled_by": handled,
        }
        if not_handled:
            entry["not_handled_by"] = not_handled
        perts.append(entry)

    coverage_summary = {
        p: {"handled": len(covered[p]),
            "not_handled": len(universe) - len(covered[p]),
            "frac": round(len(covered[p]) / len(universe), 4) if universe else None}
        for p in predictors
    }
    payload = {
        "dataset": dataset,
        "scenario": scenario,
        "perturbation_unit": "(bin_name, ko_name) — a perturbation evaluated in one cell_type",
        "predictors": predictors,
        "n_perturbations_total": len(universe),
        "n_overlapping": len(overlap),
        "note": "metrics_on_overlapping_perturbations.csv excludes _global_ rows; "
                "recompute aggregates from its per-perturbation rows.",
        "reason_codes": REASON_CODES,
        "coverage_summary": coverage_summary,
        "perturbations": perts,
    }
    out_json = metrics_csv.parent / "perturbations.json"
    with open(out_json, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"  -> {out_json.name} ({len(perts)} perturbations, "
          f"{sum(1 for e in perts if 'not_handled_by' in e)} with drops)")


def discover(results_root: Path, datasets, scenarios):
    """Yield (metrics_csv, dataset, scenario) for matching pooled dirs."""
    for mcsv in sorted(results_root.glob("*/*/pooled/metrics.csv")):
        rel = mcsv.relative_to(results_root).parts  # ds, scenario, pooled, metrics.csv
        if len(rel) != 4:
            continue
        ds, sc = rel[0], rel[1]
        if datasets and ds not in datasets:
            continue
        if scenarios and sc not in scenarios:
            continue
        yield mcsv, ds, sc


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", action="append", default=[],
                    help="restrict to dataset (repeatable; default all)")
    ap.add_argument("--scenario", action="append", default=[],
                    help="restrict to scenario (repeatable; default all)")
    ap.add_argument("--results-root", default=str(RESULTS_DIR),
                    help="results root (default benchmark.config.RESULTS_DIR)")
    args = ap.parse_args()

    root = Path(args.results_root)
    targets = list(discover(root, set(args.dataset), set(args.scenario)))
    if not targets:
        print(f"No matching pooled dirs under {root} for the given filters.")
        return
    print(f"Processing {len(targets)} pooled dir(s) under {root}\n")
    for mcsv, ds, sc in targets:
        try:
            process_pooled(mcsv, ds, sc)
        except Exception as e:
            print(f"  ERROR processing {mcsv}: {e}")
    print("\nDone.")


if __name__ == "__main__":
    main()
