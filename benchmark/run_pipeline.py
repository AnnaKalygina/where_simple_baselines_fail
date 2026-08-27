"""Unified CLI for the benchmark.

Subcommands:
  fit       --dataset … --scenario … --fold … --predictor …  (train; save weights)
  predict   --dataset … --scenario … --fold … --predictor …  (run; save predictions)
  metrics   --dataset … --scenario … --fold … --predictor … --metric …
  bs        --dataset … --scenario … --fold …                (baseline saturation)
  cf        --dataset … --scenario … --fold … --predictor …  (captured fraction)
  all       --dataset … --scenario …                          (fit + predict + metrics)
  list      print the resolved job list and exit

Every selector accepts: ``X`` (single), ``X,Y,Z`` (multi), ``0-4`` (range
for --fold), ``analytical|learned|controls|dl`` (category for --predictor),
``main_benchmark`` (group for --metric), or ``all``.

The smallest unit of work is one (dataset, scenario, fold, predictor, metric)
cell — every output is keyed on this tuple and re-runs are idempotent.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import logging
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from benchmark.config import (
    DATASET_CONFIG, RESULTS_DIR,
    checkpoint_dir, enumerate_jobs, existing_predictions_path, predictions_path,
    predictor_dir, results_fold_dir, results_pooled_dir,
)
from benchmark.data_loader import DatasetStore
from benchmark.meta_metrics import (
    METRICS_CONFIG, REAL_DATA_METRICS_CONFIG,
    bs_per_pert, cf_per_pert, compute_per_pert_metrics,
    filter_drf, filter_pos_ge_neg, filter_min_genes,
    main_benchmark_metrics, summarize_meta_metric,
    summarize_metric_distributions,
)
from benchmark.predictors.base import (
    PREDICTOR_REGISTRY, _ensure_predictors_loaded, needs_training,
    predictor_category)
from benchmark.predictors.learned import LearnedPredictor
from benchmark.predictors._shared import FINGERPRINT_FILE
from benchmark.predictors.trained import TrainedPredictor

log = logging.getLogger(__name__)


# ===================================================================
# Safe CSV writes (atomic + lock-protected read-modify-write)
# ===================================================================


def _atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    """Write a CSV via a unique temp file + os.replace (atomic rename), so a
    reader/concurrent writer never observes a half-written file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


@contextmanager
def _file_lock(path: Path) -> Iterator[None]:
    """Exclusive cross-process lock on a sibling `.lock` file. Serializes the
    read-modify-write of a shared CSV so parallel per-predictor jobs on the same
    (dataset, scenario, fold) can't lose each other's rows."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.with_name(path.name + ".lock")
    fd = os.open(str(lock), os.O_CREAT | os.O_RDWR, 0o664)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def predictor_categories() -> Dict[str, List[str]]:
    """category -> sorted predictor names, derived from the registry.

    Replaces a hand-maintained table that had already drifted: it listed the
    adopted-DL adapters under "dl" but not GEARS-ct, so `--predictor dl`
    silently skipped the container model.
    """
    _ensure_predictors_loaded()
    cats: Dict[str, List[str]] = {}
    for name, cls in PREDICTOR_REGISTRY.items():
        cats.setdefault(predictor_category(cls), []).append(name)
    return {k: sorted(v) for k, v in cats.items()}


# ===================================================================
# Selector resolution
# ===================================================================


# Universe of scenario names across all datasets — used to tell a typo (raise)
# apart from a valid scenario that just doesn't apply to this dataset (filter).
_ALL_SCENARIOS = sorted({sc for d in DATASET_CONFIG.values() for sc in d["scenarios"]})


def _resolve_datasets(spec: str) -> List[str]:
    if spec == "all":
        return list(DATASET_CONFIG.keys())
    requested = [s.strip() for s in spec.split(",") if s.strip()]
    unknown = [d for d in requested if d not in DATASET_CONFIG]
    if unknown:
        raise ValueError(f"Unknown dataset(s): {unknown}. Known: {sorted(DATASET_CONFIG)}")
    return requested


def _resolve_scenarios(spec: str, dataset: str) -> List[str]:
    ds_scenarios = list(DATASET_CONFIG[dataset]["scenarios"].keys())
    if spec == "all":
        return ds_scenarios
    requested = [s.strip() for s in spec.split(",") if s.strip()]
    unknown = [s for s in requested if s not in _ALL_SCENARIOS]
    if unknown:
        raise ValueError(f"Unknown scenario(s): {unknown}. Known: {_ALL_SCENARIOS}")
    # Known scenarios that simply don't apply to this dataset are filtered (this
    # is what lets `--dataset all --scenario UnseenCombo` work cleanly).
    return [s for s in requested if s in ds_scenarios]


def _resolve_folds(spec: str, dataset: str, scenario: str) -> List[int]:
    n = DATASET_CONFIG[dataset]["scenarios"][scenario]["n_folds"]
    if spec == "all":
        return list(range(n))
    out: List[int] = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            if "-" in token:
                a, b = token.split("-", 1)
                out.extend(range(int(a), int(b) + 1))
            else:
                out.append(int(token))
        except ValueError:
            raise ValueError(
                f"Invalid fold selector {token!r} (expected an int or an 'A-B' range)")
    # Out-of-range folds are filtered (not an error) so a single --fold spec can
    # span datasets/scenarios with different fold counts.
    return [f for f in sorted(set(out)) if 0 <= f < n]


def predictor_groups() -> Dict[str, List[str]]:
    """Named selector groups for `--predictor`, alongside the derived categories.

    A category is DERIVED (one per class, from the tier — see
    `base.predictor_category`). A group is DECLARED: a set someone chose, which
    the class hierarchy has no way to know. Both are legitimate; they answer
    different questions, so they are separate lookups rather than one table.

    `transformers` is the KxN sweep. Its membership is sourced from
    `_torch.recipe.MODELS`, where it is already stated as policy — "Keep the
    whole set, or the comparison it encodes is lost" — so the group cannot drift
    from the experiment it names. That module imports no torch, so reading it
    here is safe in every env that loads the registry.
    """
    _ensure_predictors_loaded()
    from benchmark.predictors import transformers
    return {"transformers": list(transformers.__all__)}


def _resolve_predictors(spec: str) -> List[str]:
    _ensure_predictors_loaded()
    if spec == "all":
        return sorted(PREDICTOR_REGISTRY.keys())
    categories = predictor_categories()
    groups = predictor_groups()
    out: Set[str] = set()
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if token in categories:
            out.update(categories[token])
        elif token in groups:
            out.update(groups[token])
        elif token in PREDICTOR_REGISTRY:
            out.add(token)
        else:
            # Deliberately NOT a prefix match: `--predictor transformer` (missing
            # underscore) silently becoming nine GPU training runs is worse than
            # a warning that names the typo.
            log.warning("Unknown predictor: %s", token)
    return sorted(out)


def _resolve_metrics(spec: str, use_real_data: bool = False) -> List:
    config = REAL_DATA_METRICS_CONFIG if use_real_data else METRICS_CONFIG
    if spec == "all":
        return list(config)
    if spec == "main_benchmark":
        return main_benchmark_metrics(config)
    by_name = {m.name: m for m in config}
    requested = [t.strip() for t in spec.split(",") if t.strip()]
    unknown = [t for t in requested if t not in by_name]
    if unknown:
        raise ValueError(f"Unknown metric(s): {unknown}. Known: {sorted(by_name)}")
    return [by_name[t] for t in requested]


def _resolve_job_tuples(
    args,
) -> List[Tuple[str, str, int]]:
    jobs: List[Tuple[str, str, int]] = []
    for ds in _resolve_datasets(args.dataset):
        for sc in _resolve_scenarios(args.scenario, ds):
            for fold in _resolve_folds(args.fold, ds, sc):
                jobs.append((ds, sc, fold))
    return jobs


# ===================================================================
# Subcommand implementations
# ===================================================================


def cmd_roster(args) -> None:
    """Print the predictor roster — the single derived answer to "what exists".

    Everything here is read off the classes, so it cannot drift from the code
    the way the old hand-kept category table did. `--json` for tooling.

    WHY THERE IS NO COMMITTED `ROSTER.json` (considered, rejected). Writing this
    output to a tracked file and having `verify` regenerate-and-diff it was
    proposed as the way to give non-Python consumers a stable list. But a
    committed copy of a live derivation is precisely the thing that drifts — the
    differ would exist only to police the copy — and every consumer that wanted
    it already imports `benchmark`, so it can read `PREDICTOR_REGISTRY` or shell
    out to `roster --json` and need neither the file nor the differ.
    """
    _ensure_predictors_loaded()
    rows = []
    for name in sorted(PREDICTOR_REGISTRY):
        cls = PREDICTOR_REGISTRY[name]
        rows.append({
            "name": name,
            "category": predictor_category(cls),
            "module": cls.__module__.rsplit(".", 1)[-1],
            "needs_training": needs_training(cls),
            "has_drop_rule": bool(cls.has_drop_rule),
            "scenarios": list(cls.scenarios),
            **({"cell_aware": bool(cls.cell_aware)}
               if hasattr(cls, "cell_aware") else {}),
        })
    if getattr(args, "json", False):
        print(json.dumps(rows, indent=2))
        return
    w = max(len(r["name"]) for r in rows)
    print(f"{'predictor'.ljust(w)}  {'category':12s} {'trains':6s} {'drop':4s} scenarios")
    for r in rows:
        print(f"{r['name'].ljust(w)}  {r['category']:12s} "
              f"{'yes' if r['needs_training'] else '-':6s} "
              f"{'yes' if r['has_drop_rule'] else '-':4s} {','.join(r['scenarios'])}")
    print(f"\n{len(rows)} predictors")
    for cat, names in sorted(predictor_categories().items()):
        print(f"  {cat:12s} {len(names)}")


def cmd_list(args) -> None:
    jobs = _resolve_job_tuples(args)
    predictors = _resolve_predictors(getattr(args, "predictor", "all"))
    for ds, sc, fold in jobs:
        for p in predictors:
            cls = PREDICTOR_REGISTRY.get(p)
            if cls is None or sc not in cls.scenarios:
                continue
            print(f"{ds} {sc} fold{fold} {p}")


def _maybe_preflight(args) -> None:
    """Cheap pre-benchmark gate (verify L0+L1) before training. Aborts on a
    failed invariant / leakage check so a broken dataset can't silently feed a
    long run. Bypass with --no-verify."""
    if getattr(args, "no_verify", False):
        return
    from benchmark.verify import preflight
    datasets = _resolve_datasets(args.dataset)
    log.info("Preflight verification (L0 invariants + L1 split leakage) on %s ...", datasets)
    r = preflight(datasets)
    if r.failed:
        raise SystemExit(
            f"\nPreflight FAILED ({r.failed} check(s)): {r.fail_names}\n"
            f"Fix the dataset(s), or re-run with --no-verify to bypass the gate.")


def cmd_verify(args) -> None:
    """Run the full verifier (L4 contracts + per-dataset L1/L2[/L3/L6] + L5
    coverage) over the selected datasets; exit non-zero on any failure."""
    from benchmark import verify as V
    r = V.Results()
    print("[L4] PREDICTOR-CONTRACTS")
    V.verify_predictor_contracts(r)
    print()
    full = getattr(args, "full", False)
    datasets = _resolve_datasets(args.dataset)
    sc = None if args.scenario == "all" else args.scenario
    for ds in datasets:
        print(f"[DATA] {ds}")
        V.verify_dataset(ds, r, full=full, scenario=sc)
        print()
    V.run_pseudobulk_gate_once(datasets, r)
    code = V._summary(r)
    if code != 0:
        raise SystemExit(code)


def cmd_fit(args) -> None:
    _maybe_preflight(args)
    predictors = _resolve_predictors(args.predictor)
    for ds, sc, fold in _resolve_job_tuples(args):
        store = DatasetStore(ds)
        for p in predictors:
            cls = PREDICTOR_REGISTRY[p]
            if sc not in cls.scenarios:
                log.info("skip %s/%s/fold%d %s (not in scenarios)", ds, sc, fold, p)
                continue
            model = cls()
            if isinstance(model, TrainedPredictor) and getattr(args, "seed", None):
                model.seed = args.seed
                # Never wipe another seed's run: `fit` clears the run dir, and a
                # seed mismatch reads as ordinary staleness, which retrains
                # without --force. Neither run supersedes the other, so refuse.
                other = _foreign_seed(ds, p, sc, fold, model.seed)
                if other is not None:
                    log.error("FIT %s/%s/fold%d %s: run dir holds seed %d, "
                              "refusing to overwrite it with seed %d — point "
                              "VCELL_ROOT at a different tree",
                              ds, sc, fold, p, other, model.seed)
                    continue
            if (needs_training(cls) and not getattr(args, "force", False)
                    and model.is_trained(store, sc, fold)):
                log.info("FIT %s/%s/fold%d %s: trained state already present — "
                         "skipping (--force to redo)", ds, sc, fold, p)
                continue
            log.info("FIT %s/%s/fold%d %s", ds, sc, fold, p)
            try:
                # Only the expensive tier has anything to force: it is what holds
                # a run claim and wipes a run dir before retraining.
                if isinstance(model, TrainedPredictor):
                    model.fit(store, sc, fold, force=getattr(args, "force", False),
                              steal=getattr(args, "steal_claim", False))
                else:
                    model.fit(store, sc, fold)
            except FileNotFoundError as e:
                # DL adapters raise this when MANIFEST lacks an entry for this
                # (dataset, scenario, fold). Soft-skip rather than abort.
                log.warning("FIT %s/%s/fold%d %s: skipping — %s",
                            ds, sc, fold, p, e)
                continue
            # Each tier owns its own persistence: the learned tier writes a
            # weights.npz here, container predictors already wrote their run dir
            # inside fit(), and analytical baselines have nothing to persist.
            if isinstance(model, LearnedPredictor):
                log.info("FIT %s/%s/fold%d %s -> %s", ds, sc, fold, p,
                         model.persist(store, sc, fold))


def cmd_predict(args) -> None:
    predictors = _resolve_predictors(args.predictor)
    for ds, sc, fold in _resolve_job_tuples(args):
        store = DatasetStore(ds)
        for p in predictors:
            cls = PREDICTOR_REGISTRY[p]
            if sc not in cls.scenarios:
                continue
            ppath = predictions_path(ds, p, sc, fold)
            log.info("PREDICT %s/%s/fold%d %s -> %s", ds, sc, fold, p, ppath)
            # predict NEVER trains. A predictor that has not been fitted for this
            # fold is skipped with a clear message; previously a stale/empty
            # weights.npz made this branch either retrain silently or crash.
            model = cls()
            if isinstance(model, TrainedPredictor) and getattr(args, "seed", None):
                model.seed = args.seed
            if needs_training(cls) and not model.is_trained(store, sc, fold):
                log.warning("PREDICT %s/%s/fold%d %s: not trained — run `fit` "
                            "(or `all`) first; skipping", ds, sc, fold, p)
                continue
            if isinstance(model, LearnedPredictor):
                model = cls.restore(store, sc, fold)
            try:
                preds = model.predict(store, sc, fold)
            except FileNotFoundError as e:
                log.warning("PREDICT %s/%s/fold%d %s: skipping — %s",
                            ds, sc, fold, p, e)
                continue
            split = store.split(sc, fold)
            tbi = (split.test_bin_indices
                   if split.test_bin_indices is not None and len(split.test_bin_indices) > 0
                   else np.arange(store.n_bins, dtype=np.int64))
            tki = split.test_ko_indices
            model.save_predictions(
                ppath, preds,
                gene_names=store.gene_names,
                ko_names=[store.ko_names[int(i)] for i in tki],
                bin_names=[store.bin_names[int(i)] for i in tbi],
                test_ko_indices=tki,
                test_bin_indices=tbi,
                dataset=ds, scenario=sc, fold=fold,
            )


def cmd_metrics(args) -> None:
    """Compute metrics and write ONE CSV per (dataset, scenario, fold).

    Output path: ``results/{dataset}/{scenario}/fold{N}/metrics.csv``.
    Each file is the source of truth for a single fold's metrics. Re-running
    a narrower selector updates only the matching (predictor, ko, bin, metric)
    rows within that fold's file; other folds are untouched.

    Use `pool` to derive a single pooled CSV across folds.
    """
    predictors = _resolve_predictors(args.predictor)
    metric_specs = _resolve_metrics(args.metric)
    # Group results by (dataset, scenario, fold) so each fold is written
    # independently — supports per-cell idempotency.
    per_cell: Dict[Tuple[str, str, int], List[pd.DataFrame]] = {}
    for ds, sc, fold in _resolve_job_tuples(args):
        store = DatasetStore(ds)
        for p in predictors:
            cls = PREDICTOR_REGISTRY[p]
            if sc not in cls.scenarios:
                continue
            # Panel path, or the canonical union path for panel-independent
            # DL/external predictions (single source of truth in config).
            if existing_predictions_path(ds, p, sc, fold) is None:
                log.warning("skip metrics %s/%s/fold%d %s: predictions missing",
                            ds, sc, fold, p)
                continue
            log.info("METRICS %s/%s/fold%d %s", ds, sc, fold, p)
            try:
                df = compute_per_pert_metrics(
                    ds, sc, fold, p, metric_specs, store=store,
                    missing_genes=getattr(args, "missing_genes",
                                          "fill_missing_genes_with_zero"))
            except Exception as e:
                log.error("  failed: %s", e)
                continue
            per_cell.setdefault((ds, sc, fold), []).append(df)

    if not per_cell:
        log.info("No metric rows produced.")
        return

    for (ds, sc, fold), parts in per_cell.items():
        sub = pd.concat(parts, ignore_index=True)
        odir = results_fold_dir(ds, sc, fold)
        odir.mkdir(parents=True, exist_ok=True)
        opath = odir / "metrics.csv"
        # Lock the read-modify-write: parallel per-predictor jobs on the same fold
        # otherwise both read the old file and the second write drops the first's
        # rows. The lock serializes them; the write itself is atomic (tmp+rename).
        with _file_lock(opath):
            if opath.exists():
                existing = pd.read_csv(opath)
                # Replace ALL rows for every predictor recomputed in this run, not
                # just the matching (predictor, ko, bin, metric) keys. Otherwise a
                # predictor whose KO-set SHRANK (e.g. target-aware predictors now
                # skip unresolvable KOs from eval) would keep stale rows for the
                # KOs it no longer scores. Predictors NOT in this run (e.g. DL
                # baselines) are preserved untouched.
                recomputed = set(sub["predictor"].unique())
                existing = existing[~existing["predictor"].isin(recomputed)]
                merged = pd.concat([existing, sub], ignore_index=True)
            else:
                merged = sub
            # Scenario-validity guard: drop rows for any predictor whose `.scenarios`
            # excludes this scenario. The scoring loop above already skips such
            # predictors, but a stale npz/row from an older or broader run would
            # otherwise survive the merge (e.g. Mean-over-perturbations / Two-way-mean
            # left over under UnseenBoth/UnseenCell). Unknown predictor names are kept
            # defensively.
            keep = {p for p in merged["predictor"].unique()
                    if PREDICTOR_REGISTRY.get(p) is None or sc in PREDICTOR_REGISTRY[p].scenarios}
            if len(keep) != merged["predictor"].nunique():
                dropped = sorted(set(merged["predictor"].unique()) - keep)
                log.info("  dropping %d stale row-set(s) invalid for %s: %s",
                         len(dropped), sc, dropped)
                merged = merged[merged["predictor"].isin(keep)].reset_index(drop=True)
            _atomic_write_csv(merged, opath)
        log.info("Wrote %s (%d rows)", opath, len(merged))


def cmd_pool(args) -> None:
    """Concatenate per-fold metrics CSVs into one pooled file per (dataset, scenario).

    Reads ``results/{ds}/{sc}/fold{N}/metrics.csv`` for every fold of every
    selected (dataset, scenario) and writes the union to
    ``results/{ds}/{sc}/pooled/metrics.csv``.

    This is a pure aggregation step: no metric is recomputed.
    """
    grouped: Dict[Tuple[str, str], List[pd.DataFrame]] = {}
    for ds, sc, fold in _resolve_job_tuples(args):
        fpath = results_fold_dir(ds, sc, fold) / "metrics.csv"
        if not fpath.exists():
            log.warning("pool: missing %s", fpath)
            continue
        df = pd.read_csv(fpath)
        # Ensure the fold column matches the directory name (defensive).
        if "fold" not in df.columns:
            df["fold"] = fold
        grouped.setdefault((ds, sc), []).append(df)

    if not grouped:
        log.info("No per-fold CSVs to pool.")
        return

    for (ds, sc), parts in grouped.items():
        merged = pd.concat(parts, ignore_index=True)
        odir = results_pooled_dir(ds, sc)
        odir.mkdir(parents=True, exist_ok=True)
        opath = odir / "metrics.csv"
        _atomic_write_csv(merged, opath)
        log.info("Pooled %s (%d rows from %d folds)", opath, len(merged), len(parts))


def _apply_filters(df: pd.DataFrame, args) -> pd.DataFrame:
    """Apply the optional --filter-* flags shared by cmd_bs and cmd_cf."""
    if args.filter_drf:
        df = filter_drf(df, args.filter_drf_threshold)
    if args.filter_pos_ge_neg:
        df = filter_pos_ge_neg(df)
    if args.filter_min_genes:
        df = filter_min_genes(df, args.filter_min_genes)
    return df


def _resolve_baselines(args) -> List[Optional[str]]:
    """Parse `--baseline-predictor` into a list of registered predictor names.

    Accepts a single name, a comma-list (so one `bs`/`cf` run can compare
    several baselines without overwriting the output file), or empty/None for
    the scenario default (returned as a single-element [None]).
    """
    spec = getattr(args, "baseline_predictor", None)
    if not spec:
        return [None]
    _ensure_predictors_loaded()      # register before validating names
    out: List[Optional[str]] = []
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if tok == "auto_hardest" or tok in PREDICTOR_REGISTRY:
            out.append(tok)   # 'auto_hardest' resolved per (dataset, scenario) in bs/cf_per_pert
        else:
            log.warning("Unknown baseline predictor: %s", tok)
    return out or [None]


def _drf_threshold(args) -> float:
    return float(getattr(args, "filter_drf_threshold", 0.2) or 0.2)


def cmd_bs(args) -> None:
    metric_specs = _resolve_metrics(args.metric)
    baselines = _resolve_baselines(args)
    all_rows: List[pd.DataFrame] = []
    for ds, sc, fold in _resolve_job_tuples(args):
        store = DatasetStore(ds)
        for base in baselines:
            log.info("BS %s/%s/fold%d (baseline=%s)", ds, sc, fold,
                     base or "<scenario default>")
            df = bs_per_pert(ds, sc, fold, baseline_predictor=base,
                             metric_specs=metric_specs, store=store,
                             missing_genes=getattr(args, "missing_genes",
                                                   "fill_missing_genes_with_zero"))
            all_rows.append(_apply_filters(df, args))

    if not all_rows:
        return
    out = pd.concat(all_rows, ignore_index=True)
    for (ds, sc), sub in out.groupby(["dataset", "scenario"]):
        odir = results_pooled_dir(ds, sc)
        odir.mkdir(parents=True, exist_ok=True)
        # Raw per-pert rows (cell-type-resolved via bin_name), all baselines.
        _atomic_write_csv(sub, odir / "bs.csv")
        log.info("Wrote %s (%d rows)", odir / "bs.csv", len(sub))
        # Per-cell-type headline summary (clean-filtered: pos>=neg & DRF>thr).
        summary = summarize_meta_metric(
            sub, "bs",
            ["dataset", "scenario", "baseline_predictor", "metric"],
            clean=True, drf_threshold=_drf_threshold(args),
        )
        _atomic_write_csv(summary, odir / "bs_summary.csv")
        log.info("Wrote %s (%d rows)", odir / "bs_summary.csv", len(summary))


def cmd_cf(args) -> None:
    predictors = _resolve_predictors(args.predictor)
    metric_specs = _resolve_metrics(args.metric)
    baselines = _resolve_baselines(args)
    all_rows: List[pd.DataFrame] = []
    for ds, sc, fold in _resolve_job_tuples(args):
        store = DatasetStore(ds)
        for p in predictors:
            cls = PREDICTOR_REGISTRY[p]
            if sc not in cls.scenarios:
                continue
            # A predictor need not have been run on every fold. Skip — don't
            # crash — exactly as cmd_metrics does; cf for that predictor exists
            # only on the folds it actually predicts.
            if existing_predictions_path(ds, p, sc, fold) is None:
                log.warning("skip cf %s/%s/fold%d %s: predictions missing",
                            ds, sc, fold, p)
                continue
            for base in baselines:
                log.info("CF %s/%s/fold%d %s (baseline=%s)", ds, sc, fold, p,
                         base or "<scenario default>")
                df = cf_per_pert(ds, sc, fold, p, baseline_predictor=base,
                                 metric_specs=metric_specs, store=store,
                                 missing_genes=getattr(args, "missing_genes",
                                                       "fill_missing_genes_with_zero"))
                all_rows.append(_apply_filters(df, args))

    if not all_rows:
        return
    out = pd.concat(all_rows, ignore_index=True)
    for (ds, sc), sub in out.groupby(["dataset", "scenario"]):
        odir = results_pooled_dir(ds, sc)
        odir.mkdir(parents=True, exist_ok=True)
        # Raw per-pert rows (cell-type-resolved), all (predictor, baseline) pairs.
        _atomic_write_csv(sub, odir / "cf.csv")
        log.info("Wrote %s (%d rows)", odir / "cf.csv", len(sub))
        # Per-cell-type headline summary (clean-filtered: pos>=neg & DRF>thr).
        summary = summarize_meta_metric(
            sub, "cf",
            ["dataset", "scenario", "predictor", "baseline_predictor", "metric"],
            clean=True, drf_threshold=_drf_threshold(args),
        )
        _atomic_write_csv(summary, odir / "cf_summary.csv")
        log.info("Wrote %s (%d rows)", odir / "cf_summary.csv", len(summary))


def cmd_summary(args) -> None:
    """Write ``pooled/metrics_summary.csv`` per (dataset, scenario).

    One distribution row per (predictor, metric, missing_genes, pert_set, view):
    mean/median/std across perturbations + n_perturbations / n_genes /
    n_genes_native. BOTH missing-gene modes and BOTH perturbation sets
    (full + overlapping) are emitted, so ``--missing-genes`` is ignored here.
    Predictors/folds are taken from the pooled ``metrics.csv`` (run ``pool``
    first), like the overlapping-perturbations script.
    """
    metric_specs = _resolve_metrics(args.metric)
    seen: set = set()
    for ds, sc, _fold in _resolve_job_tuples(args):
        if (ds, sc) in seen:
            continue
        seen.add((ds, sc))
        store = DatasetStore(ds)
        summ = summarize_metric_distributions(
            ds, sc, store=store, metric_specs=metric_specs)
        if summ.empty:
            log.warning("summary: no rows for %s/%s (pooled metrics.csv present?)", ds, sc)
            continue
        odir = results_pooled_dir(ds, sc)
        odir.mkdir(parents=True, exist_ok=True)
        opath = odir / "metrics_summary.csv"
        _atomic_write_csv(summ, opath)
        log.info("Wrote %s (%d rows)", opath, len(summ))


def _require_seed_isolation(args) -> None:
    """A non-default seed must run in a tree of its own. Refuses otherwise.

    Everything under `models/{ds}/{pred}/{sc}/fold{N}/` is seed-INDEPENDENT: the
    checkpoint dir, `predictions.npz`, and the `metrics.csv` rows are all keyed on
    the predictor NAME. So a second seed run against the default tree does not sit
    beside the first — `fit` wipes the run dir (`_clean_run_dir`), then `predict`
    overwrites `predictions.npz`, then `metrics` overwrites the scored rows. A
    stale fold retrains WITHOUT `--force`, so nothing else stops it either.

    The alternative — a seed segment in `checkpoint_dir` — is a breaking change to
    every existing checkpoint path and a decision belonging to whoever owns the
    trained-tier layout. This flag is for one-off seed-stability runs, so it asks
    for isolation instead of inventing a second way to address a run.
    """
    if getattr(args, "seed", None) is None:
        return
    if not os.environ.get("VCELL_ROOT"):
        sys.exit(
            "--seed needs VCELL_ROOT set to a separate tree, or this run would\n"
            "overwrite the default seed's checkpoint, predictions and metrics\n"
            "rather than sit beside them (they are all addressed by predictor\n"
            "name, not by seed). For example:\n\n"
            "  mkdir -p $SCRATCH/seed43 && ln -s $PWD/data $SCRATCH/seed43/data\n"
            "  VCELL_ROOT=$SCRATCH/seed43 python -m benchmark.run_pipeline all \\\n"
            "      --dataset adamson16 --scenario UnseenPert --fold 0 \\\n"
            "      --predictor GEARS-ct --seed 43")


def _foreign_seed(ds: str, p: str, sc: str, fold: int, seed: int) -> Optional[int]:
    """The seed already recorded in this run dir, when it is not `seed`.

    The precise backstop to `_require_seed_isolation`'s blunt one: it also fires
    when an isolated tree is reused for a second seed, which is an easy mistake to
    make and destroys a run that costs GPU-hours.
    """
    try:
        stored = json.loads(
            (checkpoint_dir(ds, p, sc, fold) / FINGERPRINT_FILE).read_text()).get("seed")
    except (OSError, ValueError):
        return None
    return int(stored) if stored is not None and int(stored) != int(seed) else None


def cmd_all(args) -> None:
    cmd_fit(args)
    cmd_predict(args)
    cmd_metrics(args)


# ===================================================================
# CLI entry
# ===================================================================


def _add_selector_args(p: argparse.ArgumentParser,
                        include_predictor: bool = True,
                        include_metric: bool = True,
                        include_filters: bool = False) -> None:
    p.add_argument("--dataset", required=True,
                   help="dataset name, comma-list, or 'all'")
    p.add_argument("--scenario", default="all",
                   help="scenario name (PascalCase), comma-list, or 'all'")
    p.add_argument("--fold", default="all",
                   help="fold index, comma-list, range (e.g. 0-4), or 'all'")
    if include_predictor:
        p.add_argument("--predictor", default="all",
                       help="predictor name, comma-list, category, or 'all'")
    if include_metric:
        p.add_argument("--metric", default="main_benchmark",
                       help="metric name, comma-list, 'main_benchmark', or 'all'")
        p.add_argument("--missing-genes",
                       choices=["fill_missing_genes_with_zero", "drop_missing_genes"],
                       default="fill_missing_genes_with_zero",
                       help="gene-axis handling of genes a predictor did not emit, for "
                            "perturbations it DID predict: 'fill_missing_genes_with_zero' "
                            "(default — score the full panel, missing genes = control/zero "
                            "delta, all methods on an identical gene set) or "
                            "'drop_missing_genes' (exclude them, Miller-style subset). "
                            "Applies to metrics/bs/cf; the 'summary' command emits both modes.")
    if include_filters:
        p.add_argument("--filter-drf", action="store_true",
                       help="drop rows where DRF <= threshold")
        p.add_argument("--filter-drf-threshold", type=float, default=0.2)
        p.add_argument("--filter-pos-ge-neg", action="store_true",
                       help="drop rows where positive control fails to beat negative")
        p.add_argument("--filter-min-genes", type=int, default=0,
                       help="drop rows with fewer than N valid genes")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="benchmark.run_pipeline",
        description="Unified benchmark CLI (fit / predict / metrics / pool / summary / "
                    "bs / cf / all / list / verify).",
    )
    sub = p.add_subparsers(dest="subcommand", required=True)

    for name, handler, with_predictor, with_metric, with_filters in [
        ("fit",     cmd_fit,     True,  False, False),
        ("predict", cmd_predict, True,  False, False),
        ("metrics", cmd_metrics, True,  True,  False),
        ("pool",    cmd_pool,    False, False, False),
        ("summary", cmd_summary, False, True,  False),
        ("bs",      cmd_bs,      False, True,  True),
        ("cf",      cmd_cf,      True,  True,  True),
        ("all",     cmd_all,     True,  True,  False),
        ("list",    cmd_list,    True,  False, False),
        ("roster",  cmd_roster,  False, False, False),
        ("verify",  cmd_verify,  False, False, False),
    ]:
        ssp = sub.add_parser(name)
        if name != "roster":     # the roster is a property of the code, not of a dataset
            _add_selector_args(ssp, with_predictor, with_metric, with_filters)
        if name in {"bs", "cf"}:
            ssp.add_argument("--baseline-predictor", default=None,
                              help="registered predictor name, or a comma-list of them "
                                   "(compared in one run, distinguished by the "
                                   "baseline_predictor column); defaults to scenario default")
        if name in {"fit", "all"}:
            ssp.add_argument("--force", action="store_true",
                             help="retrain even if this fold is already trained "
                                  "(default: skip; for container predictors this also "
                                  "re-runs the expensive GPU training and wipes the run dir)")
            ssp.add_argument("--steal-claim", action="store_true",
                             help="train into a run dir another process currently "
                                  "claims. Separate from --force on purpose: --force "
                                  "answers 'retrain an already-trained fold?', this "
                                  "answers 'collide with a live run?'. A claim older "
                                  "than the model's timeout is reclaimed without it.")
        if name in {"fit", "predict", "all"}:
            ssp.add_argument("--seed", type=int, default=None,
                             help="train/score a trained predictor under this seed "
                                  "instead of its default. Requires VCELL_ROOT to "
                                  "point at a separate tree: run dirs, predictions "
                                  "and metrics rows are addressed by predictor name, "
                                  "not by seed, so a second seed would overwrite the "
                                  "first in place. For one-off seed-stability runs.")
        if name in {"fit", "all"}:
            ssp.add_argument("--no-verify", action="store_true",
                             help="skip the pre-benchmark L0+L1 preflight")
        if name == "roster":
            ssp.add_argument("--json", action="store_true",
                             help="emit the roster as JSON for tooling")
        if name == "verify":
            ssp.add_argument("--full", action="store_true",
                             help="also run L3 (GT-axis) + L6 (prediction alignment); "
                                  "needs saved predictions")
        ssp.set_defaults(func=handler)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    # Before any work: a non-default seed that would land on the default tree is
    # refused here rather than discovered after the wipe.
    _require_seed_isolation(args)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
