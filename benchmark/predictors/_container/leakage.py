"""Hard-fail leakage assertion — **column-native** oracle over the combined h5ad
the container actually reads.

The vendored models are handed ONE combined h5ad + a per-cell ``split_name`` obs
column (see ``docker/CONTRACT.md``); the authoritative leak-safe assignment IS
that column. So the leakage gate reads ``obs[split_name]`` **directly** — the exact
column the container consumes — rather than a separately materialized copy that
could diverge.

Two modes (one check alone misses a whole class of leak):

  * **condition** mode — perturbation-holdout regimes (UnseenPert / UnseenCombo /
    UnseenDose): the held-out **perturbations** (conditions in ``test``) must be
    disjoint from the perturbations seen in ``train ∪ val``. A test pert that
    also appears in train/val is a direct leak. GEARS relies on this entirely for
    UnseenPert (its per-cell filter is skipped there).
  * **pair** mode — cell-axis regimes (UnseenCell / UnseenBoth / UnseenPair): a
    held-out perturbation may legitimately be trained in OTHER cell types, so
    condition-disjointness is the WRONG check. Instead: no held-out
    ``(covariate, perturbation)`` **pair** present in ``test`` may appear in
    ``train ∪ val``. This is the cross-cell-type leak.

Both modes additionally assert the train/val/test cell sets are disjoint. Control
cells are excluded from the pert/pair leak sets (basal is shared, never a held-out
perturbation); the "basal not starved" guard lives in the predictor's preflight.
"""
from __future__ import annotations

import argparse
import json
import sys

import anndata as ad

# Cell-axis regimes: the test split holds out whole cell types and/or scattered
# (cell_type, pert) pairs → use the (cov, pert)-pair leak check. Everything else
# is a perturbation-holdout regime → use the condition-disjointness check.
BIN_AXIS_REGIMES = ("UnseenCell", "UnseenBoth", "UnseenPair")
DEFAULT_CONTROL_LABELS = ("control", "ctrl", "ctrl_iegfp")


def _is_control(cond_series, control_labels):
    """Boolean mask of control cells: exact-label match OR contains 'ctrl'
    (matches the reference's substring convention; no gene symbol contains
    'ctrl', and 'control' — which lacks the 'ctrl' substring — is caught by the
    exact-label list)."""
    exact = cond_series.isin(list(control_labels))
    substr = cond_series.str.contains("ctrl", case=False, na=False, regex=False)
    return exact | substr


def assert_leak_safe_columns(
    h5ad_path,
    split_name: str,
    *,
    regime: str,
    covariate_col: str = "cell_type",
    control_labels=DEFAULT_CONTROL_LABELS,
    bin_axis_regimes=BIN_AXIS_REGIMES,
) -> dict:
    """Column-native leakage check over ``obs[split_name]`` of one combined h5ad.

    Returns a report dict with ``ok`` (bool) + ``problems`` (list). Never raises
    on a leak — the caller decides (``main`` exits 1; a predictor raises).
    """
    adata = ad.read_h5ad(str(h5ad_path), backed="r")  # obs is in-memory; X stays lazy
    obs = adata.obs
    if split_name not in obs.columns:
        return {"ok": False, "h5ad": str(h5ad_path), "split_name": split_name,
                "problems": [f"split column {split_name!r} not in obs"]}
    if "condition" not in obs.columns:
        return {"ok": False, "h5ad": str(h5ad_path), "split_name": split_name,
                "problems": ["obs has no 'condition' column"]}

    split = obs[split_name].astype(str)
    cond = obs["condition"].astype(str)
    has_cov = covariate_col in obs.columns
    cov = (obs[covariate_col].astype(str) if has_cov
           else cond.map(lambda _: "_all_"))
    is_ctrl = _is_control(cond, control_labels)

    train_m = (split == "train").values
    val_m = (split == "val").values
    test_m = (split == "test").values
    trainval_m = train_m | val_m
    ctrl = is_ctrl.values

    problems: list[str] = []

    # partition sanity — a cell can't be in two of train/val/test simultaneously
    # (guaranteed for a single categorical column, but assert to catch a caller
    # that hands a bad column).
    if (train_m & test_m).any() or (val_m & test_m).any() or (train_m & val_m).any():
        problems.append("train/val/test cell masks are not mutually exclusive")

    n_test = int(test_m.sum())
    if n_test == 0:
        problems.append(f"no cells labelled 'test' in {split_name!r} — degenerate split")

    # Control-cell placement is INFORMATIONAL, not a leak: control is basal and
    # shared, never a held-out perturbation, so a control cell labelled 'test' is
    # not leakage (these per-cell split columns fold-label ALL cells, controls
    # included). The genuine "basal not starved" guard — ≥1 control cell in train,
    # per covariate for cell-aware — lives in the predictor's preflight (B1).
    n_ctrl_test = int((ctrl & test_m).sum())

    cond_v = cond.values
    cov_v = cov.values
    bin_axis = regime in bin_axis_regimes

    if bin_axis:
        mode = "pair"
        test_pairs = set(zip(cov_v[test_m & ~ctrl], cond_v[test_m & ~ctrl]))
        tv_pairs = set(zip(cov_v[trainval_m & ~ctrl], cond_v[trainval_m & ~ctrl]))
        leaked = test_pairs & tv_pairs
        n_test_units, n_leaked = len(test_pairs), len(leaked)
        if leaked:
            problems.append(
                f"{n_leaked} held-out (covariate, perturbation) pairs also present "
                f"in train/val (e.g. {sorted(leaked)[:3]})")
    else:
        mode = "condition"
        test_perts = set(cond_v[test_m & ~ctrl])
        tv_perts = set(cond_v[trainval_m & ~ctrl])
        leaked = test_perts & tv_perts
        n_test_units, n_leaked = len(test_perts), len(leaked)
        if leaked:
            problems.append(
                f"{n_leaked} held-out perturbations also present in train/val "
                f"(e.g. {sorted(leaked)[:5]})")

    return {
        "ok": not problems,
        "h5ad": str(h5ad_path),
        "split_name": split_name,
        "regime": regime,
        "mode": mode,
        "covariate_col": covariate_col if has_cov else None,
        "n_train": int(train_m.sum()), "n_val": int(val_m.sum()), "n_test": n_test,
        "n_test_units": n_test_units, "n_leaked": n_leaked,
        "n_control_in_test": n_ctrl_test,  # informational (not a leak)
        "problems": problems,
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Assert a split is leak-safe (column-native over the combined "
                    "h5ad: --h5ad + --split-name + --regime).")
    ap.add_argument("--h5ad", required=True, help="combined h5ad path")
    ap.add_argument("--split-name", required=True,
                    help="per-cell split obs column, e.g. split_UnseenPert_fold_0")
    ap.add_argument("--regime", required=True,
                    help="PascalCase regime (selects condition vs pair mode)")
    ap.add_argument("--covariate-col", default="cell_type")
    a = ap.parse_args()

    report = assert_leak_safe_columns(
        a.h5ad, a.split_name, regime=a.regime, covariate_col=a.covariate_col)
    print(json.dumps(report, indent=2))
    if not report["ok"]:
        print("LEAKAGE ASSERTION FAILED", file=sys.stderr)
        sys.exit(1)
    print("leakage assertion PASSED")


if __name__ == "__main__":
    main()
