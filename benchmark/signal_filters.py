"""Self-contained perturbation signal-filter predicates — the single source of
truth for "which perturbations carry a detectable transcriptional effect".

These read the per-(bin, ko) signal statistics produced by ``perturb_dataset_analysis``
(``snr_stats.parquet``, ``e_distance.parquet``, ``signal_cutoffs.json``) and return the
set of ``(bin, canonical-ko)`` keys that PASS a given inclusion rule.

**Headline rule** (``DEFAULT_FILTER``): the energy-distance **E-test** (``etest_pass_keys``,
unadjusted ``e_test_p < 0.05``) — a metric-agnostic, per-dataset-null-calibrated
"is there a detectable effect" gate. It agrees closely with ``SNR > null-p95`` and
``E-dist > null-p95`` (the calibrated trio), whereas a fixed ``SNR > 1`` is
dataset-inconsistent and per-metric ``DRF`` is metric-inconsistent. The BH-adjusted
E-test column (``significant_adj``) is degenerate (≈0 significant everywhere) and must
NOT be used.

**Reproducibility-aware rule** (``nsr_or_rho_pass_keys``, recommended when small but
real effects must be retained): keep if **NSR > null-p95 OR ρ > null-p95**, where
``snr`` is the Monte-Carlo NSR (= ``signal/SE``; ``nsr.py``) and ``ρ`` is
``tech_dup_pearson`` (the split-half *direction* reproducibility). The two axes answer
different questions and split the work:
  * **NSR** (``snr_pass_keys``) = *detectability* (effect vs its SE). Catches fakes
    (a relabeled control has ``snr≈1``) but ALSO drops genuine **weak** effects.
  * **ρ** (``rho_pass_keys``) = *direction reproducibility* (do the two tech-dup halves
    agree on the sign pattern?). Fakes have ``ρ≈0`` (drops 99.8% of relabeled-control
    draws), while a weak-but-real effect with a consistent direction has ``ρ>0`` — so ρ
    **keeps weak-but-reproducible perts that NSR would discard**, de-confounded from
    effect size.
  * NOTE: do **not** use the split-half *magnitude* disagreement ``D(p)=noise_l2`` as an
    inclusion axis — a fake is a *reproducibly null* measurement (``V_p≈V_X``), so a
    ``D≤floor`` gate KEEPS fakes. ``D(p)`` is a QC/response-homogeneity read, not a gate.
  * Identifiability caveat: a *literal* zero-effect perturbation is statistically
    indistinguishable from a fake (no direction to reproduce), so any fake-catching gate
    drops it; it is trivially predicted by the control baseline anyway. Remove genuine
    non-targeting controls by annotation upstream, not by a statistical gate.
``nsr_or_rho`` drops only the ``(low-NSR AND low-ρ)`` noise/fake corner.

Dependency arrow: ``config -> signal_filters -> {plotting, scripts, ...}``. This module
imports only ``benchmark.config`` (+ numpy/pandas), never ``plotting`` — so figures,
tables, the recovery corpus, and metrics can all share ONE inclusion set without a
core<-plotting cycle. (Per-metric DRF stays in ``meta_metrics`` since it needs the
bs/cf dataframe, not these parquets.)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Set, Tuple

import numpy as np
import pandas as pd

from benchmark.config import PROJECT_ROOT

# Signal stats live alongside the dataset analysis outputs (VCELL_ROOT-aware via config).
EDIST_ROOT = PROJECT_ROOT / "perturb_dataset_analysis" / "results"

Key = Tuple[str, str]   # (bin_name, canonical ko)

DEFAULT_FILTER = "etest"   # the benchmark's headline inclusion rule


def canon(ko: str) -> str:
    """Order-insensitive perturbation key: 'A+B' == 'B+A'; no-op for single genes.
    Needed to join the signal-stat tables to results CSVs for combo perts."""
    return "+".join(sorted(str(ko).split("+")))


def _signal_dir(ds: str) -> Path:
    return EDIST_ROOT / ds


def _cutoffs_p95(ds: str) -> dict:
    """{bin_name: {snr, edist, ...}} per-bin null-p95 cutoffs (empty if absent)."""
    p = _signal_dir(ds) / "signal_cutoffs.json"
    if not p.exists():
        return {}
    return {str(b): v.get("cutoff_p95", {})
            for b, v in json.loads(p.read_text()).get("bins", {}).items()}


def _read(ds: str, name: str) -> pd.DataFrame:
    df = pd.read_parquet(_signal_dir(ds) / name).rename(columns={"bin": "bin_name", "ko": "ko_name"})
    df["ko_canon"] = df["ko_name"].map(canon)
    return df


def snr_pass_keys(ds: str, threshold: Optional[float] = None) -> Set[Key]:
    """(bin, canon-ko) whose SNR clears a cutoff.

    ``threshold=None`` -> per-bin null-p95 cutoff (statistical, default). A float
    applies a single fixed ``SNR > threshold`` to every bin (e.g. ``1.0``)."""
    snr = _read(ds, "snr_stats.parquet")
    if threshold is not None:
        return {(str(b), k) for b, k, s in zip(snr["bin_name"], snr["ko_canon"], snr["snr"])
                if np.isfinite(s) and s > threshold}
    cuts = _cutoffs_p95(ds)
    out: Set[Key] = set()
    for b, k, s in zip(snr["bin_name"], snr["ko_canon"], snr["snr"]):
        c = cuts.get(str(b), {}).get("snr")
        if c is not None and np.isfinite(s) and s > c:
            out.add((str(b), k))
    return out


def etest_pass_keys(ds: str, alpha: float = 0.05) -> Set[Key]:
    """(bin, canon-ko) of E-test-significant perts (unadjusted ``e_test_p < alpha``).
    The benchmark's HEADLINE inclusion rule. Do NOT use the BH-adjusted column."""
    ed = _read(ds, "e_distance.parquet")
    return {(str(b), k) for b, k, pv in zip(ed["bin_name"], ed["ko_canon"], ed["e_test_p"])
            if pd.notna(pv) and pv < alpha}


def edist_pass_keys(ds: str) -> Set[Key]:
    """(bin, canon-ko) whose E-distance exceeds the per-bin null-p95 cutoff."""
    ed = _read(ds, "e_distance.parquet")
    cuts = _cutoffs_p95(ds)
    out: Set[Key] = set()
    for b, k, d in zip(ed["bin_name"], ed["ko_canon"], ed["e_distance"]):
        c = cuts.get(str(b), {}).get("edist")
        if c is not None and np.isfinite(d) and d > c:
            out.add((str(b), k))
    return out


def rho_pass_keys(ds: str, threshold: Optional[float] = None) -> Set[Key]:
    """(bin, canon-ko) whose split-half *direction* reproducibility clears a cutoff.

    ``ρ = tech_dup_pearson`` (correlation of the two tech-dup half-deltas, all genes).
    ``threshold=None`` -> per-bin null-p95 cutoff (the ``pearson`` null, ~0.14). Keeps
    weak-but-reproducible effects (consistent direction) and drops fakes/noise (ρ≈0)."""
    snr = _read(ds, "snr_stats.parquet")
    col = "tech_dup_pearson"
    if threshold is not None:
        return {(str(b), k) for b, k, r in zip(snr["bin_name"], snr["ko_canon"], snr[col])
                if np.isfinite(r) and r > threshold}
    cuts = _cutoffs_p95(ds)
    out: Set[Key] = set()
    for b, k, r in zip(snr["bin_name"], snr["ko_canon"], snr[col]):
        c = cuts.get(str(b), {}).get("pearson")
        if c is not None and np.isfinite(r) and r > c:
            out.add((str(b), k))
    return out


def nsr_or_rho_pass_keys(ds: str) -> Set[Key]:
    """Recommended reproducibility-aware inclusion: ``NSR > null-p95 OR ρ > null-p95``.

    Union of :func:`snr_pass_keys` (detectability) and :func:`rho_pass_keys` (direction
    reproducibility). Drops only the ``(low-NSR AND low-ρ)`` corner (noise / fakes);
    keeps large effects (NSR) AND weak-but-reproducible effects (ρ). See module docstring."""
    return snr_pass_keys(ds) | rho_pass_keys(ds)


_PASS_KEY_FNS = {
    "etest": etest_pass_keys,
    "snr": snr_pass_keys,
    "rho": rho_pass_keys,
    "edist": edist_pass_keys,
    "nsr_or_rho": nsr_or_rho_pass_keys,
}


def pass_keys(ds: str, kind: str = DEFAULT_FILTER, **kw) -> Set[Key]:
    """Inclusion key set for filter ``kind`` ∈ {etest, snr, rho, edist, nsr_or_rho}."""
    if kind not in _PASS_KEY_FNS:
        raise ValueError(f"unknown filter {kind!r}; known: {sorted(_PASS_KEY_FNS)}")
    return _PASS_KEY_FNS[kind](ds, **kw)


def restrict_by_keys(df: pd.DataFrame, keep_keys: Set[Key], *,
                     bin_col: str = "bin_name", ko_col: str = "ko_name") -> pd.DataFrame:
    """Keep rows whose ``(str(bin), canon(ko))`` is in ``keep_keys``."""
    kc = df[ko_col].map(canon)
    mask = [(str(b), k) in keep_keys for b, k in zip(df[bin_col], kc)]
    return df[pd.Series(mask, index=df.index)].copy()


def restrict(df: pd.DataFrame, ds: str, kind: str = DEFAULT_FILTER, *,
             bin_col: str = "bin_name", ko_col: str = "ko_name", **kw) -> pd.DataFrame:
    """Keep only rows whose perturbation passes filter ``kind`` for dataset ``ds``."""
    return restrict_by_keys(df, pass_keys(ds, kind, **kw), bin_col=bin_col, ko_col=ko_col)


__all__ = [
    "EDIST_ROOT", "DEFAULT_FILTER", "Key", "canon",
    "snr_pass_keys", "etest_pass_keys", "edist_pass_keys",
    "rho_pass_keys", "nsr_or_rho_pass_keys",
    "pass_keys", "restrict", "restrict_by_keys",
]
