"""Paths, dataset configuration, and job enumeration for the benchmark package.

Single source of truth for:
  * filesystem paths (DATA_DIR, MODELS_DIR, RESULTS_DIR)
  * per-dataset scenario inventory (DATASET_CONFIG)
  * job enumeration over (dataset, scenario, fold) combinations
  * KO name classification (single / double / partial)

Scenario names are PascalCase everywhere: UnseenPert, UnseenCell, UnseenBoth,
UnseenPair, UnseenDose, UnseenCombo. The legacy S1–S6 codes are not used.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ===================================================================
# Path constants
#
# The repo root defaults to the canonical cluster location but is overridable
# with the VCELL_ROOT environment variable so the benchmark runs unmodified
# elsewhere (CI, a different cluster, a clone). data/, models/ and results/
# always sit directly under the root.
# ===================================================================

PROJECT_ROOT = Path(os.environ.get("VCELL_ROOT", "/cluster/work/boeva/virtual_cell_reasoning"))
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# ===================================================================
# Dataset configuration — scenarios are PascalCase identifiers
# ===================================================================

DATASET_CONFIG: Dict[str, Dict] = {
    "adamson16": {
        "scenarios": {"UnseenPert": {"n_folds": 5}},
    },
    "frangieh21": {
        "scenarios": {"UnseenPert": {"n_folds": 5}},
    },
    "norman19": {
        "scenarios": {"UnseenCombo": {"n_folds": 2}},
    },
    "sunshine23": {
        "scenarios": {"UnseenCombo": {"n_folds": 2}},
    },
    "wessels23": {
        "scenarios": {"UnseenCombo": {"n_folds": 2}},
    },
    "replogle20": {
        "scenarios": {"UnseenCombo": {"n_folds": 2}},
    },
    "replogle22": {
        "scenarios": {
            "UnseenPert": {"n_folds": 5},
            "UnseenBoth": {"n_folds": 5},
        },
    },
    "mcfaline23": {
        "scenarios": {
            "UnseenPert": {"n_folds": 5},
            "UnseenCell": {"n_folds": 3},
            "UnseenBoth": {"n_folds": 5},
            "UnseenPair": {"n_folds": 5},
        },
    },
    "jiang24": {
        "scenarios": {
            "UnseenPert": {"n_folds": 5},
            "UnseenCell": {"n_folds": 5},
            "UnseenBoth": {"n_folds": 5},
            "UnseenPair": {"n_folds": 5},
        },
    },
    "xatlas_orion": {
        "scenarios": {
            "UnseenPert": {"n_folds": 5},
            "UnseenBoth": {"n_folds": 5},
        },
    },
    "ecoli_synthetic": {
        # 3 folds per scenario: the h5ad ADOPTS the exp3/exp4 depth_hypothesis
        # splits (seeds 42/1337/7 -> folds 0/1/2) so the already-trained
        # transformer predictions can be scored without retraining. All six
        # scenarios (S1-S6 upstream) therefore carry exactly 3 seed-folds.
        "scenarios": {
            "UnseenPert": {"n_folds": 3},
            "UnseenCell": {"n_folds": 3},
            "UnseenBoth": {"n_folds": 3},
            "UnseenPair": {"n_folds": 3},
            "UnseenDose": {"n_folds": 3},
            "UnseenCombo": {"n_folds": 3},
        },
    },
}

ALL_SCENARIOS: Tuple[str, ...] = (
    "UnseenPert", "UnseenCell", "UnseenBoth", "UnseenPair",
    "UnseenDose", "UnseenCombo",
)

# Scenarios that rotate the cell-type (bin) axis: the test split holds out
# whole cell types and/or scattered (cell_type, pert) pairs. Used by
# `data_loader.split` to decide when to derive per-bin train/val/test
# memberships, and by `verify` to know which scenarios carry a bin axis.
# (UnseenPert / UnseenCombo / UnseenDose have no bin holdout → all bins train.)
BIN_AXIS_SCENARIOS: Tuple[str, ...] = ("UnseenCell", "UnseenBoth", "UnseenPair")

# ===================================================================
# Path helpers
# ===================================================================


# Expected pure-HVG count per dataset. Default is the N_HVG used by get_data
# (8192). replogle22 has fewer genes surviving QC than 8192, so
# highly_variable_genes(n_top_genes=8192) caps at the available count (7226) —
# an expected QC outcome, not a bug.
DEFAULT_N_HVG = 8192
# ecoli_synthetic is a fully synthetic SERGIO dataset with only 2853 genes total
# (all retained; no HVG subsetting), so its "highly_variable" panel is all 2853.
_HVG_OVERRIDES: Dict[str, int] = {"replogle22": 7226, "ecoli_synthetic": 2853}


def expected_hvg(dataset: str) -> int:
    """Highly-variable gene count expected for this dataset (validates the
    panel's `highly_variable` annotation is fully populated)."""
    return _HVG_OVERRIDES.get(dataset, DEFAULT_N_HVG)


# Multi-bin datasets whose UnseenPert folds follow the perturbench seed-0 block
# partition (assign_split_folds_unseen_pert_multibin), so per fold the UnseenPert
# test set is a subset of the UnseenBoth held-out block. The verifier enforces
# that convention (L1d) only for these; other multi-bin datasets (e.g.
# xatlas_orion, which uses the single-bin round-robin) are reported, not failed.
MULTIBIN_BLOCK_PARTITION_DATASETS: frozenset = frozenset(
    {"mcfaline23", "jiang24", "replogle22"})

# Datasets whose CV folds are ADOPTED verbatim from an external pipeline rather
# than generated by the native `assign_split_folds_*` helpers. Their per-fold
# test sets are independent draws (not a round-robin partition of the
# perturbations), so the verifier's UnseenPert full-coverage convention is
# reported as INFO for them, not FAIL. ecoli_synthetic adopts the
# depth_hypothesis exp3 folds (seeds 42/1337/7) so the already-trained
# transformer predictions can be scored without retraining.
ADOPTED_FOLD_DATASETS: frozenset = frozenset({"ecoli_synthetic"})


def h5ad_path(dataset: str) -> str:
    """Return the full path to the dataset's processed h5ad.

    There is exactly ONE gene panel: `{ds}_processed.h5ad`, gene set =
    HVG ∪ perturbation-target genes. The file holds single-cell expression,
    splits (obs columns), pseudobulk, DEG matrices (both halves) and the
    derived deg_arrays.
    """
    return str(DATA_DIR / dataset / f"{dataset}_processed.h5ad")


def predictor_dir(
    dataset: str, predictor: str, scenario: str, fold: int,
) -> Path:
    """Per-predictor output directory: models/{dataset}/{predictor}/{scenario}/fold{N}/.

    Every predictor (analytical, learned, control, DL) writes its
    `weights.npz` and `predictions.npz` here.
    """
    return MODELS_DIR / dataset / predictor / scenario / f"fold{fold}"


def predictions_path(
    dataset: str, predictor: str, scenario: str, fold: int,
) -> Path:
    return predictor_dir(dataset, predictor, scenario, fold) / "predictions.npz"


def existing_predictions_path(
    dataset: str, predictor: str, scenario: str, fold: int,
) -> Optional[Path]:
    """The predictions.npz that should be scored, or None if absent."""
    p = predictions_path(dataset, predictor, scenario, fold)
    return p if p.exists() else None


def weights_path(
    dataset: str, predictor: str, scenario: str, fold: int,
) -> Path:
    return predictor_dir(dataset, predictor, scenario, fold) / "weights.npz"


def checkpoint_dir(
    dataset: str, predictor: str, scenario: str, fold: int,
) -> Path:
    """Where an expensively-trained predictor keeps its run state.

    Sits beside that fold's `predictions.npz` so one run is one directory:
    `models/{dataset}/{predictor}/{scenario}/fold{N}/checkpoint/`.

    Resolved HERE, not inside a predictor, so every caller — the CLI, verify,
    lifecycle tooling — finds the same path. `VCR_TRAINED_MODEL_ROOT` relocates
    the heavy artefacts (they are GBs, and this volume runs full) while the
    small scored outputs stay in the repo tree; it is read once, in this
    function, rather than being a per-predictor branch that bypasses
    `predictor_dir` and leaves the checkpoint unfindable by anything else.
    """
    root = os.environ.get("VCR_TRAINED_MODEL_ROOT")
    if root:
        return Path(root) / dataset / predictor / scenario / f"fold{fold}"
    return predictor_dir(dataset, predictor, scenario, fold) / "checkpoint"


def results_fold_dir(dataset: str, scenario: str, fold: int) -> Path:
    """Results directory for per-fold metric CSVs (long format)."""
    return RESULTS_DIR / dataset / scenario / f"fold{fold}"


def results_pooled_dir(dataset: str, scenario: str) -> Path:
    """Results directory for pooled-across-folds metrics."""
    return RESULTS_DIR / dataset / scenario / "pooled"


def parse_target_genes(ko_name: str) -> List[str]:
    """Parse a perturbation label into its target gene symbol tokens.

    Strips the dose suffix after '@', splits combos on '+' and complexes on ';',
    drops empty tokens. The SINGLE canonical parser, reused by the predictors
    (``analytical.parse_target_genes`` re-exports this) and by ``verify`` — kept
    here in the dependency-free config leaf so neither the validator nor the
    verifier has to import the predictor stack just to parse a label.

        "GeneA"            -> ["GeneA"]
        "GeneA@0.5"        -> ["GeneA"]
        "GeneA+GeneB"      -> ["GeneA", "GeneB"]
        "GeneA;GeneB"      -> ["GeneA", "GeneB"]
    """
    base = str(ko_name).split("@")[0]
    return [p.strip() for p in base.replace(";", "+").split("+") if p.strip()]


def split_obs_column(scenario: str, fold: int) -> str:
    """Return the adata.obs column name for a (scenario, fold) split.

    Format: ``split_{scenario}_fold_{N}`` — assigned by `data/_utils.py`
    helpers (`assign_split_folds_unseen_*`).
    """
    return f"split_{scenario}_fold_{fold}"


# ===================================================================
# Job enumeration
# ===================================================================


def all_dataset_scenarios() -> List[Tuple[str, str]]:
    """Return all valid (dataset, scenario) combinations from DATASET_CONFIG."""
    return [(ds, sc) for ds, cfg in DATASET_CONFIG.items()
            for sc in cfg["scenarios"]]


def enumerate_jobs(
    dataset_filter: Optional[str] = None,
    scenario_filter: Optional[str] = None,
    fold_filter: Optional[int] = None,
) -> List[Tuple[str, str, int]]:
    """Enumerate all (dataset, scenario, fold) combinations under filters."""
    jobs: List[Tuple[str, str, int]] = []
    for ds, ds_cfg in DATASET_CONFIG.items():
        if dataset_filter and ds != dataset_filter:
            continue
        for sc, sc_cfg in ds_cfg["scenarios"].items():
            if scenario_filter and sc != scenario_filter:
                continue
            for f in range(sc_cfg["n_folds"]):
                if fold_filter is not None and f != fold_filter:
                    continue
                jobs.append((ds, sc, f))
    return jobs


# NOTE: a `classify_ko_names()` helper used to live here (dose/combo/complex KO
# index classification). It had no callers anywhere in the repo and was removed
# (2026-06-10). KO-label parsing now goes through `parse_target_genes` above.
