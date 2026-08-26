"""Predictor protocol, registry, and the gene-alignment invariant.

Every benchmark predictor — analytical, learned, control, DL — implements
the same `Predictor` interface:

  * `fit(store, scenario, fold)`       — train (no-op for analytical)
  * `predict(store, scenario, fold)`   — return (n_test_bins, n_test_kos, n_genes)
  * `save_weights(path)`               — serialize parameters (empty for analytical)
  * `load_weights(path)`               — reconstruct from weights.npz
  * `save_predictions(path, ...)`      — write self-describing NPZ (gene/ko/bin names)

The save/load roundtrip + `load_predictions()` enforce the gene-alignment
invariant: a `predictions.npz` whose `gene_names` array does not match the
dataset's `var_names` cannot be silently consumed downstream.
"""
from __future__ import annotations

import importlib
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Type

import numpy as np

from benchmark.config import existing_predictions_path, predictions_path, predictor_dir
from benchmark.data_loader import DatasetStore

log = logging.getLogger(__name__)


# ===================================================================
# Errors
# ===================================================================


class GeneAlignmentError(ValueError):
    """Raised when a predictor's NPZ gene_names don't match the dataset h5ad.

    This is the regression that motivated the refactor — if you see this
    exception, the predictor was written against a different gene ordering
    or gene set than the dataset is currently using.
    """


# ===================================================================
# Predictor protocol
# ===================================================================


class Predictor(ABC):
    """Base class for all benchmark predictors.

    Subclasses MUST set the class attributes `name`, `needs_training`, and
    `scenarios` (the list of PascalCase scenario names this predictor can
    handle). Instances are typically constructed via the registry:

        predictor = PREDICTOR_REGISTRY[name](**kwargs)

    Unresolved-target / unhandled-KO policy (uniform across predictors): a
    predictor that cannot represent a particular test KO (target gene absent
    from the panel, combo single missing, test bin never trained, …) logs a
    warning and emits NaN for that slice, which the metric layer skips — it does
    NOT raise. Set `has_drop_rule = True` for predictors that legitimately leave
    some perturbations uncovered this way; the verifier (L5) requires every other
    predictor to cover every perturbation.
    """

    name: str = ""
    needs_training: bool = False
    scenarios: List[str] = []
    # True for predictors with a per-target drop rule (cover < all perts is OK).
    # The verifier derives its coverage invariant from this attribute.
    has_drop_rule: bool = False

    # -----------------------------------------------------------------
    # Methods subclasses implement
    # -----------------------------------------------------------------

    @abstractmethod
    def fit(self, store: DatasetStore, scenario: str, fold: int) -> None:
        """Train on the fold's training split. No-op for analytical baselines."""

    @abstractmethod
    def predict(self, store: DatasetStore, scenario: str, fold: int) -> np.ndarray:
        """Return predictions shaped (n_test_bins, n_test_kos, n_genes).

        Implementations should slice `store.all_deltas` (all-cell training
        target) / `store.first_half_bulk` / `store.second_half_bulk` and
        `store.split(scenario, fold)` to figure out the indices to predict for.
        Evaluation scores against `store.first_half_deltas` (see meta_metrics).
        """

    # -----------------------------------------------------------------
    # Methods with default implementations (override if needed)
    # -----------------------------------------------------------------

    def is_trained(self, store: DatasetStore, scenario: str, fold: int) -> bool:
        """Is this predictor's trained state available on disk for this fold?

        Gates `run_pipeline predict`, which never trains: a predictor that is
        not trained is skipped with a "run fit first" message.

        Default: analytical baselines and controls carry no trained state, so
        they are always ready. Tiers that DO persist state override this and
        validate the artefact they actually need — the learned tier opens its
        `weights.npz`, container predictors check every file the container must
        load. A bare `.exists()` on a placeholder file is what made `predict`
        silently retrain, so it is not enough.

        Persistence itself is deliberately NOT on this base class: it means
        different things per tier (a small `weights.npz` of coefficients, a
        multi-GB container run dir, nothing at all), so each tier owns it
        rather than every predictor stubbing out a method it cannot honour.
        """
        return not self.needs_training

    def save_predictions(
        self,
        path: Path,
        preds: np.ndarray,
        gene_names: List[str],
        ko_names: List[str],
        bin_names: List[str],
        test_ko_indices: np.ndarray,
        test_bin_indices: Optional[np.ndarray],
        dataset: str,
        scenario: str,
        fold: int,
    ) -> None:
        """Write a self-describing predictions.npz.

        Enforces the gene-alignment invariant: every gene/ko/bin name array
        is stored alongside the deltas so downstream loaders can verify
        alignment before computing any metric.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if preds.ndim != 3:
            raise ValueError(
                f"{self.name}.save_predictions: deltas must be 3D, got shape {preds.shape}"
            )
        if len(gene_names) != preds.shape[-1]:
            raise ValueError(
                f"{self.name}.save_predictions: gene_names ({len(gene_names)}) "
                f"!= preds last dim ({preds.shape[-1]})"
            )
        if len(ko_names) != preds.shape[-2]:
            raise ValueError(
                f"{self.name}.save_predictions: ko_names ({len(ko_names)}) "
                f"!= preds 2nd-to-last dim ({preds.shape[-2]})"
            )
        if len(bin_names) != preds.shape[-3]:
            raise ValueError(
                f"{self.name}.save_predictions: bin_names ({len(bin_names)}) "
                f"!= preds 3rd-to-last dim ({preds.shape[-3]})"
            )

        np.savez(
            str(path),
            deltas=preds.astype(np.float32),
            gene_names=np.array(gene_names, dtype=object),
            ko_names=np.array(ko_names, dtype=object),
            bin_names=np.array(bin_names, dtype=object),
            test_ko_indices=np.asarray(test_ko_indices, dtype=np.int64),
            test_bin_indices=(
                np.asarray(test_bin_indices, dtype=np.int64)
                if test_bin_indices is not None and len(test_bin_indices) > 0
                else np.array([], dtype=np.int64)
            ),
            predictor_name=np.array(self.name),
            dataset=np.array(dataset),
            scenario=np.array(scenario),
            fold=np.array(fold),
        )


# ===================================================================
# Registry
# ===================================================================


PREDICTOR_REGISTRY: Dict[str, Type[Predictor]] = {}


def register(cls: Type[Predictor]) -> Type[Predictor]:
    """Decorator: add a Predictor subclass to PREDICTOR_REGISTRY."""
    if not cls.name:
        raise ValueError(f"@register: {cls.__name__} has empty name attribute")
    if cls.name in PREDICTOR_REGISTRY:
        raise ValueError(
            f"@register: predictor name {cls.name!r} already registered "
            f"by {PREDICTOR_REGISTRY[cls.name].__module__}"
        )
    PREDICTOR_REGISTRY[cls.name] = cls
    return cls


def get_predictor(name: str) -> Type[Predictor]:
    """Look up a predictor class by name. Eagerly imports the predictor modules
    so registration decorators have run before the lookup happens."""
    _ensure_predictors_loaded()
    if name not in PREDICTOR_REGISTRY:
        raise KeyError(
            f"Unknown predictor {name!r}. Known: {sorted(PREDICTOR_REGISTRY)}"
        )
    return PREDICTOR_REGISTRY[name]


_LOADED = False


def _ensure_predictors_loaded() -> None:
    """Import every predictor module so the `@register` decorators run.

    Discovers modules by walking the package rather than naming them: a
    hand-kept list silently loses predictors when someone adds a module (and was
    duplicated in three places, which is how GEARS-ct came to be missing from
    the `dl` category).

    IMPORT DISCIPLINE (relied on here): a predictor module must import cleanly in
    EVERY environment that loads the registry — including `preprocess`, which has
    no pyyaml and no torch. Keep env-specific/heavy imports INSIDE the methods
    that need them (`ContainerPredictor._recipe` imports yaml lazily; a future
    TorchPredictor must import torch inside `_train`/`_infer`). Because of that
    rule an ImportError here is a real bug, not an expected condition, so it is
    raised rather than swallowed — a silent skip would leave a registry hole that
    only shows up as "unknown predictor" much later.
    """
    global _LOADED
    if _LOADED:
        return
    import pkgutil
    import benchmark.predictors as _pkg

    failures = {}
    for mod in pkgutil.iter_modules(_pkg.__path__):
        # `_`-prefixed modules/packages are helpers (`_shared`, `_container`) and
        # register nothing; `base` is this module.
        if mod.name.startswith("_") or mod.name == "base":
            continue
        try:
            importlib.import_module(f"benchmark.predictors.{mod.name}")
        except Exception as e:                      # noqa: BLE001 - reported below
            failures[mod.name] = f"{type(e).__name__}: {e}"
    if failures:
        detail = "; ".join(f"{m} ({e})" for m, e in sorted(failures.items()))
        raise ImportError(
            f"predictor module(s) failed to import, so the registry is incomplete: "
            f"{detail}. Heavy/env-specific imports belong inside methods, not at "
            f"module level (see _ensure_predictors_loaded)."
        )
    _LOADED = True


def predictor_category(cls) -> str:
    """Grouping used by `--predictor <category>`, DERIVED — never declared.

    Tier first (a container/adapter is `dl`, anything persisting a weights.npz is
    `learned`), then the defining module for the stateless families that the
    class hierarchy genuinely does not distinguish: `analytical` and `controls`
    are both plain stateless `Predictor`s, so only their module separates them.

    Imports are deferred to call time: these modules import `base`, so importing
    them at module level would be circular.
    """
    from benchmark.predictors.learned import LearnedPredictor
    from benchmark.predictors.container_predictor import ContainerPredictor
    from benchmark.predictors.dl_adapter import DLAdapter

    if issubclass(cls, (ContainerPredictor, DLAdapter)):
        return "dl"
    if issubclass(cls, LearnedPredictor):
        return "learned"
    return cls.__module__.rsplit(".", 1)[-1]


# ===================================================================
# Prediction loading (with mandatory gene-alignment verification)
# ===================================================================


@dataclass
class PredictionBundle:
    """Loaded predictions + index mappings into the canonical DatasetStore."""
    deltas: np.ndarray
    gene_names: List[str]
    ko_names: List[str]
    bin_names: List[str]
    gene_index_in_store: np.ndarray
    ko_index_in_store: np.ndarray
    bin_index_in_store: np.ndarray
    predictor_name: str
    dataset: str
    scenario: str
    fold: int


def load_predictions(
    dataset: str,
    scenario: str,
    fold: int,
    predictor_name: str,
    *,
    store: Optional[DatasetStore] = None,
    require_exact_genes: bool = True,
) -> PredictionBundle:
    """Load `predictions.npz` and verify gene/ko/bin alignment against the store.

    With `require_exact_genes=True` (default), the predictor's gene_names must
    EXACTLY match the dataset's var_names (same order, same set). Anything
    else raises GeneAlignmentError.

    With `require_exact_genes=False`, returns a name-based intersection
    mapping (used by DL adapters whose gene panel is a subset).
    """
    p = existing_predictions_path(dataset, predictor_name, scenario, fold)
    if p is None:
        raise FileNotFoundError(
            f"load_predictions: missing predictions for {dataset}/{scenario}/fold{fold}/"
            f"{predictor_name}. Run "
            f"`python -m benchmark.run_pipeline predict --dataset {dataset} "
            f"--scenario {scenario} --fold {fold} --predictor {predictor_name}` first."
        )

    store = store or DatasetStore(dataset)
    npz = np.load(str(p), allow_pickle=True)

    pred_genes = [str(g) for g in npz["gene_names"]]
    pred_kos = [str(k) for k in npz["ko_names"]]
    pred_bins = [str(b) for b in npz["bin_names"]]
    deltas = npz["deltas"].astype(np.float32)

    if require_exact_genes:
        if pred_genes != store.gene_names:
            missing = set(store.gene_names) - set(pred_genes)
            extra = set(pred_genes) - set(store.gene_names)
            raise GeneAlignmentError(
                f"{predictor_name} on {dataset}/{scenario}/fold{fold}: "
                f"predictor has {len(pred_genes)} genes, store has {len(store.gene_names)}; "
                f"{len(missing)} missing, {len(extra)} extra"
                + (f"; first missing: {sorted(missing)[:5]}" if missing else "")
            )
        gene_idx = np.arange(len(pred_genes), dtype=np.int64)
    else:
        gene_to_store = {g: i for i, g in enumerate(store.gene_names)}
        gene_idx = np.array(
            [gene_to_store.get(g, -1) for g in pred_genes], dtype=np.int64,
        )
        if (gene_idx < 0).all():
            raise GeneAlignmentError(
                f"{predictor_name} on {dataset}/{scenario}/fold{fold}: "
                f"no predictor gene is present in the store."
            )

    # Map ko_names and bin_names through the canonical orderings.
    ko_to_store = {k: i for i, k in enumerate(store.ko_names)}
    bin_to_store = {b: i for i, b in enumerate(store.bin_names)}
    ko_idx = np.array(
        [ko_to_store.get(k, -1) for k in pred_kos], dtype=np.int64,
    )
    bin_idx = np.array(
        [bin_to_store.get(b, -1) for b in pred_bins], dtype=np.int64,
    )

    return PredictionBundle(
        deltas=deltas,
        gene_names=pred_genes,
        ko_names=pred_kos,
        bin_names=pred_bins,
        gene_index_in_store=gene_idx,
        ko_index_in_store=ko_idx,
        bin_index_in_store=bin_idx,
        predictor_name=str(npz.get("predictor_name", predictor_name)),
        dataset=str(npz.get("dataset", dataset)),
        scenario=str(npz.get("scenario", scenario)),
        fold=int(npz.get("fold", fold)),
    )


__all__ = [
    "GeneAlignmentError",
    "Predictor",
    "PREDICTOR_REGISTRY",
    "register",
    "get_predictor",
    "PredictionBundle",
    "load_predictions",
]
