"""Run a vendored training container as a first-class benchmark ``Predictor``.

DL models are trained + inferred **inside our own leak-safe splits** and scored
identically to baselines — there is no external-prediction adoption, no
fold-alignment, no MANIFEST (that path, ``dl_adapter.py``, is deleted at M3).
``fit`` trains in the model's ``.sif``; ``predict`` infers and maps the
container's ``predictions.h5ad`` onto the benchmark delta tensor.

Combined-form feed (the atheus-native + only intake, verified GEARS+PRESAGE):
the container is handed ONE combined h5ad (``data_path``) carrying the
authoritative per-cell ``split_{Scenario}_fold_{N}`` column inline plus the
marginal condition lists; the wrapper slices itself. See ``docker/CONTRACT.md``.

Each model's **run recipe** (hyperparameters + runtime wiring) lives in a
per-model ``docker/<model>/model.yaml`` co-located with the model, loaded lazily
(``_recipe``). The predictor class itself declares only benchmark *capabilities*
(``name``, ``scenarios``, ``cell_aware``, ``has_drop_rule``) as plain Python
attributes — those are read by ``verify`` in a pyyaml-less env, so they must not
depend on the yaml.

Transitional naming: while the old external-adoption predictors ``GEARS`` /
``PRESAGE`` (``dl_adapter.py``) still coexist (through M2), the container-trained
versions register under ``-ct`` names (``GEARS-ct`` / ``PRESAGE-ct``, "container
trained"). M3 deletes the adapter and collapses the name.
"""
from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from benchmark.config import (
    h5ad_path, split_obs_column, predictor_dir, checkpoint_dir, parse_target_genes)
from benchmark.data_loader import DatasetStore
from benchmark.predictors.base import register
from benchmark.predictors.trained import TrainedPredictor
# Host-side glue for container predictors (moved out of docker/harness/ into this
# package — imported normally, no sys.path shim). None of these import yaml at
# module top, so importing this predictor stays safe in the pyyaml-less env.
from benchmark.predictors._container import (
    runner, tensor_map, leakage, contract as _contract, config as _config)

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]

# The cell-type (bin) axis obs column — data_loader derives store.bin_names from
# obs['cell_type'] for cell-axis regimes (data_loader.py:511-514).
COVARIATE_COLUMN = _config.COVARIATE_COLUMN


# ===================================================================
# ContainerPredictor base
# ===================================================================


class ContainerPredictor(TrainedPredictor):
    """Base for models trained in a vendored ``.sif`` on our leak-safe splits.

    A subclass declares its benchmark *capabilities* as plain class attributes —
    ``name``, ``scenarios``, ``cell_aware``, ``has_drop_rule`` — and points
    ``model_dir`` at the directory holding its ``model.yaml`` **run recipe**
    (hyperparameters + ``sif_path`` / ``entry`` / ``code_binds`` /
    ``gene2go_path`` / ``wandb_project`` / ``model_key``). The recipe is loaded
    lazily via :meth:`_recipe` (never at import — keeps the predictor registrable
    in the pyyaml-less ``preprocess`` env). Optionally override
    ``_preflight_model_specific``.
    """

    needs_training = True
    is_container_trained = True  # logic runs in the .sif → exempt from the L4 synthetic-contract meta-check
    # subclass-provided capability declaration (plain Python, read by verify):
    model_dir: str = ""          # repo-relative dir holding this model's model.yaml

    # -------- run recipe (lazy; from docker/<model>/model.yaml) --------

    @classmethod
    def _recipe(cls) -> Dict:
        """Load + cache this model's ``model.yaml`` run recipe.

        ``import yaml`` is deferred INTO this method (not module top) on purpose:
        ``fit``/``predict`` only run in the ``vcell`` env (which has pyyaml),
        whereas ``verify --contracts/--fast`` imports every predictor in the
        pyyaml-less ``preprocess`` env — a top-level yaml import there would be
        swallowed by ``base._ensure_predictors_loaded`` and silently drop this
        predictor from the registry.
        """
        cached = cls.__dict__.get("_recipe_cache")
        if cached is not None:
            return cached
        import yaml
        if not cls.model_dir:
            raise ValueError(f"{cls.name}: model_dir not set (cannot locate model.yaml)")
        path = REPO_ROOT / cls.model_dir / "model.yaml"
        if not path.exists():
            raise FileNotFoundError(f"{cls.name}: model recipe not found at {path}")
        recipe = yaml.safe_load(path.read_text())
        _validate_recipe(recipe, f"{cls.name} ({path})")
        cls._recipe_cache = recipe
        return recipe

    @property
    def sif_path(self) -> str:
        return self._recipe()["sif_path"]

    @property
    def entry(self) -> List[str]:
        return list(self._recipe()["entry"])

    @property
    def code_binds(self) -> Dict[str, str]:
        return dict(self._recipe()["code_binds"])

    @property
    def default_hyperparameters(self) -> Dict:
        return dict(self._recipe()["hyperparameters"])

    @property
    def wandb_project(self) -> Optional[str]:
        return self._recipe().get("wandb_project")

    @property
    def model_yaml_key(self) -> str:
        return self._recipe()["model_key"]

    @property
    def train_timeout(self) -> int:
        """Seconds before a hung training run is killed. Required in the recipe:
        without it a wedged container holds the GPU until SLURM reaps the job."""
        return int(self._recipe()["train_timeout"])

    @property
    def predict_timeout(self) -> int:
        return int(self._recipe()["predict_timeout"])

    @property
    def extra_config(self) -> Dict:
        """Model-specific keys injected into the container ``config.json``."""
        recipe = self._recipe()
        extra: Dict = {}
        if recipe.get("gene2go_path"):
            extra["gene2go_path"] = recipe["gene2go_path"]
        return extra

    # -------- paths --------

    def _sif(self) -> Path:
        sif = REPO_ROOT / self.sif_path
        if not sif.exists():
            raise FileNotFoundError(
                f"{self.name}: container image not found at {sif}. Build it on the "
                f"CustomApps client (`singularity build --fakeroot`) and symlink it "
                f"here — see docker/gears/BUILD_ENV.md.")
        return sif

    def _code_binds(self) -> Dict[str, str]:
        """Resolve repo-relative code bind sources to absolute host paths."""
        return {str(REPO_ROOT / src): dest for src, dest in self.code_binds.items()}

    def _data_dir(self, store: DatasetStore) -> Path:
        return Path(h5ad_path(store.dataset)).parent

    # -------- gates --------

    def _leakage_gate(self, store: DatasetStore, scenario: str, fold: int) -> None:
        report = leakage.assert_leak_safe_columns(
            h5ad_path(store.dataset), split_obs_column(scenario, fold),
            regime=scenario, covariate_col=COVARIATE_COLUMN)
        if not report["ok"]:
            raise RuntimeError(
                f"{self.name}: LEAKAGE GATE FAILED for {store.dataset}/{scenario}/"
                f"fold{fold}: {report['problems']}")
        log.info("%s leakage gate OK (%s mode, %d test units, 0 leaked) — %s/%s/fold%d",
                 self.name, report["mode"], report["n_test_units"],
                 store.dataset, scenario, fold)

    def _preflight(self, store: DatasetStore, scenario: str, fold: int) -> None:
        """Cheap correctness asserts before training (§6 Preflight)."""
        import anndata as ad

        adata = ad.read_h5ad(h5ad_path(store.dataset), backed="r")
        obs = adata.obs
        split_col = split_obs_column(scenario, fold)
        cond = obs["condition"].astype(str)
        split = obs[split_col].astype(str)
        problems: List[str] = []

        # B1: a usable control label present AND ≥1 control cell in train (basal).
        ctrl_mask = (cond.isin(["control", "ctrl", "ctrl_iegfp"])
                     | cond.str.contains("ctrl", case=False, na=False, regex=False))
        if not ctrl_mask.any():
            problems.append("no control cells (condition ∈ {control,ctrl,ctrl_iegfp})")
        elif not (ctrl_mask & (split == "train")).any():
            problems.append("no control cell labelled 'train' — basal starved")

        # U5: split tokens must include train/val/test.
        tokens = set(split.unique())
        if not {"train", "val", "test"}.issubset(tokens):
            problems.append(f"split {split_col!r} lacks train/val/test (has {sorted(tokens)[:6]})")

        # BUG-A: for pert-holdout regimes the derived condition lists must be a
        # clean partition (train ∩ test == ∅ at the perturbation level). Control
        # is excluded: the reference derivation keeps 'control' in whatever splits
        # its cells occupy (it lacks the 'ctrl' substring), so it can appear in
        # both lists — that is basal, not a held-out perturbation leak, and the
        # wrapper maps it to 'ctrl' in both splits.
        conds = _config.derive_conditions(obs, split_col)

        def _noctrl(cs):
            return {c for c in cs if c not in ("control", "ctrl", "ctrl_iegfp")
                    and "ctrl" not in c.lower()}

        overlap = _noctrl(conds["train"]) & _noctrl(conds["test"])
        if scenario not in leakage.BIN_AXIS_REGIMES and overlap:
            problems.append(f"{len(overlap)} perturbations in BOTH train and test "
                            f"(e.g. {sorted(overlap)[:5]})")

        # B8: X must be log1p, not raw counts (sample a slice).
        sl = adata[:200]
        Xs = sl.to_memory().X if adata.isbacked else sl.X
        Xs = Xs.toarray() if hasattr(Xs, "toarray") else np.asarray(Xs)
        if Xs.size:
            xmax = float(np.nanmax(Xs))
            if xmax > 30.0:  # log1p of counts stays ≪ 30; raw counts blow past it
                problems.append(f"X max {xmax:.1f} looks like raw counts, not log1p")

        if problems:
            raise RuntimeError(
                f"{self.name} PREFLIGHT FAILED for {store.dataset}/{scenario}/"
                f"fold{fold}: {problems}")

        # model-specific (e.g. GEARS gene2go coverage) — no silent skips.
        self._preflight_model_specific(store, obs, cond, split, conds)
        log.info("%s preflight OK — %s/%s/fold%d", self.name, store.dataset, scenario, fold)

    def _preflight_model_specific(self, store, obs, cond, split, conds) -> None:
        """Hook: subclass overrides (default no-op)."""

    # -------- Predictor interface --------

    def _expected_artifacts(self) -> List[str]:
        """Run-dir-relative paths that must ALL exist for the run to count as
        trained. Declared per model in its ``model.yaml``.

        Required, not defaulted: an empty list would make ``missing_artifacts``
        return nothing, so an EMPTY run dir would report itself trained and
        ``fit``'s post-condition would pass after a crash.
        """
        artifacts = self._recipe()["expected_artifacts"]
        if not artifacts:
            raise ValueError(
                f"{self.name}: model.yaml declares an empty `expected_artifacts` "
                f"— then nothing distinguishes a trained run from an empty dir")
        return list(artifacts)

    def _recipe_fingerprint(self) -> Dict:
        """The recipe fields that change what training produces.

        Everything the container is told to do, minus the plumbing (paths, mounts,
        the W&B project) which does not alter the model.
        """
        recipe = self._recipe()
        return {"model_key": recipe["model_key"],
                "hyperparameters": recipe["hyperparameters"],
                "extra_config": self.extra_config}

    def _prepare_run_dir(self, store: DatasetStore, scenario: str, fold: int) -> Path:
        """Train from a clean dir: GEARS reuses whatever cache it finds, so a
        stale `processed_data/` would be silently trained against. Only reached
        when the CLI has decided to (re)train — an already-trained fold is
        skipped before this point, so a concurrent run's dir is not wiped."""
        run_dir = self._run_dir(store, scenario, fold)
        if run_dir.exists():                          # B4: fresh dir, no stale cache
            shutil.rmtree(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def _train(self, store: DatasetStore, scenario: str, fold: int,
               run_dir: Path) -> None:
        cfg = _config.build_config(
            "train", dataset=store.dataset, scenario=scenario, fold=fold,
            model_name=self.model_yaml_key, seed=self.seed,
            hyperparameters=self.default_hyperparameters, extra_config=self.extra_config)
        if self.wandb_project:                        # opt-in W&B training logs (train only)
            cfg["wandb"] = True
            cfg["wandb_project"] = self.wandb_project
            cfg["wandb_run"] = f"{self.name}_{store.dataset}_{scenario}_fold{fold}_seed{self.seed}"
        _contract.validate_train_config(dict(cfg))    # host-side schema gate

        log.info("%s: training %s/%s/fold%d in %s", self.name, store.dataset,
                 scenario, fold, self._sif().name)
        runner.run_container(
            self._sif(), "train", cfg, data_dir=self._data_dir(store),
            output_dir=run_dir, code_binds=self._code_binds(), entry=self.entry,
            timeout=self.train_timeout)

    def _infer(self, store: DatasetStore, scenario: str, fold: int,
               run_dir: Path) -> np.ndarray:
        preds_h5ad = run_dir / "predictions.h5ad"
        if preds_h5ad.exists():
            preds_h5ad.unlink()
        cfg = _config.build_config(
            "predict", dataset=store.dataset, scenario=scenario, fold=fold,
            model_name=self.model_yaml_key, seed=self.seed,
            hyperparameters=self.default_hyperparameters, extra_config=self.extra_config,
            output_path_host=str(preds_h5ad))
        _contract.validate_predict_config(dict(cfg))

        log.info("%s: predicting %s/%s/fold%d", self.name, store.dataset, scenario, fold)
        runner.run_container(
            self._sif(), "predict", cfg, data_dir=self._data_dir(store),
            output_dir=run_dir, code_binds=self._code_binds(), entry=self.entry,
            timeout=self.predict_timeout)

        if not preds_h5ad.exists():
            raise RuntimeError(f"{self.name}: predict wrote no {preds_h5ad}")

        # `load_and_map` places the container's output onto the store's canonical
        # (bin, ko, gene) axes by NAME; the template then validates the shape.
        return tensor_map.load_and_map(
            preds_h5ad, store, scenario, fold, model_name=self.name)


# ===================================================================
# Concrete models
# ===================================================================


@register
class GEARSContainer(ContainerPredictor):
    name = "GEARS-ct"
    model_dir = "docker/gears"        # holds model.yaml (the run recipe)
    cell_aware = False                # earns cell-aware regimes only via the M1.5 gate
    scenarios = ["UnseenPert", "UnseenCombo"]   # single regime SSOT (Python, read by verify)
    has_drop_rule = True              # gene2go coverage may drop some perts (§1.3.6/U9)

    # Host copy of the gene2go pickle (same file the container reads via the bind
    # mount) — used here for the preflight coverage check.
    _GENE2GO_HOST = REPO_ROOT / "docker" / "gears" / "gene2go_all.pkl"
    _MIN_TARGET_COVERAGE = 0.5        # stop below this (likely a gene-ID mismatch)

    def _preflight_model_specific(self, store, obs, cond, split, conds) -> None:
        """GEARS-specific: perturbation targets must be covered by gene2go, else
        conditions are silently dropped (§1.3.6). Quantify coverage; STOP if it
        falls below a floor (CHEAT-6 — never shrug off drops)."""
        if not self._GENE2GO_HOST.exists():
            log.warning("%s: gene2go pickle not found at %s — GENE-COVERAGE CHECK "
                        "SKIPPED (not silently OK; vendor gene2go to enable).",
                        self.name, self._GENE2GO_HOST)
            return
        import pickle
        with open(self._GENE2GO_HOST, "rb") as fh:
            g2g = pickle.load(fh)
        go_genes = set(g2g.keys()) if isinstance(g2g, dict) else set(g2g)

        targets = set()
        for c in conds["train"] + conds["val"] + conds["test"]:
            if c in ("control", "ctrl", "ctrl_iegfp") or "ctrl" in c.lower():
                continue  # control is not a perturbation target
            targets.update(parse_target_genes(c))
        targets.discard("")
        if not targets:
            log.warning("%s: no perturbation targets parsed — skipping coverage", self.name)
            return
        covered = {g for g in targets if g in go_genes}
        frac = len(covered) / len(targets)
        dropped = sorted(targets - covered)
        log.info("%s gene2go coverage: %d/%d targets (%.1f%%) modeled; %d dropped%s",
                 self.name, len(covered), len(targets), 100 * frac, len(dropped),
                 f" (e.g. {dropped[:5]})" if dropped else "")
        if frac < self._MIN_TARGET_COVERAGE:
            raise RuntimeError(
                f"{self.name}: only {100*frac:.1f}% of perturbation targets are in "
                f"gene2go (< {100*self._MIN_TARGET_COVERAGE:.0f}% floor) — likely a "
                f"gene-ID space mismatch, not a benign drop. Dropped e.g. {dropped[:10]}")


__all__ = ["ContainerPredictor", "GEARSContainer", "COVARIATE_COLUMN"]


# ---------------------------------------------------------------------------
# Recipe schema
# ---------------------------------------------------------------------------

#: Keys a `model.yaml` MUST declare.
_REQUIRED_RECIPE_KEYS = frozenset({
    "model_key", "sif_path", "entry", "code_binds", "hyperparameters",
    "expected_artifacts", "train_timeout", "predict_timeout",
})
#: Keys it MAY declare.
_OPTIONAL_RECIPE_KEYS = frozenset({"wandb_project", "gene2go_path"})


def _validate_recipe(recipe: Dict, where: str) -> None:
    """Reject a malformed recipe at load time, not mid-run.

    Unknown keys are an ERROR rather than ignored: the recipe is meant to be the
    single source of truth for how a model is run, and a silently-dropped typo
    (`epochs` under the wrong nesting, say) would leave that claim false while
    the model trains with defaults nobody chose.
    """
    if not isinstance(recipe, dict):
        raise ValueError(f"{where}: model.yaml must be a mapping")
    keys = set(recipe)
    missing = _REQUIRED_RECIPE_KEYS - keys
    if missing:
        raise ValueError(f"{where}: model.yaml is missing required key(s) "
                         f"{sorted(missing)}")
    unknown = keys - _REQUIRED_RECIPE_KEYS - _OPTIONAL_RECIPE_KEYS
    if unknown:
        raise ValueError(
            f"{where}: model.yaml has unrecognised key(s) {sorted(unknown)}. "
            f"Known: {sorted(_REQUIRED_RECIPE_KEYS | _OPTIONAL_RECIPE_KEYS)}. "
            f"Add it to the schema in container_predictor.py if it is real.")
