"""`TorchPredictor` — expensively-trained models that train IN THIS PROCESS.

The second expensive-trained profile in the benchmark. `ContainerPredictor` runs
its training inside a `.sif`; this one runs a torch loop here, but needs the same
surrounding machinery: a checkpoint directory, a seed, W&B, validation-driven
stopping, and an honest `is_trained`.

That machinery is deliberately DUPLICATED from `ContainerPredictor` for now
rather than abstracted. A base extracted from a single implementation is a guess
about what is shared; with two working implementations it can be factored from
evidence instead. That extraction is the next step, not this one.

Import discipline: torch and everything under `_torch/` are imported INSIDE the
methods that need them. This module is imported whenever the predictor registry
is built — including in environments that have no torch — so a module-level
import would drop every transformer from the registry (or crash the CLI).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from benchmark.config import checkpoint_dir
from benchmark.data_loader import DatasetStore
from benchmark.predictors._shared import _expected_output_shape
from benchmark.predictors._torch import recipe
from benchmark.predictors.base import Predictor

log = logging.getLogger(__name__)

CHECKPOINT_FILE = "checkpoint_best.pt"
FINGERPRINT_FILE = "fingerprint.json"


class TorchPredictor(Predictor):
    """Base for in-process torch-trained predictors.

    Subclasses supply `name`, `scenarios`, and `architecture()`.
    """

    needs_training = True
    # Every test (bin, ko) gets a prediction — the model is defined for any
    # perturbation marker, including the all-zero one. No drop rule.
    has_drop_rule = False

    #: overridden per fold via `recipe.seed_for_fold`; kept as an attribute so a
    #: subclass or a test can pin it.
    seed: Optional[int] = None
    wandb_project: Optional[str] = None

    # ------------------------------------------------------------------
    # Subclass hook
    # ------------------------------------------------------------------

    def architecture(self, n_genes: int) -> Dict:
        """Model kwargs for this variant. Override or set `model_name`."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Paths / identity
    # ------------------------------------------------------------------

    def _run_dir(self, store: DatasetStore, scenario: str, fold: int) -> Path:
        return checkpoint_dir(store.dataset, self.name, scenario, fold)

    def _seed_for(self, fold: int) -> int:
        return self.seed if self.seed is not None else recipe.seed_for_fold(fold)

    def _fingerprint(self, store: DatasetStore, scenario: str, fold: int) -> Dict:
        """What the checkpoint must agree with to be reusable.

        Gene axis included by identity, not just by count: a dataset rebuild that
        changes the panel silently invalidates every earlier checkpoint, and the
        failure would otherwise surface as a shape error deep inside the model.
        """
        return {
            "predictor": self.name,
            "dataset": store.dataset,
            "scenario": scenario,
            "fold": int(fold),
            "seed": int(self._seed_for(fold)),
            "n_genes": int(store.n_genes),
            "gene_axis_sha": _sha_of_sequence(store.gene_names),
            "architecture": {k: v for k, v in
                             self.architecture(store.n_genes).items()},
        }

    # ------------------------------------------------------------------
    # Predictor interface
    # ------------------------------------------------------------------

    def unusable_reason(
        self, store: DatasetStore, scenario: str, fold: int,
    ) -> Optional[str]:
        """Why this checkpoint cannot be used, or None if it can."""
        run_dir = self._run_dir(store, scenario, fold)
        for rel in (CHECKPOINT_FILE, FINGERPRINT_FILE):
            if not (run_dir / rel).exists():
                return f"missing {rel}"
        try:
            stored = json.loads((run_dir / FINGERPRINT_FILE).read_text())
        except (OSError, ValueError) as e:
            return f"unreadable {FINGERPRINT_FILE} ({e})"

        current = self._fingerprint(store, scenario, fold)
        for key in ("n_genes", "gene_axis_sha", "seed", "architecture"):
            if stored.get(key) != current[key]:
                return (f"checkpoint was trained with a different {key} "
                        f"({stored.get(key)!r} != {current[key]!r}); retrain "
                        f"with --force")
        return None

    def is_trained(self, store: DatasetStore, scenario: str, fold: int) -> bool:
        return self.unusable_reason(store, scenario, fold) is None

    def fit(self, store: DatasetStore, scenario: str, fold: int) -> None:
        import torch
        from benchmark.predictors._torch import data as td
        from benchmark.predictors._torch.model import GeneTransformer
        from benchmark.predictors._torch import trainer as tr

        self._check_scenario(scenario)
        seed = self._seed_for(fold)
        run_dir = self._run_dir(store, scenario, fold)
        run_dir.mkdir(parents=True, exist_ok=True)

        tr.set_seed(seed)
        fold_data = td.build_fold_data(
            store, scenario, fold, alpha=float(recipe.SAMPLE_WEIGHT["alpha"]))
        self._assert_leak_safe(store, scenario, fold, fold_data)

        train_loader, val_loader = td.make_loaders(
            fold_data,
            batch_size=int(recipe.TRAINING["batch_size"]),
            num_workers=int(recipe.TRAINING["num_workers"]),
            seed=seed)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = GeneTransformer(num_genes=store.n_genes,
                                **self.architecture(store.n_genes))
        log.info("%s: training %s/%s/fold%d on %s (%d train / %d val pairs)",
                 self.name, store.dataset, scenario, fold, device,
                 len(fold_data.train), len(fold_data.val) if fold_data.val else 0)

        run = self._start_wandb(store, scenario, fold, seed)
        try:
            report = tr.train_model(
                model, train_loader, val_loader,
                training=recipe.TRAINING, device=device,
                checkpoint_path=run_dir / CHECKPOINT_FILE, seed=seed,
                wandb_run=run)
        finally:
            if run is not None:
                run.finish()

        # Normalisation constants are part of the trained state: predict must
        # rescale with the SAME sigma the targets were divided by.
        np.savez(str(run_dir / "norm_stats.npz"), **fold_data.stats.to_dict())
        fp = self._fingerprint(store, scenario, fold)
        fp["report"] = report.to_dict()
        (run_dir / FINGERPRINT_FILE).write_text(json.dumps(fp, indent=2))

        missing = self.unusable_reason(store, scenario, fold)
        if missing:
            raise RuntimeError(
                f"{self.name}: training finished but left an unusable run dir at "
                f"{run_dir} — {missing}. Failing here rather than at predict.")

    def predict(self, store: DatasetStore, scenario: str, fold: int) -> np.ndarray:
        import torch
        from benchmark.predictors._torch import data as td
        from benchmark.predictors._torch.model import GeneTransformer
        from benchmark.predictors._torch import trainer as tr

        self._check_scenario(scenario)
        run_dir = self._run_dir(store, scenario, fold)
        reason = self.unusable_reason(store, scenario, fold)
        if reason:
            raise RuntimeError(
                f"{self.name}: no usable trained model at {run_dir} — {reason}. "
                f"Run `fit` (or `all`) first; predict never trains.")

        stats = td.NormStats.from_dict(np.load(str(run_dir / "norm_stats.npz")))
        pert, _ = td.build_perturbation_matrix(store)
        split = store.split(scenario, fold)
        tbi = split.test_bin_indices_or_all(store.n_bins)
        tki = split.test_ko_indices

        # Predict the full (test_bin x test_ko) rectangle, matching every other
        # predictor: the metric layer selects what it scores.
        pairs = np.array([(b, k) for b in tbi for k in tki], dtype=np.int64)

        view = td.PerturbationPairs(pairs, store.all_deltas, store.ctrl_bulk,
                                    pert, stats)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = GeneTransformer(num_genes=store.n_genes,
                                **self.architecture(store.n_genes))
        blob = torch.load(str(run_dir / CHECKPOINT_FILE), map_location="cpu")
        model.load_state_dict(blob["state_dict"])

        scaled = tr.infer_pairs(model, view, pairs, device,
                                batch_size=int(recipe.TRAINING["batch_size"]))
        # Back to the benchmark's units: targets were divided by sigma.
        preds = scaled * stats.sigma[None, :]

        out = np.full(_expected_output_shape(store, tbi, tki), np.nan,
                      dtype=np.float32)
        out.reshape(-1, store.n_genes)[:] = preds
        return out

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _check_scenario(self, scenario: str) -> None:
        if scenario not in self.scenarios:
            raise ValueError(
                f"{self.name} does not declare scenario {scenario!r} "
                f"(declares {self.scenarios})")

    def _assert_leak_safe(self, store, scenario, fold, fold_data) -> None:
        """Host-side gate: nothing held out may reach training.

        `_torch.data` already builds the pair index from the train mask; this
        re-checks the result, because a leak that reaches training is invisible
        in every downstream metric — it just makes the model look good.
        """
        split = store.split(scenario, fold)
        test = split.test_mask_2d(store.n_bins, store.n_kos)
        pairs = fold_data.train.pairs
        leaked = int(test[pairs[:, 0], pairs[:, 1]].sum())
        if leaked:
            raise RuntimeError(
                f"{self.name}: leakage gate FAILED — {leaked} of {len(pairs)} "
                f"training pairs are held-out cells in "
                f"{store.dataset}/{scenario}/fold{fold}")
        log.info("%s leakage gate OK (%d train pairs, 0 leaked) — %s/%s/fold%d",
                 self.name, len(pairs), store.dataset, scenario, fold)

    def _start_wandb(self, store, scenario, fold, seed):
        if not self.wandb_project:
            return None
        try:
            import wandb
        except ImportError:
            log.warning("%s: wandb requested but not installed — training will "
                        "run unlogged", self.name)
            return None
        try:
            return wandb.init(
                project=self.wandb_project,
                name=f"{self.name}_{store.dataset}_{scenario}_fold{fold}_seed{seed}",
                config={"predictor": self.name, "dataset": store.dataset,
                        "scenario": scenario, "fold": fold, "seed": seed,
                        **recipe.TRAINING,
                        **self.architecture(store.n_genes)},
                reinit=True)
        except Exception as e:                       # noqa: BLE001
            # A logging backend being down must not lose GPU-hours of training.
            log.warning("%s: wandb init failed (%s) — training unlogged",
                        self.name, e)
            return None


def _sha_of_sequence(items: List[str]) -> str:
    import hashlib
    h = hashlib.sha256()
    for s in items:
        h.update(str(s).encode())
        h.update(b"\0")
    return h.hexdigest()[:16]
