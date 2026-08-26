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

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from benchmark.data_loader import DatasetStore
from benchmark.predictors._torch import recipe
from benchmark.predictors.trained import TrainedPredictor

log = logging.getLogger(__name__)

CHECKPOINT_FILE = "checkpoint_best.pt"
NORM_STATS_FILE = "norm_stats.npz"


class TorchPredictor(TrainedPredictor):
    """Base for in-process torch-trained predictors.

    Subclasses supply `name`, `scenarios`, and `architecture()`.
    """

    # A perturbation whose targets are absent from the panel gets an all-zero
    # marker, i.e. it is indistinguishable from control. Those are emitted as NaN
    # rather than guessed, which is a drop rule.
    has_drop_rule = True
    #: None => fold-derived seed (see `_seed_for`); set to pin one.
    seed: Optional[int] = None

    #: How long before a `RUNNING.json` from this tier is presumed abandoned.
    #: The container tier reuses its `train_timeout` for this, because there the
    #: timeout really does kill the run. Nothing kills an in-process torch run,
    #: so there is no timeout to borrow — inheriting the base class's 24 h would
    #: be a number nobody chose. 12 h is above the longest observed fold
    #: (~3 h for 12x1 on one GPU) with room for a slower card or a busy node.
    claim_ttl_seconds: float = 12 * 3600

    def _claim_ttl(self) -> float:
        return float(self.claim_ttl_seconds)


    # ------------------------------------------------------------------
    # Subclass hook
    # ------------------------------------------------------------------

    def architecture(self, n_genes: int) -> Dict:
        """Model kwargs for this variant. Override or set `model_name`."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Paths / identity
    # ------------------------------------------------------------------

    def _seed_for(self, fold: int) -> int:
        """Fold-derived unless pinned, so a fold means the same thing everywhere."""
        return self.seed if self.seed is not None else recipe.seed_for_fold(fold)

    def _recipe_fingerprint(self) -> Dict:
        """What changes what training produces: the shared schedule plus this
        variant's architecture."""
        return {"training": dict(recipe.TRAINING),
                "sample_weight": dict(recipe.SAMPLE_WEIGHT),
                "architecture": self.architecture(0)}

    # ------------------------------------------------------------------
    # Predictor interface
    # ------------------------------------------------------------------

    def _expected_artifacts(self) -> List[str]:
        return [CHECKPOINT_FILE, NORM_STATS_FILE]

    def _train(self, store: DatasetStore, scenario: str, fold: int,
               run_dir: Path) -> Dict:
        import torch
        from benchmark.predictors._torch import data as td
        from benchmark.predictors._torch.model import GeneTransformer
        from benchmark.predictors._torch import trainer as tr

        seed = self._seed_for(fold)
        tr.set_seed(seed)
        fold_data = td.build_fold_data(
            store, scenario, fold, alpha=float(recipe.SAMPLE_WEIGHT["alpha"]))

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
        np.savez(str(run_dir / NORM_STATS_FILE), **fold_data.stats.to_dict())
        return report.to_dict()

    def _infer(self, store: DatasetStore, scenario: str, fold: int,
               run_dir: Path) -> np.ndarray:
        import torch
        from benchmark.predictors._torch import data as td
        from benchmark.predictors._torch.model import GeneTransformer
        from benchmark.predictors._torch import trainer as tr

        stats = td.NormStats.from_dict(np.load(str(run_dir / NORM_STATS_FILE)))
        pert, _ = td.build_perturbation_matrix(store)
        split = store.split(scenario, fold)
        tbi = split.test_bin_indices_or_all(store.n_bins)
        tki = split.test_ko_indices

        # Predict the full (test_bin x test_ko) rectangle, matching every other
        # predictor: the metric layer selects what it scores.
        pairs = np.array([(b, k) for b in tbi for k in tki], dtype=np.int64)

        # No targets passed: inference reads only x_wt/p/gene_idx, and
        # `store.all_deltas` is a computed property that would materialise a
        # ~1 GB cube for nothing.
        view = td.PerturbationPairs(pairs, store.ctrl_bulk, pert, stats)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = GeneTransformer(num_genes=store.n_genes,
                                **self.architecture(store.n_genes))
        blob = torch.load(str(run_dir / CHECKPOINT_FILE), map_location="cpu")
        model.load_state_dict(blob["state_dict"])

        scaled = tr.infer_pairs(model, view, pairs, device,
                                batch_size=int(recipe.TRAINING["batch_size"]))
        # Back to the benchmark's units: targets were divided by sigma.
        preds = scaled * stats.sigma[None, :]

        # `pairs` was built as [(b, k) for b in tbi for k in tki], which is
        # exactly C order over (bin, ko) — reshape rather than index, and let the
        # shape check in the template catch any future reordering.
        out = preds.reshape(len(tbi), len(tki), store.n_genes).astype(np.float32)

        # Decline the perturbations the model cannot represent: an all-zero
        # marker makes them identical to control, so a number here would be
        # scored as if it meant something.
        blind = set(td.unrepresentable_kos(store).tolist())
        if blind:
            cols = [j for j, k in enumerate(tki) if int(k) in blind]
            if cols:
                out[:, cols, :] = np.nan
                log.warning("%s: %d test perturbation(s) have no target gene in "
                            "the panel — emitting NaN rather than a guess",
                            self.name, len(cols))
        return out

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _leakage_gate(self, store: DatasetStore, scenario: str, fold: int) -> None:
        """Prove no held-out (bin, ko) cell can reach training.

        Answered from the split masks alone, so it runs BEFORE any data is built
        or a GPU is touched. A leak that reaches training is invisible
        downstream — every metric simply looks better — so this fails loudly.
        """
        split = store.split(scenario, fold)
        train = np.argwhere(split.train_mask_2d(store.n_bins, store.n_kos))
        test = split.test_mask_2d(store.n_bins, store.n_kos)
        leaked = int(test[train[:, 0], train[:, 1]].sum()) if len(train) else 0
        if leaked:
            raise RuntimeError(
                f"{self.name}: LEAKAGE GATE FAILED — {leaked} of {len(train)} "
                f"training pairs are held-out cells in "
                f"{store.dataset}/{scenario}/fold{fold}")
        log.info("%s leakage gate OK (%d train pairs, 0 leaked) — %s/%s/fold%d",
                 self.name, len(train), store.dataset, scenario, fold)

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
