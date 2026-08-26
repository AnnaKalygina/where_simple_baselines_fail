"""`TrainedPredictor` — the tier for models whose training costs GPU-hours.

Two things in this benchmark train expensively: `ContainerPredictor` runs the
training inside a `.sif`, `TorchPredictor` runs it in this process. They differ
completely in HOW they train, and agreed on everything around it — where the
checkpoint lives, what makes it reusable, when the leakage gate runs, how a run
is named in W&B, and what `predict` refuses to do. That agreement is this class.

It was extracted only once BOTH tiers existed and could be compared. A base
factored from a single implementation is a guess about what is shared; this one
is factored from evidence, which is why the split below is sharp: the container
keeps `_recipe`/`sif_path`/`entry`/`code_binds`/`run_container`, the torch tier
keeps its model and training loop, and everything either could have written
twice now lives here once.

The contract for subclasses is five hooks:

    _expected_artifacts()            what must exist for a run to count as trained
    _recipe_fingerprint()            what changes what training DOES
    _leakage_gate(...)               prove no held-out cell reaches training
    _train(store, sc, fold, run_dir) do the expensive thing; leave those artefacts
    _infer(store, sc, fold, run_dir)  -> (n_test_bins, n_test_kos, n_genes) deltas

`fit` and `predict` are templates: they run the gates, the ordering, and the
post-conditions, so a new expensively-trained model cannot forget them.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import socket
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from benchmark.config import checkpoint_dir
from benchmark.data_loader import DatasetStore
from benchmark.predictors._shared import _expected_output_shape
from benchmark.predictors.base import Predictor

log = logging.getLogger(__name__)

#: Written by `fit`, read by `is_trained`. Host-written on purpose: it must not
#: depend on what a particular container chooses to emit.
FINGERPRINT_FILE = "fingerprint.json"

#: Written while a run is in flight, removed when it finishes. Makes a crashed
#: run distinguishable from a running one — the two look identical on disk.
CLAIM_FILE = "RUNNING.json"


class TrainedPredictor(Predictor):
    """Base for predictors whose training is expensive and produces a checkpoint."""

    needs_training = True

    #: Seed for training. Folds may map to different seeds via `_seed_for`.
    seed: int = 42
    #: W&B project for training logs; None disables logging. HOW the run is
    #: logged differs per tier (the container passes it through its config, the
    #: torch tier calls wandb directly) — only the declaration is shared.
    wandb_project: Optional[str] = None
    #: Whether the model conditions on the cell-type covariate. Gates which
    #: regimes it may claim; promotion is a reviewed decision, not a default.
    cell_aware: bool = False

    # ------------------------------------------------------------------
    # Hooks — subclasses must implement
    # ------------------------------------------------------------------

    def _expected_artifacts(self) -> List[str]:
        """Run-dir-relative paths that must ALL exist for the run to be usable."""
        raise NotImplementedError

    def _leakage_gate(self, store: DatasetStore, scenario: str, fold: int) -> None:
        """Raise unless no held-out cell can reach training.

        Deliberately not defaulted to a no-op: a leak that reaches training is
        invisible downstream — every metric simply looks better — so a tier that
        forgot to check must fail loudly rather than inherit silence.
        """
        raise NotImplementedError

    def _train(self, store: DatasetStore, scenario: str, fold: int,
               run_dir: Path) -> Optional[Dict]:
        """Train, leaving `_expected_artifacts()` in `run_dir`.

        May return a small JSON-serialisable report (losses, best epoch, wall
        time); it is recorded alongside the fingerprint for provenance.
        """
        raise NotImplementedError

    def _infer(self, store: DatasetStore, scenario: str, fold: int,
               run_dir: Path) -> np.ndarray:
        """Predict from the checkpoint in `run_dir`.

        Must return `(n_test_bins, n_test_kos, n_genes)` in the store's canonical
        axis order; `predict` validates the shape before it is scored.
        """
        raise NotImplementedError

    def _recipe_fingerprint(self) -> Dict:
        """Everything that changes what training DOES — hyperparameters, schedule,
        architecture. Hashed into the run identity so a recipe edit invalidates
        the checkpoints it no longer describes.

        MANDATORY, despite reading like a detail: `_fingerprint` calls it
        unconditionally, so a subclass that omits it fails from `is_trained` —
        i.e. after the GPU time, not at definition.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Optional hooks
    # ------------------------------------------------------------------

    def _preflight(self, store: DatasetStore, scenario: str, fold: int) -> None:
        """Cheap checks before an expensive run (default: none)."""

    def _fingerprint(self, store: DatasetStore, scenario: str, fold: int) -> Dict:
        """The identity of a training run: what must still hold for its
        checkpoint to be reusable.

        Host-written, so every tier gets the same guarantees regardless of what
        its container emits. Each field answers a way a checkpoint silently goes
        stale:

        - ``gene_axis_sha``  the panel was rebuilt -> the network shape is wrong
        - ``split_sha``      the folds were regenerated -> cells the model TRAINED
          on are now test. This is the dangerous one: it leaves no crash behind,
          it just inflates the score, and a panel-preserving split refresh passes
          every other check.
        - ``recipe_sha``     hyperparameters changed -> the checkpoint answers a
          different question than the one being asked
        - ``seed``           a different draw
        `code_sha` is NOT here: it is advisory (see ``_ENFORCED``), so computing
        it would cost a `git` subprocess on every comparison — and this runs once
        per (dataset x scenario x fold) in any roster or sweep, through
        ``is_trained``. It is stamped in ``_write_fingerprint`` instead, which
        runs once per training run.
        """
        return {
            "predictor": self.name,
            "dataset": store.dataset,
            "scenario": scenario,
            "fold": int(fold),
            "seed": int(self._seed_for(fold)),
            "gene_axis_sha": _sha_strings(store.gene_names),
            "split_sha": _sha_split(store, scenario, fold),
            "recipe_sha": _sha_json(self._recipe_fingerprint()),
        }

    #: Fields whose disagreement makes a checkpoint unusable. `code_sha` is
    #: deliberately absent — provenance, not a gate.
    _ENFORCED = ("seed", "gene_axis_sha", "split_sha", "recipe_sha")

    def _write_fingerprint(self, store: DatasetStore, scenario: str, fold: int,
                           run_dir: Path, extra: Optional[Dict] = None) -> None:
        fp = self._fingerprint(store, scenario, fold)
        fp["code_sha"] = _git_head()          # provenance, stamped once per run
        if extra:
            fp["report"] = extra
        (run_dir / FINGERPRINT_FILE).write_text(json.dumps(fp, indent=2))

    def _staleness_reason(self, store: DatasetStore, scenario: str,
                          fold: int) -> Optional[str]:
        """Why a COMPLETE checkpoint is nonetheless not reusable, or None.

        Presence is not validity: a checkpoint whose artefacts all exist can
        still have been trained against a different panel, split or recipe.
        """
        path = self._run_dir(store, scenario, fold) / FINGERPRINT_FILE
        if not path.exists():
            return f"missing {FINGERPRINT_FILE} (trained before fingerprinting?)"
        try:
            stored = json.loads(path.read_text())
        except (OSError, ValueError) as e:
            return f"unreadable {FINGERPRINT_FILE} ({e})"

        current = self._fingerprint(store, scenario, fold)
        for key in self._ENFORCED:
            if stored.get(key) != current[key]:
                return (f"{_STALE_REASON.get(key, key)} changed since training "
                        f"({stored.get(key)!r} != {current[key]!r}); "
                        f"retrain with --force")
        return None

    def _clean_run_dir(self, run_dir: Path) -> None:
        """Remove whatever a previous run left that this one must not inherit.

        Default: nothing — a tier that overwrites its own artefacts needs no
        wipe. `fit` has already created the dir and taken the claim, so an
        override must leave `CLAIM_FILE` alone: it is what stops a second
        process from training into the same directory.
        """

    # ------------------------------------------------------------------
    # Paths / identity
    # ------------------------------------------------------------------

    def _run_dir(self, store: DatasetStore, scenario: str, fold: int) -> Path:
        """This fold's checkpoint dir, resolved centrally by `config`."""
        return checkpoint_dir(store.dataset, self.name, scenario, fold)

    def _seed_for(self, fold: int) -> int:
        """The seed this fold trains under. Override to derive it from the fold.

        Every caller must go through this rather than reading `self.seed` — the
        fingerprint records `_seed_for(fold)` as an ENFORCED field, so a tier
        that overrides it while some code path still passes `self.seed` would
        certify a checkpoint under a seed the model never saw. `TorchPredictor`
        does override it, so this is a live hazard, not a hypothetical one.
        """
        return self.seed

    def _check_scenario(self, scenario: str) -> None:
        if scenario not in self.scenarios:
            raise ValueError(
                f"{self.name} does not declare scenario {scenario!r} "
                f"(declares {self.scenarios})")

    # ------------------------------------------------------------------
    # Readiness
    # ------------------------------------------------------------------

    def missing_artifacts(self, store: DatasetStore, scenario: str,
                          fold: int) -> List[str]:
        run_dir = self._run_dir(store, scenario, fold)
        return [rel for rel in self._expected_artifacts()
                if not (run_dir / rel).exists()]

    def unusable_reason(self, store: DatasetStore, scenario: str,
                        fold: int) -> Optional[str]:
        """Why this checkpoint cannot be used, or None if it can."""
        missing = self.missing_artifacts(store, scenario, fold)
        if missing:
            return f"missing artefacts {missing}"
        return self._staleness_reason(store, scenario, fold)

    # ------------------------------------------------------------------

    def is_trained(self, store: DatasetStore, scenario: str, fold: int) -> bool:
        """Complete AND still valid for this data — never a bare `.exists()`."""
        return self.unusable_reason(store, scenario, fold) is None

    # ------------------------------------------------------------------
    # Templates
    # ------------------------------------------------------------------

    def fit(self, store: DatasetStore, scenario: str, fold: int,
            force: bool = False, steal: bool = False) -> None:
        """Gate, claim, train, then prove the run dir is actually usable.

        The post-condition matters: a crashed or half-written run must fail HERE,
        while the cause is on screen, not later at predict where the error would
        be far from what produced it.

        `force` decides whether to retrain a fold that is ALREADY trained; that
        question is settled by the caller (`run_pipeline`) before `fit` is
        reached, which is why it is unused here. `steal` is the separate, rarer
        decision to train into a directory another process currently claims.
        Keeping them apart matters: `--force` is routine, and it must not
        silently double as permission to collide with a running job.
        """
        self._check_scenario(scenario)
        self._preflight(store, scenario, fold)
        self._leakage_gate(store, scenario, fold)

        run_dir = self._run_dir(store, scenario, fold)
        run_dir.mkdir(parents=True, exist_ok=True)
        self._acquire_claim(run_dir, steal=steal)
        try:
            # Retract the previous run's certificate BEFORE touching its
            # artefacts. Without this a crash mid-retrain leaves the old
            # fingerprint standing over half-written weights: `is_trained` says
            # yes, the next `fit` skips it, and `predict` scores it. The torch
            # tier is where it bites — it writes `norm_stats.npz` only after
            # training, so the surviving stats belong to a DIFFERENT run than the
            # surviving weights. Doing it here rather than in a tier's wipe means
            # the guarantee is the template's, not one subclass's side effect.
            (run_dir / FINGERPRINT_FILE).unlink(missing_ok=True)
            self._clean_run_dir(run_dir)
            report = self._train(store, scenario, fold, run_dir)
            self._write_fingerprint(store, scenario, fold, run_dir, extra=report)
        except BaseException:
            # Leave the wreckage in place (it is often the only evidence) but say
            # how much it costs and how to reclaim it: on a near-full volume a
            # few abandoned runs are the difference between working and not.
            self._report_wreckage(run_dir)
            raise
        finally:
            (run_dir / CLAIM_FILE).unlink(missing_ok=True)

        reason = self.unusable_reason(store, scenario, fold)
        if reason:
            raise RuntimeError(
                f"{self.name}: training finished but left an unusable run dir at "
                f"{run_dir} — {reason}. Failing here rather than at predict, "
                f"where the cause would be far from the crash.")

    # ------------------------------------------------------------------
    # Concurrency
    # ------------------------------------------------------------------

    def _acquire_claim(self, run_dir: Path, *, steal: bool) -> None:
        """Take an exclusive claim on `run_dir`, or refuse to train into it.

        Atomic by construction: `O_CREAT | O_EXCL` either creates the file or
        fails, in one syscall. The previous check-then-write version had a window
        between them — two array tasks for the same (dataset, scenario, fold)
        both saw no claim, both proceeded, and the container tier's wipe then
        deleted the winner's claim on the way past. A multi-GB `rmtree` made that
        window wide enough to lose races in practice, not just in theory.

        Reclaimed automatically when the claim is unreadable or older than this
        model's timeout; overridden deliberately with `steal`. Never overridden
        by `--force`, which answers a different question.
        """
        claim_path = run_dir / CLAIM_FILE
        for attempt in ("first", "after-reclaim"):
            try:
                fd = os.open(claim_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            except FileExistsError:
                pass
            else:
                with os.fdopen(fd, "w") as fh:
                    json.dump({"host": socket.gethostname(), "pid": os.getpid(),
                               "started_at": time.time(), "predictor": self.name},
                              fh, indent=2)
                return
            if attempt == "after-reclaim":
                # Someone re-claimed between our unlink and our retry: they won.
                break
            reason = self._reclaimable(claim_path, steal=steal)
            if reason is None:
                break
            log.warning("%s: %s — reclaiming %s", self.name, reason, claim_path)
            claim_path.unlink(missing_ok=True)

        claim = _read_claim(claim_path) or {}
        age = time.time() - float(claim.get("started_at", 0)) if claim else 0.0
        who = (f"pid {claim.get('pid')} on {claim.get('host')} "
               f"({age / 60:.0f} min ago)" if claim else "another process")
        raise RuntimeError(
            f"{self.name}: {run_dir} is already claimed by {who}. Wait for it to "
            f"finish, or pass --steal-claim to train into that directory anyway "
            f"(which will corrupt the other run if it is still alive).")

    def _reclaimable(self, claim_path: Path, *, steal: bool) -> Optional[str]:
        """Why an existing claim may be taken over, or None to respect it."""
        claim = _read_claim(claim_path)
        if claim is None:
            return f"unreadable {CLAIM_FILE}"
        age = time.time() - float(claim.get("started_at", 0))
        if age > self._claim_ttl():
            return (f"stale claim from {claim.get('host')} "
                    f"(pid {claim.get('pid')}, {age / 3600:.1f} h old)")
        if steal:
            return (f"--steal-claim over a LIVE claim from {claim.get('host')} "
                    f"(pid {claim.get('pid')}, {age / 60:.0f} min old)")
        return None

    def _claim_ttl(self) -> float:
        """Seconds after which a claim is presumed abandoned."""
        return float(getattr(self, "train_timeout", 0) or 24 * 3600)

    def _report_wreckage(self, run_dir: Path) -> None:
        try:
            n_bytes = sum(f.stat().st_size for f in run_dir.rglob("*") if f.is_file())
        except OSError:
            return
        log.error("%s: training failed; %.1f GB of partial output left at %s "
                  "(delete it to reclaim, or re-run fit --force to overwrite)",
                  self.name, n_bytes / 2**30, run_dir)

    def predict(self, store: DatasetStore, scenario: str, fold: int) -> np.ndarray:
        """Load the checkpoint and infer. Never trains."""
        self._check_scenario(scenario)
        run_dir = self._run_dir(store, scenario, fold)
        reason = self.unusable_reason(store, scenario, fold)
        if reason:
            raise RuntimeError(
                f"{self.name}: no usable trained model at {run_dir} — {reason}. "
                f"Run `fit` (or `all`) first; predict never trains.")

        deltas = self._infer(store, scenario, fold, run_dir)

        split = store.split(scenario, fold)
        expected = _expected_output_shape(
            store, split.test_bin_indices_or_all(store.n_bins),
            split.test_ko_indices)
        if tuple(deltas.shape) != tuple(expected):
            raise RuntimeError(
                f"{self.name}: _infer returned {tuple(deltas.shape)}, expected "
                f"{tuple(expected)} for {store.dataset}/{scenario}/fold{fold} — "
                f"the prediction axes do not match the fold's test set.")
        if not np.isfinite(deltas).any():
            raise RuntimeError(
                f"{self.name}: no finite prediction anywhere in "
                f"{store.dataset}/{scenario}/fold{fold}. Either the model declined "
                f"every test cell (a drop rule covering the whole fold — check "
                f"perturbation coverage), or the condition/covariate mapping "
                f"failed and nothing matched. Both are worth stopping for.")
        return deltas


# ---------------------------------------------------------------------------
# Fingerprint helpers
# ---------------------------------------------------------------------------

_STALE_REASON = {
    "seed": "seed",
    "gene_axis_sha": "the gene panel",
    "split_sha": "the train/test split",
    "recipe_sha": "the training recipe",
}


def _read_claim(path: Path) -> Optional[Dict]:
    """Parse a claim file, or None if it is absent or unreadable."""
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _sha_strings(items) -> str:
    h = hashlib.sha256()
    for s in items:
        h.update(str(s).encode())
        h.update(b"\0")
    return h.hexdigest()[:16]


def _sha_json(obj) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _sha_split(store: DatasetStore, scenario: str, fold: int) -> str:
    """Hash of WHICH cells are trainable and which are held out.

    Hashes the two canonical masks rather than the index marginals, so every
    regime is covered by one expression — including UnseenPair, whose held-out
    unit is a scattered set of (bin, ko) pairs that no marginal describes.
    """
    split = store.split(scenario, fold)
    h = hashlib.sha256()
    for mask in (split.train_mask_2d(store.n_bins, store.n_kos),
                 split.test_mask_2d(store.n_bins, store.n_kos)):
        h.update(np.ascontiguousarray(mask, dtype=bool).tobytes())
        h.update(b"|")
    return h.hexdigest()[:16]


def _git_head() -> Optional[str]:
    """Short git SHA of the working tree, or None outside a repo. Advisory."""
    try:
        out = subprocess.run(
            ["git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or None
    except Exception:                                  # noqa: BLE001 - provenance only
        return None
