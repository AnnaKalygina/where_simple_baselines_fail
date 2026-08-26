"""In-process training loop for `TorchPredictor`.

Written fresh against this repo's conventions (seeding, checkpointing, W&B,
validation-driven stopping) rather than ported, so the training story matches the
rest of the benchmark. The recipe VALUES it implements live in `recipe.py`.

The loop is deliberately plain: MSE on the scaled delta, AdamW, linear warmup
then ReduceLROnPlateau, gradient clipping, early stopping, and best-checkpoint
selection on VALIDATION loss. The checkpoint that gets written is the best one,
never the last — a run that overfits after epoch 12 must not silently ship
epoch 40's weights.
"""

from __future__ import annotations

import logging
import math
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

log = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Seed every RNG the training loop touches.

    We do NOT set `torch.use_deterministic_algorithms(True)`: some CUDA kernels
    have no deterministic implementation and it would raise rather than slow
    down. Residual nondeterminism on GPU is therefore expected, which is why
    seed-to-seed spread is reported as a band rather than treated as exact.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@dataclass
class TrainReport:
    """What happened during training — persisted next to the weights."""
    best_epoch: int = -1
    best_val_loss: float = float("inf")
    final_epoch: int = -1
    stopped_early: bool = False
    epochs_run: int = 0
    train_loss_history: List[float] = field(default_factory=list)
    val_loss_history: List[float] = field(default_factory=list)
    seconds: float = 0.0
    n_parameters: int = 0

    def to_dict(self) -> Dict:
        return {
            "best_epoch": self.best_epoch,
            "best_val_loss": self.best_val_loss,
            "final_epoch": self.final_epoch,
            "stopped_early": self.stopped_early,
            "epochs_run": self.epochs_run,
            "train_loss_history": self.train_loss_history,
            "val_loss_history": self.val_loss_history,
            "seconds": self.seconds,
            "n_parameters": self.n_parameters,
        }


def _lr_scale_for_warmup(epoch: int, warmup_epochs: int) -> float:
    """Linear warmup over the first `warmup_epochs`, then 1.0."""
    if warmup_epochs <= 0 or epoch >= warmup_epochs:
        return 1.0
    return float(epoch + 1) / float(warmup_epochs)


def _run_epoch(
    model: nn.Module, loader, device: torch.device, loss_fn,
    optimizer=None, grad_clip: Optional[float] = None,
) -> float:
    """One pass. Trains when `optimizer` is given, else evaluates."""
    training = optimizer is not None
    model.train(training)
    total, n = 0.0, 0
    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for batch in loader:
            x_wt = batch["x_wt"].to(device, non_blocking=True)
            p = batch["p"].to(device, non_blocking=True)
            gene_idx = batch["gene_idx"].to(device, non_blocking=True)
            target = batch["delta"].to(device, non_blocking=True)

            pred = model(x_wt, p, gene_idx)
            loss = loss_fn(pred, target)

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            bs = x_wt.shape[0]
            total += float(loss.item()) * bs
            n += bs
    return total / max(n, 1)


def train_model(
    model: nn.Module,
    train_loader,
    val_loader,
    *,
    training: Dict,
    device: torch.device,
    checkpoint_path,
    seed: int,
    wandb_run=None,
    log_every: int = 1,
) -> TrainReport:
    """Fit `model`, writing the BEST checkpoint to `checkpoint_path`.

    Returns a `TrainReport`. Validation drives both the LR schedule and stopping;
    with no validation loader the loop still runs but cannot early-stop, and says
    so rather than quietly selecting on the training loss.
    """
    from benchmark.predictors._torch.model import count_parameters

    set_seed(seed)
    model.to(device)

    epochs = int(training["epochs"])
    warmup = int(training.get("warmup_epochs", 0))
    base_lr = float(training["lr"])
    patience = int(training.get("patience", 10))
    min_epochs = int(training.get("min_epochs", 0))
    grad_clip = training.get("grad_clip_norm")

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=base_lr,
        weight_decay=float(training.get("weight_decay", 0.0)))
    plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min",
        factor=float(training.get("lr_factor", 0.5)),
        patience=int(training.get("lr_patience", 5)))

    report = TrainReport(n_parameters=count_parameters(model))
    if val_loader is None:
        log.warning("no validation loader — early stopping and best-checkpoint "
                    "selection are disabled for this run")

    best_state = None
    since_improved = 0
    t0 = time.time()

    for epoch in range(epochs):
        # Warmup multiplies the CURRENT lr, which ReduceLROnPlateau may have
        # already lowered; apply it to the base and let the plateau factor ride
        # on top by reading the optimizer's own lr after warmup ends.
        if epoch < warmup:
            for group in optimizer.param_groups:
                group["lr"] = base_lr * _lr_scale_for_warmup(epoch, warmup)

        train_loss = _run_epoch(model, train_loader, device, loss_fn,
                                optimizer=optimizer, grad_clip=grad_clip)
        report.train_loss_history.append(train_loss)

        val_loss = (_run_epoch(model, val_loader, device, loss_fn)
                    if val_loader is not None else float("nan"))
        report.val_loss_history.append(val_loss)
        report.final_epoch = epoch
        report.epochs_run = epoch + 1

        if epoch >= warmup and val_loader is not None and math.isfinite(val_loss):
            plateau.step(val_loss)

        improved = val_loader is not None and val_loss < report.best_val_loss
        if improved:
            report.best_val_loss = float(val_loss)
            report.best_epoch = epoch
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            since_improved = 0
        else:
            since_improved += 1

        if wandb_run is not None:
            wandb_run.log({"epoch": epoch, "train_loss": train_loss,
                           "val_loss": val_loss,
                           "lr": optimizer.param_groups[0]["lr"]})
        if epoch % log_every == 0:
            log.info("epoch %3d/%d  train %.6f  val %.6f  lr %.2e%s",
                     epoch, epochs, train_loss, val_loss,
                     optimizer.param_groups[0]["lr"], "  *" if improved else "")

        if (val_loader is not None and epoch + 1 >= min_epochs
                and since_improved >= patience):
            report.stopped_early = True
            log.info("early stop at epoch %d (no val improvement in %d epochs; "
                     "best was epoch %d)", epoch, since_improved, report.best_epoch)
            break

    report.seconds = time.time() - t0

    # Ship the BEST weights, not the last ones. Without a validation set there is
    # no principled "best", so the final state is what there is — recorded as
    # best_epoch = final_epoch so the checkpoint never misrepresents itself.
    if best_state is None:
        best_state = {k: v.detach().cpu().clone()
                      for k, v in model.state_dict().items()}
        report.best_epoch = report.final_epoch
    model.load_state_dict(best_state)

    torch.save({"state_dict": best_state, "report": report.to_dict()},
               str(checkpoint_path))
    log.info("wrote %s (best epoch %d, val %.6f, %.1fs, %d params)",
             checkpoint_path, report.best_epoch, report.best_val_loss,
             report.seconds, report.n_parameters)
    return report


@torch.no_grad()
def infer_pairs(
    model: nn.Module, dataset, pairs: np.ndarray, device: torch.device,
    batch_size: int = 32,
) -> np.ndarray:
    """Predict the SCALED delta for each (bin, ko) pair, in `pairs` order.

    Returns (len(pairs), n_genes). Rescaling back to the benchmark's units is the
    caller's job — it owns the sigma the targets were divided by.
    """
    model.to(device).eval()
    out = np.empty((len(pairs), dataset.n_genes), dtype=np.float32)
    for start in range(0, len(pairs), batch_size):
        chunk = pairs[start:start + batch_size]
        x_wt = torch.from_numpy(dataset.x_wt_rows(chunk)).to(device)
        p = torch.from_numpy(dataset.p_rows(chunk)).to(device)
        gene_idx = torch.from_numpy(dataset.gene_idx_rows(len(chunk))).to(device)
        out[start:start + len(chunk)] = model(x_wt, p, gene_idx).cpu().numpy()
    return out
