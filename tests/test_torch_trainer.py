"""Training-correctness gates for the torch tier.

The headline test is `test_overfits_a_tiny_batch`: a model that cannot drive the
loss to ~zero on a handful of samples it sees over and over is broken —
mis-wired inputs, a detached graph, a frozen parameter — and no sweep result
from it means anything. It is the cheapest test that distinguishes "training"
from "appearing to train", so it runs before any real run is believed.

Kept off the GPU and tiny (a 12-gene panel, a 2-layer model) so it belongs in
the normal test suite rather than in a job script.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from benchmark.predictors._torch import data as td
from benchmark.predictors._torch import trainer as tr
from benchmark.predictors._torch.model import GeneTransformer, count_parameters


def _tiny_loader(n_genes=12, n_pairs=4, seed=0):
    """A handful of fixed (x_wt, p) -> delta examples, repeated."""
    rng = np.random.default_rng(seed)
    x_wt = rng.normal(size=(n_pairs, n_genes)).astype(np.float32)
    p = np.zeros((n_pairs, n_genes), dtype=np.float32)
    for i in range(n_pairs):
        p[i, i % n_genes] = 1.0
    delta = rng.normal(size=(n_pairs, n_genes)).astype(np.float32)
    gene_idx = np.broadcast_to(np.arange(n_genes), (n_pairs, n_genes)).copy()

    batch = {
        "x_wt": torch.from_numpy(x_wt),
        "p": torch.from_numpy(p),
        "gene_idx": torch.from_numpy(gene_idx.astype(np.int64)),
        "delta": torch.from_numpy(delta),
    }
    return [batch]          # one fixed batch, iterated every epoch


def _small_model(n_genes=12, **over):
    kwargs = dict(d_model=32, nhead=4, ff_mult=2, dropout=0.0,
                  num_genes=n_genes, layers=2, num_steps=1)
    kwargs.update(over)
    return GeneTransformer(**kwargs)


def test_overfits_a_tiny_batch(tmp_path):
    """The gate: memorising 4 samples must be easy. If it is not, training is broken."""
    loader = _tiny_loader()
    model = _small_model()
    report = tr.train_model(
        model, loader, loader,
        training={"epochs": 300, "lr": 3e-3, "weight_decay": 0.0,
                  "warmup_epochs": 0, "patience": 10_000, "min_epochs": 0,
                  "grad_clip_norm": 5.0, "lr_patience": 50, "lr_factor": 0.5},
        device=torch.device("cpu"),
        checkpoint_path=tmp_path / "ckpt.pt", seed=0)

    first, last = report.train_loss_history[0], report.train_loss_history[-1]
    assert last < first * 0.05, (
        f"failed to overfit 4 samples: loss {first:.4f} -> {last:.4f}. "
        f"Training is not actually learning.")
    assert last < 0.05


def test_all_parameters_receive_gradient():
    """Every parameter must be reachable — a silently frozen block would still
    let the loss fall, just to a worse floor."""
    loader = _tiny_loader()
    model = _small_model()
    batch = loader[0]
    loss = torch.nn.MSELoss()(
        model(batch["x_wt"], batch["p"], batch["gene_idx"]), batch["delta"])
    loss.backward()

    dead = [n for n, p in model.named_parameters()
            if p.requires_grad and (p.grad is None or torch.all(p.grad == 0))]
    assert not dead, f"parameters received no gradient: {dead}"


def test_checkpoint_holds_the_best_epoch_not_the_last(tmp_path):
    """Validation selects the checkpoint; a later, worse epoch must not ship."""
    train_loader = _tiny_loader(seed=0)
    # A val batch drawn from different noise, so val loss rises as the model
    # memorises the training batch — the best epoch is early, not last.
    val_loader = _tiny_loader(seed=99)

    model = _small_model()
    ckpt = tmp_path / "ckpt.pt"
    report = tr.train_model(
        model, train_loader, val_loader,
        training={"epochs": 60, "lr": 3e-3, "weight_decay": 0.0,
                  "warmup_epochs": 0, "patience": 10_000, "min_epochs": 0,
                  "grad_clip_norm": 5.0, "lr_patience": 50, "lr_factor": 0.5},
        device=torch.device("cpu"), checkpoint_path=ckpt, seed=0)

    assert report.best_epoch < report.final_epoch, (
        "expected the best val epoch to precede the last epoch")
    blob = torch.load(str(ckpt), map_location="cpu")
    assert blob["report"]["best_epoch"] == report.best_epoch
    # The weights on disk are the best ones, and the in-memory model was
    # restored to match them.
    for k, v in model.state_dict().items():
        assert torch.allclose(v.cpu(), blob["state_dict"][k])


def test_early_stopping_triggers_and_is_recorded(tmp_path):
    loader = _tiny_loader()
    model = _small_model()
    report = tr.train_model(
        model, loader, _tiny_loader(seed=99),
        training={"epochs": 200, "lr": 3e-3, "weight_decay": 0.0,
                  "warmup_epochs": 0, "patience": 3, "min_epochs": 2,
                  "grad_clip_norm": 5.0, "lr_patience": 50, "lr_factor": 0.5},
        device=torch.device("cpu"), checkpoint_path=tmp_path / "c.pt", seed=0)
    assert report.stopped_early
    assert report.epochs_run < 200


def test_training_without_validation_reports_honestly(tmp_path):
    """No val set: the loop still runs, but must not pretend to have selected."""
    loader = _tiny_loader()
    model = _small_model()
    report = tr.train_model(
        model, loader, None,
        training={"epochs": 5, "lr": 1e-3, "weight_decay": 0.0,
                  "warmup_epochs": 0, "patience": 10, "min_epochs": 0,
                  "grad_clip_norm": 5.0, "lr_patience": 5, "lr_factor": 0.5},
        device=torch.device("cpu"), checkpoint_path=tmp_path / "c.pt", seed=0)
    assert not report.stopped_early
    assert report.best_epoch == report.final_epoch      # nothing to select on
    assert np.isnan(report.val_loss_history[-1])


def test_seeding_makes_a_run_reproducible_on_cpu(tmp_path):
    def run(seed):
        tr.set_seed(seed)
        m = _small_model()
        tr.train_model(
            m, _tiny_loader(), None,
            training={"epochs": 3, "lr": 1e-3, "weight_decay": 0.0,
                      "warmup_epochs": 0, "patience": 10, "min_epochs": 0,
                      "grad_clip_norm": 5.0, "lr_patience": 5, "lr_factor": 0.5},
            device=torch.device("cpu"),
            checkpoint_path=tmp_path / f"c{seed}.pt", seed=seed)
        return torch.cat([p.detach().flatten() for p in m.parameters()])

    a, b, c = run(0), run(0), run(1)
    assert torch.allclose(a, b), "same seed should give the same weights on CPU"
    assert not torch.allclose(a, c), "different seeds should diverge"


def test_kxn_factorisations_differ_in_size_but_share_effective_depth():
    """The sweep's premise: 6x2 and 12x1 compute depth 12, but 6x2 ties weights
    so it has roughly half the parameters."""
    from benchmark.predictors._torch import recipe
    n_genes = 64
    p = {}
    for name in ("transformer_12x1", "transformer_6x2", "transformer_2x1"):
        m = GeneTransformer(num_genes=n_genes, **recipe.architecture(name))
        p[name] = count_parameters(m)
    assert recipe.effective_depth("transformer_12x1") == 12
    assert recipe.effective_depth("transformer_6x2") == 12
    assert p["transformer_6x2"] < p["transformer_12x1"]
    assert p["transformer_2x1"] < p["transformer_6x2"]
