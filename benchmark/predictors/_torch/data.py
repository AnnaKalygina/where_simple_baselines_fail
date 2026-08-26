"""`DatasetStore` -> training batches for the transformer sweep.

One sample is one **(bin, ko) pair**. The model sees the control profile for that
bin plus a marker for which gene(s) were perturbed, and predicts the per-gene
delta:

    x_wt      (n_genes,)  control expression for the bin, standardised
    p         (n_genes,)  0 everywhere except the perturbed gene(s), where it
                          carries the dose
    gene_idx  (n_genes,)  0..n_genes-1, the gene-identity token indices
    delta     (n_genes,)  TARGET: store.all_deltas[bin, ko] / sigma

Leak-safety is the property this module exists to guarantee: the pair index is
built from `split.train_mask_2d(...)`, and the normalisation statistics are
computed from training pairs ONLY. Standardising with statistics that saw the
held-out cells would leak their scale into every training batch — a subtle leak
that no downstream metric would reveal.

torch is imported at module level here on purpose; see `_torch/__init__.py`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from benchmark.data_loader import DatasetStore, SplitInfo

log = logging.getLogger(__name__)

# Genes with (near-)zero spread carry no signal; clamp so standardising cannot
# divide by ~0 and manufacture enormous inputs.
_SIGMA_FLOOR = 1e-6


# ---------------------------------------------------------------------------
# Label parsing
# ---------------------------------------------------------------------------

def parse_ko_dose(ko_name: str) -> Tuple[List[str], float]:
    """Split a perturbation label into its target genes and its dose.

    `config.parse_target_genes` deliberately DISCARDS the dose (it answers "which
    genes"), but the `p` vector needs the magnitude too — ecoli_synthetic has an
    UnseenDose regime built from exactly this suffix:

        "aaea"            -> (["aaea"], 1.0)        # unqualified == full dose
        "aaea@0.25"       -> (["aaea"], 0.25)
        "aaer+cra"        -> (["aaer", "cra"], 1.0)
        "h-+ns@0.50"      -> (["h-", "ns"], 0.5)    # dose applies to the combo

    Kept local for now: this is the only consumer. If a second one appears it
    belongs beside `parse_target_genes` in `benchmark/config.py`.
    """
    text = str(ko_name)
    dose = 1.0
    if "@" in text:
        base, _, dose_token = text.rpartition("@")
        try:
            dose = float(dose_token)
        except ValueError:          # not a dose suffix after all (e.g. a name containing '@')
            base = text
    else:
        base = text
    genes = [g.strip() for g in base.replace(";", "+").split("+") if g.strip()]
    return genes, dose


# ---------------------------------------------------------------------------
# Normalisation (train-only)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class NormStats:
    """Per-gene centring/scaling, estimated from training pairs only.

    `x_wt` is standardised as `(ctrl - mu) / sigma` and the target is scaled as
    `delta / sigma` — the same sigma for both, so the target is expressed in
    units of that gene's expression spread and inputs and outputs stay on a
    comparable scale.
    """
    mu: np.ndarray       # (n_genes,)
    sigma: np.ndarray    # (n_genes,)
    n_samples: int

    def to_dict(self) -> Dict[str, np.ndarray]:
        return {"mu": self.mu, "sigma": self.sigma,
                "n_samples": np.asarray(self.n_samples)}

    @classmethod
    def from_dict(cls, d) -> "NormStats":
        return cls(mu=np.asarray(d["mu"], dtype=np.float32),
                   sigma=np.asarray(d["sigma"], dtype=np.float32),
                   n_samples=int(np.asarray(d["n_samples"])))


def compute_norm_stats(store: DatasetStore, split: SplitInfo) -> NormStats:
    """Per-gene mean/std over the expression the model may legitimately see.

    That is: the perturbed profiles of TRAIN (bin, ko) pairs, plus the control
    profile of each training bin (controls are never held out — they are the
    reference every delta is measured against).

    Accumulated bin-by-bin rather than by materialising a masked copy: for
    ecoli_synthetic `all_bulk` is already ~1 GB, and a boolean-masked copy would
    duplicate it.
    """
    n_bins, n_kos = store.n_bins, store.n_kos
    mask = split.train_mask_2d(n_bins, n_kos)
    all_bulk = store.all_bulk                        # (n_bins, n_kos, n_genes)

    s1 = np.zeros(store.n_genes, dtype=np.float64)
    s2 = np.zeros(store.n_genes, dtype=np.float64)
    n = 0
    for b in range(n_bins):
        sel = mask[b]
        if not sel.any():
            continue
        block = all_bulk[b][sel].astype(np.float64, copy=False)
        s1 += block.sum(axis=0)
        s2 += np.square(block).sum(axis=0)
        n += block.shape[0]

    ctrl = store.ctrl_bulk[split.train_bin_indices_or_all(n_bins)].astype(np.float64, copy=False)
    s1 += ctrl.sum(axis=0)
    s2 += np.square(ctrl).sum(axis=0)
    n += ctrl.shape[0]

    if n == 0:
        raise ValueError("no training pairs — cannot estimate normalisation")

    mu = s1 / n
    var = np.maximum(s2 / n - np.square(mu), 0.0)
    sigma = np.maximum(np.sqrt(var), _SIGMA_FLOOR)
    return NormStats(mu=mu.astype(np.float32), sigma=sigma.astype(np.float32),
                     n_samples=int(n))


# ---------------------------------------------------------------------------
# Perturbation encoding
# ---------------------------------------------------------------------------

def build_perturbation_matrix(store: DatasetStore) -> Tuple[np.ndarray, int]:
    """(n_kos, n_genes) float32: dose at each ko's target gene(s), else 0.

    Returns the matrix and the number of kos with NO representable target. Those
    are real: ecoli_synthetic carries perturbations of `h-` and `ns`, which are
    absent from `var_names`, so their marker is all-zero and the model cannot
    distinguish them from an unperturbed control. They are counted and reported
    rather than passed over silently.
    """
    gene_to_idx = store.gene_to_index
    p = np.zeros((store.n_kos, store.n_genes), dtype=np.float32)
    unrepresentable = 0
    for k, ko_name in enumerate(store.ko_names):
        genes, dose = parse_ko_dose(ko_name)
        hit = False
        for g in genes:
            idx = gene_to_idx.get(g)
            if idx is not None:
                p[k, idx] = dose
                hit = True
        if not hit:
            unrepresentable += 1
    return p, unrepresentable


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class PerturbationPairs(Dataset):
    """(bin, ko) pairs drawn from one mask, standardised with `NormStats`.

    Holds references to the store's arrays rather than copies; `all_deltas` is
    derived once by the caller and passed in so train/val/test datasets over the
    same store share it.
    """

    def __init__(
        self,
        pairs: np.ndarray,            # (M, 2) int64: (bin_idx, ko_idx)
        deltas: np.ndarray,           # (n_bins, n_kos, n_genes)
        ctrl_bulk: np.ndarray,        # (n_bins, n_genes)
        pert_matrix: np.ndarray,      # (n_kos, n_genes)
        stats: NormStats,
    ) -> None:
        self.pairs = np.asarray(pairs, dtype=np.int64)
        self._deltas = deltas
        self._ctrl = ctrl_bulk
        self._pert = pert_matrix
        self._stats = stats
        self._x_wt = ((ctrl_bulk - stats.mu) / stats.sigma).astype(np.float32)
        self._gene_idx = np.arange(deltas.shape[-1], dtype=np.int64)

    def __len__(self) -> int:
        return int(self.pairs.shape[0])

    @property
    def n_genes(self) -> int:
        return int(self._x_wt.shape[1])

    # -- batched accessors, used by inference ------------------------------
    # Inference walks an arbitrary pair list (the test cells) rather than this
    # dataset's own index, so it addresses the underlying rows directly instead
    # of going through __getitem__ one sample at a time.

    def x_wt_rows(self, pairs: np.ndarray) -> np.ndarray:
        return self._x_wt[np.asarray(pairs)[:, 0]]

    def p_rows(self, pairs: np.ndarray) -> np.ndarray:
        return self._pert[np.asarray(pairs)[:, 1]]

    def gene_idx_rows(self, n_rows: int) -> np.ndarray:
        return np.broadcast_to(self._gene_idx, (n_rows, self.n_genes)).copy()

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        b, k = self.pairs[i]
        delta = (self._deltas[b, k] / self._stats.sigma).astype(np.float32)
        return {
            "x_wt": torch.from_numpy(self._x_wt[b]),
            "p": torch.from_numpy(self._pert[k]),
            "gene_idx": torch.from_numpy(self._gene_idx),
            "delta": torch.from_numpy(delta),
            "bin_idx": torch.tensor(int(b), dtype=torch.long),
            "ko_idx": torch.tensor(int(k), dtype=torch.long),
        }


def pairs_from_mask(mask: np.ndarray) -> np.ndarray:
    """(M, 2) int64 array of the True cells of a (n_bins, n_kos) mask."""
    return np.argwhere(np.asarray(mask, dtype=bool)).astype(np.int64)


def val_mask(store: DatasetStore, split: SplitInfo) -> np.ndarray:
    """(n_bins, n_kos) mask of validation pairs.

    `SplitInfo` carries `val_ko_indices`, so early stopping and best-checkpoint
    selection run against a real held-out slice instead of the test set. Bins:
    the val bins when the regime splits bins, otherwise the training bins.
    """
    n_bins, n_kos = store.n_bins, store.n_kos
    m = np.zeros((n_bins, n_kos), dtype=bool)
    ko = np.asarray(split.val_ko_indices, dtype=np.int64)
    if ko.size == 0:
        return m
    bins = split.val_bin_indices
    if bins is None or len(bins) == 0:
        bins = split.train_bin_indices_or_all(n_bins)
    m[np.ix_(np.asarray(bins, dtype=np.int64), ko)] = True
    # Validation must be disjoint from BOTH other sets. Under rectangle regimes
    # that is automatic (val kos are disjoint from train and test kos), but
    # UnseenPair holds out a scattered pair set, so the val-ko rectangle can
    # overlap training pairs — and validating on cells the model trained on
    # silently disables early stopping and best-checkpoint selection.
    m &= ~split.test_mask_2d(n_bins, n_kos)
    m &= ~split.train_mask_2d(n_bins, n_kos)
    return m


def ko_sampling_weights(
    deltas: np.ndarray, pairs: np.ndarray, sigma: np.ndarray, alpha: float = 1.0,
) -> np.ndarray:
    """Per-SAMPLE weights ~ ||delta||_2 ** alpha, normalised to mean 1.

    Up-weights perturbations that actually move the transcriptome; without it the
    many near-silent knockouts dominate the gradient. Computed on the pairs given
    (i.e. training pairs), so it carries no information about held-out cells.
    """
    w = np.empty(pairs.shape[0], dtype=np.float64)
    for i, (b, k) in enumerate(pairs):
        w[i] = np.linalg.norm(deltas[b, k] / sigma)
    if alpha != 1.0:
        w = np.power(w, alpha)
    total = w.sum()
    if not np.isfinite(total) or total <= 0:
        return np.ones_like(w)
    return w * (len(w) / total)


# ---------------------------------------------------------------------------
# Loader assembly
# ---------------------------------------------------------------------------

@dataclass
class FoldData:
    train: PerturbationPairs
    val: Optional[PerturbationPairs]
    stats: NormStats
    pert_matrix: np.ndarray
    train_weights: np.ndarray
    unrepresentable_kos: int


def build_fold_data(
    store: DatasetStore, scenario: str, fold: int, alpha: float = 1.0,
) -> FoldData:
    """Assemble the train/val datasets for one (scenario, fold)."""
    split = store.split(scenario, fold)
    n_bins, n_kos = store.n_bins, store.n_kos

    stats = compute_norm_stats(store, split)
    pert, unrepresentable = build_perturbation_matrix(store)
    if unrepresentable:
        log.warning(
            "%s: %d/%d perturbation(s) have no target gene in the panel — their "
            "marker is all-zero and they are indistinguishable from control",
            store.dataset, unrepresentable, n_kos)

    deltas = store.all_deltas
    ctrl = store.ctrl_bulk

    train_pairs = pairs_from_mask(split.train_mask_2d(n_bins, n_kos))
    v_pairs = pairs_from_mask(val_mask(store, split))

    train_ds = PerturbationPairs(train_pairs, deltas, ctrl, pert, stats)
    val_ds = (PerturbationPairs(v_pairs, deltas, ctrl, pert, stats)
              if v_pairs.shape[0] else None)
    weights = ko_sampling_weights(deltas, train_pairs, stats.sigma, alpha=alpha)

    return FoldData(train=train_ds, val=val_ds, stats=stats, pert_matrix=pert,
                    train_weights=weights, unrepresentable_kos=unrepresentable)


def make_loaders(
    data: FoldData, batch_size: int, num_workers: int, seed: int,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """Weighted-sampling train loader + a deterministic val loader."""
    g = torch.Generator()
    g.manual_seed(int(seed))
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(data.train_weights, dtype=torch.double),
        num_samples=len(data.train), replacement=True, generator=g)
    train_loader = DataLoader(
        data.train, batch_size=batch_size, sampler=sampler,
        num_workers=num_workers, pin_memory=True, generator=g)
    val_loader = (
        DataLoader(data.val, batch_size=batch_size, shuffle=False,
                   num_workers=num_workers, pin_memory=True)
        if data.val is not None else None)
    return train_loader, val_loader


def inference_pairs(store: DatasetStore, scenario: str, fold: int) -> np.ndarray:
    """(M, 2) pairs the benchmark asks for at predict time: the TEST cells."""
    split = store.split(scenario, fold)
    return pairs_from_mask(split.test_mask_2d(store.n_bins, store.n_kos))
