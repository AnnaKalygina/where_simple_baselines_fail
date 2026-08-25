"""Helpers shared across predictor implementations.

Kept here (not duplicated per predictor, and not hosted inside any one predictor
module) so a single fix propagates and the dependency arrows stay clean:

    config / data_loader  ->  base / _shared  ->  {analytical, controls,
                                                   dl_adapter, learned}

Everything here is predictor-agnostic: split-index resolution, output-shape
computation, masked reductions over a (n_bins, n_kos) grid, combo-pair
resolution, and target-gene resolution. None of it imports ``base.Predictor``,
so a module can pull in a shape helper without dragging in a concrete predictor
set. ``DatasetStore`` / ``SplitInfo`` are referenced for typing only (the bodies
duck-type ``store``), so this module has no runtime dependency on
``data_loader`` and stays a leaf.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Dict, List, Tuple

import numpy as np

from benchmark.config import parse_target_genes

if TYPE_CHECKING:  # type-only; avoids a runtime import (and any cycle risk)
    from benchmark.data_loader import DatasetStore

log = logging.getLogger(__name__)


def resolve_combo_pairs(store, ko_indices, *, label: str, scenario: str,
                        fold: int) -> List[Tuple[int, int, int]]:
    """Return ``(local_pos, single_idx_A, single_idx_B)`` for every RESOLVABLE
    2-way combo in ``ko_indices``.

    Permissive (matches upstream Perturbation-Models-Outperform-Baselines, which
    logs missing perts and skips them): a KO that is not a 2-way ``A+B`` combo, or
    whose single ``A``/``B`` is absent from ``store.ko_names``, is dropped with a
    warning. Callers leave the unresolved KO slices as NaN so they are skipped from
    evaluation rather than scored as a zero prediction.

    ``label`` is only used to prefix the warning (e.g. a predictor name or
    ``"GlobalEpistasis.fit"``).
    """
    ko_name_to_idx = {k: i for i, k in enumerate(store.ko_names)}
    pairs: List[Tuple[int, int, int]] = []
    not_a_combo: List[str] = []
    high_order: List[str] = []
    missing_singles: List[str] = []
    for local_pos, ki in enumerate(ko_indices):
        name = store.ko_names[int(ki)]
        if "+" not in name:
            not_a_combo.append(name)
            continue
        parts = sorted(name.split("+"))
        if len(parts) != 2:
            high_order.append(name)
            continue
        a, b = parts
        ai, bi = ko_name_to_idx.get(a), ko_name_to_idx.get(b)
        if ai is None or bi is None:
            missing_singles.append(name)
            continue
        pairs.append((local_pos, ai, bi))

    n_skip = len(not_a_combo) + len(high_order) + len(missing_singles)
    if n_skip:
        log.warning(
            "%s: skipping %d/%d unresolvable KO(s) for %s/%s/fold%d "
            "(not-combo=%s, 3+-way=%s, missing-single=%s)",
            label, n_skip, len(ko_indices), store.dataset, scenario, fold,
            not_a_combo[:3], high_order[:3], missing_singles[:3],
        )
    return pairs


def _expected_output_shape(
    store: "DatasetStore", test_bin_idx: np.ndarray, test_ko_idx: np.ndarray,
) -> Tuple[int, int, int]:
    return (len(test_bin_idx), len(test_ko_idx), store.n_genes)


# --- masked reductions over a (n_bins, n_kos) TRAIN mask -------------------
# Every training predictor sources its train cells from `split.train_mask_2d()`,
# which is the explicit per-(bin, ko) mask for the non-rectangular scenarios
# (UnseenPair, UnseenBoth) and the train_bins × train_kos rectangle otherwise.
# Reducing over that mask (rather than slicing the marginal rectangle) is what
# keeps the held-out corner out of training for UnseenBoth; for the rectangle
# scenarios the mask IS the rectangle, so the result is unchanged.


def masked_mean(deltas: np.ndarray, mask: np.ndarray, over: str) -> np.ndarray:
    """Mean of `deltas` (n_bins, n_kos, n_genes) over the True cells of a
    (n_bins, n_kos) bool `mask`, reduced along the requested axis:

      over="per_bin" -> (n_bins, n_genes): per bin, mean over kos with mask[bin]=True
                        (bins with no True kos -> NaN).
      over="per_ko"  -> (n_kos, n_genes):  per ko, mean over bins with mask[:,ko]=True
                        (kos with no True bins -> NaN).
      over="all"     -> (n_genes,):        mean over all True (bin, ko) cells
                        (no True cells -> zeros).
    """
    # A mask-weighted mean: Σ(Δ·mask) / Σ(mask) along the reduced axis. Vectorized
    # (no per-bin/per-ko Python loop); empty rows -> NaN so eval skips them, and a
    # totally-empty global mask -> zeros (predict no effect).
    mask = np.asarray(mask, dtype=bool)
    if over == "all":
        sel = deltas[mask]  # (n_true, n_genes), True cells in row-major order
        if sel.shape[0] == 0:
            return np.zeros(deltas.shape[-1], dtype=np.float32)
        return sel.mean(axis=0).astype(np.float32)
    if over == "per_bin":
        axis = 1            # reduce over kos -> (n_bins, n_genes)
    elif over == "per_ko":
        axis = 0            # reduce over bins -> (n_kos, n_genes)
    else:
        raise ValueError(f"masked_mean: unknown over={over!r} (expected per_bin|per_ko|all)")
    num = (deltas * mask[..., None]).sum(axis=axis)   # masked sum along the axis
    den = mask.sum(axis=axis)                          # cells contributing per row
    out = np.full(num.shape, np.nan, dtype=np.float32)  # empty row -> NaN (skipped)
    nz = den > 0
    out[nz] = (num[nz] / den[nz, None]).astype(np.float32)
    return out


def resolve_target_gene_indices(
    ko_name: str, gene_to_idx: Dict[str, int], *, strict: bool = True,
) -> List[int]:
    """Parse target gene names and resolve them to var_names positions.

    With strict=True (default): raises ValueError if the KO parses to no
    targets, or if any parsed target is missing from gene_to_idx. Target-based
    baselines that demand a fully-aligned panel use this mode.

    With strict=False: returns only the indices of tokens that DO resolve.
    Returns an empty list if no token resolves. The caller decides what to
    do — TargetZero / TargetScaling treat empty as "predict zero for this KO"
    (matching AE 2025 behavior for unresolvable targets).
    """
    parts = parse_target_genes(ko_name)
    if not parts:
        if strict:
            raise ValueError(
                f"KO {ko_name!r} parses to zero target gene tokens; target-based "
                f"baselines require at least one parsable target."
            )
        return []
    missing = [p for p in parts if p not in gene_to_idx]
    if missing and strict:
        raise ValueError(
            f"KO {ko_name!r}: target gene(s) {missing} not in gene_names "
            f"(parsed {parts}). Re-run preprocessing with these genes "
            f"force-included into the panel."
        )
    return [gene_to_idx[p] for p in parts if p in gene_to_idx]
