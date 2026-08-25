"""Per-row metric primitives (torch).

A library of literature-implemented metric kernels. The active benchmark
selects ~17 of these via ``meta_metrics.METRICS_CONFIG``; the rest are
kept available for future benchmark revisions to pull in.

Conventions
===========
Per-row metrics accept ``(K, G)`` tensors and return ``(K,)`` tensors,
where ``K = n_perturbations`` and ``G = n_genes``. Cross-row metrics
accept ``(K, G)`` and return a scalar.

Currently selected in METRICS_CONFIG
====================================
Error           : sq_err, abs_err
Reducers        : row_masked_mean, row_weighted_mean
Row correlation : row_pearson, row_r2, row_r2_centered, row_ccc, row_spearman
Row direction   : row_frac_correct_direction, row_fold_change_gap
Discrimination  : pds, cosine_rank, effect_size_auroc
Distributional  : var_ratio_log_error
DEG enrichment  : row_gsea_up, row_gsea_down

Library (available, not currently selected)
===========================================
sign_agreement, row_mean                            (simple helpers)
build_de_mask, build_subtree_mask                   (mask builders)
e_distance, e_distance_pca                          (need single-cell samples)
gi_score_r2, gi_precision_at_k, gi_tpr_fdp          (genetic-interaction scores)
compute_subtree_metrics,
gene_role_stratified_mse, fanout_stratified_mse     (training-time helpers)

To re-select any library kernel for the active benchmark, add a
``MetricSpec`` entry to ``meta_metrics.METRICS_CONFIG`` and a dispatch
branch in ``meta_metrics._compute_per_pert_via_kernels``.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Set

import torch
from torch import Tensor


# =====================================================================
# A. Per-element error primitives
# =====================================================================

def sq_err(pred: Tensor, target: Tensor) -> Tensor:
    """Squared error per element. (K, G) -> (K, G)."""
    return (pred - target).pow(2)


def abs_err(pred: Tensor, target: Tensor) -> Tensor:
    """Absolute error per element. (K, G) -> (K, G)."""
    return (pred - target).abs()


def sign_agreement(pred: Tensor, target: Tensor) -> Tensor:
    """Sign agreement per element. (K, G) -> (K, G) float."""
    return (pred.sign() == target.sign()).float()


# =====================================================================
# B. Row-wise reducers
# =====================================================================

def row_mean(x: Tensor) -> Tensor:
    """Unmasked mean per row. (K, G) -> (K,)."""
    return x.mean(dim=-1)


def row_masked_mean(x: Tensor, gene_mask: Tensor, min_count: int = 1) -> Tensor:
    """Masked mean per row. (K, G) -> (K,).

    Genes where gene_mask is False contribute zero. Rows with fewer than
    min_count masked genes return 0.
    """
    x_m = torch.where(gene_mask, x, torch.zeros_like(x))
    denom = gene_mask.sum(-1).clamp(min=max(min_count, 1)).float()
    return x_m.sum(-1) / denom


def row_weighted_mean(x: Tensor, weights: Tensor, gene_mask: Optional[Tensor] = None) -> Tensor:
    """Weighted mean per row. (K, G) -> (K,).

    Args:
        x: Values to average.
        weights: Per-element weights. Broadcast-expanded if needed.
        gene_mask: Optional boolean mask; False genes get zero weight.
    """
    if weights.dim() < x.dim():
        w = weights.expand_as(x).clone()
    else:
        w = weights.clone()
    if gene_mask is not None:
        w = torch.where(gene_mask, w, torch.zeros_like(w))
    wsum = w.sum(-1).clamp(min=1e-8)
    return (w * x).sum(-1) / wsum


# =====================================================================
# G. Mask builders
# =====================================================================

def build_de_mask(target: Tensor, threshold: float = 0.5) -> Tensor:
    """(K, G) bool mask: True for genes with |target| > threshold."""
    return target.abs() > threshold


def build_subtree_mask(
    ko_genes: List[str],
    gene_names: List[str],
    descendant_map: Dict[str, Set[str]],
) -> Tensor:
    """(K, G) bool mask: True for genes in the subtree of each KO."""
    name_to_idx = {n: i for i, n in enumerate(gene_names)}
    K, G = len(ko_genes), len(gene_names)
    mask = torch.zeros(K, G, dtype=torch.bool)
    for k, ko in enumerate(ko_genes):
        for desc in descendant_map.get(ko, set()):
            if desc in name_to_idx:
                mask[k, name_to_idx[desc]] = True
    return mask


# =====================================================================
# C. Non-decomposable row metrics
# =====================================================================

def row_pearson(
    pred: Tensor, target: Tensor,
    gene_mask: Optional[Tensor] = None,
    weights: Optional[Tensor] = None,
    eps: float = 1e-8,
) -> Tensor:
    """Pearson correlation per row. (K, G) -> (K,)."""
    if weights is None:
        w = torch.ones_like(pred)
    elif weights.dim() < pred.dim():
        w = weights.expand_as(pred).clone()
    else:
        w = weights.clone()
    if gene_mask is not None:
        w = torch.where(gene_mask, w, torch.zeros_like(w))
    w_sum = w.sum(dim=-1, keepdim=True).clamp(min=eps)
    p_mu = (w * pred).sum(dim=-1, keepdim=True) / w_sum
    t_mu = (w * target).sum(dim=-1, keepdim=True) / w_sum
    pc = pred - p_mu
    tc = target - t_mu
    num = (w * pc * tc).sum(dim=-1)
    den = torch.sqrt((w * pc.pow(2)).sum(dim=-1) * (w * tc.pow(2)).sum(dim=-1)).clamp(min=eps)
    return num / den


def row_r2(
    pred: Tensor, target: Tensor,
    gene_mask: Optional[Tensor] = None,
    weights: Optional[Tensor] = None,
    eps: float = 1e-8,
) -> Tensor:
    """R-squared per row. (K, G) -> (K,)."""
    if weights is None:
        w = torch.ones_like(pred)
    elif weights.dim() < pred.dim():
        w = weights.expand_as(pred).clone()
    else:
        w = weights.clone()
    if gene_mask is not None:
        w = torch.where(gene_mask, w, torch.zeros_like(w))
    ss_res = (w * (target - pred).pow(2)).sum(dim=-1)
    ss_tot = (w * target.pow(2)).sum(dim=-1)
    return 1.0 - ss_res / ss_tot.clamp(min=eps)


def row_r2_centered(
    pred: Tensor, target: Tensor,
    gene_mask: Optional[Tensor] = None,
    weights: Optional[Tensor] = None,
    eps: float = 1e-8,
) -> Tensor:
    """Standard (centered) R-squared per row. (K, G) -> (K,).

    Uses ss_tot = sum(w * (y - y_bar)^2) instead of sum(w * y^2), and clamps the
    result to >= -1.0. This matches Miller et al.'s `sklearn.metrics.r2_score`
    (centered, `max(..., -1.0)`); it is the formula the headline `"r2"` metric
    uses (see meta_metrics dispatch).
    """
    if weights is None:
        w = torch.ones_like(pred)
    elif weights.dim() < pred.dim():
        w = weights.expand_as(pred).clone()
    else:
        w = weights.clone()
    if gene_mask is not None:
        w = torch.where(gene_mask, w, torch.zeros_like(w))
    w_sum = w.sum(dim=-1, keepdim=True).clamp(min=eps)
    t_mean = (w * target).sum(dim=-1, keepdim=True) / w_sum
    ss_res = (w * (target - pred).pow(2)).sum(dim=-1)
    ss_tot = (w * (target - t_mean).pow(2)).sum(dim=-1)
    return (1.0 - ss_res / ss_tot.clamp(min=eps)).clamp(min=-1.0)


def row_ccc(
    pred: Tensor, target: Tensor,
    gene_mask: Optional[Tensor] = None,
    weights: Optional[Tensor] = None,
    eps: float = 1e-8,
) -> Tensor:
    """Lin's Concordance Correlation Coefficient per row. (K, G) -> (K,).

    CCC = 2 * cov(pred, target) / (var_pred + var_target + (mu_pred - mu_target)^2)
    """
    if weights is None:
        w = torch.ones_like(pred)
    elif weights.dim() < pred.dim():
        w = weights.expand_as(pred).clone()
    else:
        w = weights.clone()
    if gene_mask is not None:
        w = torch.where(gene_mask, w, torch.zeros_like(w))
    w_sum = w.sum(dim=-1, keepdim=True).clamp(min=eps)
    p_mu = (w * pred).sum(dim=-1, keepdim=True) / w_sum
    t_mu = (w * target).sum(dim=-1, keepdim=True) / w_sum
    pc = pred - p_mu
    tc = target - t_mu
    w1 = w_sum.squeeze(-1)
    var_p = (w * pc.pow(2)).sum(dim=-1) / w1
    var_t = (w * tc.pow(2)).sum(dim=-1) / w1
    cov = (w * pc * tc).sum(dim=-1) / w1
    mean_diff_sq = (p_mu.squeeze(-1) - t_mu.squeeze(-1)).pow(2)
    return 2.0 * cov / (var_p + var_t + mean_diff_sq + eps)


def _midrank(x: Tensor) -> Tensor:
    """Rank with midrank averaging for ties (like scipy.stats.rankdata).

    For each row, assigns each element the average of the positions it would
    occupy in a sorted array.  Ties get the same (averaged) rank, avoiding the
    spurious correlations produced by argsort-based ranking on constant or
    near-constant data.

    Fully vectorized -- no Python loops over rows or columns.

    Input:  (..., G)
    Output: (..., G)  ranks in [0, G-1] (0-based, fractional for ties)
    """
    shape = x.shape
    x_flat = x.reshape(-1, shape[-1])
    n_rows, n = x_flat.shape

    sorted_vals, sorted_idx = x_flat.sort(-1)

    # Base ranks: 0, 1, ..., n-1
    base_ranks = torch.arange(n, device=x.device, dtype=x.dtype).unsqueeze(0).expand(n_rows, -1)

    # Detect tie boundaries: where consecutive sorted values differ
    # not_tie[i] = True means sorted_vals[i] != sorted_vals[i-1] (new group starts)
    not_tie = torch.ones(n_rows, n, device=x.device, dtype=torch.bool)
    not_tie[:, 1:] = sorted_vals[:, 1:] != sorted_vals[:, :-1]

    # Group ID per element via cumsum of boundary flags (0-based)
    group_id = not_tie.long().cumsum(-1) - 1  # (n_rows, n)

    # For each group, compute sum of ranks and count, then average
    n_groups = group_id.max().item() + 1
    # Use scatter_add for vectorized group aggregation
    rank_sum = torch.zeros(n_rows, n_groups, device=x.device, dtype=x.dtype)
    group_count = torch.zeros(n_rows, n_groups, device=x.device, dtype=x.dtype)
    rank_sum.scatter_add_(-1, group_id, base_ranks)
    group_count.scatter_add_(-1, group_id, torch.ones_like(base_ranks))

    # Average rank per group
    avg_rank = rank_sum / group_count.clamp(min=1)

    # Gather back: each element gets the average rank of its group
    midranks_sorted = avg_rank.gather(-1, group_id)

    # Scatter back to original positions
    result = torch.empty_like(x_flat)
    result.scatter_(-1, sorted_idx, midranks_sorted)
    return result.reshape(shape)


def _masked_variance(x: Tensor, mask: Tensor) -> Tensor:
    """Per-row variance over masked entries only."""
    wf = mask.float()
    n = wf.sum(-1).clamp(min=1.0)
    xm = (x * wf).sum(-1) / n
    xc = x - xm.unsqueeze(-1)
    return ((xc ** 2) * wf).sum(-1) / n


def row_spearman(
    pred: Tensor, target: Tensor,
    gene_mask: Optional[Tensor] = None,
    weights: Optional[Tensor] = None,
    eps: float = 1e-8,
) -> Tensor:
    """Spearman rank correlation per row (midrank for ties). (K, G) -> (K,).

    When gene_mask is provided, masked positions are set to -inf before
    ranking so they cluster at rank 0 and do not interfere with the
    correlation among unmasked genes. Weights are passed through to
    row_pearson on the ranked values.
    """
    if gene_mask is not None:
        fill = torch.finfo(pred.dtype).min
        p_filled = torch.where(gene_mask, pred, torch.full_like(pred, fill))
        t_filled = torch.where(gene_mask, target, torch.full_like(target, fill))
        r = row_pearson(_midrank(p_filled), _midrank(t_filled), gene_mask=gene_mask, weights=weights)
    else:
        r = row_pearson(_midrank(pred), _midrank(target), weights=weights)
    # Constant rows (zero variance) -> correlation undefined; return 0
    pred_const = pred.std(-1) < eps
    tgt_const = target.std(-1) < eps
    r = torch.where(pred_const | tgt_const, torch.zeros_like(r), r)
    return r


def row_frac_correct_direction(
    pred: Tensor, target: Tensor,
    gene_mask: Optional[Tensor] = None, min_n_genes: int = 0,
) -> Tensor:
    """Fraction of genes with correct sign direction per row. (K, G) -> (K,).

    Only genes where ``target != 0`` are considered (genes with zero
    ground-truth delta carry no directional information). Rows whose
    effective gene support (after masking + nonzero filter) is below
    ``min_n_genes`` return NaN.
    """
    nonzero = target != 0
    if gene_mask is None:
        mask = nonzero
    else:
        mask = gene_mask & nonzero
    correct = (pred.sign() == target.sign()) & mask
    denom = mask.sum(-1).float()
    value = correct.sum(-1).float() / denom.clamp(min=1)
    nan = torch.full_like(value, float("nan"))
    return torch.where(denom >= min_n_genes, value, nan)



def row_fold_change_gap(
    pred: Tensor, target: Tensor, gene_mask: Optional[Tensor] = None,
) -> Tensor:
    """Mean absolute gap by magnitude quartile bins. (K, G) -> (K,)."""
    K, G = target.shape
    abs_t = target.abs()
    ae = (pred - target).abs()
    if G < 4:
        if gene_mask is not None:
            ae = torch.where(gene_mask, ae, torch.zeros_like(ae))
            denom = gene_mask.sum(-1).clamp(min=1).float()
            return ae.sum(-1) / denom
        return ae.mean(-1)
    sorted_abs, _ = abs_t.sort(-1)
    q_idx = [int(G * p) for p in [0.25, 0.5, 0.75]]
    q25 = sorted_abs[:, q_idx[0]].unsqueeze(-1)
    q50 = sorted_abs[:, q_idx[1]].unsqueeze(-1)
    q75 = sorted_abs[:, q_idx[2]].unsqueeze(-1)
    bins = (abs_t >= q25).long() + (abs_t >= q50).long() + (abs_t >= q75).long()
    result = torch.zeros(K, device=pred.device, dtype=pred.dtype)
    for b in range(4):
        b_mask = bins == b
        if gene_mask is not None:
            b_mask = b_mask & gene_mask
        b_ae = torch.where(b_mask, ae, torch.zeros_like(ae))
        b_count = b_mask.sum(-1).clamp(min=1).float()
        result = result + b_ae.sum(-1) / b_count
    return result / 4


# =====================================================================
# D. Discrimination (cross-row)  (K, G) -> float
# =====================================================================

def pds(
    pred: Tensor, target: Tensor,
    metric: str = "cosine", max_kos: int = 200, seed: int = 42,
) -> float:
    """Perturbation discrimination score.

    For each perturbation, finds the closest ground-truth profile.  PDS is
    the fraction of perturbations where the closest match is the correct one.

    Args:
        pred, target: (K, G) tensors.
        metric: ``'l1'``, ``'l2'``, ``'cosine'``, or ``'sign_cosine'``.
        max_kos: subsample if K > max_kos.
        seed: random seed for subsampling.
    """
    K = pred.shape[0]
    if K <= 1:
        return 1.0

    if K > max_kos:
        gen = torch.Generator()
        gen.manual_seed(seed)
        idx = torch.randperm(K, generator=gen)[:max_kos]
        pred = pred[idx]
        target = target[idx]
        K = max_kos

    if metric == "l1":
        dist_matrix = (pred.unsqueeze(1) - target.unsqueeze(0)).abs().sum(-1)
    elif metric == "l2":
        pred_sq = (pred ** 2).sum(dim=1)
        true_sq = (target ** 2).sum(dim=1)
        cross = pred @ target.T
        dist_sq = pred_sq[:, None] + true_sq[None, :] - 2 * cross
        dist_matrix = dist_sq.clamp(min=0).sqrt()
    elif metric == "cosine":
        pred_norm = pred / (pred.norm(dim=1, keepdim=True) + 1e-8)
        target_norm = target / (target.norm(dim=1, keepdim=True) + 1e-8)
        dist_matrix = 1.0 - pred_norm @ target_norm.T
    elif metric == "sign_cosine":
        s_pred = pred.sign()
        s_target = target.sign()
        s_pred_norm = s_pred / (s_pred.norm(dim=1, keepdim=True) + 1e-8)
        s_target_norm = s_target / (s_target.norm(dim=1, keepdim=True) + 1e-8)
        dist_matrix = 1.0 - s_pred_norm @ s_target_norm.T
    else:
        raise ValueError(
            f"Unknown metric: {metric!r}. Use 'l1', 'l2', 'cosine', or 'sign_cosine'."
        )

    nearest = torch.argmin(dist_matrix, dim=1)
    correct = (nearest == torch.arange(K, device=nearest.device)).sum().item()
    return correct / K


# Above this many perturbations, the dense (K, K) cosine-similarity matrix is too
# large to materialize (e.g. xatlas_orion ~18k perts → ~1.3 GB at float32), so
# cosine_rank switches to a memory-safe row-chunked accumulation. Every current
# dataset except xatlas_orion has K below this, so they keep the exact dense path.
_COSINE_RANK_DENSE_MAX_K = 4096


def cosine_rank(
    pred: Tensor, target: Tensor, chunk: int = 1024,
) -> Tensor:
    """Per-perturbation cosine rank (PerturbBench approach).  (K, G) -> (K,).

    For each perturbation j, computes cosine similarity of ALL predictions
    against target[j]. The midrank of the correct prediction (pred[j])
    among all predictions is the score.

    Uses midrank tie-breaking: constant predictors (Zero, MoP) get rank ≈ 0.5.

    Lower is better: 0 = correct pred is most similar, 0.5 = random, 1 = worst.

    For K <= `_COSINE_RANK_DENSE_MAX_K` this is the exact dense computation (full
    (K, K) matrix). For larger K it accumulates the same per-target comparison
    counts over row-CHUNKS of the predictions (peak memory O(chunk·K) instead of
    O(K²)) so the metric is computable on very large screens without OOM. (The
    chunked path can differ from dense at the ~1e-4 float32-noise level near rank
    ties; it only triggers for K > 4096, i.e. xatlas-scale.)
    """
    K = pred.shape[0]
    if K <= 1:
        return torch.zeros(K, device=pred.device)

    pred_norm = pred / (pred.norm(dim=1, keepdim=True) + 1e-8)
    target_norm = target / (target.norm(dim=1, keepdim=True) + 1e-8)

    if K <= _COSINE_RANK_DENSE_MAX_K:
        # Exact dense path (bit-for-bit identical to the original implementation).
        sim = pred_norm @ target_norm.T  # (K, K) — sim[i,j] = cos(pred[i], target[j])
        diag = sim.diag()
        n_better = (sim > diag.unsqueeze(0)).sum(dim=0).float()
        n_equal = (sim == diag.unsqueeze(0)).sum(dim=0).float()
        midrank = n_better + (n_equal - 1.0) / 2.0
        return midrank / (K - 1)

    # Memory-safe chunked path for xatlas-scale K. diag is taken from the matmul
    # block diagonals so the self-match still registers in n_equal.
    diag = torch.empty(K, device=pred.device)
    for s in range(0, K, chunk):
        e = min(s + chunk, K)
        diag[s:e] = (pred_norm[s:e] @ target_norm[s:e].T).diagonal()
    n_better = torch.zeros(K, device=pred.device)
    n_equal = torch.zeros(K, device=pred.device)
    for s in range(0, K, chunk):
        sim_blk = pred_norm[s:s + chunk] @ target_norm.T  # (b, K)
        n_better += (sim_blk > diag.unsqueeze(0)).sum(dim=0).float()
        n_equal += (sim_blk == diag.unsqueeze(0)).sum(dim=0).float()
    midrank = n_better + (n_equal - 1.0) / 2.0
    return midrank / (K - 1)


def effect_size_auroc(
    pred: Tensor, target: Tensor, threshold: float = 0.5,
) -> Tensor:
    """AUROC for detecting perturbation effects.  (K, G) -> (K,).

    Tie-aware Mann-Whitney U / rank AUC using midranks on ``|pred|``.
    """
    p_score = pred.abs()
    t_label = (target.abs() > threshold).float()
    n_pos = t_label.sum(-1)
    n_neg = p_score.shape[1] - n_pos
    ranks = _midrank(p_score)
    sum_rank_pos = (ranks * t_label).sum(-1)
    # 0-based ranks: U-statistic equivalent to 1-based rank-sum formula
    u_stat = sum_rank_pos - n_pos * (n_pos - 1.0) / 2.0
    denom = (n_pos * n_neg).clamp(min=1e-8)
    auc = u_stat / denom
    degenerate = (n_pos == 0) | (n_neg == 0)
    return torch.where(degenerate, torch.full_like(auc, 0.5), auc)


# =====================================================================
# C2. Signed unweighted GSEA (enrichment of DEGs in predicted ranking)
# =====================================================================

def _gsea_enrichment_score(
    ranked_indices: Tensor, gene_set_mask: Tensor, min_set_size: int = 5,
) -> Tensor:
    """Vectorized KS-like enrichment score. (K, G) indices + (K, G) bool -> (K,).

    Walks the ranked list: hit → +1/m, miss → −1/(N−m). Returns max of
    the running sum (ES). NaN for rows with fewer than min_set_size hits.
    """
    K, G = ranked_indices.shape
    hits = gene_set_mask.gather(1, ranked_indices)          # (K, G) bool in rank order
    m = hits.sum(dim=1, keepdim=True).float()               # (K, 1)
    steps = torch.where(hits, 1.0 / m.clamp(min=1),
                        -1.0 / (G - m).clamp(min=1))        # (K, G)
    es = steps.cumsum(dim=1).max(dim=1).values               # (K,)
    return torch.where(m.squeeze(1) >= min_set_size, es,
                       torch.full_like(es, float('nan')))


def row_gsea_up(
    pred: Tensor, target: Tensor, deg_mask: Tensor, min_set_size: int = 5,
) -> Tensor:
    """GSEA enrichment of upregulated DEGs at the top of predicted ranking. (K, G) -> (K,)."""
    s_up = deg_mask & (target > 0)
    ranked = pred.argsort(dim=1, descending=True)
    return _gsea_enrichment_score(ranked, s_up, min_set_size)


def row_gsea_down(
    pred: Tensor, target: Tensor, deg_mask: Tensor, min_set_size: int = 5,
) -> Tensor:
    """GSEA enrichment of downregulated DEGs at the bottom of predicted ranking. (K, G) -> (K,)."""
    s_down = deg_mask & (target < 0)
    ranked = (-pred).argsort(dim=1, descending=True)
    return _gsea_enrichment_score(ranked, s_down, min_set_size)


# =====================================================================
# E. Distributional (cross-row)
# =====================================================================

def var_ratio_log_error(
    preds: Tensor, targets: Tensor,
    eps: float = 1e-8, var_floor: float = 1e-4,
) -> float:
    """Median ``|log(var_pred / var_true)|`` over genes with ``var_true`` above floor.

    Lower is better; 0 means median log-ratio is identity. Monotonic in
    deviation from unit variance ratio (unlike raw ratio as a
    higher-is-better score).
    """
    pred_var = preds.var(dim=0)
    target_var = targets.var(dim=0)
    mask = target_var >= var_floor
    if mask.sum() == 0:
        return 0.0
    ratio = pred_var[mask] / (target_var[mask] + eps)
    log_err = (ratio + eps).log().abs()
    return log_err.median().item()


def e_distance(pred: Tensor, target: Tensor) -> float:
    """Energy distance between predicted and true perturbation profiles.

    E-distance = 2*E[||X-Y||] - E[||X-X'||] - E[||Y-Y'||].
    """
    K = pred.shape[0]
    if K <= 1:
        return float((pred - target).norm().item())
    cross = torch.cdist(pred.float(), target.float(), p=2).mean()
    self_pred = torch.cdist(pred.float(), pred.float(), p=2).mean()
    self_target = torch.cdist(target.float(), target.float(), p=2).mean()
    return float((2 * cross - self_pred - self_target).item())


def e_distance_pca(
    pred: Tensor, target: Tensor, n_components: int = 50,
    pca_mean: Optional[Tensor] = None,
    pca_Vh: Optional[Tensor] = None,
) -> float:
    """Energy distance in PCA space (top-``n_components``).

    If ``pca_mean`` and ``pca_Vh`` are provided, projects both ``pred`` and
    ``target`` with that fixed basis (e.g. fit on ground truth only).
    Otherwise fits PCA on ``cat(pred, target)`` (legacy behavior).
    """
    if pca_mean is not None and pca_Vh is not None:
        mean = pca_mean
        k = int(
            min(
                n_components,
                int(pca_Vh.shape[0]),
                pred.shape[1],
                pred.shape[0] + target.shape[0],
            )
        )
        if k < 1:
            return 0.0
        pred_c = pred.float() - mean
        tgt_c = target.float() - mean
        V = pca_Vh[:k].to(pred.device)
        proj_p = pred_c @ V.T
        proj_t = tgt_c @ V.T
        return e_distance(proj_p, proj_t)

    combined = torch.cat([pred.float(), target.float()], dim=0)
    n_comp = min(n_components, combined.shape[0], combined.shape[1])
    if n_comp < 1:
        return 0.0
    mean = combined.mean(dim=0)
    centered = combined - mean
    _U, _S, Vh = torch.linalg.svd(centered, full_matrices=False)
    proj = centered @ Vh[:n_comp].T
    K = pred.shape[0]
    return e_distance(proj[:K], proj[K:])


# =====================================================================
# F. Genetic interaction
# =====================================================================

def gi_score_r2(pred_gi: Tensor, true_gi: Tensor) -> float:
    """R-squared of predicted genetic interaction scores."""
    return float(row_r2(pred_gi.unsqueeze(0), true_gi.unsqueeze(0)).item())


def gi_precision_at_k(
    pred_gi: Tensor, true_gi: Tensor, k: int = 10,
) -> float:
    """Precision@k for genetic interaction ranking."""
    k = min(k, pred_gi.shape[0])
    if k == 0:
        return 0.0
    _, p_idx = pred_gi.abs().topk(k)
    _, t_idx = true_gi.abs().topk(k)
    p_set = set(p_idx.tolist())
    t_set = set(t_idx.tolist())
    return len(p_set & t_set) / k


def gi_tpr_fdp(
    pred_gi: Tensor, true_gi: Tensor, threshold: Optional[float] = None,
) -> float:
    """True positive rate for genetic interaction detection."""
    t = true_gi.abs()
    p = pred_gi.abs()
    if threshold is None:
        nonzero = t[t > 0]
        threshold = float(nonzero.median().item()) if nonzero.numel() > 0 else 0.5
    true_sig = t > threshold
    pred_sig = p > threshold
    if true_sig.sum() == 0:
        return 0.0
    return float((true_sig & pred_sig).sum().item() / true_sig.sum().item())


# =====================================================================
# Training orchestration helpers
# =====================================================================

def compute_subtree_metrics(
    pred: Tensor, target: Tensor, p: Tensor,
    gene_names: List[str], descendant_map: Dict[str, set],
) -> Dict[str, float]:
    """In-subtree vs out-of-subtree MAE and MSE.

    Parameters
    ----------
    pred, target : (B, N) tensors.
    p : (B, N) one-hot perturbation indicator.
    gene_names : length-N list.
    descendant_map : ``{gene: {descendant_gene_1, ...}}``.
    """
    B, N = pred.shape
    ko_genes = [gene_names[int(torch.argmax(p[b]).item())] for b in range(B)]
    sub_mask = build_subtree_mask(ko_genes, gene_names, descendant_map).to(pred.device)
    ko_mask = torch.zeros(B, N, dtype=torch.bool, device=pred.device)
    for b in range(B):
        ko_mask[b, int(torch.argmax(p[b]).item())] = True
    in_mask = sub_mask & ~ko_mask
    out_mask = ~sub_mask & ~ko_mask

    se = sq_err(pred, target)
    ae = abs_err(pred, target)
    if in_mask.any():
        mae_in = ae[in_mask].mean().item()
        mse_in = se[in_mask].mean().item()
    else:
        mae_in = mse_in = 0.0
    if out_mask.any():
        mae_out = ae[out_mask].mean().item()
        mse_out = se[out_mask].mean().item()
    else:
        mae_out = mse_out = 0.0

    return {
        "mae_in": mae_in, "mae_out": mae_out,
        "mse_in": mse_in, "mse_out": mse_out,
        "ratio": mae_in / (mae_out + 1e-8),
    }


def gene_role_stratified_mse(
    preds: Tensor, targets: Tensor,
    ko_genes: List[str], gene_roles: Dict[str, List[int]],
    gene_names: List[str], ko_info: Dict,
) -> Dict[str, float]:
    """Compute MSE separately for MR, TF, and leaf KO genes."""
    role_of_gene = {}
    for role, indices in gene_roles.items():
        for idx in indices:
            role_of_gene[gene_names[idx]] = role

    role_masks: Dict[str, List[int]] = {role: [] for role in ["mr", "tf", "leaf"]}
    for i, ko_name in enumerate(ko_genes):
        base = ko_name.split("@")[0]
        genes = [g.strip() for g in base.split(";")]
        primary = genes[0]
        role = role_of_gene.get(primary, "leaf")
        role_masks[role].append(i)

    results: Dict[str, float] = {}
    for role, indices in role_masks.items():
        if indices:
            idx = torch.tensor(indices, dtype=torch.long)
            results[f"{role}_mse"] = sq_err(preds[idx], targets[idx]).mean().item()
            results[f"{role}_count"] = len(indices)
    return results


def fanout_stratified_mse(
    preds: Tensor, targets: Tensor,
    ko_genes: List[str], gene_out_degrees: Dict[str, int],
) -> Dict[str, Dict]:
    """Bin KOs by out-degree, compute MSE per bin."""
    degrees = []
    for ko_name in ko_genes:
        base = ko_name.split("@")[0].split(";")[0].strip()
        degrees.append(gene_out_degrees.get(base, 0))
    degrees_t = torch.tensor(degrees)
    unique_degrees = sorted(set(degrees))
    if len(unique_degrees) <= 4:
        sample_bins = [str(d) for d in degrees]
    else:
        quartiles = torch.quantile(
            degrees_t.float(), torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0]),
        )
        edges = sorted(set(int(q.item()) for q in quartiles))
        sample_bins = []
        for d in degrees:
            for i in range(len(edges) - 1):
                if d <= edges[i + 1]:
                    sample_bins.append(f"{edges[i]}-{edges[i+1]}")
                    break
            else:
                sample_bins.append(f"{edges[-1]}+")
    bin_indices: Dict[str, List[int]] = defaultdict(list)
    for i, b in enumerate(sample_bins):
        bin_indices[b].append(i)
    results = {}
    for label, indices in sorted(bin_indices.items()):
        idx = torch.tensor(indices, dtype=torch.long)
        results[label] = {
            "mse": sq_err(preds[idx], targets[idx]).mean().item(),
            "count": len(indices),
        }
    return results
