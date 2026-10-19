# wsb-13: talk figures 20.10

**Ready for slides:** in progress.

Scope (agreed with Anna 20.10, 00:30): the runbook `methodology_talk_2026-10-20.md` (four intrinsic metrics, three figures) first, then the brief's extras in tier order if time remains.

## Setup

- **Data:** `/cluster/work/boeva/virtual_cell_reasoning/data/replogle22/replogle22_k562_downloaded.h5ad` (Zenodo 10044268, `ReplogleWeissman2022_K562_essential.h5ad`, the file `data/replogle22/get_data.py` downloads). Not re-downloaded. 310,385 cells x 8,563 genes, raw counts, **no per-perturbation cap** (the capped file `replogle22_processed.h5ad` is not used). The Figshare file `VCC/VCC2026/data/raw/replogle2022/K562_essential_raw_singlecell_01.h5ad` holds the same matrix.
- **Genome-wide K562 (for F10):** already on the cluster, `VCC/VCC2026/data/raw/replogle2022/K562_gwps_raw_singlecell_01.h5ad` (Figshare 35775507, 1,989,578 cells x 8,248 genes). Zenodo 10044268 was not needed.
- **Sparse cache (not in git):** `/cluster/work/boeva/akalygina/wsb13_cache/k562/` (CSR counts + obs/var arrays), built by `analysis/talk_2026_10_20/to_sparse.py`.
- **Code:** `analysis/talk_2026_10_20/metrics.py` (formulas), `run_metrics.py` (driver, per-perturbation parallel, checkpointed).

## Seeds

Base seed `20261020`, passed to `numpy.random.default_rng` as a list:

| Use | Seed |
|---|---|
| Control thirds (100 splits) and control bootstraps, all controls | `[20261020, 1]` |
| Context K1 / K2 of split 0 (S0-a) | `[20261020, 2]` / `[20261020, 3]` |
| Perturbation `i` (index in sorted list of names): halves, bootstraps, EM starts, D null, S0 splits and downsampling | `[20261020, 100, i]` |
| Pseudo-perturbation `j` (M2): subset draw / its control context | `[20261020, 200, j]` / `[20261020, 300, j]` |

## Definition decisions and deviations

1. **Depth L_i.** The runbook says L_i = sum over all genes. The file stores only 8,563 pre-filtered genes. I use the row sum over the stored genes (median 98.7% of `UMI_count`). `UMI_count` is kept in the source and not used.
2. **QC.** All cells already have at least 713 detected genes, so the 200-gene gate removes 0 cells.
3. **Control splits and control bootstraps are shared across perturbations.** The 100 control thirds (K1, K2, K3) and the 100 control bootstrap resamples are drawn once (seed above) and reused for every perturbation; the perturbed-cell halves and bootstraps are drawn per perturbation. This makes perturbations comparable on the same control noise and keeps the run within the night's budget.
4. **Lowess trend for D.** `statsmodels` lowess with its default `frac = 2/3`, `it = 3`, and `delta = 1%` of the log-mean range (speed). The runbook names no bandwidth. The trend is evaluated at a perturbation's log means by linear interpolation, constant beyond the fitted range. The trend uses all of G0; the statistic uses G (target removed).
5. **sd conventions.** sigma0 = sample sd (ddof 1) of K3 scores; the single-normal BIC uses the maximum-likelihood mean and sd; tau2_direct uses Var_g with ddof 0.
6. **shape.** The runbook defines "mixture" (identified pi in (0.1, 0.9)), "shift" and "none" (pi not identified). An identified pi outside (0.1, 0.9) is not covered; it gets `shape = edge`.
7. **Escaper diagnostic.** Computed when the target is in the stored matrix and its mean control count is at least 0.5 (`escaper_eligible`), from the posterior of the half-B cells, whether or not pi is identified. Use it together with `pi`.
