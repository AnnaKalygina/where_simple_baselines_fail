# wsb-13: talk figures 20.10

**Ready for slides:** Figure 1 (filtering, claim NOT supported, see R3) and Figure 3 (metrics) ready; Figure 2 (shapes) ready. All three in the vault inbox.

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
8. **Magnitude interval: the literal percentile bootstrap is biased, so two intervals are reported.** Inside a bootstrap resample, y_b² again carries the sampling variance of the resampled means, while se²_b only removes it once, so M_b centers on the raw Σ y² instead of on the debiased M (it is shifted up by about Σ_g se²_g). Test values: MRPS7 M = 3.55, literal interval [9.59, 15.36]; a control pseudo-perturbation M = −0.09, literal interval [6.20, 10.91]. Read literally, "no detectable effect" would almost never fire, including for nulls. Decision: `M_ci_low`/`M_ci_high` = M + the 2.5/97.5 percentiles of (M_b − mean M_b) (the bootstrap spread, centered on M); this is what every flag ("no effect", shape, Figure 1) uses. The literal percentiles are kept as `M_ci_low_literal`/`M_ci_high_literal`. Anna to confirm.

## Results

### R1. M1 metrics on K562 essential (runbook 3.1 to 3.6)
- **Status:** done.
- **Headline:** 2,021 perturbations with n ≥ 20 (36 excluded, listed in `setup_k562_essential.json`), 10,691 controls, |G0| = 8,548 genes. 39.4% have rho ≥ 0.5. Shapes: 924 shift, 690 none, 348 mixture, 59 edge. tau2 and tau2_direct agree (Spearman 0.9996, n = 1,978). Consistency "mean = pi d" holds for mixtures (Spearman of mean_B(s) − mu0 vs pi (mu1 − mu0): 0.999).
- **Files:** `results/talk_2026-10-20/M1_k562_essential.csv` (one row per perturbation); per-cell half-B and K3 scores on the cluster only, `/cluster/work/boeva/akalygina/wsb13_cache/scores_halfB_K3_k562_essential.parquet`.
- **Deviations:** decisions 1 to 8 above.
- **Warning, null-score skew (runbook 3.4 "shape of the null scores"):** the K3 control scores are often far from normal: median |skew| 0.88, 67% of perturbations above 0.5. Skew grows with magnitude (Spearman 0.65 with M) and is largest exactly where mixtures are called (median skew: mixture 1.96, shift 1.07, none 0.09). Among n ≥ 150, the mixture rate is 15% for |skew| < 0.5 and 53% for |skew| ≥ 1. The response direction appears to run along an axis on which controls already have a right tail, and a normal component 0 cannot fit that tail, so **a large share of the mixture calls may be misfit of the normal null rather than two subpopulations**. Not changed. **Sensitivity check (E1) answers this: the calls are robust**, 317 of 348 survive an empirical null.
- **Commit:** 35921fa

### R2. M2 null (50 control pseudo-perturbations, n = 123 = median n_p)
- **Status:** done; nulls pass, so figures were made.
- **Headline:** M median −0.10 (range −0.44 to 0.92), centered interval contains 0 in 50/50 (literal interval in 0/50, see decision 8); rho median −0.008, max 0.11, none ≥ 0.5; dBIC > 10 in 3/50, pi identified in 2/50 (both `edge`), mixture 0/50; widened 1/50. D is not centered at 0 in the null (median 0.57, within its own null, D_null95 median 0.72): log of a noisy phi is biased up at small n.
- **Files:** `M2_null_k562_essential.csv`, `M2_null_summary.csv`.
- **Commit:** 35921fa

### R3. Figure 1, reliability filter vs response shape (n ≥ 150)
- **Status:** done. **The figure does not support the expected claim.**
- **Headline:** reliable (rho ≥ 0.5) n = 400 vs unreliable n = 362: mixture 60.8% vs 7.2%; widened 79.5% vs 14.9%; no detectable effect 0% vs 54.1%; mixture with escapers (kd_non > 0.8) 6.5% vs 1.7%. Median tau2 0.025 vs 0.00095. The unreliable group is mostly "no effect", not heterogeneous.
- **Caveats for reading it:** (i) the mixture and widening tests gain power with effect size, so the comparison is not power-matched; (ii) control-score skew does not drive the split: with an empirical null (E1) it is 61.0% vs 6.1%; (iii) escaper diagnostic eligible for 199 of 243 reliable mixtures and 20 of 26 unreliable mixtures.
- **Brief F3 overlap (unreliable, n ≥ 150, n = 362):** (a) M interval contains 0: 196; (b) D > D_null95: 54; (c) dBIC > 10 and 0.1 < pi_raw < 0.9 (brief wording, no |d'| rule): 29. a∧b 13, a∧c 4, b∧c 10, a∧b∧c 0, none of the three 110. Files `F3_flags_overlap.csv`, `F3_flags_per_perturbation.csv`.
- **Files:** `fig1_filtering.png`, `fig1_filtering_table.csv`.
- **Commit:** 35921fa

### R4. S0 checks and Figure 3
- **Status:** done.
- **S0-a (split-half stability, Spearman-Brown):** M 0.97, tau2 0.98, pi 0.95 (only 166 perturbations with pi identified in both halves), D 0.91, rho 0.98.
- **S0-b (n ≥ 200, 414 perturbations, 100 cells, 10 repeats):** full vs downsampled Spearman M 0.999, tau2 0.998, pi 0.96 (n = 203), D 0.92. corr with n before/after: M 0.28/0.28, tau2 0.27/0.28, pi −0.03/0.14, D 0.00/0.29, rho 0.43/0.27. Across all 2,021 perturbations D has Spearman −0.62 with n (M 0.02, tau2 0.00): **raw D is not n-independent**; D − D_null95 has +0.29.
- **S0-c (downsampled values, n = 414):** Spearman M–tau2 0.996, M–rho 0.997, tau2–rho 0.999, M–pi 0.89, M–D 0.73, pi–D 0.47. R² on rank of M: tau2 0.82, pi 0.86, D 0.46, rho 0.96. At a fixed 100 cells rho is a monotone function of tau2, so their 1.0 is by construction. At full n, M–tau2 is still 0.998: **tau2 essentially restates magnitude** in this screen. D is the metric least explained by magnitude.
- **Files:** `S0_checks.csv`, `S0c_spearman_matrix_downsampled.csv`, `S0_inputs_k562_essential.csv`, `fig3_metrics.png`, `headline_numbers.json`.
- **Commit:** 35921fa

### R5. Figure 2, three response shapes
- **Status:** done. UMAP (optional) not made.
- **Examples (runbook rules, automatic, swappable):** INTS8, clean shift (largest M among rho ≥ 0.5 with dBIC < 2; n 323, rho 0.96, M 119, dBIC 0.2); PSMD7, mixture (pi closest to 0.5 among dBIC > 10, |d'| ≥ 1, n ≥ 150, not escapers; n 373, pi 0.50, dBIC 106, d' 5.8, kd_non 0.67); KIF11, unreliable despite many cells (largest n among rho < 0.5 flagged mixture or widened; n 974, rho 0.16, widened, pi 0.07 so shape `edge`).
- **Construction:** x = s_i on split 0 (direction from half A), half-B cells and K3 controls; y = PC1 of half-B residuals x_i − mean(K1) − s_i w over G, K3 projected. Seeds reproduce split 0 of the M1 run exactly.
- **Visible caveat:** in all three panels the controls themselves have a long tail along the response direction (the skew in R1).
- **Files:** `fig2_shapes.png`, `fig2_examples.json`.
- **Commit:** 0a413bc

## Extras (after the runbook, in the brief's tier order)

### E1. Mixture sensitivity to the normal null (not in the brief; checks the key claim)
- **Status:** done.
- **Method:** on the saved half-B and K3 scores, component 0 = Gaussian kernel density of the K3 scores (scipy `gaussian_kde`, Scott bandwidth), fixed; component 1 normal, free; EM with 10 starts (seed `[20261020, 400, i]`). A call needs pi in (0.1, 0.9), |d'| ≥ 1, dBIC > 10 against a single free normal **and** dBIC > 10 against the K3 density shifted by a free constant ("all cells shift, keeping the control shape").
- **Headline:** 342 empirical-null mixtures vs 348 runbook mixtures, 317 in both. Figure 1 groups (n ≥ 150): reliable 61.0% (runbook 60.8%), unreliable 6.1% (7.2%). The skew warning in R1 does not change the Figure 1 conclusion.
- **Not done:** the false-positive rate of this variant on the M2 nulls (their scores were not saved).
- **Files:** `mixture_sensitivity_k562_essential.csv`, `mixture_sensitivity_fig1_groups.csv`; code `mixture_sensitivity.py`.

### E2. Brief F1 and figure T1-B
- **Status:** done.
- **Headline:** fraction with rho ≥ 0.5 by n bin: <50 19.6% (n 189), 50–99 33.8% (533), 100–149 33.3% (537), 150–249 40.9% (523), 250–499 76.5% (213), ≥500 88.5% (26). Median tau2: 0.0093, 0.0079, 0.0038, 0.0038, 0.0213, 0.0281. tau2 is not constant across bins: perturbations with more cells also have larger tau2 in this screen.
- **Files:** `F1_reliability_by_cell_bin.csv`, `T1-B_reliability_by_cell_bin.png`.
- **Commit:** 620e2b2

### E3. Brief F3 and figure T1-C
- **Status:** done (numbers in R3, checked twice: computed in `analyze.py` and again as exclusive categories in `brief_tier1.py`, both sum to 362).
- **Headline (exclusive):** no effect only 179, widened only 31, mixture only 15, no effect + widened 13, no effect + mixture 4, widened + mixture 10, all three 0, none 110.
- **Files:** `T1-C_F3_flags.png`, `T1-C_F3_exclusive_counts.csv`.
- **Commit:** 620e2b2

## Blockers

- **Vault pull request.** The cluster has no `gh` CLI and no GitHub token (SSH push works). The inbox branch `inbox/where_simple_baselines_fail/talk-figures` is pushed, but the pull request and `gh pr merge --auto --squash` need Anna (or a token).
- **SLURM.** A 48-core request could not fit on any node tonight (estimated start 08:26); rerun with 36 cores (job 9783148).
