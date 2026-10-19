# wsb-13: talk figures 20.10

**Ready for slides:** Figure 1 (filtering, claim NOT supported, see R3) and Figure 3 (metrics) ready; Figure 2 (shapes) ready. All three in the vault inbox. Extras: T1-A, T1-B, T1-C, T2-A, T2-B, F10 figure and tables for M3, F2, F7, F8, F9; F6 impossible (no predictions).

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
- **Commit:** bcdd9a3

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

### E4. Brief M3 per-gene table and figure T2-A
- **Status:** done.
- **Examples (brief rules, which differ slightly from the runbook's Figure 2 rules):** GATA1 (highest M: n 108, M 289, rho 0.93, pi 0.87, dBIC 46, mixture); PSMD7 (pi closest to 0.5 among dBIC > 10: n 373, pi 0.50, dBIC 106); KIF11 (largest n among rho < 0.5: n 974, rho 0.16, pi 0.07, dBIC 36, `edge`).
- **M3 columns:** per G0 gene and set (controls, the three perturbations): A_prime and B_prime (1e6 scale), mu_1e4 and var_1e4 (u), delta_C, logdiff, varterm (formulas as in the brief), is_target.
- **Files:** `M3_per_gene_k562_essential.csv.gz` (per-gene summary, 4 x 8,548 rows), `T2-A_response_shapes.png`, `M3_T2A_examples.json`.
- **Commit:** 960f176

### E5. Brief F6 (direction and scale errors, PRESAGE / Ridge / baseline, replogle22 UnseenPert)
- **Status:** skipped: the inputs do not exist. No replogle22 UnseenPert predictions in the current pipeline format for PRESAGE, Ridge or the regime baseline anywhere on the cluster (`models/replogle22/` holds only `PRESAGE-ct/UnseenBoth/fold0`). The only UnseenPert predictions are the old PRESAGE/GEARS/scGPT drops in `models/atheus_csb/*_replogle22_s1/` (trust `legacy_caution`, may predate the leakage fix, K562 and RPE1 mixed). Producing F6 needs a benchmark run of Ridge, Mean + Target Scaling and PRESAGE-ct on replogle22 UnseenPert first.

### E6. Remaining M1 columns, F2, F7, F9 and figures T1-A, T2-B
- **Status:** done. Space-C rho from this pass reproduces M1 rho exactly (max abs difference 4e-16), so the splits are identical.
- **New columns** (`M1_full_k562_essential.csv` = M1 + extras): `breadth_pr` (median 696), `phi` and `phi_loo` (median 0.048 and 0.047; cos² to the cell-weighted mean effect of the 796 perturbations with rho ≥ 0.5, over G0 minus target), `mito_AmB` (5/50/95%: −1953, −4.1, 2320 on the 1e6 scale), `mito_frac_median` (controls 0.107), `log2_depth_ratio`, `depth_R`, `depth_cv`.
- **F2 (gene panel):** rho ≥ 0.5 on G0 39.4%, top-2,000 HVG 51.6%, top-8,192 HVG 39.4%. HVG from scanpy `highly_variable_genes` (default `seurat` flavor) on normalize_total(1e4) + log1p of all 310,385 cells; panels are HVG ∩ G0 (11 of 2,000 and 12 of 8,192 HVG fall outside G0 and are dropped). File `F2_panels.csv`.
- **F7 (spaces):** rho ≥ 0.5 in C 39.4%, A 37.3%, B 35.7%; Spearman of rho A vs C 0.993 (per-perturbation values in `F7_F9_spaces_weights.csv`).
- **F9:** R = mean(L)·mean(1/L): perturbations 5/50/95% 1.094, 1.139, 1.239; controls 1.125. Depth CV median 0.354 (controls 0.334). NB-weighted pseudobulk (space A with w_ig = L_i/(1 + phi_g theta_g L_i)): rho ≥ 0.5 34.2%. Decision: phi_g and theta_g = phi-hat and mu/1e4 of all controls (G0), phi clipped at 0 (399 genes with phi ≤ 0 get Poisson weights).
- **T1-A (Spearman of all descriptors, all 2,021, full n, ordered by |corr with M|):** with M: tau2 1.00, rho 0.95, mu1 0.87, phi 0.77, D 0.48, median depth −0.42, pi_raw 0.42, dBIC 0.33, breadth −0.25, mito_AmB 0.04, mito frac 0.03, n 0.02. The S0-a side bar exists only for M, tau2, pi, D and rho; **S0-a/b were not computed for breadth_pr, phi, mito_AmB, depth columns** (deviation from "S0 checks for every descriptor").
- **Files:** `M1_full_k562_essential.csv`, `M1_extras_k562_essential.csv`, `F2_F7_F9_headline.json`, `T1-A_descriptor_heatmap.png`, `T1-A_spearman_all_descriptors.csv`, `T2-B_reliability_spaces.png`, `extras_setup_k562_essential.json`; code `run_extras.py`, `post_extras.py`.
- **Commit:** 9e6129c

### E7. F10, essential vs genome-wide K562
- **Status:** done.
- **Data:** Figshare raw single-cell files for both screens (same matrix as Zenodo for essential), all cells, each against its own non-targeting controls (10,691 and 75,328). Shared genes: G0 of both screens, 8,186. Perturbations with n ≥ 20: 2,021 and 9,787; 2,003 shared.
- **Design (simplified from M1 for a 2-million-cell file):** one random split per perturbation (halves) and one control-thirds split, seed `[20261020, 500, 1 or 2]`; magnitude as M1 (target excluded); 50 disjoint control pseudo-perturbations of the median size (123 and 179) against the remaining controls give the threshold.
- **(a) Norm of the mean effect** (unweighted mean over perturbations): essential 1.47, genome-wide 0.46 (cell-weighted 1.69 vs 0.33). The genome-wide mean is 3.2 times shorter.
- **(b) Magnitude:** median M 6.0 vs 0.62; threshold = largest null M (0.80 vs 0.44): 79.7% vs 57.8% of perturbations above it.
- **(c) Cosine between the two means:** 0.88.
- **(d) Shared perturbations (2,003):** their own magnitude is the same in both screens (median M 5.96 vs 5.86, Spearman 0.83). Distance to the leave-one-out screen mean: median 4.16 (essential) vs 3.84 (genome-wide), larger in genome-wide for only 32%. Baseline Saturation (see definition below): median 0.71 vs 0.78 (n = 1,869 with both defined), lower in genome-wide for only 30%; Spearman between screens 0.34. **The expectation "the same perturbation sits farther from the genome-wide mean and looks harder" is not supported.**
- **Baseline Saturation definition used (decision, the brief gives the general formula only):** Pearson across shared genes minus target, b = leave-one-out unweighted mean effect of the screen, truth = half-B effect, positive control = half-A effect, m(0, y) = 0: BS = clip(corr(b, y_B) / corr(y_A, y_B), 0, 1), empty when corr(y_A, y_B) ≤ 0. **No Target Scaling** (the paper's UnseenPert baseline is Mean + Target Scaling). Pearson ignores scale, so it cannot react to the shorter genome-wide mean by construction; the distance column is the scale-aware view.
- **Files:** `F10_headline.json`, `F10_shared_perturbations.csv`, `F10_per_perturbation_essential.csv`, `F10_per_perturbation_gwps.csv`, `F10_screen_mean.png`; code `f10_stream.py`, `f10_analyze.py`.
- **Commit:** 9e6129c

### E8. F8, Norman 2019 combinations
- **Status:** done.
- **NA:** 131 pairs with both singles measured (111,445 cells, 11,855 controls, |G0| = 9,616 on the downloaded file, all cells). NA quantiles 10/25/50/75/90%: 0.05, 0.07, 0.12, 0.30, 0.54.
- **Errors (UnseenCombo, `_panel_miller_align`, folds 0 and 1, pipeline truth = first-half pseudobulk minus control, both targets excluded):** models run: PRESAGE, GEARS, scGPT, Additive (also present but not used: Ridge, LinearAdditive, GlobalEpistasis, LatentAdditive and others). Identity D + S = squared error holds on every row (max relative difference 1.6e-14). Median direction share D / (D + S): PRESAGE 0.88, GEARS 0.91, scGPT 0.95, Additive 0.83. Spearman of NA with scale error: PRESAGE 0.59, GEARS 0.48, scGPT 0.09, Additive 0.71; with direction error: 0.00, −0.15, −0.22, −0.06.
- **Deviation:** scGPT predictions are NaN on 111 genes for most pairs (outside its vocabulary), and fully NaN for 6 rows (3 per fold); GEARS fully NaN for 1 row. Errors use the genes with a finite prediction (`n_genes`, `n_genes_nan_prediction` per row); fully NaN rows are empty.
- **Files:** `F8_non_additivity_norman19.csv`, `F8_direction_scale_errors_norman19.csv`, `F8_errors_with_NA_norman19.csv`, `F8_headline.json`; code `f8_combos.py`.
- **Commit:** 9e6129c

### Not done
- F4 (GEM-group halves), F5, T1 (RPE1 transfer): not in tonight's scope.
- Optional UMAP for Figure 2.

## Blockers

- **Vault pull request.** The cluster has no `gh` CLI and no GitHub token (SSH push works). The inbox branch `inbox/where_simple_baselines_fail/talk-figures` is pushed, but the pull request and `gh pr merge --auto --squash` need Anna (or a token).
- **SLURM.** A 48-core request could not fit on any node tonight (estimated start 08:26); rerun with 36 cores (job 9783148).
