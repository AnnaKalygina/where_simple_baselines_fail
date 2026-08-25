# Potential train/test leakage in the DL data loader for cell-type-aware regimes

**Scope:** the shared DL benchmarking loader in
`/cluster/work/boeva/akalygina/literature-repos/Perturbation-Models-Outperform-Baselines`
(Miller et al. pipeline), for the regimes where **cell type matters**: `UnseenCell` (s2),
`UnseenBoth` (s3), `UnseenPair` (s4).

**Why this note:** to let anyone training DL models with this loader confirm that, for each
regime, the **training set actually excludes the held-out (cell type, perturbation) cells**.
The risk is silent: it inflates scores rather than crashing.

## TL;DR

- Splits are stored as a **per-cell `obs[split_name]`** column (values `train`/`val`/`test`),
  **not** as boolean masks. The leakage-safe way to use them is to filter **cells** by that
  per-cell label. A coarser filter by **perturbation (`condition`) lists** is NOT safe for
  s2/s3/s4.
- For s2/s3/s4 the held-out unit is a **(cell type × perturbation)** subset. A held-out test
  perturbation is still present in **training** in *other* cell types, so it appears in
  `train_conditions`. Therefore selecting training cells by `condition.isin(train_conditions)`
  **pulls the held-out test cells back into training** → leakage. Only the per-cell
  `obs[split_name]` label separates them.
- **PRESAGE and cellflow do it correctly** (filter by the per-cell label, covariate-aware).
- **GEARS and scGPT do NOT use covariates at all today** — they set `cell_type="NOTHING"`,
  split by marginal `condition` lists, and write a *dummy* `covariate='none'` to their
  predictions. So for them s2/s3/s4 collapse to a cell-agnostic perturbation task. The
  `# TODO: remove when doing covariates` markers indicate covariate conditioning is intended
  but **not implemented**. If covariate conditioning is added **without switching the data
  subsetting to the per-cell label**, GEARS/scGPT will leak on s2/s3/s4.
- The DL **model artifacts** (`metadata.json` in
  `/cluster/work/boeva/virtual_cell_reasoning/models/atheus_csb`) record only perturbation
  `test_conditions` — **no cell type / covariate / pair information at all** — so you cannot
  verify the held-out (cell, pert) set from the artifacts; you must inspect the training code
  and the per-cell split column.

## The principle

The data is a grid of `(cell_type, perturbation)` units. For s2/s3/s4 the test set is a
subset of grid *cells*, and a test perturbation/cell type is generally still seen elsewhere
in training. So the split is only faithfully represented at **per-cell granularity**.
Reducing it to a list of *training perturbations* is lossy and, for these regimes, leaky.

## How each model consumes the split (exact locations)

Loader file paths are relative to the Miller repo above.

### Correct — filter cells by the per-cell `obs[split_name]` label (covariate-aware)
- **cellflow** — `docker/cellflow/cellflow_wrapper.py`
  - `:305`  `split_col = adata.obs[split_name].astype(str)`
  - `:322`  `(split_col == "train") & ~is_ctrl & condition_str.isin(train_conditions)`
  - `:329` / `:336`  same for `val` / `test`.
  - The `split_col == "..."` per-cell filter is what makes this leakage-safe; the
    `condition.isin(...)` is an additional (redundant-but-safe) intersection.
- **PRESAGE** — `docker/presage/cellsimbench_datamodule.py`
  - `:472-475`  builds a perturbation mask, then **`split_adata = split_adata[split_adata.obs[self.split_name] == name]`** — keeps only cells whose per-cell label matches.
  - `:504-513`  same for the `test` stage (`obs[split_name] == 'test'`).
  - Consistent with PRESAGE prediction files carrying real `covariate` + sparse (cell, pert) pairs.
- **sclambda** — `docker/sclambda/sclambda_wrapper.py:713`  `self.adata[self.adata.obs[split_name] == 'train']`.
- **fmlp** — `docker/fmlp/fmlp_wrapper.py:356`  `self.adata[self.adata.obs[split_name] == 'train']`.

### Cell-agnostic / marginal `condition`-list split — unsafe for s2/s3/s4 if covariates are later enabled
- **GEARS** — `docker/gears/`
  - `gears_wrapper.py:891-892`  `adata_gears.obs['cell_type'] = "NOTHING"  # TODO: Needs to be removed when doing covariates`.
  - `gears_wrapper.py:961-980`  `train_conditions = self.config['train_conditions']` … `set2conditions = {'train': convert_conditions(train_conditions), 'val': …, 'test': …}` → GEARS `PertData` splits by **condition**, not per-cell label.
  - `utils.py:103`  representative pattern: `split_cells = adata[adata.obs['condition'].isin(conditions)]`.
- **scGPT** — `docker/scgpt/`
  - `scgpt_wrapper.py:619-620`  `adata_scgpt.obs['cell_type'] = "NOTHING"  # TODO: Remove this when we have covariates`.
  - `scgpt_wrapper.py:138`  encoder docstring: "Encode **without covariates** …".
  - `scgpt_wrapper.py:729-754`  builds the split from `train_conditions`/`test_conditions` via `convert_conditions(...)` (condition-level).
  - `scgpt_wrapper.py:1190-1193`  writes a **dummy** covariate to predictions:
    `'covariate': ['none'] * len(condition_list)  # … even though the model doesn't use covariates`.

### Where the condition lists come from (the lossy reduction)
- `cellsimbench/core/data_manager.py:516-518` — `get_perturbation_conditions`:
  `train_conditions = obs[split_data == 'train']['condition'].unique()` (and val/test). A
  perturbation that is `train` in one cell type and `test` in another ends up in **both**
  the train and test lists. (A granular accessor exists —
  `get_covariate_condition_pairs(split_name, split_type)` at `:531-555`, returning
  `(covariate, condition)` pairs — but GEARS/scGPT do not use it for training.)

## Why a condition-list filter leaks on s2/s3/s4 (mechanism + evidence)

For these regimes `train_conditions ⊇ test_conditions`. Verified from the shipped configs
(`models/atheus_csb/<model>/<run>/metadata.json`, `config.{train,test}_conditions`):

| regime | dataset | n_train_cond | n_test_cond | train ∩ test |
|---|---|---|---|---|
| UnseenBoth (s3) | mcfaline23 f0 | 512 (all) | 102 | **102 (all test perts)** |
| UnseenPair (s4) | mcfaline23 f0 | 501 | 240 | **229** |
| UnseenBoth (s3) | replogle22 f0 | 1992 (all) | 398 | **398 (all test perts)** |

Because every test perturbation is also a train perturbation, a filter of the form
`adata.obs['condition'].isin(train_conditions)` selects **all** cells of that perturbation —
including the held-out `(test_cell_type, test_pert)` cells. A cell-type-aware model trained on
that data has seen the exact evaluation cells. (UnseenCell is the same story: the held-out
cell type's perturbations are all in `train_conditions`.) Only `obs[split_name] == 'train'`
removes the held-out cells.

GEARS/scGPT avoid this **today only because they are cell-agnostic** (`cell_type="NOTHING"`),
so they never claim to do cell-type-conditioned prediction — their s2/s3/s4 results are
effectively perturbation-level, pooled over all cell types. The moment covariate conditioning
is switched on (the `TODO`s), the condition-list subsetting must be replaced by the per-cell
label or the leakage above is introduced.

## The artifact (metadata.json) gap

In `/cluster/work/boeva/virtual_cell_reasoning/models/atheus_csb`, each model fold's
`metadata.json` records `config.test_conditions` as a **perturbation list only**. There is
**no cell type / covariate / (cell, pert) pair** recorded; scGPT/GEARS even emit a dummy
`covariate='none'` in predictions. Consequence: for the cell-type-aware regimes you **cannot**
reconstruct which `(cell type, perturbation)` pairs were held out from the artifacts alone.
The only DL artifact that exposes the true held-out cell pairing is **PRESAGE** (its
prediction h5ads carry real `covariate` + sparse pairs); GEARS/scGPT emit the full
`perturbations × all cell types` cross product. To audit a GEARS/scGPT run you must inspect
the training code path + the per-cell split column directly.

## Checklist — is a training run leakage-free for s2/s3/s4?

1. Does training select cells by **`obs[split_name] == 'train'`** (good) or by
   **`condition.isin(train_conditions)`** alone (leaky for s2/s3/s4)?
2. Is `cell_type` (the covariate) actually fed to the model, or overwritten to a constant
   (e.g. `"NOTHING"`)? If overwritten, the model is cell-agnostic and the regime is not the
   cell-type task it claims to be.
3. If you enable covariate conditioning, did you also switch data subsetting to the per-cell
   label? (cellflow `:322` and PRESAGE `:475` are the reference-correct patterns.)
4. Sanity check: `set(train_cells.obs.index) ∩ set(test_cells.obs.index) == ∅` AND no
   `(cell_type, condition)` pair appears in both train and test.

## Should we bring the cellsimbench docker containers into our repo?

The deeper fix for "is everything verified and aligned?" is to stop depending on an
external, separately-preprocessed training pipeline and instead train from **our** h5ad
(the single source of truth that already carries the per-cell split columns + masks). This
would have eliminated, at the source, the divergences we had to reconcile after the fact:
the mcfaline23 14-extra-perturbation fold shuffle, the replogle22 `K562` vs `k562` covariate
mismatch, and the very split-handling ambiguity this note is about. Options, with trade-offs:

**A. Vendor the full GPU training containers (GEARS/scGPT/PRESAGE/cellflow) + train from our h5ad.**
- Pro: end-to-end control — one set of splits, verified leakage-free, covariate conditioning
  implemented correctly, predictions that record real `(cell type, perturbation)` metadata
  (closing the artifact gap above); fully reproducible and re-runnable on new datasets/folds.
- Con: large CUDA/torch images + model checkpoints land on a CephFS that is ~98% full;
  training is a different concern (GPU scheduling) from this repo's role (shared data + a CPU
  scoring benchmark over h5ads); forking the trainers means our DL numbers diverge from the
  published Miller results — a feature if we want *corrected* numbers, a liability if we want
  to cite *their* numbers. **Also note our standing convention: new/heavy code lives outside
  the shared `virtual_cell_reasoning` data dir** (in `akalygina/` or a dedicated repo), so the
  containers should NOT go into this repo's tree regardless.
- Verdict: do this as a **separate, pinned fork** (not inside the shared data repo) only if we
  decide we need *retrained, cell-aware, leakage-corrected* DL results.

**B. Vendor only the data-loading layer (`cellsimbench/core/data_manager.py` + the model
datamodules), not the trainers.**
- Pro: this is where alignment + leakage actually live; adapt it to read our h5ad's per-cell
  `obs[split_name]` directly and we guarantee the split logic matches our baselines, with no
  GPU/image burden. The model containers stay upstream.
- Con: still a partial fork to keep in sync with upstream; doesn't retrain anything by itself.

**C. Keep upstream as-is; add a lightweight pre-train verification shim (recommended minimum).**
- A small script (in `akalygina/`, runnable in CPU env) that, given a training config and our
  h5ad, asserts the checklist above before any GPU run: train cells are selected by the
  per-cell label, the covariate is actually fed (not `"NOTHING"`), and
  `train (cell,pert) ∩ test (cell,pert) == ∅`. Catches leakage without owning the trainers.
- Pro: cheapest, immediately useful to the colleague, no repo bloat.
- Con: a guard, not a fix — it flags leakage but the upstream loader must still be corrected.

**Recommendation:** start with **C** (a guard the colleague can run before every training job),
move to **B** if we want the split logic permanently aligned to our h5ad, and reserve **A**
(a separate pinned fork in `akalygina/`, never inside this data repo) for the case where we
explicitly want retrained cell-aware / leakage-corrected DL numbers.

## Related (internal)

We found and are fixing the **same class of bug** in our own baseline predictors: for
UnseenBoth they reconstructed the train region as the marginal `train_bins × train_kos`
rectangle, which re-includes the held-out corner (measured: 103/103 test cells in the train
rectangle on mcfaline23 fold0). The fix routes every predictor through an explicit per-(bin,
ko) mask — i.e. the same per-cell-granularity principle PRESAGE/cellflow use here.
