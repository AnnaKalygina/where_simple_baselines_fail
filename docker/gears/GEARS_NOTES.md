# GEARS container — notes & design reference

*How the vendored GEARS container behaves and **why** it's wired this way. Operational
build/run steps: `BUILD_ENV.md`. The machine contract: `../CONTRACT.md`. Umbrella design:
`../../TRAINING_INFRA_PLAN.md`.*

*History: this was the M0 `VENDORING_PLAN.md`; trimmed on 2026-08-14 to the durable
reference once M0 (build) + M1 (scored loop) landed — the executed build-procedure,
work-item, and smoke-gate sections were removed. Line refs below index the atheus source
`gears_wrapper.py` (~1579 L) and are **approximate** after the 2026-08-14 cleanup (the shim
inline + seed fix shifted the vendored wrapper by a few dozen lines).*

Source investigated: `/cluster/work/boeva/atheus/Perturbation-Models-Outperform-Baselines/docker/gears/`
(`gears_wrapper.py`, `run_model.py`, `gene2go_all.pkl` 9.46 MB).

---

## 1. What the container actually is (verified contract)

- **Entrypoint** `run_model.py {train|predict} /config.json` → `GEARSWrapper(config).train()|predict()`.
  CLI + mount layout match `../CONTRACT.md`. ✓
- **Its own GEARS** is *vendored inside* `gears_wrapper.py` (the "BASIC GEARS IMPLEMENTATION"
  block) — a copy that adds `ctrl_adata_override` for cov-specific basal. It still imports
  `PertData`, `GEARS_Model`, `gears.utils`, `gears.inference` from the pip package
  **`git+https://github.com/millerh1/GEARS.git`** (the fork). So the pin supplies the
  nn.Module + data utils, but the **training/predict loop is the vendored code** (see
  `BUILD_ENV.md`, "Pin ≠ training recipe").
- **gene2go** is passed explicitly to `PertData(..., gene2go=...)` → avoids a runtime
  download. The container path is `config['gene2go_path']` (= `/app_code/gears/gene2go_all.pkl`,
  bind-mounted, set by `model.yaml`).

### 1.1 Config schema the wrapper *actually* reads

| key | mode | meaning |
|---|---|---|
| `mode` | both | must equal the CLI verb |
| `data_path` | both | **one combined h5ad** read by `DataManager.load_dataset()` |
| `split_name` | both | name of a **per-cell obs column** with values `train/val/test` |
| `covariate_key` | both | obs column giving the real cell type (single canonical key; F4 collapsed the old `covariate_field`/`covariate_key` pair) |
| `seed` | both | drives all RNGs + the GEARS custom split (F1; see §1.3.9) |
| `hyperparameters` | both | dict — keys in §1.4 |
| `gene2go_path` | opt | in-container path to the gene2go pickle |
| `train/val/test_conditions` | train | condition lists → split dict |
| `output_dir`,`checkpoint_dir` | train | model + `processed_data/` written here |
| `model_path` | predict | trained-model dir; **must contain `processed_data/`** |
| `test_conditions`,`output_path` | predict | conditions to predict; where `predictions.h5ad` lands |

This is a **different serialization of the same authority**, not a rival contract: the
per-cell `split_{Scenario}_fold_{N}` column IS the leak-safe assignment; the combined form
carries it inline. See §1.5 for why GEARS' native split machinery wants the inline form.

### 1.2 Data ingestion

The wrapper obtains its adata from one call — `DataManager(config).load_dataset()` — at both
train and predict, reading one h5ad from `config['data_path']`. `DataManager` (+ `PathEncoder`)
is a ~20-line shim (`sc.read_h5ad` + existence check, no DEG gate) **inlined directly into
`gears_wrapper.py`** as of the 2026-08-14 cleanup (originally the shared `docker/_harness_container/harness.py`;
see §2 D3). Predict re-reads the same h5ad, so it must still contain the test cells + the
`split_name` column (§1.3.5).

### 1.3 Non-obvious behaviors that will bite if unhandled

1. **Control label filter:** keeps a row iff its condition does *not* contain the substring
   `ctrl` **or** is exactly `'control'`/`'ctrl_iegfp'`. ⇒ a control literally labelled
   **`'ctrl'` is DROPPED**. Controls must be `control` (or `ctrl_iegfp`). adamson16 = `control` ✓.
2. **Regime-aware per-cell filter:** for `UnseenPert` the per-cell filter is **skipped**
   (condition lists suffice); for cell-aware regimes it keeps only `split ∈ {train,val}`
   cells, dropping `test`. Missing column ⇒ a *warning*, not an error, and silent leakage
   risk. So `split_name` **must** be present/correct for UnseenCell/Both/Pair.
3. **`processed_data/` co-location:** training writes `output_dir/processed_data/…`; predict
   **reloads it from `model_path/processed_data/`** and errors if absent. ⇒ fit's `output_dir`
   and predict's `model_path` must be the **same directory**, persisting between the two calls.
   (`ContainerPredictor._run_dir` guarantees this; redirectable via `VCR_CONTAINER_MODEL_ROOT`.)
4. **PertData cache:** if `processed_data/…/cell_graphs.pkl` exists it **loads instead of
   reprocessing** — stale data reused silently. ⇒ one clean dir per (dataset,regime,fold)
   (`fit()` rmtrees the dir first).
5. **Predict cov enumeration needs test cells:** for cell-aware regimes the wrapper reads the
   held-out cov(s) + their control cells from the *original* adata restricted to
   `obs[split_name]=='test'`. ⇒ predict's `data_path` must still contain the test cells + the
   split column. Single-cell-line UnseenPert is moot (cov collapses to `'NOTHING'`).
6. **gene2go silent drop:** conditions whose target genes aren't all in `gene2go_all.pkl` are
   silently removed from the split dict + predict set. ⇒ coverage can shrink below the
   expected pert set without an error. Guarded by the predictor's `_preflight_model_specific`
   coverage floor (stop < 50%).
7. **`device='cuda'` hardcoded** — no CPU path ⇒ every run + smoke needs a GPU node.
8. **Predict output shape:** `predictions.h5ad` with `X` = absolute log1p, one row per
   `(cov, condition)`, obs `covariate` (cov label or literal `"none"` in single-cell-line
   mode), `condition`, `pair_key`. This matches `_container/tensor_map.py` (covariate/cell_type
   + condition; single-bin path for `"none"`). ✓
9. **Seeds (F1, fixed 2026-08-14):** the split seed (`prepare_split(seed=…)`) and all RNGs
   (`random`/`numpy`/`torch`/`cuda`/pyg) are now driven by `config['seed']` via
   `GEARSWrapper._set_seed()` (called in `__init__`). **Previously** `prepare_split(seed=42)`
   and `seed_everything(0)` were hardcoded and the config seed was inert. No early stopping
   (GEARS keeps best-val by `mse_de` internally). We do **not** force
   `torch.use_deterministic_algorithms(True)` (raises on PyG scatter), so same-seed runs are
   close but not bitwise-identical — residual CUDA nondeterminism is expected.
10. **`X` must be log1p** — GEARS trains on log1p expression; the combined h5ad must carry
    log1p `X` (asserted by the predictor preflight, B8).

### 1.4 `hyperparameters` keys

`batch_size, test_batch_size, use_mse_loss, epochs, lr, weight_decay, hidden_size,
num_go_gnn_layers, num_gene_gnn_layers, decoder_hidden_size, num_similar_genes_go_graph,
num_similar_genes_co_express_graph, coexpress_threshold, uncertainty, uncertainty_reg,
direction_lambda`. Values are pinned in **`docker/gears/model.yaml`** — the GEARS authors'
recommended defaults (model_initialize/train signatures), incl. `epochs: 20` (the atheus
config had halved it to 10). `use_direction_loss` was dropped (the wrapper never reads it).

### 1.5 Why GEARS needs the inline (combined) form — it's native, not an atheus quirk

GEARS' own `PertData` (upstream `gears/pertdata.py`) has one data path and its split unit is
the **condition**, never the cell:
- `prepare_split(...)` reduces every split type (incl. `custom`) to
  `self.set2conditions = {'train':[conds], 'val':[…], 'test':[…]}` — condition lists.
- `get_dataloader(...)` builds batches by `for p in set2conditions[split]:
  cell_graphs.extend(dataset_processed[p])` — it unions **all** cells of condition `p` (across
  every cell type) into whichever split `p` is listed in. There is **no per-cell or
  per-(cell_type,condition) gating anywhere.**

Consequences:
1. Condition lists are **marginal** → a condition cannot be train-for-covA and test-for-covB.
   In a multi-cell-type dataset that leaks. GEARS was designed for single-cell-line
   perturb-seq, where a condition is uniquely train or test, so this was never a problem.
2. The atheus wrapper's per-cell filter (drop `test`-labelled cells before `new_data_process`,
   §1.3.2) is the **minimal correct patch** making GEARS' condition-keyed model leak-safe
   under many cell types — and it works only if the container is handed the per-cell column.
3. GEARS has **no API to consume pre-partitioned files**. Feeding it `train_h5ad`+`val_h5ad`
   would force concatenating them back into one adata with a `split` column and rebuilding
   `set2conditions` — i.e. reconstructing the combined form anyway. Pre-subsetting fights
   GEARS' data model.

So the inline form is **GEARS' native shape**. The split machinery + DataLoader are pure
GEARS; the leak patch is the atheus wrapper. Cell-aware models with per-cell dataloaders
(PRESAGE/STATE) could take a pre-subsetted form directly — that's what per-model variation is for.

---

## 2. Design decisions (why it's wired this way)

### 2.1 The shared atheus convention — why the wiring generalizes

*Verified by reading two containers (`gears`, `presage`). The other atheus models
(`scgpt`, `fmlp`, `pdae`, `sclambda`) are expected to share it but **not yet verified** — read
each before adding. STATE and cellflow are **not** in atheus → separate vendoring.*

The atheus PMOB fork standardized all its model containers on one convention; GEARS and
PRESAGE independently share: **entry** `run_model.py {train|predict} /config.json`; **config
schema** (`mode, data_path, split_name, {train,val,test}_conditions, covariate_key,
hyperparameters, output_dir/output_path/model_path` + model-specific extras); **intake** = ONE
combined h5ad + a per-cell `split_name` column + condition lists (not pre-subsetted files;
both re-read the full h5ad at predict); **our-side in-container need** = exactly `DataManager`
+ `PathEncoder`; **output** = `predictions.h5ad`, absolute expression, `covariate` + `condition`.

Consequences for reuse (current layout after the 2026-08-14 cleanup):
1. The `DataManager`+`PathEncoder` shim is ~20 trivial lines **inlined per model wrapper**
   (was the shared `docker/_harness_container/harness.py`; inlining dropped the extra bind mount).
2. The host side is **one shared `ContainerPredictor`** (`benchmark/predictors/container_predictor.py`)
   + **one config-builder** (`benchmark/predictors/_container/config.py`) emitting the
   combined-form config. Per-model variation is confined to: `model.yaml` (sif/entry/binds/
   hyperparameters/gene2go/wandb), the predictor's `.scenarios` (supported regimes), and a few
   model-specific config keys. **No per-model dataloader rewrite.**
3. The **combined form is the canonical contract** (`../CONTRACT.md`); the pre-subsetted
   `train_h5ad`+`val_h5ad` form is used by no vendored model and was dropped from the contract.
4. **Leakage is a column-native check**, not file subsetting: `_container/leakage.py` reads
   `obs[split_name]` directly (disjoint cell sets; no held-out (cov,pert) pair in train/val;
   condition-list partition for UnseenPert) — more faithful than checking a materialized copy.
   (`materialize_splits.py` was deleted in the cleanup.)

**Honest scope note.** Because atheus already containerized these models on a shared contract,
our infra is *not* "rebuild each model's dataloader." It is: **own the split columns, provide
them in the atheus-native form, independently verify each model honored them, and score every
model's output uniformly with no external prediction adoption.**

**D1 — Column-native (combined-form) data feed.** Feed GEARS one h5ad + the real split column
→ reuses its tested, regime-aware, leakage-safe path and satisfies predict's need for test
cells. Leakage is enforced *and independently verified* by the pre-launch column-native gate
(`_container/leakage.py`); the wrapper's own per-cell filter is belt-and-suspenders on the
same trusted column.

**D2 — Host-side adapter translates → native config; wrapper stays byte-identical.**
`ContainerPredictor` + `_container/config.py` build GEARS' native `config.json`. The vendored
`gears_wrapper.py`/`run_model.py` are copied **unchanged** except the D3 shim (now inlined).
All schema idiosyncrasy lives in host-side Python we unit-test.

**D3 — Own the data-loading utility; drop the `cellsimbench` dependency entirely (no shim
package).** The GEARS path touched only two cellsimbench symbols — `DataManager` (an h5ad
reader + an unwanted DEG gate) and `PathEncoder` (a trivial `json` encoder). We reimplement
both ourselves (DataManager = `sc.read_h5ad` + existence check, **no DEG gate**) and they now
live **inlined at the top of `gears_wrapper.py`** (originally a separate owned file). The
container holds no external `cellsimbench` namespace, real or look-alike. Remaining
`cellsimbench` *mentions* in the wrapper are inert: `proj_name` W&B string, filenames
(`cellsimbench_gears`, `cellsimbench_split_dict.pkl`), the `_convert_to_cellsimbench_format`
method name, and a **lazy** `OneHotLinearRegressionModel` import (residualization only) — none
are runtime dependencies.

**D4 — Pinned Singularity image.** `gears.def` (Bootstrap: docker, base pinned **by digest**),
GEARS fork pinned **by commit SHA**, pip deps pinned by version — but env-only (code
bind-mounted; see `BUILD_ENV.md`).

---

## 3. Bug-guard checklist (the "no surprises" table)

| # | Failure mode | Guard |
|---|---|---|
| B1 | control labelled `ctrl` silently dropped (§1.3.1) | preflight asserts a usable control label + ≥1 control cell in train |
| B2 | missing/incorrect `split_name` ⇒ silent cell-aware leakage (§1.3.2) | config always sets `split_name`; pre-launch column-native leakage gate |
| B3 | predict can't find `processed_data/` (§1.3.3) | fit `output_dir==checkpoint_dir==_run_dir`; predict `model_path==_run_dir`; assert `processed_data/` present before predict |
| B4 | stale PertData cache reused (§1.3.4) | `fit()` rmtrees the run dir first (fresh per dataset/regime/fold) |
| B5 | predict emits in-distribution dump for cell-aware (§1.3.5) | predict `data_path` = h5ad **with** test cells; assert `sum(split=='test')>0` for cell-aware |
| B6 | gene2go silently shrinks pert set (§1.3.6) | preflight coverage floor; log dropped perts explicitly (no silent cap) |
| B7 | CPU node ⇒ CUDA error (§1.3.7) | all container runs + smoke target a GPU node |
| B8 | wrong `X` space (not log1p) ⇒ garbage (§1.3.10) | preflight assert on `X` log1p-range |
| B9 | runtime download breaks offline | gene2go passed explicitly; smoke ran offline |
| B10 | PertData writes outside the rw mount | writes land under `/model_output` |
| B11 | ~~covariate column name mismatch (`covariate_field` vs `covariate_key`)~~ | **resolved (F4):** one canonical `covariate_key`; `covariate_field` removed |
| B12 | tensor_map cov mismatch for `"none"` legacy output | single-bin path; re-asserted in the `tensor_map` self-test |

---

## 4. Adversarial review — what to watch (unknowns, bug risks, cheats)

`[VERIFIED]` = checked against real files; `[OPEN]` = genuine unknown; `[RULING]` = needs a decision.

### 4.A Genuine unknowns
- **U2 — millerh1/GEARS fork vs upstream.** `[OPEN]` The image installs the **fork**; it may
  differ in `new_data_process`, `get_dataloader`, or save/load format. Diff `pertdata.py` at
  build; the offline smoke passed.
- **U3 — co-expression-graph transductive leakage (IMPORTANT).** `[RULING]` For UnseenPert the
  per-cell filter is skipped, so `new_data_process` sees test-perturbation cells and builds
  GEARS' gene–gene co-expression graph over them. Folding test-perturbation *expression* into a
  gene-gene prior — leak or not by our standard? Upstream GEARS does exactly this, and the
  reference numbers were calibrated with it. **Stance:** match GEARS' published protocol (the
  graph uses no split *labels* or (cov,pert) *pairing*, only marginal co-expression; for
  cell-aware regimes test cells are already dropped, so it's moot there). A stricter standard
  (rebuild the graph from train+val only) would **not** reproduce the reference — expected, not noise.
- **U7 — gene-ID space.** `[VERIFIED for adamson16]` var_names are HGNC symbols; gene2go is
  symbol-keyed; targets ∩ gene2go = **88/89** (only `C7orf26` drops). **Re-check per dataset** —
  an Ensembl-keyed dataset would need symbol mapping or every condition drops.
- **U8 — compute/memory.** `[OPEN]` Predict is memory-heavy (rebuilds a PyG graph per unit in
  a multiprocessing pool): OOM'd at 32G, OK at 128G for adamson16. Measure before large datasets.
- **U9 — gene-coverage × scoring (reproduce-gate).** `[OPEN]` GEARS predicts over its modeled
  gene set (≈71% of var in gene2go); the benchmark leaves GEARS-missing genes NaN (skipped),
  never zero-filled. If the reference scored on a different gene set we won't reproduce —
  confirm the reference's gene-scoring convention; report the modeled-gene count.

### 4.B Bug risks
- **BUG-A — UnseenPert condition-list derivation** must be a clean condition-level partition
  (train ∩ test = ∅); GEARS relies on it entirely (per-cell filter skipped). Asserted in preflight.
- **BUG-C — GEARS predict config ≠ canonical predict.** Feed `data_path` (full h5ad *with* test
  cells) + `test_conditions`, **not** a `context_h5ad`; the latter → silent fallback to
  *training* covs (wrong for UnseenCell).
- **BUG-F — covariate label alignment.** Cell-aware output `covariate` labels must be ⊆ store
  `bin_names`; mismatch → NaN → zero-scored → the model looks dead due to a *mapping* bug.
- **BUG-G — the column-native leakage check has TWO modes.** UnseenPert → condition-list
  disjointness; cell-aware → no held-out (cov,pert) pair among train/val cells. Both implemented
  in `_container/leakage.py`; one alone misses a whole class.

### 4.C Shortcut / cheat temptations (named so we don't take them)
- **CHEAT-5 (the big one) — treating adamson16/UnseenPert as sufficient validation.** It runs
  the cov-agnostic `'NOTHING'` path and skips the per-cell filter, cov-aware predict, and
  test-cov enumeration — **none** of the leakage-critical machinery. Validation must include a
  cell-aware case (replogle22 UnseenBoth) before the container is "working" for those regimes.
- **CHEAT-3 — loosening reproduce-within-noise.** Pre-register the target + band using the
  authors' exact hyperparameters + same fork commit *before* running; pass/fail honestly.
- **CHEAT-6 — shrugging off gene/pert drops.** Quantify coverage; if kept < a set fraction,
  **stop** (likely ID mismatch) rather than proceed and call it fine.
- **CHEAT-2 — "it ran and the Pearson looks plausible."** A plausible metric is not evidence of
  correctness; the cheap preflight asserts must run every time.

---

## 5. Run reports (added 2026-09-07)

GEARS declares `has_drop_rule = True` and drops every condition whose targets are outside
`gene2go` — at `gears_wrapper.py:~1135` for the train/val/test split lists, and again in the
predict path for test conditions. Both drops were **logged and discarded**: the wrapper wrote
no report at all, so a scored run carried no machine-readable record of what it had omitted,
and an all-NaN row in the metrics could not be explained without re-reading the SLURM log.
This is the same defect class as scGPT's V9 (see `docker/scgpt/SCGPT_NOTES.md` §7).

Two files are now written, mirroring scGPT:

| file | contains | required? |
|---|---|---|
| `gears_training_report.json` | `n_cells_total` / `n_cells_trainval` / `n_test_cells_in_training` from the per-cell split filter, `dropped_conditions_by_split` + `n_dropped_conditions` from the gene2go filter, `epochs` / `lr` / `weight_decay`, seed and split name | **yes** — in `model.yaml: expected_artifacts` |
| `gears_predict_report.json` | `dropped_test_conditions` — the conditions predict could not model | no (predict runs after the completeness check) |

`GEARSContainer._train_report` lifts the first into `fingerprint.json["report"]`, where it is
**descriptive, never identity**: staleness compares only `TrainedPredictor._ENFORCED`
(`seed`, `gene_axis_sha`, `split_sha`, `recipe_sha`), so a report may carry timings without
invalidating the checkpoint it describes.

> **Putting the training report in `expected_artifacts` invalidates every GEARS-ct checkpoint
> trained before this change.** `TrainedPredictor.unusable_reason` requires each expected
> artifact to exist, so pre-existing folds now read as incomplete and must be retrained. That
> was the deliberate choice: a run whose drop set cannot be recovered should not count as
> trained.
