# PRESAGE container — notes & design reference

Companion to `../CONTRACT.md` (the machine contract) and `../gears/BUILD_ENV.md`
(the build route, which is identical). This file records what PRESAGE *is*, what
changed when it was vendored, and which of its behaviours will bite if unhandled.

Sources:
- upstream **`github.com/Genentech/PRESAGE`** @ `2c7b231c60cf110c4aab0acf8ecd2c2b3268fffb`,
  **Genentech Non-Commercial Software License v1.0** (see `LICENSE`, `NOTICE`,
  `MODIFICATIONS.md` — these must travel with the code; `.gitignore` un-ignores
  `docker/**/*.md` for exactly this reason). Do not push the image to a registry.
- integration wrapper vendored from
  `/cluster/work/boeva/atheus/Perturbation-Models-Outperform-Baselines/docker/presage/`
  (`presage_wrapper.py`, `cellsimbench_datamodule.py` → `presage_datamodule_csb.py`).

---

## 1. What PRESAGE is

A **perturbation-label → Δexpression** regressor. It never sees expression as
input. Given the perturbed gene(s) and (optionally) a cell-type covariate, it
predicts the pseudobulk delta against the matching control mean, over the whole
measured gene panel.

Forward pass (`ComboPRESAGE`, a local extension of upstream's model adding combo
and covariate support):

```
pert_inds (binary, n_genes)
  └─ gather frozen prior gene embeddings        (n_genes, emb_dim, n_pathways)
       from the Zenodo pathway-embedding cache; median-norm normalised + masked
  └─ GeneEmbeddingTransformation (MLP) + LeakyReLU      pathway_item_* params
  └─ SUM over the genes in a combo                      additive-in-embedding-space
  └─ Pool (GAT-weighted, sum)                           gat_weight, softmax_temperature
  └─ + cov_embed(one_hot(cell_type))                    ← the cell-aware part
  └─ ItemNet (MLP)                                      item_hidden_size
  └─ Δexpression (n_genes)
```

Loss is plain MSE on the delta (`compute_loss` is `((pred - expr) ** 2).mean()`).
The `mse/cosine/vector_norm_loss_scale` hyperparameters are **inert** — carried
only because upstream's config schema expects them.

**Why PRESAGE matters here:** it is the first container model that genuinely
conditions on cell type, so it is what makes `UnseenCell` / `UnseenBoth` /
`UnseenPair` and the M1.5 cross-cell-type leakage validation possible at all.
GEARS runs those regimes only through a degenerate covariate.

### 1.1 Why it is leak-safe where GEARS is not

PRESAGE slices on the **per-cell** split column, not on marginal condition lists:

```python
split_adata = adata[condition_mask | control_mask]
split_adata = split_adata[split_adata.obs[self.split_name] == name]   # hard per-cell filter
```

That second line is the whole difference. A condition-list split cannot express
"this KO is held out *in RPE1* but trained in K562", which is precisely what the
cell-axis regimes require. `V11` asserts the post-condition so a future edit that
loosens the filter fails loudly instead of quietly training on held-out cells.

---

## 2. Vendoring changes (`VENDORED (Vn)` markers in the source)

Each is marked in-place with the same `Vn` tag used below.

| tag | change | why |
|---|---|---|
| **V1** | drop the `/capstor/store/cscs/...` (CSCS Alps) and `ref/PRESAGE` source candidates; keep `$PRESAGE_SRC` → `/presage_src`; raise naming the paths tried | dead paths on this cluster; a silent fallthrough surfaced three frames later as a bare `ImportError` |
| **V2** | inline `PathEncoder` + a minimal `DataManager`; drop the `cellsimbench` imports and the `/cellsimbench_datamodule.py` existence probe | same de-vendoring `gears_wrapper.py` did — no host package to install, no separate harness bind |
| **V3** | read `seed` from the **top-level** config, not `hyperparameters` | our contract emits it top-level for both modes; the original would have raised, or (with a stray recipe seed) pinned every run and turned a seed-stability check into a CUDA-nondeterminism measurement. GEARS' F1, again |
| **V4** | `chdir` into the run dir for training; seed `./cache` as a **symlink farm** over the baked cache; drop the post-hoc `copytree` | upstream hardcodes `./cache/pathway_embeddings/` relative to CWD **and writes to it**. Training previously ran at whatever CWD the job started in, so concurrent folds raced (`predict` already carried a tempdir workaround for the same corruption). Anchoring it in `/model_output` fixes that and removes the writable-overlay the atheus runner needed. Symlinks rather than a copy because the baked cache is **6.4 GB** — copying per fold would cost ~64 GB over a ten-fold sweep to duplicate bytes that never change. PRESAGE only ADDS files here, so new ones land real in the run dir; an attempted overwrite fails EROFS against the read-only image rather than corrupting the shared cache. Predict's restore uses `copytree(symlinks=True)` for the same reason |
| **V5** | save the **best-validation** weights, not the last epoch's | the original saved `harness.state_dict()` after `fit`, leaving `ModelCheckpoint(save_top_k=1)`'s file unused — so with `EarlyStopping(patience=10)` the scored weights were, by construction, ~10 epochs past the optimum. Now raises rather than silently falling back |
| **V6** | resolve every artefact **relative to `model_path`**; record relative paths; reject absolute ones | the original recovered train-time paths via `.replace("/model_output/", "/pretrained_model/")`, an atheus two-mount convention. Our contract binds one dir at `/model_output` for both calls, so that rewrite pointed at paths that never exist |
| **V7** | remove `perts_as_delta_ref` as a tunable (pinned `False`) | its inference half was never implemented — `_generate_predictions` logged "not implemented yet" and used the control mean anyway — so `true` would have trained against one reference and predicted against another |
| **V8** | assert gene-name uniqueness instead of `var_names_make_unique()` | the rename produces `GENE-1`, which matches nothing in `store.gene_names` and is dropped by the host's gene intersection — silently, since `tensor_map`'s guard fires on duplicates, not renames. Verified: **zero** duplicate gene names across every processed dataset, and where a `gene_name` column exists it is identical to the index |
| **V9** | predict uses the explicit `test` split instead of inferring it | the original read `obs.loc[obs.pert == matched, split_name].iloc[0]`. Under the cell-axis regimes a held-out perturbation legitimately also appears in `train` cells of other cell types, so this could return `"train"` — and the "test" set would be built from **training cells** and scored as held out. Silent, and it inflates the score |
| **V10** | refuse to build *training* rows centred on a control mean borrowed from outside the split | `compute_means` falls back to full-adata controls when a covariate has none in-split. At inference that is legitimate (every predictor gets the held-out cell type's basal state; the benchmark subtracts control itself). At training it puts held-out cells into the loss |
| **V11** | assert train/val datasets contain no off-split cells | see §1.1 — make the guarantee structural, not incidental |
| **V12** | lift wrapper-hardcoded hyperparameters into `model.yaml`; honour `num_heads` instead of overwriting it; require the early-stopping settings | a hyperparameter the recipe cannot express makes "the recipe is the single source of truth" false. Same defect class as GEARS' silently-halved `epochs` |
| **V15** | write the per-run `cellsimbench_processed.h5ad` with `compression="gzip"` | the wrapper densifies X before writing (and `load_preprocessed` densifies again on read), so this is a dense float32 copy of the entire input, written once PER FOLD. Measured 2.7 GB for adamson16; extrapolates to ~13.6 GB for replogle22 and ~32 GB for jiang24, on a volume already at ~100%. Compression is free of behaviour change |
| **V14** | wire W&B into the Lightning trainer (`logger=False` was hardcoded), and make it **fail safe to offline** | `model.yaml` declares `wandb_project` and `ContainerPredictor._train` injects `wandb`/`wandb_project`/`wandb_run` into `config.json` — and the wrapper read none of it. A declared setting that does nothing is worse than no setting: it made `TRAINING_VALIDATION.md`'s Gate A ("per-run training dynamics from W&B") impossible while looking configured. Endpoint/credentials come from `WANDB_BASE_URL`/`WANDB_API_KEY` in the job env; singularity passes the host env through (no `--cleanenv`). Wiring it also made a dead tracking server dangerous — an online logger with no endpoint blocks on login in a tty-less batch job and would take a multi-hour run with it — so with no `WANDB_API_KEY` visible the wrapper forces `WANDB_MODE=offline`. Metrics then land at `<run_dir>/wandb/offline-run-*` beside the checkpoint, and `wandb sync <dir>` uploads them when the server returns. An explicit `WANDB_MODE` always wins |
| **V13** | resolve the covariate from `covariate_key` only; drop the `covariate_field` fallback | `covariate_field` is the key F4 deleted host-side. Nothing rejects it any more — it is not in `contract._KNOWN_FIELDS`, so `_check_extras` passes it through as a scalar extra, and a stray `covariate_field:` under `extra_config:` in `model.yaml` would silently override the canonical covariate on the one model whose whole claim rests on it. Rejecting it host-side was declined: that would put a dead model-specific key back into the shared schema, which is what `_KNOWN_FIELDS` exists to prevent. Deleting the reader leaves nothing to guard |

**Not vendored** (deliberately): `build.sh` (docker-only; we build SIFs on
CustomApps), `utils.py` (only `_normalize_perturbation` was on the live path),
`presage_gated.yaml` (an atheus-invented gated-residual variant — its code is not
even in the readable source tree), the `cellsimbench` host package, and
`slurm/sbatch_presage.sh` in this repo (dead: it calls a `presage/` directory that
does not exist).

---

## 3. Hyperparameter audit vs the authors (DONE, 2026-08-26)

Run against upstream `Genentech/PRESAGE` @ `2c7b231`. **Upstream publishes no
hyperparameter config**: `train.py`'s argparse default points at `./config/gears_adata.yml`
and `shell_scripts/run_presage.sh` at `./configs/singles_config.json` — neither
directory exists in the repo. So there are exactly two recoverable author
reference points, and **they disagree with each other**:

- **(A) argparse defaults** in `src/train_presage.py` — what you get running it bare.
- **(B) the authors' own sweep** embedded at `train_presage.py:366-500`, under
  `if args["data.dataset"] in single_datasets:` — i.e. the config they actually
  tuned for **single-perturbation** screens, which is our regime (adamson16,
  replogle22).

**25 of 27 comparable values match (A) exactly**, including every value that had
been hardcoded in the atheus wrapper body — `lr 2.4e-3`, `weight_decay 2.17e-13`,
`momentum 0.9`, `Adam`, `batch_norm True`, `n_neigh_prune 5`. V12's lift into
`model.yaml` preserved author-faithful values rather than inventing them. Upstream
also hardcodes `num_heads = 4` (`presage.py:131`), matching ours.

### The four genuine deviations

| hyperparameter | ours | (A) argparse | (B) authors' sweep | note |
|---|---|---|---|---|
| `batch_size` | **32** | 16 | **256** | matches **neither** — an atheus choice |
| `softmax_temperature` | 0.15 | 0.15 | **0.1** | ours = (A) |
| `lr` | 2.4e-3 | 2.4e-3 | **log-uniform 5e-4 … 1.5e-3** | ours = (A), but **1.6× above the sweep's upper bound** |
| `weight_decay` | 2.17e-13 | 2.17e-13 | **1e-15** | ours = (A) |

`max_epochs: 1000` is **confirmed correct**: it matches the sweep exactly, and the
argparse `10000` is just a ceiling.

Everything else in the sweep agrees with ours: `n_nmf_embedding 128`, all six
node2vec params, `n_neigh_prune 5`, `pathway_item_{hidden_size 128, nlayers 2}`,
`pathway_pool_type sum`, `pathway_weight_type gat`, `pool_nlayers 2`,
`gat_weight 0.85`, `batch_norm True`, `learnable_gene_embedding False`,
`dim_red_alg Node2Vec`, `item_hidden_size 512`, `item_nlayers 0`,
`use_pseudobulk True`, `preprocessing_zscore False`, `noisy_pseudobulk False`.

### Decision (2026-08-26): pinned to (B), the authors' sweep

`model.yaml` now carries `batch_size 256`, `softmax_temperature 0.1`,
`weight_decay 1e-15`. For `lr` the authors swept a *range* rather than fixing a
value and did not publish the winner, so we take **8.66e-4, the geometric mean of
5e-4…1.5e-3** — the midpoint under their own log-uniform prior. That number is
**ours**, and is labelled as such in the recipe; the argparse 2.4e-3 is 1.6× above
their upper bound and so cannot be what they tuned to.

Every sweep key we do *not* set was checked: for all of them the argparse default
**equals** the sweep value (`min_genes_per_kg 10`, `gex_coexpression_style
"coexpression"`, `min_cos -1.0`, `contrastive_loss_scale 0`, `use_pseudobulk
True`, `preprocessing_zscore False`, `noisy_pseudobulk False`), so omitting them
is safe under either reference.

> **Consequence to watch on small datasets.** The authors swept on
> `replogle_k562_essential_unfiltered` (thousands of perturbations). adamson16's
> training split is **62 pseudobulk rows**, so `batch_size 256` is effectively
> *full-batch* there — one optimiser step per epoch. That is a faithful
> application of their config, not a bug, but it changes training dynamics
> substantially versus atheus's 32, and it is exactly what Gate A's loss curves
> exist to catch. Expect the epoch count at which early stopping fires to differ.

**Upstream exposes no early stopping at all** — no `EarlyStopping`, no
`ModelCheckpoint`, no `patience` anywhere in `train_presage.py`. Our
`early_stopping_patience`/`min_delta` and the best-val checkpoint selection (V5)
are the CellSimBench harness's regime, not the authors'. That is defensible — it
is what makes `max_epochs: 1000` safe — but it must be reported as ours, not
theirs.

## 4. Things that will bite

- **`predict` needs the training artefacts**, not just weights: it rebuilds the
  datamodule from `presage_data/cellsimbench_processed.h5ad` + `splits.json` and
  restores `cache/`. All are in `expected_artifacts`; none are regenerable.
- **W&B server down?** Not a blocker. Runs log offline into the run dir (V14); when
  `ctr-biomed-15:8082` is back, `wandb sync <run_dir>/wandb/offline-run-*` uploads
  them with their original timestamps. Do not disable `wandb_project` to work
  around an outage — that silently forfeits Gate A's loss curves for that run.
- **Memory.** PRESAGE pseudobulks per `(covariate, perturbation)`; atheus ran it at
  64 G, and 192 G for jiang24/mcfaline23. Size from the smoke, don't assume. The
  driver is `load_preprocessed`, which densifies the whole input in RAM
  (`presage_datamodule_csb.py`): ~1.9 GB for adamson16, ~13.6 GB for replogle22,
  ~32 GB for jiang24. V15 compresses the on-disk copy but does **not** change this
  — the RAM cost is inherent to the datamodule and is what sets `--mem`.
- **Disk per run.** The run dir's cache is a symlink farm (V4), so it should be a
  few MB, not 6.4 GB. The smoke asserts symlinks are present. Do not "fix" any
  future cache problem by sharing one writable cache directory across folds —
  that is precisely the race V4 exists to remove.
- **Covariate auto-disable.** The wrapper sets `covariate_field = None` when the
  column is missing or has `< 2` unique values, and the model then silently stops
  conditioning. `ContainerPredictor._preflight` now makes that a hard failure for a
  `cell_aware` model in a cell-axis regime, rather than something you find out from
  a suspiciously flat result.
- **UnseenCell has a real methodological limit**, not a bug: a one-hot covariate
  category that never appeared in training has an untrained embedding. Report it;
  do not engineer around it. See `TRAINING_VALIDATION.md`.
- **Image size.** The atheus Docker equivalent is 19.2 GB (their largest) — conda
  env + the Zenodo cache. Set `SINGULARITY_TMPDIR` off `/tmp` when building.
- **x86-64 only** (atheus recorded the image failing to run on aarch64). Fine on
  LeoMed.

---

## 5. Adversarial review — what to watch

**Genuine unknowns.** Whether upstream's own defaults match the two provenances in
§3 (open). Whether `cov_embed` meaningfully changes predictions at all — that is
what the M1.5 positive control tests, and it is not answerable from adamson16.

**Shortcut temptations, named so we don't take them.**
- *"The cell-axis regimes are declared, so PRESAGE supports them."* Declaring
  `cell_aware = True` is a claim, not evidence. The gate is the positive control.
- *"It ran leak-free on adamson16, so the leakage machinery works."* adamson16 is
  single-cell-line: it exercises the degenerate-covariate path and skips everything
  the pair-mode gate exists for. This is `GEARS_NOTES.md` CHEAT-5 restated.
- *"V5 changed the number, so compare it to the atheus drop."* The atheus drop was
  produced by the last-epoch bug. It is a Gate-C corroborating cross-check, not a
  reproduce-within-noise bar.


---

## 6. Build provenance (first successful build, 2026-08-26)

Built on the CustomApps client per `../gears/BUILD_ENV.md`; `.sif` lives at
`/cluster/customapps/biomed/boeva/akalygina/singularity/presage/` and is
symlinked into this directory. Facts worth keeping, because each cost a build
cycle to learn:

| fact | detail |
|---|---|
| image size | **6.75 GB** (GEARS is 3.4 GB) |
| baked cache | **6.4 GB** at `/opt/presage_cache`, from Zenodo record 15587986 |
| upstream licence | ships as **`LICENSE.txt`**, not `LICENSE` — a hardcoded `cp .../LICENSE` failed the first build. Now globbed |
| cache permissions | the Zenodo tarball carries mode **0750** on its directories, and `tar --no-same-owner` keeps the mode while remapping ownership to the build user (root, under fakeroot). At run time we are not root, so the cache was **unreadable** and `_seed_cache` would have failed on the first training run. Fixed with `chmod -R a+rX` plus a `find` assertion |
| numba / matplotlib | `import scanpy` triggers `@njit(cache=True)`, which wants to write next to a read-only source. It survives only because singularity bind-mounts a writable `$HOME` — luck, not design. `NUMBA_CACHE_DIR` / `MPLCONFIGDIR` are now set in `%environment` |
| base image | **not digest-pinned.** `condaforge/mambaforge:latest`; reproducibility rests on the upstream conda lock plus `/opt/presage_lockfile.txt`. See the DIGEST note in `presage.def` |

**Why there is no `%test`.** It had one and it was actively misleading: `%test`
runs **as root**, so it reported `cache OK` for a cache the real user could not
read, and its genuine failures did **not** abort the build (the `.sif` was
written anyway). The checks that matter are now `%post` assertions, which *are*
fatal — `test -f /presage_src/presage.py`, `test -d .../pathway_embeddings`, and
the world-readable `find`. Runtime verification belongs on the cluster against
real conditions (non-root, real bind mounts), which is what actually caught the
permissions bug.


---

## 7. Smoke result (2026-08-26, job 9529035, V100)

adamson16 / UnseenPert / fold0, `max_epochs=1`, recipe pinned to the authors'
sweep. **SMOKE OK.**

```
TRAIN rc 0     18.5 s | best_epoch 0 | best_val_loss 0.0025558
               n_train_rows 62, n_val_rows 9
               batch_size_configured 256 -> effective 62   (V17 clamp)
PREDICT rc 0   predictions (18, 8250), finite 1.0, range -0.049 .. 4.403
delta tensor   (1, 18, 8250), finite 1.000
disk           run dir 957 MB | presage_data 494 MB | cache 323 MB
               cache = 112 symlinks + 1 real file
```

### What the earlier attempts caught

The run before this one (job 9529014) **failed**, and that was the point:
`batch_size 256` against a 63-row split gave `len(train_loader) == 0` under
upstream's `drop_last=True`, so Lightning ran **zero optimiser steps** and exited
cleanly. **V5's refusal to save unvalidated weights is what stopped it** —
otherwise the run would have written an untrained checkpoint and produced
predictions from random initialisation, and every downstream check would still
have passed, because the delta tensor would have been the right shape and
perfectly finite. That is the exact failure the gates exist for.

### Measured effect of the fixes

| fix | before | after |
|---|---|---|
| V4 symlink farm | would copy 6.4 GB/fold | 112 symlinks, 4.5 KB for the 5.4 GB `other_embeddings` |
| V15 gzip | `presage_data` 2.7 GB | **494 MB** (5.5x); run dir 3.2 GB -> 957 MB |
| V17 clamp | 0 training batches | 62 (full batch), both values recorded |

V15 extrapolates replogle22's per-fold copy from ~13.6 GB to roughly 2.5 GB,
which is what makes a fold sweep viable on a full volume.

### What this still does NOT show

adamson16 is single-cell-line: `cov_categories=None`, `n_covariates=0`, so the
covariate embedding was never constructed and the cell-aware path — the reason
PRESAGE is here — remains unexercised. Nothing in this run bears on the M1.5 gate.
