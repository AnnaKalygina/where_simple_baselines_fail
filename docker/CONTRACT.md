# Training-container contract

Authoritative design: `../TRAINING_INFRA_PLAN.md`; per-container notes:
e.g. `gears/GEARS_NOTES.md`. This file is the **machine contract** every model
container in `docker/<model>/` must honor. Entry point:

```
python run_model.py {train|predict} /config.json
```

Mounts (bound by `benchmark/predictors/_container/runner.py`): `/data` (ro),
`/config.json` (ro), `/model_output` (rw). The host writes **host** paths into the
config and the runner rewrites the input-h5ad field (`data_path`) to
`/data/<basename>` and `output_path` to `/model_output/<basename>`; the
in-container write paths (`output_dir`, `checkpoint_dir`, `model_path`) are the
constant `/model_output`.

## Canonical form: **combined (column-native)**

The container is handed **one combined h5ad** (`data_path`) that carries the
authoritative per-cell leak-safe split **inline**, as an obs column
`split_{Scenario}_fold_{N}` ∈ `{train, val, test}` (`split_name`), plus the
marginal condition lists derived from that column. The model slices itself.

This is canonical because it is the **native and only** intake of the vendored
models (verified for GEARS and PRESAGE): GEARS' split unit is the *condition*,
never a pre-partitioned file (see `gears/GEARS_NOTES.md` §1.5). A pre-subsetted
(`train_h5ad`+`val_h5ad`) form is used by **no** vendored model and is not part of
this contract.

Leakage-safety is **not** a byproduct of file subsetting here — it is enforced by
a host-side **column-native** gate (`benchmark/predictors/_container/leakage.py`,
reading `obs[split_name]` directly) before the container launches, and
independently re-guarded by the wrapper's own per-cell filter on the same trusted
column.

## The recipe — `docker/<model>/model.yaml`

Every containerised model declares how it is run in one file. It is validated on
load (`ContainerPredictor._validate_recipe`): a **missing required key is an
error, and so is an unrecognised one** — a silently-ignored typo would make
"the recipe is the single source of truth" false while the model trains with
defaults nobody chose.

| key | required | meaning |
|---|---|---|
| `model_key` | yes | the config `model` field; the container's own registry key |
| `sif_path` | yes | repo-relative path to the image |
| `entry` | yes | argv run via `singularity exec` |
| `code_binds` | yes | host dir → in-container mount |
| `hyperparameters` | yes | **the authors' recommended values**, pinned; do not tune toward a target |
| `expected_artifacts` | yes | run-dir-relative paths that must ALL exist for the run to count as trained |
| `train_timeout` / `predict_timeout` | yes | seconds; a hung container otherwise holds a GPU until SLURM reaps the job |
| `wandb_project` | no | training logs |
| `extra_config` | no | **the one place** model-specific keys go (GEARS puts `gene2go_path` here). They are merged into the container's config verbatim |

Model-specific settings belong under `extra_config`, never as new top-level keys:
the top-level schema is closed, so a private key added there is rejected as
unrecognised — deliberately. Otherwise the allowed-key set would grow into the
union of every model's private vocabulary and the unknown-key check would stop
catching typos.

`expected_artifacts` is what makes "is this trained?" answerable. Declare every
file predict actually needs — including caches predict RELOADS rather than
rebuilds. GEARS' `processed_data/` is the cautionary case: its predict path
raises if the cache is missing instead of regenerating it, so the cache is
durable state, and pruning it leaves a checkpoint that can never predict again.

Capabilities (`scenarios`, `cell_aware`, `has_drop_rule`) are declared in Python
on the predictor class, NOT here — `verify` reads them in a pyyaml-less env.

## TRAIN — `/config.json`

*(The container always reads `/config.json`. On the host the file is written to
the run dir as `train_config.json` / `predict_config.json`, so a predict run no
longer overwrites the record of how training was configured. Nothing in the
container changes.)*
```jsonc
{
  "mode": "train",
  "model": "gears",
  "dataset": "adamson16",
  "scenario": "UnseenPert",           // PascalCase-native
  "regime":   "UnseenPert",           // == scenario; explicit provenance
  "fold": 0,
  "seed": 42,                          // drives all RNGs + the GEARS custom split
  "data_path": "/data/adamson16_processed.h5ad",  // ONE combined h5ad (host path in; rewritten to /data/…)
  "split_name": "split_UnseenPert_fold_0",        // per-cell obs column: train/val/test
  "covariate_key": "cell_type",        // the single canonical covariate obs column
  "train_conditions": ["AARS", "..."], // unique obs.condition where split==train (control/'*' removed)
  "val_conditions":   ["..."],         // …==val
  "test_conditions":  ["..."],         // …==test  (the held-out perturbations)
  "output_dir":     "/model_output",   // model + processed_data/ written here
  "checkpoint_dir": "/model_output",   // resume dir (same)
  "gene2go_path":   "/app_code/gears/gene2go_all.pkl",  // from `extra_config:` — GEARS-specific,
                                                        // bind-mounted from the host repo (not baked).
                                                        // PRESAGE sends `presage_cache_path` here instead.
  "hyperparameters": { /* per-model, pinned; see docker/<model>/model.yaml */ }
}
```
The condition lists are a **verbatim** reproduction of the reference derivation
(`cellsimbench.core.data_manager.get_perturbation_conditions`): per-split
`obs.condition.unique()`, then drop any label containing the substring `ctrl` or
equal to `*`. They are marginal metadata; the authoritative leak-safe assignment
is the per-cell `split_name` column, which the wrapper's per-cell filter uses for
cell-aware regimes.

## PREDICT — `/config.json`
```jsonc
{
  "mode": "predict",
  "data_path": "/data/adamson16_processed.h5ad", // SAME combined h5ad — MUST still
                                                  // contain the test cells + split_name column
  "split_name": "split_UnseenPert_fold_0",
  "covariate_key": "cell_type",
  "test_conditions": ["..."],          // conditions to predict (the held-out perturbations)
  "model_path":  "/model_output",      // trained-model dir; the SAME host dir train wrote to.
                                       // Artefact paths recorded at train time are resolved
                                       // RELATIVE to it — never re-based onto another mount.
  "output_path": "/model_output/predictions.h5ad",   // host path in; rewritten
  "gene2go_path": "/app_code/gears/gene2go_all.pkl",   // `extra_config:` again, same in both modes
  "hyperparameters": { /* same block as train */ }
}
```
`model_path` and the train `output_dir`/`checkpoint_dir` MUST resolve to the
**same** host directory across the two calls (predict reloads
`processed_data/`). Predict re-reads the full h5ad so it can enumerate the
held-out covariate(s) and their control cells for cell-aware regimes.

## OUTPUT — `/model_output/`
1. `predictions.h5ad` — **absolute log1p expression** (the benchmark subtracts
   control itself). Required `obs`: `condition`, and `covariate` OR `cell_type`
   (a degenerate `"none"`/`"NOTHING"` covariate is allowed for single-cell-line
   runs). `var_names` = gene symbols (unpredicted genes may be omitted → NaN,
   tolerated). Mapped to the benchmark's `(n_test_bins, n_test_kos, n_genes)`
   tensor by `benchmark/predictors/_container/tensor_map.py` using the exact test
   indices — reusing the proven combo-`canon` / gene-intersection /
   covariate-case-normalize primitives (no rewrite).
2. Whatever provenance the model itself wants (GEARS writes `metadata.json`).
   Optional and model-specific — the benchmark does not read it.

   *(An earlier version of this contract required a `train_manifest.json`. Nothing
   ever wrote or read one; provenance is now the host's job — see below. Do not
   implement it.)*

### Files the HOST writes into the same run dir — do not create or delete these

| file | written by | purpose |
|---|---|---|
| `fingerprint.json` | host, after training | run identity: gene axis, split, recipe and seed hashes. `is_trained` compares it, so a panel rebuild, a regenerated split or a recipe edit invalidates the checkpoint instead of being silently served |
| `RUNNING.json` | host, during training | claim marker: stops two SLURM array tasks training into one dir. Removed when the run ends |
| `train_config.json` / `predict_config.json` | host, per call | the exact config the container was given |

A container should write only its own artefacts. Clobbering `fingerprint.json`
would make a stale checkpoint look valid — the failure this contract works
hardest to prevent.

## Regimes
`UnseenPert · UnseenCell · UnseenBoth · UnseenPair · UnseenDose · UnseenCombo`
(PascalCase only). Per-model support is the predictor's `.scenarios` class
attribute (`benchmark/predictors/container_predictor.py`); the per-model run
recipe (hyperparameters + wiring) is `docker/<model>/model.yaml`.
