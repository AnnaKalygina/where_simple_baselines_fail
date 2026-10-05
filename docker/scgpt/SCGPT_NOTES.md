# scGPT-ct — design notes

Vendored from `/cluster/work/boeva/atheus/Perturbation-Models-Outperform-Baselines/docker/scgpt/`
(`scgpt_wrapper.py`, 1757 lines). Read `docker/CONTRACT.md` first; the build route is
`docker/gears/BUILD_ENV.md` (model-independent — there is deliberately no second copy here).

---

## 1. What scGPT is, and what "training" means here

**scGPT-ct FINE-TUNES; it does not pretrain.** A foundation checkpoint —
`whole-human`, 33 M human cells, 12 layers / d_model 512 — is loaded, and then every
parameter is trained on the perturbation task. There is no freezing, no LoRA, no adapter,
and no pretraining code anywhere in the drop. Only the perturbation encoder
(`GeneEncoder(3, d_model)`, which replaces scGPT's plain `nn.Embedding` and adds a
LayerNorm) and the expression-decoder head start from scratch.

The forward pass, per cell:

```
  input : one CONTROL cell's expression profile  +  a per-gene perturbation flag
  embed : total_embs = e_token(gene) + e_value(expression) + e_pert(flag)
  trunk : 12-layer transformer encoder
  output: that cell's predicted POST-perturbation profile (absolute log1p)
```

### 1.1 Why cell identity travels through the control cell, and nowhere else

`TransformerGenerator` — scGPT's perturbation model — **has no covariate input port**.
Its `_encode` signature is `(src, values, input_pert_flags, src_key_padding_mask)`: no
`batch_labels`. Its `domain_spec_batchnorm` argument is accepted and never applied. The
`use_batch_labels` + DSBN machinery people associate with scGPT lives in
`TransformerModel`, the *cell-embedding* model used for integration and annotation — not
here. The atheus subclass goes one step further and pops `n_covariates` off its kwargs
(`:188`).

So the only channel carrying cell identity into a prediction is **the control cell fed as
input**. Two independent confirmations that the channel is real: our own
`DL_UNSEENCELL_MISSING_CELLTYPES.md:76-79` measured the legacy scGPT drop's per-covariate
prediction blocks as distinct (pairwise max|Δ| ≈ 1.8–2.8 in log space); and STATE names
the same mechanism — *"The control set is constructed by randomly sampling S control cells
from the same cell line ℓ, and optionally same batch b."*

That is why **V7 (covariate-matched control pairing) is the mechanism that makes scGPT-ct
cell-aware**, and why no covariate embedding was added. Adding one would be a fork of the
model, with randomly-initialised parameters that have no pretrained counterpart and a table
size that changes per dataset (48 batch levels in mcfaline23 vs 2 in jiang24). The
published precedent for an extra additive channel is scGenePT
(`e = e_token + e_count + e_pert + e_language`) — and note it is used at the **gene**
level, never the cell level.

### 1.2 Why it was NOT leak-safe, and what fixed it

Upstream GEARS `PertData` — which scGPT's perturbation path reuses — does two things that
break on the cell axis:

* it splits on a **condition list** (`set2conditions`). On cell-axis regimes
  `train_conditions ⊇ test_conditions`, because a perturbation held out in cell type A
  legitimately remains `train` in cell type B. Filtering cells by that list pulls held-out
  cells into training.
* it draws control donors as
  `self.ctrl_adata[np.random.randint(0, len(self.ctrl_adata), num_samples), :]` — **all**
  controls, any cell type, any split. `cell_type` is validated for presence and then never
  used.

`split_name` appeared in the atheus wrapper at exactly two lines (`:1166`, `:1171`), both
inside the opt-in weighted-residual path — i.e. **there was no per-cell split filter on the
training path at all**. The atheus GEARS wrapper *had* been patched
(`gears_wrapper.py:908-932`); scGPT never was. **V6** applies that patch and **V7**
rebuilds the pairing.

---

## 2. Vendoring changes (`VENDORED (Vn)` markers in the source)

Each is marked in-place with the same `Vn` tag used here.

| tag | change | why |
|---|---|---|
| **V1** | inline `PathEncoder` + a minimal `DataManager`; drop the `cellsimbench` imports | Same de-vendoring `gears_wrapper.py` and `presage_wrapper.py` did — no host package to install, no extra bind mount. |
| **V2** | do not vendor `weighted_residual_loss`, `utils.py`, `build.sh` | `weighted_residual_loss` is an atheus-invented extension (disabled by default), the analogue of PRESAGE's `presage_gated.yaml`. `utils.py` is 100 lines that nothing imports. `build.sh` is docker-only. |
| **V3** | read `seed` from the **top-level** config; seed python/numpy/torch; give the control-donor and inference-pool draws their own generator | The atheus source seeded **nothing** — no `torch.manual_seed`, no `np.random.seed` — and both `np.random.randint` sites used the global RNG. Two `predict` runs on one checkpoint therefore returned different numbers, and any seed-stability check measured RNG noise. `prepare_split` also hard-coded `seed=42` twice. |
| **V4** | resolve the **foundation** weights from `extra_config.scgpt_pretrained_path` and the **fine-tuned** weights from `config['model_path']`; delete `model_loc`/`model_loc_local` | **Highest severity.** The source read `hyperparameters['model_loc'] = "/pretrained_model/"` in *both* modes and relied on the harness re-binding that one path per mode. Our contract binds one dir at `/model_output` for both calls, so a verbatim port would resolve predict to the FOUNDATION weights and score an un-fine-tuned model — finite, sane-ranged, entirely meaningless. |
| **V5** | assert the loaded-parameter count after `load_pretrained(..., strict=False)`, in **both** modes | `strict=False` drops every key/shape mismatch silently, so a disagreement yields a randomly-initialised trunk that fine-tunes, converges and scores clean. **Checked:** the one obvious trigger — flash `Wqkv.*` vs vanilla `in_proj_*` — *is* handled upstream by a rename keyed on `model.use_fast_transformer`, so this is a guard, not a live known failure. It still catches a vocabulary-size change, an architecture disagreement (V16), and the FA1/FA2 `self_attn._impl.Wqkv` layout. |
| **V6** | **replace condition-list subsetting with the per-cell `obs[split_name]` label**, and bucket dataloaders on a per-graph split tag | THE leak fix — see §1.2. Every cell graph now carries its source cell's split label, and `SCGPTPertData.get_dataloader` replaces GEARS' wholesale. Unlike the GEARS wrapper we apply the filter in **every** regime including UnseenPert: GEARS had to special-case it because dropping test cells removes keys its `get_dataloader` needs, whereas scGPT's predict path builds its own graphs from control cells and never reads `dataset_processed`. |
| **V7** | **covariate-matched, batch-matched, split-restricted control pairing**, with the tier that fired counted | The mechanism of §1.1. Train graphs may only borrow `split=='train'` controls of the same covariate; predict conditions on the *target* covariate's controls. Fallback tiers are counted into the report, never logged and forgotten. The split restriction matters even on single-cell-line data: `assign_split_folds_unseen_pert` round-robins control **cells** into folds, so adamson16/UnseenPert/fold0 has 1 444 `test`-labelled control cells that an unrestricted pool would pair into training rows. |
| **V8** | read `covariate_key`, not `covariate_field`; delete the three hardcoded `cov_field = "cell_type"` literals | Our contract sends `covariate_key`; `covariate_field` is the key F4 deleted host-side. Nothing populated it, so the branch always fell through to `obs['cell_type'] = "NOTHING"` — cell-blind by accident. The reader is deleted rather than kept as a fallback (PRESAGE V13's reasoning: a stray `extra_config: covariate_field:` must not be able to override the canonical key on the one model whose claim rests on it). |
| **V9** | make the OOV drop **counted and floored**: record dropped genes/conditions, add `has_drop_rule = True` and a host preflight coverage floor | Three silent unbounded drops: genes outside the scGPT vocabulary, then perturbations whose targets did not survive, then test conditions absent from the panel. Measured on adamson16: **7 285/8 250 genes (11.7 % dropped)** and **78/89 targets**, with `AARS` dropped from the fold-0 test set — so one all-NaN `(bin, ko)` slice at M1 is *expected* and must be reconciled against the report, not treated as failure. |
| **V10** | **refuse** to save when no best-validation checkpoint was selected; add the authors' early stopping; record `best_epoch`/`best_val_loss` | The source fell back to `self.model` (the last epoch) when `best_model` was None. That fallback is exactly what PRESAGE's V5 refusal caught: a zero-optimiser-step run still writes a checkpoint whose predictions are correctly shaped, finite and log1p-ranged. |
| **V11** | own `drop_last` (set `False`) and record configured vs effective batching; add a predict-time loader that does **not** unpickle the graphs | `drop_last` came from `cell-gears==0.0.2` and was never verified — with `drop_last=True` a batch wider than the split gives zero optimiser steps and exits rc 0 with an untrained checkpoint (PRESAGE's V17). Separately, the atheus predict path called `pert_data.load()`, unpickling a multi-GB `cell_graphs.pkl` that `_generate_predictions` never reads. |
| **V12** | lift wrapper-hardcoded hyperparameters into `model.yaml` | `num_de_genes`, `n_cls`, the optimizer, `StepLR` gamma, the grad-clip norm, `do_sample`, and the control-donors-per-cell count were all values in the body that the recipe could not express. |
| **V13** | one `CONTROL_LABELS` constant | The list was hardcoded in **five** places (`:775`, `:910`, `:961`, `:1130`, `:1555`), twice with a `# TODO: We should be passing the control value as a parameter` beside it. Five copies drift independently. |
| **V14** | wire W&B, defaulting **safely to offline** | The source contained zero references to wandb or tensorboard, while `model.yaml` declares `wandb_project` and `ContainerPredictor._train` injects `wandb`/`wandb_project`/`wandb_run`. A declared setting that does nothing is worse than no setting: it made Gate A impossible while looking configured. Offline is the default because `ctr-biomed-15:8082` is down and an online logger with no endpoint blocks on login in a tty-less batch job. The online path is gated on an explicit `WANDB_MODE` — keying it on `WANDB_API_KEY` is what silently sent a PRESAGE run online, because `sbatch` inherits the submitting shell. |
| **V15** | **refuse** a stale per-fold cache; fingerprint it | `:824-825` warned and **proceeded** on an existing `perturb_processed.h5ad`, and `:847`/`:880` branched purely on `cell_graphs.pkl` existence — so a re-used fold dir silently trained on the previous fold's graphs and split. The host fingerprint hashes the *host's* split, so it would have certified that run as valid. |
| **V16** | reconcile the checkpoint's architecture with the recipe's, and save the **resolved** values | Train used `model_configs.get(k, hyperparams[k])`, so the checkpoint always won and the recipe's architecture values were INERT; `_save_model` then wrote the RECIPE's values into the run-dir `args.json`, making them LIVE at predict. They agree today only by luck. A recipe edit to `nlayers` would change nothing at train and silently build a different model at predict, where `strict=False` hides the mismatch. Disagreement is now an error. |
| **V17** | default `loss_reduction` to `mean`, matching upstream | Upstream `scgpt.loss.masked_mse_loss` ends `return loss / mask.sum()` — a **mean**. The atheus reimplementation defaulted to `reduction='sum'` and never overrode it, making its loss ~`batch x n_genes` (≈1e5) times larger than what `lr: 1e-4` — the authors' own value — was tuned against. Adam is nearly invariant to loss scale, so this is *not* a 5-orders-of-magnitude LR change; but `clip_grad_norm_(1.0)` and the fp16 `GradScaler` are **not** scale-invariant. With gradients ~1e5× larger, clipping fires on essentially every step (replacing Adam's per-parameter step with a fixed-norm one) and fp16 gradients risk overflowing (max ~65504), making the scaler skip steps. Either way the optimisation is not the authors'. |
| **V18** | *merged into V23* | The train/val gene-subset asymmetry and the `randperm`-returns-positions bug are one defect with one fix; kept as a single marker rather than two. |
| **V19** | dead cluster paths — **checked, absent** | `grep -E "capstor\|/iopsstor\|/users/\|/cluster/\|/store/\|/scratch/"` over the vendored source returns nothing. The atheus drop carried only container-internal paths. Recorded because "checked, absent" is a result. |
| **V20** | record the resolved flash backend and **refuse** a silent fallback | scGPT flips `use_fast_transformer` to False with a mere *warning* when flash-attn will not import, and FA2's MHA falls back to dense on float32 input. All of those produce correct-looking numbers on a slower path while `model.yaml` still claims `use_fast_transformer: true`. Note for the record: **the atheus Dockerfile installs no flash-attn at all** while its config says `true`, so every atheus scGPT number was produced on the slow path — one reason Gate C is corroborating only. |
| **V21** | densify defensively | `X.toarray()` was called unconditionally, so a dense h5ad raised `AttributeError` deep inside the graph builder rather than at load. Every dataset here is CSR today; this makes that an assumption the code states rather than one it relies on. |
| **V22** | give predict its own `predict_batch_size`; assert the output h5ad's shape | `pred_perturb` with `include_zero_gene="all"` runs the **full untruncated** gene sequence (~7.3 k for adamson16) rather than `max_seq_len`, so predict's memory profile is unrelated to training's. Lowering `batch_size` to survive predict would silently retune training. The output asserts (unique `var_names`, finite, log1p-ranged) move a `tensor_map`-time failure to before the write. |
| **V23** | make train and val use the same gene-subset policy, indexed correctly, under explicit generators | Train drew a *random* `max_seq_len` subset per batch via `torch.randperm(len(input_gene_ids))[:max_seq_len]` — which returns **positions, not gene ids**, coinciding with gene ids only because `include_zero_gene == "all"` makes the input `arange`; under `"batch-wise"` it indexed the wrong space. Validation meanwhile took a plain prefix, so `best_val_loss` selected on an arbitrary gene prefix. |
| **V25** | flag the perturbed gene with a flat `1` at predict, not `np.sign(p)` | Inherited from GEARS, where the sign encodes direction. scGPT's flag vocabulary is {0 = unperturbed, 1 = perturbed, 2 = pad}, and `np.sign(0) == 0` — so a perturbation whose target is the **first gene of the panel** was flagged "unperturbed" at predict and the model was handed a control cell with no perturbation. It would then faithfully predict the control profile, for that one perturbation, silently. Training already used a flat `1`; predict now agrees. |
| **V26** | resolve genes to the vocabulary by **Ensembl accession**, not by HGNC symbol (symbol identity first, accession for the remainder, symbol wins collisions) | The atheus source matched by symbol. HGNC renames genes, so genes scGPT knows were discarded under their old names — `AARS`/`AARS1`, `ATP5B`/`ATP5F1B`, `SRPR`/`SRPRA`, `SLMO2`/`PRELID3B`. On adamson16 that silently cost **965 genes and 11 of 89 perturbation targets**, i.e. the "11.7 % OOV rate" was a property of OUR JOIN, not of scGPT. See §8. |
| **V27** | refuse a NEGATIVE perturbation index instead of folding it through `abs()` | `SCGPTPertData.get_pert_idx` returns `[-1]` as its "gene not in `gene_names`" sentinel, and `int(np.abs(-1)) == 1` — so the sentinel silently marked **gene index 1** as the perturbed gene and the model trained on a confidently mislabelled graph. Found while extracting `_pert_flag_vector` (the V25 site). Currently unreachable — the V9/V26 drop rule removes such conditions before graphs are built — so reaching it means the drop rule and the index lookup disagree, which is worth an exception rather than a quietly wrong flag. |
| **V24** | host-owned files — **checked, absent** | The container writes only `best_model.pt`, `vocab.json`, `args.json`, `metadata.json`, `scgpt_training_report.json` and its `processed_data/` tree. No `fingerprint.json`, `RUNNING.json`, `train_config.json` or `predict_config.json`. Clobbering `fingerprint.json` would make a stale checkpoint look valid — the failure the contract works hardest to prevent. |

### Deviation recorded: batch matching at predict

The design called for predicting per `(covariate, batch)` and pooling the results.
`_sample_control_pool` instead draws the control pool **stratified by batch** within the
covariate. That is the same estimator — the mean over a stratified sample equals the
stratum-size-weighted mean of per-stratum predictions — at `1/n_batches` the forward-pass
cost, which on mcfaline23 (48 batches per cell type) is a 48× difference. Recorded here
rather than left implicit.

---

## 3. Hyperparameter provenance

Pinned to the **authors' own perturbation fine-tuning tutorial**
(`scgpt.readthedocs.io/en/latest/tutorial_perturbation.html`) — the published recipe for
exactly this task. Where atheus deviated from it, the authors win. Full table with inline
comments in `model.yaml`; the deviations are:

| value | authors | atheus | ours | note |
|---|---|---|---|---|
| `max_epochs` | 15 | 10 | **15** | |
| `batch_size` | 64 | 32 | **64** | |
| `max_seq_len` | 1536 | 1200 | **1536** | train only; predict is untruncated (V22) |
| `schedule_interval` | 1 | 15 | **1** | atheus's 15 > `max_epochs` 10, so its StepLR **never fired** |
| `early_stopping_patience` | 5 | (none) | **5** | this is the AUTHORS' setting, not ours |
| `MVC` | False | true | **false** | atheus computed the MVC head every step and never added it to the loss |
| `loss_reduction` | mean | sum | **mean** | see V17 |
| `lr` | 1e-4 | 1e-4 | **1e-4** | agree |

`predict_batch_size` is **ours**, and flagged as ours: upstream has no such knob because it
does not separate the two memory regimes (V22).

---

## 4. Things that will bite

* **The `cell_graphs.pkl` scale.** One PyG `Data` per (cell × condition), each with a dense
  `[n_genes, 2]` float32 `x` plus a dense `[1, n_genes]` `y`, in one pickle held fully in
  RAM. Estimated ~5.6 GB for adamson16 and far more for the multi-cell-line datasets. V11
  keeps predict from loading it and the recipe leaves it out of `expected_artifacts`, but
  training still materialises it. **Measure at the smoke before touching a large dataset**;
  a lazy `Dataset` that builds graphs on `__getitem__` is the real fix and is deferred.
* **GPU model matters.** flash-attn 2.x requires compute capability ≥ 8.0. `--gres=gpu:1`
  can land on a V100 (sm70) and die at the first forward pass. Always name the model.
* **Predict runs the full gene sequence.** ~7.3 k tokens, not `max_seq_len`. With flash
  attention this is memory-linear; without it, attention scores alone are tens of GB.
* **`schedule_interval: 1` now actually decays the LR.** It did not under atheus. Any
  comparison against an atheus number is affected by this as well as by V17.

---

## 5. Shortcut temptations

* *"`use_fast_transformer: false` is simpler."* It is — and it is a different compute path
  than the recipe records. If it is ever taken, it must be recorded here as a deviation,
  not silently switched (V20).
* *"The NaN slice at M1 is a bug."* On adamson16/UnseenPert/fold0 exactly one test KO
  (`AARS`) is outside the scGPT vocabulary. The gate is that the all-NaN KO set **equals**
  the container's reported drop set — an unexplained NaN is still a failure.
* *"The atheus drop is the reference."* It was produced by the defects in §2, on the slow
  attention path, with an unnormalised loss. Gate C is corroborating only.

---

## 6. Build provenance

| item | value |
|---|---|
| base image | `pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime` @ `sha256:831247999fbf7e08f61b3e39f6d77ee434f38f6f07f769d00db451e853878067` |
| base contents (probed) | python 3.11.10, torch 2.5.1+cu121 |
| scGPT | `github.com/bowang-lab/scGPT` @ `cebd6fae655b9c585a4807daa3ac31bb764f06b4` (declares 0.2.5) |
| why a SHA, not `0.2.4` | FA2 support (`flash_attn_compat.py`, PR #351), `torchtext` removed, prebuilt FA2 wheels. Recorded as a deviation from the published release. |
| flash-attn | 2.8.3, prebuilt wheel; ABI variant read from `torch._C._GLIBCXX_USE_CXX11_ABI` at build |
| cell-gears / torch-geometric | 0.0.2 / 2.6.1 |
| scientific stack | **pinned** to the set `gears.sif` resolved on 2026-08-13 (its `/opt/gears_lockfile.txt`): pandas 2.3.3, scipy 1.17.1, h5py 3.16.0, anndata 0.12.19, scanpy 1.11.5, statsmodels 0.14.6, numba 0.67.0, scikit-learn 1.9.0. See "the two-week drift" below. |
| anndata | `0.12.19`, matching `gears.sif` — **not** the host `vcell` env's 0.11.4. See "the anndata reversal" below. |
| scgpt install | `--no-deps` + an explicit `datasets` pin. scGPT's metadata requires `scvi-tools>=0.16,<1.0` (2023-era, drags in old jax/flax/orbax) but **nothing in the package imports it** — verified by installing `--no-deps` at this commit and importing. The only hard third-party import `import scgpt` adds is `datasets` (`scbank/databank.py:10`); `einops`/`torchtext` are guarded, `scib` is deferred inside an eval helper. |
| C compiler | **not installed, deliberately.** With the stack pinned every wheel is prebuilt, so nothing compiles — and the absence means a future pin drifting to an sdist-only release fails loudly instead of silently source-building. `gears.def` likewise installs no compiler. |
| pretrained weights | whole-human, 33 M cells; `best_model.pt` md5 `9922ec94305126e6e4f9c1575cf493ae`, baked at `/opt/scgpt_pretrained` |
| licences | scGPT MIT, cell-gears MIT, flash-attn BSD-3-Clause — all permissive; no PRESAGE-style NOTICE obligation |
| `%test` | deliberately absent. It runs as root and is non-fatal, which made PRESAGE's actively misleading; every check is a fatal `%post` assertion instead. |


### The two-week drift, and why the stack is pinned

The first build attempt left the scientific stack to the resolver and died at
`statsmodels`. Two packages had moved in the fortnight since `gears.sif` was built,
and both breakages are real rather than cosmetic:

| | gears.sif, 2026-08-13 | resolver, 2026-08-27 | consequence |
|---|---|---|---|
| `statsmodels` | 0.14.6 | 0.15.0 | 0.15.0 ships no cp311 wheel → pip takes the sdist → needs a C compiler the runtime base lacks → `FATAL` |
| `pandas` | 2.3.3 | 3.0.5 | a MAJOR bump. pip itself reports *"anndata 0.12.19 requires pandas<3"* — as a **warning**, so an unpinned build would have produced an image whose anndata was broken, and said so only in passing |

The second is the one that matters: the compiler error was loud and cost a build
cycle; the pandas bump was quiet and would have cost a debugging session. Hence a
single `pip install` carrying every pin (so the resolver sees all constraints at
once) plus explicit `assert`s on `pandas`/`anndata`/`statsmodels` in the `%post`
import gate, so drift is reported at build time rather than at training time.

### The anndata reversal

An earlier draft pinned `anndata==0.11.4` to match the host `vcell` env, following
`presage.def:69-75` — the container writes h5ads the host reads back, so a newer
writer could in principle emit an encoding the host cannot read. That principle
stands; the pin does not, for two reasons:

* `scanpy 1.11.5` resolves `anndata 0.12.19`, so 0.11.4 has to be forced *down*
  after the fact — the "install a pin afterwards to override an earlier
  resolution" pattern that leaves everything resolved against the old version
  quietly mismatched.
* The concern is **empirically refuted for the files we actually produce**:
  `gears.sif` runs anndata 0.12.19, and the host (0.11.4) reads its
  `predictions.h5ad` — the GEARS M0 smoke maps it through host-side `tensor_map`.

So: writer-newer-than-reader is fine here, and consistency with the sibling
container is worth more than an exact version match. If a future anndata does
break the round trip, the smoke driver's `anndata` version check is what catches it.

### Validated before building, not after

The import graph and the weight load were checked on a login node with no GPU, by
installing scGPT `--no-deps` into a throwaway `--target` inside `gears.sif` (same
base image, same python 3.11.10, same torch 2.5.1+cu121):

```
GeneVocab loaded WITHOUT torchtext: 60697 tokens
TransformerGenerator built: 51.86 M params
load_pretrained: 153/176 tensors changed, 144 in transformer_encoder.layers
```

144 = 12 layers x 12 tensors, so **V5's assertion passes with a wide margin** — and
this is the direct measurement behind V5's "guard, not a live failure" claim: the
checkpoint was pretrained with `fast_transformer: true` (so its attention weights
are named `Wqkv.*`), it was loaded into a **vanilla-attention** model, and the
trunk still landed. The `Wqkv.` -> `in_proj_` rename keyed on
`model.use_fast_transformer` really does work.


---

## 7. Smoke / M1 results

### G1 — Tier-2 smoke: **PASSED** (2026-08-28, job 9543487, post-V26)

adamson16 / UnseenPert / fold0, `max_epochs=1`, RTX 4090 (sm89), 4 CPU / 64 GB.
Driver: `/cluster/home/akalygina/scgpt_smoke_driver.py`. `TRAIN rc 0`, `PREDICT rc 0`,
7/7 expected artefacts, `SMOKE OK`.

Run 9542491 passed the same gates **before** V26 and is superseded; where the two differ
the pre-V26 value is given in brackets. The V26 delta: predictions `(18, 8086)` not
`(17, 7285)`, **zero** dropped perturbations and **zero** dropped test conditions, and a
delta-tensor finite fraction of **0.980** (= 8 086/8 250) rather than 0.834.

| measured | value | what it settles |
|---|---|---|
| `flash_attn_backend` / effective | `fa2` / `true` | V20 — the fast path the recipe claims is live, not a silent dense fallback |
| flash modules | 24 = 12 `FlashTransformerEncoderLayer` + 12 `MHA` | all 12 layers on the FA2 path |
| pretrained keys loaded | 153/178 total, **144 transformer** | V5 — the trunk loaded; the 25 unloaded are the perturbation encoder + decoder head, which train from scratch by design |
| `n_test_cells_in_training` | **0** (61 992 cells → 51 171 train+val) | V6 — the documented leak is closed |
| optimiser steps | **707**/epoch [636], `batch_size_effective == configured == 64` | V11 — not a zero-step run |
| `best_epoch` / `best_val_loss` | 1 / 0.0548 [0.0540] | V10 — best-val selection ran |
| ctrl pairing tiers | `{split_only: 2}`, `covariate_col: null` | correct: adamson16 has ONE cell type, so covariate matching auto-disables. Counts distinct `(cov, batch, splits)` **pools**, not cells — the two are the train and val pools |
| epoch wall time | **529.9 s** [497.9] (+ ~4 min data prep) | 15 epochs ≈ 2.2 h, inside `train_timeout: 21600`. Size M1 from this, not from row counts. V26 added 4 550 train graphs (45 200 [40 650]) by making 11 more perturbations trainable |
| `cell_graphs.pkl` | **4.66 GB** [3.75] (output dir 6.1 GB total) | R2 — vs the ~5.6 GB planning estimate. Predict did not need it. **V26 grew it ~24 %** (more genes per dense graph); re-check before replogle22/jiang24 |
| peak RSS | **13.8 GB** [11.7] train, unchanged by predict | 64 GB is ample for adamson16; extrapolate before replogle22/jiang24, allowing for V26's larger panel |
| predictions | `(18, 8086)` [(17, 7285)], finite frac 1.0, range `[-0.020, 4.38]` | log1p-ranged and finite |
| delta tensor | `(1, 18, 8250)`, finite frac **0.980** [0.834] | exactly `8086/8250` — every test condition is now predicted, so the only NaNs left are genes outside the vocabulary |
| gene resolution | 8 250 → **8 086** kept [7 285]: 7 285 by symbol + **801 by accession**, 164 unresolved [965], 1 collision | V26 — the accession route is live and recovered what the symbol join lost |
| `gene_id_join` / `ensembl_column` | `ensembl_first` / `ensembl_id` | V26 — the column was auto-detected by value shape, not assumed from its name |
| drop rule | `dropped_perturbations` **[]**, `dropped_test_conditions` **[]**, no all-NaN test KO | V9 + V26 — adamson16 now loses no perturbation at all; pre-V26 it lost 11, incl. the held-out `AARS` |

**Two defects the smoke caught, both fixed:**

1. `use_fast_transformer` was absent from `model.yaml` while the wrapper hard-requires it →
   `KeyError` after 4 minutes of data prep. Added with provenance. A full diff of every
   `hyperparams[...]` read against the recipe's keys now shows no other gap.
2. `dropped_test_conditions` was computed at predict, logged, and **discarded** — predict
   wrote no report. A test-only KO outside the vocab therefore became an all-NaN slice with
   nothing machine-readable to explain it, and the train-side `dropped_perturbations`
   structurally cannot name it (the V6 filter removes test cells before the drop scan).
   Fixed by `_write_predict_report` → `scgpt_predict_report.json`.

   The live case: **`AARS`** is held out as test on fold0 and absent from the checkpoint
   vocabulary, so it is legitimately unpredictable — 11 dropped targets, matching the
   78/89 survival measured during planning.

### The drop rule was a symbol-join artifact — RESOLVED by V26

The G1 smoke's 11 dropped targets were not scGPT's limitation but our symbol join's. Fixed
by V26 (see §8): adamson16 goes from 7 285 to 8 086 genes and from 11 dropped targets to
**none**. Datasets without an accession column (replogle22, jiang24, wessels23,
xatlas_orion) are unchanged and still drop by symbol.

---

## 8. Gene identity: Ensembl-first resolution (V26)

scGPT tokenises against a fixed 60 664-name vocabulary, so every gene must be matched to a
vocabulary entry. Matching by **HGNC symbol** — what the atheus source did — is wrong,
because symbols drift and accessions do not. `gene_info_scgpt.csv` carries `feature_id`
(Ensembl) beside `feature_name` and is a clean bijection (60 664 unique on both axes).

**Resolution order, per gene:** symbol identity -> Ensembl accession (if it yields a name no
symbol match has claimed) -> `NaN`, dropped. **Symbol identity wins collisions**, which makes
the change strictly ADDITIVE — the accession route may only add genes, never displace one the
old join kept. `_ensure_symbol_scgpt` asserts both post-conditions: the surviving mapping is
injective, and it is a superset of the symbol-only result.

Measured on adamson16 (`ensembl_first` vs `symbol_only`, real data, both modes run):

| mode | genes kept of 8 250 | by symbol | by accession | collisions | targets lost of 89 |
|---|---|---|---|---|---|
| `symbol_only` (pre-V26) | 7 285 | 7 285 | 0 | 0 | **11** |
| `ensembl_first` (default) | **8 086** | 7 285 | 801 | 1 | **0** |

The one collision is real: `RP11-343N15.1` (ENSG00000230806) resolves by accession to
`SRGAP2-AS1`, which our own gene named `SRGAP2-AS1` (ENSG00000233501) already holds by
symbol. It is dropped with a warning rather than emitting a duplicate token — unhandled,
`gene_ids` would carry the same vocabulary id twice and `np.where(p == gene_names)[0][0]`
would silently resolve to the first.

### Three things that make the detection robust

The accession column is found by the **shape of its values** (`^ENSG\d{11}`), never by name,
because across our own datasets it is spelt three different ways and is not always complete:

| spelling | datasets |
|---|---|
| `ensembl_id` | adamson16, frangieh21, mcfaline23, sunshine23 |
| `ensemble_id` (misspelled) | norman19 |
| `gene_id` | replogle20 |
| **none at all** | **replogle22, jiang24, wessels23, xatlas_orion, ecoli_synthetic** |

1. Detection is by value, so the misspelling and `gene_id` are picked up automatically.
2. Resolution is **per gene**, so a partly-populated column still contributes what it has —
   frangieh21's is only 77.5 % accessions.
3. `ENSEMBL_COLUMN_MIN_FRACTION` is deliberately low (0.1), not a majority: the pattern is
   specific enough that coincidence is not a real failure mode, every value is validated
   against the bijection anyway, and a high floor would discard a real partial column.

**This does NOT fix every dataset.** replogle22, jiang24, wessels23 and xatlas_orion carry no
accessions and are resolved by symbol exactly as before. The per-run report records
`ensembl_column: null` for them, so a symbol-only run is visible rather than looking identical
to an accession-resolved one.

### Where it is recorded per run

`scgpt_training_report.json` (and so `fingerprint.json`) carries `gene_id_join`,
`ensembl_column`, `n_genes_matched_by_symbol`, `n_genes_matched_by_ensembl`,
`n_genes_unresolved` and `n_gene_id_collisions`.

### Namespaces — why nothing needs translating back

The vocabulary name exists in exactly one place and is read by exactly one consumer:

| column | namespace | read by |
|---|---|---|
| `var_names` | **ours** | predictions axis, DE genes, the predict-time re-read check (`CONTRACT.md:141`) |
| `var['gene_name']` | **ours** | `pert_data.gene_names` -> `get_pert_idx`, condition matching, prediction graphs |
| `var['symbol_scgpt']` | **scGPT** | the tokenizer, and nothing else |

So predictions, `obs['condition']` and the gene axis never leave our symbols, and comparability
with every other predictor holds by construction — there is no inverse map to maintain and no
egress translation that could be forgotten. The mapping is written into
`perturb_processed.h5ad` at train and re-read at predict, so it is frozen into the trained
artefact and predict cannot resolve differently than fine-tuning did.

`extra_config.gene_id_join` selects the mode; `symbol_only` reproduces the pre-V26 join
exactly. It lives in `model.yaml`, so it is hashed into `recipe_sha` and a change invalidates
checkpoints rather than being served silently.

---

## 9. Shared implementations (why some code is NOT here)

An infrastructure review across `docker/{gears,presage,scgpt}` removed the copies that let
the three containers drift apart. What moved, and why it matters here:

* **`_pert_flag_vector`** (module level, this file) is now the ONLY place the per-gene
  perturbation flag row is built. Train and predict each had their own copy, and they
  disagreed — that *is* V25. One definition is why it cannot recur. The callers hold
  different shapes (`(k, n_genes)` at train, `(n_genes,)` at predict), so `n_genes` is passed
  explicitly rather than derived inside.
* **`_write_report`** replaced a train writer and a predict writer that shared six of eight
  body lines; the mode-specific fields (`architecture`, `model_path`) are passed by the caller.
* **`create_cell_graph_dataset_for_prediction` was DELETED.** It was unreachable — upstream
  defines that name as a module-level function in `gears.utils` and `GEARS.predict` calls the
  module-level one, so nothing dispatched to our method — and it implemented the **pre-V7
  unmatched control draw**, the exact behaviour V7 exists to prevent. Dead code that would
  have silently reintroduced cell-blindness if anyone had called it.
* Host-side, `ContainerPredictor` now owns `_check_target_coverage` (the drop-rule floor,
  shared with GEARS) and `_read_json_report` (shared with PRESAGE), and the control-label
  predicate comes from `_container/leakage.py` rather than four inline copies — the host twin
  of the V13 fix this file already applies with `CONTROL_LABELS`.
