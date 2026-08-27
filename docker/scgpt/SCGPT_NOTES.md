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

_Not yet run. Fill from the Tier-2 smoke and the M1 gate; see `TRAINING_VALIDATION.md`._
