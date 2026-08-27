# Modifications

`docker/scgpt/scgpt_wrapper.py` is a derivative work. Its ancestry:

* **scGPT** — https://github.com/bowang-lab/scGPT, MIT (see `LICENSE`). The model classes
  it subclasses (`TransformerGenerator`) and the tokenizer/loader utilities it calls are
  imported from the installed package, unmodified; the package is pinned in `scgpt.def` at
  commit `cebd6fae655b9c585a4807daa3ac31bb764f06b4`.
* **GEARS / cell-gears** — https://github.com/snap-stanford/GEARS, MIT. `SCGPTPertData`
  subclasses `gears.PertData`.
* The immediate source vendored here is the atheus PMOB drop at
  `Perturbation-Models-Outperform-Baselines/docker/scgpt/scgpt_wrapper.py` (1757 lines).

MIT requires the copyright notice to travel with the code; it does not require this file.
It exists because the repo's convention is that a vendored wrapper states what was changed —
otherwise a reader cannot tell our behaviour from upstream's.

## What was changed

Every change carries a numbered `VENDORED (Vn)` marker in the source. The table with the
reasoning is `SCGPT_NOTES.md` §2. In summary:

* **Structural** — the training split is taken from the per-cell `obs[split_name]` label
  instead of a condition list (V6), and control donors are drawn from a covariate-,
  batch- and split-matched pool instead of the global control set (V7). These change what
  the model trains on.
* **Contract** — weights resolve per mode instead of via a re-bound `/pretrained_model/`
  mount (V4); the covariate comes from `covariate_key` (V8); the seed is read and applied
  (V3); W&B is wired (V14).
* **Recipe fidelity** — hyperparameters hardcoded in the wrapper body were lifted into
  `model.yaml` and reset to the authors' published perturbation-tutorial values where atheus
  had deviated: `max_epochs`, `batch_size`, `max_seq_len`, `schedule_interval`, `MVC`, and
  the loss reduction (V12, V16, V17).
* **Refusals** — the code now fails rather than continuing where a silent pass would produce
  a plausible wrong number: no best-validation checkpoint (V10), an incompletely loaded
  pretrained trunk (V5), a stale per-fold cache (V15), an inactive fast path that the recipe
  claims is active (V20), an architecture/checkpoint disagreement (V16).
* **Not vendored** — `weighted_residual_loss` and its artefact builders (an atheus-invented
  extension, disabled by default), `utils.py` (nothing imports it), and `build.sh`.

No upstream scGPT or GEARS source file is modified; both are installed from their pinned
upstream releases.
