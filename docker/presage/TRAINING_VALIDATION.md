# PRESAGE-ct — training-validation card

Evidence that **our container trains PRESAGE correctly**. Every downstream
benchmark number is licensed by this card, so an unfilled gate is a reason not to
report, not a formality. Three gates, per the agreed strategy:

- **A — per-run sanity.** Does *this* run look like training happened?
- **B — paper replication.** Does our container reproduce PRESAGE on PRESAGE's own
  dataset, split and metric? This is the gate that licenses everything else.
- **C — external cross-check.** Does it land near the externally-trained atheus
  drop? **Corroborating only.** The atheus run scored last-epoch weights (V5), so
  agreement is reassuring and disagreement is not, by itself, evidence of a bug.

Status legend: ☐ not run · ◐ run, not yet reviewed · ☑ passed · ✗ failed

---

## Gate A — per-run training sanity

Repeat per scored run. W&B project `vcr-presage-ct`, run name
`PRESAGE-ct_{dataset}_{scenario}_fold{n}_seed{seed}`.

| check | what it rules out | status |
|---|---|---|
| ☐ `val_loss` decreases and early-stops before `max_epochs: 1000` | not training / not converging; a run that hits the ceiling means the ceiling is doing the deciding | |
| ☐ selected epoch ≪ final epoch, and `best_epoch` / `best_val_loss` appear in `fingerprint.json["report"]` | V5 regressed and last-epoch weights are being scored again | |
| ☐ overfit a tiny batch (a handful of perturbations, early stopping off) → near-zero train loss | the model cannot fit even memorisable data: wiring, not data | |
| ☐ train-set fit ≫ val-set fit | the model is predicting a constant (a real risk: the target is a delta, and predicting 0 is a decent baseline) | |
| ☐ predictions are not constant across perturbations | ditto, at inference | |
| ☐ seed 42 vs 43 within a stability band comparable to GEARS' | run-to-run noise is being read as signal. Only meaningful now that V3 makes the seed actually apply | |

## Gate B — paper replication *(the licensing gate)*

☐ **Not yet run.** The §3 hyperparameter audit it was blocked on is now **done**
(2026-08-26): 25/27 comparable values matched the authors' argparse defaults, and
the recipe is pinned to the authors' own single-dataset sweep, with `lr` taken as
the geometric mean of the range they swept. One value in the recipe is ours rather
than theirs (`lr = 8.66e-4`) and one regime is ours entirely (early stopping +
best-val selection — upstream has none). Both must be stated in any replication
claim.

| field | value |
|---|---|
| paper / repo | Genentech PRESAGE @ `2c7b231c60cf110c4aab0acf8ecd2c2b3268fffb` |
| dataset + split | *(fill from the paper — must be the paper's, not ours)* |
| metric reported | |
| paper's number | |
| our container's number | |
| verdict | |

If our number lands well outside the paper's, that is a **training-correctness
finding**, not a benchmark result — fix the container before running anything else.

## Gate C — cross-check vs the external atheus drop

Corroborating only. The as-adopted `PRESAGE` predictions in
`models/atheus_csb/presage_*` come from the pre-V5 wrapper (last-epoch weights)
and the pre-V9 predict path, so a difference is *expected* and does not, on its
own, indicate a bug in either.

| dataset / regime / fold | atheus drop | PRESAGE-ct | note |
|---|---|---|---|
| ☐ adamson16 / UnseenPert / 0 | | | |

---

## Cell-axis reporting gate (M1.5)

`PRESAGEContainer` declares `cell_aware = True` and the four cell-axis regimes up
front. **No cell-axis number is reported until all three below pass** — the
declaration is a claim about capability, not evidence of it.

On **replogle22** (K562 + RPE1), `UnseenBoth`, fold 0:

| check | what it rules out | status |
|---|---|---|
| ☐ host leakage gate passes in **pair** mode | a held-out `(cell_type, perturbation)` pair reaching train/val through the other cell type | |
| ☐ in-container V11 assert passes (no off-split cells in train/val) | the per-cell filter having been loosened | |
| ☐ **positive control**: the same KO under K562 and RPE1 predicts *differently* | **the one that matters** — if `cov_embed` is inert the two are identical, the model is cell-blind, and every other check still passes | |

adamson16/UnseenPert green is **not** evidence here: it is single-cell-line, so it
runs the degenerate-covariate path and exercises none of this.

### Known limit, to report rather than fix

Under `UnseenCell` a covariate category that never appears in training has an
**untrained** embedding — the model has no basis for conditioning on a cell type
it has not seen. That is a property of the method as published, not a defect in
this container, and it belongs in the results discussion.
