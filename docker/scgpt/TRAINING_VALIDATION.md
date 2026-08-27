# scGPT-ct — training-validation card

Evidence that **our container fine-tunes scGPT correctly**. Every downstream benchmark
number is licensed by this card, so an unfilled gate is a reason not to report, not a
formality.

* **Gate A — per-run sanity.** Does *this* run look like fine-tuning happened?
* **Gate B — paper replication.** Does our container reproduce scGPT on scGPT's own dataset,
  split and metric? This is the gate that licenses everything else.
* **Gate C — external cross-check.** Does it land near the externally-trained atheus drop?
  **Corroborating only.**

Status: `☐ not run · ◐ run, not yet reviewed · ☑ passed · ✗ failed`

---

## Gate A — per-run training sanity

Repeat per scored run. W&B project `vcr-scgpt-ct`, run name
`scGPT-ct_{dataset}_{scenario}_fold{n}_seed{seed}`.

| check | what it rules out | status |
|---|---|---|
| `val_loss` decreases, and early stopping fires before `max_epochs: 15` | not training / not converging | ☐ |
| selected epoch ≪ final epoch, and `best_epoch`/`best_val_loss` appear in `fingerprint.json["report"]` | V10 regressing to last-epoch weights | ☐ |
| overfit a tiny batch (few perturbations, early stopping off) → near-zero train loss | wiring failure | ☐ |
| train-set fit ≫ val-set fit | predicting a constant. A real risk here: scGPT predicts an *absolute* profile from a control profile, so "return the input" is a strong degenerate solution | ☐ |
| predictions are not constant across perturbations | the same, at inference | ☐ |
| `pretrained_transformer_keys_loaded` ≥ `nlayers` in the report | V5 — a randomly-initialised trunk that trains and scores clean | ☐ |
| `use_fast_transformer_effective` is `true` and `flash_attn_backend == "fa2"` | V20 — a slower compute path than the recipe records | ☐ |
| seed 42 vs 43 within a stability band comparable to GEARS' | reading run-to-run noise as signal; only meaningful now that V3 makes the seed actually apply | ☐ |

---

## Gate B — paper replication *(the licensing gate)*

| field | value |
|---|---|
| paper / repo | scGPT (Cui et al., Nat. Methods 2024); `bowang-lab/scGPT` @ `cebd6fae655b9c585a4807daa3ac31bb764f06b4` |
| dataset + split | *fill from the paper — must be the paper's (Adamson / Norman as published), not ours* |
| metric reported | *fill* |
| paper's number | *fill* |
| our container's number | *fill* |
| verdict | ☐ |

Two things are **ours, not theirs**, and must be stated in any replication claim:

* `predict_batch_size` — upstream does not separate the train and predict memory regimes (V22).
* The covariate-matched control pairing (V7). On a single-cell-line dataset it reduces to
  upstream's behaviour apart from the split restriction, so Gate B is a fair test of the
  rest; on multi-cell-line data it is a deliberate departure.

Everything else in `model.yaml` is the authors' published perturbation-tutorial value —
including the four that atheus deviated on (`max_epochs`, `batch_size`, `max_seq_len`,
`schedule_interval`), the early stopping, and the `mean` loss reduction.

If our number lands well outside the paper's, that is a **training-correctness finding**,
not a benchmark result — fix the container before running anything else.

---

## Gate C — cross-check vs the external atheus drop

**Corroborating only.** A difference is *expected*, and the reasons are known rather than
speculative: the atheus drop was produced with no per-cell split filter (V6), pooled control
donors (V7), no seeding (V3), an unnormalised loss (V17), an LR scheduler that never fired
and a wasted MVC head (V16), and — because its image installed no flash-attn while its
config requested it — on the vanilla attention path (V20).

| dataset / regime / fold | atheus drop | scGPT-ct | note |
|---|---|---|---|
| adamson16 / UnseenPert / 0 | | | ☐ |

---

## Cell-axis reporting gate (M1.5)

**No cell-axis number is reported until all three below pass.** `cell_aware = True` is a
claim about the control-matching in V7, not about an embedding — scGPT has no covariate
port — so the evidence has to be that matching the covariate *changes the prediction*.

On mcfaline23 (a172 / t98g / u87mg), `UnseenPert` fold 0 — chosen deliberately: it is
already in `scenarios`, so the control runs **before** any capability flag is flipped.

| check | status |
|---|---|
| host leakage gate passes in **pair** mode | ☐ |
| in-container structural assert: `n_test_cells_in_training == 0`, and every train graph's control donor is `split=='train'` and same-covariate (`ctrl_pairing_tiers` in the report) | ☐ |
| **positive control: the same KO under two cell types predicts *differently*** — **the one that matters** | ☐ |

Quantify the third; do not eyeball it. The null is not zero: the control pool is sampled, so
two predictions of the *same* (cell type, KO) at different seeds already differ. Measure that
sampling noise explicitly and require the cross-cell-type difference to exceed it, and to
correlate positively across test KOs with the same quantity in `store.first_half_deltas`.

Recorded expectation, so it is not mistaken for a fault: mcfaline23 has **zero** `val` and
`test` control cells (all 24 576 are `train`), so V7's val fallback tier will fire. It must
appear in `ctrl_pairing_tiers`, not in a log line.

adamson16/UnseenPert green is **not** evidence here — one cell line, degenerate covariate,
exercises none of this.

---

### Known limits, to report rather than fix

* **Predict sees a longer sequence than train.** Training truncates to `max_seq_len: 1536`;
  `pred_perturb` runs the full panel untruncated. That is upstream's own behaviour, not a
  defect in this container, but it is a real train/predict distribution shift.
* **The vocabulary drop rule is lossy.** 11.7 % of adamson16's panel and 11/89 of its
  perturbation targets are outside the scGPT vocabulary. `has_drop_rule = True` and the
  preflight floor make it visible and bounded; they do not make it go away.
