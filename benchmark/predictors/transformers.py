"""Adopted transformer predictors for the ecoli_synthetic dataset.

The depth_hypothesis repo trained nine recurrence-depth transformer variants
(`transformer_12x1 … transformer_2x1`) on the synthetic E. coli data across the
six scenarios (S1-S6) x three seeds (42/1337/7). Those runs saved a
`test_predictions.npz` per (scenario, seed) whose `predictions` array is a
`(n_test_bins, n_test_kos, n_genes)` delta tensor (raw log1p-delta space, full
2853-gene axis) carrying its own `test_ko_indices` (into the source 9159-KO
vocabulary) and `test_bin_indices` (0-9).

Because `data/ecoli_synthetic/get_data.py` reconstructs the exact same exp3 folds
(seeds 42/1337/7 -> folds 0/1/2), those predictions correspond, cell-for-cell, to
our (scenario, fold) test sets — so we ADOPT them without retraining. Each class
below is a thin `Predictor` whose `predict()` loads the matching npz and name-maps
its axes onto the store (KO index -> name via the source vocab sidecar; bin index
i -> "bin_{i:02d}"; genes by identity/name). `fit` is a no-op; `is_external` is
deliberately left unset so the DL-adapter alignment gate (scGPT/GEARS/PRESAGE) is
skipped — these are name-aligned by construction.

Override the run location with env `VCELL_ECOLI_EXP4_RUNS`.
"""
from __future__ import annotations

import glob
import logging
import os
from functools import lru_cache
from pathlib import Path

import numpy as np

from benchmark.config import DATA_DIR
from benchmark.predictors.base import Predictor, register

log = logging.getLogger(__name__)

EXP4_RUNS = Path(os.environ.get(
    "VCELL_ECOLI_EXP4_RUNS",
    "/cluster/work/boeva/akalygina/virtual-cell-reasoning/"
    "depth_hypothesis/runs/exp4"))
VOCAB_PATH = DATA_DIR / "ecoli_synthetic" / "_exp3_axis_vocab.npz"

_SCENARIO_CODE = {
    "UnseenPert": "S1", "UnseenCell": "S2", "UnseenBoth": "S3",
    "UnseenPair": "S4", "UnseenDose": "S5", "UnseenCombo": "S6",
}
_SEEDS = [42, 1337, 7]  # fold index -> seed

MODELS = [
    "transformer_12x1", "transformer_6x2", "transformer_6x1",
    "transformer_4x3", "transformer_4x1", "transformer_3x4",
    "transformer_3x1", "transformer_2x6", "transformer_2x1",
]


@lru_cache(maxsize=1)
def _source_vocab():
    """(ko_gene_names [9159], gene_names [2853]) — the exp3 axis labels, written
    by get_data.py next to the h5ad. Cached; tiny."""
    if not VOCAB_PATH.exists():
        raise FileNotFoundError(
            f"ecoli_synthetic axis vocab missing: {VOCAB_PATH}. "
            f"Run data/ecoli_synthetic/get_data.py first.")
    z = np.load(str(VOCAB_PATH), allow_pickle=True)
    return ([str(x) for x in z["ko_gene_names"]],
            [str(x) for x in z["gene_names"]])


class _AdoptedTransformer(Predictor):
    """Base for the nine adopted transformer variants (name per subclass).

    `is_adopted` marks these as externally-computed predictions adopted by file
    lookup; `is_external` is deliberately NOT set (that would trigger the DL
    adapter's fold-alignment gate at predict time — these are name-aligned by
    construction). `needs_training=False` so `cmd_predict` just calls `predict`.
    """
    needs_training = False
    is_adopted = True     # externally-computed; no synthetic L4 contract
    has_drop_rule = True  # adopted preds may not cover every (bin, ko) test cell
    scenarios = ["UnseenPert", "UnseenCell", "UnseenBoth", "UnseenPair",
                 "UnseenDose", "UnseenCombo"]

    def fit(self, store, scenario, fold):  # noqa: D401 — external, no-op
        return None

    def _resolve_npz(self, scenario: str, fold: int) -> Path:
        code = _SCENARIO_CODE[scenario]
        seed = _SEEDS[fold]
        pat = str(EXP4_RUNS / f"{code}_{self.name}_s{seed}_*" /
                  "test_predictions.npz")
        hits = sorted(glob.glob(pat))
        if not hits:
            raise FileNotFoundError(
                f"{self.name}: no exp4 prediction for {scenario}/fold{fold} "
                f"(pattern {pat})")
        return Path(hits[-1])  # latest timestamped run

    def predict(self, store, scenario, fold):
        path = self._resolve_npz(scenario, fold)
        z = np.load(str(path))
        preds = np.asarray(z["predictions"], dtype=np.float32)   # (nb, nk, ng)
        src_ko_idx = np.asarray(z["test_ko_indices"]).astype(int)
        src_bin_idx = np.asarray(z["test_bin_indices"]).astype(int)

        ko_vocab, gene_vocab = _source_vocab()
        if preds.shape[-1] != len(gene_vocab):
            raise ValueError(
                f"{self.name}: pred gene axis {preds.shape[-1]} != vocab "
                f"{len(gene_vocab)} for {scenario}/fold{fold}")
        src_ko_pos = {ko_vocab[i]: j for j, i in enumerate(src_ko_idx)}
        src_bin_pos = {f"bin_{int(b):02d}": i for i, b in enumerate(src_bin_idx)}

        # gene axis: src order == store order (both meta/gene_names) by construction;
        # verify and fall back to a name map if it ever diverges.
        if list(store.gene_names) == list(gene_vocab):
            gene_col = None  # identity
        else:
            g2s = {g: i for i, g in enumerate(store.gene_names)}
            gene_col = np.array([g2s.get(g, -1) for g in gene_vocab], dtype=int)
            if (gene_col < 0).all():
                raise ValueError(f"{self.name}: no gene overlap with store")

        split = store.split(scenario, fold)
        tbi = split.test_bin_indices_or_all(store.n_bins)
        tki = split.test_ko_indices
        out = np.full((len(tbi), len(tki), store.n_genes), np.nan, dtype=np.float32)

        n_missing = 0
        for a, b in enumerate(tbi):
            bn = store.bin_names[int(b)]
            si = src_bin_pos.get(bn)
            if si is None:
                n_missing += len(tki)
                continue
            for c, k in enumerate(tki):
                sj = src_ko_pos.get(store.ko_names[int(k)])
                if sj is None:
                    n_missing += 1
                    continue
                vec = preds[si, sj]
                if gene_col is None:
                    out[a, c, :] = vec
                else:
                    valid = gene_col >= 0
                    out[a, c, gene_col[valid]] = vec[valid]
        if n_missing:
            log.warning("%s: %d/%d (bin, ko) test cells had no matching prediction "
                        "-> NaN (skipped) for %s/%s/fold%d", self.name, n_missing,
                        len(tbi) * len(tki), store.dataset, scenario, fold)
        return out


# Materialize + register the nine concrete variants.
for _m in MODELS:
    _cls = type(_m, (_AdoptedTransformer,), {"name": _m})
    register(_cls)
    globals()[_m] = _cls

__all__ = ["MODELS", *MODELS]
