"""Training recipe + architecture mapping for the in-repo transformer sweep.

These are IN-REPO CONSTANTS, deliberately. The values were transcribed once from
the original depth-hypothesis sweep configs; nothing here reads that project at
run time, and the repo carries no dependency on it. Same intent as
`docker/gears/model.yaml`, which holds GEARS' recipe next to GEARS.

No torch import — a predictor class must be able to declare its hyperparameters
in an environment that has no torch (see `_torch/__init__.py`).
"""

from __future__ import annotations

import re
from typing import Dict, List

# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------

# `transformer_KxN`: an encoder block of K layers, applied N times with tied
# weights. The sweep is a designed experiment, not an arbitrary set: 12x1, 6x2,
# 4x3, 3x4 and 2x6 all reach effective depth K*N = 12 by different
# layer/recurrence factorisations, and 6x1/4x1/3x1/2x1 are the plain-depth
# references. Keep the whole set, or the comparison it encodes is lost.
MODELS: List[str] = [
    "transformer_12x1", "transformer_6x2", "transformer_6x1",
    "transformer_4x3", "transformer_4x1", "transformer_3x4",
    "transformer_3x1", "transformer_2x6", "transformer_2x1",
]

_KXN = re.compile(r"^transformer_(\d+)x(\d+)$")

# Shared across every variant; only `layers`/`num_steps` differ, which is what
# makes the sweep a controlled comparison.
ARCH_BASE: Dict[str, object] = {
    "d_model": 256,
    "nhead": 8,
    "ff_mult": 4,
    "dropout": 0.1,
}


def architecture(name: str) -> Dict[str, object]:
    """`transformer_KxN` -> GeneTransformer kwargs (minus `num_genes`)."""
    m = _KXN.match(name)
    if not m:
        raise ValueError(
            f"cannot parse model name {name!r}; expected transformer_KxN "
            f"(e.g. transformer_6x2)")
    return {**ARCH_BASE, "layers": int(m.group(1)), "num_steps": int(m.group(2))}


def effective_depth(name: str) -> int:
    """K*N — how deep the computation is, regardless of how it is factorised."""
    a = architecture(name)
    return int(a["layers"]) * int(a["num_steps"])


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

TRAINING: Dict[str, object] = {
    "epochs": 40,
    "lr": 2e-4,
    "weight_decay": 1e-2,
    "warmup_epochs": 3,
    "lr_patience": 5,        # ReduceLROnPlateau on val loss
    "lr_factor": 0.5,
    "patience": 10,          # early stop
    "min_epochs": 5,
    "grad_clip_norm": 5.0,
    "batch_size": 32,
    "num_workers": 4,
}

# Per-KO sampling weights ~ ||delta||_2 ** alpha, so perturbations with a real
# transcriptional effect are seen more often than near-silent ones. NOTE this is
# NOT `store.per_pert_weights`, which is DEG/log2fc-derived and answers a
# different question (significance, not effect magnitude).
SAMPLE_WEIGHT: Dict[str, object] = {"method": "l2_norm", "alpha": 1.0}

# fold index -> seed. Fixed so a fold means the same thing across variants.
FOLD_SEEDS: List[int] = [42, 1337, 7]


def seed_for_fold(fold: int) -> int:
    return FOLD_SEEDS[fold] if 0 <= fold < len(FOLD_SEEDS) else 42 + fold
