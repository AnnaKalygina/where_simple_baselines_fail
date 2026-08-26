"""Schema validators for the container train/predict config (see docker/CONTRACT.md).

Fail-fast, dependency-free checks so a malformed config is caught on the HOST
before a container is launched — and so a vendored wrapper (which won't have the
`benchmark` package installed) can validate the config it receives with the same
rules. The six regime names are hardcoded here because they ARE part of the
contract (they must not drift with the benchmark package).

Canonical form is the **combined (column-native)** config: one `data_path`
h5ad + a per-cell `split_name` column + marginal condition lists. The
pre-subsetted `train_h5ad`/`val_h5ad` form is NOT part of this contract (no
vendored model uses it); see docker/CONTRACT.md.
"""
from __future__ import annotations

REGIMES = ("UnseenPert", "UnseenCell", "UnseenBoth", "UnseenPair",
           "UnseenDose", "UnseenCombo")


class ContractError(ValueError):
    """A train/predict config violates the container contract."""


def _require(cfg: dict, key: str, types, where: str):
    if key not in cfg:
        raise ContractError(f"{where}: missing required field {key!r}")
    if not isinstance(cfg[key], types):
        tn = getattr(types, "__name__", str(types))
        raise ContractError(f"{where}: field {key!r} must be {tn}, "
                            f"got {type(cfg[key]).__name__}")
    return cfg[key]


def _optional(cfg: dict, key: str, types, where: str):
    if key in cfg and not isinstance(cfg[key], types):
        tn = getattr(types, "__name__", str(types))
        raise ContractError(f"{where}: optional field {key!r} must be {tn} "
                            f"when present, got {type(cfg[key]).__name__}")


#: Fields this schema knows about. Anything else is a model-specific extra,
#: injected from `model.yaml`'s `extra_config:` block (e.g. GEARS' `gene2go_path`,
#: PRESAGE's `presage_cache_path`). Those are NOT named here on purpose: this
#: module is the shared host<->container contract, and listing one model's keys in
#: it makes every other model look like a special case. They are still checked —
#: `_check_extras` requires them to be JSON scalars, which is all this layer can
#: meaningfully say about a key it does not own.
_KNOWN_FIELDS = frozenset({
    "mode", "model", "dataset", "scenario", "fold", "seed",
    "data_path", "split_name", "covariate_key",
    "train_conditions", "val_conditions", "test_conditions",
    "hyperparameters", "output_dir", "checkpoint_dir", "model_path", "output_path",
    "early_stopping", "checkpoint", "wandb", "wandb_project", "wandb_run",
})


def _check_extras(cfg: dict, where: str) -> None:
    """Model-specific extras must be JSON scalars (they are hashed + serialized)."""
    bad = {k: type(cfg[k]).__name__ for k in set(cfg) - _KNOWN_FIELDS
           if not isinstance(cfg[k], (str, int, float, bool))}
    if bad:
        raise ContractError(
            f"{where}: model-specific extra field(s) {bad} must be JSON scalars "
            f"(they come from model.yaml `extra_config:`)")


def validate_train_config(cfg: dict) -> dict:
    """Validate a combined-form TRAIN config; raise ContractError or return cfg."""
    where = "train config"
    if cfg.get("mode") != "train":
        raise ContractError(f"{where}: mode must be 'train', got {cfg.get('mode')!r}")
    _require(cfg, "model", str, where)
    _require(cfg, "dataset", str, where)
    scenario = _require(cfg, "scenario", str, where)
    if scenario not in REGIMES:
        raise ContractError(f"{where}: scenario {scenario!r} not one of {REGIMES}")
    fold = _require(cfg, "fold", int, where)
    if fold < 0:
        raise ContractError(f"{where}: fold must be >= 0, got {fold}")
    _require(cfg, "seed", int, where)
    # Combined-form intake: one h5ad + a per-cell split column + condition lists.
    _require(cfg, "data_path", str, where)
    _require(cfg, "split_name", str, where)
    _require(cfg, "covariate_key", str, where)     # single canonical covariate key (F4)
    for k in ("train_conditions", "val_conditions", "test_conditions"):
        _require(cfg, k, list, where)
    # optional but type-checked when present
    for k, t in (("hyperparameters", dict),
                 ("output_dir", str), ("checkpoint_dir", str),
                 ("early_stopping", dict), ("checkpoint", dict),
                 # W&B experiment tracking (train only; endpoint via WANDB_* env).
                 ("wandb", bool), ("wandb_project", str), ("wandb_run", str)):
        _optional(cfg, k, t, where)
    _check_extras(cfg, where)
    return cfg


def validate_predict_config(cfg: dict) -> dict:
    """Validate a combined-form PREDICT config; raise ContractError or return cfg."""
    where = "predict config"
    if cfg.get("mode") != "predict":
        raise ContractError(f"{where}: mode must be 'predict', got {cfg.get('mode')!r}")
    # Predict re-reads the SAME combined h5ad (must still carry the test cells +
    # split column) and reloads the trained model dir.
    #
    # Deliberately fewer REQUIRED fields than the train validator: predict needs
    # only what it takes to locate the model and the test cells. `model`,
    # `dataset`, `fold` and the train/val condition lists describe how training
    # was configured, which predict does not re-decide — a container that needs
    # them reads them back from its own checkpoint, not from this config.
    _require(cfg, "data_path", str, where)
    _require(cfg, "split_name", str, where)
    _require(cfg, "model_path", str, where)
    _require(cfg, "output_path", str, where)
    _require(cfg, "test_conditions", list, where)
    for k, t in (("seed", int), ("covariate_key", str),
                 ("hyperparameters", dict)):
        _optional(cfg, k, t, where)
    _check_extras(cfg, where)
    return cfg
