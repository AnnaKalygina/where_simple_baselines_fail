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
    regime = _require(cfg, "regime", str, where)
    if regime not in REGIMES:
        raise ContractError(f"{where}: regime {regime!r} not one of {REGIMES}")
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
                 ("gene2go_path", str), ("early_stopping", dict),
                 ("checkpoint", dict),
                 # W&B experiment tracking (train only; endpoint via WANDB_* env).
                 ("wandb", bool), ("wandb_project", str), ("wandb_run", str)):
        _optional(cfg, k, t, where)
    return cfg


def validate_predict_config(cfg: dict) -> dict:
    """Validate a combined-form PREDICT config; raise ContractError or return cfg."""
    where = "predict config"
    if cfg.get("mode") != "predict":
        raise ContractError(f"{where}: mode must be 'predict', got {cfg.get('mode')!r}")
    # Predict re-reads the SAME combined h5ad (must still carry the test cells +
    # split column) and reloads the trained model dir.
    _require(cfg, "data_path", str, where)
    _require(cfg, "split_name", str, where)
    _require(cfg, "model_path", str, where)
    _require(cfg, "output_path", str, where)
    _require(cfg, "test_conditions", list, where)
    for k, t in (("seed", int), ("covariate_key", str),
                 ("hyperparameters", dict), ("gene2go_path", str)):
        _optional(cfg, k, t, where)
    return cfg


def _selftest() -> None:
    ok_train = {"mode": "train", "model": "gears", "dataset": "adamson16",
                "scenario": "UnseenPert", "regime": "UnseenPert", "fold": 0,
                "seed": 42, "data_path": "/data/adamson16_processed.h5ad",
                "split_name": "split_UnseenPert_fold_0", "covariate_key": "cell_type",
                "train_conditions": ["AARS"], "val_conditions": ["BRCA1"],
                "test_conditions": ["TP53"],
                "output_dir": "/model_output", "checkpoint_dir": "/model_output",
                "hyperparameters": {"epochs": 20}}
    validate_train_config(ok_train)

    ok_predict = {"mode": "predict", "data_path": "/data/adamson16_processed.h5ad",
                  "split_name": "split_UnseenPert_fold_0", "model_path": "/model_output",
                  "output_path": "/model_output/predictions.h5ad",
                  "test_conditions": ["TP53"], "covariate_key": "cell_type"}
    validate_predict_config(ok_predict)

    def _expect_fail(fn, cfg, needle):
        try:
            fn(cfg)
        except ContractError as e:
            assert needle in str(e), f"wrong error: {e}"
        else:
            raise AssertionError(f"expected ContractError containing {needle!r}")

    _expect_fail(validate_train_config, {**ok_train, "mode": "predict"}, "mode must be 'train'")
    _expect_fail(validate_train_config, {**ok_train, "scenario": "S1"}, "not one of")
    _expect_fail(validate_train_config, {k: v for k, v in ok_train.items() if k != "data_path"},
                 "missing required field 'data_path'")
    _expect_fail(validate_train_config, {k: v for k, v in ok_train.items() if k != "split_name"},
                 "missing required field 'split_name'")
    _expect_fail(validate_train_config, {k: v for k, v in ok_train.items() if k != "covariate_key"},
                 "missing required field 'covariate_key'")
    _expect_fail(validate_train_config, {**ok_train, "test_conditions": "TP53"},
                 "must be list")
    _expect_fail(validate_train_config, {**ok_train, "fold": -1}, "fold must be >= 0")
    _expect_fail(validate_predict_config, {k: v for k, v in ok_predict.items()
                                           if k != "output_path"}, "missing required field 'output_path'")
    _expect_fail(validate_predict_config, {k: v for k, v in ok_predict.items()
                                           if k != "data_path"}, "missing required field 'data_path'")
    print("contract.py SELFTEST PASSED (combined-form: valid configs accepted; "
          "9 invalid configs rejected)")


if __name__ == "__main__":
    _selftest()
