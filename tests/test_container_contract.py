"""The host<->container config contract.

Ported from `contract.py`'s `__main__` selftest so it runs on every change
rather than when someone remembers the command. Same cases, expressed as
parametrised tests so a failure names the field that broke instead of stopping
at the first one.

The contract matters more as models are added: it is the schema every container
wrapper is written against, and a malformed config that slips through here
surfaces as a confusing crash deep inside someone else's container.
"""

from __future__ import annotations

import pytest

from benchmark.predictors._container.contract import (
    ContractError, validate_predict_config, validate_train_config)


OK_TRAIN = {
    "mode": "train", "model": "gears", "dataset": "adamson16",
    "scenario": "UnseenPert", "fold": 0,
    "seed": 42, "data_path": "/data/adamson16_processed.h5ad",
    "split_name": "split_UnseenPert_fold_0", "covariate_key": "cell_type",
    "train_conditions": ["AARS"], "val_conditions": ["BRCA1"],
    "test_conditions": ["TP53"],
    "output_dir": "/model_output", "checkpoint_dir": "/model_output",
    "hyperparameters": {"epochs": 20},
}

OK_PREDICT = {
    "mode": "predict", "data_path": "/data/adamson16_processed.h5ad",
    "split_name": "split_UnseenPert_fold_0", "model_path": "/model_output",
    "output_path": "/model_output/predictions.h5ad",
    "test_conditions": ["TP53"], "covariate_key": "cell_type",
}


def _without(cfg: dict, key: str) -> dict:
    return {k: v for k, v in cfg.items() if k != key}


# ---------------------------------------------------------------------------
# Valid configs must be accepted
# ---------------------------------------------------------------------------

def test_a_valid_train_config_is_accepted():
    validate_train_config(dict(OK_TRAIN))


def test_a_valid_predict_config_is_accepted():
    validate_predict_config(dict(OK_PREDICT))


# ---------------------------------------------------------------------------
# Each way a config can be wrong
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cfg,needle", [
    ({**OK_TRAIN, "mode": "predict"}, "mode must be 'train'"),
    ({**OK_TRAIN, "scenario": "S1"}, "not one of"),
    (_without(OK_TRAIN, "data_path"), "missing required field 'data_path'"),
    (_without(OK_TRAIN, "split_name"), "missing required field 'split_name'"),
    (_without(OK_TRAIN, "covariate_key"), "missing required field 'covariate_key'"),
    ({**OK_TRAIN, "test_conditions": "TP53"}, "must be list"),
    ({**OK_TRAIN, "fold": -1}, "fold must be >= 0"),
], ids=["wrong-mode", "legacy-scenario-name", "no-data_path", "no-split_name",
        "no-covariate_key", "conditions-not-a-list", "negative-fold"])
def test_invalid_train_config_is_rejected(cfg, needle):
    with pytest.raises(ContractError, match=needle.replace("'", "'")):
        validate_train_config(cfg)


@pytest.mark.parametrize("cfg,needle", [
    (_without(OK_PREDICT, "output_path"), "missing required field 'output_path'"),
    (_without(OK_PREDICT, "data_path"), "missing required field 'data_path'"),
], ids=["no-output_path", "no-data_path"])
def test_invalid_predict_config_is_rejected(cfg, needle):
    with pytest.raises(ContractError, match=needle):
        validate_predict_config(cfg)


def test_scenario_must_be_pascal_case_not_a_legacy_sN_name():
    """`S1`..`S6` are the old external naming. Accepting them here would let a
    container be configured with a scenario the benchmark cannot map back."""
    with pytest.raises(ContractError, match="not one of"):
        validate_train_config({**OK_TRAIN, "scenario": "S3"})


# ---------------------------------------------------------------------------
# Model-specific extras (`model.yaml` -> `extra_config:`)
# ---------------------------------------------------------------------------
# The schema deliberately does NOT name these — GEARS' `gene2go_path`, PRESAGE's
# `presage_cache_path` — because this module is the shared contract, and listing
# one model's keys in it makes every other model look like a special case. What
# it can still say is that a key it does not own must be a JSON scalar: extras
# are hashed into the run fingerprint and serialised into config.json, so a
# nested value fails later and far from its cause.


@pytest.mark.parametrize("extra", [
    {"gene2go_path": "/app_code/gears/gene2go_all.pkl"},      # GEARS
    {"presage_cache_path": "/opt/presage_cache"},             # PRESAGE
    {"some_flag": True, "some_count": 3, "some_ratio": 0.5},  # any scalar
])
def test_scalar_extras_pass_without_being_named_in_the_schema(extra):
    validate_train_config({**OK_TRAIN, **extra})
    validate_predict_config({**OK_PREDICT, **extra})


@pytest.mark.parametrize("bad", [
    {"gene2go_path": ["a", "b"]},
    {"presage_cache_path": {"nested": 1}},
])
def test_non_scalar_extras_are_rejected(bad):
    for validate, cfg in ((validate_train_config, OK_TRAIN),
                          (validate_predict_config, OK_PREDICT)):
        with pytest.raises(ContractError, match="JSON scalars"):
            validate({**cfg, **bad})


def test_a_known_field_keeps_its_own_type_check_and_is_not_treated_as_an_extra():
    """`hyperparameters` is a dict on purpose; the scalar rule must not catch it."""
    validate_train_config({**OK_TRAIN, "hyperparameters": {"epochs": 20}})
    with pytest.raises(ContractError, match="must be dict"):
        validate_train_config({**OK_TRAIN, "hyperparameters": "epochs=20"})


# ---------------------------------------------------------------------------
# `regime` was removed from the schema
# ---------------------------------------------------------------------------
# It was written as a verbatim copy of `scenario` and read by no wrapper — the
# same defect `_container/config.py` records having already fixed once, when
# `covariate_field`/`covariate_key` were the duplicated pair. A field nothing
# reads is a field that can silently disagree with the one that matters.


def test_regime_is_no_longer_emitted_by_build_config():
    from benchmark.predictors._container.config import build_config
    import inspect
    assert "regime" not in inspect.getsource(build_config)


def test_a_stray_regime_is_now_just_an_extra_and_must_be_scalar():
    """Removing it from the schema must not turn a leftover `regime:` into a
    hard error for anyone mid-migration — it falls through to the extras rule."""
    validate_train_config({**OK_TRAIN, "regime": "UnseenPert"})
    with pytest.raises(ContractError, match="JSON scalars"):
        validate_train_config({**OK_TRAIN, "regime": ["UnseenPert"]})
