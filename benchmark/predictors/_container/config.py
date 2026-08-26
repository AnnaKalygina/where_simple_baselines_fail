"""Build the combined-form container ``config.json`` (see ``docker/CONTRACT.md``).

Host-side, dependency-light, unit-testable construction of the ``config.json`` the
vendored model wrappers read. All schema idiosyncrasy is *normalized* here (not
hidden): one canonical covariate key, one ``data_path`` h5ad + a per-cell
``split_name`` obs column + marginal condition lists — the wrapper slices itself.

Formerly the ``AtheusConfigBuilder`` class in ``container_predictor.py``; now plain
module functions (it held no state).
"""
from __future__ import annotations

from typing import Dict, List, Optional

from benchmark.config import h5ad_path, split_obs_column

# In-container canonical rw mount point (must match runner + docker/CONTRACT.md).
C_MODEL_OUTPUT = "/model_output"

# The cell-type (bin) axis obs column — data_loader derives store.bin_names from
# obs['cell_type'] for cell-axis regimes (data_loader.py:511-514). This is the
# single canonical covariate key handed to the container (F4: was duplicated as
# both covariate_field and covariate_key, always set to the same value).
COVARIATE_COLUMN = "cell_type"


def derive_conditions(obs, split_col: str) -> Dict[str, List[str]]:
    """Per-split unique perturbation labels — a **verbatim** port of the reference
    ``cellsimbench.core.data_manager.get_perturbation_conditions`` (so we
    reproduce): ``obs.condition.unique()`` within each split value, then drop any
    label containing the substring ``ctrl`` or equal to ``*``.

    Note the reference's substring filter: ``'control'`` (which has no ``ctrl``
    substring) is KEPT and later mapped to ``ctrl`` by the wrapper; a control
    literally labelled ``ctrl``/``ctrl_iegfp`` is dropped here. Either way this
    matches the reference exactly.
    """
    split_data = obs[split_col].astype(str)
    cond = obs["condition"].astype(str)

    def _conds(value: str) -> List[str]:
        uniq = cond[split_data == value].unique().tolist()
        return [c for c in uniq if "ctrl" not in c and c != "*"]

    return {"train": _conds("train"), "val": _conds("val"), "test": _conds("test")}


def build_config(mode: str, *, dataset: str, scenario: str, fold: int,
                 model_name: str, seed: int, hyperparameters: dict,
                 extra_config: Optional[dict] = None,
                 output_path_host: Optional[str] = None) -> dict:
    """Build the combined-form ``config.json`` for ``mode`` ('train'|'predict')."""
    import anndata as ad

    data_path = h5ad_path(dataset)                    # host path; runner → /data/<name>
    split_name = split_obs_column(scenario, fold)
    obs = ad.read_h5ad(data_path, backed="r").obs     # obs in memory, X lazy
    if split_name not in obs.columns:
        raise KeyError(f"{dataset}: split column {split_name!r} not in obs")
    conds = derive_conditions(obs, split_name)

    cfg: dict = {
        "mode": mode,
        "data_path": data_path,
        "split_name": split_name,
        "covariate_key": COVARIATE_COLUMN,            # single canonical covariate key
        "hyperparameters": dict(hyperparameters),
    }
    cfg.update(dict(extra_config or {}))

    if mode == "train":
        cfg.update({
            "model": model_name, "dataset": dataset, "scenario": scenario,
            "fold": int(fold), "seed": int(seed),
            "train_conditions": conds["train"],
            "val_conditions": conds["val"],
            "test_conditions": conds["test"],
            "output_dir": C_MODEL_OUTPUT,             # in-container rw mount
            "checkpoint_dir": C_MODEL_OUTPUT,
        })
    elif mode == "predict":
        if not output_path_host:
            raise ValueError("predict config needs output_path_host")
        cfg.update({
            "seed": int(seed),                        # SAME seed as train (predict-path prepare_split + RNGs)
            "test_conditions": conds["test"],         # the held-out perturbations
            "model_path": C_MODEL_OUTPUT,             # must contain processed_data/
            "output_path": output_path_host,          # host; runner → /model_output/<name>
        })
    else:
        raise ValueError(f"mode must be train|predict, got {mode!r}")
    return cfg


__all__ = ["C_MODEL_OUTPUT", "COVARIATE_COLUMN", "derive_conditions", "build_config"]
