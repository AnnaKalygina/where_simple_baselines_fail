"""In-process torch training internals for `TorchPredictor`.

Underscore-prefixed on purpose: `base._ensure_predictors_loaded` skips `_`-named
modules, so nothing here is imported when the predictor registry is built. That
keeps the registry importable in environments without torch (`preprocess`), and
lets the modules in this package import torch at module level — they are pulled
in lazily, from inside `TorchPredictor._train`/`_infer`.

`recipe.py` is the exception: it holds plain constants and imports no torch, so a
predictor class can declare its hyperparameters without dragging torch in.
"""
