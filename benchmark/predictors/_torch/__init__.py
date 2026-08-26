"""In-process torch training internals for `TorchPredictor`.

Underscore-prefixed so `base._ensure_predictors_loaded` skips it: nothing here is
auto-imported when the predictor registry is built. Note the prefix alone does
not guarantee that — `_container` is `_`-prefixed too but IS imported eagerly, by
`container_predictor` at module level. What keeps this package lazy is that
`torch_predictor` imports it only from inside `_train`/`_infer`.

That laziness is what lets these modules import torch at module level while the
registry still loads in environments without torch (`preprocess`).

`recipe.py` is the exception: it holds plain constants and imports no torch, so a
predictor class can declare its hyperparameters without dragging torch in.
"""
