"""The KxN transformer depth sweep, trained in this repo.

Nine `GeneTransformer` variants: an encoder block of K layers applied N times
with tied weights. The set is a designed comparison, not an arbitrary
collection — `12x1, 6x2, 4x3, 3x4, 2x6` all reach effective depth K*N = 12 by
different factorisations, against the plain-depth references `6x1, 4x1, 3x1,
2x1`. Dropping variants dissolves the comparison.

These used to be ADOPTED predictors: thin adapters that read predictions
computed in a separate project and scored them here, with `needs_training =
False` and a run location pointed at by an env var. They now train in-process
through `TorchPredictor`, and this repo depends on nothing outside itself.

Currently exercised on `ecoli_synthetic`. The architecture puts one token per
gene, so attention cost is quadratic in panel size; larger real panels need a
feasibility measurement before being enabled here.
"""

from __future__ import annotations

from typing import Dict

from benchmark.predictors._torch import recipe
from benchmark.predictors.base import register
from benchmark.predictors.torch_predictor import TorchPredictor

# Every regime ecoli_synthetic defines. The model conditions on a per-gene
# perturbation marker plus a control profile, so nothing about it is specific to
# a regime; what changes is only which (bin, ko) cells are held out.
_SCENARIOS = ["UnseenPert", "UnseenCell", "UnseenBoth",
              "UnseenPair", "UnseenDose", "UnseenCombo"]


class _KxNTransformer(TorchPredictor):
    """A single `transformer_KxN` variant; K and N are parsed from `name`."""

    scenarios = _SCENARIOS
    wandb_project = "vcr-transformers"

    def architecture(self, n_genes: int) -> Dict:
        return recipe.architecture(self.name)

    @property
    def effective_depth(self) -> int:
        """K*N — the quantity the sweep holds constant across factorisations."""
        return recipe.effective_depth(self.name)


# One registered predictor per variant. Generated rather than written out nine
# times: only the name differs, and hand-copying invites the model list and the
# registry to drift apart.
for _name in recipe.MODELS:
    _cls = type(_name, (_KxNTransformer,),
                {"name": _name,
                 # `type()` on an ABC-derived base records __module__ as "abc",
                 # which would file these under the wrong predictor category.
                 "__module__": __name__,
                 "__doc__": f"GeneTransformer {_name} "
                            f"(effective depth {recipe.effective_depth(_name)})."})
    register(_cls)
    globals()[_name] = _cls

__all__ = list(recipe.MODELS)
