"""
scGPT model wrapper for the in-repo training container.

Vendored from the atheus PMOB drop
(``atheus/Perturbation-Models-Outperform-Baselines/docker/scgpt/scgpt_wrapper.py``,
1757 lines). Every change carries a numbered ``VENDORED (Vn)`` marker in-place;
the table with the reasoning is docker/scgpt/SCGPT_NOTES.md §2.

scGPT here is FINE-TUNED, not pretrained: a foundation checkpoint (whole_human,
cellxgene-census, 12 layers / d_model 512) is loaded and every parameter is then
trained on the perturbation task. See SCGPT_NOTES.md §1.

Contract: docker/CONTRACT.md. Entry: run_model.py {train|predict} /config.json.
"""

import logging
import pickle
import json
import os
import random
import time
import copy
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Mapping

import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Bernoulli
from torch.nn import functional as F
from torch.optim.lr_scheduler import StepLR
from anndata import AnnData
from tqdm import tqdm

from gears import PertData
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

# scGPT imports
from scgpt.model import TransformerGenerator
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scgpt.utils import map_raw_id_to_vocab_id
from scgpt.utils.util import load_pretrained

log = logging.getLogger(__name__)

warnings.filterwarnings("ignore")


# VENDORED (V1): drop the `cellsimbench` dependency, exactly as gears_wrapper.py
# and presage_wrapper.py did. The two symbols the atheus wrapper imported from it
# — a Path-aware JSON encoder and a slim data loader — are inlined here, so there
# is no host package to install and no extra bind mount.
class PathEncoder(json.JSONEncoder):
    """JSON encoder that serializes ``pathlib.Path`` as ``str``."""

    def default(self, obj):
        if isinstance(obj, Path):
            return str(obj)
        return json.JSONEncoder.default(self, obj)


class DataManager:
    """Minimal data loader: read one combined h5ad from ``config['data_path']``.

    The wrapper uses only ``DataManager(config)`` and ``.load_dataset()``. The
    combined h5ad carries the per-cell split column and every covariate inline;
    the wrapper slices it itself (see docker/CONTRACT.md).
    """

    def __init__(self, dataset_config: Dict):
        self.config = dataset_config
        self.adata = None

    def load_dataset(self):
        path = Path(self.config["data_path"])
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")
        print(f"[harness] Loading dataset from {path} ...", flush=True)
        self.adata = sc.read_h5ad(path)
        self.adata.var.index.name = None
        print(f"[harness] Loaded AnnData with shape: {self.adata.shape}", flush=True)
        return self.adata


# VENDORED (V13): the atheus source hard-coded this list in FIVE places
# (`:775`, `:910`, `:961`, `:1130`, `:1555`), twice with a
# `# TODO: We should be passing the control value as a parameter` beside it.
# Five copies drift independently; one constant cannot.
CONTROL_LABELS = ("ctrl", "ctrl_iegfp", "control", "non-targeting")

#: scGPT's own canonical control label inside PertData.
SCGPT_CTRL = "ctrl"


def _is_control_label(name) -> bool:
    """Whole-string control match.

    Substring matching would corrupt gene names that merely contain "ctrl"
    (e.g. RctrlSEL), so this compares whole strings — the same rule
    presage_wrapper.py:104-117 uses.
    """
    return str(name) in CONTROL_LABELS


class GeneEncoder(nn.Module):
    """
    Gene/token encoder with built-in normalization.
    This is recommended by scGPT authors for better training stability.
    """
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings, embedding_dim, padding_idx=padding_idx
        )
        self.enc_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.embedding(x)  # (batch, seq_len, embsize)
        x = self.enc_norm(x)  # Apply LayerNorm for stability
        return x


def masked_mse_loss(
    input: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
    weights: torch.Tensor = None, reduction: str = 'sum'
) -> torch.Tensor:
    """
    Compute the masked MSE loss between input and target.

    VENDORED (V17). Upstream scGPT's own ``scgpt.loss.masked_mse_loss`` ends
    ``return loss / mask.sum()`` — a MEAN. The atheus reimplementation defaulted to
    ``reduction='sum'`` and never overrode it, so its training loss was
    ``batch x n_input_genes`` (~1e5) times larger than the loss ``lr: 1e-4`` — the
    authors' own value — was tuned against. We default to ``'mean'``, matching
    upstream.

    Why this matters less than it looks, and still matters: Adam is very nearly
    invariant to a constant rescaling of the loss (m and sqrt(v) scale together),
    so the LR is NOT effectively rescaled by 1e5. But two things in this training
    loop are NOT scale-invariant:
      * ``clip_grad_norm_(..., 1.0)`` — with gradients ~1e5x larger, clipping fires
        on essentially every step, replacing Adam's per-parameter step with a
        fixed-norm one;
      * the fp16 ``GradScaler`` — gradients that large risk overflowing fp16
        (max ~65504), making the scaler skip steps and back off repeatedly.
    Either way the optimisation is not the authors'. Pinned as the recipe key
    ``loss_reduction`` so the choice is explicit and auditable rather than a
    default nobody chose.

    Args:
        input: Predicted values [batch_size, n_features]
        target: Target values [batch_size, n_features]
        mask: Mask indicating which elements to include [batch_size, n_features]
        weights: Not used, kept for compatibility
        reduction: 'mean' | 'sum'.

    Returns:
        Masked MSE loss
    """
    mask = mask.float()
    squared_diff = (input - target) ** 2
    masked_diff = squared_diff * mask
    loss = masked_diff.sum()

    if reduction == 'mean':
        normalization = mask.sum()
        if normalization > 0:
            return loss / normalization
        return torch.tensor(0.0, device=input.device)
    if reduction == 'sum':
        return loss
    raise ValueError(f"Invalid reduction mode: {reduction}. Must be 'mean' or 'sum'.")


# VENDORED (V2): `masked_weighted_residual_loss`, `_build_weighted_residual_artifacts`
# and `_lookup_batch_weights` are NOT vendored. They implement an atheus-invented
# extension (`weighted_residual_loss`, disabled by default in scgpt.yaml) — the
# analogue of PRESAGE's `presage_gated.yaml`. `utils.py` and `build.sh` are not
# vendored either: nothing imports `utils.py`, and `build.sh` is docker-only.


class TransformerGeneratorWithLayerNorm(TransformerGenerator):
    """TransformerGenerator with LayerNorm on perturbation embeddings.

    NOTE (design, not a defect): scGPT's perturbation model has NO covariate
    input port. ``_encode`` below sums exactly three channels — gene token,
    expression value, perturbation flag. The parent class accepts
    ``domain_spec_batchnorm`` but never applies it, and takes no ``batch_labels``
    (that machinery lives in ``TransformerModel``, the cell-embedding model, not
    here). Cell identity therefore reaches the prediction ONLY through the input
    control cell's expression profile — which is why V7's covariate-matched
    control pairing is the mechanism that makes scGPT-ct cell-aware, and why no
    covariate embedding is added here. See SCGPT_NOTES.md §1.
    """

    def __init__(
        self,
        *args,
        **kwargs
    ):
        # Remove n_covariates if it exists in kwargs (the model has nowhere to put it)
        kwargs.pop('n_covariates', None)
        super().__init__(*args, **kwargs)

        # Override the perturbation encoder to use GeneEncoder with normalization
        # This follows the recommendation from scGPT authors for better training stability
        # The pert_pad_id is already set in parent class as self.pert_pad_id
        self.pert_encoder = GeneEncoder(3, self.d_model, padding_idx=self.pert_pad_id)

    def _encode(
        self,
        src: Tensor,
        values: Tensor,
        input_pert_flags,
        src_key_padding_mask: Tensor,
    ) -> Tensor:
        """Encode without covariates, using LayerNorm on perturbation embeddings."""
        src = self.encoder(src)  # (batch, seq_len, embsize)
        self.cur_gene_token_embs = src
        values = self.value_encoder(values)  # (batch, seq_len, embsize)
        perts = self.pert_encoder(input_pert_flags)  # (batch, seq_len, embsize) - now with LayerNorm

        total_embs = src + values + perts

        output = self.transformer_encoder(
            total_embs, src_key_padding_mask=src_key_padding_mask
        )
        return output  # (batch, seq_len, embsize)

    def forward(
        self,
        src: Tensor,
        values: Tensor,
        input_pert_flags: Tensor,
        src_key_padding_mask: Tensor,
        CLS: bool = False,
        CCE: bool = False,
        MVC: bool = False,
        ECS: bool = False,
        do_sample: bool = False,
    ) -> Mapping[str, Tensor]:
        """
        Args:
            src (:obj:`Tensor`): token ids, shape [batch_size, seq_len]
            values (:obj:`Tensor`): token values, shape [batch_size, seq_len]
            src_key_padding_mask (:obj:`Tensor`): mask for src, shape [batch_size, seq_len]
            CLS/CCE/MVC/ECS (:obj:`bool`): optional auxiliary heads.

        Returns:
            dict of output Tensors.
        """
        if self.explicit_zero_prob and not do_sample and not self.training:
            do_sample = True
            log.warning("Auto set do_sample to True when model is in eval mode.")

        # binning input gene values
        if self.n_input_bins > 0:
            from scgpt.preprocess import binning

            processed_values = torch.stack(
                [binning(row, n_bins=self.n_input_bins) for row in values], dim=0
            ).to(values.device)
        else:
            processed_values = values

        transformer_output = self._encode(
            src, processed_values, input_pert_flags, src_key_padding_mask
        )
        output = {}
        mlm_output = self.decoder(transformer_output, values)
        if self.explicit_zero_prob and do_sample:
            bernoulli = Bernoulli(probs=mlm_output["zero_probs"])
            output["mlm_output"] = bernoulli.sample() * mlm_output["pred"]
        else:
            output["mlm_output"] = mlm_output["pred"]  # (batch, seq_len)
        if self.explicit_zero_prob:
            output["mlm_zero_probs"] = mlm_output["zero_probs"]

        cell_emb = self._get_cell_emb_from_layer(transformer_output, values)
        if CLS:
            output["cls_output"] = self.cls_decoder(cell_emb)  # (batch, n_cls)
        if MVC:
            mvc_output = self.mvc_decoder(
                cell_emb,
                self.cur_gene_token_embs,
            )  # (batch, seq_len)
            if self.explicit_zero_prob and do_sample:
                bernoulli = Bernoulli(probs=mvc_output["zero_probs"])
                output["mvc_output"] = bernoulli.sample() * mvc_output["pred"]
            else:
                output["mvc_output"] = mvc_output["pred"]  # (batch, seq_len)
            if self.explicit_zero_prob:
                output["mvc_zero_probs"] = mvc_output["zero_probs"]
        if ECS:
            # customized cosine similarity (pytorch #78064)
            cell_emb_normed = F.normalize(cell_emb, p=2, dim=1)
            cos_sim = torch.mm(cell_emb_normed, cell_emb_normed.t())  # (batch, batch)
            mask = torch.eye(cos_sim.size(0)).bool().to(cos_sim.device)
            cos_sim = cos_sim.masked_fill(mask, 0.0)
            cos_sim = F.relu(cos_sim)
            output["loss_ecs"] = torch.mean(1 - (cos_sim - self.ecs_threshold) ** 2)

        return output

    def pred_perturb(
        self,
        batch_data,
        include_zero_gene="batch-wise",
        gene_ids=None,
        amp=True,
        do_sample=True,
    ) -> Tensor:
        """
        Args:
            batch_data: a PyG batch of prediction graphs.

        Returns:
            output Tensor of shape [N, seq_len]

        VENDORED (V12): ``do_sample`` was hard-coded ``True`` at the call site;
        it is now the recipe's ``do_sample``.
        """
        self.eval()
        device = next(self.parameters()).device
        batch_data.to(device)
        batch_size = len(batch_data.pert)
        x: torch.Tensor = batch_data.x
        ori_gene_values = x[:, 0].view(batch_size, -1)  # (batch_size, n_genes)
        pert_flags = x[:, 1].long().view(batch_size, -1)

        if include_zero_gene not in ("all", "batch-wise"):
            raise ValueError(
                f"pred_perturb: unsupported include_zero_gene={include_zero_gene!r}; "
                "expected 'all' or 'batch-wise'."
            )
        assert gene_ids is not None
        if include_zero_gene == "all":
            input_gene_ids = torch.arange(ori_gene_values.size(1), device=device)
        else:  # batch-wise
            input_gene_ids = (
                ori_gene_values.nonzero()[:, 1].flatten().unique().sort()[0]
            )
        input_values = ori_gene_values[:, input_gene_ids]
        input_pert_flags = pert_flags[:, input_gene_ids]

        mapped_input_gene_ids = map_raw_id_to_vocab_id(input_gene_ids, gene_ids)
        mapped_input_gene_ids = mapped_input_gene_ids.repeat(batch_size, 1)

        src_key_padding_mask = torch.zeros_like(
            input_values, dtype=torch.bool, device=device
        )
        with torch.cuda.amp.autocast(enabled=amp):
            output_dict = self(
                mapped_input_gene_ids,
                input_values,
                input_pert_flags,
                src_key_padding_mask=src_key_padding_mask,
                CLS=False,
                CCE=False,
                MVC=False,
                ECS=False,
                do_sample=do_sample,
            )
        output_values = output_dict["mlm_output"].float()
        pred_gene_values = torch.zeros_like(ori_gene_values)
        pred_gene_values[:, input_gene_ids] = output_values
        return pred_gene_values


def _as_row(x) -> np.ndarray:
    """Return a single cell's expression as a dense ``(1, n_genes)`` float array.

    VENDORED (V21): the atheus source called ``X.toarray()`` unconditionally, so a
    dense h5ad raised ``AttributeError`` deep inside the graph builder rather than
    at load. Every dataset here is CSR today; this makes that an assumption the
    code states rather than one it relies on.
    """
    if hasattr(x, "toarray"):
        return np.asarray(x.toarray(), dtype=np.float32).reshape(1, -1)
    return np.asarray(x, dtype=np.float32).reshape(1, -1)


class SCGPTPertData(PertData):
    """GEARS ``PertData`` subclass: custom splits, per-cell split tagging, and
    covariate-matched control pairing.

    Two structural departures from both upstream GEARS and the atheus drop:

    * **V6** every cell graph carries the source cell's ``obs[split_name]`` label,
      and :meth:`get_dataloader` buckets on it. Upstream buckets on
      ``set2conditions`` — a CONDITION-level list — which on cell-axis regimes
      pulls held-out cells into training, because a perturbation held out in cell
      type A legitimately remains ``train`` in cell type B and so appears in both
      lists.
    * **V7** control donors are drawn from a covariate- (and batch-) matched,
      split-restricted pool. Upstream GEARS draws
      ``self.ctrl_adata[np.random.randint(0, len(self.ctrl_adata), num_samples)]``
      — all controls, any cell type, any split.
    """

    def __init__(self, data_path):
        super().__init__(data_path)
        # Set by the wrapper before new_data_process(); see _configure_pairing().
        self.split_col = None
        self.cov_col = None
        self.batch_col = None
        self.rng = np.random.default_rng(0)
        self._pool_cache: Dict = {}
        self.pairing_tier_counts: Dict[str, int] = {}
        self._ctrl_meta = None
        self._ctrl_X_cache = None
        self.num_samples_per_cell = 1   # recipe: ctrl_samples_per_cell (V12)
        self.num_de_genes = 20          # recipe: num_de_genes (V12)
        self.split_sizes: Dict[str, int] = {}

    # ---------------------------------------------------------------- pairing

    def configure_pairing(self, *, split_col, cov_col, batch_col, seed,
                          num_samples_per_cell=1, num_de_genes=20):
        """Declare the columns V6/V7 key on. Must be called before graph building."""
        self.split_col = split_col
        self.cov_col = cov_col
        self.batch_col = batch_col
        self.num_samples_per_cell = int(num_samples_per_cell)
        self.num_de_genes = int(num_de_genes)
        self.rng = np.random.default_rng(int(seed))  # VENDORED (V3)
        self._pool_cache = {}
        self.pairing_tier_counts = {
            "cov+batch+split": 0, "cov+split": 0, "split_only": 0,
        }

    def _ctrl_X(self):
        """Control expression matrix, materialised once.

        ``self.ctrl_adata`` is an AnnData *view*; indexing ``.X`` on a view
        re-subsets the parent matrix on every access, which inside the per-cell
        donor loop is O(n_cells) subsetting per graph.
        """
        if getattr(self, "_ctrl_X_cache", None) is None:
            self._ctrl_X_cache = self.ctrl_adata.X
        return self._ctrl_X_cache

    def _ctrl_metadata(self) -> pd.DataFrame:
        """Per-control-cell (cov, batch, split) frame, positionally aligned."""
        if self._ctrl_meta is None:
            obs = self.ctrl_adata.obs
            n = self.ctrl_adata.n_obs
            self._ctrl_meta = pd.DataFrame({
                "cov": (obs[self.cov_col].astype(str).values
                        if self.cov_col and self.cov_col in obs.columns
                        else np.array([""] * n)),
                "batch": (obs[self.batch_col].astype(str).values
                          if self.batch_col and self.batch_col in obs.columns
                          else np.array([""] * n)),
                "split": (obs[self.split_col].astype(str).values
                          if self.split_col and self.split_col in obs.columns
                          else np.array([""] * n)),
            })
        return self._ctrl_meta

    def _ctrl_pool(self, cov: str, batch: str, allowed_splits: tuple) -> np.ndarray:
        """Positional indices into ``ctrl_adata`` for one (cov, batch, split) request.

        VENDORED (V7). Tiered, with the tier that fired COUNTED — a silent
        fallback is what makes a cell-blind model look cell-aware. Raises rather
        than crossing into another covariate: pairing a cell type's perturbed
        cells with another cell type's controls destroys the only channel that
        carries cell identity into the prediction.
        """
        key = (cov, batch, allowed_splits)
        cached = self._pool_cache.get(key)
        if cached is not None:
            return cached

        meta = self._ctrl_metadata()
        split_ok = meta["split"].isin(allowed_splits).to_numpy()
        if not self.split_col:
            split_ok = np.ones(len(meta), dtype=bool)

        tiers = []
        if self.cov_col:
            if self.batch_col:
                tiers.append(("cov+batch+split",
                              split_ok & (meta["cov"] == cov).to_numpy()
                              & (meta["batch"] == batch).to_numpy()))
            tiers.append(("cov+split", split_ok & (meta["cov"] == cov).to_numpy()))
            # NOTE: there is deliberately no split-only tier here. Relaxing the
            # covariate when a covariate has no eligible controls is precisely the
            # silent cross-covariate borrow this function exists to prevent — it
            # would pair one cell type's perturbed cells with another's basal
            # state, making the prediction cell-blind while obs['covariate'] still
            # claims otherwise. Raise instead; the caller widens `allowed_splits`
            # when a legitimate widening exists (val -> train).
        else:
            tiers.append(("split_only", split_ok))

        for name, mask in tiers:
            idx = np.flatnonzero(mask)
            if len(idx):
                self.pairing_tier_counts[name] = self.pairing_tier_counts.get(name, 0) + 1
                self._pool_cache[key] = idx
                return idx

        raise RuntimeError(
            f"no control cells available for cov={cov!r} batch={batch!r} "
            f"splits={allowed_splits} out of {len(meta)} control cells. Refusing to "
            f"borrow controls from another covariate (V7) — that would make the "
            f"prediction cell-blind while obs['covariate'] still says otherwise."
        )

    # ------------------------------------------------------ upstream overrides

    # PertData.get_pert_idx signature varies by gears version: some require
    # (pert_category, adata_), others only (pert_category). Try the 2-arg form,
    # fall back to 1-arg. Also catch IndexError raised by upstream gears when a
    # perturbation gene is not in self.gene_names (can happen for OOV combo
    # parts or when var has been pre-restricted). Return [-1] so the downstream
    # create_cell_graph skips it (already guards 0 <= idx < n_genes).
    def get_pert_idx(self, pert_category, adata_=None):
        try:
            try:
                return super().get_pert_idx(pert_category, adata_)
            except TypeError:
                return super().get_pert_idx(pert_category)
        except IndexError:
            return [-1]

    def prepare_split(
        self,
        split="simulation",
        seed=1,
        train_gene_set_size=0.75,
        combo_seen2_train_frac=0.75,
        combo_single_split_test_set_fraction=0.1,
        test_perts=None,
        only_test_set_perts=False,
        test_pert_genes=None,
        split_dict_path=None,
    ):
        """Extended prepare_split adding ``split='custom'`` via ``split_dict_path``.

        Under V6 the condition lists this loads are PROVENANCE only — they record
        which conditions each split contained. What actually buckets a graph into
        train/val is the per-cell tag read by :meth:`get_dataloader`.
        """
        self.split = split
        self.seed = seed
        self.subgroup = None
        self.train_gene_set_size = train_gene_set_size

        if split == "custom":
            if not split_dict_path or not os.path.exists(str(split_dict_path)):
                raise ValueError(
                    f"prepare_split(split='custom') needs a readable split_dict_path; "
                    f"got {split_dict_path!r}"
                )
            with open(split_dict_path, "rb") as f:
                self.set2conditions = pickle.load(f)
            log.info("Loaded custom split from %s", split_dict_path)
            return

        super().prepare_split(
            split=split,
            seed=seed,
            train_gene_set_size=train_gene_set_size,
            combo_seen2_train_frac=combo_seen2_train_frac,
            combo_single_split_test_set_fraction=combo_single_split_test_set_fraction,
            test_perts=test_perts,
            only_test_set_perts=only_test_set_perts,
            test_pert_genes=test_pert_genes,
        )

    def load(self, data_path=None):
        if data_path is None or not os.path.exists(data_path):
            raise ValueError(f"load: not a processed-dataset directory: {data_path!r}")
        log.info("Loading data from %s ...", data_path)
        adata_path = os.path.join(data_path, 'perturb_processed.h5ad')
        self.adata = sc.read_h5ad(adata_path)
        self.adata.var.index.name = None
        self.dataset_name = os.path.basename(os.path.normpath(data_path))
        self.dataset_path = data_path

        pyg_path = os.path.join(data_path, 'data_pyg')
        os.makedirs(pyg_path, exist_ok=True)   # was a non-recursive os.mkdir
        dataset_fname = os.path.join(pyg_path, 'cell_graphs.pkl')

        self.ctrl_adata = self.adata[self.adata.obs['condition'] == SCGPT_CTRL]
        self.gene_names = self.adata.var.gene_name

        if os.path.isfile(dataset_fname):
            log.info("Local copy of pyg dataset detected. Loading ...")
            with open(dataset_fname, "rb") as fh:
                self.dataset_processed = pickle.load(fh)
        else:
            log.info("Creating pyg object for each cell in the data ...")
            self.dataset_processed = self.create_dataset_file()
            log.info("Saving new dataset pyg object at %s", dataset_fname)
            with open(dataset_fname, "wb") as fh:
                pickle.dump(self.dataset_processed, fh)
        log.info("Done.")

    def load_adata_only(self, data_path):
        """Predict-time loader: read the processed h5ad WITHOUT the graph pickle.

        VENDORED (V11). ``_generate_predictions`` builds its own graphs from
        control cells via ``create_cell_graph_dataset_for_prediction`` and never
        touches ``dataset_processed``; the atheus predict path nonetheless
        unpickled the whole multi-GB ``cell_graphs.pkl``. Measured on adamson16
        that is ~5.6 GB of RAM read and discarded.
        """
        adata_path = os.path.join(data_path, 'perturb_processed.h5ad')
        if not os.path.exists(adata_path):
            raise FileNotFoundError(f"processed adata not found at {adata_path}")
        self.adata = sc.read_h5ad(adata_path)
        self.adata.var.index.name = None
        self.dataset_name = os.path.basename(os.path.normpath(data_path))
        self.dataset_path = data_path
        self.ctrl_adata = self.adata[self.adata.obs['condition'] == SCGPT_CTRL]
        self.gene_names = self.adata.var.gene_name
        self.dataset_processed = None
        log.info("Loaded processed adata %s (graphs NOT loaded — V11)", self.adata.shape)

    def create_cell_graph(self, X, y, de_idx, pert, pert_idx=None,
                          cov="", split="", batch=""):
        pert_feats = np.zeros(len(X[0]))
        if pert_idx is not None:
            n_genes = len(pert_feats)
            for p in pert_idx:
                idx = int(np.abs(p))
                # Some GEARS PertData versions return indices into a superset of
                # the dataset's gene list (e.g. pert genes not in HVG). Skip
                # those — they have no feature column to mark.
                if 0 <= idx < n_genes:
                    pert_feats[idx] = 1
        pert_feats = np.expand_dims(pert_feats, 0)
        feature_mat = torch.Tensor(np.concatenate([X, pert_feats])).T
        # VENDORED (V6): `split` is the source cell's per-cell obs[split_name]
        # label. get_dataloader buckets on THIS, not on a condition list.
        return Data(x=feature_mat, edge_index=None, edge_attr=None,
                    y=torch.Tensor(y), de_idx=de_idx, pert=pert,
                    cov=cov, split=split, batch=batch)

    def create_cell_graph_dataset(self, split_adata, pert_category, num_samples=1,
                                  num_de_genes=20):
        """Build one cell graph per (perturbed cell x control donor).

        VENDORED (V7): the control donor is drawn from a covariate- and
        batch-matched pool restricted to the *same* split as the perturbed cell
        (train graphs may only see train controls). VENDORED (V12):
        ``num_de_genes`` was hard-coded at 20.
        """
        adata_ = split_adata[split_adata.obs['condition'] == pert_category]
        de_genes = adata_.uns['rank_genes_groups_cov_all']
        obs = adata_.obs

        def _col(name):
            if name and name in obs.columns:
                return obs[name].astype(str).values
            return np.array([""] * adata_.n_obs)

        cov_vals = _col(self.cov_col)
        batch_vals = _col(self.batch_col)
        split_vals = _col(self.split_col)

        Xs, ys, metas = [], [], []

        if pert_category != SCGPT_CTRL:
            pert_idx = self.get_pert_idx(pert_category, adata_)
            pert_de_category = obs['condition_name'][0]
            de_idx = np.where(adata_.var_names.isin(
                np.array(de_genes[pert_de_category][:num_de_genes])))[0]

            for i, cell_z in enumerate(adata_.X):
                split_i = split_vals[i]
                # Train graphs may only borrow train controls. Val graphs prefer
                # val controls and fall back to train+val (mcfaline23 has ZERO
                # val controls, so this tier is load-bearing, not theoretical).
                allowed = ("train",) if split_i == "train" else ("val", "train")
                pool_idx = self._ctrl_pool(cov_vals[i], batch_vals[i], allowed)
                donors = self.rng.integers(0, len(pool_idx), num_samples)
                for d in donors:
                    Xs.append(self._ctrl_X()[pool_idx[int(d)]])
                    ys.append(cell_z)
                    metas.append((cov_vals[i], split_i, batch_vals[i]))
        else:
            pert_idx = None
            de_idx = [-1] * num_de_genes
            for i, cell_z in enumerate(adata_.X):
                Xs.append(cell_z)
                ys.append(cell_z)
                metas.append((cov_vals[i], split_vals[i], batch_vals[i]))

        cell_graphs = []
        for X, y, (cov, split, batch) in zip(Xs, ys, metas):
            cell_graphs.append(self.create_cell_graph(
                _as_row(X), _as_row(y), de_idx, pert_category, pert_idx,
                cov=cov, split=split, batch=batch))
        return cell_graphs

    def create_dataset_file(self):
        # GEARS' new_data_process calls create_dataset_file() purely for the side
        # effect, then pickles self.dataset_processed; upstream gears <0.0.3 does
        # self.dataset_processed = self.create_dataset_file(). Set AND return so
        # both call patterns work.
        self.dataset_processed = {}
        for p in tqdm(self.adata.obs['condition'].unique()):
            self.dataset_processed[p] = self.create_cell_graph_dataset(
                self.adata, p, num_samples=self.num_samples_per_cell,
                num_de_genes=self.num_de_genes)
        return self.dataset_processed

    def get_dataloader(self, batch_size, test_batch_size=None):
        """Bucket graphs by the PER-CELL split tag, not by a condition list.

        VENDORED (V6) — this replaces GEARS' ``get_dataloader`` wholesale. Also
        VENDORED (V11): ``drop_last=False`` is set explicitly here rather than
        inherited from ``cell-gears``, so a batch wider than the training split
        cannot yield zero optimiser steps and an rc-0 untrained checkpoint.
        """
        test_batch_size = test_batch_size or batch_size
        buckets: Dict[str, list] = {"train": [], "val": [], "test": []}
        untagged = 0
        for graphs in self.dataset_processed.values():
            for g in graphs:
                tag = getattr(g, "split", "") or ""
                if tag in buckets:
                    buckets[tag].append(g)
                else:
                    untagged += 1
        if untagged:
            raise RuntimeError(
                f"{untagged} cell graphs carry no usable per-cell split tag. The "
                f"V6 rebuild requires every graph to be tagged at build time; an "
                f"untagged graph means a stale cell_graphs.pkl (see V15)."
            )
        if not buckets["train"]:
            raise RuntimeError("empty training split after per-cell bucketing")
        if not buckets["val"]:
            raise RuntimeError(
                "empty validation split after per-cell bucketing — best-val "
                "checkpoint selection (V10) would have nothing to select on")

        self.split_sizes = {k: len(v) for k, v in buckets.items()}
        log.info("get_dataloader (per-cell V6): %s", self.split_sizes)

        self.dataloader = {
            "train_loader": DataLoader(buckets["train"], batch_size=batch_size,
                                       shuffle=True, drop_last=False),
            "val_loader": DataLoader(buckets["val"], batch_size=test_batch_size,
                                     shuffle=False, drop_last=False),
        }
        return self.dataloader

    def create_cell_graph_for_prediction(self, X, pert_idx, pert_gene):
        """Build one prediction graph: a control profile + the perturbation flags.

        VENDORED (V25): upstream set ``pert_feats[abs(p)] = np.sign(p)``, inherited
        from GEARS where the sign encodes direction. scGPT's flag vocabulary is
        {0 = unperturbed, 1 = perturbed, 2 = pad}, and ``np.sign(0) == 0`` — so a
        perturbation whose target is the FIRST gene of the panel was flagged
        "unperturbed" at predict, and the model was handed a control cell with no
        perturbation at all. It would then faithfully predict the control profile,
        for one perturbation, silently. Training already used a flat ``1``
        (`create_cell_graph`); this makes predict agree.
        """
        pert_feats = np.zeros(len(X))
        n_genes = len(pert_feats)
        for p in pert_idx:
            idx = int(np.abs(p))
            if 0 <= idx < n_genes:
                pert_feats[idx] = 1
        feature_mat = torch.Tensor(np.vstack([X, pert_feats])).T
        return Data(x=feature_mat, pert=pert_gene)

    def create_cell_graph_dataset_for_prediction(self, pert_gene, ctrl_adata,
                                                 gene_names, device, num_samples=300):
        """VENDORED (V3): draws from ``self.rng`` so predict is reproducible."""
        pert_idx = [np.where(p == np.array(gene_names))[0][0] for p in pert_gene]
        picks = self.rng.integers(0, len(ctrl_adata), num_samples)
        Xs = _dense_matrix(ctrl_adata[picks, :].X)
        return [self.create_cell_graph_for_prediction(X, pert_idx, pert_gene).to(device)
                for X in Xs]


def _dense_matrix(X) -> np.ndarray:
    """Dense ``(n, n_genes)`` view of a sparse or dense expression block (V21)."""
    return np.asarray(X.toarray() if hasattr(X, "toarray") else X, dtype=np.float32)


class SCGPTWrapper:
    """scGPT fine-tuning wrapper: train and predict against docker/CONTRACT.md."""

    def __init__(self, config: Dict):
        self.config = config
        self.model = None
        self.pert_data = None
        self.gene_ids = None
        self.n_genes = None
        self.vocab = None
        self.best_model = None
        self.data_manager = DataManager(self.config)
        self.hyperparams = config['hyperparameters']

        # VENDORED (V3): the contract emits `seed` TOP-LEVEL, in both modes. The
        # atheus source read no seed at all — no torch/np/random seeding anywhere,
        # and the control-donor and inference-pool draws used the *global*
        # np.random, so two predicts on one checkpoint returned different numbers
        # and any seed-stability check measured RNG noise.
        self.seed = int(self.config['seed'])
        self._set_seeds()

        # VENDORED (V8): our contract sends `covariate_key`; the atheus source read
        # `covariate_field`, which F4 deleted host-side. Nothing populated it, so
        # the branch always fell through to `obs['cell_type'] = "NOTHING"` and the
        # model was cell-blind by accident. The `covariate_field` reader is deleted
        # rather than kept as a fallback (PRESAGE V13's reasoning: a stray
        # `extra_config: covariate_field:` must not be able to override the
        # canonical key on the one model whose claim rests on it).
        self.covariate_key = self.config.get('covariate_key') or 'cell_type'
        self.batch_key = self.config.get('batch_key') or None
        self.split_name = self.config.get('split_name')

        # Resolved at data-conversion time; recorded in the training report.
        self._resolved_cov_col = None
        self._resolved_batch_col = None
        self._report: Dict = {}

    def _set_seeds(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        log.info("Seeded python/numpy/torch with seed=%d (V3)", self.seed)

    # ------------------------------------------------------------------ train

    def train(self):
        """Fine-tune scGPT from the baked foundation checkpoint."""
        log.info("Starting scGPT training process (seed=%d)...", self.seed)
        t0 = time.time()

        log.info("Loading dataset ...")
        adata = self.data_manager.load_dataset()
        log.info("Loaded data with shape: %s", adata.shape)

        log.info("Converting to scGPT format ...")
        self.pert_data = self._convert_to_scgpt_format(adata)

        log.info("Recording split provenance ...")
        self._prepare_scgpt_splits()

        log.info("Creating data loaders ...")
        self.pert_data.get_dataloader(
            batch_size=self.hyperparams['batch_size'],
            test_batch_size=self.hyperparams['batch_size'],
        )

        log.info("Initializing scGPT model ...")
        self._initialize_scgpt_model(mode='train')

        log.info("Starting model training ...")
        self._train_scgpt()

        output_dir = Path(self.config['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        log.info("Saving model to %s", output_dir)
        self._save_model(output_dir)
        self._save_metadata(output_dir)
        self._report['wall_seconds'] = round(time.time() - t0, 1)
        self._write_training_report(output_dir)
        log.info("Training completed successfully")

    # ---------------------------------------------------------------- predict

    def predict(self):
        """Generate predictions using the fine-tuned checkpoint."""
        log.info("Starting scGPT prediction process (seed=%d)...", self.seed)

        model_path = Path(self.config['model_path'])
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}")
        log.info("Loading model from %s", model_path)

        # Pin the covariate/batch policy to whatever TRAINING resolved, before
        # anything reads it. Re-deriving it here could silently disagree with the
        # policy the checkpoint was fine-tuned under.
        self._load_training_report()

        self.pert_data = self._convert_to_scgpt_format_for_prediction()
        self._initialize_scgpt_model(mode='predict')

        test_conditions = self.config['test_conditions']
        log.info("Generating predictions for %d conditions", len(test_conditions))
        predictions_adata = self._generate_predictions(test_conditions)

        output_path = self.config['output_path']
        log.info("Saving predictions to %s", output_path)
        predictions_adata.write_h5ad(output_path)
        log.info("Prediction completed successfully")

    # ------------------------------------------------------------ OOV masking

    def _scgpt_gene_info_path(self) -> Path:
        """Location of the scGPT reference gene list, used for OOV masking.

        VENDORED (V8): the atheus source probed three paths (the first being the
        atheus Docker layout ``/app/data/ref/...``) and returned ``None`` on miss —
        whereupon ``_ensure_symbol_scgpt`` SKIPPED OOV masking entirely, every gene
        kept its name, and the ones absent from the vocabulary silently became
        ``<pad>`` tokens. This now comes from ``extra_config`` and raises, naming
        the paths tried (the V1 class).
        """
        configured = self.config.get('scgpt_gene_info_path')
        candidates = [Path(configured)] if configured else []
        candidates.append(Path(__file__).resolve().parent / "gene_info_scgpt.csv")
        for p in candidates:
            if p.exists():
                return p
        raise FileNotFoundError(
            "scGPT reference gene list not found. Tried: "
            + ", ".join(str(p) for p in candidates)
            + ". Set extra_config.scgpt_gene_info_path in docker/scgpt/model.yaml."
        )

    def _ensure_symbol_scgpt(self, adata: AnnData) -> None:
        """Fill ``var['symbol_scgpt']`` and mask genes outside the scGPT vocabulary.

        VENDORED (V9): the mask is unchanged, but the drop is now COUNTED and
        recorded. Downstream this drops genes from the panel and then drops any
        perturbation whose targets did not survive — both silent and unbounded in
        the atheus source. ``SCGPTContainer.has_drop_rule = True`` plus the host
        preflight coverage floor are the other half of this fix.
        """
        if getattr(self, "_oov_masked", False):
            return   # idempotent: this is called again via _prep_adata_for_scgpt,
                     # by which point the panel is already subset and re-running
                     # would overwrite the report with post-drop counts.
        if "symbol_scgpt" not in adata.var.columns:
            if "gene_name" in adata.var.columns:
                adata.var["symbol_scgpt"] = adata.var["gene_name"].astype(str)
                log.info("symbol_scgpt missing: filled from var['gene_name']")
            else:
                adata.var["symbol_scgpt"] = pd.Index(adata.var_names).astype(str)
                log.info("symbol_scgpt missing: filled from var_names")

        gene_info_path = self._scgpt_gene_info_path()
        gene_info = pd.read_csv(gene_info_path)
        valid = set(gene_info["feature_name"].astype(str))
        sym = adata.var["symbol_scgpt"].astype(str)
        oov = ~sym.isin(valid) & sym.notna()
        n_oov = int(oov.sum())
        if n_oov:
            adata.var.loc[oov, "symbol_scgpt"] = np.nan
        self._report['n_genes_in'] = int(adata.n_vars)
        self._report['n_genes_oov_dropped'] = n_oov
        self._oov_masked = True
        log.info(
            "OOV mask: %d/%d genes are not in the scGPT vocabulary and will be "
            "dropped (reference: %s)", n_oov, adata.n_vars, gene_info_path,
        )

    # -------------------------------------------------------- data conversion

    def _cache_key(self, adata_scgpt) -> str:
        """Fingerprint of everything the per-fold graph cache depends on (V15)."""
        import hashlib
        split_counts = (adata_scgpt.obs[self.split_name].astype(str).value_counts()
                        .sort_index().to_dict()) if self.split_name in adata_scgpt.obs else {}
        payload = json.dumps({
            "genes": list(map(str, adata_scgpt.var_names)),
            "conditions": sorted(map(str, adata_scgpt.obs['condition'].unique())),
            "split_name": self.split_name,
            "split_counts": {str(k): int(v) for k, v in split_counts.items()},
            "covariate_key": self._resolved_cov_col,
            "batch_key": self._resolved_batch_col,
            "control_labels": list(CONTROL_LABELS),
            "ctrl_samples_per_cell": int(self.hyperparams.get('ctrl_samples_per_cell', 1)),
            "seed": self.seed,
        }, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def _resolve_covariate_columns(self, adata) -> None:
        """Decide which obs columns V7 matches controls on, and record the decision.

        Auto-disables (rather than failing) when a column is absent or has <2
        levels — the PRESAGE V13 pattern. Every outcome is recorded in the
        training report, never left to a log line.
        """
        cov = self.covariate_key
        if cov in adata.obs.columns and adata.obs[cov].astype(str).nunique() >= 2:
            self._resolved_cov_col = cov
            log.info("covariate matching ON: %r (%d levels)",
                     cov, adata.obs[cov].astype(str).nunique())
        else:
            self._resolved_cov_col = None
            log.info("covariate matching OFF: %r absent or single-level "
                     "(single-cell-line dataset)", cov)

        bkey = self.batch_key
        if (bkey and bkey in adata.obs.columns
                and adata.obs[bkey].astype(str).nunique() >= 2):
            self._resolved_batch_col = bkey
            log.info("batch matching ON: %r (%d levels)",
                     bkey, adata.obs[bkey].astype(str).nunique())
        else:
            self._resolved_batch_col = None
            log.info("batch matching OFF: %r absent or single-level", bkey)

        self._report['covariate_col'] = self._resolved_cov_col
        self._report['batch_col'] = self._resolved_batch_col

    def _convert_to_scgpt_format(self, adata: AnnData) -> SCGPTPertData:
        """Convert the combined h5ad to scGPT PertData format, leak-safely."""
        log.info("Converting data to scGPT format ...")
        adata_scgpt = adata.copy()

        # ================= VENDORED (V6): the per-cell split filter =================
        # THE leak fix. The atheus source built its split from the marginal
        # train/val/test CONDITION lists (`convert_conditions`), and `split_name`
        # appeared nowhere on the training path. On cell-axis regimes
        # train_conditions is a superset of test_conditions — a perturbation held
        # out in cell type A legitimately remains `train` in cell type B — so
        # filtering by condition list pulls held-out CELLS into training. Nothing
        # raises; the score simply inflates.
        # This is the patch the atheus GEARS wrapper already had
        # (gears_wrapper.py:908-932) and scGPT never got.
        if not self.split_name:
            raise KeyError("config is missing `split_name` — refusing to train "
                           "without the per-cell split column (V6)")
        if self.split_name not in adata_scgpt.obs.columns:
            raise KeyError(
                f"per-cell split column {self.split_name!r} not in obs "
                f"(have: {sorted(adata_scgpt.obs.columns)[:20]} ...). "
                f"The combined h5ad must carry it — see docker/CONTRACT.md.")
        labels = adata_scgpt.obs[self.split_name].astype(str)
        n_before = adata_scgpt.n_obs
        counts_before = labels.value_counts().to_dict()
        adata_scgpt = adata_scgpt[labels.isin(['train', 'val'])].copy()
        log.info("V6 per-cell split filter: %d -> %d cells (dropped %s)",
                 n_before, adata_scgpt.n_obs,
                 {k: int(v) for k, v in counts_before.items()
                  if k not in ('train', 'val')})
        self._report['n_cells_total'] = int(n_before)
        self._report['n_cells_trainval'] = int(adata_scgpt.n_obs)
        self._report['split_counts_in'] = {str(k): int(v) for k, v in counts_before.items()}
        # Structural post-condition: no test-labelled cell may survive.
        self._report['n_test_cells_in_training'] = int(
            (adata_scgpt.obs[self.split_name].astype(str) == 'test').sum())
        if self._report['n_test_cells_in_training']:
            raise RuntimeError("V6 filter failed: test cells survived into training")
        # ===========================================================================

        self._resolve_covariate_columns(adata_scgpt)
        self._ensure_symbol_scgpt(adata_scgpt)

        # Drop OOV genes, then any perturbation whose targets did not survive.
        # Combos ("A+B") need each part checked individually — comparing the whole
        # string against gene names dropped every combo.
        adata_scgpt = adata_scgpt[:, adata_scgpt.var.symbol_scgpt.notna().values]
        known_genes = set(adata_scgpt.var.symbol_scgpt.values)
        all_perts = list(adata_scgpt.obs['condition'].unique())
        perts_to_keep, perts_dropped = [], []
        for p in all_perts:
            if _is_control_label(p) or 'ctrl' in str(p):
                perts_to_keep.append(p)
                continue
            parts = [g for g in str(p).split('+') if g]
            if parts and all(g in known_genes for g in parts):
                perts_to_keep.append(p)
            else:
                perts_dropped.append(str(p))
        adata_scgpt = adata_scgpt[adata_scgpt.obs['condition'].isin(perts_to_keep)]
        # VENDORED (V9): counted, not silent.
        self._report['n_genes_kept'] = int(adata_scgpt.n_vars)
        self._report['dropped_perturbations'] = sorted(perts_dropped)
        if perts_dropped:
            log.warning("V9 drop rule: %d perturbation(s) dropped for OOV targets: %s",
                        len(perts_dropped), sorted(perts_dropped)[:10])

        # Remove rows whose condition contains 'ctrl' but is not a control label.
        cond_str = adata_scgpt.obs['condition'].astype(str)
        keep_mask = ~cond_str.str.contains('ctrl') | cond_str.map(_is_control_label)
        # Copy once here: the three subsets above leave a view, and
        # _prep_adata_for_scgpt mutates .var/.obs, which would make AnnData
        # materialise it implicitly (and repeatedly) instead.
        adata_scgpt = adata_scgpt[keep_mask.values].copy()

        adata_scgpt = self._prep_adata_for_scgpt(adata_scgpt)

        # scGPT/GEARS PertData requires a `cell_type` obs column. Give it the real
        # covariate when we have one (V8) — it is what V7 matches controls on.
        if self._resolved_cov_col:
            adata_scgpt.obs['cell_type'] = (
                adata_scgpt.obs[self._resolved_cov_col].astype(str).values)
        else:
            adata_scgpt.obs['cell_type'] = "single"

        output_dir = Path(self.config['output_dir'])
        processed_data_dir = output_dir / 'processed_data'
        processed_data_dir.mkdir(parents=True, exist_ok=True)
        processed_dataset_path = processed_data_dir / 'cellsimbench_scgpt'
        cell_graphs_file = processed_dataset_path / 'data_pyg' / 'cell_graphs.pkl'
        cache_key_file = processed_data_dir / 'cache_key.json'

        pert_data = SCGPTPertData(str(processed_data_dir))
        pert_data.configure_pairing(
            split_col=self.split_name,
            cov_col='cell_type' if self._resolved_cov_col else None,
            batch_col=self._resolved_batch_col,
            seed=self.seed,
            num_samples_per_cell=self.hyperparams.get('ctrl_samples_per_cell', 1),
            num_de_genes=self.hyperparams.get('num_de_genes', 20),
        )

        # ============ VENDORED (V15): refuse a stale cache, never warn-and-proceed ============
        # The atheus source WARNED and proceeded on an existing perturb_processed.h5ad
        # (`:824-825`) and branched purely on cell_graphs.pkl existence (`:847`), so a
        # re-used fold directory silently trained on the PREVIOUS fold's graphs and
        # split. The host fingerprint hashes the host's split, so it would have
        # certified that run as valid.
        key = self._cache_key(adata_scgpt)
        self._report['cache_key'] = key
        reuse = False
        if cell_graphs_file.exists():
            recorded = None
            if cache_key_file.exists():
                try:
                    recorded = json.loads(cache_key_file.read_text()).get('cache_key')
                except (OSError, ValueError):
                    recorded = None
            if recorded == key:
                reuse = True
                log.info("Reusing processed dataset (cache_key %s matches)", key)
            else:
                raise RuntimeError(
                    f"{cell_graphs_file} exists but its cache_key ({recorded!r}) does "
                    f"not match this run ({key!r}). That cache was built from a "
                    f"different gene axis / split / covariate policy — training on it "
                    f"would silently score this fold with another fold's data (V15). "
                    f"Delete {processed_data_dir} and re-run.")
        # =====================================================================================

        if reuse:
            pert_data.load(data_path=str(processed_dataset_path))
        else:
            log.info("Creating new processed scGPT dataset ...")
            pert_data.new_data_process(dataset_name='cellsimbench_scgpt', adata=adata_scgpt)
            cache_key_file.write_text(json.dumps({'cache_key': key}, indent=2))

        if getattr(pert_data, 'dataset_processed', None) is None:
            raise RuntimeError(
                "pert_data.dataset_processed is None after new_data_process — "
                "SCGPTPertData.create_dataset_file should have set it.")

        self._report['ctrl_pairing_tiers'] = dict(pert_data.pairing_tier_counts)
        log.info("V7 control-pairing tiers used: %s", pert_data.pairing_tier_counts)
        return pert_data

    def _convert_to_scgpt_format_for_prediction(self) -> SCGPTPertData:
        """Load the processed dataset written by train — WITHOUT the graph pickle."""
        model_dir = Path(self.config['model_path'])
        processed_data_dir = model_dir / 'processed_data'
        processed_dataset_path = processed_data_dir / 'cellsimbench_scgpt'
        split_dict_file = processed_data_dir / 'cellsimbench_split_dict.pkl'

        if not (processed_dataset_path / 'perturb_processed.h5ad').exists():
            raise FileNotFoundError(
                f"Processed scGPT data not found at '{processed_dataset_path}'. "
                "Run training first.")
        log.info("Loading processed scGPT data from: %s", processed_dataset_path)

        pert_data = SCGPTPertData(str(processed_data_dir))
        pert_data.rng = np.random.default_rng(self.seed)   # V3
        pert_data.load_adata_only(str(processed_dataset_path))   # V11

        if split_dict_file.exists():
            pert_data.prepare_split(split='custom', seed=self.seed,
                                    split_dict_path=str(split_dict_file))
        else:
            raise FileNotFoundError(
                f"Split provenance not found at '{split_dict_file}'. Run training first.")
        return pert_data

    def _prep_adata_for_scgpt(self, adata: AnnData) -> AnnData:
        """Rename conditions into scGPT/GEARS form and assert the input shape."""
        log.info("Preprocessing AnnData for scGPT ...")
        self._ensure_symbol_scgpt(adata)
        adata.var["gene_name"] = adata.var.symbol_scgpt

        def fix_condition(condition):
            if _is_control_label(condition):
                return SCGPT_CTRL
            if "ctrl" not in str(condition) and "+" not in str(condition):
                return "ctrl+" + str(condition)
            return condition

        adata.obs["condition"] = adata.obs["condition"].astype(str).apply(fix_condition)

        assert "condition" in adata.obs.columns, "Condition column is missing"
        assert SCGPT_CTRL in adata.obs.condition.values, \
            "Control condition is missing or is not named 'ctrl'"
        assert any("+" in c for c in adata.obs.condition.values), \
            "Perturbations are missing or are not delimited with '+'"

        if self.hyperparams['dolog1p']:
            log.info("Applying log1p transformation ...")
            sc.pp.log1p(adata)

        if 'hvg' not in adata.uns.keys():
            adata.uns['hvg'] = {'indices': np.arange(adata.n_vars)}

        assert "gene_name" in adata.var.columns, "gene_name column is missing"
        log.info("scGPT preprocessing completed. Final shape: %s", adata.shape)
        return adata

    def _prepare_scgpt_splits(self):
        """Record which conditions each split contained — PROVENANCE ONLY.

        VENDORED (V6): in the atheus source this function WAS the split. It now
        only writes the record; ``SCGPTPertData.get_dataloader`` buckets graphs by
        the per-cell tag. The pickle is still written (and still loaded at predict)
        so the run dir documents the split training actually used.
        """
        obs = self.pert_data.adata.obs
        split_labels = obs[self.split_name].astype(str)
        # train/val come from the per-cell labels of the cells that actually
        # trained. `test` cannot: V6 removed those cells before PertData ever saw
        # them, so deriving it here would record an empty list and misrepresent the
        # fold. It comes from the config, which is what predict will be asked for.
        split_dict = {
            name: sorted(obs.loc[split_labels == name, 'condition'].astype(str).unique())
            for name in ('train', 'val')
        }
        split_dict['test'] = sorted(map(str, self.config.get('test_conditions', [])))
        log.info("Split provenance: train=%d val=%d conditions (per-cell derived), "
                 "test=%d (from config — those cells were filtered out by V6)",
                 len(split_dict['train']), len(split_dict['val']), len(split_dict['test']))

        processed_data_dir = Path(self.config['output_dir']) / 'processed_data'
        processed_data_dir.mkdir(parents=True, exist_ok=True)
        split_path = processed_data_dir / 'cellsimbench_split_dict.pkl'
        with open(split_path, 'wb') as f:
            pickle.dump(split_dict, f)
        log.info("Saved split provenance to %s", split_path)
        self.pert_data.prepare_split(split='custom', seed=self.seed,
                                     split_dict_path=str(split_path))

    # ------------------------------------------------------------ model setup

    def _weights_dir(self, mode: str) -> Path:
        """Where to load weights from, per mode.

        ================== VENDORED (V4): the two-mount rewrite ==================
        HIGHEST SEVERITY. The atheus source read
        ``hyperparameters['model_loc'] = "/pretrained_model/"`` in BOTH modes and
        relied on the harness re-binding that one path: to the foundation
        checkpoint at train, and to the trained run dir at predict. Our contract
        binds ONE directory at /model_output for both calls (docker/CONTRACT.md),
        so a verbatim port would either crash on a missing /pretrained_model/ or —
        far worse — resolve predict to the FOUNDATION weights and score an
        un-fine-tuned model, producing finite, sane-ranged, entirely meaningless
        predictions. `model_loc`/`model_loc_local` are deleted from the recipe.
        =========================================================================
        """
        if mode == 'train':
            d = Path(self.config.get('scgpt_pretrained_path')
                     or os.environ.get('SCGPT_PRETRAINED')
                     or '/opt/scgpt_pretrained')
            what = "foundation checkpoint"
        elif mode == 'predict':
            d = Path(self.config['model_path'])
            what = "fine-tuned checkpoint"
        else:
            raise ValueError(f"_weights_dir: unknown mode {mode!r}")
        if not d.exists():
            raise FileNotFoundError(
                f"{what} directory not found at {d}. For train this comes from "
                f"extra_config.scgpt_pretrained_path (baked at /opt/scgpt_pretrained "
                f"by scgpt.def); for predict it is config['model_path'].")
        return d

    def _resolve_architecture(self, model_configs: Dict) -> Dict:
        """Reconcile the checkpoint's architecture with the recipe's.

        VENDORED (V16): the atheus source used
        ``model_configs.get("embsize", hyperparams['embsize'])`` at train — so the
        checkpoint always won and the recipe's architecture values were INERT —
        while ``_save_model`` wrote the RECIPE's values into the run-dir args.json,
        making them LIVE at predict. They agree today only by luck. A recipe edit
        to ``nlayers`` would change nothing during training and then silently build
        a different model at predict, where ``strict=False`` drops the mismatched
        half. Disagreement is now an error.
        """
        keys = ('embsize', 'nheads', 'd_hid', 'nlayers', 'n_layers_cls')
        resolved, mismatches = {}, []
        for k in keys:
            ckpt_v = model_configs.get(k)
            recipe_v = self.hyperparams[k]
            if ckpt_v is not None and int(ckpt_v) != int(recipe_v):
                mismatches.append(f"{k}: checkpoint={ckpt_v} recipe={recipe_v}")
            resolved[k] = int(ckpt_v) if ckpt_v is not None else int(recipe_v)
        if mismatches:
            raise RuntimeError(
                "architecture disagreement between the checkpoint and model.yaml: "
                + "; ".join(mismatches)
                + ". These must match — load_pretrained(strict=False) would silently "
                  "drop every mismatched tensor and train a partly random model (V5).")
        return resolved

    def _initialize_scgpt_model(self, mode: str):
        """Build the model and load weights for ``mode`` ('train' | 'predict')."""
        log.info("Initializing scGPT model (mode=%s) ...", mode)
        model_dir = self._weights_dir(mode)
        model_config_file = model_dir / "args.json"
        model_file = model_dir / "best_model.pt"
        vocab_file = model_dir / "vocab.json"
        for f in (model_config_file, model_file, vocab_file):
            if not f.exists():
                raise FileNotFoundError(f"required checkpoint file missing: {f}")

        self.vocab = GeneVocab.from_file(vocab_file)
        for s in self.hyperparams['special_tokens']:
            if s not in self.vocab:
                self.vocab.append_token(s)

        self.pert_data.adata.var["id_in_vocab"] = [
            1 if gene in self.vocab else -1
            for gene in self.pert_data.adata.var["gene_name"]
        ]
        gene_ids_in_vocab = np.array(self.pert_data.adata.var["id_in_vocab"])
        log.info("Matched %d/%d genes in vocabulary of size %d",
                 int(np.sum(gene_ids_in_vocab >= 0)), len(gene_ids_in_vocab),
                 len(self.vocab))

        genes = self.pert_data.adata.var["gene_name"].tolist()
        self.vocab.set_default_index(self.vocab["<pad>"])
        self.gene_ids = np.array(
            [self.vocab[g] if g in self.vocab else self.vocab["<pad>"] for g in genes],
            dtype=int)
        self.n_genes = len(genes)

        with open(model_config_file, "r") as f:
            model_configs = json.load(f)
        arch = self._resolve_architecture(model_configs)
        self._resolved_arch = arch
        log.info("Resolved architecture: %s", arch)

        self.model = TransformerGeneratorWithLayerNorm(
            len(self.vocab),
            arch['embsize'],
            arch['nheads'],
            arch['d_hid'],
            arch['nlayers'],
            nlayers_cls=arch['n_layers_cls'],
            n_cls=self.hyperparams.get('n_cls', 1),
            vocab=self.vocab,
            dropout=self.hyperparams['dropout'],
            pad_token=self.hyperparams['pad_token'],
            pad_value=self.hyperparams['pad_value'],
            pert_pad_id=self.hyperparams['pert_pad_id'],
            do_mvc=self.hyperparams['MVC'],
            cell_emb_style=self.hyperparams['cell_emb_style'],
            mvc_decoder_style=self.hyperparams['mvc_decoder_style'],
            use_fast_transformer=self.hyperparams['use_fast_transformer'],
        )
        self._record_fast_path_state()

        log.info("Loading weights from %s", model_file)
        pretrained_dict = torch.load(model_file, map_location='cpu')
        self._load_pretrained_checked(pretrained_dict, mode)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)
        log.info("Model initialized on device: %s", device)

    def _record_fast_path_state(self):
        """Record whether the flash fast path is ACTUALLY active.

        VENDORED (V20): scGPT flips ``use_fast_transformer`` to False with a mere
        *warning* when flash-attn will not import, and FA2's MHA falls back to
        dense attention when it receives float32. All of those produce
        correct-looking numbers on a slower path while model.yaml still says
        ``use_fast_transformer: true``. The recipe's claim is therefore checked
        against the built model, not trusted.
        """
        requested = bool(self.hyperparams['use_fast_transformer'])
        effective = bool(getattr(self.model, 'use_fast_transformer', False))
        try:
            from scgpt.model.flash_attn_compat import flash_attn_backend
        except Exception:
            flash_attn_backend = None
        n_flash = sum(1 for m in self.model.modules()
                      if type(m).__name__ in ('FlashMHA', 'MHA', 'FlashTransformerEncoderLayer'))
        self._report.update({
            'use_fast_transformer_requested': requested,
            'use_fast_transformer_effective': effective,
            'flash_attn_backend': flash_attn_backend,
            'n_flash_modules': int(n_flash),
        })
        log.info("fast path: requested=%s effective=%s backend=%s flash_modules=%d",
                 requested, effective, flash_attn_backend, n_flash)
        if requested and not effective:
            raise RuntimeError(
                "model.yaml requests use_fast_transformer: true but scGPT fell back "
                "to vanilla attention (flash-attn did not import). That is a slower, "
                "different compute path than the recipe records. Either fix the image "
                "or set use_fast_transformer: false and record it as a deviation in "
                "SCGPT_NOTES.md (V20).")

    def _load_pretrained_checked(self, pretrained_dict, mode: str):
        """``load_pretrained`` + assert that the transformer trunk actually loaded.

        VENDORED (V5): ``load_pretrained(..., strict=False)`` drops every key or
        shape mismatch WITHOUT a word, so a disagreement yields a randomly
        initialised trunk that fine-tunes, converges, and scores clean.
        (The one obvious trigger — flash ``Wqkv.*`` vs vanilla ``in_proj_*`` — is
        handled upstream by a rename keyed on ``model.use_fast_transformer``; this
        guard is for the ones that are not: a changed vocabulary size, an
        architecture disagreement, and the FA1/FA2 ``self_attn._impl.Wqkv``
        layout.)
        """
        before = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
        self.model = load_pretrained(model=self.model,
                                     pretrained_params=pretrained_dict,
                                     strict=False, verbose=False)
        after = self.model.state_dict()
        changed = [k for k in after
                   if k in before and after[k].shape == before[k].shape
                   and not torch.equal(after[k], before[k])]
        n_layer_keys = sum(1 for k in changed
                           if k.startswith('transformer_encoder.layers.'))
        frac = len(changed) / max(len(after), 1)
        self._report.update({
            f'{mode}_pretrained_keys_loaded': len(changed),
            f'{mode}_pretrained_keys_total': len(after),
            f'{mode}_pretrained_transformer_keys_loaded': n_layer_keys,
        })
        log.info("load_pretrained[%s]: %d/%d tensors changed (%.1f%%), "
                 "%d in transformer_encoder.layers",
                 mode, len(changed), len(after), 100 * frac, n_layer_keys)
        nlayers = self._resolved_arch['nlayers']
        if n_layer_keys < nlayers:
            raise RuntimeError(
                f"load_pretrained[{mode}] populated only {n_layer_keys} tensors in "
                f"transformer_encoder.layers for a {nlayers}-layer model. The "
                f"checkpoint and the built architecture disagree, and strict=False "
                f"hid it — this would train/score a partly random trunk (V5).")
        if frac < 0.5:
            raise RuntimeError(
                f"load_pretrained[{mode}] populated only {100*frac:.1f}% of the "
                f"model's tensors — well below the 50% floor. Refusing to continue (V5).")

    # --------------------------------------------------------------- training

    def _init_wandb(self):
        """Start W&B if the host asked for it, defaulting SAFELY to offline.

        VENDORED (V14): the atheus source contained ZERO references to wandb or
        tensorboard, while model.yaml declares ``wandb_project`` and
        ``ContainerPredictor._train`` injects ``wandb``/``wandb_project``/
        ``wandb_run`` into config.json. A declared setting that does nothing is
        worse than no setting: it makes Gate A's loss curves impossible while
        looking configured.

        Offline is the DEFAULT, not a fallback. The self-hosted endpoint
        (ctr-biomed-15:8082) is down, and an online logger with no reachable
        endpoint blocks on login in a tty-less batch job and takes the whole
        multi-hour run with it. The online path is gated on an EXPLICIT
        ``WANDB_MODE``: keying it on ``WANDB_API_KEY`` is what silently sent a
        PRESAGE run online against a dead host, because sbatch inherits the
        submitting shell's environment.
        """
        if not self.config.get('wandb'):
            self._wandb = None
            return
        if not os.environ.get('WANDB_MODE'):
            os.environ['WANDB_MODE'] = 'offline'
            log.warning("WANDB_MODE unset -> forcing offline. Runs land in "
                        "<run_dir>/wandb/offline-run-* and replay with `wandb sync`.")
        try:
            import wandb
            self._wandb = wandb.init(
                project=self.config.get('wandb_project', 'vcr-scgpt-ct'),
                name=self.config.get('wandb_run', 'scgpt_training'),
                dir=str(self.config['output_dir']),
                config={'hyperparameters': self.hyperparams,
                        'seed': self.seed,
                        'split_name': self.split_name,
                        'covariate_col': self._resolved_cov_col,
                        'batch_col': self._resolved_batch_col},
                reinit=True,
            )
            log.info("W&B initialised: project=%s run=%s mode=%s",
                     self.config.get('wandb_project'), self.config.get('wandb_run'),
                     os.environ.get('WANDB_MODE'))
        except Exception as e:   # never let logging kill a training run
            log.warning("W&B init failed (%s) — continuing without it", e)
            self._wandb = None

    def _wandb_log(self, payload: Dict):
        if getattr(self, '_wandb', None) is not None:
            try:
                self._wandb.log(payload)
            except Exception as e:
                log.warning("W&B log failed (%s)", e)

    def _train_scgpt(self):
        """Fine-tune, selecting on validation loss."""
        log.info("Starting scGPT training loop ...")
        hp = self.hyperparams
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        criterion = masked_mse_loss
        # VENDORED (V12): optimizer, scheduler gamma and grad-clip norm were all
        # hard-coded in the wrapper body.
        if str(hp.get('optimizer', 'Adam')) != 'Adam':
            raise ValueError(f"unsupported optimizer {hp.get('optimizer')!r}; only Adam is wired")
        optimizer = torch.optim.Adam(self.model.parameters(), lr=hp['lr'],
                                     weight_decay=hp['weight_decay'])
        scheduler = StepLR(optimizer, hp['schedule_interval'],
                           gamma=hp.get('scheduler_gamma', 0.9))
        scaler = torch.cuda.amp.GradScaler(enabled=hp['amp'])

        train_loader = self.pert_data.dataloader["train_loader"]
        val_loader = self.pert_data.dataloader["val_loader"]

        # VENDORED (V11): record configured vs effective batching so a
        # zero-optimiser-step run is visible in the report rather than inferred.
        n_train = self.pert_data.split_sizes['train']
        steps_per_epoch = len(train_loader)
        self._report.update({
            'n_train_graphs': n_train,
            'n_val_graphs': self.pert_data.split_sizes['val'],
            'batch_size_configured': int(hp['batch_size']),
            'batch_size_effective': int(min(hp['batch_size'], n_train)),
            'n_train_steps_per_epoch': int(steps_per_epoch),
        })
        if steps_per_epoch == 0:
            raise RuntimeError(
                f"training loader yields ZERO batches for {n_train} graphs — the run "
                f"would exit rc 0 with an untrained checkpoint (V11).")
        log.info("train graphs=%d val graphs=%d steps/epoch=%d",
                 n_train, self._report['n_val_graphs'], steps_per_epoch)

        self._init_wandb()

        best_val_loss = float('inf')
        best_epoch = None
        self.best_model = None
        patience = int(hp.get('early_stopping_patience', 0) or 0)
        epochs_without_improvement = 0

        for epoch in range(1, hp['max_epochs'] + 1):
            epoch_start = time.time()
            train_loss = self._train_epoch(
                epoch=epoch, train_loader=train_loader, device=device,
                optimizer=optimizer, scheduler=scheduler, scaler=scaler,
                criterion=criterion, hyperparams=hp)
            val_loss = self._validate_epoch(val_loader, device, criterion, hp)
            elapsed = time.time() - epoch_start
            lr_now = scheduler.get_last_lr()[0]
            log.info("Epoch %3d | %5.1fs | train %.4f | val %.4f | lr %.3g",
                     epoch, elapsed, train_loss, val_loss, lr_now)
            self._wandb_log({'epoch': epoch, 'train_loss': train_loss,
                             'val_loss': val_loss, 'lr': lr_now,
                             'epoch_seconds': elapsed})

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                self.best_model = copy.deepcopy(self.model)
                epochs_without_improvement = 0
                log.info("  new best model (val %.4f)", val_loss)
            else:
                epochs_without_improvement += 1
                if patience and epochs_without_improvement >= patience:
                    log.info("Early stopping at epoch %d (no improvement for %d epochs)",
                             epoch, patience)
                    break

            scheduler.step()

        self._report.update({
            'best_epoch': best_epoch,
            'best_val_loss': None if best_epoch is None else float(best_val_loss),
            'max_epochs': int(hp['max_epochs']),
            'epochs_run': epoch,
        })
        if getattr(self, '_wandb', None) is not None:
            try:
                self._wandb.finish()
            except Exception:
                pass
        log.info("Training completed (best epoch %s, val %.4f)", best_epoch, best_val_loss)

    @staticmethod
    def _select_input_genes(n_genes, ori_gene_values, hyperparams, device, generator):
        """Choose the gene positions fed to the transformer this step.

        VENDORED (V23): the atheus train path did
        ``input_gene_ids = torch.randperm(len(input_gene_ids))[:max_seq_len]`` —
        which returns POSITIONS, not gene ids, and coincides with gene ids only
        because ``include_zero_gene == "all"`` makes the input ``arange``. Under
        ``"batch-wise"`` it silently indexed the wrong space. Validation
        meanwhile took a plain prefix ``[:max_seq_len]``, so train and val saw
        different gene subsets and ``best_val_loss`` selected on an arbitrary
        prefix. Both now use the same policy, indexed correctly, under an
        explicit generator.
        """
        if hyperparams['include_zero_gene'] == "all":
            input_gene_ids = torch.arange(n_genes, device=device, dtype=torch.long)
        else:
            input_gene_ids = ori_gene_values.nonzero()[:, 1].flatten().unique().sort()[0]
        max_len = hyperparams['max_seq_len']
        if len(input_gene_ids) > max_len:
            perm = torch.randperm(len(input_gene_ids), device=device, generator=generator)
            input_gene_ids = input_gene_ids[perm[:max_len]]
        return input_gene_ids

    def _forward_batch(self, batch_data, device, hyperparams, generator):
        """Shared train/val forward. Returns (output_values, target_values, mask)."""
        batch_size = len(batch_data.y)
        batch_data.to(device)
        x = batch_data.x
        ori_gene_values = x[:, 0].view(batch_size, self.n_genes)
        pert_flags = x[:, 1].long().view(batch_size, self.n_genes)
        target_gene_values = batch_data.y

        input_gene_ids = self._select_input_genes(
            self.n_genes, ori_gene_values, hyperparams, device, generator)
        input_values = ori_gene_values[:, input_gene_ids]
        input_pert_flags = pert_flags[:, input_gene_ids]
        target_values = target_gene_values[:, input_gene_ids]

        mapped_input_gene_ids = map_raw_id_to_vocab_id(input_gene_ids, self.gene_ids)
        mapped_input_gene_ids = mapped_input_gene_ids.repeat(batch_size, 1)
        src_key_padding_mask = torch.zeros_like(input_values, dtype=torch.bool,
                                                device=device)

        with torch.cuda.amp.autocast(enabled=hyperparams['amp']):
            output_dict = self.model(
                mapped_input_gene_ids,
                input_values,
                input_pert_flags,
                src_key_padding_mask=src_key_padding_mask,
                CLS=hyperparams['CLS'],
                CCE=hyperparams['CCE'],
                MVC=hyperparams['MVC'],
                ECS=hyperparams['ECS'],
            )
        return output_dict["mlm_output"], target_values, torch.ones_like(
            input_values, dtype=torch.bool)

    def _train_epoch(self, epoch, train_loader, device, optimizer, scheduler,
                     scaler, criterion, hyperparams):
        """Train one epoch; returns the mean loss over batches."""
        self.model.train()
        gen = torch.Generator(device=device)
        gen.manual_seed(self.seed * 1000 + epoch)     # V3/V23: deterministic per epoch
        total, n_batches = 0.0, 0
        num_batches = len(train_loader)
        pbar = tqdm(enumerate(train_loader), total=num_batches,
                    desc=f"Training epoch {epoch}", leave=False)

        for batch, batch_data in pbar:
            with torch.cuda.amp.autocast(enabled=hyperparams['amp']):
                output_values, target_values, masked_positions = self._forward_batch(
                    batch_data, device, hyperparams, gen)
                loss = criterion(output_values, target_values, masked_positions,
                                 reduction=hyperparams['loss_reduction'])  # V17

            self.model.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            with warnings.catch_warnings(record=True):
                warnings.filterwarnings("always")
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    hyperparams.get('grad_clip_norm', 1.0),   # V12
                    error_if_nonfinite=False if scaler.is_enabled() else True,
                )
            scaler.step(optimizer)
            scaler.update()

            total += float(loss.item())
            n_batches += 1
            if batch % 5 == 0:
                pbar.set_postfix({'loss': f'{total / max(n_batches, 1):.4f}'})

        return total / max(n_batches, 1)

    def _validate_epoch(self, val_loader, device, criterion, hyperparams):
        """Validate one epoch; returns the mean loss over batches."""
        self.model.eval()
        # V23: a FIXED generator, so the val gene subset is identical every epoch
        # and best_val_loss is comparable across epochs.
        gen = torch.Generator(device=device)
        gen.manual_seed(self.seed)
        total, n_batches = 0.0, 0
        with torch.no_grad():
            for batch_data in val_loader:
                with torch.cuda.amp.autocast(enabled=hyperparams['amp']):
                    output_values, target_values, masked_positions = self._forward_batch(
                        batch_data, device, hyperparams, gen)
                    loss = criterion(output_values, target_values, masked_positions,
                                     reduction=hyperparams['loss_reduction'])
                total += float(loss.item())
                n_batches += 1
        return total / n_batches if n_batches else float('inf')

    # ------------------------------------------------------------- prediction

    def _load_training_report(self) -> Dict:
        """Read the covariate policy training actually used, and pin predict to it."""
        path = Path(self.config['model_path']) / 'scgpt_training_report.json'
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found — cannot pin the predict-time covariate policy to "
                f"what training used. Retrain with this wrapper.")
        rep = json.loads(path.read_text())
        self._resolved_cov_col = rep.get('covariate_col')
        self._resolved_batch_col = rep.get('batch_col')
        log.info("Pinned from training: covariate_col=%r batch_col=%r",
                 self._resolved_cov_col, self._resolved_batch_col)
        return rep

    def _predict_control_pools(self) -> Dict[str, AnnData]:
        """Control cells to condition on, per target covariate.

        ================== VENDORED (V7): matched controls at inference ==================
        The control profile is the ONLY channel through which cell identity reaches
        an scGPT prediction (the model has no covariate port — see
        TransformerGeneratorWithLayerNorm). So predicting cell type T means feeding
        T's OWN control cells.

        These come from the ORIGINAL combined h5ad, not from the processed adata:
        under UnseenCell the entire held-out bin is labelled `test` and V6 removed
        it from training, so the processed adata contains no cells of the very cell
        type we are asked to predict. Reading the held-out covariate's CONTROL cells
        at inference is contract-sanctioned (docker/CONTRACT.md: predict "re-reads
        the full h5ad so it can enumerate the held-out covariate(s) and their
        control cells") and matches PRESAGE's V10 precedent: control-derived
        quantities from a held-out covariate are legitimate at inference and
        illegitimate at training. Controls are unperturbed, and the benchmark
        subtracts control itself.
        =================================================================================
        """
        orig = sc.read_h5ad(self.config['data_path'])
        orig.var.index.name = None
        genes = list(self.pert_data.adata.var_names)
        missing = [g for g in genes if g not in set(orig.var_names)]
        if missing:
            raise RuntimeError(
                f"{len(missing)} genes of the trained panel are absent from "
                f"{self.config['data_path']} (e.g. {missing[:5]}) — the h5ad changed "
                f"since training.")
        orig = orig[:, genes]

        is_ctrl = orig.obs['condition'].astype(str).map(_is_control_label).to_numpy()
        split = orig.obs[self.split_name].astype(str).to_numpy() if (
            self.split_name in orig.obs.columns) else np.array([''] * orig.n_obs)

        if self._resolved_cov_col and self._resolved_cov_col in orig.obs.columns:
            cov_all = orig.obs[self._resolved_cov_col].astype(str).to_numpy()
            targets = sorted(set(cov_all[split == 'test'])) or sorted(set(cov_all))
        else:
            cov_all = np.array([''] * orig.n_obs)
            targets = ['']

        pools, sizes = {}, {}
        for cov in targets:
            mask = is_ctrl & (cov_all == cov)
            if not mask.any():
                raise RuntimeError(
                    f"no control cells for target covariate {cov!r} in "
                    f"{self.config['data_path']}. scGPT conditions on the control "
                    f"profile, so it cannot predict a cell type whose controls are "
                    f"absent (V7).")
            pools[cov] = orig[mask].copy()
            sizes[cov] = int(mask.sum())
        log.info("V7 predict control pools: %s", sizes)
        self._report['predict_ctrl_pool_sizes'] = sizes
        self._report['predict_target_covariates'] = targets
        del orig
        return pools

    def _sample_control_pool(self, pool: AnnData, n: int) -> np.ndarray:
        """Draw ``n`` control profiles, stratified by batch when we have one.

        DEVIATION, recorded: the design called for predicting per (covariate,
        batch) and pooling the results. Batch-stratified SAMPLING from the
        covariate's pool is the same estimator — the mean over a stratified sample
        equals the stratum-size-weighted mean of per-stratum predictions — at
        1/n_batches the forward-pass cost (mcfaline23 has 48 batches per cell type).
        Recorded in SCGPT_NOTES.md rather than left implicit.
        """
        rng = self.pert_data.rng
        if (self._resolved_batch_col
                and self._resolved_batch_col in pool.obs.columns
                and pool.obs[self._resolved_batch_col].astype(str).nunique() > 1):
            b = pool.obs[self._resolved_batch_col].astype(str).to_numpy()
            groups = [np.flatnonzero(b == lv) for lv in sorted(set(b))]
            weights = np.array([len(g) for g in groups], dtype=float)
            weights /= weights.sum()
            per = np.maximum(1, np.round(weights * n).astype(int))
            picks = np.concatenate([g[rng.integers(0, len(g), k)]
                                    for g, k in zip(groups, per)])
            picks = picks[rng.permutation(len(picks))][:n]
        else:
            picks = rng.integers(0, pool.n_obs, n)
        return _dense_matrix(pool[picks, :].X)

    def _generate_predictions(self, test_conditions: List[str]) -> AnnData:
        """Predict every test condition under every target covariate."""
        log.info("Generating scGPT predictions ...")
        hp = self.hyperparams
        gene_list = self.pert_data.gene_names.values.tolist()
        gene_set = set(gene_list)

        prediction_conditions, dropped = [], []
        for cond in test_conditions:
            if _is_control_label(cond):
                continue
            clean = str(cond).replace('ctrl+', '')
            parts = [g for g in clean.split('+') if g]
            valid = [g for g in parts if g in gene_set]
            if len(valid) != len(parts) or not valid:
                dropped.append(str(cond))   # V9: counted, not just warned
                continue
            prediction_conditions.append((cond, valid))
        if dropped:
            log.warning("V9 drop rule: %d test condition(s) not predictable "
                        "(targets outside the scGPT panel): %s",
                        len(dropped), dropped[:10])
        self._report['dropped_test_conditions'] = sorted(dropped)
        if not prediction_conditions:
            raise ValueError("no predictable test conditions after the OOV drop rule")
        log.info("Predicting %d/%d test conditions",
                 len(prediction_conditions), len(test_conditions))

        pools = self._predict_control_pools()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.eval()
        self.model.to(device)

        # VENDORED (V22): predict runs the FULL untruncated gene sequence
        # (pred_perturb with include_zero_gene="all" does not apply max_seq_len),
        # so its memory profile is unrelated to training's. It gets its own batch
        # size rather than borrowing `batch_size` — lowering that to survive
        # predict would silently retune training too.
        predict_bs = int(hp.get('predict_batch_size', hp['batch_size']))
        results_pred: Dict = {}

        with torch.no_grad():
            for cov, pool in pools.items():
                log.info("  cov=%r: %d control cells available", cov, pool.n_obs)
                for cond, genes in tqdm(prediction_conditions,
                                        desc=f"Predicting [cov={cov or 'all'}]",
                                        leave=False):
                    Xs = self._sample_control_pool(pool, hp['pool_size'])
                    pert_idx = [np.where(g == np.array(gene_list))[0][0] for g in genes]
                    cell_graphs = [
                        self.pert_data.create_cell_graph_for_prediction(
                            X, pert_idx, genes).to(device) for X in Xs]
                    loader = DataLoader(cell_graphs, batch_size=predict_bs, shuffle=False)
                    running, n = None, 0
                    for batch_data in loader:
                        pred = self.model.pred_perturb(
                            batch_data,
                            include_zero_gene=hp['include_zero_gene'],
                            gene_ids=self.gene_ids,
                            amp=hp['amp'],
                            do_sample=hp.get('do_sample', True),
                        ).sum(dim=0)
                        # VENDORED (V19): reduce in the loop. The atheus source kept
                        # all `pool_size` per-cell predictions per (cov, cond) and only
                        # averaged at the very end.
                        running = pred if running is None else running + pred
                        n += batch_data.num_graphs
                    results_pred[(cov, str(cond))] = (running / max(n, 1)).cpu().numpy()

        return self._format_predictions_as_anndata(results_pred)

    def _format_predictions_as_anndata(self, predictions_dict: Dict) -> AnnData:
        """Build the contract's predictions.h5ad: absolute log1p, one row per (cov, cond)."""
        prediction_list, condition_list, covariate_list = [], [], []
        for (cov, cond), mean_profile in predictions_dict.items():
            prediction_list.append(mean_profile)
            condition_list.append(cond)
            covariate_list.append(cov if cov else "none")

        if not prediction_list:
            raise ValueError("No valid predictions found for test conditions")

        prediction_matrix = np.vstack(prediction_list).astype(np.float32)
        cov_aware = any(c not in ("", "none") for c in covariate_list)
        obs_idx = ([f"{c}__{cond}" for c, cond in zip(covariate_list, condition_list)]
                   if cov_aware else list(condition_list))
        obs_df = pd.DataFrame(
            {'covariate': covariate_list, 'condition': condition_list},
            index=pd.Index(obs_idx),
        )

        adata_pred = AnnData(X=prediction_matrix, obs=obs_df)
        adata_pred.var_names = self.pert_data.adata.var_names

        # VENDORED (V22): assert the shape the host contract depends on, here,
        # rather than discovering it in tensor_map after a multi-hour GPU run.
        if not adata_pred.var_names.is_unique:
            dupes = adata_pred.var_names[adata_pred.var_names.duplicated()].tolist()
            raise RuntimeError(
                f"predictions.h5ad has duplicate var_names (e.g. {dupes[:5]}); the "
                f"host gene intersection cannot key on a non-unique axis.")
        if not np.isfinite(prediction_matrix).all():
            raise RuntimeError("predictions.h5ad contains non-finite values")
        vmax = float(prediction_matrix.max())
        if vmax > 30.0:
            raise RuntimeError(
                f"predictions.h5ad max is {vmax:.1f} — that looks like raw counts, "
                f"not log1p. The contract requires absolute log1p expression.")

        log.info("scGPT output adata %s; cov-aware=%s, n_covariates=%d, max=%.2f",
                 adata_pred.shape, cov_aware, len(set(covariate_list)), vmax)
        return adata_pred

    # ---------------------------------------------------------------- artefacts

    def _save_model(self, output_dir: Path):
        """Persist the BEST-VALIDATION weights, or refuse.

        VENDORED (V10): the atheus source fell back to ``self.model`` (the last
        epoch) when ``best_model`` was None. That fallback is exactly what PRESAGE's
        V5 refusal caught: a run that took zero optimiser steps still writes a
        checkpoint whose predictions are correctly shaped, finite and log1p-ranged,
        and passes every downstream check. Refusing to save weights no validation
        ever selected is the only thing that catches it.
        """
        if self.best_model is None:
            raise RuntimeError(
                "no best-validation checkpoint was selected — refusing to save "
                "last-epoch weights silently (V10). Either validation never ran or "
                "every epoch produced a non-finite loss; both mean this run did not "
                "train.")
        torch.save(self.best_model.state_dict(), output_dir / 'best_model.pt')
        self.vocab.save_json(output_dir / 'vocab.json')

        # VENDORED (V16): write the RESOLVED architecture — what the model was
        # actually built with — not the recipe's copy of it.
        model_config = dict(self._resolved_arch)
        model_config['dropout'] = self.hyperparams['dropout']
        (output_dir / 'args.json').write_text(json.dumps(model_config, indent=2))
        log.info("Model saved to %s (best epoch %s)", output_dir,
                 self._report.get('best_epoch'))

    def _save_metadata(self, output_dir: Path):
        """Model-side provenance. The benchmark does not read this."""
        metadata = {
            'model_type': 'scGPT',
            'config': self.config,
            'data_shape': list(self.pert_data.adata.shape) if self.pert_data else None,
            'n_genes': self.n_genes,
            'vocab_size': len(self.vocab) if self.vocab else None,
        }
        with open(output_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2, cls=PathEncoder)

    def _write_training_report(self, output_dir: Path):
        """The report the host lifts into fingerprint.json via _train_report.

        Everything a reader auditing this run needs and the host-side fingerprint
        (a hash of the split) cannot show: which epoch was selected, which control
        pairing tiers fired, what the drop rule dropped, and whether the fast path
        was actually live.
        """
        report = dict(self._report)
        report['seed'] = self.seed
        report['split_name'] = self.split_name
        report['data_path'] = self.config['data_path']
        report['architecture'] = getattr(self, '_resolved_arch', None)
        path = output_dir / 'scgpt_training_report.json'
        path.write_text(json.dumps(report, indent=2, cls=PathEncoder, sort_keys=True))
        log.info("Wrote training report to %s", path)
