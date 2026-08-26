"""GeneTransformer — the architecture under study in the KxN depth sweep.

COPIED VERBATIM from the original depth-hypothesis implementation (only this
docstring and the dropped back-compat aliases differ). It is deliberately not
"improved": the sweep compares K layers applied N times against plain depth, and
any change to the forward pass would change what the comparison measures. Treat
edits here as changing the experiment, not refactoring code.

Vendored by copy, not import — this repo depends on nothing outside itself.

Note on cost: one token per gene, so self-attention is quadratic in panel size.
Fine for ecoli_synthetic (2853 genes); a real dataset like adamson16 (8250) is
~8.4x more attention work per layer and needs its own feasibility measurement.
"""

import torch
import torch.nn as nn


class GeneTransformer(nn.Module):
    """Transformer encoder for gene expression prediction.

    Parameters
    ----------
    d_model : int
        Token embedding dimension.
    nhead : int
        Number of attention heads.
    ff_mult : int
        Feed-forward multiplier (dim_feedforward = ff_mult * d_model).
    dropout : float
        Dropout rate.
    num_genes : int
        Number of genes (for the identity embedding table).
    num_steps : int
        N — number of recurrences (the encoder block is applied N times).
    layers : int
        K — number of transformer layers in the encoder block.
    """

    def __init__(self, d_model=256, nhead=8, ff_mult=4, dropout=0.1,
                 num_genes=100, num_steps=1, layers=1, **kwargs):
        super().__init__()
        self.num_steps = num_steps

        # --- Token embedding (additive fusion) ---
        self.gene_emb = nn.Embedding(num_genes, d_model)
        self.scalar_proj = nn.Sequential(
            nn.Linear(2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        # --- Transformer encoder (K layers, applied N times) ---
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff_mult * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)

        # --- Output head ---
        self.head = nn.Linear(d_model, 1)

    def forward(self, x_wt, p, gene_idx):
        # Additive fusion: gene identity + scalar features
        gene_emb = self.gene_emb(gene_idx)                  # (B, N, d_model)
        scalars = torch.stack([x_wt, p.float()], dim=-1)    # (B, N, 2)
        tok = gene_emb + self.scalar_proj(scalars)           # (B, N, d_model)

        # Apply encoder N times (weight-tied recurrence)
        H = tok
        for _ in range(self.num_steps):
            H = self.encoder(H)                              # (B, N, d_model)

        return self.head(H).squeeze(-1)                      # (B, N)


def count_parameters(model: nn.Module) -> int:
    """Trainable parameter count — logged so a variant's size is on record."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
