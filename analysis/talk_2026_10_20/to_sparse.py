"""Convert the dense Replogle 2022 K562 essential h5ad into a CSR count matrix plus obs/var arrays."""
import sys
from pathlib import Path

import h5py
import numpy as np
import scipy.sparse as sp

SRC = Path("/cluster/work/boeva/virtual_cell_reasoning/data/replogle22/replogle22_k562_downloaded.h5ad")
OUT = Path(sys.argv[1])
BLOCK = 1213 * 4


def obs_col(obs, name):
    """Return one obs column as a numpy array."""
    g = obs[name]
    if isinstance(g, h5py.Group):
        return g["categories"][:].astype(str)[g["codes"][:]]
    v = g[:]
    return v.astype(str) if v.dtype.kind in "SO" else v


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with h5py.File(SRC, "r") as f:
        X = f["X"]
        n = X.shape[0]
        blocks = []
        for start in range(0, n, BLOCK):
            blocks.append(sp.csr_matrix(X[start:start + BLOCK]))
            print(start, flush=True)
        M = sp.vstack(blocks, format="csr")
        obs = f["obs"]
        cols = {c: obs_col(obs, c) for c in ["gene", "guide_id", "batch", "UMI_count", "percent_mito", "ngenes"]}
        cols["cell_barcode"] = obs_col(obs, obs.attrs["_index"])
        var = f["var"]
        genes = var[var.attrs["_index"]][:].astype(str)
    sp.save_npz(OUT / "counts_csr.npz", M, compressed=False)
    np.savez(OUT / "obs_var.npz", genes=genes, **cols)
    print("done", M.shape, M.nnz)


if __name__ == "__main__":
    main()
