"""Stream a Replogle 2022 raw single-cell h5ad and accumulate per-group sums of x and x^2 in space C (for F10)."""
import sys
from multiprocessing import Pool

import h5py
import numpy as np
import scipy.sparse as sp

KAPPA = 1e4
MIN_CELLS = 20
N_NULL = 50
BLOCK = 4096
CTRL = "non-targeting"
G = {}


def categorical(group, name):
    """Decode an old-style anndata categorical column."""
    cats = group["__categories"][name][:].astype(str)
    return cats[group[name][:]]


def labels(gene, seed):
    """Half labels for perturbed cells, thirds for controls, and 50 disjoint control pseudo-groups."""
    rng = np.random.default_rng(seed)
    names, inv, sizes = np.unique(gene, return_inverse=True, return_counts=True)
    is_pert = (names != CTRL) & (sizes >= MIN_CELLS)
    perts = names[is_pert]
    pos = np.full(len(names), -1)
    pos[is_pert] = np.arange(is_pert.sum())
    lab = np.full(len(gene), -1, np.int64)
    order = np.argsort(inv, kind="stable")
    bounds = np.concatenate([[0], np.cumsum(sizes)])
    for k in np.flatnonzero(is_pert):
        perm = rng.permutation(order[bounds[k]:bounds[k + 1]])
        h = len(perm) // 2
        lab[perm[:h]] = 2 * pos[k]
        lab[perm[h:]] = 2 * pos[k] + 1
    ctrl = rng.permutation(np.flatnonzero(gene == CTRL))
    for t, part in enumerate(np.array_split(ctrl, 3)):
        lab[part] = 2 * len(perts) + t
    n_null = int(np.median(sizes[is_pert]))
    null_lab = np.full(len(gene), -1, np.int64)
    pool = rng.permutation(ctrl)[: N_NULL * n_null]
    for j in range(N_NULL):
        null_lab[pool[j * n_null:(j + 1) * n_null]] = j
    return perts, lab, null_lab, n_null


def work(rng_):
    """Accumulate sums over one row range."""
    start, stop = rng_
    n_groups, n_genes = G["n_groups"], G["n_genes"]
    s1 = np.zeros((n_groups, n_genes))
    s2 = np.zeros((n_groups, n_genes))
    t1 = np.zeros((N_NULL, n_genes))
    t2 = np.zeros((N_NULL, n_genes))
    det = np.zeros(n_genes)
    with h5py.File(G["path"], "r") as f:
        X = f["X"]
        for a in range(start, stop, BLOCK):
            b = min(a + BLOCK, stop)
            lab, nl = G["lab"][a:b], G["null_lab"][a:b]
            keep = lab >= 0
            if not keep.any():
                continue
            c = X[a:b].astype(np.float64)
            dep = c.sum(1)
            x = np.log1p(KAPPA * c / dep[:, None])
            rows = np.flatnonzero(keep)
            ind = sp.csr_matrix((np.ones(len(rows)), (lab[rows], rows)), shape=(n_groups, b - a))
            s1 += ind @ x
            s2 += ind @ (x * x)
            ctrl_rows = np.flatnonzero(lab >= G["first_ctrl"])
            det += (c[ctrl_rows] > 0).sum(0)
            nr = np.flatnonzero(nl >= 0)
            if len(nr):
                indn = sp.csr_matrix((np.ones(len(nr)), (nl[nr], nr)), shape=(N_NULL, b - a))
                t1 += indn @ x
                t2 += indn @ (x * x)
    return s1, s2, t1, t2, det


def main():
    """Entry: path, output npz, workers, seed tag."""
    path, out, workers, tag = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    with h5py.File(path, "r") as f:
        gene = categorical(f["obs"], "gene")
        genes = categorical(f["var"], "gene_name")
        n = f["X"].shape[0]
    perts, lab, null_lab, n_null = labels(gene, [20261020, 500, tag])
    G.update(path=path, lab=lab, null_lab=null_lab, n_groups=2 * len(perts) + 3, n_genes=len(genes),
             first_ctrl=2 * len(perts))
    edges = np.linspace(0, n, workers + 1).astype(int)
    tot = None
    with Pool(workers) as pool:
        for k, r in enumerate(pool.imap_unordered(work, list(zip(edges[:-1], edges[1:])))):
            tot = list(r) if tot is None else [u + v for u, v in zip(tot, r)]
            print(k, flush=True)
    counts = np.bincount(lab[lab >= 0], minlength=G["n_groups"])
    np.savez(out, s1=tot[0], s2=tot[1], null_s1=tot[2], null_s2=tot[3], ctrl_detect=tot[4], counts=counts,
             perts=perts, genes=genes, n_null=n_null, n_ctrl=int((gene == CTRL).sum()), n_cells=n)
    print("done", len(perts), counts[-3:], n_null, flush=True)


if __name__ == "__main__":
    main()
