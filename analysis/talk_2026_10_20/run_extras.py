"""Brief extras on K562 essential: breadth_pr, phi inputs, mito columns, depth, F2 panels, F7 spaces and F9 NB weights."""
import pickle
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

import metrics as mx
import run_metrics as rm

CACHE = rm.CACHE / "extras"
HVG_SIZES = (2000, 8192)


def hvg_masks():
    """Top-2,000 and top-8,192 HVG (scanpy default flavor, all cells) as masks over G0."""
    import anndata as ad
    import scanpy as sc
    a = ad.AnnData(rm.D["counts"].astype(np.float32))
    a.var_names = rm.D["genes"].astype(str)
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    g0_names = rm.D["genes"][rm.D["g0"]].astype(str)
    out = {}
    for k in HVG_SIZES:
        h = sc.pp.highly_variable_genes(a, n_top_genes=k, inplace=False)
        names = set(a.var_names[h["highly_variable"].values])
        out[k] = np.isin(g0_names, list(names))
        rm.D[f"hvg{k}_outside_g0"] = int(len(names) - out[k].sum())
    return out


def control_split_terms():
    """Per split, the K1 and K2 effect baselines in spaces A, B and NB-weighted A."""
    ctx = rm.D["ctx"]
    cc = rm.D["ctrl_c"]
    dep = ctx.depth
    rng = np.random.default_rng([rm.SEED, 1])
    terms = {s: np.empty((2, mx.N_SPLITS, cc.shape[1])) for s in ("A", "B", "NB")}
    for r in range(mx.N_SPLITS):
        perm = rng.permutation(ctx.n)
        thirds = np.array_split(perm, 3)
        for j in range(2):
            k = thirds[j]
            for s, v in pseudobulks(cc[k], ctx.u[k], dep[k]).items():
                terms[s][j, r] = v
    return terms


def pseudobulks(c, u, dep):
    """Space A, space B and NB-weighted space A pseudobulk profiles of one cell set."""
    phi_theta = rm.D["nb_phi_theta"]
    w = dep[:, None] / (1 + phi_theta * dep[:, None])
    theta = (w * c).sum(0) / (w * dep[:, None]).sum(0)
    return {"A": np.log1p(mx.KAPPA * c.sum(0) / dep.sum()), "B": np.log1p(u.mean(0)),
            "NB": np.log1p(mx.KAPPA * theta)}


def setup_extras():
    """Everything the extras need on top of run_metrics.setup()."""
    rm.setup()
    ctrl = rm.D["ctrl"]
    rm.D["ctrl_c"] = rm.D["counts"][ctrl][:, rm.D["g0"]].toarray().astype(np.float64)
    ctx = rm.D["ctx"]
    theta = ctx.mu / mx.KAPPA
    rm.D["nb_phi_theta"] = np.clip(np.nan_to_num(ctx.phi, nan=0.0), 0, None) * theta
    rm.D["hvg"] = hvg_masks()
    rm.D["split_terms"] = control_split_terms()
    mt = np.flatnonzero(np.char.startswith(rm.D["genes"].astype(str), "MT-"))
    rm.D["mt"] = mt
    rm.D["ctrl_mito"] = mito_terms(ctrl)


def mito_terms(rows):
    """(sum over MT genes of A' minus B' on the 1e6 scale, median per-cell MT fraction) for a cell set."""
    c = rm.D["counts"][rows][:, rm.D["mt"]].toarray().astype(np.float64)
    dep = rm.D["depth"][rows]
    a = 1e6 * c.sum(0) / dep.sum()
    b = (1e6 * c / dep[:, None]).mean(0)
    return float((a - b).sum()), float(np.median(c.sum(1) / dep))


def run_one(i):
    """Extras for perturbation i; checkpointed."""
    path = CACHE / f"{i:05d}.pkl"
    if path.exists():
        return i
    name = rm.D["perts"][i]
    rows = np.flatnonzero(rm.D["gene"] == name)
    c = rm.D["counts"][rows][:, rm.D["g0"]].toarray().astype(np.float64)
    dep = rm.D["depth"][rows]
    x, u = mx.normalize(c, dep)
    ctx = rm.D["ctx"]
    genes = np.ones(len(rm.D["g0"]), bool)
    gi = rm.D["gene_index"].get(name)
    if gi is not None and gi in rm.D["g0_pos"]:
        genes[rm.D["g0_pos"][gi]] = False
    rng = np.random.default_rng([rm.SEED, 100, i])
    mx.magnitude_ci(x, ctx, genes, rng, 0.0)
    halves = mx.split_halves(len(rows), rng)
    panels = {"G0": genes, "HVG2000": genes & rm.D["hvg"][2000], "HVG8192": genes & rm.D["hvg"][8192]}
    rs = {k: [] for k in ["C_G0", "C_HVG2000", "C_HVG8192", "A_G0", "B_G0", "NB_G0"]}
    st = rm.D["split_terms"]
    for r, (a, b) in enumerate(halves):
        ya = x[a].mean(0) - ctx.k1_means[r]
        yb = x[b].mean(0) - ctx.k2_means[r]
        for p, m in panels.items():
            rs[f"C_{p}"].append(np.corrcoef(ya[m], yb[m])[0, 1])
        pa, pb = pseudobulks(c[a], u[a], dep[a]), pseudobulks(c[b], u[b], dep[b])
        for s in ("A", "B", "NB"):
            rs[f"{s}_G0"].append(np.corrcoef((pa[s] - st[s][0, r])[genes], (pb[s] - st[s][1, r])[genes])[0, 1])
    row = {"perturbation": name}
    for k, v in rs.items():
        rh = float(np.median(v))
        row[f"rho_sb_{k}"] = 2 * rh / (1 + rh)
    m, y, se2, _ = mx.magnitude(x, ctx, genes)
    aa = np.clip((y ** 2 - se2)[genes], 0, None)
    row["breadth_pr"] = float(aa.sum() ** 2 / (aa ** 2).sum()) if aa.any() else np.nan
    amb, mfrac = mito_terms(rows)
    row["mito_AmB"] = amb - rm.D["ctrl_mito"][0]
    row["mito_frac_median"] = mfrac
    row["log2_depth_ratio"] = float(np.log2(np.median(dep) / np.median(ctx.depth)))
    row["depth_R"] = float(dep.mean() * (1 / dep).mean())
    row["depth_cv"] = float(dep.std(ddof=1) / dep.mean())
    with open(path, "wb") as f:
        pickle.dump(dict(row=row, y=y.astype(np.float32)), f)
    return i


def collect():
    """Write the extras table, the effect matrix for phi, and setup facts."""
    rows, ys = [], []
    for i in range(len(rm.D["perts"])):
        with open(CACHE / f"{i:05d}.pkl", "rb") as f:
            r = pickle.load(f)
        rows.append(r["row"])
        ys.append(r["y"])
    pd.DataFrame(rows).to_csv(rm.OUT / "M1_extras_k562_essential.csv", index=False)
    np.save(rm.CACHE / "y_spaceC_G0_k562_essential.npy", np.stack(ys))
    ctx = rm.D["ctx"]
    facts = dict(hvg2000_outside_G0=rm.D["hvg2000_outside_g0"], hvg8192_outside_G0=rm.D["hvg8192_outside_g0"],
                 hvg2000_in_G0=int(rm.D["hvg"][2000].sum()), hvg8192_in_G0=int(rm.D["hvg"][8192].sum()),
                 n_mt_genes=int(len(rm.D["mt"])), control_mito_frac_median=rm.D["ctrl_mito"][1],
                 control_depth_R=float(ctx.depth.mean() * (1 / ctx.depth).mean()),
                 control_depth_cv=float(ctx.depth.std(ddof=1) / ctx.depth.mean()),
                 nb_genes_with_phi_le_0=int((rm.D["nb_phi_theta"] <= 0).sum()))
    pd.Series(facts).to_json(rm.OUT / "extras_setup_k562_essential.json", indent=1)
    pd.Series(rm.D["genes"][rm.D["g0"]].astype(str)).to_csv(rm.CACHE / "g0_genes.csv", index=False, header=False)


def main():
    """Setup, parallel extras per perturbation, collect."""
    workers = int(sys.argv[1])
    CACHE.mkdir(exist_ok=True)
    setup_extras()
    print("setup done", flush=True)
    order = [int(i) for i in np.argsort(-rm.D["sizes"])]
    with Pool(workers) as pool:
        for k, _ in enumerate(pool.imap_unordered(run_one, order)):
            if k % 200 == 0:
                print(k, flush=True)
    collect()
    print("done", flush=True)


if __name__ == "__main__":
    main()
