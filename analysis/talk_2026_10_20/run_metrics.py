"""Compute M1 (per perturbation), M2 (control pseudo-perturbations) and the S0 inputs on K562 essential."""
import os
import pickle
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

import metrics as mx

CACHE = Path("/cluster/work/boeva/akalygina/wsb13_cache")
OUT = Path(__file__).resolve().parents[2] / "results" / "talk_2026-10-20"
SEED = 20261020
MIN_CELLS = 20
MIN_GENES = 200
G0_DETECT = 0.05
N_NULL = 50
DOWN_N = 100
DOWN_MIN = 200
DOWN_REPEATS = 10
CTRL = "non-targeting"

D = {}


def setup():
    """Load counts, apply QC, build G0 and the control contexts."""
    counts = sp.load_npz(CACHE / "k562" / "counts_csr.npz")
    ov = np.load(CACHE / "k562" / "obs_var.npz", allow_pickle=True)
    keep = ov["ngenes"] >= MIN_GENES
    if not keep.all():
        counts = counts[keep]
    gene = ov["gene"][keep]
    genes = ov["genes"]
    depth = np.asarray(counts.sum(1)).ravel()
    ctrl = np.flatnonzero(gene == CTRL)
    detect = np.asarray((counts[ctrl] > 0).mean(0)).ravel()
    is_mt = np.char.startswith(genes.astype(str), "MT-")
    g0 = np.flatnonzero((detect >= G0_DETECT) & ~is_mt)
    cc = counts[ctrl][:, g0].toarray().astype(np.float64)
    xc, uc = mx.normalize(cc, depth[ctrl])
    ctx = mx.ControlContext(xc, uc, depth[ctrl], [SEED, 1])
    k1, k2, _ = ctx.thirds
    names, sizes = np.unique(gene[gene != CTRL], return_counts=True)
    D.update(counts=counts, gene=gene, genes=genes, depth=depth, ctrl=ctrl, g0=g0, ctx=ctx,
             trend=ctx.trend(np.ones(len(g0), bool)),
             ctx_k1=ctx.sub(k1, [SEED, 2]), ctx_k2=ctx.sub(k2, [SEED, 3]),
             gene_index={g: i for i, g in enumerate(genes)}, g0_pos={g: i for i, g in enumerate(g0)},
             perts=names[sizes >= MIN_CELLS], sizes=sizes[sizes >= MIN_CELLS], excluded=names[sizes < MIN_CELLS],
             n_qc_removed=int((~keep).sum()),
             ctrl_counts_full=counts[ctrl])
    D["trend_k1"] = D["ctx_k1"].trend(np.ones(len(g0), bool))
    D["trend_k2"] = D["ctx_k2"].trend(np.ones(len(g0), bool))


def cells_of(rows):
    """Dense (x, u, depth) on G0 for a set of cell rows."""
    c = D["counts"][rows][:, D["g0"]].toarray().astype(np.float64)
    x, u = mx.normalize(c, D["depth"][rows])
    return x, u, D["depth"][rows]


def run_pert(i):
    """All M1 columns, scores and S0 values for perturbation i; checkpointed."""
    name = D["perts"][i]
    path = CACHE / "m1" / f"{i:05d}.pkl"
    if path.exists():
        return
    rng = np.random.default_rng([SEED, 100, i])
    rows = np.flatnonzero(D["gene"] == name)
    x, u, dep = cells_of(rows)
    genes = np.ones(len(D["g0"]), bool)
    gi = D["gene_index"].get(name)
    in_g0 = gi is not None and gi in D["g0_pos"]
    if in_g0:
        genes[D["g0_pos"][gi]] = False
    target_u = ctrl_u = None
    if gi is not None:
        cc = np.asarray(D["ctrl_counts_full"][:, gi].todense()).ravel()
        if cc.mean() >= 0.5:
            ctrl_u = mx.KAPPA * cc / D["ctx"].depth
            target_u = mx.KAPPA * np.asarray(D["counts"][rows][:, gi].todense()).ravel() / dep
    row, sb, s0, _ = mx.full_metrics(x, u, dep, D["ctx"], genes, D["trend"], rng, target_u, ctrl_u)
    row.update(perturbation=name, target=name, target_in_matrix=gi is not None, target_in_G0=in_g0,
               escaper_eligible=target_u is not None)
    perm = rng.permutation(len(rows))
    a, b = perm[: len(rows) // 2], perm[len(rows) // 2:]
    s0a = mx.lite_metrics(x[a], u[a], dep[a], D["ctx_k1"], genes, D["trend_k1"], rng)
    s0b = mx.lite_metrics(x[b], u[b], dep[b], D["ctx_k2"], genes, D["trend_k2"], rng)
    down = None
    if len(rows) >= DOWN_MIN:
        reps = []
        for _ in range(DOWN_REPEATS):
            k = rng.choice(len(rows), DOWN_N, replace=False)
            reps.append(mx.lite_metrics(x[k], u[k], dep[k], D["ctx"], genes, D["trend"], rng))
        down = {c: float(np.nanmedian([r[c] for r in reps])) if np.isfinite([r[c] for r in reps]).any() else np.nan
                for c in reps[0]}
    scores = pd.DataFrame({"perturbation": name,
                           "cell_set": ["half_B"] * len(sb) + ["K3"] * len(s0),
                           "score": np.concatenate([sb, s0]).astype(np.float32)})
    with open(path, "wb") as f:
        pickle.dump(dict(row=row, half_a=s0a, half_b=s0b, down=down, scores=scores), f)


def run_null(j):
    """One control pseudo-perturbation scored against the remaining controls; checkpointed."""
    path = CACHE / "m2" / f"{j:03d}.pkl"
    if path.exists():
        return
    rng = np.random.default_rng([SEED, 200, j])
    ctx = D["ctx"]
    n_p = int(np.median(D["sizes"]))
    idx = rng.choice(ctx.n, n_p, replace=False)
    rest = np.setdiff1d(np.arange(ctx.n), idx)
    sub = ctx.sub(rest, [SEED, 300, j])
    genes = np.ones(len(D["g0"]), bool)
    row, *_ = mx.full_metrics(ctx.x[idx], ctx.u[idx], ctx.depth[idx], sub, genes,
                              sub.trend(genes), rng)
    row.update(perturbation=f"null_{j:02d}")
    with open(path, "wb") as f:
        pickle.dump(row, f)


def collect():
    """Gather checkpoints into the result tables."""
    rows, s0, scores = [], [], []
    for i, name in enumerate(D["perts"]):
        with open(CACHE / "m1" / f"{i:05d}.pkl", "rb") as f:
            r = pickle.load(f)
        rows.append(r["row"])
        rec = {"perturbation": name, "n_cells": r["row"]["n_cells"]}
        for tag in ("half_a", "half_b", "down"):
            for c, v in (r[tag] or {}).items():
                rec[f"{c}_{tag}"] = v
        s0.append(rec)
        scores.append(r["scores"])
    m1 = pd.DataFrame(rows)
    lead = ["perturbation", "target", "target_in_matrix", "target_in_G0", "n_cells", "median_depth"]
    m1 = m1[lead + [c for c in m1.columns if c not in lead]]
    m1.to_csv(OUT / "M1_k562_essential.csv", index=False)
    pd.DataFrame(s0).to_csv(OUT / "S0_inputs_k562_essential.csv", index=False)
    pd.concat(scores).to_parquet(CACHE / "scores_halfB_K3_k562_essential.parquet", index=False)
    m2 = []
    for j in range(N_NULL):
        with open(CACHE / "m2" / f"{j:03d}.pkl", "rb") as f:
            m2.append(pickle.load(f))
    pd.DataFrame(m2).to_csv(OUT / "M2_null_k562_essential.csv", index=False)
    info = dict(n_cells_total=int(D["counts"].shape[0]), n_controls=int(len(D["ctrl"])),
                n_qc_removed=D["n_qc_removed"], n_G0=int(len(D["g0"])), n_perts_included=int(len(D["perts"])),
                n_perts_excluded=int(len(D["excluded"])), excluded=list(map(str, D["excluded"])))
    pd.Series(info).to_json(OUT / "setup_k562_essential.json", indent=1)


def run_task(t):
    """Dispatch one pool task."""
    kind, k = t
    (run_pert if kind == "p" else run_null)(k)
    return t


def main():
    """Entry point: setup, parallel per-perturbation tasks, collect."""
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else os.cpu_count()
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else None
    (CACHE / "m1").mkdir(exist_ok=True)
    (CACHE / "m2").mkdir(exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    setup()
    print("setup done", D["counts"].shape, len(D["ctrl"]), len(D["g0"]), len(D["perts"]), flush=True)
    order = np.argsort(-D["sizes"])
    tasks = [("p", int(i)) for i in order][:limit] if limit else [("p", int(i)) for i in order]
    tasks = ([("n", j) for j in range(N_NULL)] if not limit else []) + tasks
    with Pool(workers) as pool:
        for k, t in enumerate(pool.imap_unordered(run_task, tasks)):
            if k % 50 == 0:
                print(k, t, flush=True)
    if not limit:
        collect()
    print("done", flush=True)


if __name__ == "__main__":
    main()
