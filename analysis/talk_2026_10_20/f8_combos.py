"""F8: non-additivity of Norman 2019 pairs and the direction / scale errors of UnseenCombo predictions."""
import json

import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp

from analyze import OUT

RAW = "/cluster/work/boeva/virtual_cell_reasoning/data/norman19/norman19_downloaded.h5ad"
TRUTH = "/cluster/work/boeva/virtual_cell_reasoning/data/norman19/norman19_Miller_align.h5ad"
PRED = "/cluster/work/boeva/virtual_cell_reasoning/models/norman19/_panel_miller_align/{m}/UnseenCombo/fold{k}/predictions.npz"
MODELS = ["PRESAGE", "GEARS", "scGPT", "Additive"]
KAPPA = 1e4


def read_raw():
    """CSR counts, perturbation labels and gene names of the downloaded Norman 2019 file."""
    with h5py.File(RAW, "r") as f:
        X = f["X"]
        n, g = X.attrs["shape"]
        kind = sp.csc_matrix if X.attrs["encoding-type"] == "csc_matrix" else sp.csr_matrix
        m = kind((X["data"][:], X["indices"][:], X["indptr"][:]), shape=(n, g)).tocsr()
        p = f["obs"]["perturbation"]
        lab = p["categories"][:].astype(str)[p["codes"][:]]
        genes = f["var"]["_index"][:].astype(str)
    return m, lab, genes


def stats(m, rows, g0):
    """Mean and sample variance of x = log1p(kappa c / L) over G0 for a set of cells."""
    c = m[rows]
    dep = np.asarray(c.sum(1)).ravel()
    x = np.log1p(KAPPA * c[:, g0].toarray() / dep[:, None])
    return x.mean(0), x.var(0, ddof=1), len(rows)


def non_additivity():
    """NA per pair with both singles measured, space C, all controls, both targets removed."""
    m, lab, genes = read_raw()
    ctrl = np.flatnonzero(lab == "control")
    detect = np.asarray((m[ctrl] > 0).mean(0)).ravel()
    g0 = np.flatnonzero((detect >= 0.05) & ~np.char.startswith(genes, "MT-"))
    g0_names = genes[g0]
    mc, vc, nc = stats(m, ctrl, g0)
    cache = {}

    def get(name):
        if name not in cache:
            cache[name] = stats(m, np.flatnonzero(lab == name), g0)
        return cache[name]

    rows = []
    singles = {l for l in np.unique(lab) if "_" not in l and l != "control"}
    for pair in sorted(l for l in np.unique(lab) if "_" in l):
        a, b = pair.split("_")
        if a not in singles or b not in singles:
            continue
        (mab, vab, nab), (ma, va, na), (mb, vb, nb) = get(pair), get(a), get(b)
        keep = ~np.isin(g0_names, [a, b])
        resid = (mab - mc) - (ma - mc) - (mb - mc)
        se2 = vab / nab + va / na + vb / nb + vc / nc
        se2_ab = vab / nab + vc / nc
        num = float(((resid ** 2 - se2)[keep]).sum())
        den = float((((mab - mc) ** 2 - se2_ab)[keep]).sum())
        rows.append(dict(pair=f"{a}+{b}", n_pair=nab, n_a=na, n_b=nb, NA=num / den if den > 0 else np.nan,
                         NA_numerator=num, NA_denominator=den))
    return pd.DataFrame(rows), dict(n_cells=int(m.shape[0]), n_controls=int(nc), n_G0=int(len(g0)))


def truth():
    """Pipeline truth: first-half pseudobulk minus control pseudobulk, per ko name."""
    with h5py.File(TRUTH, "r") as f:
        pb = f["uns"]["pseudobulk"]
        kos = pb["ko_names"][:].astype(str)
        fh = pb["first_half_bulk"][0]
        ctrl = pb["ctrl_bulk"][0]
        genes = f["var"]["_index"][:].astype(str)
    return {k: fh[i] - ctrl for i, k in enumerate(kos)}, genes


def errors():
    """Direction and scale errors per pair, model and fold, with the exact identity checked per row."""
    y_of, genes = truth()
    rows = []
    for mname in MODELS:
        for k in (0, 1):
            z = np.load(PRED.format(m=mname, k=k), allow_pickle=True)
            assert list(z["gene_names"].astype(str)) == list(genes)
            for j, ko in enumerate(z["ko_names"].astype(str)):
                pred = z["deltas"][0, j]
                keep = ~np.isin(genes, ko.split("+")) & np.isfinite(pred)
                y, yh = y_of[ko][keep].astype(np.float64), pred[keep].astype(np.float64)
                yy, hh, yh_y = y @ y, yh @ yh, yh @ y
                cos2 = yh_y ** 2 / (yy * hh) if hh > 0 else np.nan
                beta = yh_y / hh if hh > 0 else np.nan
                d = yy * (1 - cos2)
                s = (1 - beta) ** 2 * hh
                sq = float(((yh - y) ** 2).sum())
                rows.append(dict(pair=ko, model=mname, fold=k, cos=yh_y / np.sqrt(yy * hh) if hh > 0 else np.nan,
                                 beta_star=beta, direction_error=d, scale_error=s, squared_error=sq,
                                 identity_abs_diff=abs(d + s - sq), identity_rel_diff=abs(d + s - sq) / sq,
                                 n_genes=int(keep.sum()), n_genes_nan_prediction=int((~np.isfinite(pred)).sum())))
    return pd.DataFrame(rows)


def main():
    """Write NA, errors and their join; print headline numbers."""
    na, info = non_additivity()
    err = errors()
    na.to_csv(OUT / "F8_non_additivity_norman19.csv", index=False)
    err.to_csv(OUT / "F8_direction_scale_errors_norman19.csv", index=False)
    j = err.merge(na, on="pair", how="left")
    j.to_csv(OUT / "F8_errors_with_NA_norman19.csv", index=False)
    head = dict(info, n_pairs_NA=len(na), NA_quantiles=na.NA.quantile([0.1, 0.25, 0.5, 0.75, 0.9]).round(3).tolist(),
                identity_max_rel_diff=float(err.identity_rel_diff.max()), n_error_rows=len(err),
                n_pairs_with_NA_in_errors=int(j.NA.notna().groupby(j.pair).first().sum()))
    for mname in MODELS:
        e = j[j.model == mname]
        frac = (e.direction_error / e.squared_error)
        head[mname] = dict(n=len(e), median_direction_share=float(frac.median()),
                           spearman_NA_vs_direction_error=float(e[["NA", "direction_error"]].corr("spearman").iloc[0, 1]),
                           spearman_NA_vs_scale_error=float(e[["NA", "scale_error"]].corr("spearman").iloc[0, 1]))
    (OUT / "F8_headline.json").write_text(json.dumps(head, indent=1))
    print(json.dumps(head, indent=1))


if __name__ == "__main__":
    main()
