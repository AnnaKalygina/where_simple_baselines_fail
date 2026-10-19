"""F10: the mean perturbation effect of K562 essential vs genome-wide, each screen against its own controls."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyze import GRAY, NAVY, OUT, RED, TURQ

CACHE = "/cluster/work/boeva/akalygina/wsb13_cache"
SCREENS = {"essential": TURQ, "genome-wide": RED}
FILES = {"essential": "essential", "genome-wide": "gwps"}


def load(screen):
    """Per-perturbation and control summaries of one screen, on all stored genes."""
    z = np.load(f"{CACHE}/f10_sums_{FILES[screen]}.npz", allow_pickle=True)
    s1, s2, cnt = z["s1"], z["s2"], z["counts"].astype(float)
    p = len(z["perts"])
    na, nb = cnt[0:2 * p:2], cnt[1:2 * p:2]
    sa, sb = s1[0:2 * p:2], s1[1:2 * p:2]
    qa, qb = s2[0:2 * p:2], s2[1:2 * p:2]
    n = na + nb
    mean = (sa + sb) / n[:, None]
    var = ((qa + qb) - n[:, None] * mean ** 2) / (n[:, None] - 1)
    k = slice(2 * p, 2 * p + 3)
    n_th = cnt[k]
    nc = n_th.sum()
    cm = s1[k].sum(0) / nc
    cv = (s2[k].sum(0) - nc * cm ** 2) / (nc - 1)
    genes = z["genes"].astype(str)
    g0 = (z["ctrl_detect"] / nc >= 0.05) & ~np.char.startswith(genes, "MT-")
    return dict(perts=z["perts"].astype(str), genes=genes, g0=g0, n=n, mean=mean, var=var, cm=cm, cv=cv, nc=nc,
                mean_a=sa / na[:, None], mean_b=sb / nb[:, None],
                k1=s1[2 * p] / n_th[0], k2=s1[2 * p + 1] / n_th[1],
                null_s1=z["null_s1"], null_s2=z["null_s2"], n_null=int(z["n_null"]), n_cells=int(z["n_cells"]))


def restrict(d, shared):
    """Per-perturbation effects, magnitudes, half effects and null magnitudes on the shared genes."""
    col = {g: i for i, g in enumerate(d["genes"])}
    idx = np.array([col[g] for g in shared])
    pos = {g: i for i, g in enumerate(shared)}
    y = d["mean"][:, idx] - d["cm"][idx]
    se2 = d["var"][:, idx] / d["n"][:, None] + d["cv"][idx] / d["nc"]
    ya = d["mean_a"][:, idx] - d["k1"][idx]
    yb = d["mean_b"][:, idx] - d["k2"][idx]
    mask = np.ones_like(y, bool)
    for k, p in enumerate(d["perts"]):
        if p in pos:
            mask[k, pos[p]] = False
    m = np.where(mask, y ** 2 - se2, 0).sum(1)
    nn = d["n_null"]
    null_m = []
    c1 = d["cm"][idx] * d["nc"]
    c2 = d["cv"][idx] * (d["nc"] - 1) + d["nc"] * d["cm"][idx] ** 2
    for j in range(d["null_s1"].shape[0]):
        a1, a2 = d["null_s1"][j, idx], d["null_s2"][j, idx]
        mu = a1 / nn
        v = (a2 - nn * mu ** 2) / (nn - 1)
        nr = d["nc"] - nn
        mr = (c1 - a1) / nr
        vr = ((c2 - a2) - nr * mr ** 2) / (nr - 1)
        null_m.append(float(((mu - mr) ** 2 - (v / nn + vr / nr)).sum()))
    return dict(y=y, se2=se2, ya=ya, yb=yb, mask=mask, M=m, null_M=np.array(null_m))


def corr(a, b):
    """Pearson correlation."""
    a, b = a - a.mean(), b - b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 0 else np.nan


def per_pert(r, perts):
    """Distance to the leave-one-out mean and Baseline Saturation (Pearson, mean baseline) per perturbation."""
    y, mask = r["y"], r["mask"]
    p = len(y)
    ybar = y.mean(0)
    rows = []
    for k in range(p):
        b = (p * ybar - y[k]) / (p - 1)
        g = mask[k]
        r_ab = corr(r["ya"][k, g], r["yb"][k, g])
        bs = np.clip(corr(b[g], r["yb"][k, g]) / r_ab, 0, 1) if r_ab > 0 else np.nan
        rows.append(dict(perturbation=perts[k], M=r["M"][k], dist_to_mean=float(np.linalg.norm(y[k, g] - b[g])),
                         r_half_split0=r_ab, BS_pearson=bs))
    return pd.DataFrame(rows)


def main():
    """Compute F10 (a) to (d), write tables, figure and headline numbers."""
    d = {s: load(s) for s in SCREENS}
    shared = sorted(set(d["essential"]["genes"][d["essential"]["g0"]]) & set(d["genome-wide"]["genes"][d["genome-wide"]["g0"]]))
    r = {s: restrict(d[s], shared) for s in SCREENS}
    head = {"n_shared_genes": len(shared)}
    tabs = {}
    for s in SCREENS:
        ybar = r[s]["y"].mean(0)
        ybar_w = (r[s]["y"] * d[s]["n"][:, None]).sum(0) / d[s]["n"].sum()
        thr = float(r[s]["null_M"].max())
        head[s] = dict(n_cells=d[s]["n_cells"], n_controls=int(d[s]["nc"]), n_perts=len(d[s]["perts"]),
                       norm_mean_effect=float(np.linalg.norm(ybar)), norm_mean_effect_cell_weighted=float(np.linalg.norm(ybar_w)),
                       M_quantiles=dict(zip(["p10", "p25", "p50", "p75", "p90"], np.percentile(r[s]["M"], [10, 25, 50, 75, 90]).round(3).tolist())),
                       threshold_max_null_M=thr, null_M_p95=float(np.percentile(r[s]["null_M"], 95)), n_null=len(r[s]["null_M"]),
                       null_size=d[s]["n_null"], frac_M_above_threshold=float((r[s]["M"] > thr).mean()))
        r[s]["ybar"] = ybar
        tabs[s] = per_pert(r[s], d[s]["perts"])
        tabs[s].assign(n_cells=d[s]["n"]).to_csv(OUT / f"F10_per_perturbation_{FILES[s]}.csv", index=False)
    a, b = r["essential"]["ybar"], r["genome-wide"]["ybar"]
    head["cosine_mean_effects"] = float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))
    both = tabs["essential"].merge(tabs["genome-wide"], on="perturbation", suffixes=("_ess", "_gw"))
    both.to_csv(OUT / "F10_shared_perturbations.csv", index=False)
    head["n_shared_perts"] = len(both)
    head["shared_median_dist_ess"] = float(both.dist_to_mean_ess.median())
    head["shared_median_dist_gw"] = float(both.dist_to_mean_gw.median())
    head["shared_frac_dist_gw_gt_ess"] = float((both.dist_to_mean_gw > both.dist_to_mean_ess).mean())
    ok = both.BS_pearson_ess.notna() & both.BS_pearson_gw.notna()
    head["shared_BS_n"] = int(ok.sum())
    head["shared_median_BS_ess"] = float(both.BS_pearson_ess[ok].median())
    head["shared_median_BS_gw"] = float(both.BS_pearson_gw[ok].median())
    head["shared_frac_BS_gw_lt_ess"] = float((both.BS_pearson_gw[ok] < both.BS_pearson_ess[ok]).mean())
    (OUT / "F10_headline.json").write_text(json.dumps(head, indent=1))
    print(json.dumps(head, indent=1))
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(6, 4.5))
    for s, col in SCREENS.items():
        v = np.sort(r[s]["M"])
        ax.plot(np.clip(v, 1e-2, None), np.linspace(0, 1, len(v)), color=col, lw=2, label=s)
        ax.axvline(max(head[s]["threshold_max_null_M"], 1e-2), color=col, lw=1, ls="--")
    ax.set_xscale("log")
    ax.set_xlabel("magnitude M (log1p², clipped at 0.01)")
    ax.set_ylabel("fraction of perturbations")
    ax.legend(frameon=False, fontsize=12, loc="upper left")
    bx.scatter(both.BS_pearson_ess[ok], both.BS_pearson_gw[ok], s=8, color=NAVY, alpha=0.5, linewidths=0)
    bx.plot([0, 1], [0, 1], color=GRAY, lw=1)
    bx.set_xlabel("saturation, essential")
    bx.set_ylabel("saturation, genome-wide")
    bx.set_xlim(0, 1.02)
    bx.set_ylim(0, 1.02)
    for x_ in (ax, bx):
        x_.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "F10_screen_mean.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
