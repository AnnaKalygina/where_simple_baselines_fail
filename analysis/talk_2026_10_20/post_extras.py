"""phi / phi_loo, F2, F7 and F9 summaries, figure T2-B and the full-descriptor heatmap T1-A."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyze import GRAY, LABELS, NAVY, OUT, RED, TURQ, spearman

CACHE = "/cluster/work/boeva/akalygina/wsb13_cache"
DESCRIPTORS = {"M": "M", "breadth_pr": "breadth", "rho": "ρ", "tau2": "τ²", "phi": "φ", "phi_loo": "φ loo",
               "pi_raw": "π (raw)", "dBIC": "ΔBIC", "mu1": "μ1", "D": "D", "mito_AmB": "mito A−B",
               "mito_frac_median": "mito frac", "n_cells": "n cells", "median_depth": "depth",
               "log2_depth_ratio": "depth ratio"}
S0A = {"M": "M", "tau2": "tau2", "pi_raw": "pi", "D": "D", "rho": "rho"}


def phi_columns(m1):
    """cos^2 between each effect and the cell-weighted mean effect of reliable perturbations, with and without p."""
    y = np.load(f"{CACHE}/y_spaceC_G0_k562_essential.npy").astype(np.float64)
    genes = pd.read_csv(f"{CACHE}/g0_genes.csv", header=None)[0].astype(str).values
    pos = {g: i for i, g in enumerate(genes)}
    rel = (m1.rho >= 0.5).values
    w = m1.n_cells.values.astype(float) * rel
    tot = (y * w[:, None]).sum(0)
    phi, loo = np.empty(len(y)), np.empty(len(y))
    for k in range(len(y)):
        g = np.ones(len(genes), bool)
        if m1.perturbation.iloc[k] in pos:
            g[pos[m1.perturbation.iloc[k]]] = False
        for out, mean in ((phi, tot / w.sum()), (loo, (tot - w[k] * y[k]) / (w.sum() - w[k]))):
            a, b = y[k, g], mean[g]
            out[k] = (a @ b) ** 2 / ((a @ a) * (b @ b))
    return phi, loo


def plot_t2b(m, head):
    """Reliable fraction per space / weighting, and the per-perturbation depth factor R."""
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(6, 4.5), gridspec_kw={"width_ratios": [1.1, 1]})
    keys = [("rho_sb_C_G0", "C"), ("rho_sb_A_G0", "A"), ("rho_sb_B_G0", "B"), ("rho_sb_NB_G0", "A, NB")]
    vals = [100 * (m[k] >= 0.5).mean() for k, _ in keys]
    ax.bar(range(len(keys)), vals, color=[RED, TURQ, TURQ, NAVY], width=0.65)
    for i, v in enumerate(vals):
        ax.text(i, v + 1, f"{v:.1f}", ha="center", va="bottom", fontsize=12, color=NAVY)
    ax.set_xticks(range(len(keys)), [l for _, l in keys])
    ax.set_xlabel("space")
    ax.set_ylabel("ρ ≥ 0.5 (%)")
    ax.set_ylim(0, max(vals) * 1.25)
    bx.hist(m.depth_R, bins=40, color=TURQ)
    bx.axvline(head["control_depth_R"], color=RED, lw=1.5)
    bx.set_xlabel("R = mean(L) · mean(1/L)")
    bx.set_ylabel("perturbations")
    for a in (ax, bx):
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "T2-B_reliability_spaces.png", dpi=300)
    plt.close(fig)


def plot_t1a(m, s0):
    """Spearman matrix of all descriptors ordered by |corr with M|, with a stability side bar."""
    cols = list(DESCRIPTORS)
    sm = pd.DataFrame(np.eye(len(cols)), index=cols, columns=cols)
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            sm.loc[a, b] = sm.loc[b, a] = spearman(m[a], m[b])[0]
    order = sm["M"].abs().sort_values(ascending=False).index.tolist()
    sm = sm.loc[order, order]
    sm.to_csv(OUT / "T1-A_spearman_all_descriptors.csv")
    stab = s0.set_index("metric").S0a_spearman_brown
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(6, 4.5), gridspec_kw={"width_ratios": [6, 1], "wspace": 0.05})
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("nr", [TURQ, "white", RED])
    ax.imshow(sm.values, cmap=cmap, vmin=-1, vmax=1)
    lab = [DESCRIPTORS[c] for c in order]
    ax.set_xticks(range(len(order)), lab, rotation=90, fontsize=12)
    ax.set_yticks(range(len(order)), lab, fontsize=12)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    vals = [stab.get(S0A[c], np.nan) if c in S0A else np.nan for c in order]
    y = np.arange(len(order))
    bx.barh(y, np.nan_to_num(vals), color=NAVY, height=0.7)
    bx.set_ylim(len(order) - 0.5, -0.5)
    bx.set_yticks([])
    bx.set_xlim(0, 1)
    bx.set_xticks([0, 1])
    bx.set_xlabel("S0-a")
    bx.spines[["top", "right", "left"]].set_visible(False)
    fig.subplots_adjust(left=0.2, right=0.98, bottom=0.24, top=0.99)
    fig.savefig(OUT / "T1-A_descriptor_heatmap.png", dpi=300)
    plt.close(fig)
    return order


def main():
    """Join M1 with the extras, compute phi, summarize F2/F7/F9, draw T1-A and T2-B."""
    m1 = pd.read_csv(OUT / "M1_k562_essential.csv")
    ex = pd.read_csv(OUT / "M1_extras_k562_essential.csv")
    m = m1.merge(ex, on="perturbation", validate="one_to_one")
    m["phi"], m["phi_loo"] = phi_columns(m)
    m.to_csv(OUT / "M1_full_k562_essential.csv", index=False)
    head = json.loads((OUT / "extras_setup_k562_essential.json").read_text())
    rep = {"check_C_G0_equals_M1_rho_max_abs_diff": float((m.rho_sb_C_G0 - m.rho).abs().max())}
    for k in ["rho_sb_C_G0", "rho_sb_C_HVG2000", "rho_sb_C_HVG8192", "rho_sb_A_G0", "rho_sb_B_G0", "rho_sb_NB_G0"]:
        rep[f"frac_ge_0_5_{k}"] = float((m[k] >= 0.5).mean())
    rep["spearman_rho_A_vs_C"] = spearman(m.rho_sb_A_G0, m.rho_sb_C_G0)[0]
    rep["depth_R_quantiles"] = m.depth_R.quantile([0.05, 0.5, 0.95]).round(4).tolist()
    rep["depth_cv_median"] = float(m.depth_cv.median())
    rep["control_depth_R"], rep["control_depth_cv"] = head["control_depth_R"], head["control_depth_cv"]
    rep["phi_median"], rep["phi_loo_median"] = float(m.phi.median()), float(m.phi_loo.median())
    rep["breadth_pr_median"] = float(m.breadth_pr.median())
    rep["mito_AmB_quantiles"] = m.mito_AmB.quantile([0.05, 0.5, 0.95]).round(2).tolist()
    pd.Series(rep).to_json(OUT / "F2_F7_F9_headline.json", indent=1)
    m[["perturbation", "n_cells", "rho_sb_C_G0", "rho_sb_C_HVG2000", "rho_sb_C_HVG8192"]].to_csv(OUT / "F2_panels.csv", index=False)
    m[["perturbation", "n_cells", "rho_sb_A_G0", "rho_sb_B_G0", "rho_sb_C_G0", "rho_sb_NB_G0", "depth_R", "depth_cv"]].to_csv(
        OUT / "F7_F9_spaces_weights.csv", index=False)
    plot_t2b(m, head)
    s0 = pd.read_csv(OUT / "S0_checks.csv")
    order = plot_t1a(m, s0)
    print(json.dumps(rep, indent=1, default=float))
    print("T1-A order", order)


if __name__ == "__main__":
    main()
