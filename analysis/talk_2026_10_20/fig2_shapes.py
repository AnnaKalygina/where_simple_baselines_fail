"""Figure 2: response score vs residual PC1 for three automatically chosen perturbations."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import metrics as mx
import run_metrics as rm
from analyze import GRAY, NAVY, OUT, RED, TURQ

ESCAPER_KD = 0.8


def choose(m1):
    """The runbook's three examples, with the rule that selected each."""
    clean = m1[(m1.rho >= 0.5) & (m1.dBIC < 2)].sort_values("M", ascending=False)
    mix = m1[(m1.dBIC > 10) & (m1.dprime.abs() >= 1) & (m1.n_cells >= 150)].copy()
    mix["gap"] = (mix.pi_raw - 0.5).abs()
    non_esc = mix[~(mix.kd_non > ESCAPER_KD)]
    mix_pick = (non_esc if len(non_esc) else mix).sort_values("gap")
    unrel = m1[m1.rho < 0.5]
    flagged = unrel[(unrel["shape"] == "mixture") | unrel.widened]
    unrel_pick = (flagged if len(flagged) else unrel).sort_values("n_cells", ascending=False)
    return [("clean shift", clean.perturbation.iloc[0], "largest M among rho >= 0.5 with dBIC < 2"),
            ("mixture", mix_pick.perturbation.iloc[0],
             "pi closest to 0.5 among dBIC > 10, |d'| >= 1, n >= 150" + (", not escapers" if len(non_esc) else "")),
            ("unreliable, many cells", unrel_pick.perturbation.iloc[0],
             "largest n among rho < 0.5" + (" flagged mixture or widened" if len(flagged) else ""))]


def panel_data(name):
    """Scores and residual PC1 for half-B cells and K3 controls, reproducing split 0 of run_metrics."""
    i = int(np.flatnonzero(rm.D["perts"] == name)[0])
    rng = np.random.default_rng([rm.SEED, 100, i])
    rows = np.flatnonzero(rm.D["gene"] == name)
    x, _, _ = rm.cells_of(rows)
    ctx = rm.D["ctx"]
    genes = np.ones(len(rm.D["g0"]), bool)
    gi = rm.D["gene_index"].get(name)
    if gi is not None and gi in rm.D["g0_pos"]:
        genes[rm.D["g0_pos"][gi]] = False
    mx.magnitude_ci(x, ctx, genes, rng, 0.0)
    a, b = mx.split_halves(len(rows), rng)[0]
    _, _, k3 = ctx.thirds
    k1_mean = ctx.k1_means[0][genes]
    ya = x[a].mean(0)[genes] - k1_mean
    w = ya / np.linalg.norm(ya)
    rb = x[b][:, genes] - k1_mean
    r3 = ctx.x[k3][:, genes] - k1_mean
    sb, s3 = rb @ w, r3 @ w
    rb -= np.outer(sb, w)
    r3 -= np.outer(s3, w)
    center = rb.mean(0)
    _, _, vt = np.linalg.svd(rb - center, full_matrices=False)
    pc = vt[0]
    return sb, s3, (rb - center) @ pc, (r3 - center) @ pc


def norm_pdf(s, mu, sd):
    """Normal density."""
    return np.exp(mx.norm_logpdf(s, mu, sd))


def main():
    """Choose examples, draw the three panels, record the choice."""
    m1 = pd.read_csv(OUT / "M1_k562_essential.csv")
    picks = choose(m1)
    rm.setup()
    fig = plt.figure(figsize=(6, 4.5))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 3], hspace=0.05, wspace=0.35)
    record = []
    for k, (kind, name, rule) in enumerate(picks):
        r = m1.set_index("perturbation").loc[name]
        sb, s3, pb, p3 = panel_data(name)
        top = fig.add_subplot(gs[0, k])
        ax = fig.add_subplot(gs[1, k], sharex=top)
        ax.scatter(s3, p3, s=2, color=GRAY, alpha=0.3, linewidths=0, rasterized=True)
        ax.scatter(sb, pb, s=4, color=TURQ, alpha=0.7, linewidths=0, rasterized=True)
        lo, hi = np.percentile(np.concatenate([sb, s3]), [0.2, 99.8])
        grid = np.linspace(lo, hi, 300)
        bins = np.linspace(lo, hi, 40)
        top.hist(s3, bins=bins, density=True, color=GRAY, alpha=0.35)
        top.hist(sb, bins=bins, density=True, histtype="step", color=TURQ, lw=1.2)
        top.plot(grid, (1 - r.pi_raw) * norm_pdf(grid, r.mu0, r.sigma0), color=GRAY, lw=1.5)
        top.plot(grid, r.pi_raw * norm_pdf(grid, r.mu1, r.sigma1), color=RED, lw=1.5)
        top.set_title(name, fontsize=12, color=NAVY)
        top.axis("off")
        ax.set_xlim(lo, hi)
        ax.set_xlabel("response score\n(log1p units)", fontsize=12)
        if k == 0:
            ax.set_ylabel("residual PC1 (log1p units)", fontsize=12)
        ax.tick_params(labelsize=12)
        ax.spines[["top", "right"]].set_visible(False)
        record.append(dict(panel=k + 1, kind=kind, perturbation=name, rule=rule, n_cells=int(r.n_cells),
                           rho=float(r.rho), M=float(r.M), pi=None if pd.isna(r.pi) else float(r.pi),
                           pi_raw=float(r.pi_raw), dBIC=float(r.dBIC), dprime=float(r.dprime),
                           D=float(r.D), D_null95=float(r.D_null95), kd_non=None if pd.isna(r.kd_non) else float(r.kd_non),
                           shape=r["shape"]))
    fig.savefig(OUT / "fig2_shapes.png", dpi=300, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    (OUT / "fig2_examples.json").write_text(json.dumps(record, indent=1))
    print(json.dumps(record, indent=1))


if __name__ == "__main__":
    main()
