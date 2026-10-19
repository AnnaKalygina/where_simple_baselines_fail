"""Brief M3 per-gene table and figure T2-A (half-B vs K3 score histograms with the fitted mixture)."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import metrics as mx
import run_metrics as rm
from analyze import GRAY, NAVY, OUT, RED, TURQ

SCORES = rm.CACHE / "scores_halfB_K3_k562_essential.parquet"


def choose(m1):
    """Brief M3 defaults: highest M; pi closest to 0.5 among dBIC > 10; largest n among rho < 0.5."""
    hi = m1.sort_values("M", ascending=False).perturbation.iloc[0]
    mix = m1[m1.dBIC > 10].assign(gap=lambda d: (d.pi_raw - 0.5).abs()).sort_values("gap").perturbation.iloc[0]
    big = m1[m1.rho < 0.5].sort_values("n_cells", ascending=False).perturbation.iloc[0]
    return [("highest magnitude", hi), ("pi closest to 0.5, dBIC > 10", mix), ("largest n, rho < 0.5", big)]


def gene_stats(rows):
    """A', B' (1e6 scale), mean and variance of u (1e4 scale), and mean of x per G0 gene."""
    c = rm.D["counts"][rows][:, rm.D["g0"]].toarray().astype(np.float64)
    dep = rm.D["depth"][rows]
    x, u = mx.normalize(c, dep)
    return dict(A=1e6 * c.sum(0) / dep.sum(), B=(1e6 * c / dep[:, None]).mean(0),
                mu=u.mean(0), var=u.var(0, ddof=1), xbar=x.mean(0))


def m3_table(picks):
    """Long per-gene table for the controls and the three perturbations."""
    genes = rm.D["genes"][rm.D["g0"]].astype(str)
    ctrl = gene_stats(rm.D["ctrl"])
    parts = [pd.DataFrame({"set": "controls", "gene": genes, "A_prime": ctrl["A"], "B_prime": ctrl["B"],
                           "mu_1e4": ctrl["mu"], "var_1e4": ctrl["var"]})]
    for _, name in picks:
        p = gene_stats(np.flatnonzero(rm.D["gene"] == name))
        parts.append(pd.DataFrame({
            "set": name, "gene": genes, "A_prime": p["A"], "B_prime": p["B"], "mu_1e4": p["mu"], "var_1e4": p["var"],
            "delta_C": p["xbar"] - ctrl["xbar"],
            "logdiff": np.log1p(p["mu"]) - np.log1p(ctrl["mu"]),
            "varterm": 0.5 * (p["var"] / (1 + p["mu"]) ** 2 - ctrl["var"] / (1 + ctrl["mu"]) ** 2),
            "is_target": genes == name}))
    t = pd.concat(parts, ignore_index=True)
    t.to_csv(OUT / "M3_per_gene_k562_essential.csv.gz", index=False)
    return t


def plot_t2a(m1, picks):
    """Three histograms of half-B vs K3 scores with the fitted normal-null mixture."""
    sc = pd.read_parquet(SCORES, filters=[("perturbation", "in", [n for _, n in picks])])
    fig, axes = plt.subplots(1, 3, figsize=(6, 4.5))
    for ax, (_, name) in zip(axes, picks):
        r = m1.set_index("perturbation").loc[name]
        d = sc[sc.perturbation == name]
        sb, s3 = d.loc[d.cell_set == "half_B", "score"].values, d.loc[d.cell_set == "K3", "score"].values
        lo, hi = np.percentile(np.concatenate([sb, s3]), [0.2, 99.8])
        bins = np.linspace(lo, hi, 35)
        grid = np.linspace(lo, hi, 300)
        ax.hist(s3, bins=bins, density=True, color=GRAY, alpha=0.45, label="controls K3")
        ax.hist(sb, bins=bins, density=True, histtype="step", color=TURQ, lw=1.5, label="perturbed half B")
        ax.plot(grid, (1 - r.pi_raw) * np.exp(mx.norm_logpdf(grid, r.mu0, r.sigma0)), color=NAVY, lw=1.5)
        ax.plot(grid, r.pi_raw * np.exp(mx.norm_logpdf(grid, r.mu1, r.sigma1)), color=RED, lw=1.5)
        ax.set_title(name, fontsize=12, color=NAVY)
        ax.set_xlabel("score (log1p units)")
        ax.set_yticks([])
        ax.spines[["top", "right", "left"]].set_visible(False)
    axes[0].set_ylabel("density")
    fig.tight_layout()
    fig.savefig(OUT / "T2-A_response_shapes.png", dpi=300)
    plt.close(fig)


def main():
    """Pick, tabulate, plot, record."""
    m1 = pd.read_csv(OUT / "M1_k562_essential.csv")
    picks = choose(m1)
    rm.setup()
    m3_table(picks)
    plot_t2a(m1, picks)
    cols = ["n_cells", "M", "rho", "pi_raw", "pi", "dBIC", "dprime", "mu0", "sigma0", "mu1", "sigma1", "shape", "kd_non"]
    rec = [dict(rule=k, perturbation=n, **{c: (None if pd.isna(v) else (v if isinstance(v, str) else float(v)))
                                           for c, v in m1.set_index("perturbation").loc[n, cols].items()})
           for k, n in picks]
    (OUT / "M3_T2A_examples.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main()
