"""Brief F1 (reliability and tau2 per cell-count bin) and F3 overlaps as figures T1-B and T1-C."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyze import GRAY, NAVY, OUT, RED, TURQ

BINS = [0, 50, 100, 150, 250, 500, np.inf]
BIN_LABELS = ["<50", "50–99", "100–149", "150–249", "250–499", "≥500"]


def f1_table(m1):
    """Fraction with rho >= 0.5 and tau2 quantiles per n_cells bin."""
    m1 = m1.assign(bin=pd.cut(m1.n_cells, BINS, right=False, labels=BIN_LABELS))
    t = m1.groupby("bin", observed=False).agg(
        n=("perturbation", "size"), frac_rho_ge_0_5=("rho", lambda r: float((r >= 0.5).mean())),
        n_tau2_defined=("tau2", lambda v: int(v.notna().sum())),
        tau2_p25=("tau2", lambda v: v.quantile(0.25)), tau2_median=("tau2", "median"),
        tau2_p75=("tau2", lambda v: v.quantile(0.75)))
    t.to_csv(OUT / "F1_reliability_by_cell_bin.csv")
    return t, m1


def plot_t1b(t, m1):
    """Bars of the reliable fraction per bin and tau2 boxes per bin."""
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(6, 4.5), sharex=True, gridspec_kw={"height_ratios": [1, 1.2]})
    x = np.arange(len(BIN_LABELS))
    ax.bar(x, 100 * t.frac_rho_ge_0_5.values, color=RED, width=0.65)
    for k, n in enumerate(t.n.values):
        ax.text(k, 100 * t.frac_rho_ge_0_5.values[k] + 2, f"n={n}", ha="center", va="bottom", fontsize=12, color=NAVY)
    ax.set_ylim(0, 100)
    ax.set_ylabel("ρ ≥ 0.5 (%)")
    data = [m1.loc[(m1.bin == b) & (m1.tau2 > 0), "tau2"].values for b in BIN_LABELS]
    bx.boxplot(data, positions=x, widths=0.55, showfliers=False, medianprops={"color": RED},
               boxprops={"color": NAVY}, whiskerprops={"color": NAVY}, capprops={"color": NAVY})
    bx.set_yscale("log")
    bx.set_ylabel("τ²")
    bx.set_xticks(x, BIN_LABELS)
    bx.set_xlabel("cells per perturbation")
    for a in (ax, bx):
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "T1-B_reliability_by_cell_bin.png", dpi=300)
    plt.close(fig)


def plot_t1c():
    """Exclusive overlap counts of the three F3 flags."""
    f = pd.read_csv(OUT / "F3_flags_per_perturbation.csv")
    a, b, c = f.a_ci_contains_0, f.b_widened, f.c_mixture_brief
    cats = [("no effect only", a & ~b & ~c), ("widened only", ~a & b & ~c), ("mixture only", ~a & ~b & c),
            ("no effect + widened", a & b & ~c), ("no effect + mixture", a & ~b & c),
            ("widened + mixture", ~a & b & c), ("all three", a & b & c), ("none", ~a & ~b & ~c)]
    counts = [int(v.sum()) for _, v in cats]
    pd.DataFrame({"category": [k for k, _ in cats], "count": counts}).to_csv(OUT / "T1-C_F3_exclusive_counts.csv", index=False)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    y = np.arange(len(cats))
    colors = [TURQ if k != "none" else GRAY for k, _ in cats]
    ax.barh(y, counts, color=colors)
    for k, v in enumerate(counts):
        ax.text(v + 2, k, str(v), va="center", fontsize=12, color=NAVY)
    ax.set_yticks(y, [k for k, _ in cats])
    ax.invert_yaxis()
    ax.set_xlabel(f"perturbations (n ≥ 150, ρ < 0.5; total {len(f)})")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "T1-C_F3_flags.png", dpi=300)
    plt.close(fig)
    return dict(zip([k for k, _ in cats], counts))


def main():
    """Write F1 and F3 figures and tables."""
    m1 = pd.read_csv(OUT / "M1_k562_essential.csv")
    t, m1b = f1_table(m1)
    plot_t1b(t, m1b)
    print(t.round(4).to_string())
    print(plot_t1c())


if __name__ == "__main__":
    main()
