"""Null summary, S0 checks, Figure 1 and Figure 3 tables and PNGs from the M1/M2/S0 tables."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

OUT = Path(__file__).resolve().parents[2] / "results" / "talk_2026-10-20"
NAVY, RED, TURQ, GRAY = "#002060", "#C54746", "#4A9AA8", "#9AA3AD"
METRICS = ["M", "tau2", "pi", "D"]
LABELS = {"M": "M", "tau2": "τ²", "pi": "π", "D": "D", "rho": "ρ", "n_cells": "n cells"}
MIN_N_FIG1 = 150
ESCAPER_KD = 0.8

plt.rcParams.update({"font.family": "sans-serif", "font.size": 12, "text.color": NAVY, "axes.labelcolor": NAVY,
                     "axes.edgecolor": NAVY, "xtick.color": NAVY, "ytick.color": NAVY,
                     "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white"})


def spearman(a, b):
    """Spearman rho and n over pairs where both are finite."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return np.nan, int(ok.sum())
    return float(stats.spearmanr(a[ok], b[ok])[0]), int(ok.sum())


def null_summary(m2):
    """Distribution of every numeric M2 column plus the runbook's expected-value checks."""
    num = m2.select_dtypes("number")
    q = num.quantile([0.05, 0.5, 0.95]).T
    q.columns = ["p05", "median", "p95"]
    q.to_csv(OUT / "M2_null_summary.csv")
    return dict(n=len(m2), M_median=float(m2.M.median()),
                frac_M_ci_contains_0=float(((m2.M_ci_low <= 0) & (m2.M_ci_high >= 0)).mean()),
                rho_median=float(m2.rho.median()), frac_rho_ge_0_5=float((m2.rho >= 0.5).mean()),
                frac_dBIC_gt_10=float((m2.dBIC > 10).mean()), frac_widened=float(m2.widened.mean()),
                frac_pi_identified=float(m2.pi.notna().mean()))


def fig1_table(m1):
    """Runbook Figure 1 percentages per reliability group, n >= 150."""
    pop = m1[m1.n_cells >= MIN_N_FIG1].copy()
    pop["group"] = np.where(pop.rho >= 0.5, "reliable", "unreliable")
    pop["mixture"] = pop["shape"] == "mixture"
    pop["no_effect"] = (pop.M_ci_low <= 0) & (pop.M_ci_high >= 0)
    pop["mixture_escaper"] = pop.mixture & (pop.kd_non > ESCAPER_KD)
    pop["mixture_not_escaper"] = pop.mixture & (pop.kd_non <= ESCAPER_KD)
    rows = []
    for g, d in pop.groupby("group"):
        rows.append(dict(group=g, n=len(d), pct_mixture=100 * d.mixture.mean(), pct_widened=100 * d.widened.mean(),
                         pct_no_effect=100 * d.no_effect.mean(), pct_mixture_escaper=100 * d.mixture_escaper.mean(),
                         pct_mixture_not_escaper=100 * d.mixture_not_escaper.mean(),
                         n_mixture_escaper_eligible=int((d.mixture & d.escaper_eligible).sum()),
                         tau2_median=float(d.tau2.median()), n_tau2_defined=int(d.tau2.notna().sum())))
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "fig1_filtering_table.csv", index=False)
    return t, pop


def f3_overlaps(m1):
    """Brief F3: flags among n >= 150 and rho < 0.5, with overlap counts."""
    d = m1[(m1.n_cells >= 150) & (m1.rho < 0.5)]
    a = (d.M_ci_low <= 0) & (d.M_ci_high >= 0)
    b = d.D > d.D_null95
    c = (d.dBIC > 10) & (d.pi_raw > 0.1) & (d.pi_raw < 0.9)
    combos = {"n": len(d), "a_ci_contains_0": int(a.sum()), "b_widened": int(b.sum()), "c_mixture_brief": int(c.sum()),
              "a_and_b": int((a & b).sum()), "a_and_c": int((a & c).sum()), "b_and_c": int((b & c).sum()),
              "a_b_c": int((a & b & c).sum()), "none": int((~a & ~b & ~c).sum())}
    pd.Series(combos).to_csv(OUT / "F3_flags_overlap.csv", header=["count"])
    flags = d[["perturbation", "n_cells", "rho"]].assign(a_ci_contains_0=a, b_widened=b, c_mixture_brief=c)
    flags.to_csv(OUT / "F3_flags_per_perturbation.csv", index=False)
    return combos


def s0_checks(m1, s0):
    """S0-a stability, S0-b downsampling, S0-c Spearman matrix and R2 on rank of M."""
    full = m1.set_index("perturbation")
    s0 = s0.set_index("perturbation")
    rows = []
    for d in METRICS + ["rho"]:
        r, n = spearman(s0[f"{d}_half_a"], s0[f"{d}_half_b"])
        sb = 2 * r / (1 + r) if np.isfinite(r) else np.nan
        big = s0[s0.n_cells >= 200]
        f_vals = full.loc[big.index, d]
        r_fd, n_fd = spearman(f_vals, big[f"{d}_down"])
        r_before, _ = spearman(f_vals, big.n_cells)
        r_after, _ = spearman(big[f"{d}_down"], big.n_cells)
        r_all, _ = spearman(full[d], full.n_cells)
        rows.append(dict(metric=d, S0a_spearman_halves=r, S0a_spearman_brown=sb, S0a_n=n,
                         S0b_spearman_full_vs_down=r_fd, S0b_n=n_fd, S0b_corr_n_before=r_before,
                         S0b_corr_n_after=r_after, corr_n_all_perts=r_all))
    s0t = pd.DataFrame(rows)
    big = s0[s0.n_cells >= 200]
    mat_cols = {d: big[f"{d}_down"] for d in METRICS + ["rho"]}
    mat_cols["n_cells"] = big.n_cells
    mat = pd.DataFrame(mat_cols)
    names = list(mat.columns)
    sm = pd.DataFrame(np.eye(len(names)), index=names, columns=names)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            sm.loc[a, b] = sm.loc[b, a] = spearman(mat[a], mat[b])[0]
    sm.to_csv(OUT / "S0c_spearman_matrix_downsampled.csv")
    rank_m = mat["M"].rank()
    r2 = []
    for c in names:
        ok = np.isfinite(mat[c]) & np.isfinite(rank_m)
        r2.append(dict(metric=c, R2_on_rank_M=float(stats.pearsonr(mat[c][ok], rank_m[ok])[0] ** 2),
                       R2_rank_on_rank_M=float(spearman(mat[c], rank_m)[0] ** 2), n=int(ok.sum())))
    r2 = pd.DataFrame(r2)
    s0t = s0t.merge(r2.rename(columns={"n": "S0c_n"}), on="metric", how="left")
    s0t.to_csv(OUT / "S0_checks.csv", index=False)
    return s0t, sm


def plot_fig1(t, pop):
    """Bars per reliability group and tau2 by group."""
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(6, 4.5), gridspec_kw={"width_ratios": [1.9, 1]})
    cats = [("pct_mixture", "mixture"), ("pct_widened", "widened"), ("pct_no_effect", "no effect"),
            ("pct_mixture_escaper", "mixture,\nescapers")]
    t = t.set_index("group")
    x = np.arange(len(cats))
    for k, (g, col) in enumerate([("reliable", TURQ), ("unreliable", RED)]):
        if g in t.index:
            ax.bar(x + (k - 0.5) * 0.38, [t.loc[g, c] for c, _ in cats], 0.38, color=col,
                   label=f"{g} (n = {int(t.loc[g, 'n'])})")
    ax.set_xticks(x, [l for _, l in cats], fontsize=12, rotation=35, ha="right")
    ax.set_ylabel("perturbations (%)")
    ax.legend(frameon=False, fontsize=12, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    data, pos = [], []
    for k, g in enumerate(["reliable", "unreliable"]):
        v = pop.loc[(pop.group == g) & (pop.tau2 > 0), "tau2"].values
        data.append(v)
        pos.append(k)
        jitter = np.random.default_rng(20261020).uniform(-0.15, 0.15, len(v))
        bx.scatter(k + jitter, v, s=6, color=[TURQ, RED][k], alpha=0.5, linewidths=0)
    bx.boxplot(data, positions=pos, widths=0.5, showfliers=False,
               medianprops={"color": NAVY}, boxprops={"color": NAVY}, whiskerprops={"color": NAVY},
               capprops={"color": NAVY})
    bx.set_yscale("log")
    bx.set_xticks(pos, ["reliable", "unreliable"], rotation=35, ha="right")
    bx.set_ylabel("τ² (per-cell signal / noise)")
    bx.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_filtering.png", dpi=300)
    plt.close(fig)


def plot_fig3(s0t, sm):
    """Spearman heatmap of the downsampled metrics and the cell-count / stability bars."""
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(6, 4.5), gridspec_kw={"width_ratios": [1.25, 1]})
    names = list(sm.index)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("nr", [TURQ, "white", RED])
    ax.imshow(sm.values, cmap=cmap, vmin=-1, vmax=1)
    for i in range(len(names)):
        for j in range(len(names)):
            v = sm.values[i, j]
            ax.text(j, i, "" if not np.isfinite(v) else f"{v:.2f}".replace("0.", ".").replace("-.", "−."),
                    ha="center", va="center", fontsize=9 if len(names) > 5 else 12, color=NAVY)
    lab = [LABELS[n] for n in names]
    ax.set_xticks(range(len(names)), lab, rotation=90)
    ax.set_yticks(range(len(names)), lab)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    t = s0t.set_index("metric").loc[METRICS]
    y = np.arange(len(METRICS))
    h = 0.26
    bx.barh(y - h, t.S0b_corr_n_before.abs(), h, color=GRAY, label="|corr(d, n)| full")
    bx.barh(y, t.S0b_corr_n_after.abs(), h, color=RED, label="|corr(d, n)| at 100 cells")
    bx.barh(y + h, t.S0a_spearman_brown, h, color=TURQ, label="stability (S0-a)")
    bx.set_yticks(y, [LABELS[m] for m in METRICS])
    bx.invert_yaxis()
    bx.set_xlim(0, 1)
    bx.set_xlabel("Spearman (unitless)")
    bx.legend(frameon=False, fontsize=9, loc="lower right", bbox_to_anchor=(1.0, 1.0))
    bx.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "fig3_metrics.png", dpi=300)
    plt.close(fig)


def main():
    """Run every summary and write tables, figures and a headline JSON."""
    m1 = pd.read_csv(OUT / "M1_k562_essential.csv")
    m2 = pd.read_csv(OUT / "M2_null_k562_essential.csv")
    s0 = pd.read_csv(OUT / "S0_inputs_k562_essential.csv")
    head = {"null": null_summary(m2)}
    t, pop = fig1_table(m1)
    head["fig1"] = t.to_dict("records")
    head["F3"] = f3_overlaps(m1)
    s0t, sm = s0_checks(m1, s0)
    head["S0"] = s0t.to_dict("records")
    head["tau2_vs_direct"] = spearman(m1.tau2, m1.tau2_direct)
    head["frac_rho_ge_0_5_all"] = float((m1.rho >= 0.5).mean())
    head["shape_counts"] = m1["shape"].value_counts().to_dict()
    plot_fig1(t, pop)
    plot_fig3(s0t, sm)
    (OUT / "headline_numbers.json").write_text(json.dumps(head, indent=1, default=float))
    print(json.dumps(head, indent=1, default=float))


if __name__ == "__main__":
    main()
