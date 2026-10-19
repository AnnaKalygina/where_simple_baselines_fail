"""Sensitivity of the mixture calls to the normal null: component 0 as the K3 kernel density instead of a normal."""
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy import optimize
from scipy.stats import gaussian_kde

import metrics as mx
from analyze import OUT, ESCAPER_KD, MIN_N_FIG1

SCORES = "/cluster/work/boeva/akalygina/wsb13_cache/scores_halfB_K3_k562_essential.parquet"
SEED = 20261020
LOG_FLOOR = -700.0
G = {}


def log_kde(kde, s):
    """Log kernel density with a floor far below any observed value."""
    return np.maximum(np.log(np.maximum(kde(s), 1e-300)), LOG_FLOOR)


def em_fixed_null(s, lp0, rng, sd_ref):
    """EM for (1 - pi) f0 + pi N(mu1, sd1) with log f0 given; best of N_EM_STARTS starts."""
    best = None
    for _ in range(mx.N_EM_STARTS):
        pi, mu1 = rng.uniform(0.05, 0.95), rng.choice(s)
        sd1 = max(s.std() * rng.uniform(0.5, 1.5), 1e-3 * sd_ref)
        ll_old = -np.inf
        for _ in range(1000):
            a = np.log1p(-pi) + lp0
            b = np.log(pi) + mx.norm_logpdf(s, mu1, sd1)
            m = np.logaddexp(a, b)
            ll = float(m.sum())
            g = np.exp(b - m)
            gs = g.sum()
            if gs <= 1e-12 or gs >= len(s) - 1e-12:
                pi = float(np.clip(gs / len(s), 1e-9, 1 - 1e-9))
                break
            pi, mu1 = gs / len(s), float((g * s).sum() / gs)
            sd1 = max(float(np.sqrt((g * (s - mu1) ** 2).sum() / gs)), 1e-3 * sd_ref)
            if ll - ll_old < 1e-9 * abs(ll):
                break
            ll_old = ll
        ll = float(np.logaddexp(np.log1p(-pi) + lp0, np.log(pi) + mx.norm_logpdf(s, mu1, sd1)).sum())
        if best is None or ll > best[0]:
            best = (ll, pi, mu1, sd1)
    return best


def one(name):
    """Empirical-null mixture, shifted-null and single-normal fits for one perturbation."""
    i, sb, s0 = G["groups"][name]
    rng = np.random.default_rng([SEED, 400, i])
    kde = gaussian_kde(s0)
    mu0, sd0 = s0.mean(), s0.std(ddof=1)
    ll_mix, pi, mu1, sd1 = em_fixed_null(sb, log_kde(kde, sb), rng, sd0)
    span = np.ptp(np.concatenate([sb, s0]))
    res = optimize.minimize_scalar(lambda d: -log_kde(kde, sb - d).sum(),
                                   bounds=(-span, span), method="bounded")
    ll_shift = -float(res.fun)
    ll1 = float(mx.norm_logpdf(sb, sb.mean(), sb.std()).sum())
    n = len(sb)
    bic_mix = -2 * ll_mix + 3 * np.log(n)
    return dict(perturbation=name, emp_pi=pi, emp_mu1=mu1, emp_sigma1=sd1, emp_dprime=(mu1 - mu0) / sd0,
                emp_dBIC_vs_normal=(-2 * ll1 + 2 * np.log(n)) - bic_mix,
                emp_dBIC_vs_shift=(-2 * ll_shift + np.log(n)) - bic_mix, emp_shift=float(res.x))


def main():
    """Fit every perturbation, write the table and the Figure 1 percentages under the robust rule."""
    workers = int(sys.argv[1])
    m1 = pd.read_csv(OUT / "M1_k562_essential.csv")
    sc = pd.read_parquet(SCORES)
    idx = {p: k for k, p in enumerate(m1.perturbation)}
    G["groups"] = {p: (idx[p], d.loc[d.cell_set == "half_B", "score"].to_numpy(np.float64),
                       d.loc[d.cell_set == "K3", "score"].to_numpy(np.float64)) for p, d in sc.groupby("perturbation")}
    with Pool(workers) as pool:
        rows = pool.map(one, list(m1.perturbation), chunksize=8)
    t = m1[["perturbation", "n_cells", "rho", "shape", "kd_non", "null_skew"]].merge(pd.DataFrame(rows), on="perturbation")
    t["mixture_runbook"] = t["shape"] == "mixture"
    t["mixture_empirical_null"] = ((t.emp_pi > 0.1) & (t.emp_pi < 0.9) & (t.emp_dprime.abs() >= mx.DPRIME_MIN)
                                   & (t.emp_dBIC_vs_normal > mx.DBIC_MIN) & (t.emp_dBIC_vs_shift > mx.DBIC_MIN))
    t.to_csv(OUT / "mixture_sensitivity_k562_essential.csv", index=False)
    pop = t[t.n_cells >= MIN_N_FIG1].assign(group=lambda d: np.where(d.rho >= 0.5, "reliable", "unreliable"))
    summ = pop.groupby("group").agg(n=("perturbation", "size"), pct_mixture_runbook=("mixture_runbook", "mean"),
                                    pct_mixture_empirical_null=("mixture_empirical_null", "mean"))
    summ[["pct_mixture_runbook", "pct_mixture_empirical_null"]] *= 100
    esc = pop[pop.mixture_empirical_null]
    summ["pct_mixture_empirical_null_escaper"] = 100 * esc.groupby("group").apply(lambda d: (d.kd_non > ESCAPER_KD).sum()) / summ.n
    summ["n_runbook_mixtures_kept"] = pop[pop.mixture_runbook].groupby("group").mixture_empirical_null.sum()
    summ.to_csv(OUT / "mixture_sensitivity_fig1_groups.csv")
    print(summ.round(2).to_string())
    print("all perts: runbook mixtures", int(t.mixture_runbook.sum()), "empirical-null mixtures", int(t.mixture_empirical_null.sum()),
          "both", int((t.mixture_runbook & t.mixture_empirical_null).sum()))


if __name__ == "__main__":
    main()
