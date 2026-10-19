"""Intrinsic per-perturbation metrics (M, tau2, pi, D) and the reliability filter, as in the 20.10 runbook."""
import numpy as np
from scipy import stats
from statsmodels.nonparametric.smoothers_lowess import lowess

KAPPA = 1e4
N_SPLITS = 100
N_BOOT = 100
N_DNULL = 200
N_EM_STARTS = 10
DBIC_MIN = 10.0
DPRIME_MIN = 1.0


def normalize(counts, depth):
    """Return (x, u): log1p(kappa c / L) and kappa c / L for a dense count block."""
    u = KAPPA * counts / depth[:, None]
    return np.log1p(u), u


def mean_var(a):
    """Column means and sample variances (ddof=1)."""
    m = a.mean(0)
    return m, a.var(0, ddof=1)


def dispersion(u, inv_depth_mean):
    """Moment estimate (mu, phi) per gene on u = kappa c / L."""
    mu, v = mean_var(u)
    with np.errstate(divide="ignore", invalid="ignore"):
        phi = (v - mu * inv_depth_mean) / mu ** 2
    return mu, phi


def dispersion_from_sums(s1, s2, n, inv_depth_mean):
    """Same as dispersion() from column sums of u and u**2."""
    mu = s1 / n
    v = (s2 - n * mu ** 2) / (n - 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        phi = (v - mu * inv_depth_mean) / mu ** 2
    return mu, phi


def fit_trend(mu, phi, genes):
    """Lowess trend of log phi on log mu over genes with phi > 0 and mu >= 0.1; returns (x, y) knots."""
    ok = genes & (phi > 0) & (mu >= 0.1)
    lx, ly = np.log(mu[ok]), np.log(phi[ok])
    fit = lowess(ly, lx, delta=0.01 * (lx.max() - lx.min()), return_sorted=True)
    return fit[:, 0], fit[:, 1]


def d_statistic(mu, phi, genes, trend):
    """Median over eligible genes of log phi minus the control trend at log mu."""
    ok = genes & (phi > 0) & (mu >= 0.1)
    if not ok.any():
        return np.nan
    lx = np.log(mu[ok])
    return float(np.median(np.log(phi[ok]) - np.interp(lx, trend[0], trend[1])))


class ControlContext:
    """Control-side quantities shared by all perturbations scored against one control pool."""

    def __init__(self, xc, uc, depth, seed):
        rng = np.random.default_rng(seed)
        self.x, self.u, self.depth = xc, uc, depth
        self.n = xc.shape[0]
        self.mean, self.var = mean_var(xc)
        self.inv_depth = KAPPA / depth
        self.u_sum = uc.sum(0)
        self.u2_sum = (uc ** 2).sum(0)
        mu, phi = dispersion(uc, self.inv_depth.mean())
        self.mu, self.phi = mu, phi
        k1_means, k2_means = np.empty((N_SPLITS, xc.shape[1])), np.empty((N_SPLITS, xc.shape[1]))
        for r in range(N_SPLITS):
            perm = rng.permutation(self.n)
            k1, k2, k3 = np.array_split(perm, 3)
            if r == 0:
                self.thirds = (k1, k2, k3)
            k1_means[r], k2_means[r] = xc[k1].mean(0), xc[k2].mean(0)
        self.k1_means, self.k2_means = k1_means, k2_means
        w = rng.multinomial(self.n, np.full(self.n, 1.0 / self.n), size=N_BOOT).astype(np.float64)
        m1 = w @ xc / self.n
        m2 = w @ (xc ** 2) / self.n
        self.boot_mean = m1
        self.boot_var = (m2 - m1 ** 2) * self.n / (self.n - 1)

    def trend(self, genes):
        """Lowess dispersion trend on this pool."""
        return fit_trend(self.mu, self.phi, genes)

    def sub(self, idx, seed):
        """Context restricted to a subset of control rows."""
        return ControlContext(self.x[idx], self.u[idx], self.depth[idx], seed)


def magnitude(xp, ctx, genes):
    """Debiased magnitude M, the effect y and the squared standard errors over genes."""
    n = xp.shape[0]
    mp, vp = mean_var(xp)
    y = mp - ctx.mean
    se2 = vp / n + ctx.var / ctx.n
    return float(np.sum((y ** 2 - se2)[genes])), y, se2, vp


def magnitude_ci(xp, ctx, genes, rng, m):
    """Bootstrap interval of M: literal percentiles of M_b, and M plus the centered percentiles of M_b."""
    n = xp.shape[0]
    w = rng.multinomial(n, np.full(n, 1.0 / n), size=N_BOOT).astype(np.float64)
    m1 = w @ xp / n
    vp = (w @ (xp ** 2) / n - m1 ** 2) * n / (n - 1)
    y = m1 - ctx.boot_mean
    se2 = vp / n + ctx.boot_var / ctx.n
    mb = np.sum((y ** 2 - se2)[:, genes], axis=1)
    lit = np.percentile(mb, [2.5, 97.5])
    cen = m + np.percentile(mb - mb.mean(), [2.5, 97.5])
    return float(cen[0]), float(cen[1]), float(lit[0]), float(lit[1])


def split_halves(n, rng):
    """N_SPLITS random (A, B) halves of n perturbed cells."""
    out = []
    for _ in range(N_SPLITS):
        perm = rng.permutation(n)
        out.append((perm[: n // 2], perm[n // 2:]))
    return out


def reliability(xp, ctx, genes, halves):
    """Median split-half Pearson r_half over splits, its Spearman-Brown rho, and split-0 effects."""
    rs = np.empty(len(halves))
    for r, (a, b) in enumerate(halves):
        ya = xp[a].mean(0)[genes] - ctx.k1_means[r][genes]
        yb = xp[b].mean(0)[genes] - ctx.k2_means[r][genes]
        rs[r] = np.corrcoef(ya, yb)[0, 1]
    r_half = float(np.median(rs))
    rho = 2 * r_half / (1 + r_half)
    return r_half, rho


def tau2_from_rho(rho, n):
    """Wang et al. estimator rho / (n (1 - rho)), empty outside (0, 1)."""
    return rho / (n * (1 - rho)) if 0 < rho < 1 else np.nan


def norm_logpdf(s, mu, sd):
    """Normal log density."""
    return -0.5 * np.log(2 * np.pi * sd ** 2) - 0.5 * ((s - mu) / sd) ** 2


def fit_mixture(s, mu0, sd0, rng):
    """EM for (1 - pi) N(mu0, sd0) + pi N(mu1, sd1) with component 0 fixed; best of N_EM_STARTS."""
    lp0 = norm_logpdf(s, mu0, sd0)
    sd_floor = 1e-3 * sd0
    best = None
    for _ in range(N_EM_STARTS):
        pi = rng.uniform(0.05, 0.95)
        mu1 = rng.choice(s)
        sd1 = max(s.std() * rng.uniform(0.5, 1.5), sd_floor)
        ll_old = -np.inf
        for _ in range(1000):
            a = np.log1p(-pi) + lp0
            b = np.log(pi) + norm_logpdf(s, mu1, sd1)
            m = np.logaddexp(a, b)
            ll = float(m.sum())
            g = np.exp(b - m)
            gs = g.sum()
            if gs <= 1e-12 or gs >= len(s) - 1e-12:
                pi = float(np.clip(gs / len(s), 1e-9, 1 - 1e-9))
                break
            pi = gs / len(s)
            mu1 = float((g * s).sum() / gs)
            sd1 = max(float(np.sqrt((g * (s - mu1) ** 2).sum() / gs)), sd_floor)
            if ll - ll_old < 1e-9 * abs(ll):
                break
            ll_old = ll
        a = np.log1p(-pi) + lp0
        b = np.log(pi) + norm_logpdf(s, mu1, sd1)
        ll = float(np.logaddexp(a, b).sum())
        if best is None or ll > best[0]:
            best = (ll, pi, mu1, sd1)
    return best


def responders(xp, ctx, genes, halves, rng):
    """Responder-fraction model on split 0: direction from half A, scores on half B and K3."""
    a, b = halves[0]
    k1, _, k3 = ctx.thirds
    k1_mean = ctx.k1_means[0][genes]
    ya = xp[a].mean(0)[genes] - k1_mean
    w = ya / np.linalg.norm(ya)
    sb = (xp[b][:, genes] - k1_mean) @ w
    s0 = (ctx.x[k3][:, genes] - k1_mean) @ w
    mu0, sd0 = float(s0.mean()), float(s0.std(ddof=1))
    ll_mix, pi, mu1, sd1 = fit_mixture(sb, mu0, sd0, rng)
    n = len(sb)
    ll1 = float(norm_logpdf(sb, sb.mean(), sb.std()).sum())
    dbic = (-2 * ll1 + 2 * np.log(n)) - (-2 * ll_mix + 3 * np.log(n))
    dprime = (mu1 - mu0) / sd0
    post = np.exp(np.log(pi) + norm_logpdf(sb, mu1, sd1)
                  - np.logaddexp(np.log1p(-pi) + norm_logpdf(sb, mu0, sd0), np.log(pi) + norm_logpdf(sb, mu1, sd1)))
    return dict(pi_raw=pi, mu0=mu0, sigma0=sd0, mu1=mu1, sigma1=sd1, dprime=dprime, dBIC=dbic,
                null_skew=float(stats.skew(s0)), mean_shift_s=float(sb.mean() - mu0),
                pi_times_shift=pi * (mu1 - mu0)), sb, s0, post, b, k3, w


def identified_pi(r):
    """Reported pi: only when dBIC > 10 and |d'| >= 1."""
    return r["pi_raw"] if (r["dBIC"] > DBIC_MIN and abs(r["dprime"]) >= DPRIME_MIN) else np.nan


def shape_label(pi, ci_low, ci_high):
    """mixture / edge / shift / none from the identified pi and the M interval."""
    if not np.isnan(pi):
        return "mixture" if 0.1 < pi < 0.9 else "edge"
    if np.isnan(ci_low):
        return ""
    return "none" if ci_low <= 0 <= ci_high else "shift"


def dispersion_shift(up, depth_p, trend, genes):
    """D for one perturbed cell set against a control trend."""
    mu, phi = dispersion(up, (KAPPA / depth_p).mean())
    return d_statistic(mu, phi, genes, trend)


def dispersion_null(ctx, n_p, genes, rng):
    """95th percentile of D over N_DNULL control subsets of size n_p, each against the remaining controls."""
    vals = np.empty(N_DNULL)
    inv = ctx.inv_depth
    for k in range(N_DNULL):
        idx = rng.choice(ctx.n, n_p, replace=False)
        us = ctx.u[idx]
        s1, s2 = us.sum(0), (us ** 2).sum(0)
        mask = np.ones(ctx.n, bool)
        mask[idx] = False
        rest = ctx.n - n_p
        mu_r, phi_r = dispersion_from_sums(ctx.u_sum - s1, ctx.u2_sum - s2, rest, inv[mask].mean())
        trend = fit_trend(mu_r, phi_r, genes)
        mu_s, phi_s = dispersion_from_sums(s1, s2, n_p, inv[idx].mean())
        vals[k] = d_statistic(mu_s, phi_s, genes, trend)
    return float(np.nanpercentile(vals, 95))


def lite_metrics(xp, up, depth_p, ctx, genes, trend, rng):
    """M, tau2, pi, D and rho only, for the S0 checks."""
    n = xp.shape[0]
    m, *_ = magnitude(xp, ctx, genes)
    halves = split_halves(n, rng)
    r_half, rho = reliability(xp, ctx, genes, halves)
    r, *_ = responders(xp, ctx, genes, halves, rng)
    return dict(M=m, tau2=tau2_from_rho(rho, n), pi=identified_pi(r), pi_raw=r["pi_raw"],
                D=dispersion_shift(up, depth_p, trend, genes), rho=rho)


def full_metrics(xp, up, depth_p, ctx, genes, trend, rng, target_u=None, ctrl_target_u=None):
    """Every M1 column for one perturbed cell set, plus its per-cell scores."""
    n = xp.shape[0]
    m, y, se2, vp = magnitude(xp, ctx, genes)
    lo, hi, lo_lit, hi_lit = magnitude_ci(xp, ctx, genes, rng, m)
    halves = split_halves(n, rng)
    r_half, rho = reliability(xp, ctx, genes, halves)
    yg, se2g = y[genes], se2[genes]
    tau2_direct = (yg.var() - se2g.mean()) / vp[genes].mean()
    r, sb, s0, post, b_idx, k3_idx, _ = responders(xp, ctx, genes, halves, rng)
    pi = identified_pi(r)
    kd_resp = kd_non = np.nan
    if target_u is not None and ctrl_target_u is not None:
        resp = post > 0.5
        base = ctrl_target_u.mean()
        if resp.any():
            kd_resp = float(target_u[b_idx][resp].mean() / base)
        if (~resp).any():
            kd_non = float(target_u[b_idx][~resp].mean() / base)
    d = dispersion_shift(up, depth_p, trend, genes)
    d95 = dispersion_null(ctx, n, genes, rng)
    row = dict(n_cells=n, median_depth=float(np.median(depth_p)), M=m, M_ci_low=lo, M_ci_high=hi,
               M_ci_low_literal=lo_lit, M_ci_high_literal=hi_lit,
               r_half=r_half, rho=rho, tau2=tau2_from_rho(rho, n), tau2_direct=float(tau2_direct),
               pi=pi, **r, shape=shape_label(pi, lo, hi), kd_resp=kd_resp, kd_non=kd_non,
               D=d, D_null95=d95, widened=bool(d > d95))
    return row, sb, s0, k3_idx
