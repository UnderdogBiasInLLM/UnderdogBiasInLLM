"""Estimators used in the paper.

All regressions are weighted least squares with standard errors clustered by the
inference cluster of each setting, using the G/(G-1) small-sample correction and
two-sided t(G-1) tests (Appendix A.1). Estimates are returned as proportions;
tables multiply by 100.
"""
import numpy as np
import pandas as pd
from scipy import stats


def _summary(beta, cov, groups):
    se = np.sqrt(np.maximum(np.diag(cov), 0))
    t = np.divide(beta, se, out=np.zeros_like(beta), where=se > 0)
    p = 2 * stats.t.sf(np.abs(t), groups - 1)
    return se, p


def cluster_ols(y, x, weight, cluster, centered=True):
    """Weighted least squares with cluster-robust covariance.

    Returns a dict with coefficients, covariance, standard errors, p-values,
    number of clusters, degrees of freedom (G-1), and R^2 (centered or, for
    within-transformed data, uncentered).
    """
    y = np.asarray(y, float)
    x = np.asarray(x, float)
    w = np.asarray(weight, float)
    codes, uniques = pd.factorize(np.asarray(cluster), sort=True)
    g = len(uniques)
    bread = np.linalg.inv(x.T @ (w[:, None] * x))
    beta = bread @ (x.T @ (w * y))
    resid = y - x @ beta
    scores = np.zeros((g, x.shape[1]))
    np.add.at(scores, codes, x * (w * resid)[:, None])
    cov = g / (g - 1) * bread @ (scores.T @ scores) @ bread
    se, p = _summary(beta, cov, g)
    total = y - np.average(y, weights=w) if centered else y
    sst = np.dot(w, total ** 2)
    r2 = 1 - np.dot(w, resid ** 2) / sst if sst else np.nan
    return {'beta': beta, 'cov': cov, 'se': se, 'p': p, 'clusters': g, 'df': g - 1,
            'nobs': len(y), 'r2': r2}


def contrast(fit, vector):
    """Linear combination of coefficients with its cluster-robust SE and t(G-1) p-value."""
    v = np.asarray(vector, float)
    est = float(v @ fit['beta'])
    se = float(np.sqrt(max(v @ fit['cov'] @ v, 0)))
    p = float(2 * stats.t.sf(abs(est / se), fit['df'])) if se > 0 else np.nan
    return {'est': est, 'se': se, 'p': p, 'df': fit['df']}


def mean_estimate(values, weight, cluster):
    """Weighted mean (or mean paired difference) with cluster-robust SE."""
    fit = cluster_ols(values, np.ones((len(values), 1)), weight, cluster)
    return contrast(fit, [1.0])


# ---------------------------------------------------------------- Equation (3)
def overturn_asymmetry(real, decision, weight, cluster):
    """Equation (3): O_im = alpha + Delta^U G_i + e, with O = disagreement with the
    real outcome and G_i = 1 when the real outcome favors the stronger party.

    p^U = alpha + Delta^U (stronger-party wins overturned toward the underdog),
    p^S = alpha (underdog wins overturned toward the stronger party).
    """
    real = np.asarray(real, int)
    decision = np.asarray(decision, int)
    overturn = (decision != real).astype(float)
    g = 1 - real
    fit = cluster_ols(overturn, np.column_stack([np.ones(len(real)), g]), weight, cluster)
    return {'pU': contrast(fit, [1, 1]), 'pS': contrast(fit, [1, 0]),
            'asym': contrast(fit, [0, 1]), 'fit': fit}


def treatment_asymmetry(real, d0, dt, weight, cluster):
    """Equation (A.3): stack C0 and treatment decisions of the same units,
    O = alpha + beta G + gamma T + delta (G x T). Returns beta (C0 asymmetry),
    beta + delta (treatment asymmetry), and delta (change)."""
    real = np.asarray(real, int)
    n = len(real)
    g = np.tile(1 - real, 2)
    t = np.repeat([0.0, 1.0], n)
    o = np.concatenate([(np.asarray(d0) != real), (np.asarray(dt) != real)]).astype(float)
    x = np.column_stack([np.ones(2 * n), g, t, g * t])
    fit = cluster_ols(o, x, np.tile(weight, 2), np.tile(np.asarray(cluster), 2))
    return {'c0': contrast(fit, [0, 1, 0, 0]), 'treat': contrast(fit, [0, 1, 0, 1]),
            'delta': contrast(fit, [0, 0, 0, 1]), 'fit': fit}


# ---------------------------------------------------------------- fixed-effects stacks
def within_fe(responses, weight, cluster):
    """Stack k+1 decisions per unit (column 0 is the omitted reference), absorb the
    unit fixed effect by within transformation, and regress on the k condition
    indicators. Used for Equation (1) (reference = real outcome), Equation (A.2)
    (reference = C0) and Equation (B.1) (reference = C0). Returns one coefficient per
    non-reference column and the within R^2."""
    y = np.asarray(responses, float)
    n, m1 = y.shape
    k = m1 - 1
    y = y - y.mean(axis=1, keepdims=True)
    ind = np.vstack([np.zeros(k), np.eye(k)])
    within = ind - ind.mean(axis=0, keepdims=True)
    x = np.broadcast_to(within, (n, m1, k)).reshape(n * m1, k)
    fit = cluster_ols(y.ravel(), x, np.repeat(np.asarray(weight, float), m1),
                      np.repeat(np.asarray(cluster), m1), centered=False)
    coefs = [contrast(fit, np.eye(k)[j]) for j in range(k)]
    return {'coefs': coefs, 'r2': fit['r2'], 'nobs': fit['nobs'], 'clusters': fit['clusters']}


# ---------------------------------------------------------------- Equation (A.1)
def persuasion(d0, d_under, d_strong, weight, cluster):
    """Equation (A.1): units with a stronger-party C0 decision receive the
    underdog-supporting opinion, the others the stronger-party-supporting opinion.
    Persuaded_i = alpha + delta C0Strong_i. Returns both persuasion rates and the
    anchoring asymmetry Delta^A = delta."""
    d0 = np.asarray(d0, int)
    c0strong = (d0 == 0).astype(float)
    persuaded = np.where(d0 == 0, np.asarray(d_under, int) == 1, np.asarray(d_strong, int) == 0).astype(float)
    fit = cluster_ols(persuaded, np.column_stack([np.ones(len(d0)), c0strong]), weight, cluster)
    return {'under': contrast(fit, [1, 1]), 'stronger': contrast(fit, [1, 0]),
            'anchoring': contrast(fit, [0, 1])}


# ---------------------------------------------------------------- heterogeneity
def subgroup_asymmetry(real, decision, groups, weight, cluster):
    """Equation (3) within each subgroup and a joint Wald (F) test of equal
    asymmetry across groups; a cluster spanning groups counts once."""
    real = np.asarray(real, int)
    decision = np.asarray(decision, int)
    groups = np.asarray(groups, int)
    weight = np.asarray(weight, float)
    cluster = np.asarray(cluster)
    k = groups.max() + 1
    overturn = (decision != real).astype(float)
    gstrong = 1 - real
    ind = np.eye(k)[groups]
    fit = cluster_ols(overturn, np.column_stack([ind, ind * gstrong[:, None]]), weight, cluster)
    rows = []
    for j in range(k):
        s = groups == j
        r = overturn_asymmetry(real[s], decision[s], weight[s], cluster[s])
        r.update(units=int(s.sum()), clusters=int(np.unique(cluster[s]).size))
        rows.append(r)
    restr = np.zeros((k - 1, 2 * k))
    for j in range(1, k):
        restr[j - 1, k + j] = 1
        restr[j - 1, k] = -1
    diff = restr @ fit['beta']
    f = float(diff @ np.linalg.solve(restr @ fit['cov'] @ restr.T, diff) / (k - 1))
    p = float(stats.f.sf(f, k - 1, fit['df']))
    return {'groups': rows, 'wald_f': f, 'wald_p': p, 'df': (k - 1, fit['df'])}


def interacted_terciles(real, decision, length_group, controls, weight, cluster):
    """Housing record-length specification (Table A.4, Panel C): a single regression of
    O on G interacted with record-length terciles, optionally adding control terciles and
    their interactions with G. Returns the coefficient on G x Long (long minus short)."""
    real = np.asarray(real, int)
    overturn = (np.asarray(decision, int) != real).astype(float)
    g = (1 - real).astype(float)
    cols = [np.ones(len(real)), g]
    target = None
    for i, grp in enumerate([length_group] + list(controls)):
        grp = np.asarray(grp, int)
        for level in (1, 2):
            dummy = (grp == level).astype(float)
            if dummy.std() == 0:  # a median-split control has no middle group
                continue
            cols += [dummy, dummy * g]
            if i == 0 and level == 2:
                target = len(cols) - 1  # G x longest record-length tercile
    fit = cluster_ols(overturn, np.column_stack(cols), weight, cluster)
    return contrast(fit, np.eye(len(cols))[target])


# ---------------------------------------------------------------- level metrics
def decision_metrics(real, decision, weight, cluster):
    """Levels used in the post-training tables, each with cluster-robust SE."""
    real = np.asarray(real, int)
    d = np.asarray(decision, int)
    ov = overturn_asymmetry(real, d, weight, cluster)
    out = {'pU': ov['pU'], 'pS': ov['pS'], 'asym': ov['asym'],
           'shift': mean_estimate(d - real, weight, cluster),
           'support': mean_estimate(d, weight, cluster),
           'agree': mean_estimate((d == real).astype(float), weight, cluster)}
    bal = contrast(ov['fit'], [-1, -0.5])
    bal['est'] += 1
    out['balanced'] = bal
    return out


def paired_metric_changes(real, d_a, d_b, weight, cluster):
    """Within-unit paired changes (b minus a) of the decision metrics."""
    real = np.asarray(real, int)
    a = np.asarray(d_a, int)
    b = np.asarray(d_b, int)
    err = (b != real).astype(float) - (a != real).astype(float)
    fit = cluster_ols(err, np.column_stack([np.ones(len(real)), 1 - real]), weight, cluster)
    return {'pU': contrast(fit, [1, 1]), 'pS': contrast(fit, [1, 0]), 'asym': contrast(fit, [0, 1]),
            'shift': mean_estimate(b - a, weight, cluster),
            'support': mean_estimate(b - a, weight, cluster),
            'agree': mean_estimate((b == real).astype(float) - (a == real).astype(float), weight, cluster),
            'balanced': contrast(fit, [-1, -0.5])}


def t_interval(est, se, df, level):
    q = stats.t.ppf(0.5 + level / 2, df)
    return est - q * se, est + q * se
