"""Build every table of the paper from the derived samples and frozen decisions.

Each builder returns a Table: rows of formatted cells (estimate, significance stars,
clustered standard error in parentheses). Percentages and percentage points are
reported to two decimals.
"""
import numpy as np
import pandas as pd

from . import estimators as est
from .data import (CUES, MODELS, SETTING_NAMES, SETTINGS, arrays, load_posttraining,
                   load_setting, terciles)


# ---------------------------------------------------------------- formatting
def stars(p):
    return '***' if p < .001 else '**' if p < .01 else '*' if p < .05 else ''


def num(x, scale=100, dp=2):
    s = f'{x * scale:.{dp}f}'
    return '0.' + '0' * dp if float(s) == 0 else s


def pval(p):
    return '<.001' if p < .001 else f'{p:.3f}'.lstrip('0')


class Cell:
    def __init__(self, value, star='', se=None):
        self.value, self.star, self.se = value, star, se

    def text(self):
        out = self.value + self.star
        return out + (f' ({self.se})' if self.se is not None else '')


def c(r, star=True, se=True, scale=100):
    """Cell from an estimate dict {'est','se','p'}."""
    return Cell(num(r['est'], scale), stars(r['p']) if star else '', num(r['se'], scale) if se else None)


def plain(text):
    return Cell(text)


class Table:
    def __init__(self, label, title, columns, notes=''):
        self.label, self.title, self.columns, self.notes = label, title, columns, notes
        self.rows = []

    def panel(self, text):
        self.rows.append(('panel', text, []))

    def row(self, label, cells):
        self.rows.append(('row', label, cells))

    def frame(self):
        recs = []
        for kind, label, cells in self.rows:
            recs.append([label] + [x.text() for x in cells] + [''] * (len(self.columns) - len(cells)))
        return pd.DataFrame(recs, columns=['row'] + self.columns)

    def latex(self):
        def esc(s):
            return s.replace('%', r'\%').replace('_', r'\_')
        n = len(self.columns) + 1
        lines = [r'\begin{tabular}{l' + 'r' * len(self.columns) + '}', r'\toprule',
                 ' & '.join([''] + [esc(x) for x in self.columns]) + r' \\', r'\midrule']
        for kind, label, cells in self.rows:
            if kind == 'panel':
                lines.append(rf'\multicolumn{{{n}}}{{l}}{{\textit{{{esc(label)}}}}} \\')
            else:
                lines.append(' & '.join([esc(label)] + [
                    x.value.replace('-', '$-$', 1) + (f'$^{{{x.star}}}$' if x.star else '')
                    + (f' ({x.se})' if x.se is not None else '') for x in cells]) + r' \\')
        lines += [r'\bottomrule', r'\end{tabular}']
        return '\n'.join(lines)

    def tokens(self):
        """(number, stars) pairs printed in the table, including panel headings."""
        out = []
        for kind, label, cells in self.rows:
            if kind == 'panel':
                out += [(t, '') for t in self.header_numbers.get(label, [])]
            for x in cells:
                v = x.value.lstrip('<').lstrip('+')
                if '.' in v:
                    out.append((v, x.star))
                if x.se is not None:
                    out.append((x.se, ''))
        return out

    header_numbers = {}


def panel_with_numbers(table, text, numbers):
    table.header_numbers = dict(table.header_numbers)
    table.header_numbers[text] = numbers
    table.panel(text)


# ---------------------------------------------------------------- shared computations
class Results:
    """Computes and caches every estimate once."""

    def __init__(self):
        self.d = {s: load_setting(s) for s in SETTINGS}
        self.cache = {}

    def get(self, key, fn):
        if key not in self.cache:
            self.cache[key] = fn()
        return self.cache[key]

    def baseline(self, s, m):
        d = self.d[s]
        real, w, cl = arrays(d)
        return self.get(('base', s, m), lambda: est.overturn_asymmetry(real, d[f'{m}|C0'], w, cl))

    def support_fe(self, s):
        d = self.d[s]
        real, w, cl = arrays(d)
        resp = np.column_stack([real] + [d[f'{m}|C0'] for m in MODELS])
        return self.get(('fe1', s), lambda: est.within_fe(resp, w, cl))

    def opinion_sample(self, s):
        d = self.d[s]
        return d[d[f'{MODELS[0]}|OPINION_U'].notna()] if s == 'marketplace' else d

    def opinions(self, s, m):
        d = self.opinion_sample(s)
        real, w, cl = arrays(d)
        def run():
            per = est.persuasion(d[f'{m}|C0'], d[f'{m}|OPINION_U'], d[f'{m}|OPINION_S'], w, cl)
            fe = est.within_fe(d[[f'{m}|C0', f'{m}|OPINION_U', f'{m}|OPINION_S']].to_numpy(float), w, cl)
            return {**per, 'tauU': fe['coefs'][0], 'tauS': fe['coefs'][1], 'r2': fe['r2'],
                    'nobs': fe['nobs'], 'units': len(d), 'clusters': fe['clusters']}
        return self.get(('op', s, m), run)

    def treatment(self, s, m, cond):
        d = self.d[s]
        real, w, cl = arrays(d)
        return self.get(('tr', s, m, cond), lambda: est.treatment_asymmetry(
            real, d[f'{m}|C0'], d[f'{m}|{cond}'], w, cl))

    def mask_levels(self, s, m):
        d = self.d[s]
        real, w, cl = arrays(d)
        a, b = d[f'{m}|C0'].to_numpy(int), d[f'{m}|MASK'].to_numpy(int)
        def run():
            return {'sup0': np.average(a, weights=w), 'sup1': np.average(b, weights=w),
                    'dsup': est.mean_estimate(b - a, w, cl),
                    'agr0': np.average(a == real, weights=w), 'agr1': np.average(b == real, weights=w),
                    'dagr': est.mean_estimate((b == real).astype(float) - (a == real), w, cl)}
        return self.get(('ml', s, m), run)

    def cues(self, s, m):
        d = self.d[s]
        real, w, cl = arrays(d)
        return self.get(('cue', s, m), lambda: est.within_fe(
            d[[f'{m}|C0'] + [f'{m}|{x}' for x in CUES]].to_numpy(float), w, cl))

    def groups(self, s):
        d = self.d[s]
        if s == 'marketplace':
            majority = d[['buyer_votes', 'seller_votes']].max(axis=1).to_numpy()
            g = np.where(majority >= 15, 0, np.where(majority >= 12, 1, 2))
            return g, ['15--17 votes', '12--14 votes', '9--11 votes'], None
        g, cuts = terciles(d.record_length_chars)
        lo, hi = int(cuts[0]), int(cuts[1])
        return g, [f'Short (<={lo:,})', f'Medium ({lo + 1:,}--{hi:,})', f'Long (>={hi + 1:,})'], cuts

    def heterogeneity(self, s, m):
        d = self.d[s]
        real, w, cl = arrays(d)
        g = self.groups(s)[0]
        return self.get(('het', s, m), lambda: est.subgroup_asymmetry(real, d[f'{m}|C0'], g, w, cl))


# ---------------------------------------------------------------- main-text tables
def table_data_overview(R):
    t = Table('tab:data_overview', 'Constructed samples', ['Adjudication units', 'Clusters'])
    for s in SETTINGS:
        d = R.d[s]
        t.row(SETTING_NAMES[s], [plain(f'{len(d):,}'), plain(f'{d.cluster_id.nunique():,}')])
    return t


def table_reversals(R):
    t = Table('tab:reversals', 'Overturn rates and underdog support shifts (Table 2)',
              ['Stronger-party wins overturned (%)', 'Underdog wins overturned (%)',
               'Overturn asymmetry (pp)', 'Underdog support shift (pp)'])
    party = {'marketplace': 'buyer', 'housing': 'resident', 'labor': 'employee', 'divorce': 'female spouse'}
    for s in SETTINGS:
        real, w, _ = arrays(R.d[s])
        share = num(np.average(real, weights=w))
        panel_with_numbers(t, f'{SETTING_NAMES[s]} ({party[s]} won {share}%)', [share])
        fe = R.support_fe(s)
        for i, m in enumerate(MODELS):
            b = R.baseline(s, m)
            t.row(m, [c(b['pU'], False, False), c(b['pS'], False, False), c(b['asym'], se=False),
                      c(fe['coefs'][i], se=False)])
    return t


def table_boundary(R):
    t = Table('tab:boundary_conditions', 'Asymmetry by decision ambiguity and record length (Table 3)', MODELS)
    for s, what in [('marketplace', 'decision ambiguity'), ('labor', 'record length'),
                    ('housing', 'record length'), ('divorce', 'record length')]:
        t.panel(f'{SETTING_NAMES[s]} ({what})')
        labels = R.groups(s)[1]
        het = {m: R.heterogeneity(s, m) for m in MODELS}
        for j, lab in enumerate(labels):
            t.row(lab, [c(het[m]['groups'][j]['asym'], se=False) for m in MODELS])
        t.row('Equal-asymmetry test (p-value)', [plain(pval(het[m]['wald_p'])) for m in MODELS])
    return t


def table_opinion_main(R):
    t = Table('tab:human_feedback_anchor', 'Persuasion rates under advisory opinions (Table 4)',
              ['Underdog-supporting (%)', 'Stronger-party-supporting (%)', 'Anchoring asymmetry (pp)'])
    for s in SETTINGS:
        t.panel(SETTING_NAMES[s])
        for m in MODELS:
            o = R.opinions(s, m)
            t.row(m, [c(o['under'], False, False), c(o['stronger'], False, False), c(o['anchoring'], se=False)])
    return t


def table_treatment_main(R, cond, label, title, name):
    t = Table(label, title, ['Baseline', name, f'{name} - Baseline'])
    for s in SETTINGS:
        t.panel(SETTING_NAMES[s])
        for m in MODELS:
            r = R.treatment(s, m, cond)
            t.row(m, [c(r['c0'], se=False), c(r['treat'], se=False), c(r['delta'], se=False)])
    return t


def table_cues_main(R):
    t = Table('tab:social_identity_tags', 'Change in support for the cue recipient (Table 6)',
              ['Caregiving: underdog', 'Caregiving: stronger party', 'Disability: underdog', 'Disability: stronger party'])
    for s in SETTINGS:
        t.panel(SETTING_NAMES[s])
        for m in MODELS:
            co = R.cues(s, m)['coefs']
            cells = []
            for j, sign in enumerate([1, -1, 1, -1]):
                cells.append(c({'est': sign * co[j]['est'], 'se': co[j]['se'], 'p': co[j]['p']}, se=False))
            t.row(m, cells)
    return t


# ---------------------------------------------------------------- Appendix A
def table_full_overturns(R):
    t = Table('tab:full_overturns', 'Baseline estimates (Table A.1)',
              ['p^U', 'p^S', 'Asymmetry Delta^U', 'Support shift beta_m'])
    unit = {'marketplace': 'disputes', 'housing': 'complaint heads', 'labor': 'pleas', 'divorce': 'claims'}
    for s in SETTINGS:
        d = R.d[s]
        fe = R.support_fe(s)
        r2 = f'{fe["r2"]:.3f}'
        panel_with_numbers(t, f'{SETTING_NAMES[s]} ({len(d):,} {unit[s]}; {d.cluster_id.nunique():,} clusters; '
                              f'Eq. (1): {fe["nobs"]:,} observations, within R^2={r2})', [r2])
        for i, m in enumerate(MODELS):
            b = R.baseline(s, m)
            t.row(m, [c(b['pU'], False), c(b['pS'], False), c(b['asym']), c(fe['coefs'][i])])
    return t


def table_heterogeneity(R, s, label, title):
    t = Table(label, title, ['p^U', 'p^S', 'Asymmetry Delta^U'])
    labels = R.groups(s)[1]
    for m in MODELS:
        h = R.heterogeneity(s, m)
        t.panel(f'{m} (equal-Delta^U test: p{"" if h["wald_p"] < .001 else "="}{pval(h["wald_p"])})')
        t.header_numbers = dict(t.header_numbers)
        t.header_numbers[t.rows[-1][1]] = [pval(h['wald_p']).lstrip('<')]
        for j, lab in enumerate(labels):
            g = h['groups'][j]
            t.row(f'{lab} (N={g["units"]:,}; {g["clusters"]:,} clusters)',
                  [c(g['pU'], False), c(g['pS'], False), c(g['asym'])])
    return t


def table_length_other(R):
    t = Table('tab:heterogeneity_length_other', 'Record length, housing and divorce (Table A.4)', MODELS)
    for s, panel in [('housing', 'A. Housing'), ('divorce', 'B. Divorce Property')]:
        t.panel(f'Panel {panel}: coefficient on G_i by record-length tercile')
        labels = R.groups(s)[1]
        het = {m: R.heterogeneity(s, m) for m in MODELS}
        for j, lab in enumerate(labels):
            n = het[MODELS[0]]['groups'][j]['units']
            t.row(f'{lab}; N={n:,}', [c(het[m]['groups'][j]['asym']) for m in MODELS])
        t.row('Equal-Delta^U test (p-value)', [plain(pval(het[m]['wald_p'])) for m in MODELS])
    t.panel('Panel C. Housing: coefficient on G_i x Long (long-minus-short difference in asymmetry)')
    d = R.d['housing']
    real, w, cl = arrays(d)
    g = R.groups('housing')[0]
    apol, _ = terciles(d.apology_ack_count)
    evt, _ = terciles(d.dated_event_count)
    for lab, controls in [('(i) No controls', []), ('(ii) + apology/acknowledgment terciles', [apol]),
                          ('(iii) + dated-event terciles', [apol, evt])]:
        t.row(lab, [c(R.get(('lenC', m, lab), lambda m=m, controls=controls: est.interacted_terciles(
            real, d[f'{m}|C0'], g, controls, w, cl))) for m in MODELS])
    return t


def table_opinion_details(R):
    t = Table('tab:opinion_details', 'Advisory opinions: persuasion and support changes (Table A.5)',
              ['Persuasion, pro-underdog', 'Persuasion, pro-stronger', 'Anchoring Delta^A',
               'tau_U', 'tau_S', 'Within R^2'])
    unit = {'marketplace': 'disputes', 'housing': 'complaint heads', 'labor': 'pleas', 'divorce': 'claims'}
    for s in SETTINGS:
        o = R.opinions(s, MODELS[0])
        t.panel(f'{SETTING_NAMES[s]} ({o["units"]:,} {unit[s]}; {o["clusters"]:,} clusters; '
                f'Eq. (A.2): {o["nobs"]:,} observations)')
        for m in MODELS:
            o = R.opinions(s, m)
            t.row(m, [c(o['under'], False), c(o['stronger'], False), c(o['anchoring']),
                      c(o['tauU']), c(o['tauS']), plain(f'{o["r2"]:.3f}')])
    return t


def table_treatment_full(R, cond, label, title):
    t = Table(label, title, ['C0 asymmetry (beta)', 'Treatment asymmetry (beta+delta)', 'Change (delta)'])
    for s in SETTINGS:
        t.panel(SETTING_NAMES[s])
        for m in MODELS:
            r = R.treatment(s, m, cond)
            t.row(m, [c(r['c0'], False), c(r['treat'], False), c(r['delta'])])
    return t


def table_mask_details(R):
    t = Table('tab:mask_details', 'Identity masking: support and agreement (Table A.7)',
              ['Support C0', 'Support MASK', 'Support MASK-C0', 'Agreement C0', 'Agreement MASK', 'Agreement MASK-C0'])
    for s in SETTINGS:
        t.panel(SETTING_NAMES[s])
        for m in MODELS:
            x = R.mask_levels(s, m)
            t.row(m, [plain(num(x['sup0'])), plain(num(x['sup1'])), c(x['dsup']),
                      plain(num(x['agr0'])), plain(num(x['agr1'])), c(x['dagr'])])
    return t


# ---------------------------------------------------------------- Appendix B
def table_cue_regressions(R):
    t = Table('tab:social_cue_regressions', 'Vulnerability cues: change in underdog support (Table B.1)', MODELS)
    names = ['Caregiving cue, underdog', 'Caregiving cue, stronger party',
             'Disability cue, underdog', 'Disability cue, stronger party']
    for s in SETTINGS:
        fit = {m: R.cues(s, m) for m in MODELS}
        f0 = fit[MODELS[0]]
        t.panel(f'{SETTING_NAMES[s]} ({f0["nobs"]:,} observations; {f0["clusters"]:,} clusters)')
        for j, lab in enumerate(names):
            t.row(lab, [c(fit[m]['coefs'][j]) for m in MODELS])
        t.row('Within R^2', [plain(f'{fit[m]["r2"]:.4f}') for m in MODELS])
    return t


def divorce_mask_rows(R):
    d = R.d['divorce']
    out = []

    def run(label, sub, weights):
        x = d[sub]
        real = x.real_underdog.to_numpy(int)
        w = weights[sub]
        cl = x.cluster_id.to_numpy()
        a, b = x['Qwen|C0'].to_numpy(int), x['Qwen|MASK'].to_numpy(int)
        r = est.treatment_asymmetry(real, a, b, w, cl)
        agr = est.mean_estimate((b == real).astype(float) - (a == real), w, cl)
        out.append((label, int(sub.sum()), r, agr))

    ones = np.ones(len(d))
    allu = np.ones(len(d), bool)
    run('Female plaintiff', d.female_plaintiff.to_numpy() == 1, ones)
    run('Female defendant', d.female_plaintiff.to_numpy() == 0, ones)
    run('Post-divorce property action', d.post_divorce_action.to_numpy() == 1, ones)
    run('Divorce and property together', d.post_divorce_action.to_numpy() == 0, ones)
    run('Claim-equal weights', allu, ones)
    run('Equal case weights', allu, 1 / d.groupby('cluster_id').unit_id.transform('size').to_numpy(float))
    cell = d.female_plaintiff.astype(str) + d.real_underdog.astype(str)
    run('Equal role-by-outcome cells', allu, 1 / cell.map(cell.value_counts()).to_numpy(float))
    run('Fully formatted responses only', ((d['Qwen|C0|fmt'] == 1) & (d['Qwen|MASK|fmt'] == 1)).to_numpy(), ones)
    return out


def table_divorce_mask(R):
    t = Table('tab:appendix_divorce_mask_robustness', 'Divorce masking (Qwen): strata and sensitivity (Table B.2)',
              ['Claims', 'C0', 'MASK', 'MASK - C0', 'Agreement change (pp)'])
    rows = R.get('divmask', lambda: divorce_mask_rows(R))
    for i, (lab, n, r, agr) in enumerate(rows):
        if i in (0, 2, 4):
            t.panel(["Panel A. Female spouse's litigation role", 'Panel B. Record type', 'Panel C. Full sample'][i // 2])
        t.row(lab, [plain(f'{n:,}'), c(r['c0'], False), c(r['treat'], False), c(r['delta']), c(agr)])
    return t


METRIC_COLS = ['pU', 'pS', 'asym', 'shift', 'agree', 'balanced']
METRIC_NAMES = ['Stronger -> underdog (%)', 'Underdog -> stronger (%)', 'Asymmetry (pp)',
                'Support shift (pp)', 'Agreement (%)', 'Balanced agreement (%)']


def base_post_results(R, s):
    def run():
        bp = load_posttraining('base_post', s)
        pre = bp[bp.model == 'Pretrained'].set_index('unit_id')
        post = bp[bp.model == 'Post-trained'].set_index('unit_id').loc[pre.index]
        real, w, cl = pre.real_underdog.to_numpy(int), pre.weight.to_numpy(float), pre.cluster_id.to_numpy()
        a, b = pre.underdog_support.to_numpy(int), post.underdog_support.to_numpy(int)
        return {'pre': est.decision_metrics(real, a, w, cl), 'post': est.decision_metrics(real, b, w, cl),
                'diff': est.paired_metric_changes(real, a, b, w, cl), 'units': len(pre),
                'clusters': pre.cluster_id.nunique(), 'frame': (pre, post)}
    return R.get(('bp', s), run)


def table_base_post(R, full):
    t = Table('tab:base_post_full' if full else 'tab:base_post',
              'Pretrained versus post-trained Qwen3.5-9B' + (' (Table B.3)' if full else ' (Table 7)'),
              METRIC_NAMES if full else METRIC_NAMES[:3])
    cols = METRIC_COLS if full else METRIC_COLS[:3]
    for s in ['housing', 'labor', 'divorce']:
        r = base_post_results(R, s)
        t.panel(SETTING_NAMES[s])
        for key, lab in [('pre', 'Pretrained'), ('post', 'Post-trained'), ('diff', 'Post-trained - pretrained')]:
            t.row(lab, [c(r[key][k], star=(key == 'diff' or k == 'asym'), se=full) for k in cols])
    return t


def sft_results(R, s):
    def run():
        x = load_posttraining('sft', s)
        x = x[x.split == 'test']
        real, w, cl = x.real_underdog.to_numpy(int), x.weight.to_numpy(float), x.cluster_id.to_numpy()
        dec = {k: x[k].to_numpy(int) for k in ['C0', 'M1', 'M2']}
        out = {k: est.decision_metrics(real, v, w, cl) for k, v in dec.items()}
        for a, b in [('C0', 'M1'), ('C0', 'M2'), ('M1', 'M2')]:
            out[f'{b}-{a}'] = est.paired_metric_changes(real, dec[a], dec[b], w, cl)
        out['units'], out['clusters'] = len(x), x.cluster_id.nunique()
        return out
    return R.get(('sft', s), run)


def table_sft(R, full):
    t = Table('tab:targeted_sft_full' if full else 'tab:targeted_sft',
              'Standard and targeted fine-tuning, held-out test samples' + (' (Table B.4)' if full else ' (Table 9)'),
              METRIC_NAMES if full else ['Stronger -> underdog (%)', 'Underdog -> stronger (%)', 'Asymmetry (pp)',
                                         'Agreement (%)', 'Balanced agreement (%)'])
    cols = METRIC_COLS if full else ['pU', 'pS', 'asym', 'agree', 'balanced']
    rows = ([('C0', 'Before fine-tuning (C0)'), ('M1', 'Standard SFT (M1)'), ('M2', 'Targeted SFT (M2)'),
             ('M1-C0', 'M1 - C0'), ('M2-C0', 'M2 - C0'), ('M2-M1', 'M2 - M1')] if full else
            [('C0', 'Before fine-tuning'), ('M1', 'Standard SFT'), ('M2', 'Targeted SFT'),
             ('M1-C0', 'Standard SFT - Before'), ('M2-M1', 'Targeted SFT - Standard SFT')])
    for s in ['housing', 'labor']:
        r = sft_results(R, s)
        t.panel(f'{SETTING_NAMES[s]} (test sample: {r["units"]:,} units; {r["clusters"]:,} clusters)')
        for key, lab in rows:
            diff = '-' in key
            t.row(lab, [c(r[key][k], star=diff or (full and k == 'asym'), se=full) for k in cols])
    return t


def all_tables(R):
    return [table_data_overview(R), table_reversals(R), table_boundary(R), table_opinion_main(R),
            table_treatment_main(R, 'MASK', 'tab:identity_masking', 'Asymmetry under identity masking (Table 5)', 'MASK'),
            table_cues_main(R), table_base_post(R, False),
            table_treatment_main(R, 'DEBIAS', 'tab:debias_instruction', 'Asymmetry under the debiasing instruction (Table 8)', 'Instruction'),
            table_sft(R, False),
            table_full_overturns(R),
            table_heterogeneity(R, 'marketplace', 'tab:heterogeneity_consensus', 'Asymmetry by juror consensus (Table A.2)'),
            table_heterogeneity(R, 'labor', 'tab:heterogeneity_labor', 'Asymmetry by record length, labor (Table A.3)'),
            table_length_other(R), table_opinion_details(R),
            table_treatment_full(R, 'MASK', 'tab:identity_masking_full', 'Identity masking (Table A.6)'),
            table_mask_details(R),
            table_treatment_full(R, 'DEBIAS', 'tab:debias_instruction_full', 'Debiasing instruction (Table A.8)'),
            table_cue_regressions(R), table_divorce_mask(R), table_base_post(R, True), table_sft(R, True)]
