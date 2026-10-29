"""Reproduce every table and in-text estimate of the paper from the frozen decisions.

Usage:  python reproduce.py

Writes output/tables/<label>.csv and .tex, output/inline_values.json, and
output/check_report.txt, then compares every printed number (with its
significance stars) against expected/paper_tables.json and expected/inline_values.json.
Exits with status 1 if any value differs.
"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.analysis import estimators as est  # noqa: E402
from src.analysis.data import MODELS, arrays, load_posttraining, terciles  # noqa: E402
from src.analysis.figures import vote_split_figure  # noqa: E402
from src.analysis.tables import Results, all_tables, num, stars  # noqa: E402

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'output'


def inline_values(R):
    """Numbers reported in the text of the paper and its appendices."""
    v = {}
    # Appendix A.1: fully formatted C0 responses only (retail marketplace)
    d = R.d['marketplace']
    for m in ['Qwen', 'DeepSeek']:
        s = (d[f'{m}|C0|fmt'] == 1).to_numpy()
        real, w, cl = arrays(d, s)
        v[f'marketplace_fully_formatted_asymmetry_{m}'] = num(est.overturn_asymmetry(real, d.loc[s, f'{m}|C0'], w, cl)['asym']['est'])
    # Appendix A.4 and A.5: labor C0 counts and MASK agreement changes
    d = R.d['labor']
    real, w, cl = arrays(d)
    for m in ['DeepSeek', 'Haiku 5.5']:
        v[f'labor_c0_employee_decisions_{m}'] = int(d[f'{m}|C0'].sum())
        x = R.mask_levels('labor', m)['dagr']
        lo, hi = est.t_interval(x['est'], x['se'], x['df'], .95)
        v[f'labor_mask_agreement_change_{m}'] = [num(x['est']), num(x['se']), num(lo), num(hi)]
    v['labor_real_employee_share'] = num(np.average(real, weights=w))
    # Appendix A.2: record-length descriptives
    g, _ = terciles(d.record_length_chars)
    v['labor_employer_statement_mean_chars_by_tercile'] = [round(d.employer_statement_chars[g == j].mean()) for j in range(3)]
    v['labor_tercile_clusters'] = [int(d.cluster_id[g == j].nunique()) for j in range(3)]
    dv = R.d['divorce']
    g, _ = terciles(dv.record_length_chars)
    v['divorce_shortest_tercile_mean_statement_chars_female_male'] = [
        round(dv.female_statement_chars[g == 0].mean()), round(dv.male_statement_chars[g == 0].mean())]
    # Section 4.1 and Appendix A.2: C0 asymmetry in claims where the female spouse is the defendant
    s = (dv.female_plaintiff == 0).to_numpy()
    real, w, cl = arrays(dv, s)
    for m in MODELS:
        a = est.overturn_asymmetry(real, dv.loc[s, f'{m}|C0'], w, cl)['asym']
        v[f'divorce_female_defendant_asymmetry_{m}'] = [num(a['est']), num(a['se']), stars(a['p'])]
    # Appendix B.2: equivalence test (90% case-clustered interval of residual MASK asymmetry)
    real, w, cl = arrays(dv)
    r = est.treatment_asymmetry(real, dv['Qwen|C0'], dv['Qwen|MASK'], w, cl)['treat']
    v['divorce_mask_residual_asymmetry_90ci'] = [num(x) for x in est.t_interval(r['est'], r['se'], r['df'], .90)]
    # Appendix D.5: ties in conditional label scoring (divorce property)
    bp = load_posttraining('base_post', 'divorce')
    pre = bp[bp.model == 'Pretrained'].set_index('unit_id')
    post = bp[bp.model == 'Post-trained'].set_index('unit_id').loc[pre.index]
    v['divorce_ties_pretrained_posttrained'] = [int(pre.exact_tie.sum()), int(post.exact_tie.sum())]
    real, w, cl = pre.real_underdog.to_numpy(int), pre.weight.to_numpy(float), pre.cluster_id.to_numpy()
    a = np.where(pre.exact_tie == 1, 1, pre.underdog_support).astype(int)
    b = np.where(post.exact_tie == 1, 1, post.underdog_support).astype(int)
    ch = est.paired_metric_changes(real, a, b, w, cl)['asym']['est']
    v['divorce_ties_to_female'] = [num(est.overturn_asymmetry(real, a, w, cl)['asym']['est']),
                                   num(est.overturn_asymmetry(real, b, w, cl)['asym']['est']), num(ch)]
    keep = ((pre.exact_tie == 0) & (post.exact_tie == 0)).to_numpy()
    v['divorce_tied_claims_either_model'] = int((~keep).sum())
    a, b = pre.underdog_support.to_numpy(int)[keep], post.underdog_support.to_numpy(int)[keep]
    v['divorce_ties_excluded'] = [num(est.overturn_asymmetry(real[keep], a, w[keep], cl[keep])['asym']['est']),
                                  num(est.overturn_asymmetry(real[keep], b, w[keep], cl[keep])['asym']['est']),
                                  num(est.paired_metric_changes(real[keep], a, b, w[keep], cl[keep])['asym']['est'])]
    # Appendix B.4: fine-tuning splits
    for s in ['housing', 'labor']:
        x = load_posttraining('sft', s)
        v[f'sft_{s}_units_train_val_test'] = [int((x.split == k).sum()) for k in ['train', 'val', 'test']]
        clusters = [int(x.cluster_id[x.split == k].nunique()) for k in ['train', 'val', 'test']]
        if s == 'housing':
            v['sft_housing_test_clusters'] = clusters[2]
        else:
            v['sft_labor_clusters_train_val_test'] = clusters
        v[f'sft_{s}_m2_upweighted'] = int(x.m2_upweighted.sum())
    # Samples
    for s, d in R.d.items():
        v[f'{s}_units_clusters'] = [len(d), int(d.cluster_id.nunique())]
    return v


def main():
    R = Results()
    (OUT / 'tables').mkdir(parents=True, exist_ok=True)
    expected = json.loads((ROOT / 'expected' / 'paper_tables.json').read_text(encoding='utf-8'))
    report, failures = [], 0
    for t in all_tables(R):
        name = t.label.replace('tab:', '')
        t.frame().to_csv(OUT / 'tables' / f'{name}.csv', index=False)
        (OUT / 'tables' / f'{name}.tex').write_text(t.latex(), encoding='utf-8')
        if t.label not in expected:
            report.append(f'{t.label}: written (no numeric check)')
            continue
        got = Counter(map(tuple, t.tokens()))
        want = Counter(map(tuple, expected[t.label]))
        missing, extra = want - got, got - want
        ok = not missing and not extra
        failures += not ok
        report.append(f'{t.label}: {"OK" if ok else "MISMATCH"} ({sum(want.values())} values)')
        if not ok:
            report.append(f'   paper only: {sorted(missing.elements())[:20]}')
            report.append(f'   reproduced only: {sorted(extra.elements())[:20]}')
    (OUT / 'figures').mkdir(parents=True, exist_ok=True)
    vote_split_figure(R.d['marketplace'], OUT / 'figures' / 'xianyu_vote_split').to_csv(
        OUT / 'figures' / 'xianyu_vote_split.csv', index=False)
    report.append('fig:vote_split: written to output/figures/')
    got = inline_values(R)
    (OUT / 'inline_values.json').write_text(json.dumps(got, indent=1), encoding='utf-8')
    want = json.loads((ROOT / 'expected' / 'inline_values.json').read_text(encoding='utf-8'))
    for k, w in want.items():
        ok = got.get(k) == w
        failures += not ok
        report.append(f'{k}: {"OK" if ok else f"MISMATCH (paper {w}, reproduced {got.get(k)})"}')
    report.append(f'\n{failures} mismatch(es).')
    (OUT / 'check_report.txt').write_text('\n'.join(report), encoding='utf-8')
    print('\n'.join(report))
    sys.exit(1 if failures else 0)


if __name__ == '__main__':
    main()
