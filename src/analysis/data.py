"""Load the derived samples and frozen decisions into one wide frame per setting."""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SETTINGS = ['marketplace', 'housing', 'labor', 'divorce']
SETTING_NAMES = {'marketplace': 'Retail Marketplace', 'housing': 'Housing',
                 'labor': 'Labor', 'divorce': 'Divorce Property'}
MODELS = ['Qwen', 'DeepSeek', 'Luna', 'Haiku 5.5']
CUES = ['CUE_CARE_U', 'CUE_CARE_S', 'CUE_DIS_U', 'CUE_DIS_S']


def load_setting(setting):
    """Units with one column per model x condition (decision favors the underdog)
    and per model x condition formatting flag (suffix '|fmt')."""
    units = pd.read_csv(ROOT / 'data' / 'derived' / f'units_{setting}.csv', dtype={'unit_id': str, 'cluster_id': str})
    dec = pd.read_csv(ROOT / 'decisions' / f'decisions_{setting}.csv', dtype={'unit_id': str})
    if dec.duplicated(['unit_id', 'model', 'condition']).any():
        raise ValueError(f'{setting}: duplicate decisions')
    if not set(dec.unit_id) <= set(units.unit_id):
        raise ValueError(f'{setting}: decisions for units outside the sample')
    dec['key'] = dec.model + '|' + dec.condition
    wide = dec.pivot(index='unit_id', columns='key', values='underdog_support')
    fmt = dec.pivot(index='unit_id', columns='key', values='fully_formatted').add_suffix('|fmt')
    d = units.set_index('unit_id').join(wide).join(fmt).reset_index()
    for m in MODELS:
        if d[f'{m}|C0'].isna().any():
            raise ValueError(f'{setting}: {m} C0 missing for some units')
    return d


def arrays(d, subset=None):
    s = d if subset is None else d[subset]
    return s.real_underdog.to_numpy(int), s.weight.to_numpy(float), s.cluster_id.to_numpy()


def terciles(values):
    """Tercile groups 0, 1, 2 with cut points at the 1/3 and 2/3 quantiles (upper bounds
    inclusive). If the two cut points coincide (heavily tied counts), units are split at
    the median into groups 0 and 2."""
    v = np.asarray(values, float)
    cuts = np.quantile(v, [1 / 3, 2 / 3])
    if cuts[0] == cuts[1]:
        return (v > np.median(v)).astype(int) * 2, cuts
    return np.where(v <= cuts[0], 0, np.where(v <= cuts[1], 1, 2)), cuts


def load_posttraining(kind, setting):
    return pd.read_csv(ROOT / 'posttraining' / f'{kind}_{setting}.csv', dtype={'unit_id': str, 'cluster_id': str})
