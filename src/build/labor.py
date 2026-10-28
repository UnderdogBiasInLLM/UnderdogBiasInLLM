"""Labor setting: rebuild the CPEE labor-service plea sample (paper Appendix C.3).

Source: CPEE (Zhao et al. 2022), ``CPEE_dataset.zip`` from https://github.com/LiliizZ/CPEE.
See ``data/download/labor.md``. ``raw_dir`` may be the zip file itself, a folder holding the
zip, or a folder holding the extracted ``train_cpee.json``, ``val_cpee.json`` and
``test_cpee.json`` (directly or inside a ``CPEE_dataset/`` subfolder).

Screening chain (counts printed by ``python -m src.build.labor <raw_dir>``):

1. pool the train, validation and test partitions                    158,625 records
2. cause is exactly ["劳务合同纠纷"] (labor-service contract)           11,119
3. defendant statement non-empty after stripping and not "无"          5,374
4. drop exact duplicates of a complete record, keeping the first
   occurrence in train -> validation -> test order                     4,944
5. drop defendant statements that are only a heading or placeholder    3,702 (6,309 pleas)
6. drop partially granted pleas (label 1)                               747 pleas
   -> 5,562 fully granted (2) or fully denied (0) pleas in 3,359 cases; the other 343
      records contain no fully granted or denied plea.

Each retained plea is one adjudication unit; pleas from the same source record form one
cluster. Plaintiff = employee (underdog), defendant = employer (stronger party); a fully
granted plea is a real outcome for the employee (``real_underdog = 1``).
"""
import hashlib
import json
import re
import sys
import unicodedata
import zipfile
from collections import Counter
from pathlib import Path

import pandas as pd

CAUSE = '劳务合同纠纷'
SPLITS = ('train', 'val', 'test')          # deduplication keeps the first occurrence in this order
GRANTED, PARTIAL, DENIED = 2, 1, 0         # CPEE plea-level labels

# A defendant statement whose punctuation- and whitespace-free NFKC form is empty, one of these
# absence markers, or only an answer heading (e.g. "辩称", "答辩意见及证据") carries no content.
ABSENCE_ONLY = {'无', '无答辩', '未答辩', '未作答辩', '未予答辩', '未进行答辩',
                '辩称无答辩', '无答辩意见', '未发表答辩意见', '没有答辩'}
HEADING = re.compile(r'(?:辩称|答辩(?:意见|状|材料)?)'
                     r'(?:(?:及|和|与)(?:相关)?(?:证据(?:材料)?|提交证据))?')

UNIT_COLUMNS = ['unit_id', 'cluster_id', 'weight', 'real_underdog',
                'record_length_chars', 'employer_statement_chars']
RECORD_FIELDS = ('plai', 'defe', 'fact', 'plea')

PAPER_FLOW = {'records': 158625, 'labor_service': 11119, 'defendant_statement': 5374,
              'deduplicated': 4944, 'substantive_statement': 3702, 'pleas_in_records': 6309,
              'partially_granted_pleas': 747, 'records_without_binary_plea': 343,
              'pleas': 5562, 'cases': 3359, 'granted_pleas': 4499, 'denied_pleas': 1063}


# ----------------------------------------------------------------------------- source files
def _partitions(raw_dir):
    """Yield (split, list of records) for train, val, test from a folder or the release zip."""
    raw = Path(raw_dir)
    zips = [raw] if raw.is_file() else sorted(raw.glob('*.zip'))
    for folder in (raw, raw / 'CPEE_dataset'):
        if raw.is_dir() and all((folder / f'{s}_cpee.json').is_file() for s in SPLITS):
            for split in SPLITS:
                yield split, json.loads((folder / f'{split}_cpee.json').read_bytes())
            return
    for path in zips:
        with zipfile.ZipFile(path) as archive:
            names = {Path(n).name: n for n in archive.namelist()
                     if not n.startswith('__MACOSX') and n.endswith('_cpee.json')}
            if all(f'{s}_cpee.json' in names for s in SPLITS):
                for split in SPLITS:
                    yield split, json.loads(archive.read(names[f'{split}_cpee.json']))
                return
    raise FileNotFoundError('CPEE train/val/test_cpee.json not found; see data/download/labor.md')


def _record_digest(record):
    """Hash of the complete source record (canonical JSON), used for exact deduplication."""
    text = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def _is_placeholder(statement):
    compact = ''.join(c for c in unicodedata.normalize('NFKC', statement)
                      if not c.isspace() and not unicodedata.category(c).startswith('P'))
    return not compact or compact in ABSENCE_ONLY or bool(HEADING.fullmatch(compact))


def _chars(text):
    """Non-whitespace characters; a missing (null) field counts as zero."""
    return sum(not c.isspace() for c in (text or ''))


# ----------------------------------------------------------------------------- screening
def screen(raw_dir):
    """Apply the Appendix C.3 chain. Returns (pleas, flow).

    ``pleas`` is a list of dicts, one per retained plea, in source order, with unit_id,
    cluster_id, real_underdog, and the prompt fields plai, defe, fact (verbatim, possibly
    null) and plea (the single target plea string). ``flow`` holds the screening counts.
    """
    flow, seen, pleas = Counter(), set(), []
    for split, rows in _partitions(raw_dir):
        flow['records'] += len(rows)
        for index, row in enumerate(rows):
            if row['cause'] != [CAUSE]:
                continue
            flow['labor_service'] += 1
            labels = row['label']
            if not labels or len(labels) != len(row['plea']) or any(v not in (0, 1, 2) for v in labels):
                raise ValueError('Unexpected CPEE plea/label structure')
            defense = (row['defe'] or '').strip()
            if not defense or defense == '无':
                continue
            flow['defendant_statement'] += 1
            key = _record_digest(row)
            if key in seen:
                continue
            seen.add(key)
            flow['deduplicated'] += 1
            if _is_placeholder(defense):
                continue
            flow['substantive_statement'] += 1
            flow['pleas_in_records'] += len(labels)
            flow['partially_granted_pleas'] += labels.count(PARTIAL)
            targets = [i for i, v in enumerate(labels) if v != PARTIAL]
            if not targets:
                flow['records_without_binary_plea'] += 1
                continue
            case_id = f'cpee-labor-{split}-{index:06d}'
            flow['cases'] += 1
            for i in targets:
                pleas.append({'unit_id': f'{case_id}-plea-{i:02d}', 'cluster_id': case_id,
                              'real_underdog': int(labels[i] == GRANTED),
                              'plai': row['plai'], 'defe': row['defe'], 'fact': row['fact'],
                              'plea': row['plea'][i]})
        del rows
    flow['pleas'] = len(pleas)
    flow['granted_pleas'] = sum(p['real_underdog'] for p in pleas)
    flow['denied_pleas'] = flow['pleas'] - flow['granted_pleas']
    return pleas, dict(flow)


def _units(pleas):
    units = pd.DataFrame([{
        'unit_id': p['unit_id'], 'cluster_id': p['cluster_id'], 'weight': 1,
        'real_underdog': p['real_underdog'],
        # Record length: non-whitespace characters of plai + defe + fact + target plea.
        'record_length_chars': sum(_chars(p[k]) for k in RECORD_FIELDS),
        'employer_statement_chars': _chars(p['defe']),
    } for p in pleas], columns=UNIT_COLUMNS)
    if not units.unit_id.is_unique:
        raise ValueError('Duplicate plea identifiers')
    return units.sort_values('unit_id').reset_index(drop=True)


def build_units(raw_dir):
    """One row per plea: unit_id, cluster_id (source record), weight (1), real_underdog
    (1 = plea fully granted = employee), record_length_chars, employer_statement_chars."""
    return _units(screen(raw_dir)[0])


def load_records(raw_dir, unit_ids=None):
    """Prompt source fields for the requested pleas (all pleas if ``unit_ids`` is None).

    Returns a list of dicts (unit_id, cluster_id, real_underdog, plai, defe, fact, plea) in the
    order of ``unit_ids``; plai/defe/fact are the verbatim source values (str or None) and plea
    is the single target plea string.
    """
    pleas = {p['unit_id']: p for p in screen(raw_dir)[0]}
    if unit_ids is None:
        return [pleas[k] for k in sorted(pleas)]
    missing = [k for k in unit_ids if k not in pleas]
    if missing:
        raise KeyError(f'{len(missing)} unit IDs are not in the rebuilt sample, e.g. {missing[0]}')
    return [pleas[k] for k in unit_ids]


def build_sample(raw_dir):
    """Units table, all records keyed by unit_id, and the screening counts in one pass."""
    pleas, flow = screen(raw_dir)
    return _units(pleas), {p['unit_id']: p for p in pleas}, flow


def main(argv):
    """Usage: python -m src.build.labor <raw_dir> [units_out.csv]"""
    if len(argv) not in (1, 2):
        raise SystemExit(main.__doc__)
    units, _, flow = build_sample(argv[0])
    for key, expected in PAPER_FLOW.items():
        print(f'{key:30s} {flow.get(key, 0):>8,d}  paper {expected:>8,d}'
              + ('' if flow.get(key, 0) == expected else '  MISMATCH'))
    if len(argv) == 2:
        units.to_csv(argv[1], index=False, lineterminator='\n')
        print(f'wrote {len(units)} units to {argv[1]}')
    return int(any(flow.get(k, 0) != v for k, v in PAPER_FLOW.items()))


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
