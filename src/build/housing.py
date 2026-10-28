"""Housing setting: records for the prompt builder.

The repository ships only the sample's identifiers (``data/derived/units_housing.csv``: one
row per complaint head, ``unit_id`` = ``<report id>-head-<NN>``, ``cluster_id`` = report id).
The decisions are public on the Housing Ombudsman Service website (see
``data/download/housing.md``); the repository contains no report text.

To re-send prompts, supply a JSONL file with one object per complaint head::

    {"unit_id": "...", "case_record": "...", "complaint_head": "..."}

where ``case_record`` is the report's retained narrative (with the assessment, determination
and recorded finding withheld) and ``complaint_head`` the verbatim target complaint head; see
``src/prompts/housing.py``.
"""
import json
from pathlib import Path

DEFAULT_FILENAME = 'housing_records.jsonl'


def _path(raw_dir):
    path = Path(raw_dir)
    return path / DEFAULT_FILENAME if path.is_dir() else path


def load_records(raw_dir, unit_ids=None):
    """Records {unit_id, case_record, complaint_head} in the order of ``unit_ids``.

    ``raw_dir`` is the JSONL file or a folder containing ``housing_records.jsonl``. With
    ``unit_ids`` None, every record in the file is returned in file order. Requested unit IDs
    missing from the file raise a KeyError listing them.
    """
    records = {}
    with _path(raw_dir).open(encoding='utf-8') as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            missing = {'unit_id', 'case_record', 'complaint_head'} - set(row)
            if missing:
                raise ValueError(f'line {number}: missing {sorted(missing)}')
            if row['unit_id'] in records:
                raise ValueError(f'line {number}: duplicate unit_id {row["unit_id"]}')
            records[row['unit_id']] = {k: row[k] for k in ('unit_id', 'case_record', 'complaint_head')}
    if unit_ids is None:
        return list(records.values())
    absent = [u for u in unit_ids if u not in records]
    if absent:
        raise KeyError(f'{len(absent)} requested complaint heads are not in the records file, '
                       f'e.g. {absent[:5]}')
    return [records[u] for u in unit_ids]
