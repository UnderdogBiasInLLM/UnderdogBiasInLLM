"""Re-send one setting's prompts to one model under one condition and read the decisions.

    python -m src.run.run --setting housing --model qwen --condition C0 --raw data/raw/housing --limit 20
    python -m src.run.run --setting labor --model haiku --condition MASK --dry-run

Steps:
1. Units: every unit of ``data/derived/units_<setting>.csv``; for a treatment, the units that
   have that condition in ``decisions/decisions_<setting>.csv`` (all units, except the
   retail marketplace opinion arms, which use the disputes with juror votes on both sides).
2. Records: ``src.build.<setting>.load_records(raw, unit_ids)`` from the source data you
   downloaded (``--raw``; see ``data/download/<setting>.md``). For housing, ``--raw`` is a
   JSONL file (or a folder holding ``housing_records.jsonl``) of
   {unit_id, case_record, complaint_head}. Retail marketplace runs need ``--image-dir`` so
   that image parts are local files.
3. Requests: ``src.prompts.<setting>.build_request(record, condition)`` (plus the provider
   model ID where the builder takes one).
4. Calls: ``src.run.clients`` with the paper's model configuration, one independent
   request per unit; API keys come from the environment. Every reply is appended to
   ``runs/<setting>/<model>/<condition>/responses.jsonl`` (git-ignored). Re-running the same
   command resumes: units that already have a readable decision are skipped, and a unit is
   re-sent (up to ``--max-attempts`` model replies) only while it has none.
5. Decisions: ``src.parse.<setting>.read_decision`` on the first usable reply (final and
   complete, not truncated or refused, without reasoning output, from the requested model;
   see ``clients.usable``) with a unique explicit decision; results are merged into ``runs/decisions_<setting>.csv`` in the format of
   ``decisions/decisions_<setting>.csv`` (unit_id, model, condition, underdog_support,
   fully_formatted).

``--dry-run`` prints the first request body (images shown as placeholders) and sends nothing.
"""
import argparse
import csv
import hashlib
import importlib
import inspect
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from . import clients

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / 'runs'
SETTINGS = ('marketplace', 'housing', 'labor', 'divorce')
CONDITIONS = ('C0', 'OPINION_U', 'OPINION_S', 'MASK', 'DEBIAS',
              'CUE_CARE_U', 'CUE_CARE_S', 'CUE_DIS_U', 'CUE_DIS_S')
DECISION_COLUMNS = ['unit_id', 'model', 'condition', 'underdog_support', 'fully_formatted']


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(',', ':')).encode('utf-8')).hexdigest()


def sample_units(setting, condition):
    """Unit IDs of the full constructed sample, restricted to the treatment's sample."""
    with (ROOT / 'data' / 'derived' / f'units_{setting}.csv').open(encoding='utf-8', newline='') as f:
        units = [row['unit_id'] for row in csv.DictReader(f)]
    shipped = ROOT / 'decisions' / f'decisions_{setting}.csv'
    if condition != 'C0' and shipped.exists():
        with shipped.open(encoding='utf-8', newline='') as f:
            in_arm = {row['unit_id'] for row in csv.DictReader(f) if row['condition'] == condition}
        if in_arm:
            units = [u for u in units if u in in_arm]
    return units


def build_requests(build, prompts, raw, units, args):
    """{unit_id: request} in sample order.

    Optional loader arguments (``image_dir`` for the retail marketplace images, ``cache_dir``
    for divorce property) are passed when given; the provider model ID is passed to prompt
    builders that take one.
    """
    options = {'image_dir': args.image_dir, 'cache_dir': args.cache_dir}
    accepted = inspect.signature(build.load_records).parameters
    kwargs = {k: v for k, v in options.items() if v is not None and k in accepted}
    loaded = build.load_records(raw, units, **kwargs)
    records = loaded if isinstance(loaded, dict) else {r['unit_id']: r for r in loaded}
    extra = ({'model': clients.MODELS[args.model]}
             if 'model' in inspect.signature(prompts.build_request).parameters else {})
    return {u: prompts.build_request(records[u], args.condition, **extra) for u in units}


def read_response(parse, raw_text, condition):
    """(underdog_support 0/1/None, fully_formatted bool) from a setting's parser."""
    result = parse.read_decision(raw_text, condition)
    if isinstance(result, tuple):
        return result
    formatted = parse.fully_formatted(raw_text, condition) if result is not None else False
    return result, bool(formatted)


class Journal:
    """Append-only record of every reply for one setting x model x condition."""

    def __init__(self, folder):
        self.path = folder / 'responses.jsonl'
        folder.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()

    def entries(self):
        if not self.path.exists():
            return []
        with self.path.open(encoding='utf-8') as f:
            return [json.loads(line) for line in f if line.strip()]

    def append(self, entry):
        with self.lock, self.path.open('a', encoding='utf-8', newline='\n') as f:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')


def decide(parse, entries, condition):
    """First usable reply with a readable decision, per unit: {unit_id: (support, formatted)}."""
    out = {}
    for e in entries:
        if e['unit_id'] in out or not e.get('usable'):
            continue
        support, formatted = read_response(parse, e.get('text') or '', condition)
        if support is not None:
            out[e['unit_id']] = (int(support), int(bool(formatted)))
    return out


def write_decisions(setting, model_label, condition, decisions):
    """Merge into runs/decisions_<setting>.csv, replacing rows of the same model x condition."""
    path = RUNS / f'decisions_{setting}.csv'
    rows = []
    if path.exists():
        with path.open(encoding='utf-8', newline='') as f:
            rows = [r for r in csv.DictReader(f) if (r['model'], r['condition']) != (model_label, condition)]
    rows += [{'unit_id': u, 'model': model_label, 'condition': condition,
              'underdog_support': s, 'fully_formatted': ff} for u, (s, ff) in decisions.items()]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=DECISION_COLUMNS, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--setting', required=True, choices=SETTINGS)
    parser.add_argument('--model', required=True, choices=sorted(clients.MODELS))
    parser.add_argument('--condition', required=True, choices=CONDITIONS)
    parser.add_argument('--raw', type=Path, help='downloaded source data (default data/raw/<setting>)')
    parser.add_argument('--image-dir', type=Path, help='retail marketplace: folder for extracted original images')
    parser.add_argument('--cache-dir', type=Path, help='divorce property: folder for the rebuilt-row cache')
    parser.add_argument('--limit', type=int, help='only the first N units of the sample')
    parser.add_argument('--workers', type=int, default=4, help='concurrent requests (default 4)')
    parser.add_argument('--max-attempts', type=int, default=3,
                        help='model replies per unit while none is readable (default 3)')
    parser.add_argument('--dry-run', action='store_true', help='print the first request body; send nothing')
    args = parser.parse_args(argv)

    build = importlib.import_module(f'src.build.{args.setting}')
    prompts = importlib.import_module(f'src.prompts.{args.setting}')
    parse = importlib.import_module(f'src.parse.{args.setting}')
    units = sample_units(args.setting, args.condition)
    if args.limit is not None:
        units = units[:args.limit]
    raw = args.raw or ROOT / 'data' / 'raw' / args.setting
    requests_by_unit = build_requests(build, prompts, raw, units, args)

    if args.dry_run:
        first = next(iter(requests_by_unit.values()))
        body = clients.request_body(args.model, first, args.setting)
        print(json.dumps(body, ensure_ascii=False, indent=2))
        print(f'[dry run] {len(requests_by_unit)} requests built; nothing sent', file=sys.stderr)
        return

    folder = RUNS / args.setting / args.model / args.condition
    journal = Journal(folder)
    previous = journal.entries()
    done = decide(parse, previous, args.condition)
    replies = {}
    for e in previous:
        if e.get('response_received'):
            replies[e['unit_id']] = replies.get(e['unit_id'], 0) + 1
    pending = [u for u in units if u not in done and replies.get(u, 0) < args.max_attempts]
    print(f'{args.setting} / {clients.MODEL_LABELS[args.model]} / {args.condition}: '
          f'{len(units)} units, {len(done)} already decided, {len(pending)} to send', file=sys.stderr)

    def work(unit_id):
        request = requests_by_unit[unit_id]
        for attempt in range(replies.get(unit_id, 0) + 1, args.max_attempts + 1):
            entry = {'unit_id': unit_id, 'attempt': attempt, 'model': clients.MODELS[args.model],
                     'request_sha256': digest(request), 'sent_at': now()}
            try:
                reply = clients.call(args.model, request, args.setting)
            except Exception as exc:  # recorded; the unit stays pending for a later run
                journal.append({**entry, 'received_at': now(), 'response_received': False,
                                'complete': False, 'error': f'{type(exc).__name__}: {exc}'})
                return 'error'
            ok, reason = clients.usable(args.model, reply)
            journal.append({**entry, 'received_at': now(), 'response_received': True,
                            'usable': ok, 'unusable_reason': reason, **reply})
            if ok and read_response(parse, reply['text'], args.condition)[0] is not None:
                return 'decided'
        return 'unreadable'

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        outcomes = list(pool.map(work, pending))
    decisions = decide(parse, journal.entries(), args.condition)
    decisions = {u: decisions[u] for u in units if u in decisions}
    path = write_decisions(args.setting, clients.MODEL_LABELS[args.model], args.condition, decisions)
    summary = {k: outcomes.count(k) for k in sorted(set(outcomes))}
    print(f'sent: {summary}; decided {len(decisions)} of {len(units)} units -> {path}',
          file=sys.stderr)
    missing = len(units) - len(decisions)
    if missing:
        print(f'{missing} units have no readable decision yet; re-run the same command to retry them',
              file=sys.stderr)


if __name__ == '__main__':
    main()
