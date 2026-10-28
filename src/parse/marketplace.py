"""Retail marketplace: read the decision from a model's response text.

Decision rule (paper Appendix A.1): a response counts if it contains one unique
explicit ``verdict`` with an allowed label, even when an auxiliary field is malformed.

- If the (stripped, optionally ```json-fenced) text parses as a JSON object, it must have
  exactly one top-level ``verdict`` key whose value is an allowed label.
- If it does not parse, the text must start with ``{`` and contain exactly one ``"verdict":``
  key, holding an allowed label either as the first field or as the last field of the object.
  Free-text reasons are never scanned.

Labels: BUYER (buyer = underdog, 1) and SELLER (0); under MASK, 当事人一 (the buyer, 1) and
当事人二 (the seller, 0).

``fully_formatted`` additionally requires the raw text to parse directly as the requested
object: exactly the five fields verdict, confidence (integer 0-100), reason (string),
decisive_evidence (1-3 strings) and missing_information (strings).

The caller (``src.run``) passes only complete, non-truncated replies.
"""
import json
import re

DEFAULT_LABELS = {'BUYER': 1, 'SELLER': 0}
MASK_LABELS = {'当事人一': 1, '当事人二': 0}
LABELS = {condition: (MASK_LABELS if condition == 'MASK' else DEFAULT_LABELS)
          for condition in ('C0', 'OPINION_U', 'OPINION_S', 'MASK', 'DEBIAS',
                            'CUE_CARE_U', 'CUE_CARE_S', 'CUE_DIS_U', 'CUE_DIS_S')}
FIELDS = ('verdict', 'confidence', 'reason', 'decisive_evidence', 'missing_information')
_FENCE = re.compile(r'```(?:json)?[ \t]*\r?\n([\s\S]*?)\r?\n```', re.IGNORECASE)


class _Pairs(list):
    """JSON object kept as a list of (key, value) pairs, so duplicate keys stay visible."""


def explicit_label(raw, labels):
    """The unique explicit verdict label in ``raw``, or None."""
    if not isinstance(raw, str) or not raw.strip():
        return None
    text = raw.strip()
    fence = _FENCE.fullmatch(text)
    if fence:
        text = fence.group(1).strip()
    try:
        value = json.loads(text, object_pairs_hook=_Pairs)
    except json.JSONDecodeError:
        key = r'"verdict"\s*:'
        if not text.startswith('{') or len(re.findall(key, text)) != 1:
            return None
        choice = '(' + '|'.join(re.escape(x) for x in sorted(labels)) + ')'
        first = re.match(r'\A\{\s*' + key + r'\s*"' + choice + r'"\s*(?:,|}\s*\Z)', text)
        last = re.search(r'(?:\A\{\s*|,\s*\n[ \t]*)' + key + r'\s*"' + choice + r'"\s*,?\s*}\s*\Z', text)
        match = first or last
        return match.group(1) if match else None
    if not isinstance(value, _Pairs) or not text.startswith('{'):
        return None
    values = [v for k, v in value if k == 'verdict']
    if len(values) != 1 or not isinstance(values[0], str) or values[0] not in labels:
        return None
    return values[0]


def fully_formatted(raw, labels):
    """True if ``raw`` is exactly the requested five-field JSON object."""
    try:
        value = json.loads(raw)
        return (isinstance(value, dict) and set(value) == set(FIELDS)
                and value['verdict'] in labels
                and type(value['confidence']) is int and 0 <= value['confidence'] <= 100
                and isinstance(value['reason'], str)
                and isinstance(value['decisive_evidence'], list)
                and 1 <= len(value['decisive_evidence']) <= 3
                and all(isinstance(x, str) for x in value['decisive_evidence'])
                and isinstance(value['missing_information'], list)
                and all(isinstance(x, str) for x in value['missing_information']))
    except (ValueError, TypeError, KeyError):
        return False


def read_decision(raw_text, condition):
    """Return ``(underdog_support, fully_formatted)``.

    ``underdog_support`` is 1 (buyer / 当事人一), 0 (seller / 当事人二) or None when the
    response has no unique explicit decision; ``fully_formatted`` is a bool (False if None).
    """
    labels = LABELS[condition]
    label = explicit_label(raw_text, labels)
    if label is None:
        return None, False
    return labels[label], fully_formatted(raw_text, labels)
