"""Housing setting: read the decision from a model's response text.

Decision field and labels (1 = decision for the resident, the underdog):

- every condition except MASK: ``complaint_outcome``; UPHELD -> 1, NOT_UPHELD -> 0;
- MASK: ``verdict``; PARTY_A (the complainant) -> 1, PARTY_B -> 0.

Decision rule (paper Appendix A.1): a response counts if it contains one
unique explicit decision for the target head, even when an auxiliary field is malformed. It is
never inferred from the reason text or from the real outcome.

1. If the stripped text (optionally inside one ```json fence) parses as a JSON object with
   exactly one decision-field key whose value is an allowed label, that label is the decision.
   In this single-object case the doubled negation ``NOT_NOT_UPHELD`` is read as NOT_UPHELD.
2. Otherwise (e.g. invalid JSON from a malformed auxiliary field, a misspelled decision-field
   name, or an invalid placeholder object followed by a corrected one), the response is read
   if every quoted key-value pair whose value is an allowed label names the same label, e.g.
   ``"complaint_outcome": "UPHELD"``; two different labels, or none, make it unreadable.

``fully_formatted`` additionally requires the raw text to be exactly the requested object: the
five fields (decision field, confidence integer 0-100, reason non-empty string,
decisive_evidence 1-3 strings, missing_information strings), no other or duplicate keys and no
surrounding Markdown.

The caller (``src.run``) passes only complete, non-truncated, non-refused replies.
"""
import json
import re

CONDITIONS = ('C0', 'OPINION_U', 'OPINION_S', 'MASK', 'DEBIAS',
              'CUE_CARE_U', 'CUE_CARE_S', 'CUE_DIS_U', 'CUE_DIS_S')
DEFAULT_LABELS = {'UPHELD': 1, 'NOT_UPHELD': 0}   # complaint upheld -> resident support
MASK_LABELS = {'PARTY_A': 1, 'PARTY_B': 0}        # Party A is always the complainant
LABELS = {c: (MASK_LABELS if c == 'MASK' else DEFAULT_LABELS) for c in CONDITIONS}
FIELD = {c: ('verdict' if c == 'MASK' else 'complaint_outcome') for c in CONDITIONS}
AUXILIARY = ('confidence', 'reason', 'decisive_evidence', 'missing_information')
# A doubled negation occasionally written as the only decision of a single JSON object
# (Haiku 5.5) is read as NOT_UPHELD; it never counts as fully formatted.
SINGLE_OBJECT_ALIASES = {'NOT_NOT_UPHELD': 'NOT_UPHELD'}
_FENCE = re.compile(r'```(?:json)?[ \t]*\r?\n([\s\S]*?)\r?\n```', re.IGNORECASE)


class _Pairs(list):
    """JSON object kept as (key, value) pairs so that duplicate keys stay visible."""


def explicit_label(raw, field, labels):
    """The unique explicit decision label in ``raw``, or None."""
    if not isinstance(raw, str) or not raw.strip():
        return None
    text = raw.strip()
    fence = _FENCE.fullmatch(text)
    if fence:
        text = fence.group(1).strip()
    try:
        value = json.loads(text, object_pairs_hook=_Pairs)
    except json.JSONDecodeError:
        value = None
    if isinstance(value, _Pairs):
        values = [v for k, v in value if k == field]
        if len(values) == 1 and isinstance(values[0], str):
            label = SINGLE_OBJECT_ALIASES.get(values[0], values[0]) if field == 'complaint_outcome' else values[0]
            if label in labels:
                return label
    # Rule 2: every quoted label value in the text names the same label.
    choice = '|'.join(re.escape(x) for x in sorted(labels))
    found = set(re.findall(r'"[^"\n]{1,64}"\s*:\s*"(' + choice + r')"', text))
    return found.pop() if len(found) == 1 else None


def _unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate JSON key')
        result[key] = value
    return result


def fully_formatted(raw, condition):
    """True if ``raw`` is exactly the requested five-field JSON object."""
    field, labels = FIELD[condition], LABELS[condition]
    try:
        value = json.loads(raw, object_pairs_hook=_unique_keys)
    except (ValueError, TypeError):
        return False
    if not isinstance(value, dict) or set(value) != {field, *AUXILIARY} or value[field] not in labels:
        return False
    if type(value['confidence']) is not int or not 0 <= value['confidence'] <= 100:
        return False
    if not isinstance(value['reason'], str) or not value['reason'].strip():
        return False
    for key in ('decisive_evidence', 'missing_information'):
        if not isinstance(value[key], list) or not all(isinstance(v, str) for v in value[key]):
            return False
    return 1 <= len(value['decisive_evidence']) <= 3


def read_decision(raw_text, condition):
    """Return ``(underdog_support, fully_formatted)``.

    ``underdog_support`` is 1 (resident / Party A), 0 (landlord / Party B) or None when the
    response has no unique explicit decision; ``fully_formatted`` is a bool (False if None).
    """
    labels = LABELS[condition]
    label = explicit_label(raw_text, FIELD[condition], labels)
    if label is None:
        return None, False
    return labels[label], fully_formatted(raw_text, condition)
