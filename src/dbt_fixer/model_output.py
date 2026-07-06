"""Tolerant-but-never-trusting extraction of a JSON object from raw model text.

Every model pass in this package (the structured-fix proposal, and the
fix-refuter gate added in a later sprint) answers in a single JSON object,
but real model output is never trusted at face value: it may wrap the
answer in a ` ```json ` fence, wrap it in a plain ` ``` ` fence, emit
reasoning with earlier unrelated fenced blocks before the final answer, or
simply return prose instead of JSON at all. `extract_json_object` is the
single, shared choke point every schema parser in this package runs its raw
model output through first.

This module never raises for malformed input: anything that is not
extractable as a single top-level JSON object resolves to `None`, which
every caller in this package treats as an explicit "no usable output"
signal, never as an empty-but-valid result.
"""

from __future__ import annotations

import json
import re
from typing import Optional

# A fenced code block, capturing its optional language tag and its body.
# Every fenced block is harvested, not just the first -- a model often
# emits SQL/diff fences while reasoning before its final JSON answer, and
# grabbing the first fence would parse the wrong block.
_FENCE_BLOCK_RE = re.compile(r"```([^\n`]*)\n(.*?)```", re.DOTALL)


def _json_candidates(raw: object) -> list[str]:
    """Ordered list of substrings to try to parse as a JSON object.

    Priority (first one that parses as a JSON object wins):

    1. Any ` ```json `-tagged fenced block, last one first -- the final
       answer is the model's last word.
    2. Any other fenced block, last one first.
    3. The whole raw string (bare JSON with no fence at all).
    """

    if not isinstance(raw, str) or not raw.strip():
        return []

    json_tagged: list[str] = []
    other_fenced: list[str] = []
    for match in _FENCE_BLOCK_RE.finditer(raw):
        lang = match.group(1).strip().lower()
        body = match.group(2).strip()
        if not body:
            continue
        (json_tagged if lang == "json" else other_fenced).append(body)

    candidates: list[str] = []
    candidates.extend(reversed(json_tagged))
    candidates.extend(reversed(other_fenced))
    stripped = raw.strip()
    if stripped:
        candidates.append(stripped)
    return candidates


def extract_json_object(raw: object) -> Optional[dict]:
    """Best-effort extraction of a single top-level JSON object from `raw`.

    Tolerates a ` ```json ` fenced block wrapping the object (with
    surrounding prose outside the fence ignored), as well as reasoning that
    emits earlier fenced blocks before the final JSON answer.

    Returns `None` -- never raises -- for anything that is not a
    well-formed JSON *object*: scalars, arrays, and unparseable text all
    resolve to `None`, and free-form prose with no valid JSON anywhere in
    it resolves to `None` as well.
    """

    for candidate in _json_candidates(raw):
        try:
            parsed = json.loads(candidate)
        except (json.JSONDecodeError, ValueError, TypeError):
            continue
        if isinstance(parsed, dict):
            return parsed
    return None
