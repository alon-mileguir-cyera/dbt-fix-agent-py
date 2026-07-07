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

**Two distinct extraction contracts.** `extract_json_object` above is
deliberately *tolerant*: it is the Sprint 2 structured-fix proposal pass's
parser, and it is allowed to dig a JSON object out of surrounding prose or
pick the last of several fenced blocks, because the proposal pass is not
the safety backstop -- the allowlist, re-audit, and refuter gates are.
`extract_strict_json_object` is the opposite contract, used only by the
fix-refuter gate (`dbt_fixer.refuter`): it accepts a response only when,
after stripping surrounding whitespace, the *entire* response is either a
single JSON object literal or a single fenced block whose body is a single
JSON object literal, and nothing else. Any prose outside the JSON/fence,
any extra fenced block, any concatenated second JSON value, or any trailing
garbage causes strict extraction to return `None` -- there is no
best-effort fallback here, because the refuter's entire purpose is to be
the strict, adversarial, fail-closed check, and a model that hedges with
commentary around otherwise valid-looking JSON must not be rewarded with a
parse.
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


# A single fenced code block spanning the *entire* (stripped) response, with
# nothing before or after it -- unlike `_FENCE_BLOCK_RE`, this is anchored
# with `^`/`$` (via `fullmatch`) so it only matches when the fence is the
# whole response, not merely a substring of it.
_FULL_FENCE_RE = re.compile(r"```[^\n`]*\n(.*)```", re.DOTALL)


def _parse_exactly_one_json_object(text: str) -> Optional[dict]:
    """Parse `text` as JSON only if it is *exactly* one JSON object.

    Unlike the tolerant path above, this does not scan for candidate
    substrings: `json.loads` is handed the whole string as-is, so any
    trailing content after a complete JSON value (e.g. a second
    concatenated JSON object, or trailing prose) raises `JSONDecodeError`
    ("Extra data") and is correctly rejected rather than silently ignored.
    """

    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, ValueError, TypeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def extract_strict_json_object(raw: object) -> Optional[dict]:
    """Extract a JSON object only when `raw` is *exactly* that, nothing else.

    Accepts, after stripping leading/trailing whitespace:

    1. A bare JSON object literal spanning the whole string (e.g.
       `'{"a": 1}'`), or
    2. A single fenced code block (```` ``` ```` or ` ```json `) spanning
       the whole string, whose body is itself exactly one JSON object.

    Rejects -- always returning `None`, never raising -- any response that
    has prose before or after the JSON/fence, more than one fenced block,
    more than one JSON object (concatenated or otherwise), a JSON array or
    scalar instead of an object, or unparseable text. This is intentionally
    stricter than `extract_json_object`: it is the choke point for the
    fix-refuter gate, which must fail closed on any ambiguity about what a
    model actually said.
    """

    if not isinstance(raw, str):
        return None

    text = raw.strip()
    if not text:
        return None

    fence_match = _FULL_FENCE_RE.fullmatch(text)
    if fence_match is not None:
        body = fence_match.group(1).strip()
        if not body:
            return None
        return _parse_exactly_one_json_object(body)

    # No single fence spans the whole response. If the response contains a
    # fence marker at all (e.g. multiple fenced blocks, or a fence plus
    # surrounding prose), it cannot be "exactly one JSON object" by this
    # strict contract's rules -- reject outright rather than falling back to
    # scanning for a substring.
    if "```" in text:
        return None

    return _parse_exactly_one_json_object(text)
