"""Fail-closed extraction of a JSON object from raw model text.

Every model pass in this package (the structured-fix proposal, and the
fix-refuter gate added in a later sprint) answers in a single JSON object,
but real model output is never trusted at face value: it may wrap the
answer in a ` ```json ` fence, wrap it in a plain ` ``` ` fence, or simply
return prose instead of JSON at all. `extract_json_object` is the single,
shared choke point every schema parser in this package runs its raw model
output through first.

This module never raises for malformed input: anything that is not
extractable as a single top-level JSON object resolves to `None`, which
every caller in this package treats as an explicit "no usable output"
signal, never as an empty-but-valid result.

Both public extractors use the same strict framing contract: after stripping
surrounding whitespace, the *entire* response must be either one JSON object
literal or one fenced block whose body is one JSON object literal. Any prose
outside the JSON/fence, extra fenced block, concatenated second JSON value, or
trailing garbage returns `None`. This prevents a repository-controlled JSON
fence echoed in model narration from being mistaken for the authoritative
proposal. The proposal pass may ask its separate, tool-free finalizer for a
cleanly framed response; the parser itself never guesses.
"""

from __future__ import annotations

import json
import re
from typing import Optional

def extract_json_object(raw: object) -> Optional[dict]:
    """Extract exactly one whole-response JSON object, or return `None`.

    This proposal-parser entry point intentionally shares the refuter's
    strict framing rules. In particular, it never accepts a JSON fence amid
    narration, because that fence may have been echoed from untrusted
    repository content.
    """

    return extract_strict_json_object(raw)


# A single fenced code block spanning the *entire* (stripped) response, with
# nothing before or after it. `fullmatch` makes the whole-response requirement
# explicit; no substring scan is allowed.
_FULL_FENCE_RE = re.compile(r"```[^\n`]*\n(.*)```", re.DOTALL)


def _parse_exactly_one_json_object(text: str) -> Optional[dict]:
    """Parse `text` as JSON only if it is *exactly* one JSON object.

    This does not scan for candidate substrings: `json.loads` is handed the
    whole string as-is, so any trailing content after a complete JSON value
    (e.g. a second
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
    scalar instead of an object, or unparseable text. It is the shared choke
    point for proposal and fix-refuter output, both of which must fail closed
    on any ambiguity about what a model actually said.
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
