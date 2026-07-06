"""Tests for `dbt_fixer.model_output.extract_json_object`.

Covers the success path (bare JSON, ```json``` fenced, plain fenced, and
reasoning-with-multiple-fences-picks-the-last-one) plus the never-raises
failure paths: non-string input, unparseable text, and JSON that parses but
is not an object.
"""

from __future__ import annotations

from dbt_fixer.model_output import extract_json_object


def test_extracts_bare_json_object() -> None:
    raw = '{"edits": [], "rationale": "no fix needed"}'

    result = extract_json_object(raw)

    assert result == {"edits": [], "rationale": "no fix needed"}


def test_extracts_json_tagged_fenced_block() -> None:
    raw = """Here is my answer:

```json
{"edits": [], "rationale": "fenced"}
```
"""

    result = extract_json_object(raw)

    assert result == {"edits": [], "rationale": "fenced"}


def test_extracts_plain_fenced_block_without_language_tag() -> None:
    raw = """```
{"edits": [], "rationale": "plain fence"}
```"""

    result = extract_json_object(raw)

    assert result == {"edits": [], "rationale": "plain fence"}


def test_prefers_last_json_tagged_fence_over_earlier_ones() -> None:
    raw = """Reasoning below.

```json
{"edits": [], "rationale": "draft one, discard"}
```

Actually, final answer:

```json
{"edits": [], "rationale": "final answer"}
```
"""

    result = extract_json_object(raw)

    assert result == {"edits": [], "rationale": "final answer"}


def test_prefers_json_tagged_fence_over_other_fenced_blocks() -> None:
    raw = """Some SQL I looked at:

```sql
select 1
```

```json
{"edits": [], "rationale": "the real answer"}
```
"""

    result = extract_json_object(raw)

    assert result == {"edits": [], "rationale": "the real answer"}


def test_returns_none_for_non_string_input() -> None:
    assert extract_json_object(None) is None
    assert extract_json_object(42) is None
    assert extract_json_object(["not", "a", "string"]) is None


def test_returns_none_for_unparseable_prose() -> None:
    raw = "I could not find a safe fix for this failure, sorry."

    assert extract_json_object(raw) is None


def test_returns_none_for_json_array_or_scalar() -> None:
    assert extract_json_object("[1, 2, 3]") is None
    assert extract_json_object("42") is None
    assert extract_json_object('"just a string"') is None


def test_returns_none_for_empty_or_whitespace_only_string() -> None:
    assert extract_json_object("") is None
    assert extract_json_object("   \n  ") is None
