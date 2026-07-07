"""Tests for `dbt_fixer.model_output.extract_json_object`.

Covers the success path (bare JSON, ```json``` fenced, plain fenced, and
reasoning-with-multiple-fences-picks-the-last-one) plus the never-raises
failure paths: non-string input, unparseable text, and JSON that parses but
is not an object.
"""

from __future__ import annotations

from dbt_fixer.model_output import extract_json_object, extract_strict_json_object


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


# ---------------------------------------------------------------------------
# extract_strict_json_object: the fix-refuter gate's non-tolerant choke
# point. Unlike extract_json_object above, this must reject anything that
# is not *exactly* one JSON object (bare, or as the sole content of one
# fenced block) -- no digging through surrounding prose, no picking a
# "last" fence among several.
# ---------------------------------------------------------------------------


def test_strict_accepts_bare_json_object() -> None:
    raw = '{"refuted": false, "could_not_refute": true, "reason": "clean"}'

    result = extract_strict_json_object(raw)

    assert result == {"refuted": False, "could_not_refute": True, "reason": "clean"}


def test_strict_accepts_bare_json_object_with_surrounding_whitespace_only() -> None:
    raw = '  \n  {"refuted": false, "could_not_refute": true, "reason": "clean"}  \n  '

    result = extract_strict_json_object(raw)

    assert result == {"refuted": False, "could_not_refute": True, "reason": "clean"}


def test_strict_accepts_single_json_tagged_fenced_block() -> None:
    raw = """```json
{"refuted": false, "could_not_refute": true, "reason": "fenced, nothing else"}
```"""

    result = extract_strict_json_object(raw)

    assert result == {
        "refuted": False,
        "could_not_refute": True,
        "reason": "fenced, nothing else",
    }


def test_strict_accepts_single_plain_fenced_block() -> None:
    raw = """```
{"refuted": false, "could_not_refute": true, "reason": "plain fence"}
```"""

    result = extract_strict_json_object(raw)

    assert result == {
        "refuted": False,
        "could_not_refute": True,
        "reason": "plain fence",
    }


def test_strict_rejects_prose_before_a_fenced_json_block() -> None:
    raw = (
        "Sure thing, here is my analysis...\n"
        "```json\n"
        '{"refuted": false, "could_not_refute": true, "reason": "looks fine"}\n'
        "```"
    )

    assert extract_strict_json_object(raw) is None


def test_strict_rejects_prose_after_a_fenced_json_block() -> None:
    raw = (
        "```json\n"
        '{"refuted": false, "could_not_refute": true, "reason": "looks fine"}\n'
        "```\n"
        "Hope that helps!"
    )

    assert extract_strict_json_object(raw) is None


def test_strict_rejects_prose_on_both_sides_of_a_fenced_json_block() -> None:
    raw = (
        "Sure thing, here is my analysis...\n"
        "```json\n"
        '{"refuted": false, "could_not_refute": true, "reason": "looks fine"}\n'
        "```\n"
        "Hope that helps!"
    )

    assert extract_strict_json_object(raw) is None


def test_strict_rejects_multiple_fenced_json_objects() -> None:
    raw = (
        "```json\n"
        '{"refuted": false, "could_not_refute": false, "reason": "draft"}\n'
        "```\n"
        "```json\n"
        '{"refuted": false, "could_not_refute": true, "reason": "final"}\n'
        "```"
    )

    assert extract_strict_json_object(raw) is None


def test_strict_rejects_two_bare_json_objects_concatenated() -> None:
    raw = (
        '{"refuted": false, "could_not_refute": false, "reason": "draft"}'
        '{"refuted": false, "could_not_refute": true, "reason": "final"}'
    )

    assert extract_strict_json_object(raw) is None


def test_strict_rejects_two_bare_json_objects_on_separate_lines() -> None:
    raw = (
        '{"refuted": false, "could_not_refute": false, "reason": "draft"}\n'
        '{"refuted": false, "could_not_refute": true, "reason": "final"}'
    )

    assert extract_strict_json_object(raw) is None


def test_strict_rejects_unparseable_prose() -> None:
    assert extract_strict_json_object("this is not json at all, just prose") is None


def test_strict_rejects_non_string_input() -> None:
    assert extract_strict_json_object(None) is None
    assert extract_strict_json_object(42) is None
    assert extract_strict_json_object(["not", "a", "string"]) is None


def test_strict_rejects_json_array_or_scalar() -> None:
    assert extract_strict_json_object("[1, 2, 3]") is None
    assert extract_strict_json_object("42") is None
    assert extract_strict_json_object('"just a string"') is None


def test_strict_rejects_empty_or_whitespace_only_string() -> None:
    assert extract_strict_json_object("") is None
    assert extract_strict_json_object("   \n  ") is None


def test_strict_rejects_empty_fenced_block() -> None:
    assert extract_strict_json_object("```json\n\n```") is None
