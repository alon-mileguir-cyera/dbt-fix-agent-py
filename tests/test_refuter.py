"""Tests for `dbt_fixer.refuter`: the Fix-Refuter Gate, fully offline.

Every test injects a fake `RefuterRunner` (a plain callable) -- never a
real model call -- matching the `conftest.py`-enforced offline contract.
The only "real" primitive exercised here is the bounded-timeout wrapper's
daemon thread, which never touches the network or a subprocess.
"""

from __future__ import annotations

import json
import time

import pytest

from dbt_fixer.fencing import fence_context
from dbt_fixer.refuter import (
    RefuterResponse,
    build_refuter_prompt,
    parse_refuter_response,
    run_fix_refuter_gate,
)

_DO_NOTHING_DIFF = "--- a/models/a.sql\n+++ b/models/a.sql\n"
_WHITESPACE_DIFF = "   \n\t\n"


def _fenced_context():
    return fence_context(
        {
            "pr_url": "https://github.com/example/repo/pull/1",
            "pr_title": "Fix broken not_null test",
            "pr_description": "restores a deleted line",
            "pr_diff": "--- a/models/a.sql\n+++ b/models/a.sql\n-select 1\n",
            "failure_context": "- check: not_null_a_id\n  status: FAIL\n",
        }
    )


def _confident_pass_response() -> str:
    return json.dumps(
        {
            "refuted": False,
            "could_not_refute": True,
            "reason": "the diff restores exactly the deleted line and nothing else",
        }
    )


def _confident_refutation_response(reason: str = "the diff is a no-op") -> str:
    return json.dumps({"refuted": True, "could_not_refute": False, "reason": reason})


# ---------------------------------------------------------------------------
# refuter_kills_do_nothing_patch
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("candidate_diff", [_DO_NOTHING_DIFF, _WHITESPACE_DIFF, "", "   "])
@pytest.mark.parametrize("trial", range(10))
def test_refuter_kills_do_nothing_or_whitespace_candidate(candidate_diff, trial):
    runner = lambda prompt: _confident_refutation_response("this diff changes nothing")
    verdict = run_fix_refuter_gate(
        fenced_context=_fenced_context(),
        candidate_diff=candidate_diff,
        refuter_runner=runner,
        timeout_seconds=5.0,
    )
    assert verdict.passed is False
    assert verdict.refuted is True


def test_refuter_passes_a_genuinely_confident_response():
    runner = lambda prompt: _confident_pass_response()
    verdict = run_fix_refuter_gate(
        fenced_context=_fenced_context(),
        candidate_diff="--- a/models/a.sql\n+++ b/models/a.sql\n+select 1\n",
        refuter_runner=runner,
        timeout_seconds=5.0,
    )
    assert verdict.passed is True
    assert verdict.refuted is False


# ---------------------------------------------------------------------------
# refuter_default_refuted_on_ambiguity
# ---------------------------------------------------------------------------


def _hedged_response() -> str:
    return json.dumps(
        {
            "refuted": False,
            "could_not_refute": False,
            "reason": "it might be fine but I'm not fully sure",
        }
    )


def _partial_response() -> str:
    # Missing the required "reason" key.
    return json.dumps({"refuted": False, "could_not_refute": True})


def _malformed_json_response() -> str:
    return "this is not json at all, just prose"


def _empty_response() -> str:
    return ""


def _both_true_response() -> str:
    return json.dumps({"refuted": True, "could_not_refute": True, "reason": "contradictory"})


@pytest.mark.parametrize(
    "runner_factory",
    [
        lambda: (lambda prompt: _hedged_response()),
        lambda: (lambda prompt: _partial_response()),
        lambda: (lambda prompt: _malformed_json_response()),
        lambda: (lambda prompt: _empty_response()),
        lambda: (lambda prompt: _both_true_response()),
        lambda: (lambda prompt: (_ for _ in ()).throw(RuntimeError("boom"))),
    ],
)
def test_refuter_defaults_to_refuted_on_every_ambiguous_or_failing_mode(runner_factory):
    verdict = run_fix_refuter_gate(
        fenced_context=_fenced_context(),
        candidate_diff="--- a/models/a.sql\n+++ b/models/a.sql\n+select 1\n",
        refuter_runner=runner_factory(),
        timeout_seconds=5.0,
    )
    assert verdict.passed is False
    assert verdict.refuted is True


# ---------------------------------------------------------------------------
# refuter_requires_strict_json_schema
# ---------------------------------------------------------------------------


def test_parse_rejects_missing_could_not_refute_flag():
    raw = json.dumps({"refuted": False, "reason": "looks fine"})
    assert parse_refuter_response(raw) is None


def test_parse_rejects_extra_top_level_key():
    raw = json.dumps(
        {
            "refuted": False,
            "could_not_refute": True,
            "reason": "fine",
            "confidence": "high",
        }
    )
    assert parse_refuter_response(raw) is None


def test_parse_rejects_wrong_types():
    raw = json.dumps({"refuted": "false", "could_not_refute": True, "reason": "fine"})
    assert parse_refuter_response(raw) is None


def test_parse_rejects_non_dict_json():
    assert parse_refuter_response(json.dumps(["refuted", True])) is None


def test_parse_accepts_exact_schema():
    raw = json.dumps({"refuted": False, "could_not_refute": True, "reason": "clean"})
    parsed = parse_refuter_response(raw)
    assert parsed == RefuterResponse(refuted=False, could_not_refute=True, reason="clean")


def test_parse_rejects_prose_wrapped_fenced_json():
    # The exact regression case from evaluation feedback: a model that
    # hedges with commentary around an otherwise schema-valid fenced JSON
    # object must not be accepted -- the refuter's strict-JSON contract has
    # no tolerance for surrounding prose, unlike the Sprint 2 proposal
    # parser's deliberately tolerant `extract_json_object`.
    raw = (
        "Sure thing, here is my analysis...\n"
        "```json\n"
        '{"refuted": false, "could_not_refute": true, "reason": "looks fine"}\n'
        "```\n"
        "Hope that helps!"
    )
    assert parse_refuter_response(raw) is None


def test_parse_rejects_multiple_fenced_json_objects():
    raw = (
        "```json\n"
        '{"refuted": false, "could_not_refute": false, "reason": "draft"}\n'
        "```\n"
        "```json\n"
        '{"refuted": false, "could_not_refute": true, "reason": "final"}\n'
        "```"
    )
    assert parse_refuter_response(raw) is None


def test_gate_treats_prose_wrapped_fenced_json_as_refuted():
    # Full end-to-end reproduction via the gate entrypoint: a runner
    # returning valid-looking JSON wrapped in chatty prose must resolve to
    # a failed (refuted) verdict, never `passed=True`.
    runner = lambda prompt: (
        "Sure thing, here is my analysis...\n"
        "```json\n"
        '{"refuted": false, "could_not_refute": true, "reason": "looks fine"}\n'
        "```\n"
        "Hope that helps!"
    )
    verdict = run_fix_refuter_gate(
        fenced_context=_fenced_context(),
        candidate_diff="--- a/models/a.sql\n+++ b/models/a.sql\n+select 1\n",
        refuter_runner=runner,
        timeout_seconds=5.0,
    )
    assert verdict.passed is False
    assert verdict.refuted is True


def test_gate_treats_multiple_json_objects_as_refuted():
    runner = lambda prompt: (
        '{"refuted": false, "could_not_refute": false, "reason": "draft"}\n'
        '{"refuted": false, "could_not_refute": true, "reason": "final"}'
    )
    verdict = run_fix_refuter_gate(
        fenced_context=_fenced_context(),
        candidate_diff="--- a/models/a.sql\n+++ b/models/a.sql\n+select 1\n",
        refuter_runner=runner,
        timeout_seconds=5.0,
    )
    assert verdict.passed is False
    assert verdict.refuted is True


def test_gate_treats_schema_violation_as_refuted():
    runner = lambda prompt: json.dumps({"refuted": False, "reason": "missing flag"})
    verdict = run_fix_refuter_gate(
        fenced_context=_fenced_context(),
        candidate_diff="--- a/models/a.sql\n+++ b/models/a.sql\n+select 1\n",
        refuter_runner=runner,
        timeout_seconds=5.0,
    )
    assert verdict.passed is False
    assert verdict.refuted is True


# ---------------------------------------------------------------------------
# refuter_uses_fresh_context_and_fenced_inputs
# ---------------------------------------------------------------------------


def test_prompt_contains_fenced_failure_context_and_fenced_candidate_diff():
    fenced = _fenced_context()
    candidate_diff = "--- a/models/a.sql\n+++ b/models/a.sql\n+select 1\n"
    prompt = build_refuter_prompt(fenced, candidate_diff)

    assert fenced.render() in prompt
    assert f"<<<UNTRUSTED:candidate_diff:{fenced.nonce}>>>" in prompt
    assert candidate_diff in prompt
    assert f"<<<END_UNTRUSTED:candidate_diff:{fenced.nonce}>>>" in prompt


def test_prompt_is_freshly_built_each_call_with_no_carried_state():
    fenced = _fenced_context()
    prompt_one = build_refuter_prompt(fenced, "--- a\n+++ b\n+one\n")
    prompt_two = build_refuter_prompt(fenced, "--- a\n+++ b\n+two\n")

    # Each call is a fully self-contained prompt string built purely from
    # its own arguments -- no accumulation of prior candidate diffs.
    assert "+one" not in prompt_two
    assert "+two" not in prompt_one


def test_gate_invokes_runner_with_the_freshly_built_prompt():
    captured = {}

    def runner(prompt: str) -> str:
        captured["prompt"] = prompt
        return _confident_pass_response()

    fenced = _fenced_context()
    candidate_diff = "--- a/models/a.sql\n+++ b/models/a.sql\n+select 1\n"
    run_fix_refuter_gate(
        fenced_context=fenced,
        candidate_diff=candidate_diff,
        refuter_runner=runner,
        timeout_seconds=5.0,
    )
    assert candidate_diff in captured["prompt"]
    assert fenced.render() in captured["prompt"]


# ---------------------------------------------------------------------------
# refuter_bounded_timeout_enforced
# ---------------------------------------------------------------------------


def test_runner_blocking_past_timeout_resolves_to_refuted():
    def slow_runner(prompt: str) -> str:
        time.sleep(0.5)
        return _confident_pass_response()

    start = time.monotonic()
    verdict = run_fix_refuter_gate(
        fenced_context=_fenced_context(),
        candidate_diff="--- a\n+++ b\n+x\n",
        refuter_runner=slow_runner,
        timeout_seconds=0.05,
    )
    elapsed = time.monotonic() - start

    assert verdict.passed is False
    assert verdict.refuted is True
    assert "timeout" in verdict.reason.lower() or "did not respond" in verdict.reason.lower()
    # The gate must return promptly, not wait for the slow runner to finish.
    assert elapsed < 0.4


def test_runner_blocking_forever_resolves_to_refuted_without_hanging_the_test():
    def hangs_forever(prompt: str) -> str:
        event = __import__("threading").Event()
        event.wait()  # never set; would block forever if not on a daemon thread
        return "unreachable"

    start = time.monotonic()
    verdict = run_fix_refuter_gate(
        fenced_context=_fenced_context(),
        candidate_diff="--- a\n+++ b\n+x\n",
        refuter_runner=hangs_forever,
        timeout_seconds=0.05,
    )
    elapsed = time.monotonic() - start

    assert verdict.passed is False
    assert verdict.refuted is True
    assert elapsed < 0.4


def test_runner_comfortably_under_timeout_completes_normally():
    def fast_runner(prompt: str) -> str:
        time.sleep(0.01)
        return _confident_pass_response()

    verdict = run_fix_refuter_gate(
        fenced_context=_fenced_context(),
        candidate_diff="--- a\n+++ b\n+x\n",
        refuter_runner=fast_runner,
        timeout_seconds=5.0,
    )
    assert verdict.passed is True
    assert verdict.refuted is False


# ---------------------------------------------------------------------------
# narration-around-JSON rescue (mirror of the proposal finalization fix)
# ---------------------------------------------------------------------------


def _fenced_ok(reason="no flaw"):
    import json
    return json.dumps({"refuted": False, "could_not_refute": True, "reason": reason})


def test_narrated_refuter_answer_is_rescued_by_finalization():
    from dbt_fixer.fencing import fence_context
    from dbt_fixer.refuter import run_fix_refuter_gate

    calls = []
    def runner(prompt):
        calls.append(prompt)
        if len(calls) == 1:
            return "Based on my investigation:\n" + _fenced_ok()  # preamble breaks strict parse
        return _fenced_ok()  # clean on the finalization re-prompt

    verdict = run_fix_refuter_gate(
        fenced_context=fence_context({"failure_context": "x"}),
        candidate_diff="diff",
        refuter_runner=runner,
        timeout_seconds=30.0,
    )
    assert verdict.passed and not verdict.refuted
    assert len(calls) == 2
    assert "previous response did not consist solely" in calls[1]


def test_rescued_refutation_still_refutes():
    from dbt_fixer.fencing import fence_context
    from dbt_fixer.refuter import run_fix_refuter_gate
    import json

    def runner(prompt):
        # narrated first, then a clean but REFUTING answer -> must fail closed
        if "previous response" in prompt:
            return json.dumps({"refuted": True, "could_not_refute": False, "reason": "drift found"})
        return "Here is my finding: " + json.dumps(
            {"refuted": True, "could_not_refute": False, "reason": "drift found"}
        )

    verdict = run_fix_refuter_gate(
        fenced_context=fence_context({"failure_context": "x"}),
        candidate_diff="diff",
        refuter_runner=runner,
        timeout_seconds=30.0,
    )
    assert not verdict.passed and verdict.refuted


def test_second_miss_still_fails_closed():
    from dbt_fixer.fencing import fence_context
    from dbt_fixer.refuter import run_fix_refuter_gate

    verdict = run_fix_refuter_gate(
        fenced_context=fence_context({"failure_context": "x"}),
        candidate_diff="diff",
        refuter_runner=lambda p: "narration, never any json at all",
        timeout_seconds=30.0,
    )
    assert not verdict.passed and verdict.refuted
