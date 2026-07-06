"""Tests for `dbt_fixer.bounds`: the env-override parsing and the
`ExecutionBudget` primitive's independent, simultaneous enforcement of the
wall-clock timeout, tool-call cap, and turn limit.
"""

from __future__ import annotations

import pytest

from dbt_fixer.bounds import (
    Bounds,
    DEFAULT_MAX_TOOL_CALLS,
    DEFAULT_MAX_TURNS,
    DEFAULT_TIMEOUT_SECONDS,
    ENV_MAX_TOOL_CALLS,
    ENV_MAX_TURNS,
    ENV_TIMEOUT_SECONDS,
    ExecutionBudget,
    TimeoutExceededError,
    ToolCallCapExceededError,
    TurnLimitExceededError,
    load_bounds,
)


class FakeClock:
    """A deterministic, manually-advanced clock for testing time-based limits."""

    def __init__(self, start: float = 0.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, delta: float) -> None:
        self.now += delta


# --- env parsing -------------------------------------------------------------


def test_defaults_when_unset():
    bounds, warnings = load_bounds({})
    assert bounds == Bounds(
        timeout_seconds=DEFAULT_TIMEOUT_SECONDS,
        max_tool_calls=DEFAULT_MAX_TOOL_CALLS,
        max_turns=DEFAULT_MAX_TURNS,
    )
    assert warnings == ()


@pytest.mark.parametrize("bad_value", ["not-a-number", "-5", "0"])
def test_malformed_timeout_falls_back(bad_value):
    bounds, warnings = load_bounds({ENV_TIMEOUT_SECONDS: bad_value})
    assert bounds.timeout_seconds == DEFAULT_TIMEOUT_SECONDS
    assert warnings and ENV_TIMEOUT_SECONDS in warnings[0]


def test_out_of_range_timeout_falls_back():
    bounds, warnings = load_bounds({ENV_TIMEOUT_SECONDS: "999999"})
    assert bounds.timeout_seconds == DEFAULT_TIMEOUT_SECONDS
    assert warnings


@pytest.mark.parametrize(
    "bad_value",
    ["nan", "NaN", "+nan", "-nan", " nan ", "inf", "-inf", "Infinity", "1e400"],
)
def test_non_finite_timeout_falls_back_and_never_disables_the_timeout(bad_value):
    """Regression test: NaN/inf-adjacent float strings must never bypass the
    range check via IEEE-754 comparison semantics (every ordering comparison
    against NaN is False, so a naive `value < min or value > max` check would
    silently treat NaN as "in range"). A live NaN timeout bound would make
    `ExecutionBudget.check_timeout()` never fire, disabling the wall-clock
    timeout entirely -- exactly the hang-forever failure mode this primitive
    exists to prevent.
    """

    bounds, warnings = load_bounds({ENV_TIMEOUT_SECONDS: bad_value})
    assert bounds.timeout_seconds == DEFAULT_TIMEOUT_SECONDS
    assert warnings and ENV_TIMEOUT_SECONDS in warnings[0]

    # Also prove the resulting budget actually enforces the timeout: with a
    # NaN bound bypassing the guard, this would never raise.
    clock = FakeClock()
    budget = ExecutionBudget(bounds, clock=clock)
    clock.advance(DEFAULT_TIMEOUT_SECONDS + 1)
    with pytest.raises(TimeoutExceededError):
        budget.check_timeout()


@pytest.mark.parametrize("bad_value", ["nope", "-1", "0"])
def test_malformed_max_tool_calls_falls_back(bad_value):
    bounds, warnings = load_bounds({ENV_MAX_TOOL_CALLS: bad_value})
    assert bounds.max_tool_calls == DEFAULT_MAX_TOOL_CALLS
    assert warnings


@pytest.mark.parametrize("bad_value", ["nope", "-1", "0"])
def test_malformed_max_turns_falls_back(bad_value):
    bounds, warnings = load_bounds({ENV_MAX_TURNS: bad_value})
    assert bounds.max_turns == DEFAULT_MAX_TURNS
    assert warnings


def test_valid_overrides_are_respected_with_no_warnings():
    bounds, warnings = load_bounds(
        {ENV_TIMEOUT_SECONDS: "10", ENV_MAX_TOOL_CALLS: "3", ENV_MAX_TURNS: "2"}
    )
    assert bounds.timeout_seconds == 10.0
    assert bounds.max_tool_calls == 3
    assert bounds.max_turns == 2
    assert warnings == ()


def test_blank_values_use_defaults_without_warning():
    bounds, warnings = load_bounds(
        {ENV_TIMEOUT_SECONDS: "  ", ENV_MAX_TOOL_CALLS: "", ENV_MAX_TURNS: "   "}
    )
    assert bounds.timeout_seconds == DEFAULT_TIMEOUT_SECONDS
    assert bounds.max_tool_calls == DEFAULT_MAX_TOOL_CALLS
    assert bounds.max_turns == DEFAULT_MAX_TURNS
    assert warnings == ()


# --- ExecutionBudget: independent, simultaneous enforcement -----------------


def test_timeout_enforced_via_check_timeout():
    clock = FakeClock()
    budget = ExecutionBudget(Bounds(timeout_seconds=5, max_tool_calls=100, max_turns=100), clock=clock)
    clock.advance(6)
    with pytest.raises(TimeoutExceededError):
        budget.check_timeout()


def test_timeout_not_yet_exceeded_does_not_raise():
    clock = FakeClock()
    budget = ExecutionBudget(Bounds(timeout_seconds=5, max_tool_calls=100, max_turns=100), clock=clock)
    clock.advance(4.9)
    budget.check_timeout()  # must not raise


def test_tool_call_cap_enforced_deterministically():
    clock = FakeClock()
    budget = ExecutionBudget(Bounds(timeout_seconds=1000, max_tool_calls=2, max_turns=100), clock=clock)
    assert budget.record_tool_call() == 1
    assert budget.record_tool_call() == 2
    with pytest.raises(ToolCallCapExceededError):
        budget.record_tool_call()
    assert budget.tool_calls_used == 3  # the failed attempt still counted before raising


def test_turn_limit_enforced_deterministically():
    clock = FakeClock()
    budget = ExecutionBudget(Bounds(timeout_seconds=1000, max_tool_calls=100, max_turns=2), clock=clock)
    assert budget.record_turn() == 1
    assert budget.record_turn() == 2
    with pytest.raises(TurnLimitExceededError):
        budget.record_turn()


def test_timeout_takes_precedence_over_tool_call_recording():
    clock = FakeClock()
    budget = ExecutionBudget(Bounds(timeout_seconds=5, max_tool_calls=100, max_turns=100), clock=clock)
    clock.advance(10)
    with pytest.raises(TimeoutExceededError):
        budget.record_tool_call()
    # the timeout check happens before the counter increments
    assert budget.tool_calls_used == 0


def test_timeout_takes_precedence_over_turn_recording():
    clock = FakeClock()
    budget = ExecutionBudget(Bounds(timeout_seconds=5, max_tool_calls=100, max_turns=100), clock=clock)
    clock.advance(10)
    with pytest.raises(TimeoutExceededError):
        budget.record_turn()
    assert budget.turns_used == 0


def test_all_three_limits_are_independent():
    clock = FakeClock()
    bounds = Bounds(timeout_seconds=1000, max_tool_calls=1, max_turns=1)
    budget = ExecutionBudget(bounds, clock=clock)
    budget.record_tool_call()
    budget.record_turn()
    # both caps are now individually exhausted; each raises its own named error
    with pytest.raises(ToolCallCapExceededError):
        budget.record_tool_call()
    with pytest.raises(TurnLimitExceededError):
        budget.record_turn()


def test_execution_budget_defaults_to_real_monotonic_clock():
    budget = ExecutionBudget(Bounds(timeout_seconds=1000, max_tool_calls=10, max_turns=10))
    # must not raise; proves the default clock works without a fake
    budget.check_timeout()
    assert budget.elapsed_seconds >= 0
