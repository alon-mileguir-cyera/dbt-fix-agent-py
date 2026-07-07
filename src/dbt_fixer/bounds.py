"""The shared bounded-execution primitive.

Every model-calling pass in this package (structured-fix proposal, the
fix-refuter gate, ...) must run *through* this module's `ExecutionBudget`,
never around it. It enforces three independent limits simultaneously so a
stalled or runaway pass fails deterministically instead of hanging or
looping forever:

- **Wall-clock timeout** (`DBT_FIXER_TIMEOUT_SECONDS`, default 300s):
  checked on every `record_tool_call` / `record_turn`, and independently via
  `check_timeout`.
- **Tool-call cap** (`DBT_FIXER_MAX_TOOL_CALLS`, default 40): the maximum
  number of repo-tool invocations (read/search) one pass may make.
- **Turn limit** (`DBT_FIXER_MAX_TURNS`, default 8): the maximum number of
  model turns one pass may take.

Environment-contract table (this module's slice of it -- see `dbt_fixer.env`
for the rest):

| Variable                        | Required | Default | Valid range   | Malformed handling                        |
|-----------------------------------|----------|---------|----------------|---------------------------------------------|
| `DBT_FIXER_TIMEOUT_SECONDS`       | no       | `300`   | `[1, 3600]`    | falls back to `300`, records a warning       |
| `DBT_FIXER_MAX_TOOL_CALLS`        | no       | `40`    | `[1, 500]`     | falls back to `40`, records a warning        |
| `DBT_FIXER_MAX_TURNS`             | no       | `8`     | `[1, 100]`     | falls back to `8`, records a warning         |

All three variables are optional-by-design and *never* raise: a malformed
value degrades to the documented default (never clamps, never crashes),
exactly like every other numeric bound in this package (see
`dbt_fixer._numeric`).

The primitive itself is deliberately clock-injectable (`clock` parameter)
so tests can simulate exceeding any bound without a real sleep -- there is
no code path in this module that performs an actual wall-clock wait.

**`run_with_hard_timeout`.** Every external-boundary call this package
makes -- a model runner (the fix-refuter gate, the proposal pass) or a
subprocess runner (the re-audit gate, the dbt parse gate) -- is bounded not
just by the cooperative `ExecutionBudget` above but by this module's other
primitive: a *hard*, interrupting wall-clock timeout that does not depend
on the callee cooperating at all. `run_with_hard_timeout` runs the given
zero-argument callable in a daemon background thread and returns as soon
as either the callable finishes or `timeout_seconds` elapses, whichever is
first -- so a callee that blocks, sleeps, or hangs forever (a stalled model
API call, a `dbt parse` invocation stuck on a network mount, a
deliberately-adversarial test fake) can never hang the calling thread, and
never blocks process/interpreter exit either (the worker thread is a
daemon). Every module in this package that needs a hard timeout around an
untrusted external call routes through this one function, so "bounded by a
timeout" always means the exact same enforcement mechanism everywhere.
"""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from typing import Callable, List, Mapping, Optional, Tuple

from ._numeric import parse_bounded_number

ENV_TIMEOUT_SECONDS = "DBT_FIXER_TIMEOUT_SECONDS"
ENV_MAX_TOOL_CALLS = "DBT_FIXER_MAX_TOOL_CALLS"
ENV_MAX_TURNS = "DBT_FIXER_MAX_TURNS"

DEFAULT_TIMEOUT_SECONDS: float = 300.0
DEFAULT_MAX_TOOL_CALLS: int = 40
DEFAULT_MAX_TURNS: int = 8

_TIMEOUT_RANGE: Tuple[float, float] = (1.0, 3600.0)
_MAX_TOOL_CALLS_RANGE: Tuple[int, int] = (1, 500)
_MAX_TURNS_RANGE: Tuple[int, int] = (1, 100)


class BoundedExecutionError(RuntimeError):
    """Base class for every limit this primitive enforces."""


class TimeoutExceededError(BoundedExecutionError):
    """The wall-clock timeout for this pass has been exceeded."""


class ToolCallCapExceededError(BoundedExecutionError):
    """The maximum number of repo-tool calls for this pass has been exceeded."""


class TurnLimitExceededError(BoundedExecutionError):
    """The maximum number of model turns for this pass has been exceeded."""


@dataclass(frozen=True)
class Bounds:
    """The three independent limits one bounded pass must respect."""

    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    max_tool_calls: int = DEFAULT_MAX_TOOL_CALLS
    max_turns: int = DEFAULT_MAX_TURNS


def load_bounds(env: Optional[Mapping[str, str]] = None) -> Tuple[Bounds, Tuple[str, ...]]:
    """Parse the three bound-override env vars into a `Bounds`.

    Never raises. Returns the parsed bounds plus a tuple of human-readable
    warnings for every value that fell back to its default because it was
    present but malformed (missing/blank values are not warned about).
    """

    if env is None:
        import os

        env = os.environ

    warnings: List[str] = []
    timeout_seconds = parse_bounded_number(
        env,
        ENV_TIMEOUT_SECONDS,
        default=DEFAULT_TIMEOUT_SECONDS,
        min_value=_TIMEOUT_RANGE[0],
        max_value=_TIMEOUT_RANGE[1],
        warnings=warnings,
        caster=float,
    )
    max_tool_calls = parse_bounded_number(
        env,
        ENV_MAX_TOOL_CALLS,
        default=DEFAULT_MAX_TOOL_CALLS,
        min_value=_MAX_TOOL_CALLS_RANGE[0],
        max_value=_MAX_TOOL_CALLS_RANGE[1],
        warnings=warnings,
        caster=int,
    )
    max_turns = parse_bounded_number(
        env,
        ENV_MAX_TURNS,
        default=DEFAULT_MAX_TURNS,
        min_value=_MAX_TURNS_RANGE[0],
        max_value=_MAX_TURNS_RANGE[1],
        warnings=warnings,
        caster=int,
    )
    return (
        Bounds(
            timeout_seconds=timeout_seconds,
            max_tool_calls=max_tool_calls,
            max_turns=max_turns,
        ),
        tuple(warnings),
    )


class ExecutionBudget:
    """Tracks elapsed time, tool calls, and turns for one bounded pass.

    Every `record_tool_call()` / `record_turn()` call re-checks the
    wall-clock timeout first, so all three limits are enforced independently
    *and* simultaneously -- a pass that is under its tool-call cap but has
    blown its timeout still stops on the very next thing it tries to do.

    `clock` defaults to `time.monotonic` but is injectable so unit tests can
    simulate the passage of time deterministically, without a real sleep.
    """

    def __init__(self, bounds: Bounds, *, clock: Optional[Callable[[], float]] = None) -> None:
        import time

        self._bounds = bounds
        self._clock = clock or time.monotonic
        self._start = self._clock()
        self._tool_calls = 0
        self._turns = 0

    @property
    def bounds(self) -> Bounds:
        return self._bounds

    @property
    def elapsed_seconds(self) -> float:
        return self._clock() - self._start

    @property
    def tool_calls_used(self) -> int:
        return self._tool_calls

    @property
    def turns_used(self) -> int:
        return self._turns

    def check_timeout(self) -> None:
        """Raise `TimeoutExceededError` if the wall-clock timeout has passed."""

        elapsed = self.elapsed_seconds
        if elapsed > self._bounds.timeout_seconds:
            raise TimeoutExceededError(
                f"wall-clock timeout of {self._bounds.timeout_seconds}s exceeded "
                f"(elapsed={elapsed:.2f}s)"
            )

    def record_tool_call(self) -> int:
        """Record one repo-tool invocation; raise if the cap or timeout is exceeded."""

        self.check_timeout()
        self._tool_calls += 1
        if self._tool_calls > self._bounds.max_tool_calls:
            raise ToolCallCapExceededError(
                f"tool-call cap of {self._bounds.max_tool_calls} exceeded "
                f"(calls_made={self._tool_calls})"
            )
        return self._tool_calls

    def record_turn(self) -> int:
        """Record one model turn; raise if the turn limit or timeout is exceeded."""

        self.check_timeout()
        self._turns += 1
        if self._turns > self._bounds.max_turns:
            raise TurnLimitExceededError(
                f"turn limit of {self._bounds.max_turns} exceeded (turns_taken={self._turns})"
            )
        return self._turns


def run_with_hard_timeout(
    func: Callable[[], object], timeout_seconds: float
) -> Tuple[str, object]:
    """Invoke `func()` in a daemon thread, bounded by `timeout_seconds`.

    Returns a `(kind, value)` pair:

    - `("ok", return_value)` if `func` returned normally within the bound;
    - `("error", exception)` if `func` raised within the bound;
    - `("timeout", None)` if neither happened before `timeout_seconds`
      elapsed.

    The worker thread is a daemon, so a `func` that blocks, sleeps, or
    hangs forever (a stalled model call, a stuck subprocess, a
    deliberately-adversarial test fake) can never block this function
    itself past `timeout_seconds`, nor block process/interpreter exit.
    This is the one shared hard-interrupting-timeout mechanism every
    external-boundary call in this package must route through.
    """

    result_queue: "queue.Queue[tuple[str, object]]" = queue.Queue(maxsize=1)

    def _target() -> None:
        try:
            result_queue.put(("ok", func()))
        except Exception as exc:  # the callee is an untrusted external boundary
            result_queue.put(("error", exc))

    thread = threading.Thread(target=_target, daemon=True)
    thread.start()

    try:
        return result_queue.get(timeout=timeout_seconds)
    except queue.Empty:
        return ("timeout", None)
