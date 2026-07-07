"""`python -m dbt_fixer.entrypoint` -- the package's one, always-exit-0 CLI surface.

This module is the outermost boundary of the whole package: everything
upstream of it (`dbt_fixer.env`, `dbt_fixer.pipeline`, and -- once later
sprints wire them in here -- the proposal/gate/delivery pipeline) already
converts its own expected failure modes into a typed, non-exceptional
result. This module's job is narrower and absolute: no matter what happens
underneath it -- a missing required env var, an unparseable failure
context, or a genuinely unexpected bug -- the process:

1. exits `0`, always;
2. emits *exactly one* line matching
   ``^dbt-fixer-status: (proposed|no_safe_fix|failed)$`` as the *last*
   line of stdout;
3. never lets a raw traceback become the last thing printed.

**Sprint 1 scope.** This entrypoint currently wires only Stage 1
(`dbt_fixer.pipeline.run_stage1`: environment validation + failure-context
intake). The structured-fix-proposal pass and the allowlist / re-audit /
fix-refuter / dbt-parse gates already exist as library modules
(`dbt_fixer.fix_pipeline`, `dbt_fixer.retry_loop`, ...) but are not yet
invoked from here -- a later sprint's contract adds that wiring. This means
`proposed` is architecturally unreachable from this entrypoint today: even
a run whose environment is valid and whose failure context parses cleanly
into a concrete target still resolves to `no_safe_fix`, with an honest
reason naming the identified target and stating that no fix pipeline ran.
This is a deliberate design constraint, not an oversight -- see the sprint
contract's note that "only failed/no-op paths are reachable this sprint."
"""

from __future__ import annotations

import sys
from typing import Mapping, Optional

from .logging_utils import get_logger
from .pipeline import run_stage1
from .status import (
    STDOUT_REASON_PREFIX,
    STDOUT_STATUS_PREFIX,
    RunResult,
)

__all__ = ["compute_run_result", "render_stdout_lines", "main"]

logger = get_logger("entrypoint")

_FALLBACK_STATUS = "failed"
_FALLBACK_REASON = (
    "an unexpected internal error occurred before a result could be computed"
)


def compute_run_result(env: Optional[Mapping[str, str]] = None) -> RunResult:
    """Compute this run's single, terminal `RunResult`. Never raises.

    Delegates to `dbt_fixer.pipeline.run_stage1`, which itself never
    raises; the `except Exception` below is a second, defensive backstop
    (this module's own fail-closed guarantee) in case a defect in the
    wiring here -- not in `run_stage1` itself -- produces an unexpected
    error.
    """

    try:
        outcome = run_stage1(env)
    except Exception as exc:  # pragma: no cover - defensive: run_stage1 never raises today
        logger.exception("unexpected error running stage 1: %s", exc)
        return RunResult(
            status="failed", reason=f"unexpected internal error in stage 1: {exc!r}"
        )

    if outcome.terminal is not None:
        return outcome.terminal

    # Environment validation and intake both succeeded: a concrete
    # `FailureTarget` was identified. This sprint's entrypoint does not yet
    # invoke the proposal/gate pipeline against that target, so the only
    # honest outcome is `no_safe_fix` -- never a guess, never `proposed`.
    try:
        target = outcome.intake.target if outcome.intake is not None else None
        identifiers = ", ".join(target.identifiers) if target is not None else ""
        identifiers = identifiers or "<unnamed>"
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("unexpected error rendering identified target: %s", exc)
        return RunResult(
            status="failed", reason=f"unexpected internal error describing target: {exc!r}"
        )

    return RunResult(
        status="no_safe_fix",
        reason=(
            f"identified a concrete failure target ({identifiers}) but this build's "
            "fix-proposal pipeline is not yet wired into the entrypoint; no fix was "
            "attempted"
        ),
    )


def _single_line(text: str) -> str:
    """Collapse `text` to one line so it can never masquerade as, or push
    past, the fixed-shape status line that must be the true last line of
    stdout."""

    return " ".join(text.split())


def render_stdout_lines(result: RunResult) -> list[str]:
    """Render `result` as the fixed stdout lines: an optional reason line,
    followed always by the single, line-anchored status line."""

    lines: list[str] = []
    if result.reason:
        lines.append(f"{STDOUT_REASON_PREFIX}: {_single_line(result.reason)}")
    lines.append(f"{STDOUT_STATUS_PREFIX}: {result.status}")
    return lines


def main(argv: Optional[list[str]] = None, env: Optional[Mapping[str, str]] = None) -> int:
    """Run one dbt_fixer pass and print its fixed stdout contract.

    Always returns `0`. Guarantees the status line is printed exactly once,
    as the last line of stdout, even if computing or rendering the result
    itself fails unexpectedly.
    """

    status = _FALLBACK_STATUS
    reason = _FALLBACK_REASON

    try:
        result = compute_run_result(env)
        status = result.status
        reason = result.reason
    except Exception as exc:  # pragma: no cover - defensive: compute_run_result never raises
        logger.exception("unhandled exception computing run result: %s", exc)

    try:
        if reason:
            print(f"{STDOUT_REASON_PREFIX}: {_single_line(reason)}")
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("failed to print reason line: %s", exc)

    # The status line is always the true last thing printed, regardless of
    # whether the reason line above succeeded.
    try:
        print(f"{STDOUT_STATUS_PREFIX}: {status}")
    except Exception as exc:  # pragma: no cover - stdout itself is broken; nothing left to do
        logger.exception("failed to print status line: %s", exc)

    return 0


if __name__ == "__main__":
    sys.exit(main())
