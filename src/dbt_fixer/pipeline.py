"""Stage 1 of the fixer pipeline: environment validation + failure-context intake.

This is the exact seam later sprints build the proposal/apply/gate pipeline
on top of: `run_stage1` either resolves a run to a terminal `RunResult`
right now (bad environment, or a failure_context that can't be parsed into
a target), or hands back a validated `FixerConfig` and a successfully
parsed `IntakeResult` for the next stage to consume.

Nothing upstream of this function is trusted to not raise -- `run_stage1`
itself never raises. Every exception `dbt_fixer.env.load_config` or
`dbt_fixer.intake.resolve_intake` could realistically produce is caught here
and converted into a typed, terminal `RunResult` with status `failed` (for
environment/config problems) so a malformed or corrupt input can never
surface as an unhandled stack trace.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

from .env import EnvValidationError, FixerConfig, load_config
from .intake import IntakeResult, resolve_intake
from .status import RunResult


@dataclass(frozen=True)
class Stage1Outcome:
    """The result of running Stage 1.

    `terminal is None` means: environment validation and intake both
    succeeded, a `FailureTarget` was found, and the run should proceed to
    the proposal pipeline (Sprint 2+). Any other case sets `terminal` to
    the `RunResult` this run must resolve to, and the caller should stop
    here.
    """

    config: Optional[FixerConfig]
    intake: Optional[IntakeResult]
    terminal: Optional[RunResult]


def run_stage1(env: Optional[Mapping[str, str]] = None) -> Stage1Outcome:
    """Validate the environment and parse the failure context.

    Never raises. Maps every failure mode to a terminal, typed `RunResult`:

    - Missing/invalid required environment variable, or any unexpected
      exception while validating it -> `status="failed"`.
    - An empty or unparseable `failure_context` (or any unexpected
      exception while parsing it) -> `status="no_safe_fix"`.
    - A successfully parsed target -> `terminal=None`; the caller proceeds.
    """

    try:
        config = load_config(env)
    except EnvValidationError as exc:
        return Stage1Outcome(
            config=None,
            intake=None,
            terminal=RunResult(status="failed", reason=f"environment validation failed: {exc}"),
        )
    except Exception as exc:  # defensive: env validation must never crash the process
        return Stage1Outcome(
            config=None,
            intake=None,
            terminal=RunResult(
                status="failed", reason=f"unexpected error validating environment: {exc!r}"
            ),
        )

    try:
        intake = resolve_intake(config)
    except Exception as exc:  # defensive: intake must never crash the process
        return Stage1Outcome(
            config=config,
            intake=None,
            terminal=RunResult(status="failed", reason=f"unexpected error during intake: {exc!r}"),
        )

    if not intake.ok:
        return Stage1Outcome(
            config=config,
            intake=intake,
            terminal=RunResult(
                status="no_safe_fix",
                reason=intake.no_safe_fix_reason or "no safe fix identified",
            ),
        )

    # SAFETY: never auto-fix a judgment-critical failure (tenant isolation,
    # RAP bypass, destructive ops, credentials exposure). These need human review;
    # attempting them would lean on the re-audit - the same auditor that is
    # least reliable on exactly these judgment calls. Decline up front.
    target = intake.target
    if target is not None:
        jc = target.judgment_critical_blocking_ids
        if jc:
            return Stage1Outcome(
                config=config,
                intake=intake,
                terminal=RunResult(
                    status="no_safe_fix",
                    reason=(
                        "the failing check(s) require human review and are never "
                        f"auto-fixed: {', '.join(jc)}"
                    ),
                ),
            )

        # Nothing BLOCKING to fix: every failing check is advisory (a human-review
        # flag, e.g. destructive-operation-safety, now advisory, or SQL style).
        # The fixer is only responsible for blocking checks, and the allowlist
        # forbids the doc/style edits that would clear an advisory anyway, so a
        # proposal round here always ends in no_safe_fix. Decline up front - this
        # also keeps the fixer from ever attempting to touch an advisory finding
        # (e.g. editing a destructive post-hook), which it must not auto-fix,
        # instead of relying on a downstream allowlist pattern to catch it.
        if not target.blocking_identifiers:
            advisory_ids = ", ".join(target.identifiers) or "none"
            return Stage1Outcome(
                config=config,
                intake=intake,
                terminal=RunResult(
                    status="no_safe_fix",
                    reason=(
                        "only advisory findings are present (no blocking check to "
                        f"fix); these are for human review, not auto-fix: {advisory_ids}"
                    ),
                ),
            )

    return Stage1Outcome(config=config, intake=intake, terminal=None)
