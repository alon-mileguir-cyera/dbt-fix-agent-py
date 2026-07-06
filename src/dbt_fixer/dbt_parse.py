"""The dbt Parse Gate: a best-effort, non-authoritative sanity check.

Unlike every other gate in this package, this one is allowed to not run at
all -- and when it doesn't, that is recorded honestly as `"skipped"`,
never conflated with `"passed"` or `"failed"`. It exists purely as a fast,
opportunistic extra signal: if a real `dbt` executable happens to be on
`PATH`, run `dbt parse` against the touched project directory (inside a
scratch copy with the candidate diff applied) under a bounded timeout, and
kill the candidate on a nonzero exit or a timeout. If `dbt` is not
available, or the scratch copy/candidate-apply step itself cannot be
completed, this gate skips rather than guessing -- the allowlist and
re-audit gates remain the only two gates whose pass is ever required for a
`proposed` outcome; a skip here never blocks and never itself grants that
outcome (see `dbt_fixer.retry_loop`, which never treats `skipped` as
equivalent to `passed`).

**Locating the touched project directory.** A candidate diff's touched
files are read via `dbt_fixer.diffparse._diff_paths` (the same read-only
diff-path extraction the allowlist gate already relies on). For each
touched path, in order, this module walks upward from that file's
directory -- never above the scratch root -- looking for the nearest
`dbt_project.yml`. The first touched path that resolves to a project
directory wins; if none of them do, there is no sensible place to run
`dbt parse` at all, and the gate skips rather than running it somewhere
ambiguous (e.g. the scratch root itself) where a failure would carry no
real meaning.

**Injectable everything.** `which` (defaults to `shutil.which`) and
`subprocess_runner` are both injectable so this module is exercised with
zero real subprocess or filesystem-PATH access in tests, matching
`conftest.py`'s enforced offline contract. A real `subprocess_runner`
implementation is expected to translate an actual timeout into
`DbtParseTimeoutError` and a genuine invocation failure (e.g. the
executable vanished between the `which` check and the call) into
`DbtInvocationError`; this module never spawns a subprocess directly.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Literal, Optional, Tuple

from .diffparse import DiffParseError, PatchApplyError, _diff_paths, apply_diff
from .pathsafe import PathTraversalError, resolve_within_root
from .reaudit import ProcessOutcome
from .scratch import ScratchCopyError, scratch_copy

__all__ = [
    "DbtInvocationError",
    "DbtParseTimeoutError",
    "DbtSubprocessRunner",
    "DbtParseOutcome",
    "DbtParseVerdict",
    "find_touched_project_dir",
    "run_dbt_parse_gate",
]

DBT_PROJECT_FILENAME = "dbt_project.yml"

DbtParseOutcome = Literal["passed", "failed", "skipped"]

WhichFunc = Callable[[str], Optional[str]]


class DbtInvocationError(RuntimeError):
    """Raised by a `DbtSubprocessRunner` when `dbt` cannot be started at all.

    Distinct from a completed process that exits nonzero (an ordinary
    parse failure) or a `DbtParseTimeoutError` (a bounded-timeout breach):
    this is for the case the process never even started. Because this
    gate is best-effort, this always resolves to `"skipped"`, never
    `"failed"`.
    """


class DbtParseTimeoutError(RuntimeError):
    """Raised by a `DbtSubprocessRunner` when `dbt parse` exceeds its bound.

    Unlike `DbtInvocationError`, a timeout is treated as an ordinary gate
    failure (`"failed"`, killing the candidate) -- the executable ran, and
    simply did not finish in time, which is itself meaningful signal about
    the candidate, not an environment problem.
    """


# `(argv, cwd, timeout_seconds) -> ProcessOutcome`, sharing the same
# `ProcessOutcome` shape the re-audit gate's `SubprocessRunner` returns
# (`dbt_fixer.reaudit.ProcessOutcome`) since both are just "run this
# command, bounded, and tell me what happened."
DbtSubprocessRunner = Callable[[list, Path, float], ProcessOutcome]


@dataclass(frozen=True)
class DbtParseVerdict:
    """The dbt Parse Gate's outcome for one candidate diff.

    `outcome` is one of the three distinct, never-conflated states this
    gate can resolve to. `passed` and `skipped` are convenience properties
    so call sites never need to compare `outcome` against a string
    literal directly.
    """

    outcome: DbtParseOutcome
    reason: str = ""
    project_dir: Optional[str] = None

    @property
    def passed(self) -> bool:
        return self.outcome == "passed"

    @property
    def skipped(self) -> bool:
        return self.outcome == "skipped"


def _display_relative(path: Path, root: "str | Path") -> str:
    """Best-effort repo-relative rendering of `path` for human-readable reasons.

    Falls back to the plain string form of `path` if it cannot be expressed
    relative to `root` (e.g. because of a symlink-resolution mismatch
    between the two) -- this is purely cosmetic and never affects gate
    logic, so it must never raise.
    """

    try:
        relative = path.resolve().relative_to(Path(root).resolve())
    except ValueError:
        return str(path)
    return str(relative) if str(relative) != "." else "."


def find_touched_project_dir(scratch_root: "str | Path", touched_paths: Iterable[str]) -> Optional[Path]:
    """Find the nearest `dbt_project.yml`-containing ancestor of a touched path.

    Walks upward from each touched path's directory, in the order
    `touched_paths` is given, never above `scratch_root`. Returns the
    first ancestor directory found containing `dbt_project.yml`, or
    `None` if no touched path resolves to one (an ambiguous case where
    running `dbt parse` anywhere would carry no real meaning).

    A touched path that fails to resolve safely within `scratch_root`
    (path traversal) is silently skipped in favor of the next candidate,
    rather than raising -- this function is purely a best-effort locator,
    never a validator.
    """

    resolved_root = Path(scratch_root).resolve()
    for touched_path in touched_paths:
        try:
            resolved = resolve_within_root(resolved_root, touched_path)
        except PathTraversalError:
            continue

        current = resolved.parent
        while True:
            if (current / DBT_PROJECT_FILENAME).is_file():
                return current
            if current == resolved_root:
                break
            parent = current.parent
            if parent == current:
                break
            current = parent

    return None


def run_dbt_parse_gate(
    *,
    repo_root: "str | Path",
    candidate_diff: str,
    timeout_seconds: float,
    subprocess_runner: DbtSubprocessRunner,
    which: WhichFunc = shutil.which,
) -> DbtParseVerdict:
    """Run the best-effort dbt Parse Gate for one candidate diff.

    Args:
        repo_root: The original, never-mutated checkout root. A fresh
            scratch copy is made and torn down entirely inside this call.
        candidate_diff: The unified diff text for this round's candidate.
        timeout_seconds: The bound handed to `subprocess_runner`.
        subprocess_runner: The injectable `dbt parse` invocation callable
            (a real one in production, a fake in every test). Raising
            `DbtInvocationError` resolves to `"skipped"`; raising
            `DbtParseTimeoutError` resolves to `"failed"`.
        which: Injectable PATH-lookup callable, defaulting to
            `shutil.which`. Returning `None` (no `dbt` on PATH) resolves
            to `"skipped"` before any scratch copy is even created.

    Returns:
        A `DbtParseVerdict`. Never raises: every failure mode (no `dbt` on
        PATH, scratch-copy failure, candidate-apply failure, no locatable
        project directory, invocation failure, timeout, nonzero exit)
        resolves to a `DbtParseVerdict` field, never an exception escaping
        this function.
    """

    dbt_path = which("dbt")
    if not dbt_path:
        return DbtParseVerdict(
            outcome="skipped",
            reason="no dbt executable found on PATH; dbt parse gate skipped",
        )

    try:
        touched_paths: Tuple[str, ...] = tuple(_diff_paths(candidate_diff))
    except DiffParseError as exc:
        return DbtParseVerdict(
            outcome="skipped",
            reason=f"candidate diff could not be parsed for the dbt parse gate: {exc}",
        )

    if not touched_paths:
        return DbtParseVerdict(
            outcome="skipped",
            reason="candidate diff touches no files; nothing to parse",
        )

    repo_root = Path(repo_root)

    try:
        with scratch_copy(repo_root) as scratch_root:
            try:
                apply_diff(scratch_root, candidate_diff)
            except (DiffParseError, PatchApplyError, PathTraversalError) as exc:
                return DbtParseVerdict(
                    outcome="skipped",
                    reason=(
                        "candidate diff could not be applied to a scratch copy for "
                        f"the dbt parse gate: {exc}"
                    ),
                )

            project_dir = find_touched_project_dir(scratch_root, touched_paths)
            if project_dir is None:
                return DbtParseVerdict(
                    outcome="skipped",
                    reason=(
                        "no dbt_project.yml found above any touched path; dbt parse "
                        "gate skipped"
                    ),
                )

            project_dir_display = _display_relative(project_dir, scratch_root)

            try:
                outcome = subprocess_runner([dbt_path, "parse"], project_dir, timeout_seconds)
            except DbtParseTimeoutError as exc:
                return DbtParseVerdict(
                    outcome="failed",
                    reason=f"dbt parse timed out after {timeout_seconds}s: {exc}",
                    project_dir=project_dir_display,
                )
            except DbtInvocationError as exc:
                return DbtParseVerdict(
                    outcome="skipped",
                    reason=f"dbt could not be invoked: {exc}",
                )
    except ScratchCopyError as exc:
        return DbtParseVerdict(
            outcome="skipped",
            reason=f"could not create a scratch copy for the dbt parse gate: {exc}",
        )

    if outcome.returncode != 0:
        return DbtParseVerdict(
            outcome="failed",
            reason=(
                f"dbt parse exited with code {outcome.returncode} in "
                f"{project_dir_display!r}; stderr: {outcome.stderr.strip()!r}"
            ),
            project_dir=project_dir_display,
        )

    return DbtParseVerdict(
        outcome="passed",
        reason=f"dbt parse succeeded in {project_dir_display!r}",
        project_dir=project_dir_display,
    )
