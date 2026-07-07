"""Shared path-containment guard: reject any path that resolves outside a root.

Both the model-facing `RepoTools` read/search toolkit (`dbt_fixer.tools.repo_tools`)
and the structured-edit applier (`dbt_fixer.applier`) must apply exactly the same
containment rule to every path they are handed, so this logic is defined once,
here, rather than duplicated (and risking drift) across the two call sites.

A relative path is rejected -- before any filesystem access is attempted -- if
it is:

- not a non-empty string,
- absolute,
- contains a literal `..` path component, or
- resolves (after following any symlinks in the joined path) to a location
  that is not the root itself or a descendant of the root.

The last check is what catches a symlink planted *inside* the root that
points *outside* it: `Path.resolve()` follows symlinks, so the final
containment comparison is always against the fully-resolved, real path, not
the literal joined string.

**Outcome-first design.** `check_within_root` is the primary implementation:
it never raises, and instead returns a `PathCheckResult` carrying a closed
`PathCheckOutcome` enum value naming exactly which rule (if any) rejected the
path. `resolve_within_root` is a thin, raising convenience wrapper over it,
kept for call sites (and existing tests) that prefer exception-based control
flow; it does not duplicate `check_within_root`'s rules, so the two can never
drift apart.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional


class PathTraversalError(ValueError):
    """Raised when a relative path would resolve outside its sanctioned root.

    Never carries any content from the rejected path's target -- only the
    (attacker-supplied) path string itself, which is safe to surface in an
    error message or log line.
    """


class PathCheckOutcome(Enum):
    """The closed set of results a containment check can produce."""

    OK = "ok"
    EMPTY_OR_INVALID = "empty_or_invalid"
    ABSOLUTE_PATH_REJECTED = "absolute_path_rejected"
    TRAVERSAL_REJECTED = "traversal_rejected"
    OUTSIDE_ROOT_REJECTED = "outside_root_rejected"

    @property
    def ok(self) -> bool:
        return self is PathCheckOutcome.OK


@dataclass(frozen=True)
class PathCheckResult:
    """The non-exceptional result of one `check_within_root` call.

    `resolved_path` is set if and only if `outcome.ok` is true. `reason` is
    always a specific, human-readable explanation -- present on every
    non-`OK` outcome, never a generic "invalid path" message.
    """

    outcome: PathCheckOutcome
    resolved_path: Optional[Path] = None
    reason: str = ""

    @property
    def ok(self) -> bool:
        return self.outcome.ok


def check_within_root(root: Path, relative_path: str) -> PathCheckResult:
    """Check whether `relative_path` resolves within `root`. Never raises.

    This is the canonical containment rule both `resolve_within_root` and
    every path-safe tool ultimately defer to. See the module docstring for
    the full rule set.
    """

    if not isinstance(relative_path, str) or relative_path.strip() == "":
        return PathCheckResult(
            outcome=PathCheckOutcome.EMPTY_OR_INVALID,
            reason=f"path must be a non-empty string, got {relative_path!r}",
        )

    candidate = Path(relative_path)

    # Reject absolute paths outright: joining an absolute path onto the root
    # with `/` would silently discard the root entirely in pathlib, so this
    # must be checked before any join happens.
    if candidate.is_absolute():
        return PathCheckResult(
            outcome=PathCheckOutcome.ABSOLUTE_PATH_REJECTED,
            reason=f"absolute paths are not allowed: {relative_path!r}",
        )

    # Reject any literal parent-directory traversal component up front, as a
    # clear, explicit signal independent of what resolve() does.
    if ".." in candidate.parts:
        return PathCheckResult(
            outcome=PathCheckOutcome.TRAVERSAL_REJECTED,
            reason=f"path traversal ('..') is not allowed: {relative_path!r}",
        )

    resolved_root = Path(root).resolve()
    joined = resolved_root / candidate
    # resolve() follows symlinks (and does not require the target to exist),
    # so a symlink inside the root that points outside it is caught by the
    # containment check below.
    resolved = joined.resolve()

    if resolved != resolved_root and resolved_root not in resolved.parents:
        return PathCheckResult(
            outcome=PathCheckOutcome.OUTSIDE_ROOT_REJECTED,
            reason=f"path resolves outside the root: {relative_path!r}",
        )

    return PathCheckResult(outcome=PathCheckOutcome.OK, resolved_path=resolved)


def resolve_within_root(root: Path, relative_path: str) -> Path:
    """Resolve `relative_path` against `root`, guaranteeing containment.

    A raising convenience wrapper around `check_within_root` for call sites
    that prefer exception-based control flow (e.g. a caller that has no
    useful recovery path other than aborting).

    Args:
        root: The sanctioned root directory. Does not need to be
            pre-resolved; this function resolves it itself.
        relative_path: A path string that must be relative (not absolute)
            and must not contain a `..` component.

    Returns:
        The fully resolved (symlinks followed) absolute path, guaranteed to
        be `root` itself or a descendant of it.

    Raises:
        PathTraversalError: If `relative_path` is not a non-empty string,
            is absolute, contains a `..` component, or resolves (following
            symlinks) to a location outside `root`.
    """

    result = check_within_root(root, relative_path)
    if not result.ok:
        raise PathTraversalError(result.reason)
    assert result.resolved_path is not None  # guaranteed by check_within_root when ok
    return result.resolved_path
