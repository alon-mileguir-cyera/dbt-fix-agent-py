"""Tests for `dbt_fixer.pathsafe.resolve_within_root`.

Covers the success path (a plain in-bounds relative path resolves cleanly)
and the three distinct rejection paths: `..` traversal, absolute paths, and
a symlink planted inside the root that points outside it.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from dbt_fixer.pathsafe import (
    PathCheckOutcome,
    PathTraversalError,
    check_within_root,
    resolve_within_root,
)


def test_resolves_in_bounds_relative_path(tmp_path: Path) -> None:
    (tmp_path / "models").mkdir()
    target = tmp_path / "models" / "stg_customers.sql"
    target.write_text("select 1", encoding="utf-8")

    resolved = resolve_within_root(tmp_path, "models/stg_customers.sql")

    assert resolved == target.resolve()


def test_rejects_dotdot_traversal(tmp_path: Path) -> None:
    with pytest.raises(PathTraversalError):
        resolve_within_root(tmp_path, "../etc/passwd")


def test_rejects_dotdot_traversal_embedded_in_middle(tmp_path: Path) -> None:
    (tmp_path / "models").mkdir()
    with pytest.raises(PathTraversalError):
        resolve_within_root(tmp_path, "models/../../etc/passwd")


def test_rejects_absolute_path(tmp_path: Path) -> None:
    with pytest.raises(PathTraversalError):
        resolve_within_root(tmp_path, "/etc/passwd")


def test_rejects_empty_or_non_string_path(tmp_path: Path) -> None:
    with pytest.raises(PathTraversalError):
        resolve_within_root(tmp_path, "")

    with pytest.raises(PathTraversalError):
        resolve_within_root(tmp_path, "   ")

    with pytest.raises(PathTraversalError):
        resolve_within_root(tmp_path, None)  # type: ignore[arg-type]


@pytest.mark.skipif(os.name == "nt", reason="symlinks require elevated privileges on Windows")
def test_rejects_symlink_escaping_root(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside_secret.txt"
    outside.write_text("secret", encoding="utf-8")

    root = tmp_path / "repo"
    root.mkdir()
    escape_link = root / "escape.sql"
    escape_link.symlink_to(outside)

    with pytest.raises(PathTraversalError):
        resolve_within_root(root, "escape.sql")


# --- check_within_root: the non-raising, Outcome-enum counterpart ----------


def test_check_within_root_ok_for_in_bounds_path(tmp_path: Path) -> None:
    (tmp_path / "models").mkdir()
    target = tmp_path / "models" / "stg_customers.sql"
    target.write_text("select 1", encoding="utf-8")

    result = check_within_root(tmp_path, "models/stg_customers.sql")

    assert result.ok is True
    assert result.outcome is PathCheckOutcome.OK
    assert result.resolved_path == target.resolve()
    assert result.reason == ""


def test_check_within_root_never_raises_for_traversal(tmp_path: Path) -> None:
    result = check_within_root(tmp_path, "../etc/passwd")
    assert result.ok is False
    assert result.outcome is PathCheckOutcome.TRAVERSAL_REJECTED
    assert result.resolved_path is None
    assert result.reason


def test_check_within_root_never_raises_for_absolute_path(tmp_path: Path) -> None:
    result = check_within_root(tmp_path, "/etc/passwd")
    assert result.ok is False
    assert result.outcome is PathCheckOutcome.ABSOLUTE_PATH_REJECTED
    assert result.reason


def test_check_within_root_never_raises_for_empty_or_none(tmp_path: Path) -> None:
    for bad in ("", "   ", None):
        result = check_within_root(tmp_path, bad)  # type: ignore[arg-type]
        assert result.ok is False
        assert result.outcome is PathCheckOutcome.EMPTY_OR_INVALID
        assert result.reason


@pytest.mark.skipif(os.name == "nt", reason="symlinks require elevated privileges on Windows")
def test_check_within_root_never_raises_for_symlink_escape(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside_secret_check.txt"
    outside.write_text("secret", encoding="utf-8")

    root = tmp_path / "repo2"
    root.mkdir()
    escape_link = root / "escape.sql"
    escape_link.symlink_to(outside)

    result = check_within_root(root, "escape.sql")
    assert result.ok is False
    assert result.outcome is PathCheckOutcome.OUTSIDE_ROOT_REJECTED
    assert result.reason


def test_resolve_within_root_and_check_within_root_agree_on_every_case(tmp_path: Path) -> None:
    """The raising wrapper and the Outcome-returning primary implementation
    must never disagree: `resolve_within_root` is defined purely in terms
    of `check_within_root`'s own verdict."""

    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "ok.sql").write_text("select 1", encoding="utf-8")

    cases = ["models/ok.sql", "../escape", "/absolute", "", "models/../../escape"]
    for case in cases:
        result = check_within_root(tmp_path, case)
        if result.ok:
            assert resolve_within_root(tmp_path, case) == result.resolved_path
        else:
            with pytest.raises(PathTraversalError):
                resolve_within_root(tmp_path, case)
