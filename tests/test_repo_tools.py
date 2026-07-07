"""Tests for `dbt_fixer.tools.repo_tools.RepoTools`.

Covers construction validation, byte-for-byte read correctness, glob search
correctness (including `**` recursion), and the path-traversal / symlink
rejection behavior for both `read_file` (raises) and `search_files`
(silently excludes escaping matches, but still raises for a traversal
attempt in the `pattern`/`relative_dir` arguments themselves).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from dbt_fixer.pathsafe import PathTraversalError
from dbt_fixer.tools.repo_tools import (
    RepoFileNotFoundError,
    RepoIsADirectoryError,
    RepoReadOutcome,
    RepoSearchOutcome,
    RepoTools,
)


def _make_repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    (root / "models" / "staging").mkdir(parents=True)
    (root / "models" / "staging" / "stg_customers.sql").write_text(
        "select * from raw.customers", encoding="utf-8"
    )
    (root / "models" / "marts").mkdir(parents=True)
    (root / "models" / "marts" / "customers.sql").write_text(
        "select * from staging.stg_customers", encoding="utf-8"
    )
    (root / "README.md").write_text("# repo", encoding="utf-8")
    return root


def test_construction_rejects_missing_or_non_directory_root(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        RepoTools(tmp_path / "does-not-exist")

    a_file = tmp_path / "a_file.txt"
    a_file.write_text("x", encoding="utf-8")
    with pytest.raises(NotADirectoryError):
        RepoTools(a_file)


def test_read_file_returns_exact_content(tmp_path: Path) -> None:
    root = _make_repo(tmp_path)
    tools = RepoTools(root)

    content = tools.read_file("models/staging/stg_customers.sql")

    assert content == "select * from raw.customers"


def test_read_file_raises_for_missing_file(tmp_path: Path) -> None:
    root = _make_repo(tmp_path)
    tools = RepoTools(root)

    with pytest.raises(RepoFileNotFoundError):
        tools.read_file("models/staging/does_not_exist.sql")


def test_read_file_raises_for_directory(tmp_path: Path) -> None:
    root = _make_repo(tmp_path)
    tools = RepoTools(root)

    with pytest.raises(RepoIsADirectoryError):
        tools.read_file("models/staging")


def test_read_file_rejects_dotdot_traversal(tmp_path: Path) -> None:
    root = _make_repo(tmp_path)
    tools = RepoTools(root)

    with pytest.raises(PathTraversalError):
        tools.read_file("../outside.txt")


def test_read_file_rejects_absolute_path(tmp_path: Path) -> None:
    root = _make_repo(tmp_path)
    tools = RepoTools(root)

    with pytest.raises(PathTraversalError):
        tools.read_file("/etc/passwd")


@pytest.mark.skipif(os.name == "nt", reason="symlinks require elevated privileges on Windows")
def test_read_file_raises_for_symlink_escaping_root(tmp_path: Path) -> None:
    outside = tmp_path / "outside_secret.sql"
    outside.write_text("select secret", encoding="utf-8")
    root = _make_repo(tmp_path)
    (root / "escape.sql").symlink_to(outside)
    tools = RepoTools(root)

    with pytest.raises(PathTraversalError):
        tools.read_file("escape.sql")


def test_search_files_finds_recursive_glob_matches(tmp_path: Path) -> None:
    root = _make_repo(tmp_path)
    tools = RepoTools(root)

    matches = tools.search_files("**/*.sql", relative_dir="models")

    assert matches == (
        "models/marts/customers.sql",
        "models/staging/stg_customers.sql",
    )


def test_search_files_combined_pattern_default_dir(tmp_path: Path) -> None:
    root = _make_repo(tmp_path)
    tools = RepoTools(root)

    matches = tools.search_files("models/**/*.sql")

    assert matches == (
        "models/marts/customers.sql",
        "models/staging/stg_customers.sql",
    )


def test_search_files_does_not_match_unrelated_extensions(tmp_path: Path) -> None:
    root = _make_repo(tmp_path)
    tools = RepoTools(root)

    matches = tools.search_files("**/*.sql", relative_dir="models")

    assert "README.md" not in matches
    assert all(m.endswith(".sql") for m in matches)


def test_search_files_rejects_dotdot_in_pattern(tmp_path: Path) -> None:
    root = _make_repo(tmp_path)
    tools = RepoTools(root)

    with pytest.raises(PathTraversalError):
        tools.search_files("../*.sql")


def test_search_files_rejects_absolute_relative_dir(tmp_path: Path) -> None:
    root = _make_repo(tmp_path)
    tools = RepoTools(root)

    with pytest.raises(PathTraversalError):
        tools.search_files("*.sql", relative_dir="/etc")


def test_search_files_raises_for_missing_relative_dir(tmp_path: Path) -> None:
    root = _make_repo(tmp_path)
    tools = RepoTools(root)

    with pytest.raises(RepoFileNotFoundError):
        tools.search_files("*.sql", relative_dir="does-not-exist")


@pytest.mark.skipif(os.name == "nt", reason="symlinks require elevated privileges on Windows")
def test_search_files_silently_excludes_symlink_escaping_matches(tmp_path: Path) -> None:
    outside = tmp_path / "outside_secret.sql"
    outside.write_text("select secret", encoding="utf-8")
    root = _make_repo(tmp_path)
    (root / "models" / "escape.sql").symlink_to(outside)
    tools = RepoTools(root)

    matches = tools.search_files("**/*.sql", relative_dir="models")

    assert "models/escape.sql" not in matches
    assert matches == (
        "models/marts/customers.sql",
        "models/staging/stg_customers.sql",
    )


def test_search_files_respects_max_results_cap(tmp_path: Path) -> None:
    root = _make_repo(tmp_path)
    tools = RepoTools(root)

    matches = tools.search_files("**/*.sql", relative_dir="models", max_results=1)

    assert len(matches) == 1


# --- try_read_file / try_search_files: the non-raising Outcome-enum API ----


def test_try_read_file_ok(tmp_path: Path) -> None:
    root = _make_repo(tmp_path)
    tools = RepoTools(root)

    result = tools.try_read_file("models/staging/stg_customers.sql")

    assert result.ok is True
    assert result.outcome is RepoReadOutcome.OK
    assert result.content == "select * from raw.customers"
    assert result.reason == ""


def test_try_read_file_never_raises_for_traversal(tmp_path: Path) -> None:
    root = _make_repo(tmp_path)
    tools = RepoTools(root)

    result = tools.try_read_file("../etc/passwd")

    assert result.ok is False
    assert result.outcome is RepoReadOutcome.PATH_REJECTED
    assert result.content is None
    assert result.reason


def test_try_read_file_never_raises_for_absolute_path(tmp_path: Path) -> None:
    root = _make_repo(tmp_path)
    tools = RepoTools(root)

    result = tools.try_read_file("/etc/passwd")

    assert result.ok is False
    assert result.outcome is RepoReadOutcome.PATH_REJECTED


def test_try_read_file_never_raises_for_missing_file(tmp_path: Path) -> None:
    root = _make_repo(tmp_path)
    tools = RepoTools(root)

    result = tools.try_read_file("models/does_not_exist.sql")

    assert result.ok is False
    assert result.outcome is RepoReadOutcome.NOT_FOUND
    assert result.reason


def test_try_read_file_never_raises_for_directory(tmp_path: Path) -> None:
    root = _make_repo(tmp_path)
    tools = RepoTools(root)

    result = tools.try_read_file("models/staging")

    assert result.ok is False
    assert result.outcome is RepoReadOutcome.IS_A_DIRECTORY
    assert result.reason


@pytest.mark.skipif(os.name == "nt", reason="symlinks require elevated privileges on Windows")
def test_try_read_file_never_raises_for_symlink_escape(tmp_path: Path) -> None:
    outside = tmp_path / "outside_secret2.sql"
    outside.write_text("select secret", encoding="utf-8")
    root = _make_repo(tmp_path)
    (root / "models" / "escape2.sql").symlink_to(outside)
    tools = RepoTools(root)

    result = tools.try_read_file("models/escape2.sql")

    assert result.ok is False
    assert result.outcome is RepoReadOutcome.PATH_REJECTED


def test_try_search_files_ok(tmp_path: Path) -> None:
    root = _make_repo(tmp_path)
    tools = RepoTools(root)

    result = tools.try_search_files("**/*.sql", relative_dir="models")

    assert result.ok is True
    assert result.outcome is RepoSearchOutcome.OK
    assert result.matches == (
        "models/marts/customers.sql",
        "models/staging/stg_customers.sql",
    )
    assert result.reason == ""


def test_try_search_files_never_raises_for_traversal_in_relative_dir(tmp_path: Path) -> None:
    root = _make_repo(tmp_path)
    tools = RepoTools(root)

    result = tools.try_search_files("*.sql", relative_dir="../etc")

    assert result.ok is False
    assert result.outcome is RepoSearchOutcome.PATH_REJECTED
    assert result.matches == ()
    assert result.reason


def test_try_search_files_never_raises_for_absolute_relative_dir(tmp_path: Path) -> None:
    root = _make_repo(tmp_path)
    tools = RepoTools(root)

    result = tools.try_search_files("*.sql", relative_dir="/etc")

    assert result.ok is False
    assert result.outcome is RepoSearchOutcome.PATH_REJECTED


def test_try_search_files_never_raises_for_missing_relative_dir(tmp_path: Path) -> None:
    root = _make_repo(tmp_path)
    tools = RepoTools(root)

    result = tools.try_search_files("*.sql", relative_dir="does-not-exist")

    assert result.ok is False
    assert result.outcome is RepoSearchOutcome.NOT_FOUND
    assert result.reason


def test_try_search_files_never_raises_for_non_directory_relative_dir(tmp_path: Path) -> None:
    root = _make_repo(tmp_path)
    tools = RepoTools(root)

    result = tools.try_search_files("*.sql", relative_dir="models/staging/stg_customers.sql")

    assert result.ok is False
    assert result.outcome is RepoSearchOutcome.NOT_A_DIRECTORY


def test_try_search_files_and_search_files_agree_on_matches(tmp_path: Path) -> None:
    root = _make_repo(tmp_path)
    tools = RepoTools(root)

    raising = tools.search_files("**/*.sql", relative_dir="models")
    non_raising = tools.try_search_files("**/*.sql", relative_dir="models")

    assert non_raising.ok is True
    assert non_raising.matches == raising
