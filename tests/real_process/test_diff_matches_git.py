"""Real-process comparison: `dbt_fixer.diffing` output vs. an actual `git diff`.

This module is explicitly marked `real_process` and is therefore exempt
from the offline-only `conftest.py` guard -- it is the one sanctioned place
in this package's test suite that shells out to a real `git` binary, purely
to *prove* (not to implement) that our pure-`difflib`-based diff generation
produces the same hunk content real `git diff` would, for the add-only,
delete-only, and mixed-modification cases.

The `diff --git`/`index` header lines are intentionally not compared:
`dbt_fixer.diffing` emits its own synthetic `diff --git a/{path} b/{path}`
header rather than a real blob-hash `index` line (which would require
actually shelling out to git to compute, defeating the point of a pure,
offline-testable diff generator). Everything from the `---`/`+++` file
markers through the end of each hunk is compared exactly.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from dbt_fixer.diffing import generate_unified_diff

pytestmark = pytest.mark.real_process


def _run_git(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=True,
    )


def _strip_diff_git_and_index_lines(diff_text: str) -> str:
    """Drop `diff --git ...` and `index ...` header lines from a diff.

    These are the only lines `dbt_fixer.diffing` does not attempt to
    replicate byte-for-byte (see module docstring); every other line
    (`---`, `+++`, `@@ ... @@`, and the `+`/`-`/context body lines) must
    match a real `git diff` exactly.
    """

    _dropped_prefixes = ("diff --git ", "index ", "new file mode ", "deleted file mode ")
    kept = [
        line
        for line in diff_text.splitlines(keepends=True)
        if not line.startswith(_dropped_prefixes)
    ]
    return "".join(kept)


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _run_git(["init", "-q"], cwd=repo)
    _run_git(["config", "user.email", "test@example.com"], cwd=repo)
    _run_git(["config", "user.name", "Test"], cwd=repo)
    return repo


def _commit_all(repo: Path, message: str) -> None:
    _run_git(["add", "-A"], cwd=repo)
    _run_git(["commit", "-q", "-m", message], cwd=repo)


def test_modified_file_matches_real_git_diff(tmp_path: Path, git_repo: Path) -> None:
    (git_repo / "models").mkdir()
    (git_repo / "models" / "a.sql").write_text("select 1\nselect 2\n", encoding="utf-8")
    _commit_all(git_repo, "initial")

    before_root = tmp_path / "before"
    before_root.mkdir()
    (before_root / "models").mkdir()
    (before_root / "models" / "a.sql").write_text("select 1\nselect 2\n", encoding="utf-8")

    (git_repo / "models" / "a.sql").write_text("select 1\nselect 999\n", encoding="utf-8")

    real_diff = _run_git(["diff", "--no-color"], cwd=git_repo).stdout
    ours = generate_unified_diff(before_root, git_repo, ["models/a.sql"])

    assert _strip_diff_git_and_index_lines(real_diff) == _strip_diff_git_and_index_lines(ours)


def test_added_file_matches_real_git_diff(tmp_path: Path, git_repo: Path) -> None:
    (git_repo / "models").mkdir()
    (git_repo / "models" / "existing.sql").write_text("select 1\n", encoding="utf-8")
    _commit_all(git_repo, "initial")

    before_root = tmp_path / "before"
    before_root.mkdir()
    (before_root / "models").mkdir()
    (before_root / "models" / "existing.sql").write_text("select 1\n", encoding="utf-8")

    (git_repo / "models" / "new.sql").write_text("select 42\n", encoding="utf-8")
    # An untracked new file is invisible to plain `git diff`; `add -N`
    # ("intent to add") registers it in the index with empty content so
    # `git diff` reports it as an addition against `/dev/null`, without
    # actually staging its content (unlike a real `git add`).
    _run_git(["add", "-N", "models/new.sql"], cwd=git_repo)

    real_diff = _run_git(["diff", "--no-color", "--", "models/new.sql"], cwd=git_repo).stdout
    ours = generate_unified_diff(before_root, git_repo, ["models/new.sql"])

    assert _strip_diff_git_and_index_lines(real_diff) == _strip_diff_git_and_index_lines(ours)


def test_deleted_file_matches_real_git_diff(tmp_path: Path, git_repo: Path) -> None:
    (git_repo / "models").mkdir()
    (git_repo / "models" / "gone.sql").write_text("select 1\n", encoding="utf-8")
    _commit_all(git_repo, "initial")

    before_root = tmp_path / "before"
    before_root.mkdir()
    (before_root / "models").mkdir()
    (before_root / "models" / "gone.sql").write_text("select 1\n", encoding="utf-8")

    (git_repo / "models" / "gone.sql").unlink()

    real_diff = _run_git(["diff", "--no-color"], cwd=git_repo).stdout
    ours = generate_unified_diff(before_root, git_repo, ["models/gone.sql"])

    assert _strip_diff_git_and_index_lines(real_diff) == _strip_diff_git_and_index_lines(ours)


def test_mixed_add_modify_delete_matches_real_git_diff(tmp_path: Path, git_repo: Path) -> None:
    (git_repo / "models").mkdir()
    (git_repo / "models" / "modify_me.sql").write_text("select 1\n", encoding="utf-8")
    (git_repo / "models" / "delete_me.sql").write_text("select 2\n", encoding="utf-8")
    (git_repo / "models" / "keep_me.sql").write_text("select 3\n", encoding="utf-8")
    _commit_all(git_repo, "initial")

    before_root = tmp_path / "before"
    before_root.mkdir()
    (before_root / "models").mkdir()
    (before_root / "models" / "modify_me.sql").write_text("select 1\n", encoding="utf-8")
    (before_root / "models" / "delete_me.sql").write_text("select 2\n", encoding="utf-8")
    (before_root / "models" / "keep_me.sql").write_text("select 3\n", encoding="utf-8")

    (git_repo / "models" / "modify_me.sql").write_text("select 111\n", encoding="utf-8")
    (git_repo / "models" / "delete_me.sql").unlink()
    (git_repo / "models" / "add_me.sql").write_text("select 4\n", encoding="utf-8")
    _run_git(["add", "-N", "models/add_me.sql"], cwd=git_repo)

    real_diff = _run_git(["diff", "--no-color"], cwd=git_repo).stdout
    ours = generate_unified_diff(
        before_root,
        git_repo,
        ["models/modify_me.sql", "models/delete_me.sql", "models/add_me.sql", "models/keep_me.sql"],
    )

    assert _strip_diff_git_and_index_lines(real_diff) == _strip_diff_git_and_index_lines(ours)
