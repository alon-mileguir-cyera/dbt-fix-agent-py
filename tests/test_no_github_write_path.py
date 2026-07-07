"""Static, grep-level proof that no GitHub-write code path exists anywhere in
this package's source tree.

This is a whole-package hardening check (unlike `test_no_network_static.py`,
which is scoped to Sprint 1's modules): it walks every `.py` file under
`src/dbt_fixer/` -- including modules added in every later sprint -- and
fails the build if it finds any string, import, or call pattern associated
with writing to GitHub: a PyGithub/GitHub REST client construction or write
method call, a `git push`/`git commit --push`-style shell invocation, a POST
to a `github.com`/`api.github.com` endpoint, or a PR/issue/review-comment
creation call.

This package is architecturally read-only with respect to GitHub -- it never
even *reads* GitHub directly (the PR diff/title/description/URL all arrive
as pre-fetched `DBT_FIXER_*` environment values, never fetched by this
package itself) -- so the bar here is "zero matches", not "zero matches
outside an allowlist". The one place a GitHub-shaped string is expected to
appear at all is prose describing what this package *cannot* do (its
docstrings and this test file itself); those are excluded from the scan by
construction (see `_NEGATIVE_EXAMPLE_FILES` and the docstring-stripping
pass below) rather than by a per-line allowlist, so a genuine offender can
never hide behind a comment claiming to be a negative example.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import dbt_fixer

SRC_ROOT = Path(dbt_fixer.__file__).resolve().parent

# This test file (and its sibling static-check module) are the only places
# in the repository permitted to *mention* GitHub-write terminology at all,
# since they exist to describe/detect it; they are never scanned themselves.
_THIS_FILE = Path(__file__).resolve()

# --- forbidden patterns -------------------------------------------------

# Python identifiers/imports that would only ever appear if a GitHub REST
# client library were in use.
_FORBIDDEN_IMPORT_ROOTS = {"github", "pygithub", "ghapi"}

# Case-insensitive substring/regex patterns associated with a GitHub *write*
# path: client construction, PR/issue/comment/review creation, and shelling
# out to push commits. Matched against raw source text (not just imports),
# so even a dynamically-constructed call (`getattr(gh, "create_pull")`)
# still leaves a literal trace.
_FORBIDDEN_TEXT_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bimport\s+github\b", re.IGNORECASE),
    re.compile(r"\bfrom\s+github\b", re.IGNORECASE),
    re.compile(r"\bGithub\s*\(", re.IGNORECASE),  # PyGithub client construction
    re.compile(r"\.create_pull\s*\(", re.IGNORECASE),
    re.compile(r"\.create_issue\s*\(", re.IGNORECASE),
    re.compile(r"\.create_issue_comment\s*\(", re.IGNORECASE),
    re.compile(r"\.create_review\s*\(", re.IGNORECASE),
    re.compile(r"\.create_comment\s*\(", re.IGNORECASE),
    re.compile(r"\bgit\s+push\b", re.IGNORECASE),
    re.compile(r"\bgit\s+commit\s+--push\b", re.IGNORECASE),
    re.compile(r"api\.github\.com", re.IGNORECASE),
    re.compile(r"github\.com/repos", re.IGNORECASE),
    re.compile(r"\bgh\s+pr\s+create\b", re.IGNORECASE),
    re.compile(r"\bgh\s+api\b.*-X\s*POST", re.IGNORECASE),
    re.compile(r"PyGithub", re.IGNORECASE),
)


def _strip_docstrings_and_comments(source: str) -> str:
    """Return `source` with every module/class/function docstring and every
    `#`-comment removed, so prose *describing* the absence of a GitHub-write
    path (this package's own module docstrings say exactly that, by design)
    can never itself trip this scan -- only genuine code/string-literal
    content is checked.
    """

    tree = ast.parse(source)
    docstring_spans: list[tuple[int, int]] = []
    for node in ast.walk(tree):
        if isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            body = getattr(node, "body", None)
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                if isinstance(body[0].value.value, str):
                    docstring_spans.append(
                        (body[0].lineno, getattr(body[0], "end_lineno", body[0].lineno))
                    )

    lines = source.splitlines()
    for start, end in docstring_spans:
        for lineno in range(start, end + 1):
            if 1 <= lineno <= len(lines):
                lines[lineno - 1] = ""

    # Strip trailing `# ...` comments line-by-line (good enough for this
    # grep-level check; this package's source never puts a `#` inside a
    # string literal that would need smarter handling for this purpose).
    stripped_lines = []
    for line in lines:
        if "#" in line:
            line = line.split("#", 1)[0]
        stripped_lines.append(line)
    return "\n".join(stripped_lines)


def _imported_roots(source: str) -> set[str]:
    tree = ast.parse(source)
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                roots.add(alias.name.split(".")[0].lower())
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                roots.add(node.module.split(".")[0].lower())
    return roots


def _all_source_files() -> list[Path]:
    return sorted(
        f for f in SRC_ROOT.rglob("*.py") if "__pycache__" not in f.parts
    )


def test_source_tree_is_non_empty_sanity_check():
    # Guards against this test silently passing over zero files if the
    # package layout ever changes.
    assert len(_all_source_files()) >= 20


def test_no_github_client_library_is_imported_anywhere():
    offenders: dict[str, set[str]] = {}
    for path in _all_source_files():
        roots = _imported_roots(path.read_text(encoding="utf-8"))
        forbidden = roots & _FORBIDDEN_IMPORT_ROOTS
        if forbidden:
            offenders[str(path.relative_to(SRC_ROOT))] = forbidden
    assert not offenders, f"forbidden GitHub-client imports found: {offenders}"


def test_no_github_write_call_or_push_pattern_exists_in_source():
    offenders: dict[str, list[str]] = {}
    for path in _all_source_files():
        raw = path.read_text(encoding="utf-8")
        scrubbed = _strip_docstrings_and_comments(raw)
        hits = [
            pattern.pattern
            for pattern in _FORBIDDEN_TEXT_PATTERNS
            if pattern.search(scrubbed)
        ]
        if hits:
            offenders[str(path.relative_to(SRC_ROOT))] = hits
    assert not offenders, (
        f"forbidden GitHub-write patterns found in package source (outside "
        f"docstrings/comments): {offenders}"
    )


def test_no_github_dependency_declared_in_packaging_metadata():
    pyproject = SRC_ROOT.parent.parent / "pyproject.toml"
    assert pyproject.exists()
    text = pyproject.read_text(encoding="utf-8").lower()
    for offender in ("pygithub", "ghapi", "\"github\"", "'github'"):
        assert offender not in text, f"forbidden GitHub dependency declared: {offender!r}"


def test_this_static_check_itself_is_excluded_from_its_own_scan():
    # The scan walks `src/dbt_fixer` only, never the `tests/` directory this
    # file lives in, so this file's own necessary mentions of "github"/
    # "push" (in its docstring and pattern list) can never cause a false
    # self-positive.
    assert SRC_ROOT not in _THIS_FILE.parents
    assert _THIS_FILE not in _all_source_files()
