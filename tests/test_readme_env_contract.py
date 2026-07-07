"""Cross-check that `README.md`'s environment-contract tables are complete.

The spec's `readme_env_contract_completeness` criterion is only meaningful if
it's actively enforced: prose can drift out of sync with `src/dbt_fixer/`
the moment a new `DBT_FIXER_*` variable is added (or renamed) without a
matching README edit. This module makes that drift a test failure in both
directions:

1. Every `DBT_FIXER_*` string literal referenced anywhere in the package
   source (found via a whole-tree grep, not a hand-maintained list) must
   appear at least once in `README.md`.
2. Every `DBT_FIXER_*` string literal that appears in `README.md` must
   correspond to a real variable referenced somewhere in the package source
   (catches a typo'd or since-removed variable lingering in the docs).

This is a static/textual check, not an import-time introspection of
`FixerConfig`'s fields, deliberately: the README documents the *environment
variable names* a deployer sets, and grepping for the literal strings is the
most direct proof that nothing was renamed on one side without the other.
"""

from __future__ import annotations

import re
from pathlib import Path

import dbt_fixer

SRC_ROOT = Path(dbt_fixer.__file__).resolve().parent
README_PATH = SRC_ROOT.parent.parent / "README.md"

_ENV_VAR_PATTERN = re.compile(r"DBT_FIXER_[A-Z0-9_]+")


def _env_vars_referenced_in_source() -> set[str]:
    found: set[str] = set()
    for path in SRC_ROOT.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        found |= set(_ENV_VAR_PATTERN.findall(path.read_text(encoding="utf-8")))
    return found


def _env_vars_mentioned_in_readme() -> set[str]:
    text = README_PATH.read_text(encoding="utf-8")
    return set(_ENV_VAR_PATTERN.findall(text))


def test_readme_exists():
    assert README_PATH.exists(), f"expected a README at {README_PATH}"


def test_every_source_referenced_env_var_is_documented_in_readme():
    in_source = _env_vars_referenced_in_source()
    in_readme = _env_vars_mentioned_in_readme()
    missing = in_source - in_readme
    assert not missing, (
        f"these DBT_FIXER_* variables are referenced in source but never "
        f"mentioned in README.md: {sorted(missing)}"
    )


def test_every_readme_mentioned_env_var_actually_exists_in_source():
    in_source = _env_vars_referenced_in_source()
    in_readme = _env_vars_mentioned_in_readme()
    stale = in_readme - in_source
    assert not stale, (
        f"README.md mentions these DBT_FIXER_* variables, but they are not "
        f"referenced anywhere in package source (typo or stale docs?): {sorted(stale)}"
    )


def test_source_referenced_env_vars_sanity_check():
    # Guards against the grep silently matching zero variables if the naming
    # convention or package layout ever changes underneath this test.
    assert len(_env_vars_referenced_in_source()) >= 15


def test_core_run_configuration_table_documents_required_vs_optional():
    text = README_PATH.read_text(encoding="utf-8")
    # The two required variables must be explicitly marked required, not
    # just mentioned in passing -- a deployer scanning the table needs to
    # know which vars are fail-closed on absence.
    assert re.search(r"`DBT_FIXER_FAILURE_KIND`\s*\|\s*\*\*yes\*\*", text)
    assert re.search(r"`DBT_FIXER_REPO_PATH`\s*\|\s*\*\*yes\*\*", text)
