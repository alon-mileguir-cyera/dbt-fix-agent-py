"""Tests for `dbt_fixer.entrypoint`: the always-exit-0, single-status-line
CLI contract.

These tests call `main()`/`compute_run_result()` in-process (never spawning
a real subprocess -- that would be blocked by `conftest` anyway outside a
`real_process` module) and assert on captured stdout, matching exactly how
a future orchestrator would grep this process's output.
"""

from __future__ import annotations

import re

import pytest

import dbt_fixer.pipeline as pipeline_module
from dbt_fixer.entrypoint import compute_run_result, main, render_stdout_lines
from dbt_fixer.env import ENV_FAILURE_KIND, ENV_FAILURE_CONTEXT, ENV_REPO_PATH
from dbt_fixer.status import RunResult, STDOUT_REASON_PREFIX, STDOUT_STATUS_PREFIX

_STATUS_LINE_RE = re.compile(r"^dbt-fixer-status: (proposed|no_safe_fix|failed)$")


def _lines(capsys) -> list[str]:
    out = capsys.readouterr().out
    return out.splitlines()


# --- the fixed stdout contract ----------------------------------------------


def test_empty_environment_resolves_to_failed_with_single_status_line(capsys):
    exit_code = main(env={})
    assert exit_code == 0

    lines = _lines(capsys)
    assert lines, "expected at least the status line"
    assert _STATUS_LINE_RE.match(lines[-1])
    assert lines[-1] == f"{STDOUT_STATUS_PREFIX}: failed"
    # exactly one status line in the whole run
    assert sum(1 for line in lines if _STATUS_LINE_RE.match(line)) == 1


def test_missing_repo_path_resolves_to_failed_with_named_reason(capsys):
    exit_code = main(env={ENV_FAILURE_KIND: "ci"})
    assert exit_code == 0

    lines = _lines(capsys)
    assert lines[-1] == f"{STDOUT_STATUS_PREFIX}: failed"
    reason_lines = [line for line in lines if line.startswith(STDOUT_REASON_PREFIX)]
    assert reason_lines, "a failed run must state a specific reason"
    assert ENV_REPO_PATH in reason_lines[0]


def test_malformed_failure_kind_resolves_to_failed(tmp_path, capsys):
    env = {ENV_FAILURE_KIND: "not-a-real-kind", ENV_REPO_PATH: str(tmp_path)}
    exit_code = main(env=env)
    assert exit_code == 0

    lines = _lines(capsys)
    assert lines[-1] == f"{STDOUT_STATUS_PREFIX}: failed"
    reason_lines = [line for line in lines if line.startswith(STDOUT_REASON_PREFIX)]
    assert ENV_FAILURE_KIND in reason_lines[0]


def test_empty_failure_context_resolves_to_no_safe_fix(tmp_path, capsys):
    env = {ENV_FAILURE_KIND: "ci", ENV_REPO_PATH: str(tmp_path)}
    exit_code = main(env=env)
    assert exit_code == 0

    lines = _lines(capsys)
    assert lines[-1] == f"{STDOUT_STATUS_PREFIX}: no_safe_fix"


def test_unparseable_failure_context_resolves_to_no_safe_fix(tmp_path, capsys):
    env = {
        ENV_FAILURE_KIND: "ci",
        ENV_REPO_PATH: str(tmp_path),
        ENV_FAILURE_CONTEXT: "totally unrelated garbage text",
    }
    exit_code = main(env=env)
    assert exit_code == 0

    lines = _lines(capsys)
    assert lines[-1] == f"{STDOUT_STATUS_PREFIX}: no_safe_fix"


def test_valid_ci_target_still_resolves_to_no_safe_fix_this_sprint(tmp_path, capsys):
    """Sprint 1 scope: even a cleanly-parsed target cannot reach `proposed`
    because the proposal/gate pipeline is not yet wired into this
    entrypoint. `no_safe_fix` -- never a guess, never `proposed` -- is the
    only honest outcome available today."""

    env = {
        ENV_FAILURE_KIND: "ci",
        ENV_REPO_PATH: str(tmp_path),
        ENV_FAILURE_CONTEXT: (
            "Completed with 1 error\n"
            "Failure in test my_test (models/x.sql)\n"
            "Got 1 results, configured to fail if != 0\n"
        ),
    }
    exit_code = main(env=env)
    assert exit_code == 0

    lines = _lines(capsys)
    assert lines[-1] == f"{STDOUT_STATUS_PREFIX}: no_safe_fix"
    reason_lines = [line for line in lines if line.startswith(STDOUT_REASON_PREFIX)]
    assert "my_test" in reason_lines[0]


def test_status_line_is_always_the_last_line_and_appears_exactly_once(tmp_path, capsys):
    scenarios = [
        {},
        {ENV_FAILURE_KIND: "ci"},
        {ENV_FAILURE_KIND: "ci", ENV_REPO_PATH: str(tmp_path)},
        {
            ENV_FAILURE_KIND: "ci",
            ENV_REPO_PATH: str(tmp_path),
            ENV_FAILURE_CONTEXT: "garbage",
        },
    ]
    for env in scenarios:
        exit_code = main(env=env)
        assert exit_code == 0
        lines = _lines(capsys)
        assert lines, f"expected output for env={env!r}"
        assert _STATUS_LINE_RE.match(lines[-1]), f"bad last line for env={env!r}: {lines[-1]!r}"
        assert sum(1 for line in lines if _STATUS_LINE_RE.match(line)) == 1


# --- exit code is always 0, even on internal exceptions ---------------------


def test_exit_code_is_zero_even_when_run_stage1_raises(monkeypatch, capsys):
    def _boom(env=None):
        raise RuntimeError("simulated internal defect")

    monkeypatch.setattr(pipeline_module, "run_stage1", _boom)
    # entrypoint imports run_stage1 by name, so patch it there too.
    import dbt_fixer.entrypoint as entrypoint_module

    monkeypatch.setattr(entrypoint_module, "run_stage1", _boom)

    exit_code = main(env={})
    assert exit_code == 0

    lines = _lines(capsys)
    assert lines[-1] == f"{STDOUT_STATUS_PREFIX}: failed"
    reason_lines = [line for line in lines if line.startswith(STDOUT_REASON_PREFIX)]
    assert reason_lines and "simulated internal defect" in reason_lines[0]


def test_exit_code_is_zero_even_when_compute_run_result_itself_raises(monkeypatch, capsys):
    import dbt_fixer.entrypoint as entrypoint_module

    def _boom(env=None):
        raise RuntimeError("even more unexpected")

    monkeypatch.setattr(entrypoint_module, "compute_run_result", _boom)

    exit_code = main(env={})
    assert exit_code == 0

    lines = _lines(capsys)
    assert lines[-1] == f"{STDOUT_STATUS_PREFIX}: failed"


def test_no_unhandled_traceback_reaches_stdout(tmp_path, capsys):
    exit_code = main(env={ENV_FAILURE_KIND: "ci", ENV_REPO_PATH: str(tmp_path / "missing")})
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "Traceback (most recent call last)" not in out


# --- render_stdout_lines / compute_run_result as pure functions ------------


def test_render_stdout_lines_puts_status_line_last():
    result = RunResult(status="failed", reason="line one\nline two")
    lines = render_stdout_lines(result)
    assert lines[-1] == f"{STDOUT_STATUS_PREFIX}: failed"
    # multi-line reasons are collapsed so they can never be mistaken for
    # (or push past) the true last line
    assert "\n" not in lines[0]
    assert lines[0] == f"{STDOUT_REASON_PREFIX}: line one line two"


def test_render_stdout_lines_omits_reason_line_when_reason_is_empty():
    result = RunResult(status="no_safe_fix", reason="")
    lines = render_stdout_lines(result)
    assert lines == [f"{STDOUT_STATUS_PREFIX}: no_safe_fix"]


@pytest.mark.parametrize(
    "env",
    [
        {},
        {ENV_FAILURE_KIND: "audit"},
    ],
)
def test_compute_run_result_never_raises_for_bad_input(env):
    result = compute_run_result(env)
    assert result.status in ("failed", "no_safe_fix", "proposed")
