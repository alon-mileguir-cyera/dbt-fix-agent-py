"""Tests for `dbt_fixer.pipeline.run_stage1`: env + intake errors always map
to a clean, typed terminal `RunResult`, never an unhandled exception."""

from __future__ import annotations

import dbt_fixer.pipeline as pipeline_module
from dbt_fixer.env import ENV_FAILURE_KIND, ENV_FAILURE_CONTEXT, ENV_REPO_PATH
from dbt_fixer.pipeline import run_stage1


def test_missing_required_env_resolves_to_failed():
    outcome = run_stage1({})
    assert outcome.terminal is not None
    assert outcome.terminal.status == "failed"
    assert outcome.config is None
    assert outcome.intake is None


def test_invalid_repo_path_resolves_to_failed(tmp_path):
    env = {ENV_FAILURE_KIND: "ci", ENV_REPO_PATH: str(tmp_path / "nope")}
    outcome = run_stage1(env)
    assert outcome.terminal is not None
    assert outcome.terminal.status == "failed"


def test_unparseable_context_resolves_to_no_safe_fix(tmp_path):
    env = {
        ENV_FAILURE_KIND: "ci",
        ENV_REPO_PATH: str(tmp_path),
        ENV_FAILURE_CONTEXT: "garbage unrelated text",
    }
    outcome = run_stage1(env)
    assert outcome.terminal is not None
    assert outcome.terminal.status == "no_safe_fix"
    assert outcome.terminal.reason
    assert outcome.config is not None  # env validation succeeded before intake ran


def test_empty_context_resolves_to_no_safe_fix(tmp_path):
    env = {ENV_FAILURE_KIND: "ci", ENV_REPO_PATH: str(tmp_path)}
    outcome = run_stage1(env)
    assert outcome.terminal is not None
    assert outcome.terminal.status == "no_safe_fix"


def test_unexpected_exception_from_load_config_resolves_to_failed(monkeypatch):
    """The defensive `except Exception` around `load_config` (a genuine
    programming-error safety net, distinct from the documented
    `EnvValidationError` path) must also resolve cleanly rather than
    propagate."""

    def _boom(_env):
        raise RuntimeError("boom: unexpected programming error in load_config")

    monkeypatch.setattr(pipeline_module, "load_config", _boom)
    outcome = run_stage1({})
    assert outcome.terminal is not None
    assert outcome.terminal.status == "failed"
    assert "unexpected error validating environment" in outcome.terminal.reason
    assert "boom" in outcome.terminal.reason
    assert outcome.config is None
    assert outcome.intake is None


def test_unexpected_exception_from_resolve_intake_resolves_to_failed(monkeypatch, tmp_path):
    """The defensive `except Exception` around `resolve_intake` must also
    resolve cleanly rather than propagate, and must preserve the
    already-validated config on the outcome."""

    def _boom(_config):
        raise RuntimeError("boom: unexpected programming error in resolve_intake")

    monkeypatch.setattr(pipeline_module, "resolve_intake", _boom)
    env = {ENV_FAILURE_KIND: "ci", ENV_REPO_PATH: str(tmp_path)}
    outcome = run_stage1(env)
    assert outcome.terminal is not None
    assert outcome.terminal.status == "failed"
    assert "unexpected error during intake" in outcome.terminal.reason
    assert "boom" in outcome.terminal.reason
    assert outcome.config is not None
    assert outcome.intake is None


def test_valid_target_does_not_resolve_terminal_yet(tmp_path):
    env = {
        ENV_FAILURE_KIND: "ci",
        ENV_REPO_PATH: str(tmp_path),
        ENV_FAILURE_CONTEXT: (
            "Completed with 1 error\n\n"
            "Failure in test x (models/y.sql)\n  bad\n\nDone."
        ),
    }
    outcome = run_stage1(env)
    assert outcome.terminal is None
    assert outcome.config is not None
    assert outcome.intake is not None
    assert outcome.intake.ok
    assert outcome.intake.target.identifiers == ("x",)
