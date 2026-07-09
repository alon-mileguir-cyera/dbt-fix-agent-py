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


# ---------------------------------------------------------------------------
# SAFETY: judgment-critical failures are declined up front, never auto-fixed
# ---------------------------------------------------------------------------

_JUDGMENT_REPORT_TMPL = (
    "# Verdict: **BLOCKED**\n\n"
    "### {name} (`{cid}`)\n\n"
    "**Severity:** Critical &nbsp; **Score:** 20/100 &nbsp; **State:** **FAIL**\n\n"
    "**Evidence:**\n\n> {name} regression\n"
)


def test_judgment_critical_checks_are_declined_up_front(tmp_path, monkeypatch):
    """A tenant-isolation / RAP / destructive / sensitive-data block must
    resolve to no_safe_fix WITHOUT any proposal attempt (never auto-fixed)."""
    from dbt_fixer.intake import parse_failure_target

    for cid, name in [
        ("tenant_isolation_integrity", "Tenant Isolation Integrity"),
        ("rap_bypass_logic", "RAP Bypass Logic Safety"),
        ("destructive_operation_safety", "Destructive Operation Safety"),
        ("credentials_exposure", "Credentials Exposure"),
    ]:
        report = _JUDGMENT_REPORT_TMPL.format(cid=cid, name=name)
        target, reason = parse_failure_target("audit", report)
        assert target is not None, reason
        assert target.judgment_critical_blocking_ids == (cid,), cid


def test_mechanical_criticals_are_not_declined(tmp_path):
    """schema_contract_verification and downstream_dependency_impact are
    mechanical and remain fixable (not judgment-critical)."""
    from dbt_fixer.intake import parse_failure_target

    for cid, name in [
        ("schema_contract_verification", "Schema Contract Verification"),
        ("downstream_dependency_impact", "Downstream Dependency Impact"),
    ]:
        report = _JUDGMENT_REPORT_TMPL.format(cid=cid, name=name)
        target, reason = parse_failure_target("audit", report)
        assert target is not None, reason
        assert target.judgment_critical_blocking_ids == (), cid


def test_mixed_block_with_a_judgment_critical_still_declines(tmp_path):
    """If a judgment-critical is among the blocking checks, decline even
    though a mechanical check is also present (the efficacy gate could never
    clear the judgment-critical anyway)."""
    from dbt_fixer.intake import parse_failure_target

    report = (
        "# Verdict: **BLOCKED**\n\n"
        "### Schema Contract Verification (`schema_contract_verification`)\n\n"
        "**Severity:** Critical &nbsp; **State:** **FAIL**\n\n> mismatch\n\n"
        "### Tenant Isolation Integrity (`tenant_isolation_integrity`)\n\n"
        "**Severity:** Critical &nbsp; **State:** **FAIL**\n\n> dropped filter\n"
    )
    target, reason = parse_failure_target("audit", report)
    assert target is not None, reason
    assert "tenant_isolation_integrity" in target.judgment_critical_blocking_ids


def test_run_stage1_declines_judgment_critical_end_to_end(tmp_path):
    """Full Stage 1: a tenant-isolation BLOCK terminates as no_safe_fix with
    a human-review reason, before any proposal is attempted."""
    report = (
        "# Verdict: **BLOCKED**\n\n"
        "### Tenant Isolation Integrity (`tenant_isolation_integrity`)\n\n"
        "**Severity:** Critical &nbsp; **State:** **FAIL**\n\n"
        "**Evidence:**\n\n> a tenant filter was removed\n"
    )
    env = {
        ENV_FAILURE_KIND: "audit",
        ENV_REPO_PATH: str(tmp_path),
        ENV_FAILURE_CONTEXT: report,
    }
    outcome = run_stage1(env)
    assert outcome.terminal is not None
    assert outcome.terminal.status == "no_safe_fix"
    assert "human review" in outcome.terminal.reason
    assert "tenant_isolation_integrity" in outcome.terminal.reason


def test_run_stage1_allows_mechanical_critical_to_proceed(tmp_path):
    """A schema_contract BLOCK is fixable -> Stage 1 proceeds (terminal None)."""
    report = (
        "# Verdict: **BLOCKED**\n\n"
        "### Schema Contract Verification (`schema_contract_verification`)\n\n"
        "**Severity:** Critical &nbsp; **State:** **FAIL**\n\n"
        "**Evidence:**\n\n> declared column not in output\n"
    )
    env = {
        ENV_FAILURE_KIND: "audit",
        ENV_REPO_PATH: str(tmp_path),
        ENV_FAILURE_CONTEXT: report,
    }
    outcome = run_stage1(env)
    assert outcome.terminal is None  # proceeds to the fix attempt
