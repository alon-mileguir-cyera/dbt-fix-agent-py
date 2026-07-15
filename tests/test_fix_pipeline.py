"""Tests for `dbt_fixer.fix_pipeline.run_fix_pipeline`: the full offline
read-propose-apply-diff sequence.

Covers: byte-identical diff output across two runs of a fixed fake model
runner against a fixed sample repo (no network/subprocess calls -- enforced
by the always-on `conftest.py` guard, since this module is not marked
`real_process`); the honest "no proposal" outcome when the fake runner
answers with malformed output; and the fail-closed "apply failed" outcome
when the (schema-valid) proposal references a conflicting/invalid edit,
with the original checkout left untouched in every case.
"""

from __future__ import annotations

import json
from pathlib import Path

from dbt_fixer.bounds import Bounds, ExecutionBudget
from dbt_fixer.fencing import fence_context
from dbt_fixer.fix_pipeline import run_fix_pipeline


def _make_repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    (root / "models").mkdir(parents=True)
    (root / "models" / "a.sql").write_text("select 1\n", encoding="utf-8")
    return root


def _fenced_context():
    return fence_context({"failure_context": "compilation error in models/a.sql"})


def test_pipeline_end_to_end_is_byte_identical_across_repeated_runs(tmp_path: Path) -> None:
    repo_root = _make_repo(tmp_path)
    valid_raw = json.dumps(
        {
            "edits": [
                {"type": "whole_file_replace", "path": "models/a.sql", "content": "select 2\n"}
            ],
            "rationale": "fixed the failing model",
        }
    )
    fenced = _fenced_context()

    result_one = run_fix_pipeline(
        repo_root, fenced, lambda prompt: valid_raw, ExecutionBudget(Bounds())
    )
    result_two = run_fix_pipeline(
        repo_root, fenced, lambda prompt: valid_raw, ExecutionBudget(Bounds())
    )

    assert result_one.ok
    assert result_two.ok
    assert result_one.diff == result_two.diff
    assert result_one.diff is not None
    assert "diff --git a/models/a.sql b/models/a.sql" in result_one.diff
    assert "+select 2" in result_one.diff


def test_pipeline_never_mutates_the_original_checkout(tmp_path: Path) -> None:
    repo_root = _make_repo(tmp_path)
    original = (repo_root / "models" / "a.sql").read_text(encoding="utf-8")
    valid_raw = json.dumps(
        {
            "edits": [
                {"type": "whole_file_replace", "path": "models/a.sql", "content": "select 999\n"}
            ],
            "rationale": "fix",
        }
    )

    result = run_fix_pipeline(
        repo_root, _fenced_context(), lambda prompt: valid_raw, ExecutionBudget(Bounds())
    )

    assert result.ok
    assert (repo_root / "models" / "a.sql").read_text(encoding="utf-8") == original


def test_pipeline_returns_no_proposal_result_for_malformed_model_output(tmp_path: Path) -> None:
    repo_root = _make_repo(tmp_path)

    result = run_fix_pipeline(
        repo_root, _fenced_context(), lambda prompt: "not json at all", ExecutionBudget(Bounds())
    )

    assert not result.ok
    assert result.diff is None
    assert result.reason is not None
    assert result.proposal_pass is not None and not result.proposal_pass.ok


def test_pipeline_threads_tool_free_finalizer_for_empty_primary_output(tmp_path: Path) -> None:
    repo_root = _make_repo(tmp_path)
    final_raw = json.dumps(
        {
            "edits": [
                {"type": "whole_file_replace", "path": "models/a.sql", "content": "select 2\n"}
            ],
            "rationale": "finalized from pre-loaded evidence",
        }
    )
    finalizer_prompts: list[str] = []

    result = run_fix_pipeline(
        repo_root,
        _fenced_context(),
        lambda prompt: "",
        ExecutionBudget(Bounds()),
        finalizer_runner=lambda prompt: finalizer_prompts.append(prompt) or final_raw,
    )

    assert result.ok
    assert len(finalizer_prompts) == 1
    assert "Original structured-proposal request" in finalizer_prompts[0]
    assert result.diff is not None and "+select 2" in result.diff


def test_pipeline_fails_closed_for_a_proposal_targeting_a_missing_file(tmp_path: Path) -> None:
    repo_root = _make_repo(tmp_path)
    raw = json.dumps(
        {
            "edits": [
                {
                    "type": "whole_file_replace",
                    "path": "models/does_not_exist.sql",
                    "content": "select 1\n",
                }
            ],
            "rationale": "targets a file that isn't there",
        }
    )

    result = run_fix_pipeline(repo_root, _fenced_context(), lambda prompt: raw, ExecutionBudget(Bounds()))

    assert not result.ok
    assert result.diff is None
    assert result.reason is not None and "could not be applied" in result.reason
