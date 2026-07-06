"""Tests for `dbt_fixer.agent`: toolkit wiring, bounded tool calls, and the
static proof that no write tool is ever exposed to the model.

Constructing `boto3.client(...)`, `AwsBedrock(...)`, and `agno.agent.Agent(...)`
objects does not perform any real network I/O (confirmed by experimentation),
so these can be constructed directly under the offline-only `conftest.py`
guard; only an actual `.run()`/API call would need a fake.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dbt_fixer.agent import (
    FixerAgentConfig,
    build_bedrock_model,
    build_fixer_agent,
    build_repo_toolkit,
)
from dbt_fixer.bounds import Bounds, ExecutionBudget, ToolCallCapExceededError
from dbt_fixer.tools.repo_tools import RepoTools


def _make_repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    (root / "models").mkdir(parents=True)
    (root / "models" / "a.sql").write_text("select 1", encoding="utf-8")
    return root


def test_toolkit_exposes_exactly_read_and_search_no_write_tool(tmp_path: Path) -> None:
    root = _make_repo(tmp_path)
    repo_tools = RepoTools(root)

    toolkit = build_repo_toolkit(repo_tools)

    tool_names = set(toolkit.functions.keys())
    assert tool_names == {"read_repo_file", "search_repo_files"}

    write_like_keywords = ("write", "create", "delete", "remove", "rename", "mkdir", "unlink")
    for name in tool_names:
        assert not any(keyword in name.lower() for keyword in write_like_keywords)


def test_toolkit_entrypoints_delegate_to_repo_tools_read_and_search(tmp_path: Path) -> None:
    root = _make_repo(tmp_path)
    repo_tools = RepoTools(root)

    toolkit = build_repo_toolkit(repo_tools)

    read_fn = toolkit.functions["read_repo_file"].entrypoint
    search_fn = toolkit.functions["search_repo_files"].entrypoint

    assert read_fn("models/a.sql") == "select 1"
    assert search_fn("**/*.sql", "models") == ["models/a.sql"]


def test_toolkit_tool_calls_are_bounded_by_execution_budget(tmp_path: Path) -> None:
    root = _make_repo(tmp_path)
    repo_tools = RepoTools(root)
    budget = ExecutionBudget(Bounds(max_tool_calls=1))
    toolkit = build_repo_toolkit(repo_tools, budget=budget)
    read_fn = toolkit.functions["read_repo_file"].entrypoint

    assert read_fn("models/a.sql") == "select 1"  # first call: within budget

    with pytest.raises(ToolCallCapExceededError):
        read_fn("models/a.sql")  # second call: exceeds the cap of 1


def test_build_bedrock_model_constructs_without_network_access(tmp_path: Path) -> None:
    # boto3.client(...) construction and AwsBedrock(...) construction do not
    # perform real network I/O; the autouse conftest fixture blocking real
    # sockets/subprocesses is active for this test and this must not raise.
    config = FixerAgentConfig(repo_root=_make_repo(tmp_path))

    model = build_bedrock_model(config)

    assert model.id == config.model_id


def test_build_fixer_agent_wires_read_only_toolkit(tmp_path: Path) -> None:
    root = _make_repo(tmp_path)
    budget = ExecutionBudget(Bounds(max_tool_calls=5))
    config = FixerAgentConfig(repo_root=root)

    agent = build_fixer_agent(config, budget=budget)

    toolkits = [t for t in agent.tools if hasattr(t, "functions")]
    assert len(toolkits) == 1
    assert set(toolkits[0].functions.keys()) == {"read_repo_file", "search_repo_files"}
    assert agent.tool_call_limit == 5
