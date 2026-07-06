"""Bedrock-backed agno agent core for the dbt Fix Agent's proposal pass.

Wires an agno :class:`~agno.agent.Agent` to Amazon Bedrock via agno's
``AwsBedrock`` model wrapper, matching the pinned-model, credential, and
tooling conventions used by Cyera's sibling dbt-audit-agent-py package.

Authentication is always via boto3's default credential chain (environment
variables, shared config/credentials files, EC2/ECS/Lambda instance roles,
or an assumed role via ``AWS_PROFILE`` set in the environment). No AWS
access key, secret key, session token, or profile override is ever
hardcoded here.

No Anthropic API key or direct Anthropic client is used anywhere: model
access is exclusively through Bedrock.

**No write tool is ever exposed to the model.** The only toolkit this
module builds (`build_repo_toolkit`) wraps exactly the two read-only
methods on `dbt_fixer.tools.repo_tools.RepoTools` -- `read_file` and
`search_files` -- as `read_repo_file`/`search_repo_files`. There is no
function anywhere in this module, or reachable from the toolkit it builds,
that can create, modify, or delete a file in the checkout the model is
reading from. Structured edits are applied later, by `dbt_fixer.applier`,
against an isolated scratch copy the model itself never has tool access to.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

import boto3
from agno.agent import Agent
from agno.models.aws import AwsBedrock
from agno.tools.toolkit import Toolkit
from botocore.config import Config

from .bounds import ExecutionBudget
from .tools.repo_tools import RepoTools

# Pinned model id for the fixer, env-overridable exactly like the sibling
# auditor package's convention (unprefixed -- these two variables are a
# cross-package operator convention, not part of the `DBT_FIXER_*` contract
# owned by `dbt_fixer.env`).
BEDROCK_MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "us.anthropic.claude-sonnet-5")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

# Bounded boto3 client config so a slow/stalled Bedrock response FAILS
# instead of hanging indefinitely -- essential so a stalled model call
# cannot defeat `ExecutionBudget`'s wall-clock timeout by hanging inside
# the single `runner(prompt)` call that timeout wraps around.
_BEDROCK_CONFIG = Config(
    connect_timeout=int(os.getenv("BEDROCK_CONNECT_TIMEOUT", "10")),
    read_timeout=int(os.getenv("BEDROCK_READ_TIMEOUT", "120")),
    retries={"max_attempts": int(os.getenv("BEDROCK_MAX_ATTEMPTS", "3")), "mode": "standard"},
)


def build_repo_toolkit(repo_tools: RepoTools, *, budget: Optional[ExecutionBudget] = None) -> Toolkit:
    """Wrap a `RepoTools` instance's read/search methods as an agno toolkit.

    The returned toolkit exposes exactly two tools to the agent --
    `read_repo_file` and `search_repo_files` -- both of which delegate to
    the path-safe `RepoTools` instance and therefore inherit its
    containment guarantees. The agent itself never receives raw filesystem
    access, and no third tool of any kind (write, create, delete, rename)
    is ever added to this toolkit.

    When `budget` is supplied, every tool invocation calls
    `budget.record_tool_call()` *before* doing any real work, so a pass
    that exceeds the shared tool-call cap or wall-clock timeout raises
    `BoundedExecutionError` immediately rather than continuing to call
    tools unboundedly.
    """

    def read_repo_file(relative_path: str) -> str:
        """Read a text file from the repository under review.

        Args:
            relative_path: Path relative to the repository root, e.g.
                `"models/staging/stg_customers.sql"`. Absolute paths and
                `..` traversal are rejected.
        """

        if budget is not None:
            budget.record_tool_call()
        return repo_tools.read_file(relative_path)

    def search_repo_files(pattern: str, relative_dir: str = ".") -> list[str]:
        """Search for files in the repository under review by glob pattern.

        Args:
            pattern: A glob pattern, e.g. `"*.sql"` or `"models/**/*.sql"`.
            relative_dir: Directory (relative to the repository root) to
                search under. Defaults to the repository root itself.
        """

        if budget is not None:
            budget.record_tool_call()
        return list(repo_tools.search_files(pattern, relative_dir))

    return Toolkit(name="repo_tools", tools=[read_repo_file, search_repo_files])


@dataclass
class FixerAgentConfig:
    """Configuration for constructing the fixer's Bedrock-backed agent."""

    repo_root: "str | Path"
    model_id: str = BEDROCK_MODEL_ID
    aws_region: Optional[str] = None
    # Newer Bedrock models (Sonnet 5) deprecate `temperature` outright via the
    # Converse API; default to None (omit it) and only send it if a caller
    # explicitly sets one for an older model.
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    instructions: List[str] = field(default_factory=list)


def build_bedrock_model(config: FixerAgentConfig) -> AwsBedrock:
    """Construct the pinned Bedrock model wrapper for the fixer.

    Credentials are never passed explicitly -- `AwsBedrock` (and the boto3
    session/client it is handed) resolves them via boto3's default
    credential chain. `aws_region` may be supplied to select a region
    without touching credential resolution.
    """

    region = config.aws_region or AWS_REGION
    bedrock_runtime = boto3.client("bedrock-runtime", region_name=region, config=_BEDROCK_CONFIG)
    optional_params: dict[str, object] = {}
    if config.temperature is not None:
        optional_params["temperature"] = config.temperature
    if config.max_tokens is not None:
        optional_params["max_tokens"] = config.max_tokens
    return AwsBedrock(id=config.model_id, client=bedrock_runtime, **optional_params)


def build_fixer_agent(config: FixerAgentConfig, *, budget: Optional[ExecutionBudget] = None) -> Agent:
    """Build the core agno Agent for the dbt Fix Agent's proposal pass.

    The agent is wired to the pinned Bedrock model and to a path-safe,
    read-only toolkit scoped to `config.repo_root`. No write tool is ever
    added: `build_repo_toolkit` is the only toolkit wiring path used here,
    and it exposes exactly `read_repo_file`/`search_repo_files`.

    When `budget` is supplied, both agno's own native `tool_call_limit`
    (a first enforcement layer) and this package's `ExecutionBudget`
    (wired into every exposed tool call as a second, independent
    enforcement layer that does not depend on agno's internal bookkeeping)
    are active.
    """

    repo_tools = RepoTools(config.repo_root)
    toolkit = build_repo_toolkit(repo_tools, budget=budget)
    model = build_bedrock_model(config)

    tool_call_limit = budget.bounds.max_tool_calls if budget is not None else None

    return Agent(
        model=model,
        tools=[toolkit],
        instructions=config.instructions,
        markdown=False,
        tool_call_limit=tool_call_limit,
        reasoning=False,
    )


def build_agent_runner(agent: Agent) -> Callable[[str], str]:
    """Adapt an agno `Agent` into the plain `Callable[[str], str]` "runner"
    shape `dbt_fixer.proposal.run_proposal_pass` expects.

    Any exception raised by the underlying `agent.run` call is left to
    propagate -- `run_proposal_pass` is what actually converts an
    exception (including a `BoundedExecutionError` raised from inside a
    tool call) into a fail-closed "no proposal" outcome; this adapter
    stays a thin, honest pass-through so it never masks a real error.
    """

    def _run(prompt: str) -> str:
        run_output = agent.run(prompt)
        content = getattr(run_output, "content", None)
        if content is None:
            return ""
        return content if isinstance(content, str) else str(content)

    return _run
