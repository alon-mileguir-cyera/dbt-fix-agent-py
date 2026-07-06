# dbt Fix Agent (Shadow Mode)

A sealed, single-purpose Python package that proposes narrowly-scoped,
mechanically-gated repairs for a known-red dbt Cloud CI check or an
auditor-`BLOCKED` PR, proves the fix through independent adversarial gates,
and posts the result to Slack. **It never writes to GitHub** — there is no
write-capable credential in its environment and no code path that pushes,
comments, or opens a PR.

This README documents the package's environment contract as each sprint
lands. Sprint 1 establishes the contract's shape (fail-closed required
variables, fail-safe-default numeric bounds) and the variables owned by that
sprint's modules; later sprints add their own rows to the tables below as
their features (Bedrock model access, Slack delivery, the auditor subprocess
integration) land.

## Running the tests

```
pip install -e ".[test]"
pytest
```

The suite is fully offline: `tests/conftest.py` actively blocks real network
sockets and real subprocess spawns for every test, except a test explicitly
marked `@pytest.mark.real_process` (reserved, starting in a later sprint, for
one clearly-marked real-process integration module).

## Environment contract

### Core run configuration (`dbt_fixer.env`)

| Variable | Required | Default when unset | Malformed-value handling |
|---|---|---|---|
| `DBT_FIXER_FAILURE_KIND` | **yes** | n/a | Missing/blank/invalid (must be `ci` or `audit`) → `EnvValidationError`, run resolves `failed`. |
| `DBT_FIXER_REPO_PATH` | **yes** | n/a | Missing/blank, or path does not exist / is not a directory → `EnvValidationError`, run resolves `failed`. |
| `DBT_FIXER_PR_TITLE` | no | `""` | Free text; no validation. |
| `DBT_FIXER_PR_DESCRIPTION` | no | `""` | Free text; no validation. |
| `DBT_FIXER_PR_DIFF` | no | `""` | Free text; no validation. |
| `DBT_FIXER_PR_URL` | no | `""` | Free text; no validation. |
| `DBT_FIXER_FAILURE_CONTEXT` | no | `""` | Free text; an empty or unparseable value is handled by `dbt_fixer.intake`, resolving the run to `no_safe_fix` with a specific reason — never treated as an environment error. |
| `DBT_FIXER_SLACK_CHANNEL` | no | `None` | Free text; unset means Slack delivery is skipped (a no-op), not an error. |
| `DBT_FIXER_AUDITOR_PYTHON` | no | `None` | Free text path to the sibling auditor's interpreter; unset is a hard `no_safe_fix` at re-audit-gate time (a later sprint), never a skipped gate. |
| `DBT_FIXER_MAX_ROUNDS` | no | `3` | Non-numeric, or outside `[1, 10]` → falls back to `3` and records a warning (never crashes, never clamps to the nearest bound). |

### Bounded-execution primitive (`dbt_fixer.bounds`)

Every model-calling pass in this package runs through the same
`ExecutionBudget`, which enforces these three limits independently and
simultaneously:

| Variable | Required | Default when unset | Valid range | Malformed-value handling |
|---|---|---|---|---|
| `DBT_FIXER_TIMEOUT_SECONDS` | no | `300` | `[1, 3600]` | Falls back to `300` and records a warning. |
| `DBT_FIXER_MAX_TOOL_CALLS` | no | `40` | `[1, 500]` | Falls back to `40` and records a warning. |
| `DBT_FIXER_MAX_TURNS` | no | `8` | `[1, 100]` | Falls back to `8` and records a warning. |

None of these variables ever raise: an out-of-range or non-numeric value
degrades to the documented default rather than crashing the process or
silently clamping to the nearest valid bound.

`DBT_FIXER_TIMEOUT_SECONDS` is float-typed, so it also explicitly rejects
non-finite values that `float()` would otherwise parse successfully --
`nan`, `inf`, `-inf`, and overflow strings like `1e400` all fall back to
`300` with a recorded warning, exactly like any other malformed value. This
is checked ahead of the range comparison because IEEE 754 makes every
ordering comparison against NaN evaluate to `False`, which would otherwise
let a NaN timeout silently pass the `[1, 3600]` range check and permanently
disable the wall-clock timeout it's supposed to enforce.

## Package layout

```
src/dbt_fixer/
  env.py            # DBT_FIXER_* required/optional contract, fail-closed on required
  bounds.py          # timeout/tool-call-cap/turn-limit primitive, fail-safe on malformed
  _numeric.py        # shared fail-safe numeric-bound parsing helper
  scratch.py         # scratch-copy lifecycle (create, use, guaranteed cleanup)
  fencing.py         # untrusted-content fencing + lookalike-marker neutralization
  intake.py          # failure-context -> structured target, or an honest no_safe_fix
  pipeline.py        # stage-1 orchestration: env + intake -> terminal RunResult or continue
  status.py          # the fixed proposed/no_safe_fix/failed vocabulary and glyphs
  logging_utils.py   # stderr-only diagnostic logging (stdout stays a clean machine surface)
  pathsafe.py         # shared path-containment guard (rejects '..', absolute paths, symlink escapes)
  tools/
    repo_tools.py     # RepoTools: rooted, read-only file read/glob-search -- no write method exists
  model_output.py      # tolerant-but-never-trusting JSON-object extraction from raw model text
  proposal.py           # structured fix-proposal schema (whole_file_replace/line_range_edit) + bounded model pass
  agent.py               # Bedrock/agno agent wiring; the only toolkit it builds exposes read/search only
  applier.py              # fail-closed, two-phase application of a Proposal onto an isolated scratch copy
  diffing.py               # pure difflib unified-diff generation, matching real `git diff` semantics
  fix_pipeline.py           # Stage 2 orchestration: read -> propose -> apply -> diff, fully offline-testable
```

Later sprints add the allowlist and re-audit gates, the fix-refuter and
`dbt parse` gates, the bounded retry loop, and the Slack/stdout delivery
contract — each with its own additions to this README's environment-contract
tables.

## Sprint 2: path-safe repo tools, structured fix proposal, scratch-copy applier

**No write tool is ever exposed to a model.** `dbt_fixer.tools.repo_tools.RepoTools`
exposes exactly `read_file`/`search_files`, both scoped to a fixed repo root
via `dbt_fixer.pathsafe.resolve_within_root` (rejects non-string/empty,
absolute, and `..`-containing paths, and follows symlinks before the final
containment check). `dbt_fixer.agent.build_repo_toolkit` wraps only those two
methods as the `read_repo_file`/`search_repo_files` agno tools; there is no
create/write/delete/rename capability reachable from anything a model can
call. `read_file` raises `PathTraversalError` for a symlink that escapes the
root; `search_files` instead silently excludes individual escaping matches
found during glob enumeration (while still raising if the `pattern`/
`relative_dir` arguments themselves attempt traversal).

**Structured fix proposals are the only way a fix is ever proposed.**
`dbt_fixer.proposal.parse_proposal` enforces a closed JSON schema (exact
top-level and per-edit key sets, no extra fields, only the two edit types
`whole_file_replace`/`line_range_edit`); any mismatch — malformed JSON, a
missing field, an extra key, an unrecognized edit type, a single bad edit
among otherwise-good ones — resolves to `None` ("no proposal"), never a
partial or guessed acceptance. `dbt_fixer.proposal.run_proposal_pass` runs
this behind the Sprint 1 `ExecutionBudget`: a turn is recorded before the
model is ever called, and any `BoundedExecutionError` from the budget or the
runner itself resolves to an honest no-proposal result rather than hanging.

**Edits are applied only to an isolated scratch copy.** `dbt_fixer.applier.apply_proposal`
validates every edit in a proposal (target exists, target is a file, every
line range is in bounds, no two edits conflict) *before* mutating anything;
a single invalid or conflicting edit raises a specific `ApplyError` subclass
and leaves the scratch copy completely untouched. The original checkout
(`dbt_fixer.env.FixerConfig.repo_path`) is never passed to the applier.

**Diffs are pure-Python and match real `git diff`.** `dbt_fixer.diffing.generate_unified_diff`
uses only `difflib.unified_diff` — no subprocess, no real git — and is
verified byte-identical (aside from the `diff --git`/`index`/`new file mode`
header lines, which are normalized away in the comparison) to a real `git
diff` for add-only, delete-only, and mixed-change cases in
`tests/real_process/test_diff_matches_git.py`, the one test module in this
package marked `@pytest.mark.real_process`.

`dbt_fixer.fix_pipeline.run_fix_pipeline` wires all of the above into the
full Stage 2 sequence and is proven, offline, to produce byte-identical diff
output across repeated runs of a fixed fake model runner against a fixed
sample repo.

Two additional, unprefixed environment variables (matching the sibling
`dbt-audit-agent-py` package's operator convention, not part of the
`DBT_FIXER_*` contract above) control Bedrock model selection:

| Variable | Required | Default when unset |
|---|---|---|
| `BEDROCK_MODEL_ID` | no | `us.anthropic.claude-sonnet-5` |
| `AWS_REGION` | no | `us-east-1` |

AWS credentials are always resolved via boto3's default credential chain;
no access key, secret key, or profile is ever hardcoded.
