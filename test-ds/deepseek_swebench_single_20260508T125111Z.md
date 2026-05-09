# DeepSeek SWE-bench Single Run Plan

## Goal

Run one SWE-bench Lite `dev` instance with DeepSeek and save a trajectory locally.

## Key decisions

- Use `uv run mini-extra swebench-single` because this repo is managed through `uv`.
- Use a single explicit instance id rather than numeric index `0`, because `swebench-single` sorts `instance_id`s before resolving numeric indices.
- Use DeepSeek through the OpenAI-compatible API path, not the Anthropic-compatible path, because the repo's default tool-call wrapper is compatible with the OpenAI schema but is rejected by the Anthropic DeepSeek endpoint for this use case.
- Use a local prepared repo checkout plus a local benchmark override config because Docker is installed here but the current user cannot access `/var/run/docker.sock`.
- Use the non-interactive `default` agent class for this local fallback so step-limit exits do not block on terminal prompts.

## Command shape

```bash
DEEPSEEK_API_KEY=... uv run mini-extra swebench-single \
  --subset lite \
  --split dev \
  -i sqlfluff__sqlfluff-1625 \
  -m openai/deepseek-v4-flash \
  -c swebench.yaml \
  -c swebench_local_debug.yaml \
  -c agent.step_limit=3 \
  -c model.model_kwargs.api_key="$DEEPSEEK_API_KEY" \
  -c model.model_kwargs.api_base=https://api.deepseek.com \
  -c model.cost_tracking=ignore_errors \
  -c environment.cwd=/tmp/mini-swe-agent-swebench/sqlfluff__sqlfluff-1625/testbed \
  -c environment.timeout=120 \
  -c environment.env.PATH=/tmp/mini-swe-agent-swebench/sqlfluff__sqlfluff-1625/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
  -c environment.env.VIRTUAL_ENV=/tmp/mini-swe-agent-swebench/sqlfluff__sqlfluff-1625/venv \
  --exit-immediately \
  -y \
  -o ./deepseek-sqlfluff-1625-local.traj.json
```

This is the bounded smoke-test command that produced the verified local trajectory.

## Local setup

1. Clone `sqlfluff/sqlfluff` into `/tmp/mini-swe-agent-swebench/sqlfluff__sqlfluff-1625/testbed`.
2. Checkout commit `14e1a23a3166b9a645a16de96f694c77a5d4abb7`.
3. Create `/tmp/mini-swe-agent-swebench/sqlfluff__sqlfluff-1625/venv`.
4. Install runtime requirements and editable package.
5. Pin `setuptools<81` and `click==8.1.7` so this historical repo snapshot works with its own CLI/tests.

## Verification targets

- `uv run mini-extra swebench-single --help` works.
- A minimal DeepSeek tool-call smoke test succeeds with `openai/deepseek-v4-flash` and `https://api.deepseek.com`.
- Local target repo imports successfully.
- Two representative `sqlfluff` CLI tests pass before the agent run.
- The final trajectory file exists and contains the expected model name plus agent messages.

## Step-by-step commands

Use the explicit instance id `sqlfluff__sqlfluff-1625`. Do not use `-i 0` for this setup, because `swebench-single` sorts instance ids before resolving numeric indices.

### 1. Prepare the local testbed

```bash
WORK=/tmp/mini-swe-agent-swebench/sqlfluff__sqlfluff-1625
TESTBED="$WORK/testbed"
VENV="$WORK/venv"

mkdir -p "$WORK"

if [ ! -d "$TESTBED/.git" ]; then
  git clone https://github.com/sqlfluff/sqlfluff.git "$TESTBED"
fi

git -C "$TESTBED" fetch --all --tags --quiet
git -C "$TESTBED" checkout 14e1a23a3166b9a645a16de96f694c77a5d4abb7

if [ ! -x "$VENV/bin/python" ]; then
  python3 -m venv "$VENV"
fi

"$VENV/bin/pip" install -U pip wheel
"$VENV/bin/pip" install -r "$TESTBED/requirements.txt" -e "$TESTBED"
"$VENV/bin/pip" install 'setuptools<81' 'click==8.1.7'
```

### 2. Sanity-check the prepared repo

```bash
WORK=/tmp/mini-swe-agent-swebench/sqlfluff__sqlfluff-1625
TESTBED="$WORK/testbed"
VENV="$WORK/venv"

cd "$TESTBED" && "$VENV/bin/pytest" -q test/cli/commands_test.py::test__cli__command_dialect
cd "$TESTBED" && "$VENV/bin/pytest" -q test/cli/commands_test.py::test__cli__command_directed
```

### 3. Run a longer single-instance attempt

This command keeps the same local fallback but raises the step limit from `3` to `80`, which is a practical ceiling for a real attempt without jumping straight to the YAML default of `250`.

```bash
export DEEPSEEK_API_KEY='replace-with-your-real-key'

WORK=/tmp/mini-swe-agent-swebench/sqlfluff__sqlfluff-1625
TESTBED="$WORK/testbed"
VENV="$WORK/venv"

uv run mini-extra swebench-single \
  --subset lite \
  --split dev \
  -i sqlfluff__sqlfluff-1625 \
  -m openai/deepseek-v4-flash \
  -c swebench.yaml \
  -c swebench_local_debug.yaml \
  -c agent.step_limit=80 \
  -c model.model_kwargs.api_key="$DEEPSEEK_API_KEY" \
  -c model.model_kwargs.api_base=https://api.deepseek.com \
  -c model.cost_tracking=ignore_errors \
  -c environment.cwd="$TESTBED" \
  -c environment.timeout=300 \
  -c environment.env.PATH="$VENV/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
  -c environment.env.VIRTUAL_ENV="$VENV" \
  -o ./deepseek-sqlfluff-1625-long.traj.json
```

If this run stops with `LimitsExceeded`, increase `agent.step_limit` to `120` or remove that override to fall back to the benchmark config's default `250`-step cap.

### 4. Inspect the saved trajectory

```bash
uv run mini-extra inspect ./deepseek-sqlfluff-1625-long.traj.json
```

The output file will contain the full message history, including every model turn and every shell command that the agent executed.
