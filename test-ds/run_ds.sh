export DEEPSEEK_API_KEY=""

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
