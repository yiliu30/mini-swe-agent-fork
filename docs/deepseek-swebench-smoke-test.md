# DeepSeek + SWE-bench Smoke Test

This document describes how to set up and run a minimal SWE-bench smoke test using DeepSeek's `deepseek-v4-flash` model via the Anthropic-compatible API.

## Architecture

```
mini-extra swebench CLI
  -> swebench.py (batch runner)
    -> loads SWE-bench lite/dev dataset, sliced to 2 instances
    -> for each instance:
      -> LitellmModel (model="deepseek/deepseek-v4-flash")
        -> litellm.completion() -> https://api.deepseek.com/v1/chat/completions
      -> Docker environment (SWE-bench image)
      -> agent.run() -> model queries + bash tool calls
  -> preds.json (patches)
     |
     v
  SWE-bench evaluation harness
    -> applies patches, runs test suite
    -> reports pass/fail ratio per instance
```

The key integration point: LiteLLM's DeepSeek provider directed to the standard DeepSeek API via `DEEPSEEK_API_KEY`. (The Anthropic-compatible endpoint was not viable because it doesn't support custom tool definitions.)

## Prerequisites

- Docker installed and running
- mini-swe-agent >= 2.4.2 installed (`uv pip install -e .`)
- LiteLLM >= 1.75.5 installed
- DeepSeek API key (from `~/.claude/settings.json`)

## Configuration Gotchas

Two issues must be handled explicitly:

1. **Cost tracking**: `deepseek-v4-flash` is not in LiteLLM's model cost registry. The `_calculate_cost()` method raises `RuntimeError` unless `cost_tracking: "ignore_errors"` is set.

2. **Tool support**: DeepSeek's **Anthropic-compatible** endpoint (`api.deepseek.com/anthropic`) does **not** support custom tool definitions — it only supports Anthropic's built-in web search tools. You must use the standard DeepSeek API (`api.deepseek.com`) via the `deepseek/` LiteLLM provider for tool-calling tasks like SWE-bench.

## Step-by-Step Setup

### 1. API Connectivity Test

First, verify the model works with tool calling:

```bash
# Test via LiteLLM with tool calling
DEEPSEEK_API_KEY="$DEEPSEEK_API_KEY" \
python3 -c "
import litellm
r = litellm.completion(
    model='deepseek/deepseek-v4-flash',
    messages=[{'role': 'user', 'content': 'List files in current directory'}],
    max_tokens=100,
    drop_params=True,
    tools=[{
        'type': 'function',
        'function': {
            'name': 'bash',
            'description': 'Execute a bash command',
            'parameters': {
                'type': 'object',
                'properties': {
                    'command': {'type': 'string', 'description': 'bash command'}
                },
                'required': ['command']
            }
        }
    }],
)
msg = r.choices[0].message
print('Content:', msg.content)
print('Tool calls:', msg.tool_calls)
"
```

### 2. Create DeepSeek Benchmark Config

Create `src/minisweagent/config/benchmarks/swebench_deepseek.yaml`:

```yaml
# DeepSeek v4 Flash model overrides for SWE-bench.
# Merge on top of swebench.yaml via: -c swebench.yaml -c swebench_deepseek.yaml
model:
  model_name: "deepseek/deepseek-v4-flash"
  cost_tracking: "ignore_errors"   # Model not in LiteLLM cost registry
  model_kwargs:
    drop_params: true
    parallel_tool_calls: true
```

### 3. Create Environment Setup Script

Create `scripts/setup_deepseek_env.sh`:

```bash
#!/usr/bin/env bash
# Source before running mini-extra swebench with DeepSeek
# NOTE: Uses standard DeepSeek API (OpenAI-compatible), NOT the
# Anthropic-compatible endpoint which doesn't support custom tool definitions.
export DEEPSEEK_API_KEY="$DEEPSEEK_API_KEY"
echo "DeepSeek environment configured for mini-swe-agent"
```

### 4. Pre-Pull Docker Image

The first 2 instances in `lite/dev` are both from `sqlfluff` — they share one Docker image:

```bash
docker pull docker.io/swebench/sweb.eval.x86_64.sqlfluff_1776_sqlfluff-1625:latest
```

### 5. Run the Smoke Test

```bash
source scripts/setup_deepseek_env.sh

mini-extra swebench \
  --subset lite \
  --split dev \
  --slice "0:2" \
  -c swebench.yaml \
  -c swebench_deepseek.yaml \
  -c agent.mode=yolo \
  -c agent.cost_limit=3.0 \
  -w 1 \
  -o ./output/smoke_test_deepseek_$(date +%Y%m%d_%H%M%S)
```

**Selected instances** (lite/dev indices 0–1):
- `sqlfluff__sqlfluff-1625`
- `sqlfluff__sqlfluff-2419`

**Flag breakdown:**
| Flag | Purpose |
|------|---------|
| `--subset lite --split dev --slice "0:2"` | Exactly 2 instances, both from sqlfluff |
| `-c swebench.yaml` | Base config (agent templates, env, step limits) |
| `-c swebench_deepseek.yaml` | DeepSeek model overrides |
| `-c agent.mode=yolo` | Skip confirmation prompts (batch mode) |
| `-c agent.cost_limit=3.0` | $3 cost ceiling per instance |
| `-w 1` | Single worker, sequential execution |
| `-o ./output/...` | Timestamped output directory |

### 6. Verify Results


```bash
# Check predictions
python3 -c "
import json
preds = json.load(open('output/smoke_test_deepseek_*/preds.json'))
for iid, data in preds.items():
    patch_len = len(data.get('model_patch', ''))
    print(f'{iid}: model={data[\"model_name_or_path\"]}, patch_len={patch_len} chars')
"

# Check trajectory has model interaction
python3 -c "
import json
traj = json.load(open('output/smoke_test_deepseek_*/sqlfluff__sqlfluff-1625/sqlfluff__sqlfluff-1625.traj.json'))
msgs = [m for m in traj.get('messages', []) if m.get('role') == 'assistant']
has_tools = any(m.get('extra', {}).get('actions') for m in msgs)
print(f'Assistant messages: {len(msgs)}, has tool calls: {has_tools}')
"
```

**Success criteria:**
- Both instances have entries in `preds.json` with non-empty `model_patch`
- Trajectories show assistant messages with tool calls (proves the model connected)
- No uncaught exceptions in `exit_statuses_*.yaml`

### 7. Get the Pass Ratio (Evaluation)

Step 5 only generates patches — it doesn't tell you if they actually **resolve** the issues. To get a pass/fail ratio, submit `preds.json` to the SWE-bench evaluation harness:

```bash
# Option 1: Cloud-based (free, ~20 min flat)
# Install: pip install sb-cli
sb-cli submit swe-bench_lite test \
  --predictions_path output/smoke_test_deepseek_*/preds.json \
  --run_id deepseek-smoke-test
```

```bash
# Option 2: Local evaluation (requires swebench installed)
python -m swebench.harness.run_evaluation \
  --dataset_name princeton-nlp/SWE-Bench_Lite \
  --predictions_path output/smoke_test_deepseek_*/preds.jsonl \
  --max_workers 2 \
  --run_id deepseek-smoke-test
```

> **Note:** If you only want a quick manual check, you can verify patches yourself by applying them to the instance's Docker image and running the relevant test suite, as demonstrated in the verification section below.

**Sample output** from cloud evaluation:
```
✓ sqlfluff__sqlfluff-1625: resolved
✓ sqlfluff__sqlfluff-2419: resolved
─────────────────────────────────
  Resolved: 2/2 (100.0%)
```

## Output Structure

```
output/smoke_test_deepseek_<timestamp>/
  preds.json                              # Aggregate predictions
  exit_statuses_<ts>.yaml                 # Per-instance exit codes
  minisweagent.log                        # Full log
  sqlfluff__sqlfluff-1625/
    sqlfluff__sqlfluff-1625.traj.json     # Full trajectory
  sqlfluff__sqlfluff-2419/
    sqlfluff__sqlfluff-2419.traj.json     # Full trajectory
```

## Fallback Strategies

### A: Different model name

If `deepseek-v4-flash` is not accepted, try the generic DeepSeek chat model:

```bash
DEEPSEEK_API_KEY="$DEEPSEEK_API_KEY" \
mini-extra swebench \
  --subset lite --split dev --slice "0:2" \
  -c swebench.yaml \
  -c model.model_name=deepseek/deepseek-chat \
  -c model.cost_tracking=ignore_errors \
  -c agent.mode=yolo \
  -w 1 -o ./output/smoke_test_deepseek_std_$(date +%Y%m%d_%H%M%S)
```

### B: Dummy dataset (fastest)

If Docker image pull is too slow:

```bash
DEEPSEEK_API_KEY="$DEEPSEEK_API_KEY" \
mini-extra swebench \
  --subset _test --split test --slice "0:2" \
  -c swebench.yaml -c swebench_deepseek.yaml \
  -c agent.mode=yolo -w 1 \
  -o ./output/smoke_test_deepseek_test_$(date +%Y%m%d_%H%M%S)
```

## Time Estimate

| Step | Time |
|------|------|
| API connectivity test | ~1 min |
| Config creation | ~2 min |
| Docker image pull | ~5–15 min |
| Smoke test run (2 instances) | ~10–20 min |
| Verification | ~1 min |
| **Total** | **~20–40 min** |

## Files to Create

| File | Purpose |
|------|---------|
| `src/minisweagent/config/benchmarks/swebench_deepseek.yaml` | DeepSeek model overrides |
| `scripts/setup_deepseek_env.sh` | Environment variable setup |

No existing files need modification.
