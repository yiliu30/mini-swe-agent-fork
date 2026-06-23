# Local Qwen3 via vLLM — Best Known Configuration

Running SWE-bench with a local Qwen3 model served by vLLM.

## 1. Start the vLLM Server

```bash
vllm serve /storage/yiliu7/Qwen/Qwen3.6-35B-A3B-FP8 \
  --max-model-len 262144 \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --port 8000
```

| Flag | Why |
|------|-----|
| `--reasoning-parser qwen3` | Extracts reasoning into `reasoning_content` field. Without this, Qwen3 dumps thinking into `content` and never produces tool calls. |
| `--enable-auto-tool-choice` | Required for `tool_choice: "required"` to work. |
| `--tool-call-parser hermes` | Parses JSON tool calls. Qwen3 needs the `hermes` parser (not `functionary` or `mistral`). |
| `--max-model-len 262144` | Full context window. The model won't hit this — output token cap happens first. |

**GPU**: Pick a free GPU with `CUDA_VISIBLE_DEVICES=<N>` if needed. Model needs ~170 GB VRAM.

## 2. Config File

`src/minisweagent/config/benchmarks/swebench_vllm.yaml`:

```yaml
# Merge on top of swebench.yaml: -c swebench.yaml -c swebench_vllm.yaml
model:
  model_name: "hosted_vllm//storage/yiliu7/Qwen/Qwen3.6-35B-A3B-FP8"
  cost_tracking: "ignore_errors"

  # Override: only flag length truncation, not tool_calls (normal when tool_choice=required)
  format_error_template: |
    {% if finish_reason is defined and finish_reason == "length" -%}
    Your previous response reached the output token limit before you produced a
    tool call, so it was cut off. Respond more concisely and finish with exactly
    one bash tool call. If you need to think more, do so briefly.
    {%- else -%}
    Tool call error:

    <error>
    {{error}}
    </error>

    Here is general guidance on how to submit correct toolcalls:

    Every response needs to use the 'bash' tool at least once to execute commands.

    Call the bash tool with your command as the argument:
    - Tool: bash
    - Arguments: {"command": "your_command_here"}

    If you have completed your assignment, please consult the first message about
    how to submit your solution (you will not be able to continue working on this
    task after that).
    {%- endif %}

  model_kwargs:
    api_base: "http://localhost:8000/v1"
    tool_choice: "required"          # Qwen3 won't use tools unless forced
    drop_params: true
    max_tokens: 49152                # verified: 49K fits under 262K cap when context grows to ~210K tokens
    temperature: 0.0
    timeout: 900000                  # 15 min per response
```

### Why each setting matters

| Setting | Why |
|---------|-----|
| `cost_tracking: "ignore_errors"` | `hosted_vllm/` models aren't in LiteLLM's cost registry. Without this, `RuntimeError`. |
| `format_error_template` (custom) | Default template treats `finish_reason=tool_calls` as truncation. With `tool_choice: "required"`, every response ends that way — so only `length` is a real error. |
| `tool_choice: "required"` | **Critical**. Qwen3 is a thinking model — it reasons indefinitely unless forced to call a tool. Without this it produces 0 actions. |
| `max_tokens: 49152` | Verified instances need this — context grows to ~210K after 150+ steps, leaving 262K−210K=52K. 49K fits. For Lite, 65536 works. |
| `api_base` | Points at the local vLLM server. No API key needed. |

## 3. Run

**SWE-bench Verified (test split):**
```bash
mini-extra swebench \
  -c swebench.yaml -c swebench_vllm.yaml \
  --subset verified --split test \
  --agent.mode=yolo -w 1 \
  -o ./output/qwen3_verified_$(date +%Y%m%d_%H%M%S)
```

**SWE-bench Lite (dev split):**
```bash
mini-extra swebench \
  -c swebench.yaml -c swebench_vllm.yaml \
  --subset lite --split dev \
  --agent.mode=yolo -w 1 \
  -o ./output/qwen3_lite_$(date +%Y%m%d_%H%M%S)
```

Use `--slice "0:N"` to limit instances for smoke tests. Use `--redo-existing` to re-run already-completed instances.

## 4. Evaluate

After the run, submit `preds.json` to the SWE-bench evaluation harness:

```bash
# Local evaluation (uses Docker containers, runs test suite per instance)
python -m swebench.harness.run_evaluation \
  --dataset_name princeton-nlp/SWE-Bench_Verified \
  --predictions_path output/<run>/preds.jsonl \
  --split test --max_workers 2 \
  --run_id qwen3-run
```

Convert `preds.json` to JSONL first:
```bash
python3 -c "
import json
preds = json.load(open('output/<run>/preds.json'))
with open('output/<run>/preds.jsonl', 'w') as f:
    for iid, d in preds.items():
        if d.get('model_patch'):
            f.write(json.dumps(d) + '\n')
"
```

## 5. Benchmark Results (2026-06-22)

Tested on SWE-bench Verified, 10 instances (astropy repo). Full output at `output/qwen3_verified_10_run2_20260622_151346/`.

| Metric | Value |
|--------|-------|
| Instances run | 10 |
| Patches submitted | 4 (40%) |
| Resolved (passes all tests) | 2 (20% overall, 50% of submitted) |
| Total wall time | 41 min |
| Avg time per instance | ~4 min |

**Resolved instances:**
- `astropy__astropy-13453` — 66 actions, 1995-char patch
- `astropy__astropy-14309` — 1619-char patch

**Unresolved instances (patches submitted but tests failed):**
- `astropy__astropy-13398` — 83 actions, 4126-char patch
- `astropy__astropy-13977` — 54 actions, 927-char patch

**Failures without patches:**

| Exit status | Count | Why |
|-------------|-------|-----|
| `RepeatedFormatError` | 3 | Thinking model produced 0 tool calls — format error → retry loop → exhausted |
| `LimitsExceeded` | 2 | 250 step limit hit before completing (13236: 250 actions, 14182: 250 actions) |
| `ContextWindowExceeded` | 1 | Conversation grew past 262K tokens (12907: 168 actions before overflow) |

## 6. Key Knobs

| Knob | Effect |
|------|--------|
| `max_tokens` ↑ | More room for thinking → fewer truncations, but longer per-step latency. 49152–65536 is the usable range. |
| `tool_choice` | `"required"` = forces tool every turn (works with Qwen3). `"auto"` = model decides (Qwen3 won't use tools). |
| `--reasoning-parser qwen3` | **On**: thinking → `reasoning_content`, clean tool calls in `content`. **Off**: thinking + tool call merged in `content`, model times out on first response. |
| `step_limit` | Default 250. Qwen3 needs it — it does ~150+ steps per instance. |
| `temperature: 0.0` | Deterministic. Works fine. |

## 7. Known Issues

- **Thinking model inconsistency**: Qwen3's chain-of-thought is unbounded. Some runs hit 148 steps with 0 errors, others fail at step 8 because thinking consumed the full `max_tokens` budget and the response arrived empty or truncated. In the 10-instance Verified test, 3 instances failed this way (30%).
- **Context window overflow**: After ~150-250 steps, accumulated messages + tool outputs can push the input past 210K tokens. With `max_tokens: 49152`, this breaches the 262K cap → `ContextWindowExceededError`. Affected 1/10 instances in testing. There is no fix — the model's context window is fixed.
- **First-response timeout**: Without `--reasoning-parser qwen3`, the first response can run 10+ minutes because thinking + code block are serialized in `content`. Always use the reasoning parser.
- **Step limit**: Default 250 may not be enough. 2/10 instances exhausted steps without submitting. Bumping to 300+ trades wall-clock time for coverage.
- **No cost tracking**: `cost_tracking: "ignore_errors"` means zero cost shown.

## 8. Quick Reference

```bash
# Server
CUDA_VISIBLE_DEVICES=<N> vllm serve /path/to/model \
  --max-model-len 262144 --reasoning-parser qwen3 \
  --enable-auto-tool-choice --tool-call-parser hermes --port 8000

# Run (Verified)
mini-extra swebench \
  -c swebench.yaml -c swebench_vllm.yaml \
  --subset verified --split test \
  --agent.mode=yolo -w 1 \
  -o ./output/qwen3_verified_$(date +%Y%m%d_%H%M%S)

# Evaluate (after run)
python -m swebench.harness.run_evaluation \
  --dataset_name princeton-nlp/SWE-Bench_Verified \
  --predictions_path output/<run>/preds.jsonl \
  --split test --max_workers 2 \
  --run_id qwen3-run
```
