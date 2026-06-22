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
    max_tokens: 65536                # ~64K for thinking + action; 32K is too tight
    temperature: 0.0
    timeout: 900000                  # 15 min per response
```

### Why each setting matters

| Setting | Why |
|---------|-----|
| `cost_tracking: "ignore_errors"` | `hosted_vllm/` models aren't in LiteLLM's cost registry. Without this, `RuntimeError`. |
| `format_error_template` (custom) | Default template treats `finish_reason=tool_calls` as truncation. With `tool_choice: "required"`, every response ends that way — so only `length` is a real error. |
| `tool_choice: "required"` | **Critical**. Qwen3 is a thinking model — it reasons indefinitely unless forced to call a tool. Without this it produces 0 actions. |
| `max_tokens: 65536` | Sweet spot. 32K is too tight (thinking burns 20–30K before the tool call). Much higher causes first-response timeouts. |
| `api_base` | Points at the local vLLM server. No API key needed. |

## 3. Run

```bash
mini-extra swebench \
  -c swebench.yaml \
  -c swebench_vllm.yaml \
  --subset lite \
  --split dev \
  --slice "0:1" \
  --agent.mode=yolo \
  -w 1 \
  -o ./output/qwen3_vllm_$(date +%Y%m%d_%H%M%S)
```

For the full sweepbench dataset:
```bash
mini-extra swebench \
  -c swebench.yaml -c swebench_vllm.yaml \
  --subset lite --agent.mode=yolo -w 1 \
  -o ./output/qwen3_lite_full_$(date +%Y%m%d_%H%M%S)
```

## 4. Key Knobs

| Knob | Effect |
|------|--------|
| `max_tokens` ↑ | More room for thinking → fewer truncations, but longer per-step latency. 49152–65536 is the usable range. |
| `tool_choice` | `"required"` = forces tool every turn (works with Qwen3). `"auto"` = model decides (Qwen3 won't use tools). |
| `--reasoning-parser qwen3` | **On**: thinking → `reasoning_content`, clean tool calls in `content`. **Off**: thinking + tool call merged in `content`, model times out on first response. |
| `step_limit` | Default 250. Qwen3 needs it — it does ~150+ steps per instance. |
| `temperature: 0.0` | Deterministic. Works fine. |

## 5. Known Issues

- **Thinking model inconsistency**: Qwen3's chain-of-thought is unbounded. Some runs hit 148 steps with 0 errors, others fail at step 8 because thinking consumed the full `max_tokens` budget and the response arrived empty or truncated. This is inherent to thinking models — the agent loop can't control how much the model thinks.
- **First-response timeout**: Without `--reasoning-parser qwen3`, the first response can run 10+ minutes because thinking + code block are serialized in `content`. Always use the reasoning parser.
- **No cost tracking**: `cost_tracking: "ignore_errors"` means zero cost shown. For pass/fail ratio, submit `preds.json` to the SWE-bench evaluation harness (`sb-cli submit` or `swebench.harness.run_evaluation`).

## 6. Quick Reference

```bash
# Server
CUDA_VISIBLE_DEVICES=<N> vllm serve /path/to/model \
  --max-model-len 262144 --reasoning-parser qwen3 \
  --enable-auto-tool-choice --tool-call-parser hermes --port 8000

# Run
mini-extra swebench \
  -c swebench.yaml -c swebench_vllm.yaml \
  --subset lite --slice "0:2" --agent.mode=yolo -w 1

# Evaluate (after run)
sb-cli submit swe-bench_lite test \
  --predictions_path output/<run>/preds.json \
  --run_id qwen3-smoke
```
