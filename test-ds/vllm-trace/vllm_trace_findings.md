# vLLM tracing experiment findings

## Outcome

- mini-swe-agent **does support** a local vLLM OpenAI-compatible server for this workflow.
- The chosen vLLM setup works with:
  - model: `/mnt/disk1/yiliu7/Qwen/Qwen3.6-27B`
  - GPUs: `4,5,6,7`
  - parser flags: `--reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder`
- Built-in vLLM logging is already enough to capture the **request side** of each round:
  - rendered prompt text
  - prompt token ids
  - sampling params
  - request id
- The saved mini-swe-agent trajectory is enough to capture the **response side**:
  - assistant reasoning text
  - tool call payload
  - tool call id
  - provider usage block
  - tool output that would be consumed by the next round

## Root cause hit during setup

The `omni` env initially failed before vLLM startup:

- `torch 2.11.0+cu130` expected `ncclDevCommDestroy`
- the env was loading `/home/yiliu7/workspace/venvs/omni/lib/python3.12/site-packages/nvidia/nccl/lib/libnccl.so.2`
- that file was NCCL `2.27.5+cuda12.9`, which does **not** export `ncclDevCommDestroy`

Working workaround:

```bash
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libnccl.so.2.29.7
```

That system NCCL exports the missing symbol and allows `torch` + `vllm` to start cleanly in the existing env.

## Working server launch

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 \
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libnccl.so.2.29.7 \
VLLM_LOGGING_LEVEL=DEBUG \
/home/yiliu7/workspace/venvs/omni/bin/vllm serve /mnt/disk1/yiliu7/Qwen/Qwen3.6-27B \
  --served-model-name qwen3.6-27b-local \
  --port 8000 \
  --tensor-parallel-size 4 \
  --max-model-len 262144 \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --enable-log-requests \
  --enable-request-id-headers \
  --max-log-len 40000
```

## Server state during capture

- listener PID: `3168022`
- port: `8000`
- log file: `test-ds/vllm-trace/vllm-qwen36-27b.log`

The server was stopped after the capture so GPUs `4,5,6,7` were released.

## Captured artifacts

- mini smoke trajectory:
  - `test-ds/vllm-trace/mini-vllm-smoke.traj.json`
- SWE-bench one-round trajectory:
  - `test-ds/vllm-trace/sqlfluff-vllm-step1.traj.json`
- vLLM server log:
  - `test-ds/vllm-trace/vllm-qwen36-27b.log`

## Reproduce commands

### 1. Launch vLLM with request-side tracing

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 \
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libnccl.so.2.29.7 \
VLLM_LOGGING_LEVEL=DEBUG \
/home/yiliu7/workspace/venvs/omni/bin/vllm serve /mnt/disk1/yiliu7/Qwen/Qwen3.6-27B \
  --served-model-name qwen3.6-27b-local \
  --port 8000 \
  --tensor-parallel-size 4 \
  --max-model-len 262144 \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --enable-log-requests \
  --enable-request-id-headers \
  --max-log-len 40000
```

### 2. Launch vLLM with both request-side and output-side tracing

This is the same launch, but adds generated output logging:

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 \
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libnccl.so.2.29.7 \
VLLM_LOGGING_LEVEL=DEBUG \
/home/yiliu7/workspace/venvs/omni/bin/vllm serve /mnt/disk1/yiliu7/Qwen/Qwen3.6-27B \
  --served-model-name qwen3.6-27b-local \
  --port 8000 \
  --tensor-parallel-size 4 \
  --max-model-len 262144 \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --enable-log-requests \
  --enable-log-outputs \
  --enable-request-id-headers \
  --max-log-len 40000
```

### 3. One-round mini smoke run

```bash
cd /home/yiliu7/workspace/mini-swe-agent

MSWEA_CONFIGURED=true uv run mini \
  -c mini.yaml \
  --agent-class default \
  -m hosted_vllm/qwen3.6-27b-local \
  -t 'Use the bash tool exactly once to run pwd, then stop.' \
  -c model.model_kwargs.api_base=http://127.0.0.1:8000/v1 \
  -c model.model_kwargs.api_key=dummy \
  -c model.model_kwargs.temperature=0 \
  -c model.cost_tracking=ignore_errors \
  -c agent.step_limit=1 \
  -c environment.cwd=/home/yiliu7/workspace/mini-swe-agent \
  -o test-ds/vllm-trace/mini-vllm-smoke.traj.json
```

### 4. One-round SWE-bench single capture

```bash
cd /home/yiliu7/workspace/mini-swe-agent

WORK=/tmp/mini-swe-agent-swebench/sqlfluff__sqlfluff-1625
TESTBED="$WORK/testbed"
VENV="$WORK/venv"

MSWEA_CONFIGURED=true uv run mini-extra swebench-single \
  --subset lite \
  --split dev \
  -i sqlfluff__sqlfluff-1625 \
  --agent-class default \
  --exit-immediately \
  -m hosted_vllm/qwen3.6-27b-local \
  -c swebench.yaml \
  -c swebench_local_debug.yaml \
  -c agent.step_limit=1 \
  -c model.model_kwargs.api_base=http://127.0.0.1:8000/v1 \
  -c model.model_kwargs.api_key=dummy \
  -c model.model_kwargs.temperature=0 \
  -c model.cost_tracking=ignore_errors \
  -c environment.cwd="$TESTBED" \
  -c environment.timeout=120 \
  -c environment.env.PATH="$VENV/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
  -c environment.env.VIRTUAL_ENV="$VENV" \
  -o test-ds/vllm-trace/sqlfluff-vllm-step1.traj.json
```

### 5. Match request ids between the trajectory and the vLLM log

```bash
rg 'chatcmpl-' test-ds/vllm-trace/sqlfluff-vllm-step1.traj.json
rg 'Request chatcmpl-|Generated response chatcmpl-' test-ds/vllm-trace/vllm-qwen36-27b.log
```

## What the benchmark capture shows

### Request-side evidence from vLLM

For request id `chatcmpl-bfa963fe26f1db2c` in `sqlfluff-vllm-step1.traj.json`, the vLLM log contains:

- the full rendered prompt
- the injected bash tool schema
- the full benchmark PR description / instruction block
- `prompt_token_ids`
- sampling params

Matching log lines:

- request prompt/details: `vllm-qwen36-27b.log:4814`
- request params: `vllm-qwen36-27b.log:4815`

### Response-side evidence from the saved trajectory

The same request id appears in the trajectory at:

- response id: `sqlfluff-vllm-step1.traj.json:89`

The trajectory also preserves:

- tool call:
  - `find /testbed -type f -name "*.py" | xargs grep -l "L031" 2>/dev/null | head -20`
- tool call id:
  - `chatcmpl-tool-afecd2cfbbd53f63`
- provider usage:
  - `prompt_tokens: 1476`
  - `completion_tokens: 204`
- tool output:
  - `find: ‘/testbed’: No such file or directory`

This means the combined view is already strong:

1. **vLLM log** tells you what was sent/consumed on the request side.
2. **trajectory** tells you what tool call came back and what tool output will feed the next round.

## Important limitation

With `--enable-log-requests`, vLLM logs the **rendered prompt**, not the raw incoming OpenAI JSON body.

That is still useful for:

- understanding what text the model actually saw
- inspecting prompt token ids
- checking tool schema injection

But it is not the same as storing the original `messages=[...]` request body as structured JSON.

If you need raw JSON bodies next, the smallest next step is to add either:

- a tiny vLLM middleware via `--middleware`, or
- a local proxy in front of vLLM

## How to dump input and output on the vLLM side

### Stock vLLM: request-side dump

The current stock setup already supports:

- rendered prompt text
- prompt token ids
- sampling params
- request ids

This comes from `vllm.entrypoints.logger.RequestLogger.log_inputs()`, which logs:

- `Request %s details: prompt: %r, prompt_token_ids: %s`
- `Received request %s: params: %s`

You only get the full prompt text if:

- `--enable-log-requests` is enabled
- `VLLM_LOGGING_LEVEL=DEBUG`

### Stock vLLM: output-side dump

vLLM can also log generated output text and generated token ids, but only if:

- `--enable-log-requests`
- `--enable-log-outputs`

The implementation is in `vllm.entrypoints.logger.RequestLogger.log_outputs()`, which logs:

- `Generated response ...`
- `output: ...`
- `output_token_ids: ...`
- `finish_reason: ...`

For chat completions, the call sites are in:

- `vllm/entrypoints/openai/chat_completion/serving.py`
  - streaming delta logging around line `1047`
  - complete streaming response logging around line `1247`

### Stock vLLM: rendered prompt without generation

vLLM exposes a render route:

- `/v1/chat/completions/render`

The render server path builds a `GenerateRequest` with:

- `token_ids`
- multimodal features
- sampling params

So if you want to inspect the final tokenized prompt before generation, this route is useful.

### What stock vLLM still does not dump cleanly

Stock logging does **not** cleanly persist the raw incoming OpenAI JSON body, meaning:

- original `messages=[...]`
- original `tools=[...]`
- raw per-request kwargs as JSON

To get that exact wire payload, use:

1. a tiny ASGI middleware via `--middleware`, or
2. a local proxy in front of vLLM

That is the right next step if you want exact raw request/response bodies rather than the rendered prompt view.

## How tool calls are generated in this setup

### 1. mini-swe-agent always sends tool definitions

On the mini-swe-agent side, `LitellmModel._query()` calls:

- `litellm.completion(model=..., messages=..., tools=[BASH_TOOL], ...)`

So every round includes the bash tool schema.

### 2. vLLM injects those tools into the chat template

On the vLLM side, `OpenAIServingRender.render_chat()` converts `request.tools` into `tool_dicts`, then passes them into:

- `preprocess_chat(..., tool_dicts=tool_dicts, ...)`

Inside `preprocess_chat()`, those tools are merged into the chat-template kwargs:

- `tools=tool_dicts`

Then the full message list is rendered by the renderer.

### 3. The chat template tells Qwen how to emit a tool call

For this Qwen3Coder setup, the rendered prompt includes:

- a `# Tools` section
- the serialized tool schema
- explicit XML instructions like:
  - `<tool_call>`
  - `<function=...>`
  - `<parameter=...>`

This is visible directly in the request-side vLLM log.

### 4. The model emits XML-like tool-call text

The model does not emit OpenAI JSON directly. It emits tool calls in the Qwen3Coder XML-ish format.

### 5. vLLM parses that model text back into OpenAI-style tool calls

`vllm/tool_parsers/qwen3coder_tool_parser.py`:

- detects `<tool_call>` blocks
- extracts `<function=...>` sections
- extracts `<parameter=...>` values
- converts parameter types using the tool schema
- returns `ToolCall(type="function", function=FunctionCall(...))`

The main entry point is:

- `Qwen3CoderToolParser.extract_tool_calls()`

So the OpenAI-style `tool_calls` in the response are created by the vLLM tool parser, not emitted as native JSON by the model.

## How each turn context is concatenated

### 1. mini-swe-agent resends the full message history each round

`DefaultAgent.query()` calls:

- `self.model.query(self.messages)`

And `LitellmModel.query()` forwards the prepared `messages` list into:

- `litellm.completion(..., messages=messages, tools=[BASH_TOOL], ...)`

So each round sends the full conversation so far, not just the newest message.

### 2. vLLM receives the full `request.messages`

Inside `OpenAIServingRender.render_chat()`, vLLM passes:

- `request.messages`

into `preprocess_chat()`.

### 3. The renderer converts all messages into one rendered prompt

In `preprocess_chat()`:

- `renderer.render_chat_async([messages], chat_params, ...)`

In `HfRenderer.render_messages()`:

1. `parse_chat_messages(...)` converts the OpenAI messages into a conversation structure
2. `safe_apply_chat_template(...)` applies the model chat template to the full conversation
3. the result becomes a single prompt string / token sequence

So the “context concat” step is: **all prior turns are re-rendered into one chat-template prompt every round**.

### 4. The request-side log shows the final concatenated prompt

That is why the vLLM request log is useful: it shows the post-template prompt that the model actually consumed, including:

- system message
- user message
- tool instructions
- previous assistant/tool turns on later rounds

## Current conclusion

For your current goal, the best split is:

- **trajectory** for returned tool calls and tool outputs
- **vLLM log** for rendered per-turn prompt consumption

If you want deeper vLLM-side visibility next:

1. relaunch with `--enable-log-outputs`
2. use `/v1/chat/completions/render` for prompt/token-id inspection
3. add middleware/proxy only if you need exact raw JSON bodies

## Status commands

### Check whether the server is running

```bash
ss -ltnp '( sport = :8000 )' && tail -n 40 test-ds/vllm-trace/vllm-qwen36-27b.log
```

### Inspect the captured trajectory

```bash
cd /home/yiliu7/workspace/mini-swe-agent && uv run mini-extra inspect test-ds/vllm-trace/sqlfluff-vllm-step1.traj.json
```

## Practical conclusion

For the question you asked — tool call plus which inputs are consumed each round — the current setup is already enough for a first pass:

- use the vLLM log for the consumed prompt
- use the trajectory for the returned tool call and tool output

If you want the same visibility for **later rounds**, rerun with `agent.step_limit=2` or `3` and compare each request id in the log with the matching assistant/tool messages in the saved trajectory.
