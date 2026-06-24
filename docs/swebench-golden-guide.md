# SWE-bench Golden Guide (Model-Agnostic)

Concise best practices distilled from running 70+ SWE-bench Verified instances across Qwen3 (35B, 27B) and DeepSeek V4 Flash.

## 1. Sampling Parameters

```yaml
temperature: 1.0    # Not 0.0. Higher diversity avoids deterministic failure loops
top_p: 0.95         # Standard. Works across models
```

At temp=0.0, thinking models get stuck in identical reasoning chains. At temp=1.0, stochastic sampling occasionally breaks the loop. This alone was worth 30 points (20% → 50%) on Qwen3.

## 2. Tool-Calling Parser

Use the model's **native** tool-call parser. Never fall back to `hermes` unless there's no alternative.

| Model | Parser | Notes |
|-------|--------|-------|
| DeepSeek V4 | `deepseek_v4` | Native. Zero format errors. |
| Qwen3 coder | `qwen3_coder` | Better than hermes, but still has thinking explosions |
| Qwen3 general | `hermes` | Works but highest format error rate |

```bash
--tool-call-parser deepseek_v4   # or qwen3_coder, or hermes
--enable-auto-tool-choice
```

Native parsers eliminate format errors. The wrong parser causes 30-60% of API calls to be wasted on unparseable output.

## 3. Thinking Models

Thinking models (Qwen3, DeepSeek R1) have **unbounded chain-of-thought**. Their reasoning can consume the entire `max_tokens` budget before producing a tool call.

| Strategy | Effect |
|----------|--------|
| Use a non-thinking model | Simplest fix. DeepSeek V4 Flash (no think) = 100% submission rate |
| Enable native thinking parser | `--reasoning-parser deepseek_v4` (server-side) |
| `preserve_thinking: true` | Reduces actions by 30-50% but can cause context bloat |
| Increase `max_tokens` | 49K–65K range. Higher = more room for thinking, but more waste |

For DeepSeek V4 thinking: `extra_body: {chat_template_kwargs: {thinking: true, reasoning_effort: high}}` (NOT `thinking: {type: enabled}` — that's silently ignored).

## 4. Token Budget

```yaml
max_tokens: 49152    # Sweet spot for 262K context. Enough for thinking + tool call
```

With 262K context and ~200K input after 150+ steps: 262K - 200K = 62K headroom. 49K output leaves 13K margin. Too low (32K) truncates tool calls. Too high (65K) lets thinking explode.

## 5. Context Window

```bash
--max-model-len 131072    # Cap at 128K. Defaults (1M for DeepSeek) OOM
```

Only increase if you have GPU memory headroom. KV cache is the bottleneck, not model weights.

## 6. GPU Configuration

| Scenario | Recommendation |
|----------|---------------|
| Small model (<40B) | 1 GPU |
| Medium model (40-100B) | 2 GPUs, TP2 |
| Large model (100B+) | 4+ GPUs |
| MoE models | Fewer GPUs needed (sparse activation) |

Check with `nvidia-smi`. GPU memory usage = model weights + KV cache + CUDA overhead. Leave 10-15% headroom.

## 7. Environment

```bash
# Always set CUDA_HOME to match the venv's CUDA toolkit
CUDA_HOME=/usr/local/cuda-13

# Use the right venv. vLLM 0.23+ for DeepSeek V4, 0.22 for Qwen3
/home/yiliu7/workspace/venvs/vllm/bin/vllm serve ...
```

## 8. Config Template (mini-swe-agent)

```yaml
model:
  model_name: "hosted_vllm//path/to/model"
  cost_tracking: "ignore_errors"
  model_kwargs:
    api_base: "http://localhost:8000/v1"
    max_tokens: 49152
    temperature: 1.0
    top_p: 0.95
    timeout: 900000
```

Add `extra_body` for thinking models, `tool_choice` for models that won't call tools without forcing.

## 9. Running

```bash
# 10-instance smoke test
mini-extra swebench \
  --subset verified --split test --slice "0:10" \
  -c swebench.yaml -c swebench_my_model.yaml \
  -c agent.mode=yolo -w 1 \
  -o ./output/$(date +%Y%m%d_%H%M%S)

# Full 500-instance run (resume-safe)
mini-extra swebench \
  --subset verified --split test \
  -c swebench.yaml -c swebench_my_model.yaml \
  -c agent.mode=yolo -w 5 \
  -o ./output/full_run/
```

`-w 5` parallel workers work with vLLM. Resume by re-running with same `-o` dir.

## 10. Evaluation

```bash
python -c "
import json
preds = json.load(open('output/<run>/preds.json'))
with open('output/<run>/preds.jsonl','w') as f:
    for i,d in preds.items():
        if d.get('model_patch'): f.write(json.dumps(d)+'\n')
"

python -m swebench.harness.run_evaluation \
  --dataset_name princeton-nlp/SWE-Bench_Verified \
  --predictions_path output/<run>/preds.jsonl \
  --split test --max_workers 4 --run_id my-run
```

Clean Docker conflicts: `docker rm -f sweb.eval.*`

## 11. Quick Diagnostics

```bash
# Server health
curl http://localhost:8000/v1/models

# Generation speed
tail -f /tmp/vllm_*.log | grep "generation throughput"

# Run progress
python3 -c "
import json, glob
trajs = glob.glob('output/*/*/*.traj.json')
print(f'{len(trajs)}/10 done')
"

# Per-instance token usage
python3 -c "
import json
t = json.load(open('output/<run>/<iid>/<iid>.traj.json'))
api = t['info']['model_stats']['api_calls']
tokens = sum(m['extra']['response']['usage']['total_tokens'] 
             for m in t['messages'] if m.get('role')=='assistant')
print(f'{api} calls, {tokens:,} tokens')
"

# Trajectory browser
mini-extra inspect output/<run>/
```

## 12. Common Failure Modes

| Symptom | Cause | Fix |
|---------|-------|-----|
| `ContextWindowExceeded` | Input + max_tokens > context | Reduce max_tokens or max-model-len |
| `RepeatedFormatError` | Model produces 0 tool calls | Use native tool parser, increase temp |
| `LimitsExceeded` | 250-step limit | Model stuck in loop; check trajectory |
| `InternalServerError` | Server crashed/restarted | Resume run, skip completed instances |
| CUDA compile error | Toolkit mismatch | Set `CUDA_HOME` to match venv |
| GPU OOM at startup | KV cache too large | Reduce `max-model-len` or `gpu-memory-utilization` |

## 13. Interpretation

Submission rate ≠ pass ratio. A model can submit 100% of instances with 0% resolved (bad patches). A model with 50% submission but 50% resolved is better than 100% submission with 10% resolved.

Token usage does not predict success. In our tests, resolved instances averaged 510K tokens; unresolved 519K.

The bash-only agent scaffold caps resolution at ~60% regardless of model. File-edit tools are needed to go higher.
