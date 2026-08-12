# DeepSeek V4 Flash via Remote B200 vLLM — Best Known Configuration

Run mini-SWE-agent and the SWE-bench task containers on the current node, but use the DeepSeek vLLM server running on `b200-aws`.

## 1. Architecture

- Agent runner: current node
- SWE-bench Docker containers: current node
- vLLM server: `b200-aws`
- Connectivity: SSH local port forwarding from this node to `b200-aws:127.0.0.1:8000`

The remote API stays bound to `127.0.0.1` on `b200-aws`. It is not directly exposed on the network.

## 2. Local Environment

Create a local repo environment on this node:

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -U pip
pip install -e '.[dev]'
pip install swebench
```

Docker must be available locally because the benchmark environments still run on this node.

## 3. Benchmark Config

Use `src/minisweagent/config/benchmarks/swebench_deepseek_b200.yaml`:

```yaml
model:
  model_name: "hosted_vllm//storage/yiliu7/deepseek-ai/DeepSeek-V4-Flash/"
  cost_tracking: "ignore_errors"
  model_kwargs:
    api_base: "http://127.0.0.1:18000/v1"
    drop_params: true
    max_tokens: 49152
    temperature: 1.0
    top_p: 0.95
    timeout: 900000
environment:
  pull_timeout: 600
```

The forwarded endpoint on this node is always:

```text
http://127.0.0.1:18000/v1
```

## 4. Remote Server Assumption

This workflow assumes:

- `ssh b200-aws` works from this node
- the remote vLLM server is healthy on `b200-aws`
- the remote health endpoint is `http://127.0.0.1:8000/v1/models`

You can verify the remote server with:

```bash
mini-extra deepseek-local check
```

## 5. Run 10 SWE-bench Verified Instances

The utility manages the SSH tunnel and runs the benchmark against the forwarded API:

```bash
mini-extra deepseek-local swebench
```

Current defaults:

- subset: `verified`
- split: `test`
- slice: `0:10`
- workers: `2`
- config merge: `swebench.yaml` + `swebench_deepseek_b200.yaml` + `agent.mode=yolo`

If you want to inspect the tunnel separately:

```bash
mini-extra deepseek-local start-tunnel
mini-extra deepseek-local check
mini-extra deepseek-local stop-tunnel
```

## 6. Scoring

`mini-extra deepseek-local swebench` performs three steps:

1. runs the 10-instance SWE-bench job
2. converts `preds.json` into `preds.jsonl`
3. runs local SWE-bench harness evaluation

The evaluation command is:

```bash
python -m swebench.harness.run_evaluation \
  --dataset_name princeton-nlp/SWE-Bench_Verified \
  --predictions_path output/<run>/preds.jsonl \
  --split test \
  --max_workers 2 \
  --run_id deepseek-v4-b200
```

## 7. Expected Artifacts

The run output directory contains:

- per-instance trajectories
- `preds.json`
- `preds.jsonl`
- `minisweagent.log`

The local harness also writes its evaluation outputs in the usual SWE-bench locations.
