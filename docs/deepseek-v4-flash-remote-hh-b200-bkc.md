# DeepSeek V4 Flash via Remote `hh-b200` for Local SWE-bench

Run the DeepSeek vLLM server on `hh-b200`, keep SWE-bench execution on the current node, and connect them through an SSH tunnel.

## 1. Remote Server

Start the server on `hh-b200` inside the prepared container. This keeps the sparse attention funnel path and uses GPUs `6,7`.

```bash
ssh hh-b200
docker exec -d vllm-ds-yi bash -lc '
cd /workspace/vllm && \
source .venv/bin/activate && \
export PYTHONPATH=/workspace/funnel-topk:${PYTHONPATH} && \
export CUDA_VISIBLE_DEVICES=6,7 && \
export VLLM_SPARSE_INDEXER_PREFILL_TOPK_BACKEND=funnel_dense && \
export VLLM_SPARSE_INDEXER_PREFILL_TOPK_FUNNEL_MODE=turbo && \
export VLLM_DEEP_GEMM_WARMUP=skip && \
vllm serve DeepSeek-V4-Flash \
  --host 127.0.0.1 \
  --port 8000 \
  --trust-remote-code \
  --kv-cache-dtype fp8 \
  --block-size 256 \
  --tensor-parallel-size 2 \
  --enable-expert-parallel \
  --attention_config.use_fp4_indexer_cache=True \
  --moe-backend deep_gemm_mega_moe \
  --tokenizer-mode deepseek_v4 \
  --tool-call-parser deepseek_v4 \
  --enable-auto-tool-choice \
  --reasoning-parser deepseek_v4 \
  --kernel-config.enable_flashinfer_autotune=False \
  --speculative-config '\''{"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"greedy"}'\''
'
```

The server stays bound to `127.0.0.1:8000` on `hh-b200`.

## 2. Local Benchmark Config

Use [src/minisweagent/config/benchmarks/swebench_deepseek_hh_b200.yaml](/home/yiliu7/workspace/mini-swe-agent-fork/src/minisweagent/config/benchmarks/swebench_deepseek_hh_b200.yaml), which points the agent at the locally forwarded endpoint:

```yaml
model:
  model_name: "hosted_vllm/DeepSeek-V4-Flash"
  model_kwargs:
    api_base: "http://127.0.0.1:18000/v1"
```

## 3. Health Check

From the current node:

```bash
mini-extra deepseek-local check --host hh-b200
```

This verifies SSH reachability, the remote `/v1/models` endpoint, and any existing local tunnel.

## 4. Run 10 SWE-bench Verified Instances

The utility opens the tunnel if needed, runs generation locally, then scores locally.

```bash
mini-extra deepseek-local swebench 10 \
  --host hh-b200 \
  --config-name swebench_deepseek_hh_b200.yaml \
  --run-id deepseek-v4-flash-hh-b200-10
```

Current defaults still apply unless overridden:

- subset: `verified`
- split: `test`
- workers: `2`
- local forward: `127.0.0.1:18000 -> hh-b200:127.0.0.1:8000`

## 5. Manual Tunnel Commands

```bash
mini-extra deepseek-local start-tunnel --host hh-b200
mini-extra deepseek-local check --host hh-b200
mini-extra deepseek-local stop-tunnel --host hh-b200
```

## 6. Output

The local run directory contains `preds.json`, `preds.jsonl`, trajectories, and `minisweagent.log`. Scoring is run automatically after generation.
