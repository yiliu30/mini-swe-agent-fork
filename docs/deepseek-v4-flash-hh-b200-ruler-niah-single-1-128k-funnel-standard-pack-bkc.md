# DeepSeek V4 Flash on `hh-b200` - Funnel Standard Pack BKC

This runbook is for the experimental prefill backend that:

- packs each ragged sparse-indexer row into a dense left-aligned tensor
- calls `funnel_topk(..., mode="standard")`
- writes the selected packed offsets back into the original row-local top-k buffer

The vLLM-side implementation lives in the local checkout at `/home/yiliu7/workspace/vllm`:

- `vllm/attention/ops/common.py`
- `vllm/model_executor/models/deepseek_v2.py`
- `tests/v1/attention/test_sparse_prefill_standard_pack.py`

Prepared shared `/home` artifacts:

- patch bundle: `/home/yiliu7/workspace/vllm-funnel-standard-pack.patch`
- apply helper: `/home/yiliu7/workspace/mini-swe-agent-fork/scripts/apply_hh_b200_funnel_standard_pack.sh`

## Backend Selector

Use this environment variable on the DeepSeek server:

```bash
export VLLM_SPARSE_INDEXER_PREFILL_TOPK_BACKEND=funnel_dense_standard_pack
```

That keeps decode on the existing path and only swaps the prefill top-k selection.

## Remote Server

Start the server on `hh-b200` inside `vllm-ds-yi` with GPUs `3,4`.

If the remote repo still needs to be synced first, run this on `hh-b200` host:

```bash
bash /home/yiliu7/workspace/mini-swe-agent-fork/scripts/apply_hh_b200_funnel_standard_pack.sh
```

```bash
ssh hh-b200
docker exec -d vllm-ds-yi bash -lc '
set -euo pipefail
cd /workspace/vllm
source .venv/bin/activate
export PYTHONPATH=/workspace/funnel-topk:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=3,4
export VLLM_SPARSE_INDEXER_PREFILL_TOPK_BACKEND=funnel_dense_standard_pack
export VLLM_DEEP_GEMM_WARMUP=skip
pkill -f "/workspace/vllm/.venv/bin/vllm serve /dev/shm/.yiliu7/deepseek-ai/DeepSeek-V4-Flash" || true
exec /workspace/vllm/.venv/bin/vllm serve /dev/shm/.yiliu7/deepseek-ai/DeepSeek-V4-Flash \
  --served-model-name DeepSeek-V4-Flash \
  --host 127.0.0.1 \
  --port 8001 \
  --trust-remote-code \
  --kv-cache-dtype fp8 \
  --block-size 256 \
  --enable-expert-parallel \
  --tensor-parallel-size 2 \
  --attention_config.use_fp4_indexer_cache=True \
  --moe-backend deep_gemm_mega_moe \
  --tokenizer-mode deepseek_v4 \
  --tool-call-parser deepseek_v4 \
  --enable-auto-tool-choice \
  --reasoning-parser deepseek_v4 \
  --kernel-config.enable_flashinfer_autotune=False
'
```

Expected log marker:

```text
Using funnel_dense_standard_pack backend for DeepSeek sparse prefill top-k.
```

## Tunnel

Forward local port `18000` to remote `8001`:

```bash
ssh -f -N -L 127.0.0.1:18000:127.0.0.1:8001 hh-b200
```

Quick checks:

```bash
curl http://127.0.0.1:18000/v1/models
bash scripts/check_vllm_status.sh --local-base-url http://127.0.0.1:18000
```

## 128k Eval

Smoke run:

```bash
/home/yiliu7/workspace/lm-evaluation-harness/.venv/bin/python -P -m lm_eval \
  --model local-completions \
  --tasks niah_single_1 \
  --batch_size 1 \
  --model_args model=DeepSeek-V4-Flash,base_url=http://127.0.0.1:18000/v1/completions,tokenizer=deepseek-ai/DeepSeek-V4-Flash,tokenizer_backend=huggingface,tokenized_requests=False,num_concurrent=1,max_retries=10,max_length=131200 \
  --metadata '{"max_seq_lengths":[131072],"num_samples":5}' \
  --limit 5 \
  --log_samples \
  --output_path /home/yiliu7/workspace/mini-swe-agent-fork/output/ruler_niah_single_1_128k_funnel_standard_pack_smoke
```

Full run:

```bash
/home/yiliu7/workspace/lm-evaluation-harness/.venv/bin/python -P -m lm_eval \
  --model local-completions \
  --tasks niah_single_1 \
  --batch_size 1 \
  --model_args model=DeepSeek-V4-Flash,base_url=http://127.0.0.1:18000/v1/completions,tokenizer=deepseek-ai/DeepSeek-V4-Flash,tokenizer_backend=huggingface,tokenized_requests=False,num_concurrent=64,max_retries=10,max_length=131200 \
  --metadata '{"max_seq_lengths":[131072]}' \
  --log_samples \
  --output_path /home/yiliu7/workspace/mini-swe-agent-fork/output/ruler_niah_single_1_128k_funnel_standard_pack_full
```

## Current Limitation

From this Codex execution environment, noninteractive `ssh hh-b200 ...` currently authenticates and then hangs before remote command execution completes, so the remote server start could not be launched automatically here. The local code changes are in place and compiled successfully.
