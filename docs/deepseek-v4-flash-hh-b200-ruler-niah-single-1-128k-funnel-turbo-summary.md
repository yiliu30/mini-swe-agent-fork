# DeepSeek V4 Flash on `hh-b200` - RULER `niah_single_1` 128k Funnel Turbo Summary

## Run Summary

| | |
|---|---|
| Model | `DeepSeek-V4-Flash` |
| Serving node | remote `hh-b200` |
| Client node | current local node |
| Access path | local tunnel `127.0.0.1:18000 -> hh-b200:127.0.0.1:8001` |
| Task | `niah_single_1` |
| Sequence length under test | `131072` |
| API backend | `local-completions` |
| Batch size | `1` |
| Full-run concurrencies tested | `64`, `1` |
| GPUs | `4,5` on `hh-b200` |

## Final Result

- Smoke run at `131072`: `1.0` on `5/5`
- Full run at `131072` with `num_concurrent=1`: `0.998` on `499/500`
- Latest full re-eval at `131072`: `0.994` on `497/500`
- Earlier full run at `131072`: `0.992` on `496/500`
- API errors: `0`
- Empty responses: `0`

The extra `4096 = -1` entry is expected from the task metric schema when `--metadata '{"max_seq_lengths":[131072]}'` is used.

## Artifacts

- Smoke output: `output/ruler_niah_single_1_128k_funnel_turbo_smoke_20260804_020009/`
- Smoke results: `output/ruler_niah_single_1_128k_funnel_turbo_smoke_20260804_020009/DeepSeek-V4-Flash/results_2026-08-04T02-00-48.930164.json`
- Full `num_concurrent=1` output: `output/ruler_niah_single_1_128k_funnel_turbo_full_conc1_20260804_185001/`
- Full `num_concurrent=1` results: `output/ruler_niah_single_1_128k_funnel_turbo_full_conc1_20260804_185001/DeepSeek-V4-Flash/results_2026-08-04T19-17-13.292506.json`
- Full `num_concurrent=1` samples: `output/ruler_niah_single_1_128k_funnel_turbo_full_conc1_20260804_185001/DeepSeek-V4-Flash/samples_niah_single_1_2026-08-04T19-17-13.292506.jsonl`
- Latest full re-eval output: `output/ruler_niah_single_1_128k_funnel_turbo_reeval_20260804_045912/`
- Latest full re-eval results: `output/ruler_niah_single_1_128k_funnel_turbo_reeval_20260804_045912/DeepSeek-V4-Flash/results_2026-08-04T05-18-15.394991.json`
- Latest full re-eval samples: `output/ruler_niah_single_1_128k_funnel_turbo_reeval_20260804_045912/DeepSeek-V4-Flash/samples_niah_single_1_2026-08-04T05-18-15.394991.jsonl`
- Earlier full output: `output/ruler_niah_single_1_128k_funnel_turbo_full_20260804_020151_retry/`
- Earlier full results: `output/ruler_niah_single_1_128k_funnel_turbo_full_20260804_020151_retry/DeepSeek-V4-Flash/results_2026-08-04T02-21-11.771347.json`
- Earlier full samples: `output/ruler_niah_single_1_128k_funnel_turbo_full_20260804_020151_retry/DeepSeek-V4-Flash/samples_niah_single_1_2026-08-04T02-21-11.771347.jsonl`

## Server Start

Remote startup script used inside `vllm-ds-yi`:

```bash
#!/usr/bin/env bash
set -euo pipefail
cd /workspace/vllm
source .venv/bin/activate
export PYTHONPATH=/workspace/funnel-topk:
export CUDA_VISIBLE_DEVICES=4,5
export VLLM_SPARSE_INDEXER_PREFILL_TOPK_BACKEND=funnel_dense
export VLLM_SPARSE_INDEXER_PREFILL_TOPK_FUNNEL_MODE=turbo
export VLLM_DEEP_GEMM_WARMUP=skip
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
```

Confirmed funnel prefill activation from the remote log:

```text
(Worker_TP0_EP0 pid=88749) WARNING ... Using funnel_dense backend for top-k prefill.
(Worker_TP0_EP0 pid=88749) WARNING ... using funnel_topk ragged prefill v1 op for prefill top-k (VLLM_SPARSE_INDEXER_PREFILL_TOPK_BACKEND=funnel_dense)
```

## Eval Commands

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
  --output_path /home/yiliu7/workspace/mini-swe-agent-fork/output/ruler_niah_single_1_128k_funnel_turbo_smoke_20260804_020009
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
  --output_path /home/yiliu7/workspace/mini-swe-agent-fork/output/ruler_niah_single_1_128k_funnel_turbo_full_20260804_020151_retry
```

Latest re-eval used the same `lm_eval` command shape, with only the output directory changed to:

```bash
/home/yiliu7/workspace/mini-swe-agent-fork/output/ruler_niah_single_1_128k_funnel_turbo_reeval_20260804_045912
```

Full `num_concurrent=1` run:

```bash
/home/yiliu7/workspace/lm-evaluation-harness/.venv/bin/python -P -m lm_eval \
  --model local-completions \
  --tasks niah_single_1 \
  --batch_size 1 \
  --model_args model=DeepSeek-V4-Flash,base_url=http://127.0.0.1:18000/v1/completions,tokenizer=deepseek-ai/DeepSeek-V4-Flash,tokenizer_backend=huggingface,tokenized_requests=False,num_concurrent=1,max_retries=10,max_length=131200 \
  --metadata '{"max_seq_lengths":[131072]}' \
  --log_samples \
  --output_path /home/yiliu7/workspace/mini-swe-agent-fork/output/ruler_niah_single_1_128k_funnel_turbo_full_conc1_20260804_185001
```

## Timing

- Synthetic sample generation: about `2m 46s`
- Request phase: about `16m 08s`
- Total evaluation time: `1137.55s` (`18m 58s`)
- Full `num_concurrent=1` request phase: about `24m 18s`
- Full `num_concurrent=1` total evaluation time: about `27m 04s`

## Comparison To Baseline

Baseline reference: `docs/deepseek-v4-flash-hh-b200-ruler-niah-single-1-128k-summary.md`

| Run | Prefill mode | Full result @ 128k | Notes |
|---|---|---:|---|
| Baseline | default | `1.000` | previous run |
| Funnel turbo, `num_concurrent=64` | `funnel_dense` + `turbo` | `0.994` | latest `64`-concurrency re-eval, `497/500`, `3` wrong samples |
| Funnel turbo, `num_concurrent=1` | `funnel_dense` + `turbo` | `0.998` | `499/500`, `1` wrong sample |

The earlier funnel turbo run scored `0.992` on `496/500`. Lowering concurrency from `64` to `1` improved the funnel result from `0.994` to `0.998`, which supports concurrency as one contributor to the earlier misses.

## Cross-Run Failure Analysis

- Initial funnel-turbo failures: `179`, `262`, `387`, `486`
- Latest funnel-turbo re-eval failures: `53`, `282`, `408`
- Full `num_concurrent=1` failure: `436`
- Overlap between the two failed sets: none

This means the funnel-turbo misses were not stable across the two repeated full runs. In both runs, the wrong outputs looked like truncated or malformed number extraction rather than transport failure, and both runs still had `0` API errors.
