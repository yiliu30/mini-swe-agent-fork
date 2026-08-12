# DeepSeek V4 Flash on `hh-b200` - SWE-mini 50 Summary

## Run Summary

| | |
|---|---|
| Model | `hosted_vllm/DeepSeek-V4-Flash` |
| Serving node | remote `hh-b200` via local tunnel `127.0.0.1:18000 -> hh-b200:127.0.0.1:8000` |
| Agent node | current local node |
| Config | `src/minisweagent/config/benchmarks/swebench_deepseek_hh_b200.yaml` |
| Dataset | `MariusHobbhahn/swe-bench-verified-mini` |
| Main output | `output/deepseek_v4_hh_b200_swe_mini_50_20260803_1854/` |
| Main evaluation | `logs/run_evaluation/deepseek-v4-flash-hh-b200-swe-mini-50-partial-20260803-192257/` |
| Main report | `hosted_vllm__DeepSeek-V4-Flash.deepseek-v4-flash-hh-b200-swe-mini-50-partial-20260803-192257.json` |
| Retry output | `output/deepseek_v4_hh_b200_swe_mini_retry_6_20260803_193359/` |
| Retry evaluation | `logs/run_evaluation/deepseek-v4-flash-hh-b200-swe-mini-retry-4-20260803-200019/` |
| Retry report | `hosted_vllm__DeepSeek-V4-Flash.deepseek-v4-flash-hh-b200-swe-mini-retry-4-20260803-200019.json` |

## Final Result

- Targeted instances: `50`
- Main-pass generated predictions: `50/50`
- Main-pass non-empty patches evaluated: `44`
- Main-pass resolved: `27/50`
- Main-pass unresolved after completed evaluation: `14/50`
- Main-pass eval errors: `3/50`
- Main-pass empty patches: `6/50`
- Retry batch size: `6`
- Retry non-empty patches evaluated: `4/6`
- Retry resolved: `2/6`
- Retry unresolved: `2/6`
- Final combined evaluated non-empty patches: `48/50`
- Final combined resolved: `29/50`
- Final combined unresolved after evaluation: `16/50`
- Final combined eval errors/timeouts: `3/50`
- Final still non-evaluable after retry: `2/50`

Score views:

- End-to-end score on the 50 targeted instances: `29/50 = 58.0%`
- Score on evaluated non-empty patches after retry: `29/48 = 60.4%`
- Main-pass score on evaluated non-empty patches: `27/44 = 61.4%`
- Retry score on evaluated retry patches: `2/4 = 50.0%`

## Server Startup

Remote DeepSeek server on `hh-b200`:

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

The SWE-mini run used this remote server through the local tunnel `127.0.0.1:18000 -> hh-b200:127.0.0.1:8000`.

## Commands Used

Main generation:

```bash
mini-extra swebench \
  --subset MariusHobbhahn/swe-bench-verified-mini \
  --split test \
  --slice 0:50 \
  --workers 32 \
  --output output/deepseek_v4_hh_b200_swe_mini_50_20260803_1854 \
  -c swebench.yaml \
  -c swebench_deepseek_hh_b200.yaml \
  -c agent.mode=yolo
```

Main evaluation:

```bash
python -m swebench.harness.run_evaluation \
  --dataset_name MariusHobbhahn/swe-bench-verified-mini \
  --predictions_path output/deepseek_v4_hh_b200_swe_mini_50_20260803_1854/preds.jsonl \
  --split test \
  --max_workers 32 \
  --run_id deepseek-v4-flash-hh-b200-swe-mini-50-partial-20260803-192257
```

Retry generation:

```bash
mini-extra swebench \
  --subset MariusHobbhahn/swe-bench-verified-mini \
  --split test \
  --filter '^(django__django-11951|django__django-12325|sphinx-doc__sphinx-7590|sphinx-doc__sphinx-10466|sphinx-doc__sphinx-8548|sphinx-doc__sphinx-10435)$' \
  --workers 6 \
  --output output/deepseek_v4_hh_b200_swe_mini_retry_6_20260803_193359 \
  -c swebench.yaml \
  -c src/minisweagent/config/benchmarks/swebench_deepseek_hh_b200.yaml \
  -c agent.mode=yolo
```

Retry evaluation:

```bash
python -m swebench.harness.run_evaluation \
  --dataset_name MariusHobbhahn/swe-bench-verified-mini \
  --predictions_path output/deepseek_v4_hh_b200_swe_mini_retry_6_20260803_193359/preds.jsonl \
  --split test \
  --max_workers 4 \
  --run_id deepseek-v4-flash-hh-b200-swe-mini-retry-4-20260803-200019
```

## Main-Pass Evaluation Details

- The main generation run produced `50` trajectories and `50` prediction entries.
- Only `44` predictions had non-empty `model_patch` values, so only those `44` entered the first harness evaluation.
- The `6` non-evaluable cases were:
  - `django__django-11951`
  - `django__django-12325`
  - `sphinx-doc__sphinx-7590`
  - `sphinx-doc__sphinx-10466`
  - `sphinx-doc__sphinx-8548`
  - `sphinx-doc__sphinx-10435`
- The `3` main-pass harness errors were test timeouts:
  - `sphinx-doc__sphinx-7985`
  - `sphinx-doc__sphinx-8269`
  - `sphinx-doc__sphinx-8475`

## Retry Outcome

Retry resolved:

- `django__django-11951`
- `sphinx-doc__sphinx-10466`

Retry unresolved:

- `django__django-12325`
- `sphinx-doc__sphinx-7590`

Still not evaluable after retry:

- `sphinx-doc__sphinx-10435`
- `sphinx-doc__sphinx-8548`

## Artifacts

- Main run log: `output/deepseek_v4_hh_b200_swe_mini_50_20260803_1854/minisweagent.log`
- Main run predictions: `output/deepseek_v4_hh_b200_swe_mini_50_20260803_1854/preds.json`
- Main run filtered predictions: `output/deepseek_v4_hh_b200_swe_mini_50_20260803_1854/preds.jsonl`
- Retry log: `output/deepseek_v4_hh_b200_swe_mini_retry_6_20260803_193359/minisweagent.log`
- Retry predictions: `output/deepseek_v4_hh_b200_swe_mini_retry_6_20260803_193359/preds.json`
- Retry filtered predictions: `output/deepseek_v4_hh_b200_swe_mini_retry_6_20260803_193359/preds.jsonl`
