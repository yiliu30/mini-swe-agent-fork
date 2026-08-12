# DeepSeek V4 Flash on `hh-b200` - SWE-mini 50 Baseline Default Summary

## Run Summary

| | |
|---|---|
| Model | `hosted_vllm/DeepSeek-V4-Flash` |
| Serving node | remote `hh-b200` via local tunnel `127.0.0.1:18000 -> hh-b200:127.0.0.1:8001` |
| Agent node | current local node |
| Config | `src/minisweagent/config/benchmarks/swebench_deepseek_hh_b200.yaml` |
| Dataset | `MariusHobbhahn/swe-bench-verified-mini` |
| Main output | `output/deepseek_v4_hh_b200_swe_mini_50_baseline_default_20260803_2048/` |
| Main evaluation | `logs/run_evaluation/deepseek-v4-flash-hh-b200-swe-mini-50-baseline-default-main-20260803-212912/` |
| Main report | `hosted_vllm__DeepSeek-V4-Flash.deepseek-v4-flash-hh-b200-swe-mini-50-baseline-default-main-20260803-212912.json` |
| Retry output | `output/deepseek_v4_hh_b200_swe_mini_baseline_default_retry_2_20260803_222202/` |
| Retry evaluation | `logs/run_evaluation/deepseek-v4-flash-hh-b200-swe-mini-50-baseline-default-retry-2-20260803-222202/` |
| Retry report | `hosted_vllm__DeepSeek-V4-Flash.deepseek-v4-flash-hh-b200-swe-mini-50-baseline-default-retry-2-20260803-222202.json` |

## Baseline Definition

This baseline intentionally removed the funnel top-k environment variables:

- `VLLM_SPARSE_INDEXER_PREFILL_TOPK_BACKEND`
- `VLLM_SPARSE_INDEXER_PREFILL_TOPK_FUNNEL_MODE`

It also ran without `--speculative-config`.

That last point matters because the earlier funnel/turbo docs showed speculative decoding in the intended command, but the actual successful `hh-b200` runtime we inspected did not show `--speculative-config` in the live `vllm serve` process. Attempting to preserve `--speculative-config` for this default-topk baseline failed in practice with a `dflash_config.mask_token_id` error, so the working baseline omitted it.

## Final Result

- Targeted instances: `50`
- Main-pass generated predictions: `50/50`
- Main-pass non-empty patches submitted: `48/50`
- Main-pass completed evaluations: `44/48`
- Main-pass resolved: `31/50`
- Main-pass unresolved after completed evaluation: `13/50`
- Main-pass eval errors/timeouts: `4/50`
- Main-pass empty patches from generation: `2/50`
- Retry batch size: `2`
- Retry non-empty patches evaluated: `2/2`
- Retry resolved: `1/2`
- Retry unresolved: `1/2`
- Final combined evaluated non-empty patches: `50/50`
- Final combined resolved: `32/50`
- Final combined unresolved after evaluation: `14/50`
- Final combined eval errors/timeouts: `4/50`

Score views:

- End-to-end score on the 50 targeted instances: `32/50 = 64.0%`
- Score on evaluated non-empty patches after retry: `32/46 = 69.6%`
- Main-pass score on evaluated non-empty patches: `31/44 = 70.5%`
- Retry score on retry patches: `1/2 = 50.0%`

## Comparison vs Funnel/Turbo

Compared with the earlier funnel/turbo SWE-mini run in `docs/deepseek-v4-flash-hh-b200-swe-mini-50-summary.md`:

- Funnel/turbo final combined resolved: `29/50`
- Baseline default final combined resolved: `32/50`
- Funnel/turbo final combined evaluated non-empty patches: `48/50`
- Baseline default final combined evaluated non-empty patches: `50/50`
- Funnel/turbo final still non-evaluable after retry: `2/50`
- Baseline default final still non-evaluable after retry: `0/50`

On this 50-instance slice, the baseline default run outperformed the earlier funnel/turbo run by `+3` resolved instances.

## Apples-to-Apples Comparison

This is the fairest comparison view across the two 50-instance SWE-mini runs:

- Same model family: `DeepSeek-V4-Flash`
- Same dataset and slice: `MariusHobbhahn/swe-bench-verified-mini`, `test`, `0:50`
- Same agent node: current local node
- Same benchmark config: `src/minisweagent/config/benchmarks/swebench_deepseek_hh_b200.yaml`
- Same worker count for the main run: `32`
- Same remote serving node pattern: `hh-b200` over the local SSH tunnel
- Same tensor parallel size: `2`

Effective serving difference:

- Funnel/turbo run: set `VLLM_SPARSE_INDEXER_PREFILL_TOPK_BACKEND=funnel_dense` and `VLLM_SPARSE_INDEXER_PREFILL_TOPK_FUNNEL_MODE=turbo`
- Baseline default run: omitted both funnel top-k env vars

Important caveat:

- The older funnel/turbo doc shows `--speculative-config`, but the actual successful runtime we inspected on `hh-b200` did not show it in the live `vllm serve` process.
- The baseline default run also omitted `--speculative-config`, because preserving it failed at startup with a `dflash_config.mask_token_id` error.
- So the apples-to-apples comparison should be read as `funnel top-k enabled` vs `default top-k`, not as `speculative decoding enabled` vs `disabled`.

Apples-to-apples metrics:

| Metric | Funnel/Turbo | Baseline Default |
|---|---:|---:|
| Main-pass generated predictions | `50/50` | `50/50` |
| Main-pass non-empty patches submitted | `44/50` | `48/50` |
| Main-pass completed evaluations | `44/44` | `44/48` |
| Main-pass resolved | `27/50` | `31/50` |
| Main-pass eval errors/timeouts | `3/50` | `4/50` |
| Main-pass empty patches from generation | `6/50` | `2/50` |
| Final combined evaluated non-empty patches | `48/50` | `50/50` |
| Final combined resolved | `29/50` | `32/50` |
| Final combined eval errors/timeouts | `3/50` | `4/50` |
| Final still non-evaluable after retry | `2/50` | `0/50` |

Bottom line:

- On the same 50-instance slice, the baseline default run produced more valid submissions and more resolved instances.
- The baseline default advantage was `+4` on the main pass and `+3` after retries.

## Server Startup

Remote DeepSeek server on `hh-b200`:

```bash
ssh hh-b200
docker exec -d vllm-ds-yi bash -lc '
cd /workspace/vllm && \
source .venv/bin/activate && \
export PYTHONPATH=/workspace/funnel-topk:${PYTHONPATH:-} && \
export CUDA_VISIBLE_DEVICES=4,5 && \
export VLLM_DEEP_GEMM_WARMUP=skip && \
/workspace/vllm/.venv/bin/vllm serve /dev/shm/.yiliu7/deepseek-ai/DeepSeek-V4-Flash \
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

The working remote startup script used in the container was `/tmp/start_deepseek_baseline_default_8001.sh`.

## Commands Used

Main generation:

```bash
./.venv/bin/mini-extra swebench \
  --subset MariusHobbhahn/swe-bench-verified-mini \
  --split test \
  --slice 0:50 \
  --workers 32 \
  --output output/deepseek_v4_hh_b200_swe_mini_50_baseline_default_20260803_2048 \
  -c swebench.yaml \
  -c src/minisweagent/config/benchmarks/swebench_deepseek_hh_b200.yaml \
  -c agent.mode=yolo
```

Main evaluation:

```bash
./.venv/bin/python -m swebench.harness.run_evaluation \
  --dataset_name MariusHobbhahn/swe-bench-verified-mini \
  --predictions_path output/deepseek_v4_hh_b200_swe_mini_50_baseline_default_20260803_2048/preds.jsonl \
  --split test \
  --max_workers 32 \
  --run_id deepseek-v4-flash-hh-b200-swe-mini-50-baseline-default-main-20260803-212912
```

Retry generation:

```bash
./.venv/bin/mini-extra swebench \
  --subset MariusHobbhahn/swe-bench-verified-mini \
  --split test \
  --filter '^(sphinx-doc__sphinx-8721|sphinx-doc__sphinx-7748)$' \
  --workers 2 \
  --output output/deepseek_v4_hh_b200_swe_mini_baseline_default_retry_2_20260803_222202 \
  -c swebench.yaml \
  -c src/minisweagent/config/benchmarks/swebench_deepseek_hh_b200.yaml \
  -c agent.mode=yolo
```

Retry evaluation:

```bash
./.venv/bin/python -m swebench.harness.run_evaluation \
  --dataset_name MariusHobbhahn/swe-bench-verified-mini \
  --predictions_path output/deepseek_v4_hh_b200_swe_mini_baseline_default_retry_2_20260803_222202/preds.jsonl \
  --split test \
  --max_workers 2 \
  --run_id deepseek-v4-flash-hh-b200-swe-mini-50-baseline-default-retry-2-20260803-222202
```

## Main-Pass Evaluation Details

- The main generation run produced `50` trajectories and `50` prediction entries.
- `48` predictions had non-empty `model_patch` values and entered the main harness evaluation.
- The `2` generation-side empty-patch cases were:
  - `sphinx-doc__sphinx-8721`
  - `sphinx-doc__sphinx-7748`
- Those two were not model-failure eval results. Both ended with `RepeatedFormatError`, so the agent failed to emit a valid final submission string even though the model had drafted code changes in the trajectory.
- The `4` main-pass harness errors were test timeouts:
  - `sphinx-doc__sphinx-10435`
  - `sphinx-doc__sphinx-7985`
  - `sphinx-doc__sphinx-8269`
  - `sphinx-doc__sphinx-8475`

## Retry Outcome

Retry resolved:

- `sphinx-doc__sphinx-8721`

Retry unresolved:

- `sphinx-doc__sphinx-7748`

## Artifacts

- Main run log: `output/deepseek_v4_hh_b200_swe_mini_50_baseline_default_20260803_2048/minisweagent.log`
- Main run predictions: `output/deepseek_v4_hh_b200_swe_mini_50_baseline_default_20260803_2048/preds.json`
- Main run filtered predictions: `output/deepseek_v4_hh_b200_swe_mini_50_baseline_default_20260803_2048/preds.jsonl`
- Main evaluation console log: `output/deepseek_v4_hh_b200_swe_mini_50_baseline_default_20260803_2048/eval-main.log`
- Retry log: `output/deepseek_v4_hh_b200_swe_mini_baseline_default_retry_2_20260803_222202/minisweagent.log`
- Retry predictions: `output/deepseek_v4_hh_b200_swe_mini_baseline_default_retry_2_20260803_222202/preds.json`
- Retry filtered predictions: `output/deepseek_v4_hh_b200_swe_mini_baseline_default_retry_2_20260803_222202/preds.jsonl`
- Retry evaluation console log: `output/deepseek_v4_hh_b200_swe_mini_baseline_default_retry_2_20260803_222202/eval.log`
