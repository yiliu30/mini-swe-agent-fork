# DeepSeek V4 Flash on `hh-b200` - RULER `niah_single_1` 128k With vs Without Funnel

## Scope

This page compares all completed `niah_single_1` 128k runs on remote `hh-b200`:

- baseline vLLM startup without funnel prefill
- initial vLLM startup with funnel prefill enabled in `turbo` mode
- latest funnel-turbo re-eval
- full funnel-turbo run with `num_concurrent=1`
- full funnel-standard-v1 run after the native top-k clamp fix

All runs used:

- model `DeepSeek-V4-Flash`
- remote serving on `hh-b200`
- local tunnel `127.0.0.1:18000 -> hh-b200:127.0.0.1:8001`
- local `lm_eval` client on the current node
- task `niah_single_1`
- target length `131072`
- `batch_size=1`
- `max_length=131200`

The intended apples-to-apples difference started as the vLLM server startup mode, and later also included a concurrency check at `num_concurrent=1`.

## Result Summary

| Run | Prefill mode | Full result @ 128k | Correct | Wrong | API errors |
|---|---|---:|---:|---:|---:|
| Without funnel | default | `1.000` | `500` | `0` | `0` |
| With funnel, `num_concurrent=1` | `funnel_dense` + `turbo` | `0.998` | `499` | `1` | `0` |
| With funnel, standard v1 fixed | `funnel_dense` + `standard` | `0.994` | `497` | `3` | `0` |
| With funnel, initial run | `funnel_dense` + `turbo` | `0.992` | `496` | `4` | `0` |
| With funnel, latest re-eval | `funnel_dense` + `turbo` | `0.994` | `497` | `3` | `0` |

## Artifacts

Without funnel:

- Summary: `docs/deepseek-v4-flash-hh-b200-ruler-niah-single-1-128k-summary.md`
- Full results: `output/ruler_niah_single_1_128k_full_20260803_225125/DeepSeek-V4-Flash/results_2026-08-03T23-09-09.211915.json`

With funnel:

- Summary: `docs/deepseek-v4-flash-hh-b200-ruler-niah-single-1-128k-funnel-turbo-summary.md`
- Initial full results: `output/ruler_niah_single_1_128k_funnel_turbo_full_20260804_020151_retry/DeepSeek-V4-Flash/results_2026-08-04T02-21-11.771347.json`
- Latest full re-eval results: `output/ruler_niah_single_1_128k_funnel_turbo_reeval_20260804_045912/DeepSeek-V4-Flash/results_2026-08-04T05-18-15.394991.json`
- Full `num_concurrent=1` results: `output/ruler_niah_single_1_128k_funnel_turbo_full_conc1_20260804_185001/DeepSeek-V4-Flash/results_2026-08-04T19-17-13.292506.json`
- Full standard-v1 fixed results: `output/ruler_niah_single_1_128k_funnel_standard_v1_full_fixed_20260806_063104/DeepSeek-V4-Flash/results_2026-08-06T06-48-48.686847.json`

## What Changed

Without funnel, the server used the default sparse prefill path.

With funnel, the server added:

```bash
export PYTHONPATH=/workspace/funnel-topk:
export VLLM_SPARSE_INDEXER_PREFILL_TOPK_BACKEND=funnel_dense
export VLLM_SPARSE_INDEXER_PREFILL_TOPK_FUNNEL_MODE=turbo
```

For the standard-v1 fixed run, the only startup change was:

```bash
export PYTHONPATH=/workspace/funnel-topk:
export VLLM_SPARSE_INDEXER_PREFILL_TOPK_BACKEND=funnel_dense
export VLLM_SPARSE_INDEXER_PREFILL_TOPK_FUNNEL_MODE=standard
```

Confirmed from remote log:

```text
(Worker_TP0_EP0 pid=88749) WARNING ... Using funnel_dense backend for top-k prefill.
(Worker_TP0_EP0 pid=88749) WARNING ... using funnel_topk ragged prefill v1 op for prefill top-k (VLLM_SPARSE_INDEXER_PREFILL_TOPK_BACKEND=funnel_dense)
```

## Notes

- The baseline run achieved `500/500`.
- The initial funnel run achieved `496/500`.
- The latest funnel re-eval achieved `497/500`.
- The full funnel run with `num_concurrent=1` achieved `499/500`.
- The full funnel standard-v1 fixed run achieved `497/500`.
- All funnel runs completed without transport or API failures.
- Relative to baseline, the initial funnel run missed `4` cases, the latest `64`-concurrency re-eval missed `3`, and the full `num_concurrent=1` run missed `1`.
- Relative to baseline, the standard-v1 fixed run missed `3` cases at `num_concurrent=32`.
- Moving from `num_concurrent=64` to `num_concurrent=1` improved the funnel result from `0.994` to `0.998`, which supports concurrency as a contributor to the earlier failures.
- All result files also include `4096 = -1`, which is expected from the task metric schema when only `131072` is requested through metadata.

## Failure Overlap Analysis

The failed cases did not line up across the two funnel runs.

| Run | Failed `doc_id`s |
|---|---|
| With funnel, initial run | `179`, `262`, `387`, `486` |
| With funnel, latest re-eval | `53`, `282`, `408` |
| With funnel, `num_concurrent=1` | `436` |
| With funnel, standard v1 fixed | `8`, `118`, `261` |

- Overlap: none
- Initial-only failures: `179`, `262`, `387`, `486`
- Re-eval-only failures: `53`, `282`, `408`
- `num_concurrent=1`-only failure: `436`

This is evidence that the funnel-turbo degradation here is not tied to one fixed subset of prompts. The observed failures were numeric extraction mistakes such as truncation or malformed short answers, while server-side request completion still remained clean with `0` API errors in all runs.

## References

- Baseline page: [deepseek-v4-flash-hh-b200-ruler-niah-single-1-128k-summary.md](/home/yiliu7/workspace/mini-swe-agent-fork/docs/deepseek-v4-flash-hh-b200-ruler-niah-single-1-128k-summary.md:1)
- Funnel page: [deepseek-v4-flash-hh-b200-ruler-niah-single-1-128k-funnel-turbo-summary.md](/home/yiliu7/workspace/mini-swe-agent-fork/docs/deepseek-v4-flash-hh-b200-ruler-niah-single-1-128k-funnel-turbo-summary.md:1)
