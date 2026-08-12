# DeepSeek V4 Flash on `hh-b200` - RULER `niah_single_1` 128k Full Runs

## Scope

This page collects the completed full `500`-sample `niah_single_1` 128k runs in one place.

Shared setup across runs:

- model `DeepSeek-V4-Flash`
- serving node `hh-b200`
- local tunnel `127.0.0.1:18000 -> hh-b200:127.0.0.1:8001`
- local `lm_eval` client on the current node
- task `niah_single_1`
- target length `131072`
- `batch_size=1`
- `max_length=131200`

## Full-Run Summary

| Run | Prefill mode | `num_concurrent` | Result | Correct | Wrong | API errors | Artifact |
|---|---|---:|---:|---:|---:|---:|---|
| Baseline | default | `64` | `1.000` | `500` | `0` | `0` | `output/ruler_niah_single_1_128k_full_20260803_225125/` |
| Funnel turbo run 1 | `funnel_dense` + `turbo` | `64` | `0.992` | `496` | `4` | `0` | `output/ruler_niah_single_1_128k_funnel_turbo_full_20260804_020151_retry/` |
| Funnel turbo run 2 | `funnel_dense` + `turbo` | `64` | `0.994` | `497` | `3` | `0` | `output/ruler_niah_single_1_128k_funnel_turbo_reeval_20260804_045912/` |
| Funnel turbo run 3 | `funnel_dense` + `turbo` | `1` | `0.998` | `499` | `1` | `0` | `output/ruler_niah_single_1_128k_funnel_turbo_full_conc1_20260804_185001/` |
| Funnel standard v1 fixed | `funnel_dense` + `standard` | `32` | `0.994` | `497` | `3` | `0` | `output/ruler_niah_single_1_128k_funnel_standard_v1_full_fixed_20260807_063104/` |

## Failed Cases

| Run | Failed `doc_id`s |
|---|---|
| Baseline | none |
| Funnel turbo run 1 | `179`, `262`, `387`, `486` |
| Funnel turbo run 2 | `53`, `282`, `408` |
| Funnel turbo run 3 | `436` |
| Funnel standard v1 fixed | `8`, `118`, `261` |

- Overlap across funnel failed sets: none
- All failures were answer-quality misses, not API transport failures

## Reproduction Checks

The failed cases were replayed in isolation at full 128k prompt length.

| Source run | Replayed failed cases | Replay result | Report |
|---|---|---|---|
| Funnel turbo run 1 | `179`, `262`, `387`, `486` | did not reproduce | `docs/deepseek-v4-flash-hh-b200-ruler-niah-single-1-128k-logprob-analysis.md` |
| Funnel turbo run 2 | `53`, `282`, `408` | did not reproduce | `docs/deepseek-v4-flash-hh-b200-ruler-niah-single-1-128k-logprob-analysis-reeval.md` |
| Funnel turbo run 3 | `436` | did not reproduce | `docs/deepseek-v4-flash-hh-b200-ruler-niah-single-1-128k-logprob-analysis-conc1-doc436.md` |

For the isolated replays:

- baseline matched on all replayed prompts
- funnel matched on all replayed prompts
- funnel kept the gold token top-1 on the replayed failed prompts

## Interpretation

- The default prefill baseline stayed at `500/500`.
- Funnel turbo at `num_concurrent=64` dropped to `496/500` and `497/500` across two full runs.
- Funnel turbo at `num_concurrent=1` improved to `499/500`.
- Funnel standard v1 after the top-k clamp fix completed cleanly at `497/500` with `num_concurrent=32`.
- Since none of the failed cases reproduced in isolated replay, the current evidence points more to runtime behavior under load than to a deterministic prompt-specific corruption.

## Related Pages

- [deepseek-v4-flash-hh-b200-ruler-niah-single-1-128k-summary.md](/home/yiliu7/workspace/mini-swe-agent-fork/docs/deepseek-v4-flash-hh-b200-ruler-niah-single-1-128k-summary.md:1)
- [deepseek-v4-flash-hh-b200-ruler-niah-single-1-128k-funnel-turbo-summary.md](/home/yiliu7/workspace/mini-swe-agent-fork/docs/deepseek-v4-flash-hh-b200-ruler-niah-single-1-128k-funnel-turbo-summary.md:1)
- [deepseek-v4-flash-hh-b200-ruler-niah-single-1-128k-funnel-comparison.md](/home/yiliu7/workspace/mini-swe-agent-fork/docs/deepseek-v4-flash-hh-b200-ruler-niah-single-1-128k-funnel-comparison.md:1)
