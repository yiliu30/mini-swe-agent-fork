# DeepSeek V4 Flash on `hh-b200` - RULER `niah_single_1` 128k Logprob Analysis

## Summary

- Prompt source: `/home/yiliu7/workspace/mini-swe-agent-fork/output/ruler_niah_single_1_128k_funnel_turbo_full_20260804_020151_retry/DeepSeek-V4-Flash/samples_niah_single_1_2026-08-04T02-21-11.771347.jsonl`
- Replay set: `12` prompts (`4` misses + `8` controls)
- Serving comparison: baseline default top-k vs funnel `funnel_dense` + `turbo`
- Shared prompt set: yes
- Local tunnel: `127.0.0.1:18001 -> hh-b200:127.0.0.1:8001`

## Key Finding

- The earlier 4 funnel-run misses did not reproduce under isolated replay. Both modes matched `12/12` prompts.
- On all 12 prompts, the gold answer tokens stayed top-1 and within the returned top-20 for the funnel run.
- Funnel had a higher summed gold logprob on `9/12` prompts, including all `4/4` previously missed cases.

This points away from a deterministic single-request answer-token corruption. The earlier `0.992` funnel result is more likely tied to concurrency, batching, or longer-tail continuation behavior in the full 500-request run.


## Baseline Default Top-k

- Remote log: `/tmp/deepseek_logprob_baseline_8001.log`
- Generated matches: `12/12`
- Mean gold logprob: `-0.3129`



## Funnel Dense Turbo Top-k

- Remote log: `/tmp/deepseek_logprob_funnel_turbo_8001.log`
- Generated matches: `12/12`
- Mean gold logprob: `-0.1985`

- Funnel markers confirmed: `confirmed`



## Per-Prompt Comparison

| Id | Kind | Gold | Baseline Match | Funnel Match | Baseline Gold LP | Funnel Gold LP | Delta (Base-Funnel) | First Gen Divergence | Funnel Note |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| 1 | control | `1648297` | 1 | 1 | -0.2605 | -0.1325 | -0.1280 | 6 | both match |
| 64 | control | `6655273` | 1 | 1 | -0.2355 | -0.2777 | 0.0421 | 6 | both match |
| 128 | control | `4906794` | 1 | 1 | -0.2931 | -0.2497 | -0.0434 | 6 | both match |
| 180 | miss | `8732533` | 1 | 1 | -0.5338 | -0.2205 | -0.3133 | 6 | both match |
| 192 | control | `8626962` | 1 | 1 | -0.7882 | -0.1407 | -0.6476 | 1 | both match |
| 256 | control | `8675774` | 1 | 1 | -0.2652 | -0.3518 | 0.0867 | - | both match |
| 263 | miss | `5835658` | 1 | 1 | -0.1248 | -0.0352 | -0.0896 | - | both match |
| 320 | control | `6597984` | 1 | 1 | -0.2527 | -0.1785 | -0.0742 | 6 | both match |
| 384 | control | `5391746` | 1 | 1 | -0.3527 | -0.2270 | -0.1257 | 6 | both match |
| 388 | miss | `7816559` | 1 | 1 | -0.3962 | -0.2064 | -0.1897 | 1 | both match |
| 448 | control | `2736229` | 1 | 1 | -0.1838 | -0.3349 | 0.1511 | 12 | both match |
| 487 | miss | `9266754` | 1 | 1 | -0.0682 | -0.0268 | -0.0414 | 6 | both match |


## Miss Details

### Sample 180

- Gold answer: `8732533`
- Baseline generated: `8732533. \nThe grass is green. The sky is blue.` (match)
- Funnel generated: `8732533.` (match)
- Gold logprob delta: `-0.3133`
- First generation divergence: `6`
- First gold top-1 divergence: `none`
- First funnel gold token missing from top-20: `none`
- First funnel top-1 mismatch vs gold: `none`

### Sample 263

- Gold answer: `5835658`
- Baseline generated: `5835658.` (match)
- Funnel generated: `5835658.` (match)
- Gold logprob delta: `-0.0896`
- First generation divergence: `none`
- First gold top-1 divergence: `none`
- First funnel gold token missing from top-20: `none`
- First funnel top-1 mismatch vs gold: `none`

### Sample 388

- Gold answer: `7816559`
- Baseline generated: `**7816559**. \n\nThe text states: "One of the special magic` (match)
- Funnel generated: `7816559.` (match)
- Gold logprob delta: `-0.1897`
- First generation divergence: `1`
- First gold top-1 divergence: `none`
- First funnel gold token missing from top-20: `none`
- First funnel top-1 mismatch vs gold: `none`

### Sample 487

- Gold answer: `9266754`
- Baseline generated: `9266754.` (match)
- Funnel generated: `9266754. \n\nThe text contains a repetitive pattern of "The grass` (match)
- Gold logprob delta: `-0.0414`
- First generation divergence: `6`
- First gold top-1 divergence: `none`
- First funnel gold token missing from top-20: `none`
- First funnel top-1 mismatch vs gold: `none`


## Notes

- Gold scoring uses iterative one-token replay over the tokenizer pieces for `prompt + " " + target`.
- Matching uses the same normalization as the RULER task: trim, replace control characters with newlines, then substring-match the gold answer.
- This report is replay-based. The earlier 500-sample baseline and funnel benchmark outputs are not prompt-aligned, so this analysis uses one fixed prompt set from the funnel artifact for both server modes.
