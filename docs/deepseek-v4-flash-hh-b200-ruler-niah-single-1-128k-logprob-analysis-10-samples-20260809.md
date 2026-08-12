# DeepSeek V4 Flash on `hh-b200` - RULER `niah_single_1` 128k Logprob Analysis

## Summary

- Prompt source: `/home/yiliu7/workspace/mini-swe-agent-fork/output/ruler_niah_single_1_128k_funnel_turbo_full_20260804_020151_retry/DeepSeek-V4-Flash/samples_niah_single_1_2026-08-04T02-21-11.771347.jsonl`
- Replay set: `10` prompts (`4` misses + `6` controls)
- Serving comparison: baseline default top-k vs funnel `funnel_dense` + `turbo`
- Shared prompt set: yes
- Local tunnel: `127.0.0.1:18002 -> hh-b200:127.0.0.1:8001`


## Baseline Default Top-k

- Remote log: `/tmp/deepseek_logprob_baseline_8001.log`
- Generated matches: `10/10`
- Mean gold logprob: `-0.2813`



## Funnel Dense Turbo Top-k

- Remote log: `/tmp/deepseek_logprob_funnel_turbo_8001.log`
- Generated matches: `10/10`
- Mean gold logprob: `-0.1820`

- Funnel markers confirmed: `confirmed`



## Per-Prompt Comparison

| Id | Kind | Gold | Baseline Match | Funnel Match | Baseline Gold LP | Funnel Gold LP | Delta (Base-Funnel) | First Gen Divergence | Funnel Note |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| 1 | control | `1648297` | 1 | 1 | -0.1772 | -0.1325 | -0.0447 | 7 | both match |
| 64 | control | `6655273` | 1 | 1 | -0.3232 | -0.2777 | -0.0455 | - | both match |
| 128 | control | `4906794` | 1 | 1 | -0.1592 | -0.2497 | 0.0905 | 7 | both match |
| 180 | miss | `8732533` | 1 | 1 | -0.3806 | -0.2205 | -0.1601 | 6 | both match |
| 192 | control | `8626962` | 1 | 1 | -0.5396 | -0.1407 | -0.3989 | - | both match |
| 256 | control | `8675774` | 1 | 1 | -0.2231 | -0.3518 | 0.1288 | - | both match |
| 263 | miss | `5835658` | 1 | 1 | -0.0654 | -0.0352 | -0.0302 | - | both match |
| 320 | control | `6597984` | 1 | 1 | -0.2489 | -0.1785 | -0.0704 | 6 | both match |
| 388 | miss | `7816559` | 1 | 1 | -0.6522 | -0.2064 | -0.4458 | - | both match |
| 487 | miss | `9266754` | 1 | 1 | -0.0438 | -0.0268 | -0.0169 | 8 | both match |


## Miss Details

### Sample 180

- Gold answer: `8732533`
- Baseline generated: `8732533. \nThe grass is green. The sky is blue.` (match)
- Funnel generated: `8732533.` (match)
- Gold logprob delta: `-0.1601`
- First generation divergence: `6`
- First gold top-1 divergence: `none`
- First funnel gold token missing from top-20: `none`
- First funnel top-1 mismatch vs gold: `none`

### Sample 263

- Gold answer: `5835658`
- Baseline generated: `5835658.` (match)
- Funnel generated: `5835658.` (match)
- Gold logprob delta: `-0.0302`
- First generation divergence: `none`
- First gold top-1 divergence: `none`
- First funnel gold token missing from top-20: `none`
- First funnel top-1 mismatch vs gold: `none`

### Sample 388

- Gold answer: `7816559`
- Baseline generated: `7816559.` (match)
- Funnel generated: `7816559.` (match)
- Gold logprob delta: `-0.4458`
- First generation divergence: `none`
- First gold top-1 divergence: `none`
- First funnel gold token missing from top-20: `none`
- First funnel top-1 mismatch vs gold: `none`

### Sample 487

- Gold answer: `9266754`
- Baseline generated: `9266754. \n\nThe text contains a repetitive pattern of "The grass` (match)
- Funnel generated: `9266754. \n\nThe grass is green. The sky is blue.` (match)
- Gold logprob delta: `-0.0169`
- First generation divergence: `8`
- First gold top-1 divergence: `none`
- First funnel gold token missing from top-20: `none`
- First funnel top-1 mismatch vs gold: `none`


## Notes

- Gold scoring uses iterative one-token replay over the tokenizer pieces for `prompt + " " + target`.
- Matching uses the same normalization as the RULER task: trim, replace control characters with newlines, then substring-match the gold answer.
- This report is replay-based. The earlier 500-sample baseline and funnel benchmark outputs are not prompt-aligned, so this analysis uses one fixed prompt set from the funnel artifact for both server modes.