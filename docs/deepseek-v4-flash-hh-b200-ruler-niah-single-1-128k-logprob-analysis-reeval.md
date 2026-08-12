# DeepSeek V4 Flash on `hh-b200` - RULER `niah_single_1` 128k Logprob Analysis

## Summary

- Prompt source: `/home/yiliu7/workspace/mini-swe-agent-fork/output/ruler_niah_single_1_128k_funnel_turbo_reeval_20260804_045912/DeepSeek-V4-Flash/samples_niah_single_1_2026-08-04T05-18-15.394991.jsonl`
- Replay set: `11` prompts (`3` misses + `8` controls)
- Serving comparison: baseline default top-k vs funnel `funnel_dense` + `turbo`
- Shared prompt set: yes
- Local tunnel: `127.0.0.1:18001 -> hh-b200:127.0.0.1:8001`


## Baseline Default Top-k

- Remote log: `/tmp/deepseek_logprob_baseline_8001.log`
- Generated matches: `11/11`
- Mean gold logprob: `-0.2628`



## Funnel Dense Turbo Top-k

- Remote log: `/tmp/deepseek_logprob_funnel_turbo_8001.log`
- Generated matches: `11/11`
- Mean gold logprob: `-0.2183`

- Funnel markers confirmed: `confirmed`



## Per-Prompt Comparison

| Id | Kind | Gold | Baseline Match | Funnel Match | Baseline Gold LP | Funnel Gold LP | Delta (Base-Funnel) | First Gen Divergence | Funnel Note |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| 1 | control | `1648297` | 1 | 1 | -0.2873 | -0.1325 | -0.1548 | 6 | both match |
| 54 | miss | `7301168` | 1 | 1 | -0.1471 | -0.1284 | -0.0187 | 6 | both match |
| 64 | control | `6655273` | 1 | 1 | -0.2486 | -0.2175 | -0.0311 | 6 | both match |
| 128 | control | `4906794` | 1 | 1 | -0.1307 | -0.2497 | 0.1190 | 6 | both match |
| 192 | control | `8626962` | 1 | 1 | -0.2384 | -0.3716 | 0.1332 | - | both match |
| 256 | control | `8675774` | 1 | 1 | -0.4168 | -0.3518 | -0.0650 | 6 | both match |
| 283 | miss | `5780883` | 1 | 1 | -0.1261 | -0.0951 | -0.0310 | 6 | both match |
| 320 | control | `6597984` | 1 | 1 | -0.1098 | -0.1147 | 0.0048 | 6 | both match |
| 384 | control | `5391746` | 1 | 1 | -0.5341 | -0.2270 | -0.3071 | 6 | both match |
| 409 | miss | `2547554` | 1 | 1 | -0.4363 | -0.1783 | -0.2580 | 6 | both match |
| 448 | control | `2736229` | 1 | 1 | -0.2157 | -0.3349 | 0.1192 | 6 | both match |


## Miss Details

### Sample 54

- Gold answer: `7301168`
- Baseline generated: `7301168.` (match)
- Funnel generated: `7301168. The special magic number for exuberant-bicycle mentioned in` (match)
- Gold logprob delta: `-0.0187`
- First generation divergence: `6`
- First gold top-1 divergence: `none`
- First funnel gold token missing from top-20: `none`
- First funnel top-1 mismatch vs gold: `none`

### Sample 283

- Gold answer: `5780883`
- Baseline generated: `5780883. \n\nThe text states: "One of the special magic` (match)
- Funnel generated: `5780883.` (match)
- Gold logprob delta: `-0.0310`
- First generation divergence: `6`
- First gold top-1 divergence: `none`
- First funnel gold token missing from top-20: `none`
- First funnel top-1 mismatch vs gold: `none`

### Sample 409

- Gold answer: `2547554`
- Baseline generated: `2547554. The grass is green. The sky is blue. The` (match)
- Funnel generated: `2547554. \n\nThe grass is green. The sky is blue.` (match)
- Gold logprob delta: `-0.2580`
- First generation divergence: `6`
- First gold top-1 divergence: `none`
- First funnel gold token missing from top-20: `none`
- First funnel top-1 mismatch vs gold: `none`


## Notes

- Gold scoring uses iterative one-token replay over the tokenizer pieces for `prompt + " " + target`.
- Matching uses the same normalization as the RULER task: trim, replace control characters with newlines, then substring-match the gold answer.
- This report is replay-based. The earlier 500-sample baseline and funnel benchmark outputs are not prompt-aligned, so this analysis uses one fixed prompt set from the funnel artifact for both server modes.