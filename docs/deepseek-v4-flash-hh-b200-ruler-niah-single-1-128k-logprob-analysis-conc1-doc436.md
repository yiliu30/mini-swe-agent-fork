# DeepSeek V4 Flash on `hh-b200` - RULER `niah_single_1` 128k Logprob Analysis

## Summary

- Prompt source: `/home/yiliu7/workspace/mini-swe-agent-fork/output/ruler_niah_single_1_128k_funnel_turbo_full_conc1_20260804_185001/DeepSeek-V4-Flash/samples_niah_single_1_2026-08-04T19-17-13.292506.jsonl`
- Replay set: `1` prompts (`1` misses + `0` controls)
- Serving comparison: baseline default top-k vs funnel `funnel_dense` + `turbo`
- Shared prompt set: yes
- Local tunnel: `127.0.0.1:18001 -> hh-b200:127.0.0.1:8001`


## Baseline Default Top-k

- Remote log: `/tmp/deepseek_logprob_baseline_8001.log`
- Generated matches: `1/1`
- Mean gold logprob: `-0.0584`



## Funnel Dense Turbo Top-k

- Remote log: `/tmp/deepseek_logprob_funnel_turbo_8001.log`
- Generated matches: `1/1`
- Mean gold logprob: `-0.1990`

- Funnel markers confirmed: `confirmed`



## Per-Prompt Comparison

| Id | Kind | Gold | Baseline Match | Funnel Match | Baseline Gold LP | Funnel Gold LP | Delta (Base-Funnel) | First Gen Divergence | Funnel Note |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| 437 | miss | `7109499` | 1 | 1 | -0.0584 | -0.1990 | 0.1406 | 6 | both match |


## Miss Details

### Sample 437

- Gold answer: `7109499`
- Baseline generated: `7109499.` (match)
- Funnel generated: `7109499. \n\nThe special magic number for earthy-pig mentioned in` (match)
- Gold logprob delta: `0.1406`
- First generation divergence: `6`
- First gold top-1 divergence: `none`
- First funnel gold token missing from top-20: `none`
- First funnel top-1 mismatch vs gold: `none`


## Notes

- Gold scoring uses iterative one-token replay over the tokenizer pieces for `prompt + " " + target`.
- Matching uses the same normalization as the RULER task: trim, replace control characters with newlines, then substring-match the gold answer.
- This report is replay-based. The earlier 500-sample baseline and funnel benchmark outputs are not prompt-aligned, so this analysis uses one fixed prompt set from the funnel artifact for both server modes.