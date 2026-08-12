# DeepSeek V4 Flash on `hh-b200` - RULER `niah_single_1` 128k Logprob Analysis

## Summary

- Prompt source: `/home/yiliu7/workspace/mini-swe-agent-fork/output/ruler_niah_single_1_128k_funnel_turbo_full_20260804_020151_retry/DeepSeek-V4-Flash/samples_niah_single_1_2026-08-04T02-21-11.771347.jsonl`
- Replay set: `10` prompts (`0` misses + `10` controls)
- Serving comparison: baseline default top-k vs funnel `funnel_dense` + `turbo`
- Shared prompt set: yes
- Local tunnel: `127.0.0.1:18004 -> hh-b200:127.0.0.1:8001`


## Baseline Default Top-k

- Remote log: `/tmp/deepseek_logprob_baseline_8001.log`
- Generated matches: `10/10`
- Mean gold logprob: `-0.3169`



## Funnel Dense Turbo Top-k

- Remote log: `/tmp/deepseek_logprob_funnel_turbo_8001.log`
- Generated matches: `10/10`
- Mean gold logprob: `-0.2468`

- Funnel markers confirmed: `confirmed`



## Per-Prompt Comparison

| Id | Kind | Gold | Baseline Match | Funnel Match | Baseline Gold LP | Funnel Gold LP | Delta (Base-Funnel) | First Gen Divergence | Funnel Note |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| 1 | control | `1648297` | 1 | 1 | -0.1798 | -0.1325 | -0.0473 | - | both match |
| 64 | control | `6655273` | 1 | 1 | -0.1708 | -0.2777 | 0.1069 | - | both match |
| 128 | control | `4906794` | 1 | 1 | -0.2536 | -0.2497 | -0.0039 | 6 | both match |
| 192 | control | `8626962` | 1 | 1 | -0.8019 | -0.1407 | -0.6612 | 6 | both match |
| 256 | control | `8675774` | 1 | 1 | -0.3365 | -0.3518 | 0.0154 | 6 | both match |
| 320 | control | `6597984` | 1 | 1 | -0.1013 | -0.1460 | 0.0446 | - | both match |
| 384 | control | `5391746` | 1 | 1 | -0.2537 | -0.2270 | -0.0267 | - | both match |
| 448 | control | `2736229` | 1 | 1 | -0.3981 | -0.3349 | -0.0632 | - | both match |
| 480 | control | `4687198` | 1 | 1 | -0.2948 | -0.3594 | 0.0646 | 6 | both match |
| 500 | control | `7447431` | 1 | 1 | -0.3783 | -0.2480 | -0.1303 | 6 | both match |

## Table Columns

- `Id`: sample index from the saved RULER `samples_*.jsonl` file.
- `Kind`: whether the sample was selected as a `control` or a `miss` replay case.
- `Gold`: the reference answer string for that sample.
- `Baseline Match`: `1` if the baseline free-generation response contained the gold answer after normalization, else `0`.
- `Funnel Match`: `1` if the funnel free-generation response contained the gold answer after normalization, else `0`.
- `Baseline Gold LP`: summed logprob of the gold target tokens under the baseline server, computed by iterative one-token replay.
- `Funnel Gold LP`: summed logprob of the same gold target tokens under the funnel server, computed the same way.
- `Delta (Base-Funnel)`: `Baseline Gold LP - Funnel Gold LP`. Negative means funnel assigned higher probability to the gold continuation; positive means baseline did.
- `First Gen Divergence`: first generated token position where baseline and funnel free-generation outputs differed. `-` means their generated token sequences matched for the compared span.
- `Funnel Note`: short qualitative summary for that row, such as `both match`, `baseline-only match`, `funnel-only match`, or `both miss`.


## Miss Details


## Method

- This report is replay-based against the live vLLM server, not extracted from the original benchmark artifact.
- For each selected sample, the script reuses the saved 128k prompt from the funnel run artifact and queries two serving modes separately: baseline default top-k and funnel `funnel_dense` + `turbo`.
- Generated-output collection uses one `/v1/completions` request per sample per mode with `temperature=0`, `max_tokens=16`, and `logprobs=20`.
- Gold logprob collection uses iterative one-token replay. The script tokenizes `" " + target` into tokenizer pieces, then sends one `/v1/completions` request with `max_tokens=1` and `logprobs=20` for each next gold piece.
- For this 10-control run, every target tokenized into `4` pieces, so gold replay used `10 x 4 x 2 = 80` one-token requests and produced `80` replay tokens total.
- Generated-output collection used `20` requests total and produced `130` baseline tokens plus `120` funnel tokens.
- Total generated tokens served during this collection were `170` for baseline, `160` for funnel, and `330` combined.

## Worked Example

- `Baseline Gold LP` is the sum of the baseline server's per-token logprobs for the gold target sequence.
- Sample `1` has gold target `1648297`, which tokenized as `[" ", "164", "829", "7"]`.
- The baseline replay collected these token logprobs:
  - `" "`: `-0.1778528`
  - `"164"`: `-0.0017735`
  - `"829"`: `-0.0000738`
  - `"7"`: `-0.0000709`
- Summed together:

```text
Baseline Gold LP
= log P(" " | prompt)
+ log P("164" | prompt + " ")
+ log P("829" | prompt + " 164")
+ log P("7" | prompt + " 164829")
= -0.1778528 + -0.0017735 + -0.0000738 + -0.0000709
= -0.1798
```

- That final `-0.1798` is the table entry shown in `Baseline Gold LP` for sample `1`.
- The corresponding funnel score for the same gold sequence was `-0.1325`, which is closer to `0` and therefore indicates higher assigned probability to that gold continuation.

## Notes

- Gold scoring uses iterative one-token replay over the tokenizer pieces for `prompt + " " + target`.
- Matching uses the same normalization as the RULER task: trim, replace control characters with newlines, then substring-match the gold answer.
- This report is replay-based. The earlier 500-sample baseline and funnel benchmark outputs are not prompt-aligned, so this analysis uses one fixed prompt set from the funnel artifact for both server modes.
