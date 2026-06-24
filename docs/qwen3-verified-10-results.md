# Qwen3 35B-A3B-FP8 — SWE-bench Verified 10-Instance Run

## Run Summary

| | |
|---|---|
| Date | 2026-06-22 |
| Dataset | SWE-bench Verified (test split), first 10 instances (astropy repo) |
| Model | Qwen3.6-35B-A3B-FP8 via vLLM |
| Config | `swebench.yaml` + `swebench_vllm.yaml` |
| Wall time | 41 min 31 sec |
| Output | `output/qwen3_verified_10_run2_20260622_151346/` |

## Results at a Glance

| Metric | Value |
|--------|-------|
| Instances run | 10 |
| Patches submitted | 4 (40%) |
| Evaluated & resolved | 2 (20%) |
| Total agent messages | 1,888 |
| Total agent actions (tool calls) | 923 |

## Per-Instance Breakdown

| instance | msgs | actions | patch | exit status | runtime |
|---|---|---|---|---|---|
| astropy__astropy-12907 | 339 | 168 | 0 | ContextWindowExceededError | 18:55 |
| astropy__astropy-13033 | 10 | 2 | 0 | RepeatedFormatError | <1s |
| astropy__astropy-13236 | 503 | 250 | 0 | LimitsExceeded | 4:01 |
| **astropy__astropy-13398** | 172 | 83 | 4126 chars | Submitted | 3:36 |
| **astropy__astropy-13453** ✓ | 135 | 66 | 1995 chars | Submitted | 4:45 |
| astropy__astropy-13579 | 22 | 8 | 0 | RepeatedFormatError | 0:12 |
| **astropy__astropy-13977** | 111 | 54 | 927 chars | Submitted | 1:28 |
| astropy__astropy-14096 | 17 | 5 | 0 | RepeatedFormatError | 3:31 |
| astropy__astropy-14182 | 503 | 250 | 0 | LimitsExceeded | 2:52 |
| **astropy__astropy-14309** ✓ | 76 | 37 | 1619 chars | Submitted | 0:48 |

✓ = resolved by SWE-bench evaluation

## Exit Statuses Explained

### `Submitted`
The agent **produced a patch and submitted it**. The model reached the submission step in the agent loop — it created a `patch.txt` via `git diff` and ran the exact command:
```bash
echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt
```
The environment detected the magic string `COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`, extracted the patch, and the agent exited the loop with `exit_status: Submitted`.

**Important**: `Submitted` does **not** mean the patch is correct. It only means the model *believed* it was done and followed the submission protocol. Whether the patch actually fixes the issue is determined by the **evaluation step** — applying the patch in a fresh container and running the `FAIL_TO_PASS` / `PASS_TO_PASS` tests.

In this run: 4 Submitted → 2 resolved (50% of submitted patches were correct).

### `RepeatedFormatError`
The model produced 3 consecutive responses with **no parseable tool call**. Each time, the agent appended a format error message and asked the model to retry. After 3 failures in a row, the agent gives up. This is the #1 failure mode for thinking models like Qwen3 — the model gets lost in reasoning and forgets to produce the required code block or tool call JSON.

### `LimitsExceeded`
The agent hit the **250-step limit** before producing a submission. The model was still actively working (13236 made 250 tool calls, 14182 made 250) but ran out of budget. Bumping `step_limit` could let these finish, at the cost of more wall time.

### `ContextWindowExceededError`
After 168 actions, the accumulated message history + model response exceeded the 262K-token vLLM context window. The next API call failed with a 400 error. There is no recovery — the context window is a hard limit of the model server.

## Evaluation Details

Evaluated with `swebench.harness.run_evaluation`:

| instance | patch size | evaluation | result |
|---|---|---|---|
| astropy__astropy-13398 | 4126 chars | tests failed | unresolved |
| astropy__astropy-13453 | 1995 chars | tests passed | ✓ resolved |
| astropy__astropy-13977 | 927 chars | tests failed | unresolved |
| astropy__astropy-14309 | 1619 chars | tests passed | ✓ resolved |

## Trajectory Files

Each instance has a full `.traj.json` file recording every step:
- Every model response (including `reasoning_content`)
- Every tool call with full command arguments
- Every command output (returncode + stdout/stderr)
- Per-step cost and timestamp

Access with:
```bash
mini-extra inspect output/qwen3_verified_10_run2_20260622_151346/
```

---

# Run 3: Fixes Applied (2026-06-23)

## Changes from Run 2

| Parameter | Run 2 | Run 3 |
|-----------|-------|-------|
| `temperature` | 0.0 | 0.6 |
| `preserve_thinking` | not set | true (via `extra_body`) |
| `tool_choice` | required | required |
| `max_tokens` | 49152 | 49152 |

Config: `swebench_vllm.yaml` updated. vLLM server: GPU 3, `--gpu-memory-utilization 0.92`.

## Run Summary

| | |
|---|---|
| Date | 2026-06-23 |
| Dataset | SWE-bench Verified (test split), first 10 instances |
| Model | Qwen3.6-35B-A3B-FP8 via vLLM |
| Config | `swebench.yaml` + `swebench_vllm.yaml` (updated) |
| Output | `output/qwen3_verified_10_run3_20260623_062025/` |

## Results at a Glance

| Metric | Run 2 | Run 3 |
|--------|-------|-------|
| Instances completed | 10 | 7 |
| Patches submitted | 4 (40%) | 5 (50%) |
| Evaluated & resolved | 2 (20%) | 2 (20%) |
| Avg actions (submitted) | 60 | 47 |

Run hung on instance 8 (astropy-14096) — with `preserve_thinking`, accumulated reasoning bloat makes later turns extremely slow. Only 7/10 completed.

## Per-Instance Breakdown

| instance | msgs | actions | patch | exit status | evaluated |
|---|---|---|---|---|---|
| astropy__astropy-12907 | 49 | 24 | 504 chars | Submitted | **✓ RESOLVED** |
| astropy__astropy-13033 | 39 | 19 | 933 chars | Submitted | ✗ unresolved |
| astropy__astropy-13236 | 143 | 71 | 1016 chars | Submitted | ✗ unresolved |
| astropy__astropy-13398 | 155 | 77 | 4046 chars | Submitted | ✗ unresolved |
| astropy__astropy-13453 | 85 | 42 | 631 chars | Submitted | **✓ RESOLVED** |
| astropy__astropy-13579 | 22 | 10 | 0 | RepeatedFormatError | — |
| astropy__astropy-13977 | 2 | 0 | 0 | RepeatedFormatError | — |
| astropy__astropy-14096 | — | — | — | **hung** (never completed) | — |
| astropy__astropy-14182 | — | — | — | did not reach | — |
| astropy__astropy-14309 | — | — | — | did not reach | — |

## Comparison: Run 2 vs Run 3

```
Instance              Run 2 (before)                  Run 3 (after)
astropy-12907         ContextWindowExceeded (168a)    Submitted (24a)  → ✓ RESOLVED
astropy-13033         RepeatedFormatError (2a)        Submitted (19a)  → ✗
astropy-13236         LimitsExceeded (250a)           Submitted (71a)  → ✗
astropy-13398         Submitted (83a)                 Submitted (77a)  → ✗
astropy-13453         Submitted (66a)                 Submitted (42a)  → ✓ RESOLVED
astropy-13579         RepeatedFormatError (8a)        RepeatedFormatError (10a)
astropy-13977         Submitted (54a)                 RepeatedFormatError (0a) — REGRESSION
astropy-14096         RepeatedFormatError (5a)        HUNG (agent stuck, run aborted)
astropy-14182         LimitsExceeded (250a)           did not reach
astropy-14309         Submitted (37a) → ✓             did not reach
```

## Key Observations

- **preserve_thinking dramatically reduces actions**: 12907 went from 168→24, 13236 from 250→71
- **But causes context bloat**: After ~30 turns, reasoning content dominates the input, making each turn minutes-long. The run hung on instance 8.
- **temperature=0.6 is a double-edged sword**: Prevents deterministic failure loops but can cause first-turn format errors (13977 got 0 actions)
- **bash-only agent is the fundamental bottleneck**: Even with perfect config, the model produces inaccurate patches because it lacks file-edit tools. This is the primary gap vs Qwen's official 73.4%.

---

# Run 5: Winning Config — temp=1.0 + top_p=0.95 (2026-06-23)

## Changes from Run 3

| Parameter | Run 3 | Run 5 |
|-----------|-------|-------|
| `temperature` | 0.6 | **1.0** |
| `top_p` | not set | **0.95** |
| `preserve_thinking` | true | true |
| `tool_choice` | required | required |
| GPU config | single GPU (500 tok/s) | TP4 across 4 GPUs (245 tok/s) |

Config matching Qwen's official SWE-bench settings.

## Run Summary

| | |
|---|---|
| Date | 2026-06-23 |
| Dataset | SWE-bench Verified (test split), first 10 instances |
| Model | Qwen3.6-35B-A3B-FP8 via vLLM TP4 |
| Output | `output/qwen3_35b_tp4_10_20260623_161314/` |
| Wall time | ~2.5 hours |

## Results at a Glance

| Metric | Run 2 (temp=0.0) | Run 3 (temp=0.6) | **Run 5 (temp=1.0)** |
|--------|-----------------|-------------------|---------------------|
| Instances completed | 10/10 | 7/10 | 10/10 |
| Patches submitted | 4 (40%) | 5 (50%) | **9 (90%)** |
| Resolved | 2 (20%) | 2 (20%) | **5 (50%)** |
| Avg actions (submitted) | 60 | 47 | 49 |

## Per-Instance Breakdown

| instance | actions | patch | exit status | evaluated |
|---|---|---|---|---|
| astropy__astropy-12907 | 22 | 504 chars | Submitted | **✓ RESOLVED** |
| astropy__astropy-13033 | 30 | 1441 chars | Submitted | ✗ unresolved |
| astropy__astropy-13236 | 46 | 1923 chars | Submitted | **✓ RESOLVED** |
| astropy__astropy-13398 | 61 | 5460 chars | Submitted | ✗ unresolved |
| astropy__astropy-13453 | 54 | 408 chars | Submitted | **✓ RESOLVED** |
| astropy__astropy-13579 | 39 | 0 | RepeatedFormatError | — |
| astropy__astropy-13977 | 57 | 897 chars | Submitted | ✗ unresolved |
| astropy__astropy-14096 | 56 | 1377 chars | Submitted | **✓ RESOLVED** |
| astropy__astropy-14182 | 57 | 1102 chars | Submitted | ✗ unresolved |
| astropy__astropy-14309 | 26 | 556 chars | Submitted | **✓ RESOLVED** |

## Comparison: All Runs

```
Instance              Run 2 (temp=0.0)  Run 3 (temp=0.6)  Run 5 (temp=1.0)
astropy-12907         ✗ CE              ✓                  ✓ RESOLVED
astropy-13033         ✗ RF              ✗                  ✗
astropy-13236         ✗ LE              ✗                  ✓ RESOLVED ← first time!
astropy-13398         ✗                 ✗                  ✗
astropy-13453         ✓                 ✓                  ✓ RESOLVED
astropy-13579         ✗ RF              ✗ RF               ✗ RF (unfixable)
astropy-13977         ✗                 ✗ RF               ✗ (regressed from R2)
astropy-14096         ✗ RF              — (hung)           ✓ RESOLVED ← first time!
astropy-14182         ✗ LE              — (didn't reach)   ✗ (first patch)
astropy-14309         ✓                 — (didn't reach)   ✓ RESOLVED
```

## Why temp=1.0 Works

The thinking explosion (Qwen3 exhausts 49K tokens in reasoning before producing a tool call) still occurs at temp=1.0. But higher sampling diversity means the model *sometimes* terminates reasoning earlier by chance, leaving room for the tool call.

At temp=0.0: model deterministically thinks to exhaustion on certain instances → format error.
At temp=1.0: stochastic sampling occasionally produces a shorter reasoning chain → successful tool call.

Effect: converts ~50% of thinking-explosion failures into successes through sampling diversity. Tradeoff: slightly more actions per instance and larger patches (model explores more in its reasoning).

## Key Insight

**temp=1.0 + top_p=0.95 is the single most impactful config change for Qwen3** — worth 30 percentage points (20% → 50%) with no other changes.

---

# Run 6: DeepSeek V4 Flash (2026-06-24)

## Config

| Parameter | Value |
|-----------|-------|
| Model | DeepSeek V4 Flash (FP8, ~160B MoE) |
| vLLM | 0.23.1rc1, TP2 on GPUs 2,3 |
| `temperature` | 1.0 |
| `top_p` | 0.95 |
| `max_tokens` | 49152 |
| Tool parser | `deepseek_v4` (native) |

## Results

| Metric | Value |
|--------|-------|
| Instances completed | 10/10 |
| Patches submitted | **10 (100%)** |
| Resolved | **6 (60%)** |
| Avg actions (submitted) | 35 |
| Format errors | 0 |

## Per-Instance

| instance | actions | patch | evaluated | Qwen3 Run5 |
|---|---|---|---|---|
| astropy-12907 | 15 | 504c | ✓ | ✓ |
| astropy-13033 | 22 | 1235c | ✗ | ✗ |
| astropy-13236 | 48 | 797c | ✓ | ✓ |
| astropy-13398 | 37 | 3808c | ✗ | ✗ |
| astropy-13453 | 36 | 429c | ✓ | ✓ |
| astropy-13579 | 28 | 1724c | **✓** | ✗ RF |
| astropy-13977 | 41 | 774c | ✗ | ✗ |
| astropy-14096 | 51 | 3487c | ✓ | ✓ |
| astropy-14182 | 54 | 1056c | ✗ | ✗ |
| astropy-14309 | 18 | 597c | ✓ | ✓ |

## Full Comparison

```
              Qwen3(t0)  Qwen3(t1)  DeepSeek V4
-----------------------------------------------
12907          ✗ CE       ✓           ✓
13033          ✗ RF       ✗           ✗
13236          ✗ LE       ✓           ✓
13398          ✗          ✗           ✗
13453          ✓          ✓           ✓
13579          ✗ RF       ✗ RF        ✓ ← first time!
13977          ✗          ✗           ✗
14096          ✗ RF       ✓           ✓
14182          ✗ LE       ✗           ✗
14309          ✓          ✓           ✓
-----------------------------------------------
Resolved       20%        50%         60%
```

**Key takeaway**: DeepSeek V4's native tool-calling eliminates format errors entirely (100% submission), pushing the pass ratio to 60% — best of all models tested.

---

# Run 7: DeepSeek V4 Flash — Thinking Mode (2026-06-24)

## Correct Thinking Format

The thinking mode was initially tested with wrong format (`thinking: {type: enabled}` — silently ignored). Correct format:

```yaml
extra_body:
  chat_template_kwargs:
    thinking: true
    reasoning_effort: high   # or "max" (needs 384K+ context)
```

## Results

| Metric | No-think | Wrong Format | **Think High** |
|--------|----------|-------------|----------------|
| Resolved | 6/10 (60%) | 6/10 (60%) | **6/10 (60%)** |
| Submission | 100% | 100% | 100% |
| Avg actions | 35 | 38 | 32 |

## Per-Instance

All 3 runs resolve the exact same 6 instances:
- 12907, 13236, 13453, 13579, 14096, 14309

Same 4 unresolved across all runs:
- 13033, 13398, 13977, 14182

## Conclusion

**Thinking mode does not improve DeepSeek V4 Flash on SWE-bench.** The bottleneck is the bash-only agent scaffold. Non-think mode is the BKC (simpler, faster, same accuracy).

## Final Model Comparison

| Model | Best Run | Submission | Resolved | Format Errors |
|-------|---------|------------|----------|---------------|
| Qwen3 35B (temp=0.0) | Run 2 | 40% | 20% | 3 |
| Qwen3 35B (temp=0.6) | Run 3 | 71% | 20% | 2 |
| Qwen3 35B (temp=1.0) | Run 5 | 90% | 50% | 1 |
| Qwen3 27B | Run 4 | 20% | 10% | 2 |
| **DeepSeek V4 Flash** | **Run 6** | **100%** | **60%** | **0** |
