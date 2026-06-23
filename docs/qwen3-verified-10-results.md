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
