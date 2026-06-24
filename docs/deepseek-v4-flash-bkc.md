# DeepSeek V4 Flash via Local vLLM — Best Known Configuration

Running SWE-bench with DeepSeek V4 Flash served by local vLLM.

## 1. Model

- **Path**: `/storage/yiliu7/deepseek-ai/DeepSeek-V4-Flash`
- **Size**: ~160 GB (46 shards at FP8), ~74 GiB per GPU with TP2
- **Architecture**: 43 layers, 4096 hidden, 64 attn heads, 1 KV head, MoE (6 experts/tok)
- **Context**: 1M native, capped at 128K for KV cache headroom
- **Key advantage**: Native tool-calling via `deepseek_v4` parser — no thinking explosions

## 2. Start the vLLM Server

```bash
CUDA_HOME=/usr/local/cuda-13 \
CUDA_VISIBLE_DEVICES=2,3 \
/home/yiliu7/workspace/venvs/vllm/bin/vllm serve \
  /storage/yiliu7/deepseek-ai/DeepSeek-V4-Flash \
  --trust-remote-code \
  --kv-cache-dtype fp8 \
  --block-size 256 \
  --tensor-parallel-size 2 \
  --max-model-len 131072 \
  --attention_config.use_fp4_indexer_cache=True \
  --tokenizer-mode deepseek_v4 \
  --tool-call-parser deepseek_v4 \
  --enable-auto-tool-choice \
  --reasoning-parser deepseek_v4 \
  --port 8000
```

| Flag | Why |
|------|-----|
| `CUDA_HOME=/usr/local/cuda-13` | Match CUDA toolkit to vllm venv headers (CUDA 13) |
| `/home/yiliu7/workspace/venvs/vllm/bin/vllm` | Use vLLM 0.23.1rc1 dev build (required for DeepSeek V4) |
| `--tokenizer-mode deepseek_v4` | DeepSeek V4-specific tokenizer |
| `--tool-call-parser deepseek_v4` | **Critical.** Native tool-calling parser — eliminates thinking explosion |
| `--reasoning-parser deepseek_v4` | Extracts reasoning into `reasoning_content` |
| `--tensor-parallel-size 2` | Model fits across 2 GPUs (74 GiB each) |
| `--max-model-len 131072` | Cap context at 128K (default 1M would OOM) |
| `--kv-cache-dtype fp8` | FP8 KV cache saves memory |
| `--attention_config.use_fp4_indexer_cache=True` | FP4 indexer cache for MoE routing |

**GPU**: 2 GPUs needed (B200, 183 GB each). Model uses ~74 GiB per GPU. Startup takes ~30 minutes first time (JIT kernel compilation).

## 3. Config File

`src/minisweagent/config/benchmarks/swebench_deepseek_local.yaml`:

```yaml
# DeepSeek V4 Flash via local vLLM model overrides for SWE-bench.
# Merge on top of swebench.yaml via: -c swebench.yaml -c swebench_deepseek_local.yaml
model:
  model_name: "hosted_vllm//storage/yiliu7/deepseek-ai/DeepSeek-V4-Flash"
  cost_tracking: "ignore_errors"
  model_kwargs:
    api_base: "http://localhost:8000/v1"
    drop_params: true
    max_tokens: 49152
    temperature: 1.0
    top_p: 0.95
    timeout: 900000
```

### Why each setting matters

| Setting | Why |
|---------|-----|
| `hosted_vllm/` | litellm provider for local vLLM servers |
| `cost_tracking: "ignore_errors"` | Local model not in litellm cost registry |
| `temperature: 1.0` | Match Qwen official SWE-bench settings; higher diversity |
| `top_p: 0.95` | Qwen official setting |
| `max_tokens: 49152` | Sufficient for tool calls + reasoning |
| No `tool_choice` | DeepSeek V4 decides itself — native parser handles it reliably |
| No `preserve_thinking` | Not needed — DeepSeek V4's parser handles reasoning internally |

## 4. Run

**SWE-bench Verified (test split), 10 instances:**
```bash
source /home/yiliu7/workspace/venvs/omni/bin/activate
mini-extra swebench \
  --subset verified --split test --slice "0:10" \
  -c swebench.yaml -c swebench_deepseek_local.yaml \
  -c agent.mode=yolo -w 1 \
  -o ./output/deepseek_v4_10_$(date +%Y%m%d_%H%M%S)
```

**Full 500-instance run:**
```bash
mini-extra swebench \
  --subset verified --split test \
  -c swebench.yaml -c swebench_deepseek_local.yaml \
  -c agent.mode=yolo -w 1 \
  -o ./output/deepseek_v4_full_$(date +%Y%m%d_%H%M%S)
```

Resume-safe: re-run with same `-o` dir to skip completed instances.

## 5. Evaluate

```bash
python -c "
import json
preds = json.load(open('output/<run>/preds.json'))
with open('output/<run>/preds.jsonl','w') as f:
    for i,d in preds.items():
        if d.get('model_patch'): f.write(json.dumps(d)+'\n')
"

python -m swebench.harness.run_evaluation \
  --dataset_name princeton-nlp/SWE-Bench_Verified \
  --predictions_path output/<run>/preds.jsonl \
  --split test --max_workers 2 --run_id deepseek-v4
```

## 6. Benchmark Results (2026-06-24)

Tested on SWE-bench Verified, 10 instances. Full output at `output/deepseek_v4_10_20260624_035016/`.

### Results at a Glance

| Metric | Qwen3 Best (Run 5) | **DeepSeek V4** |
|--------|-------------------|-----------------|
| Instances completed | 10/10 | 10/10 |
| Patches submitted | 9 (90%) | **10 (100%)** |
| Resolved | 5 (50%) | **6 (60%)** |
| Avg actions (submitted) | 49 | **35** |
| Format errors | 1 | **0** |
| Wall time | ~2.5 hours | ~1.5 hours |

### Per-Instance Breakdown

| instance | actions | patch | status | Qwen3 Run5 |
|---|---|---|---|---|
| astropy__astropy-12907 | 15 | 504c | **✓ RESOLVED** | ✓ |
| astropy__astropy-13033 | 22 | 1235c | ✗ unresolved | ✗ |
| astropy__astropy-13236 | 48 | 797c | **✓ RESOLVED** | ✓ |
| astropy__astropy-13398 | 37 | 3808c | ✗ unresolved | ✗ |
| astropy__astropy-13453 | 36 | 429c | **✓ RESOLVED** | ✓ |
| astropy__astropy-13579 | 28 | 1724c | **✓ RESOLVED** | ✗ RF |
| astropy__astropy-13977 | 41 | 774c | ✗ unresolved | ✗ |
| astropy__astropy-14096 | 51 | 3487c | **✓ RESOLVED** | ✓ |
| astropy__astropy-14182 | 54 | 1056c | ✗ unresolved | ✗ |
| astropy__astropy-14309 | 18 | 597c | **✓ RESOLVED** | ✓ |

### Key Insight

**DeepSeek V4's native tool-calling (`deepseek_v4` parser) eliminates format errors entirely.** 100% submission rate with zero thinking explosions. Every API call produces a valid tool call — the model never gets stuck in reasoning-only loops.

The 4 unresolved instances (13033, 13398, 13977, 14182) fail because the bash-only agent scaffold produces incorrect patches, not because the model fails to use tools. These need file-edit tools or better prompt engineering.

### Newly Resolved vs Qwen3

- **astropy-13579**: Always failed with `RepeatedFormatError` on Qwen3 (all 4 runs). DeepSeek V4 submitted correctly and resolved it.
- No regressions vs Qwen3 Run 5.

## 7. Comparison: All Models

| Metric | Qwen3 35B (temp=0.0) | Qwen3 35B (temp=1.0) | **DeepSeek V4 Flash** |
|--------|---------------------|---------------------|----------------------|
| Submission rate | 40% | 90% | **100%** |
| Resolved | 20% | 50% | **60%** |
| Format errors | 3 | 1 | **0** |
| Avg actions | 60 | 49 | **35** |
| Gen speed | 500 tok/s | 245 tok/s | 28 tok/s |
| Per-instance time | ~4 min | ~12 min | ~9 min |
| GPU config | 1 GPU | 4 GPU (TP4) | 2 GPU (TP2) |

Despite generating at 28 tok/s (vs Qwen3's 500 tok/s), DeepSeek V4 completes instances faster because every token is productive — no wasted thinking cycles.

## 8. Thinking Mode Experiment (2026-06-24)

DeepSeek V4 Flash supports two thinking modes via `chat_template_kwargs`:

| Mode | Config |
|------|--------|
| Non-think (default) | no `extra_body` needed |
| Think High | `extra_body: {chat_template_kwargs: {thinking: true, reasoning_effort: high}}` |
| Think Max | `extra_body: {chat_template_kwargs: {thinking: true, reasoning_effort: max}}` (needs 384K+ context) |

**Important**: Thinking is NOT `thinking: {type: enabled}` — that format is silently ignored. It must be `chat_template_kwargs.thinking`.

### Thinking Mode Results

Tested Think High vs Non-think on 10 instances:

| Run | Thinking | Resolved | Submission | Actions |
|-----|----------|----------|------------|---------|
| No-think | None | **6/10 (60%)** | 100% | 35 |
| Think (wrong format) | `thinking: {type: enabled}` | 6/10 (60%) | 100% | 38 |
| Think High | `chat_template_kwargs.thinking=true` | **6/10 (60%)** | 100% | 32 |

**All three runs resolve the exact same 6 instances**: 12907, 13236, 13453, 13579, 14096, 14309.

The same 4 instances fail across all runs: 13033, 13398, 13977, 14182.

### Key Finding

**Thinking mode does not improve DeepSeek V4 Flash on SWE-bench Verified.** The bottleneck is the bash-only agent scaffold, not model reasoning capability. Thinking produces more verbose chain-of-thought but doesn't change which patches are correct. Use non-think mode (simpler, faster) as the BKC.

### Token & Turn Efficiency

Trajectory-level comparison across all 10 instances:

| Metric | No Thinking | Think High | Delta |
|--------|------------|------------|-------|
| Total API calls | 350 | 317 | -9% |
| Total tokens | 6,118,046 | 5,637,575 | -8% |
| Avg calls/instance | 35.0 | 31.7 | -3.3 |
| Avg tokens/call | 17,480 | 17,784 | +2% |
| Avg completion/call | 10,804 | 12,030 | +11% |
| Avg prompt/instance | 601,000 | 551,728 | -8% |

Thinking makes the model slightly more efficient — it reasons more per call (+11% completion tokens) but makes better decisions resulting in fewer total calls (-9%). Fewer calls means less accumulated context, reducing prompt tokens (-8%). Net effect: 8% fewer total tokens consumed per instance. However, this efficiency gain does not translate to higher resolution rate.

### Per-instance comparison: No Thinking vs Think High

| instance | API no→think | ΔAPI | Tokens no→think | ΔTokens | Compl no→think | ΔCompl | ✓ |
|---|---|---|---|---|---|---|---|
| astropy-12907 | 15→16 | +1 | 125K→154K | +23% | 3.4K→5.7K | +69% | ✓ |
| astropy-13033 | 22→25 | +3 | 177K→273K | +55% | 4.7K→6.8K | +45% | ✗ |
| astropy-13236 | 48→53 | +5 | 496K→735K | +48% | 7.4K→10.1K | +36% | ✓ |
| astropy-13398 | 37→36 | -1 | 1.07M→909K | -15% | 11.6K→9.4K | -19% | ✗ |
| astropy-13453 | 36→26 | **-10** | 513K→349K | **-32%** | 5.8K→4.4K | -24% | ✓ |
| astropy-13579 | 28→36 | +8 | 594K→1.21M | **+104%** | 19.8K→33.0K | +67% | ✓ |
| astropy-13977 | 41→31 | **-10** | 607K→471K | -22% | 8.8K→11.0K | +25% | ✗ |
| astropy-14096 | 51→46 | -5 | 1.34M→806K | **-40%** | 28.5K→22.3K | -22% | ✓ |
| astropy-14182 | 54→27 | **-27** | 1.05M→423K | **-60%** | 14.8K→9.6K | -35% | ✗ |
| astropy-14309 | 18→21 | +3 | 138K→309K | +125% | 3.2K→7.9K | +149% | ✓ |
| **AVG** | **35→32** | **-3** | **612K→564K** | **-8%** | **10.8K→12.0K** | **+11%** | |

Key observations:
- **14182**: Thinking cut calls by 50% (54→27) and tokens by 60% — largest improvement, but still unresolved
- **13453**: Thinking reduced calls by 28% (36→26) and tokens by 32% — most efficient resolved instance
- **13579**: Thinking INCREASED tokens by 104% — model over-thought, but still resolved
- **14096**: Thinking cut tokens by 40% (1.34M→806K) — better focus with thinking
- Thinking helps 5/10 instances (fewer tokens), hurts 5/10 (more tokens). Net effect: -8% tokens, no change in resolution

### Parallel Execution

5 concurrent workers (`-w 5`) work correctly with DeepSeek V4. The vLLM server handles multiple concurrent requests via batching. 10 instances complete in ~30 min with parallelism vs ~1.5 hours sequentially.

## 9. Known Issues

- **First-time startup slow**: ~30 minutes for JIT kernel compilation (TileLang/AutoTuner). Subsequent starts reuse cached compilations.
- **CUDA toolkit mismatch**: Requires `CUDA_HOME=/usr/local/cuda-13` to match vllm venv headers.
- **Memory headroom**: With 128K context, ~80 GB remains for KV cache. For 500-instance full runs, some instances with very long conversations may hit the limit. Reduce `max-model-len` further if needed.
- **No cost tracking**: `cost_tracking: "ignore_errors"` means zero cost shown.
- **Cannot use omni venv**: Requires vllm venv (`/home/yiliu7/workspace/venvs/vllm`) with vLLM 0.23.1rc1+.

## 10. Quick Reference

```bash
# Server
CUDA_HOME=/usr/local/cuda-13 CUDA_VISIBLE_DEVICES=2,3 \
/home/yiliu7/workspace/venvs/vllm/bin/vllm serve \
  /storage/yiliu7/deepseek-ai/DeepSeek-V4-Flash \
  --trust-remote-code --kv-cache-dtype fp8 --block-size 256 \
  --tensor-parallel-size 2 --max-model-len 131072 \
  --tokenizer-mode deepseek_v4 --tool-call-parser deepseek_v4 \
  --enable-auto-tool-choice --reasoning-parser deepseek_v4 --port 8000

# Run (10 instances)
source /home/yiliu7/workspace/venvs/omni/bin/activate
mini-extra swebench \
  --subset verified --split test --slice "0:10" \
  -c swebench.yaml -c swebench_deepseek_local.yaml \
  -c agent.mode=yolo -w 1 \
  -o ./output/deepseek_v4_$(date +%Y%m%d_%H%M%S)

# Evaluate
python -m swebench.harness.run_evaluation \
  --dataset_name princeton-nlp/SWE-Bench_Verified \
  --predictions_path output/<run>/preds.jsonl \
  --split test --max_workers 2 --run_id deepseek-v4
```
