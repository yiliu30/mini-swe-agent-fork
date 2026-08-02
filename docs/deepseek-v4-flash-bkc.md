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

**GPU**: 2 GPUs needed (B200, 183 GB each). Model uses ~74 GiB per GPU. Startup takes ~30 minutes first time (JIT kernel compilation), or ~2 minutes with autotune disabled (see below).

### Funnel Dense Backend (Current BKC)

The old host-side `cp -r funnel_topk ... site-packages` flow is stale for the
current turbo funnel path. The working BKC is the Docker + precompiled vLLM
setup from `/home/yiliu7/workspace/vllm/deepseek_v4_flash_docker_cmds.sh`:

- Docker image: `nvcr.io/nvidia/pytorch:26.06-py3`
- Container: `vllm-ds-precompiled-smoke`
- vLLM install: editable, precompiled mode
- funnel-topk install: editable from `/workspace/funnel-topk`

Start the server with funnel enabled:

```bash
docker exec vllm-ds-precompiled-smoke bash -lc '
  cd /workspace/vllm &&
  source /opt/vllm-precompiled-venv/bin/activate &&
  export VLLM_SPARSE_INDEXER_PREFILL_TOPK_BACKEND=funnel_dense \
         VLLM_SPARSE_INDEXER_PREFILL_TOPK_FUNNEL_MODE=turbo \
         FLASHINFER_DISABLE_VERSION_CHECK=1 \
         NCCL_IB_DISABLE=1 \
         NCCL_P2P_DISABLE=1 \
         NCCL_SHM_DISABLE=1 \
         TORCH_NCCL_BLOCKING_WAIT=1 \
         VLLM_DISABLE_PYNCCL=1 \
         VLLM_ALLREDUCE_USE_SYMM_MEM=0 \
         VLLM_DEEP_GEMM_WARMUP=skip \
         VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR=/root/.cache/vllm_flashinfer_autotune \
         CUDA_VISIBLE_DEVICES=0,1 &&
  /opt/vllm-precompiled-venv/bin/vllm serve \
    /storage/yiliu7/deepseek-ai/DeepSeek-V4-Flash/ \
    --trust-remote-code \
    --kv-cache-dtype fp8 \
    --block-size 256 \
    --enable-expert-parallel \
    --tensor-parallel-size 2 \
    --attention_config.use_fp4_indexer_cache=True \
    --tokenizer-mode deepseek_v4 \
    --reasoning-parser deepseek_v4 \
    --gpu-memory-utilization 0.75 \
    --kernel-config.enable_flashinfer_autotune=False \
    --kernel-config.enable_jit_warmup=False \
    --kernel-config.enable_cutedsl_warmup=False \
    --disable-custom-all-reduce \
    --port 8000
'
```

Current behavior in vLLM:

- `funnel_dense` calls `top_k_per_row_prefill_funnel_v1`
- `VLLM_SPARSE_INDEXER_PREFILL_TOPK_FUNNEL_MODE=turbo` is required for the new
  turbo kernel path
- `DeepSeek-V4-Flash` uses `index_topk=512`
- if the funnel adapter fails, vLLM raises instead of silently falling back

Verification:

- call site: `/home/yiliu7/workspace/vllm/vllm/model_executor/layers/sparse_attn_indexer.py`
- audit log:
  `/home/yiliu7/workspace/vllm/logs/ds_sweep/prefill_v1_audit.log`
- expected lines:
  `top_k_per_row_prefill_funnel_v1 ... top_k=512 mode=turbo`

Current GSM8K check in Docker (`2026-08-02`):

- with funnel: `flexible 0.9507 +- 0.0060`, `strict 0.9515 +- 0.0059`
- without funnel: `flexible 0.9507 +- 0.0060`, `strict 0.9507 +- 0.0060`

Reference note:
`/home/yiliu7/workspace/funnel-topk/bench_res/deepseek_v4_flash_gsm8k_docker_20260802.md`

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

## 7. 100-Instance Run (2026-06-24)

Scaled from 10 to 100 instances (first 100 of Verified test split: 22 astropy, 78 django). No-Think BKC config, 64 parallel workers.

### Results at a Glance

| Metric | 10-instance | **100-instance** |
|--------|-----------|-----------------|
| Instances | 10 | **100** |
| Submitted | 10 (100%) | **95 (95%)** |
| Resolved (submitted) | 6/10 (60%) | **69/95 (73%)** |
| Resolved (overall) | 6/10 (60%) | **69/100 (69%)** |
| Format errors | 0 | **0** |
| ContextWindowExceeded | 0 | **5** |
| Avg API calls/instance | 35 | **41** |
| Avg actions/instance | 35 | **41** |
| Avg tokens/instance | 612K | **794K** |
| Total tokens | 6.1M | **79.4M** |
| Wall clock | 26 min (seq) | **25 min (64 workers)** |
| vLLM gen throughput | 28 tok/s (1 req) | **2120 avg / 12395 max tok/s** |

### Key Observation

The 10-instance sample (all astropy) scored 60%. The 100-instance sample (78 django, 22 astropy) scored 69%. A full 500-instance run achieved **74.3%** (277/373 evaluable) — see [500-Instance Run](#500-instance-run-2026-06-28) for details.

### Exit Status Breakdown

| Status | Count | % |
|--------|-------|---|
| Submitted | 95 | 95% |
| ContextWindowExceededError | 5 | 5% |

The 5 context window errors occurred on instances with very long conversations (>200 steps). These could be recovered by restarting with `max-model-len` bumped or reducing accumulated context.

### Per-Repo Breakdown

| Repo | Instances | Submitted | Resolved (est.) |
|------|-----------|-----------|-----------------|
| astropy | 22 | 22 (100%) | ~60% |
| django | 78 | 73 (94%) | ~70% |

### Funnel Dense Backend Validation (2026-06-28)

Tested 10 and 100 instances with funnel_dense + autotune disabled (```--kernel-config.enable_flashinfer_autotune=False```). Server startup: ~2 min vs ~30 min.

| Metric | Native (10 inst) | **funnel_dense (10 inst)** | Native (100 inst) | **funnel_dense (100 inst)** |
|--------|-----------------|--------------------------|------------------|---------------------------|
| Resolved | 6/10 (60%) | **6/10 (60%)** | 69/95 (73%) | **70/97 (72%)** |
| Submitted | 10/10 | 10/10 | 95% | **97%** |
| Format errors | 0 | 0 | 0 | 0 |
| ContextWindowExceeded | 0 | 0 | 5 | **2** |
| Avg API calls | 35 | 34 | 42 | **42** |
| Avg tokens/inst | 612K | 684K | 794K | **839K** |
| vLLM avg throughput | 591 tok/s | **686 tok/s** | 2120 tok/s | **1988 tok/s** |
| Startup time | ~30 min | **~2 min** | ~30 min | **~2 min** |

**Conclusion**: funnel_dense matches native accuracy at scale (70% vs 69%). Minor improvements in submission rate (97% vs 95%) and fewer context window errors (2 vs 5). Recommended for production runs due to dramatically faster startup.

### Running at Scale

```bash
# Pre-pull Docker images (required for first run)
python3 -c "
from datasets import load_dataset; import subprocess
ds = load_dataset('princeton-nlp/SWE-Bench_Verified', split='test')
for inst in ds.select(range(100)):
    iid = inst['instance_id'].replace('__','_1776_')
    img = f'docker.io/swebench/sweb.eval.x86_64.{iid}:latest'.lower()
    subprocess.run(['docker','pull',img], timeout=120)
"

# Run with 64 workers
mini-extra swebench \
  --subset verified --split test --slice "0:100" \
  -c swebench.yaml -c swebench_deepseek_local.yaml \
  -c agent.mode=yolo -w 64 \
  -o ./output/deepseek_v4_100/

# Re-run with --redo-existing to retry failures
```

### 500-Instance Run (2026-06-28)

Full SWE-bench Verified (test split, 500 instances) on native backend. 64 parallel workers, 377 patches submitted, docker pre-pull in batches due to disk constraints.

#### Results at a Glance

| Metric | 100-instance | **500-instance** |
|--------|-------------|-----------------|
| Trajectories recovered | — | **436** |
| Patches extracted | 95 | **434** |
| Evaluated | 95 | **430** |
| Resolved | 69 (73%) | **308 (71.6%)** |
| Submitted | 95 (95%) | **434** |
| Corrupted patches | 0 | **3** |
| Missing (no trajectory) | — | **64** |
| vLLM server | native (venvs/vllm) | **native (dev vllm, autotune off)** |

#### Per-Repo Breakdown

| Repo | Resolved | Total | Rate |
|------|----------|-------|------|
| astropy | 12 | 22 | 55% |
| django | 164 | 222 | 74% |
| sympy | 56 | 75 | 75% |
| sphinx-doc | 28 | 44 | 64% |
| scikit-learn | 28 | 31 | 90% |
| pytest-dev | 14 | 19 | 74% |
| pydata | 4 | 7 | 57% |
| pylint-dev | 2 | 10 | 20% |
| **Evaluated** | **308** | **430** | **71.6%** |

57 additional instances (42 sympy, 13 sphinx-doc, 2 scikit-learn) were evaluated in a follow-up run, adding 31 resolutions.

#### Coverage Gaps

64 instances missed due to incomplete Docker image pre-pull (rate limiting):

| Repo | Missing |
|------|---------|
| matplotlib | 0/34 (no trajectories) |
| psf | 0/8 (no trajectories) |
| pydata | 7/22 (15 no trajectories) |
| django | 224/231 (4 no trajectories) |
| mwaskom | 0/2 (no trajectories) |
| pallets | 0/1 (no trajectories) |

64 missing trajectories (matplotlib, psf, mwaskom, pallets, partial pydata/django) due to Docker Hub rate limiting during pre-pull. These + the 57 unevaluated patches should all resolve when Docker images are available for the funnel_dense run.

#### Notes

- 74.3% is higher than the 65-70% projected from the 100-instance run — non-django repos (scikit-learn 93%, sympy 85%, sphinx-doc 84%) pull the average up
- 3 corrupted patches from trajectory extraction truncation (django instances)
- Server: dev vLLM build on GPUs 0,1 with `--kernel-config.enable_flashinfer_autotune=False`, startup ~2 min
- Disk constrained run (106 GB free after cleanup) — used batched approach (slice by slice)

## 8. Comparison: All Models

| Metric | Qwen3 35B (temp=1.0) | **DeepSeek V4 (10 inst)** | **DeepSeek V4 (100 inst)** | **DeepSeek V4 (430 inst)** |
|--------|---------------------|--------------------------|---------------------------|
| Submission rate | 90% | 100% | 95% | **87%** |
| Resolved | 50% | 60% | 69% | **71.6%** |
| Format errors | 1 | 0 | 0 | **0** |
| Avg actions | 49 | 35 | 41 | **40** |
| Avg tokens/instance | — | 612K | 794K | **~800K** |
| Gen speed | 500 tok/s | 28 tok/s | 2120 tok/s (batched) | **2120 tok/s** |
| Wall clock | ~2.5 hr (seq) | 26 min (seq) | 25 min (64 workers) | **~2 hr (batched)** |
| GPU | 4 (TP4) | 2 (TP2) | 2 (TP2) | 2 (TP2) |

DeepSeek V4 Flash with native tool-calling achieves **71.6%** (308/430 evaluated) on SWE-bench Verified — confirming the 69-73% estimates from 10/100-instance runs. Zero format errors across all runs. The bash-only agent scaffold limits resolution on complex Python repos (pylint-dev 20%, pydata 57%), while simpler repos like scikit-learn (90%) and sympy (75%) excel. 64 instances missed due to incomplete Docker pre-pull.

## 9. Thinking Mode Experiment (2026-06-24)

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
| Think High | `chat_template_kwargs.thinking=true, reasoning_effort=high` | **6/10 (60%)** | 100% | 32 |
| Think Max | `chat_template_kwargs.thinking=true, reasoning_effort=max` | 5/10 (50%) | 100% | 54 |

**No-Think, Think High**: 6/10 (60%), resolve same instances: 12907, 13236, 13453, 13579, 14096, 14309.

**Think Max**: 5/10 (50%). Lost 13236 (resolved in all other runs). No new instances gained.

The same 4 instances fail across all runs: 13033, 13398, 13977, 14182.

### Key Finding

**Thinking mode does not improve DeepSeek V4 Flash on SWE-bench Verified.** No-Think and Think High tie at 6/10 (60%). Think Max regresses to 5/10 (50%) — deeper reasoning causes the model to over-think, producing more actions (avg 54 vs 32-38) with worse accuracy. **Use non-think mode as the BKC** — simplest, fastest, same accuracy.

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

## 10. Known Issues

- **First-time startup slow**: ~30 minutes for JIT kernel compilation (TileLang/AutoTuner). Subsequent starts reuse cached compilations.
- **CUDA toolkit mismatch**: Requires `CUDA_HOME=/usr/local/cuda-13` to match vllm venv headers.
- **Memory headroom**: With 128K context, ~80 GB remains for KV cache. For 500-instance full runs, some instances with very long conversations may hit the limit. Reduce `max-model-len` further if needed.
- **No cost tracking**: `cost_tracking: "ignore_errors"` means zero cost shown.
- **Cannot use omni venv**: Requires vllm venv (`/home/yiliu7/workspace/venvs/vllm`) with vLLM 0.23.1rc1+.

## 11. Quick Reference

```bash
# Server (default — omni venv)
CUDA_HOME=/usr/local/cuda-13 CUDA_VISIBLE_DEVICES=2,3 \
/home/yiliu7/workspace/venvs/vllm/bin/vllm serve \
  /storage/yiliu7/deepseek-ai/DeepSeek-V4-Flash \
  --trust-remote-code --kv-cache-dtype fp8 --block-size 256 \
  --tensor-parallel-size 2 --max-model-len 131072 \
  --tokenizer-mode deepseek_v4 --tool-call-parser deepseek_v4 \
  --enable-auto-tool-choice --reasoning-parser deepseek_v4 --port 8000

# Server (funnel_dense turbo, Docker BKC)
docker exec vllm-ds-precompiled-smoke bash -lc '
  cd /workspace/vllm &&
  source /opt/vllm-precompiled-venv/bin/activate &&
  export VLLM_SPARSE_INDEXER_PREFILL_TOPK_BACKEND=funnel_dense \
         VLLM_SPARSE_INDEXER_PREFILL_TOPK_FUNNEL_MODE=turbo \
         FLASHINFER_DISABLE_VERSION_CHECK=1 \
         NCCL_IB_DISABLE=1 \
         NCCL_P2P_DISABLE=1 \
         NCCL_SHM_DISABLE=1 \
         TORCH_NCCL_BLOCKING_WAIT=1 \
         VLLM_DISABLE_PYNCCL=1 \
         VLLM_ALLREDUCE_USE_SYMM_MEM=0 \
         VLLM_DEEP_GEMM_WARMUP=skip \
         CUDA_VISIBLE_DEVICES=0,1 &&
  /opt/vllm-precompiled-venv/bin/vllm serve \
    /storage/yiliu7/deepseek-ai/DeepSeek-V4-Flash/ \
    --trust-remote-code --kv-cache-dtype fp8 --block-size 256 \
    --enable-expert-parallel --tensor-parallel-size 2 \
    --attention_config.use_fp4_indexer_cache=True \
    --tokenizer-mode deepseek_v4 --reasoning-parser deepseek_v4 \
    --gpu-memory-utilization 0.75 \
    --kernel-config.enable_flashinfer_autotune=False \
    --kernel-config.enable_jit_warmup=False \
    --kernel-config.enable_cutedsl_warmup=False \
    --disable-custom-all-reduce --port 8000
'

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
