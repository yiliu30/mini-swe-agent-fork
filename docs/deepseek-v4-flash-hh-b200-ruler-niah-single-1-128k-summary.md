# DeepSeek V4 Flash on `hh-b200` - RULER `niah_single_1` 128k Summary

## Run Summary

| | |
|---|---|
| Model | `DeepSeek-V4-Flash` |
| Serving node | remote `hh-b200` |
| Access path | local tunnel `127.0.0.1:18000 -> hh-b200:127.0.0.1:8001` |
| Client node | current local node |
| Harness repo | `/home/yiliu7/workspace/lm-evaluation-harness` |
| Harness venv | `/home/yiliu7/workspace/lm-evaluation-harness/.venv` |
| Task | `niah_single_1` |
| Sequence length under test | `131072` |
| API model backend | `local-completions` |
| Batch size | `1` |
| Full-run concurrency | `64` |

## Final Result

- Smoke run result at `131072`: `1.0`
- Full run result at `131072`: `1.0`
- Full run samples: `500/500`
- Full run finished without API errors

The extra `4096 = -1` entry is expected from the task's metric schema. The requested test length was enforced through `--metadata '{"max_seq_lengths":[131072]}'`.

## Artifacts

- Smoke output: `output/ruler_niah_single_1_128k_smoke_20260803_224600/`
- Smoke results: `output/ruler_niah_single_1_128k_smoke_20260803_224600/DeepSeek-V4-Flash/results_2026-08-03T22-51-11.798298.json`
- Smoke samples: `output/ruler_niah_single_1_128k_smoke_20260803_224600/DeepSeek-V4-Flash/samples_niah_single_1_2026-08-03T22-51-11.798298.jsonl`
- Full output: `output/ruler_niah_single_1_128k_full_20260803_225125/`
- Full results: `output/ruler_niah_single_1_128k_full_20260803_225125/DeepSeek-V4-Flash/results_2026-08-03T23-09-09.211915.json`
- Full samples: `output/ruler_niah_single_1_128k_full_20260803_225125/DeepSeek-V4-Flash/samples_niah_single_1_2026-08-03T23-09-09.211915.jsonl`

## Environment Notes

The harness environment needed two fixes before the run was stable:

- install the harness extras: `pip install -e /home/yiliu7/workspace/lm-evaluation-harness[api,ruler]`
- pin transformers back below v5:
  - `pip install 'transformers<5' 'tokenizers<0.22'`

Two runtime details also mattered:

- run `lm_eval` from outside the harness repo
- invoke Python with `-P` to avoid the NLTK `defusedxml` import guard triggered by the repo working directory

## Commands Used

Environment setup:

```bash
cd /home/yiliu7/workspace/lm-evaluation-harness
python -m venv .venv
. .venv/bin/activate
pip install -e /home/yiliu7/workspace/lm-evaluation-harness[api,ruler]
pip install 'transformers<5' 'tokenizers<0.22'
```

Smoke run:

```bash
/home/yiliu7/workspace/lm-evaluation-harness/.venv/bin/python -P -m lm_eval \
  --model local-completions \
  --tasks niah_single_1 \
  --batch_size 1 \
  --model_args model=DeepSeek-V4-Flash,base_url=http://127.0.0.1:18000/v1/completions,tokenizer=deepseek-ai/DeepSeek-V4-Flash,tokenizer_backend=huggingface,tokenized_requests=False,num_concurrent=1,max_retries=10,max_length=131200 \
  --metadata '{"max_seq_lengths":[131072]}' \
  --limit 5 \
  --log_samples \
  --output_path /home/yiliu7/workspace/mini-swe-agent-fork/output/ruler_niah_single_1_128k_smoke_20260803_224600
```

Full run:

```bash
/home/yiliu7/workspace/lm-evaluation-harness/.venv/bin/python -P -m lm_eval \
  --model local-completions \
  --tasks niah_single_1 \
  --batch_size 1 \
  --model_args model=DeepSeek-V4-Flash,base_url=http://127.0.0.1:18000/v1/completions,tokenizer=deepseek-ai/DeepSeek-V4-Flash,tokenizer_backend=huggingface,tokenized_requests=False,num_concurrent=64,max_retries=10,max_length=131200 \
  --metadata '{"max_seq_lengths":[131072]}' \
  --log_samples \
  --output_path /home/yiliu7/workspace/mini-swe-agent-fork/output/ruler_niah_single_1_128k_full_20260803_225125
```

Endpoint check:

```bash
bash scripts/check_vllm_status.sh --remote-port 8001
```

## Timing

- Smoke run wall time: about `3m 38s`
- Full run wall time: about `17m 36s`
- Full request phase only: about `14m 26s`

## Server State Observed

- `/v1/models` returned `DeepSeek-V4-Flash`
- server max length was `1048576`
- final post-run status showed:
  - `health: ok`
  - `requests_running: 0`
  - `requests_waiting: 0`
  - `requests_finished_error: 0`

Inference for both runs was served remotely on `hh-b200`; only the evaluation client and output artifacts lived on the current local node.
