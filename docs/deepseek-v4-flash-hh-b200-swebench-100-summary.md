# DeepSeek V4 Flash on `hh-b200` - SWE-bench Verified 100 Summary

## Run Summary

| | |
|---|---|
| Model | `hosted_vllm/DeepSeek-V4-Flash` |
| Serving node | remote `hh-b200` via local tunnel `127.0.0.1:18000 -> hh-b200:127.0.0.1:8000` |
| Agent node | current local node |
| Config | `swebench_deepseek_hh_b200.yaml` |
| Main output | `output/deepseek_v4_hh_b200_verified_100_run_20260803_0423/` |
| Main evaluation | `logs/run_evaluation/deepseek-v4-flash-hh-b200-100-20260803-0423-w32/` |
| Main report | `hosted_vllm__DeepSeek-V4-Flash.deepseek-v4-flash-hh-b200-100-20260803-0423-w32.json` |
| Retry output | `output/deepseek_v4_hh_b200_retry_10_20260803_0506/` |
| Retry evaluation | `logs/run_evaluation/deepseek-v4-flash-hh-b200-retry-10-20260803-0506/` |
| Retry report | `hosted_vllm__DeepSeek-V4-Flash.deepseek-v4-flash-hh-b200-retry-10-20260803-0506.json` |

## Final Result

- Targeted instances: `100`
- Main-pass generated predictions: `100/100`
- Main-pass exit statuses: `91 Submitted`, `9 RepeatedFormatError`
- Main-pass non-empty patches evaluated: `90`
- Main-pass resolved: `62/100`
- Retry batch size: `10` previously non-evaluated cases
- Retry exit statuses: `9 Submitted`, `1 RepeatedFormatError`
- Retry resolved: `7/10`
- Final combined resolved: `69/100`
- Final combined evaluated non-empty patches: `99/100`
- Final combined unresolved after evaluation: `30/100`
- Final still non-evaluated: `1/100`

Score views:

- End-to-end score on the 100 targeted instances: `69/100 = 69.0%`
- Score on evaluated non-empty patches after retry: `69/99 = 69.7%`
- Main-pass score on evaluated non-empty patches: `62/90 = 68.9%`

## Commands Used

Main generation:

```bash
mini-extra swebench \
  --subset verified \
  --split test \
  --slice 0:100 \
  --workers 32 \
  --output output/deepseek_v4_hh_b200_verified_100_run_20260803_0423 \
  -c swebench.yaml \
  -c swebench_deepseek_hh_b200.yaml \
  -c agent.mode=yolo
```

Main evaluation:

```bash
python -m swebench.harness.run_evaluation \
  --dataset_name princeton-nlp/SWE-Bench_Verified \
  --predictions_path output/deepseek_v4_hh_b200_verified_100_run_20260803_0423/preds.jsonl \
  --split test \
  --max_workers 32 \
  --run_id deepseek-v4-flash-hh-b200-100-20260803-0423-w32
```

Retry generation:

```bash
mini-extra swebench \
  --subset verified \
  --split test \
  --filter '^(astropy__astropy-13579|astropy__astropy-7166|django__django-10880|django__django-10999|django__django-11206|django__django-11211|django__django-11490|django__django-11951|django__django-12155|django__django-12406)$' \
  --workers 10 \
  --output output/deepseek_v4_hh_b200_retry_10_20260803_0506 \
  -c swebench.yaml \
  -c swebench_deepseek_hh_b200.yaml \
  -c agent.mode=yolo
```

Retry evaluation:

```bash
python -m swebench.harness.run_evaluation \
  --dataset_name princeton-nlp/SWE-Bench_Verified \
  --predictions_path output/deepseek_v4_hh_b200_retry_10_20260803_0506/preds.jsonl \
  --split test \
  --max_workers 10 \
  --run_id deepseek-v4-flash-hh-b200-retry-10-20260803-0506
```

## Main-Pass Evaluation Details

- The main generation run produced `91` `Submitted` statuses, but only `90` non-empty patches.
- `astropy__astropy-7166` was marked `Submitted` in generation but had an empty `model_patch`, so it was not included in `preds.jsonl` and was not scored in the main evaluation.
- The other `9` non-evaluated cases were `RepeatedFormatError`.
- The SWE-bench harness report JSON uses `total_instances: 500` because it is keyed to the full Verified test split. This 100-instance run only targeted `--slice 0:100`.

Main-pass non-evaluated cases:

- `astropy__astropy-13579`
- `astropy__astropy-7166`
- `django__django-10880`
- `django__django-10999`
- `django__django-11206`
- `django__django-11211`
- `django__django-11490`
- `django__django-11951`
- `django__django-12155`
- `django__django-12406`

## Retry Outcome

Retry resolved:

- `astropy__astropy-13579`
- `astropy__astropy-7166`
- `django__django-10880`
- `django__django-11206`
- `django__django-11211`
- `django__django-11951`
- `django__django-12155`

Retry unresolved:

- `django__django-10999`
- `django__django-11490`

Still not evaluated after retry:

- `django__django-12406`

## Artifacts

- Main run log: `output/deepseek_v4_hh_b200_verified_100_run_20260803_0423/minisweagent.log`
- Main run predictions: `output/deepseek_v4_hh_b200_verified_100_run_20260803_0423/preds.json`
- Main run filtered predictions: `output/deepseek_v4_hh_b200_verified_100_run_20260803_0423/preds.jsonl`
- Retry log: `output/deepseek_v4_hh_b200_retry_10_20260803_0506/minisweagent.log`
- Retry predictions: `output/deepseek_v4_hh_b200_retry_10_20260803_0506/preds.json`
- Retry filtered predictions: `output/deepseek_v4_hh_b200_retry_10_20260803_0506/preds.jsonl`
