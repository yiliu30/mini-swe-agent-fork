#!/usr/bin/env bash
set -euo pipefail

container="${1:-vllm-ds-yi}"
remote_repo="${2:-/workspace/vllm}"
source_repo="/home/yiliu7/workspace/vllm"
patch_file="/home/yiliu7/workspace/vllm-funnel-standard-pack.patch"

docker exec "${container}" bash -lc "
set -euo pipefail
test -d '${remote_repo}'
test -f '${patch_file}'
mkdir -p '${remote_repo}/tests/v1/attention'
cp '${source_repo}/vllm/attention/ops/common.py' '${remote_repo}/vllm/attention/ops/common.py'
cp '${source_repo}/vllm/model_executor/models/deepseek_v2.py' '${remote_repo}/vllm/model_executor/models/deepseek_v2.py'
cp '${source_repo}/tests/v1/attention/test_sparse_prefill_standard_pack.py' '${remote_repo}/tests/v1/attention/test_sparse_prefill_standard_pack.py'
cd '${remote_repo}'
source .venv/bin/activate
python -m compileall \
  vllm/attention/ops/common.py \
  vllm/model_executor/models/deepseek_v2.py \
  tests/v1/attention/test_sparse_prefill_standard_pack.py
VLLM_USE_PRECOMPILED=1 uv pip install --editable . --torch-backend=auto
git status --short -- \
  vllm/attention/ops/common.py \
  vllm/model_executor/models/deepseek_v2.py \
  tests/v1/attention/test_sparse_prefill_standard_pack.py
"

echo "Applied funnel standard-pack files into ${container}:${remote_repo}"
echo "Patch bundle: ${patch_file}"
