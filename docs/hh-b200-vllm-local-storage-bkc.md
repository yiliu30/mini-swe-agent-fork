# hh-b200 Local-Storage vLLM BKC

Best known configuration for running vLLM on `hh-b200` without paying the main
NFS penalty from `/home`.

## 1. Why this BKC exists

On `hh-b200`:

- `/home` is NFS-mounted
- `/` and `/storage` are local to the node

Verified from the node:

```bash
df -h / /home /storage /tmp
mount | egrep ' on /(home|storage|tmp) '
```

Current behavior:

- repo under `/home/.../workspace/vllm` is slower for editable installs
- venv under the repo is slower for many-small-file operations
- Hugging Face cache under `/home/.../.cache/huggingface` is slower for model
  downloads and metadata churn
- `uv` is much faster when its cache and temp files stay on local disk

## 2. Recommended host paths

Use local-node paths for repo, venv, and model cache:

```text
repo:        /storage/hshen/workspace/vllm
venv:        /storage/hshen/venvs/vllm-precompiled
hf cache:    /storage/hshen/.cache/huggingface
uv cache:    /tmp/uv-cache
tmpdir:      /tmp
```

Recommended host bootstrap:

```bash
mkdir -p \
  /storage/hshen/workspace \
  /storage/hshen/venvs \
  /storage/hshen/.cache/huggingface \
  /tmp/uv-cache
```

## 3. Container shape

Use the same base image as the working AWS setup:

```text
nvcr.io/nvidia/pytorch:26.06-py3
```

Recommended container launch on `hh-b200`:

```bash
docker run -d \
  --name vllm-ds-precompiled-smoke \
  --gpus all \
  --shm-size=64g \
  --network host \
  -w /workspace/vllm \
  -v /storage/hshen/workspace/vllm:/workspace/vllm \
  -v /storage:/storage \
  -v /storage/hshen/.cache/huggingface:/root/.cache/huggingface \
  -e UV_LINK_MODE=copy \
  -e TMPDIR=/tmp \
  -e UV_CACHE_DIR=/tmp/uv-cache \
  -e HF_HOME=/root/.cache/huggingface \
  -e HUGGINGFACE_HUB_CACHE=/root/.cache/huggingface/hub \
  -e TRANSFORMERS_CACHE=/root/.cache/huggingface/transformers \
  nvcr.io/nvidia/pytorch:26.06-py3 \
  sleep infinity
```

Notes:

- bind-mount the repo from `/storage`, not `/home`
- keep `uv` cache on `/tmp`
- keep Hugging Face cache on `/storage`
- if you want persistent `uv` wheels instead of ephemeral `/tmp`, use
  `/storage/hshen/.cache/uv`, but `/tmp` is the faster default for rebuilds

## 4. Repo and env setup

Inside the container:

```bash
export TMPDIR=/tmp
export UV_CACHE_DIR=/tmp/uv-cache
export HF_HOME=/root/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=/root/.cache/huggingface/hub
export TRANSFORMERS_CACHE=/root/.cache/huggingface/transformers
```

Clone or sync the repo into the storage-backed workspace:

```bash
cd /workspace
git clone <repo-url> vllm
cd /workspace/vllm
```

Create the venv on local storage:

```bash
uv venv /storage/hshen/venvs/vllm-precompiled
```

Install editable vLLM with precompiled mode:

```bash
VLLM_USE_PRECOMPILED=1 \
uv pip install \
  --python /storage/hshen/venvs/vllm-precompiled/bin/python \
  --editable /workspace/vllm \
  --torch-backend=auto
```

This keeps:

- source tree on `/storage`
- venv on `/storage`
- install/build temp/cache on `/tmp`

## 5. Hugging Face download behavior

For model download throughput on this node:

- keep `HF_HOME` off `/home`
- prefer `/storage/hshen/.cache/huggingface`
- avoid repo-local caches inside NFS workspaces

Basic smoke test:

```bash
/storage/hshen/venvs/vllm-precompiled/bin/python - <<'PY'
from huggingface_hub import snapshot_download
print(snapshot_download("facebook/opt-125m", local_files_only=False))
PY
```

## 6. Verification

Check the storage-backed layout:

```bash
df -h / /home /storage /tmp
```

Check that the repo and venv are not on `/home`:

```bash
realpath /workspace/vllm
realpath /storage/hshen/venvs/vllm-precompiled
```

Check the vLLM install:

```bash
/storage/hshen/venvs/vllm-precompiled/bin/python - <<'PY'
import vllm
print(vllm.__version__)
print(vllm.__file__)
PY
```

Check package metadata:

```bash
uv pip show \
  --python /storage/hshen/venvs/vllm-precompiled/bin/python \
  vllm
```

## 7. Current caveats

- `vllm --version` can hang in some local trees because this CLI imports most
  subcommands before printing the version. Prefer:

```bash
/storage/hshen/venvs/vllm-precompiled/bin/python - <<'PY'
import vllm
print(vllm.__version__)
PY
```

- If you only sync a committed branch state from another node, you may still
  miss uncommitted local files needed by that serving environment.

- The current `/home/yiliu7/workspace/hh-b200/create_docker.sh` was initially
  wired to `/home`-backed paths. For best install and download performance on
  `hh-b200`, prefer the `/storage` layout above.
