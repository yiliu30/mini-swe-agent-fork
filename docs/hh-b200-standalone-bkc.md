# `hh-b200` Standalone Node BKC

Best known configuration for using `hh-b200` as a general-purpose containerized
build and serving node.

## 1. Goal

This BKC is for workloads where:

- the node does the package install itself
- the node runs services or batch jobs in Docker
- the working set should stay off `/home`

The main reason is that `/home` is NFS-backed on `hh-b200`, while `/storage`,
`/tmp`, and `/` are local to the node.

## 2. Storage layout

Recommended host paths:

```text
workspace:   /storage/hshen/workspace
venvs:       /storage/hshen/venvs
app caches:  /storage/hshen/.cache
uv cache:    /tmp/uv-cache
tmpdir:      /tmp
```

Bootstrap once:

```bash
mkdir -p \
  /storage/hshen/workspace \
  /storage/hshen/venvs \
  /storage/hshen/.cache \
  /tmp/uv-cache
```

## 3. General rules

- Keep repos on `/storage`, not `/home`
- Keep virtualenvs on `/storage`, not inside NFS-backed repos
- Keep package-manager temp and cache files on local disk
- Keep large model or artifact caches on `/storage`
- Use `/tmp` for high-churn scratch data

This matters most for editable installs, wheel extraction, package metadata
walks, and Hugging Face downloads.

## 4. Container pattern

Use the same base image as the AWS B200 flow when you want parity:

```text
nvcr.io/nvidia/pytorch:26.06-py3
```

Recommended generic launch pattern:

```bash
docker run -d \
  --name <container_name> \
  --gpus all \
  --shm-size=64g \
  --network host \
  -w /workspace/app \
  -v /storage/hshen/workspace/app:/workspace/app \
  -v /storage:/storage \
  -v /storage/hshen/.cache:/root/.cache \
  -e https_proxy -e http_proxy -e HTTPS_PROXY -e HTTP_PROXY -e no_proxy -e NO_PROXY \
  -e TMPDIR=/tmp \
  -e UV_CACHE_DIR=/tmp/uv-cache \
  -e UV_LINK_MODE=copy \
  nvcr.io/nvidia/pytorch:26.06-py3 \
  sleep infinity
```

Adjust `/workspace/app` to the actual repo name.

## 5. Package install guidance

For faster installs on this node:

```bash
export TMPDIR=/tmp
export UV_CACHE_DIR=/tmp/uv-cache
export UV_LINK_MODE=copy
```

If the workload uses Hugging Face or other model caches:

```bash
export HF_HOME=/root/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=/root/.cache/huggingface/hub
export TRANSFORMERS_CACHE=/root/.cache/huggingface/transformers
```

Recommended generic flow inside the container:

```bash
cd /workspace
git clone <repo-url> app
cd /workspace/app
uv venv /storage/hshen/venvs/app
uv pip install --python /storage/hshen/venvs/app/bin/python --editable .
```

## 6. Validation

Check the mount behavior:

```bash
df -h / /home /storage /tmp
mount | egrep ' on /(home|storage|tmp) '
```

Check that the repo and venv are not on `/home`:

```bash
realpath /workspace/app
realpath /storage/hshen/venvs/app
```

Check installed package metadata:

```bash
uv pip show --python /storage/hshen/venvs/app/bin/python <package-name>
```

## 7. `vllm` Example

For `vllm`, apply the same node rules:

- repo on `/storage/hshen/workspace/vllm`
- venv on `/storage/hshen/venvs/vllm-precompiled`
- Hugging Face cache on `/storage/hshen/.cache/huggingface`
- `uv` cache on `/tmp/uv-cache`

Example install:

```bash
uv venv /storage/hshen/venvs/vllm-precompiled
VLLM_USE_PRECOMPILED=1 \
uv pip install \
  --python /storage/hshen/venvs/vllm-precompiled/bin/python \
  --editable /workspace/vllm \
  --torch-backend=auto
```

Example verification:

```bash
/storage/hshen/venvs/vllm-precompiled/bin/python - <<'PY'
import vllm
print(vllm.__version__)
print(vllm.__file__)
PY
```

## 8. Current `vllm` Caveat

The current `hh-b200` `vllm` checkout matches the AWS committed branch state,
but not the full AWS working tree. It is therefore not yet equivalent to the
AWS-serving setup.

The current smoke-test failure is:

```text
ModuleNotFoundError: No module named 'vllm.v1.attention.backends.mla.prefill_observation'
```

So the node-level BKC is valid, but the current `vllm` tree still needs sync if
you want exact AWS parity.
