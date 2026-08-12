# Current Node Standalone BKC

Best known configuration for using the current node as a general-purpose
containerized build and serving host.

## 1. Goal

This BKC is for workloads where:

- the node installs packages locally
- the node runs services or batch jobs in Docker
- the working set should use the larger local disk

On this node:

- `/data` is a separate local ext4 mount
- `/home` is on `/`
- there is no `/storage`

So the practical optimization is to keep repos, virtualenvs, and large caches on
`/data`.

## 2. Storage layout

Recommended host paths:

```text
workspace:   /data/yiliu7/workspace
venvs:       /data/yiliu7/venvs
app caches:  /data/yiliu7/.cache
uv cache:    /tmp/uv-cache
tmpdir:      /tmp
```

Bootstrap once:

```bash
mkdir -p \
  /data/yiliu7/workspace \
  /data/yiliu7/venvs \
  /data/yiliu7/.cache \
  /tmp/uv-cache
```

## 3. General rules

- Keep large repos on `/data`
- Keep virtualenvs on `/data`
- Keep package-manager temp and cache files on local disk
- Keep model and artifact caches on `/data`
- Use `/tmp` for high-churn scratch data

This matters most for editable installs, wheel extraction, metadata scans, and
large model downloads.

## 4. Container pattern

Recommended generic launch pattern:

```bash
docker run -d \
  --name <container_name> \
  --gpus all \
  --shm-size=64g \
  --network host \
  -w /workspace/app \
  -v /data/yiliu7/workspace/app:/workspace/app \
  -v /data:/data \
  -v /data/yiliu7/.cache:/root/.cache \
  -e https_proxy -e http_proxy -e HTTPS_PROXY -e HTTP_PROXY -e no_proxy -e NO_PROXY \
  -e TMPDIR=/tmp \
  -e UV_CACHE_DIR=/tmp/uv-cache \
  -e UV_LINK_MODE=copy \
  nvcr.io/nvidia/pytorch:26.06-py3 \
  sleep infinity
```

Adjust `/workspace/app` to the actual repo name.

## 5. Package install guidance

For faster installs:

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
uv venv /data/yiliu7/venvs/app
uv pip install --python /data/yiliu7/venvs/app/bin/python --editable .
```

## 6. Validation

Check the mount behavior:

```bash
df -h / /home /data /tmp
findmnt -T /home -o TARGET,SOURCE,FSTYPE,OPTIONS -n
findmnt -T /data -o TARGET,SOURCE,FSTYPE,OPTIONS -n
```

Check that the repo and venv are on `/data`:

```bash
realpath /workspace/app
realpath /data/yiliu7/venvs/app
```

Check installed package metadata:

```bash
uv pip show --python /data/yiliu7/venvs/app/bin/python <package-name>
```

## 7. `vllm` Example

For `vllm`, apply the same node rules:

- repo on `/data/yiliu7/workspace/vllm`
- venv on `/data/yiliu7/venvs/vllm-precompiled`
- Hugging Face cache on `/data/yiliu7/.cache/huggingface`
- `uv` cache on `/tmp/uv-cache`

Example install:

```bash
uv venv /data/yiliu7/venvs/vllm-precompiled
VLLM_USE_PRECOMPILED=1 \
uv pip install \
  --python /data/yiliu7/venvs/vllm-precompiled/bin/python \
  --editable /workspace/vllm \
  --torch-backend=auto
```
