#!/usr/bin/env python3

"""Replay RULER prompts against DeepSeek servers and compare token logprobs."""

import ast
import json
import os
import signal
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import requests
import typer
from jinja2 import Template

from minisweagent.utils.log import logger

app = typer.Typer(help=__doc__, rich_markup_mode="rich", add_completion=False, no_args_is_help=True)

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_SAMPLE_PATH = (
    REPO_ROOT
    / "output/ruler_niah_single_1_128k_funnel_turbo_full_20260804_020151_retry/DeepSeek-V4-Flash/"
    "samples_niah_single_1_2026-08-04T02-21-11.771347.jsonl"
)
DEFAULT_REPORT_PATH = REPO_ROOT / "docs/deepseek-v4-flash-hh-b200-ruler-niah-single-1-128k-logprob-analysis.md"
DEFAULT_HOST = "hh-b200"
DEFAULT_CONTAINER = "vllm-ds-yi"
DEFAULT_MODEL = "/dev/shm/.yiliu7/deepseek-ai/DeepSeek-V4-Flash"
DEFAULT_SERVED_MODEL = "DeepSeek-V4-Flash"
DEFAULT_LOCAL_PORT = 18001
DEFAULT_REMOTE_PORT = 8001
DEFAULT_CONTROL_IDS = (1, 64, 128, 192, 256, 320, 384, 448)
DEFAULT_MISS_IDS = (180, 263, 388, 487)
DEFAULT_TUNNEL_PIDFILE = Path("/tmp/minisweagent-hh-b200-logprob-tunnel.pid")
DEFAULT_TOKENIZER_PYTHON = Path("/home/yiliu7/workspace/lm-evaluation-harness/.venv/bin/python")
DEFAULT_TOKENIZER_NAME = "deepseek-ai/DeepSeek-V4-Flash"
DEFAULT_CUDA_VISIBLE_DEVICES = "6,7"
REMOTE_LOG_DIR = Path("/tmp")
FUNNEL_MARKERS = (
    "Using funnel_dense backend for top-k prefill.",
    "using funnel_topk ragged prefill v1 op for prefill top-k",
)
SERVER_TIMEOUT_SECONDS = 480
REQUEST_TIMEOUT_SECONDS = 900

REPORT_TEMPLATE = Template(
    """# DeepSeek V4 Flash on `hh-b200` - RULER `niah_single_1` 128k Logprob Analysis

## Summary

- Prompt source: `{{ sample_path }}`
- Replay set: `{{ cases|length }}` prompts (`{{ miss_count }}` misses + `{{ control_count }}` controls)
- Serving comparison: baseline default top-k vs funnel `funnel_dense` + `turbo`
- Shared prompt set: yes
- Local tunnel: `127.0.0.1:{{ local_port }} -> hh-b200:127.0.0.1:{{ remote_port }}`

{% for mode in modes %}
## {{ mode.title }}

- Remote log: `{{ mode.log_path }}`
- Generated matches: `{{ mode.matched }}/{{ cases|length }}`
- Mean gold logprob: `{{ mode.avg_gold_logprob_text }}`
{% if mode.marker_status %}
- Funnel markers confirmed: `{{ mode.marker_status }}`
{% endif %}

{% endfor %}
## Per-Prompt Comparison

| Id | Kind | Gold | Baseline Match | Funnel Match | Baseline Gold LP | Funnel Gold LP | Delta (Base-Funnel) | First Gen Divergence | Funnel Note |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
{% for case in cases -%}
| {{ case.sample_id }} | {{ case.kind }} | `{{ case.target }}` | {{ 1 if case.baseline.generated.matches else 0 }} | {{ 1 if case.funnel.generated.matches else 0 }} | {{ case.baseline_gold_logprob_text }} | {{ case.funnel_gold_logprob_text }} | {{ case.gold_logprob_delta_text }} | {{ case.first_generation_divergence or '-' }} | {{ case.note }} |
{% endfor %}

## Miss Details

{% for case in cases if case.kind == 'miss' -%}
### Sample {{ case.sample_id }}

- Gold answer: `{{ case.target }}`
- Baseline generated: `{{ case.baseline.generated.text_preview }}`{% if case.baseline.generated.matches %} (match){% endif %}
- Funnel generated: `{{ case.funnel.generated.text_preview }}`{% if case.funnel.generated.matches %} (match){% endif %}
- Gold logprob delta: `{{ case.gold_logprob_delta_text }}`
- First generation divergence: `{{ case.first_generation_divergence or 'none' }}`
- First gold top-1 divergence: `{{ case.first_gold_top1_divergence or 'none' }}`
- First funnel gold token missing from top-20: `{{ case.funnel.gold.first_missing_position or 'none' }}`
- First funnel top-1 mismatch vs gold: `{{ case.funnel.gold.first_top1_mismatch or 'none' }}`

{% endfor %}
## Notes

- Gold scoring uses iterative one-token replay over the tokenizer pieces for `prompt + " " + target`.
- Matching uses the same normalization as the RULER task: trim, replace control characters with newlines, then substring-match the gold answer.
- This report is replay-based. The earlier 500-sample baseline and funnel benchmark outputs are not prompt-aligned, so this analysis uses one fixed prompt set from the funnel artifact for both server modes.
"""
)


@dataclass(frozen=True)
class ServerMode:
    name: str
    title: str
    log_name: str
    extra_env: dict[str, str]
    expect_funnel_markers: bool = False


@dataclass(frozen=True)
class SampleCase:
    sample_id: int
    kind: str
    prompt: str
    target: str
    prompt_hash: str


@dataclass(frozen=True)
class TokenDetail:
    token: str
    logprob: float | None
    top_logprobs: dict[str, float]
    offset: int | None


@dataclass(frozen=True)
class GeneratedAnalysis:
    text: str
    text_preview: str
    matches: bool
    tokens: list[str]
    token_logprobs: list[float | None]
    top_logprobs: list[dict[str, float]]


@dataclass(frozen=True)
class GoldAnalysis:
    continuation: str
    total_logprob: float | None
    tokens: list[TokenDetail]
    first_missing_position: int | None
    first_top1_mismatch: int | None


@dataclass(frozen=True)
class ModeCaseResult:
    generated: GeneratedAnalysis
    gold: GoldAnalysis


@dataclass(frozen=True)
class CaseComparison:
    sample_id: int
    kind: str
    target: str
    baseline: ModeCaseResult
    funnel: ModeCaseResult
    gold_logprob_delta: float | None
    first_generation_divergence: int | None
    first_gold_top1_divergence: int | None
    note: str


@dataclass(frozen=True)
class ModeSummary:
    title: str
    log_path: str
    matched: int
    avg_gold_logprob: float | None
    marker_status: str | None = None


@dataclass(frozen=True)
class TokenizedCase:
    case: SampleCase
    continuation_pieces: list[str]


BASELINE_MODE = ServerMode(
    name="baseline",
    title="Baseline Default Top-k",
    log_name="deepseek_logprob_baseline_8001.log",
    extra_env={},
)
FUNNEL_MODE = ServerMode(
    name="funnel_turbo",
    title="Funnel Dense Turbo Top-k",
    log_name="deepseek_logprob_funnel_turbo_8001.log",
    extra_env={
        "PYTHONPATH": "/workspace/funnel-topk:${PYTHONPATH:-}",
        "VLLM_SPARSE_INDEXER_PREFILL_TOPK_BACKEND": "funnel_dense",
        "VLLM_SPARSE_INDEXER_PREFILL_TOPK_FUNNEL_MODE": "turbo",
    },
    expect_funnel_markers=True,
)


def _now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _run(cmd: list[str], *, check: bool = True, input_text: str | None = None) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(cmd, input=input_text, text=True, capture_output=True)
    if check and result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or f"Command failed: {' '.join(cmd)}")
    return result


def _ssh(host: str, command: str, *, check: bool = True, input_text: str | None = None) -> subprocess.CompletedProcess[str]:
    return _run(["ssh", "-q", host, command], check=check, input_text=input_text)


def _pid_is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _read_pidfile(pidfile: Path) -> int | None:
    if not pidfile.exists():
        return None
    try:
        return int(pidfile.read_text().strip())
    except ValueError:
        pidfile.unlink(missing_ok=True)
        return None


def _pid_matches_tunnel(pid: int, host: str, local_port: int, remote_port: int) -> bool:
    cmdline_path = Path(f"/proc/{pid}/cmdline")
    if not cmdline_path.exists():
        return False
    cmdline = cmdline_path.read_text().replace("\x00", " ")
    return (
        "ssh" in cmdline
        and host in cmdline
        and "-N" in cmdline
        and f"127.0.0.1:{local_port}:127.0.0.1:{remote_port}" in cmdline
    )


def _get_live_tunnel_pid(host: str, local_port: int, remote_port: int, pidfile: Path) -> int | None:
    pid = _read_pidfile(pidfile)
    if pid is None:
        return None
    if _pid_is_running(pid) and _pid_matches_tunnel(pid, host, local_port, remote_port):
        return pid
    pidfile.unlink(missing_ok=True)
    return None


def _ensure_tunnel(host: str, local_port: int, remote_port: int, pidfile: Path) -> int:
    if pid := _get_live_tunnel_pid(host, local_port, remote_port, pidfile):
        return pid
    process = subprocess.Popen(
        [
            "ssh",
            "-N",
            "-L",
            f"127.0.0.1:{local_port}:127.0.0.1:{remote_port}",
            "-o",
            "ExitOnForwardFailure=yes",
            host,
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    time.sleep(1)
    if process.poll() is not None:
        raise RuntimeError(f"SSH tunnel to {host} exited with code {process.returncode}.")
    pidfile.write_text(str(process.pid))
    return process.pid


def _stop_tunnel(host: str, local_port: int, remote_port: int, pidfile: Path) -> None:
    pid = _get_live_tunnel_pid(host, local_port, remote_port, pidfile)
    if pid is None:
        pidfile.unlink(missing_ok=True)
        return
    os.kill(pid, signal.SIGTERM)
    for _ in range(20):
        if not _pid_is_running(pid):
            pidfile.unlink(missing_ok=True)
            return
        time.sleep(0.1)
    os.kill(pid, signal.SIGKILL)
    pidfile.unlink(missing_ok=True)


def normalize_prediction(text: str) -> str:
    clean = "".join("\n" if ord(char) < 32 else char for char in text)
    return clean.strip()


def matches_target(text: str, target: str) -> bool:
    return target.lower() in normalize_prediction(text).lower()


def _preview(text: str, limit: int = 96) -> str:
    preview = normalize_prediction(text).replace("\n", "\\n")
    return preview[:limit] + ("..." if len(preview) > limit else "")


def _top_logprob_map(top_logprobs: Any) -> dict[str, float]:
    if isinstance(top_logprobs, dict):
        return {str(key): float(value) for key, value in top_logprobs.items()}
    return {}


def _response_tokens(choice: dict[str, Any]) -> list[str]:
    return [str(token) for token in choice.get("logprobs", {}).get("tokens", [])]


def _format_logprob(value: float | None) -> str:
    return "n/a (>20)" if value is None else f"{value:.4f}"


def first_difference(left: list[str], right: list[str]) -> int | None:
    for index, (left_token, right_token) in enumerate(zip(left, right), 1):
        if left_token != right_token:
            return index
    if len(left) != len(right):
        return min(len(left), len(right)) + 1
    return None


def extract_continuation_rows(choice: dict[str, Any], prompt: str, continuation: str) -> list[TokenDetail]:
    logprobs = choice.get("logprobs", {})
    tokens = [str(token) for token in logprobs.get("tokens", [])]
    token_logprobs = list(logprobs.get("token_logprobs", []))
    top_logprobs = list(logprobs.get("top_logprobs", []))
    offsets = list(logprobs.get("text_offset", []))
    start = len(prompt)
    end = len(prompt + continuation)
    rows: list[TokenDetail] = []
    for token, logprob, top, offset in zip(tokens, token_logprobs, top_logprobs, offsets):
        if offset is None or offset < start or offset >= end:
            continue
        rows.append(
            TokenDetail(
                token=token,
                logprob=None if logprob is None else float(logprob),
                top_logprobs=_top_logprob_map(top),
                offset=int(offset),
            )
        )
    return rows


def analyze_generated(choice: dict[str, Any], target: str) -> GeneratedAnalysis:
    text = str(choice.get("text", ""))
    logprobs = choice.get("logprobs", {})
    return GeneratedAnalysis(
        text=text,
        text_preview=_preview(text),
        matches=matches_target(text, target),
        tokens=[str(token) for token in logprobs.get("tokens", [])],
        token_logprobs=[
            None if value is None else float(value) for value in logprobs.get("token_logprobs", [])
        ],
        top_logprobs=[_top_logprob_map(value) for value in logprobs.get("top_logprobs", [])],
    )


def analyze_gold_from_rows(target: str, continuation_pieces: list[str], tokens: list[TokenDetail]) -> GoldAnalysis:
    continuation = "".join(continuation_pieces)
    total_logprob = None if any(token.logprob is None for token in tokens) else sum(
        token.logprob for token in tokens if token.logprob is not None
    )
    first_missing_position = next(
        (
            index
            for index, token in enumerate(tokens, 1)
            if token.logprob is None
        ),
        None,
    )
    first_top1_mismatch = next(
        (
            index
            for index, token in enumerate(tokens, 1)
            if token.top_logprobs and max(token.top_logprobs, key=token.top_logprobs.get) != token.token
        ),
        None,
    )
    return GoldAnalysis(
        continuation=continuation,
        total_logprob=total_logprob,
        tokens=tokens,
        first_missing_position=first_missing_position,
        first_top1_mismatch=first_top1_mismatch,
    )


def first_gold_top1_divergence(left: GoldAnalysis, right: GoldAnalysis) -> int | None:
    for index, (left_token, right_token) in enumerate(zip(left.tokens, right.tokens), 1):
        if not left_token.top_logprobs or not right_token.top_logprobs:
            continue
        if max(left_token.top_logprobs, key=left_token.top_logprobs.get) != max(
            right_token.top_logprobs, key=right_token.top_logprobs.get
        ):
            return index
    return None


def _build_server_script(
    mode: ServerMode, model: str, served_model_name: str, remote_port: int, cuda_visible_devices: str
) -> str:
    env_lines = [f"export {key}={value}" for key, value in mode.extra_env.items()]
    env_block = "\n".join(env_lines)
    return "\n".join(
        [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            "cd /workspace/vllm",
            "source .venv/bin/activate",
            f"export CUDA_VISIBLE_DEVICES={cuda_visible_devices}",
            "export VLLM_DEEP_GEMM_WARMUP=skip",
            env_block,
            "exec /workspace/vllm/.venv/bin/vllm serve "
            f"{model} "
            f"--served-model-name {served_model_name} "
            "--host 127.0.0.1 "
            f"--port {remote_port} "
            "--trust-remote-code "
            "--kv-cache-dtype fp8 "
            "--block-size 256 "
            "--enable-expert-parallel "
            "--tensor-parallel-size 2 "
            "--attention_config.use_fp4_indexer_cache=True "
            "--moe-backend deep_gemm_mega_moe "
            "--tokenizer-mode deepseek_v4 "
            "--tool-call-parser deepseek_v4 "
            "--enable-auto-tool-choice "
            "--reasoning-parser deepseek_v4 "
            "--kernel-config.enable_flashinfer_autotune=False",
        ]
    )


def _stop_remote_server(host: str, container: str, model: str) -> None:
    _ssh(
        host,
        (
            f"docker exec {container} pkill -f "
            f"'/workspace/vllm/.venv/bin/vllm serve {model}' || true"
        ),
    )


def _wait_for_server_down(base_url: str, timeout_seconds: int = 60) -> None:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            requests.get(f"{base_url}/health", timeout=3)
        except requests.RequestException:
            return
        time.sleep(1)


def _wait_for_server_up(base_url: str, timeout_seconds: int) -> None:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            response = requests.get(f"{base_url}/v1/models", timeout=10)
            response.raise_for_status()
            return
        except requests.RequestException:
            time.sleep(2)
    raise RuntimeError(f"Timed out waiting for server at {base_url}.")


def _remote_log_path(mode: ServerMode) -> Path:
    return REMOTE_LOG_DIR / mode.log_name


def _start_remote_server(
    host: str,
    container: str,
    mode: ServerMode,
    model: str,
    served_model_name: str,
    remote_port: int,
    base_url: str,
    cuda_visible_devices: str,
) -> Path:
    script_path = REMOTE_LOG_DIR / f"start_deepseek_logprob_{mode.name}_{remote_port}.sh"
    script_text = _build_server_script(mode, model, served_model_name, remote_port, cuda_visible_devices)
    _stop_remote_server(host, container, model)
    _wait_for_server_down(base_url)
    _ssh(
        host,
        f"docker exec -i {container} bash -lc 'cat > {script_path} && chmod +x {script_path}'",
        input_text=script_text,
    )
    log_path = _remote_log_path(mode)
    _ssh(host, f"docker exec {container} bash -lc 'rm -f {log_path}'")
    _ssh(host, f"docker exec -d {container} bash -lc '{script_path} > {log_path} 2>&1'")
    _wait_for_server_up(base_url, SERVER_TIMEOUT_SECONDS)
    return log_path


def _require_funnel_markers(host: str, container: str, log_path: Path) -> None:
    for marker in FUNNEL_MARKERS:
        _ssh(host, f"docker exec {container} bash -lc \"grep -F {marker!r} {str(log_path)!r}\"")


def _completion_request(base_url: str, prompt: str, max_tokens: int, *, logprobs: int = 20) -> dict[str, Any]:
    payload = {
        "model": DEFAULT_SERVED_MODEL,
        "prompt": prompt,
        "temperature": 0,
        "max_tokens": max_tokens,
        "logprobs": logprobs,
    }
    response = requests.post(f"{base_url}/v1/completions", json=payload, timeout=REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()
    data = response.json()
    choices = data.get("choices", [])
    if not choices:
        raise RuntimeError(f"Missing choices in completion response: {data}")
    if "logprobs" not in choices[0]:
        raise RuntimeError("Completion response did not include logprobs.")
    return choices[0]


def _load_cases(sample_path: Path, miss_ids: tuple[int, ...], control_ids: tuple[int, ...]) -> list[SampleCase]:
    chosen = {sample_id: "miss" for sample_id in miss_ids} | {sample_id: "control" for sample_id in control_ids}
    rows: list[SampleCase] = []
    for index, line in enumerate(sample_path.read_text().splitlines(), 1):
        if index not in chosen:
            continue
        raw = json.loads(line)
        target = raw["target"]
        if isinstance(target, str) and target.startswith("["):
            target = ast.literal_eval(target)
        rows.append(
            SampleCase(
                sample_id=index,
                kind=chosen[index],
                prompt=raw["arguments"]["gen_args_0"]["arg_0"],
                target=target[0] if isinstance(target, list) else str(target),
                prompt_hash=raw["prompt_hash"],
            )
        )
    if len(rows) != len(chosen):
        missing = sorted(set(chosen) - {row.sample_id for row in rows})
        raise RuntimeError(f"Missing sample ids in {sample_path}: {missing}")
    return sorted(rows, key=lambda row: row.sample_id)


def _tokenize_cases(
    tokenizer_python: Path,
    tokenizer_name: str,
    cases: list[SampleCase],
) -> list[TokenizedCase]:
    code = """
import json
import sys
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(sys.argv[1], trust_remote_code=True)
texts = json.load(sys.stdin)
rows = []
for text in texts:
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    rows.append([tokenizer.decode([token_id], clean_up_tokenization_spaces=False) for token_id in token_ids])
json.dump(rows, sys.stdout)
"""
    continuations = [f" {case.target}" for case in cases]
    result = _run(
        [str(tokenizer_python), "-c", code, tokenizer_name],
        input_text=json.dumps(continuations),
    )
    pieces = json.loads(result.stdout)
    return [
        TokenizedCase(case=case, continuation_pieces=list(case_pieces))
        for case, case_pieces in zip(cases, pieces, strict=True)
    ]


def _score_gold_tokens(base_url: str, prompt: str, continuation_pieces: list[str]) -> GoldAnalysis:
    prefix = prompt
    rows: list[TokenDetail] = []
    for piece in continuation_pieces:
        choice = _completion_request(base_url, prefix, 1)
        logprobs = choice.get("logprobs", {})
        top_logprobs = _top_logprob_map((logprobs.get("top_logprobs") or [{}])[0])
        sampled_tokens = _response_tokens(choice)
        sampled_token = sampled_tokens[0] if sampled_tokens else ""
        sampled_logprobs = logprobs.get("token_logprobs") or [None]
        rows.append(
            TokenDetail(
                token=piece,
                logprob=top_logprobs.get(piece)
                if piece in top_logprobs
                else (
                    float(sampled_logprobs[0])
                    if sampled_token == piece and sampled_logprobs[0] is not None
                    else None
                ),
                top_logprobs=top_logprobs,
                offset=None,
            )
        )
        prefix += piece
    return analyze_gold_from_rows("".join(continuation_pieces).strip(), continuation_pieces, rows)


def _analyze_case(base_url: str, tokenized_case: TokenizedCase) -> ModeCaseResult:
    generated_choice = _completion_request(base_url, tokenized_case.case.prompt, 16)
    return ModeCaseResult(
        generated=analyze_generated(generated_choice, tokenized_case.case.target),
        gold=_score_gold_tokens(base_url, tokenized_case.case.prompt, tokenized_case.continuation_pieces),
    )


def _summarize_case(case: SampleCase, baseline: ModeCaseResult, funnel: ModeCaseResult) -> CaseComparison:
    note = "both match"
    if baseline.generated.matches and not funnel.generated.matches:
        note = "baseline-only match"
    elif not baseline.generated.matches and funnel.generated.matches:
        note = "funnel-only match"
    elif not baseline.generated.matches and not funnel.generated.matches:
        note = "both miss"
    return CaseComparison(
        sample_id=case.sample_id,
        kind=case.kind,
        target=case.target,
        baseline=baseline,
        funnel=funnel,
        gold_logprob_delta=(
            None
            if baseline.gold.total_logprob is None or funnel.gold.total_logprob is None
            else baseline.gold.total_logprob - funnel.gold.total_logprob
        ),
        first_generation_divergence=first_difference(baseline.generated.tokens, funnel.generated.tokens),
        first_gold_top1_divergence=first_gold_top1_divergence(baseline.gold, funnel.gold),
        note=note,
    )


def _mode_summary(title: str, log_path: Path, results: list[ModeCaseResult], marker_status: str | None = None) -> ModeSummary:
    gold_logprobs = [result.gold.total_logprob for result in results if result.gold.total_logprob is not None]
    return ModeSummary(
        title=title,
        log_path=str(log_path),
        matched=sum(result.generated.matches for result in results),
        avg_gold_logprob=None if not gold_logprobs else sum(gold_logprobs) / len(gold_logprobs),
        marker_status=marker_status,
    )


def _write_outputs(
    sample_path: Path,
    output_path: Path,
    report_path: Path,
    local_port: int,
    remote_port: int,
    cases: list[CaseComparison],
    mode_summaries: list[ModeSummary],
) -> None:
    payload = {
        "sample_path": str(sample_path),
        "generated_at": datetime.now().isoformat(),
        "local_port": local_port,
        "remote_port": remote_port,
        "modes": [asdict(summary) for summary in mode_summaries],
        "cases": [asdict(case) for case in cases],
    }
    output_path.write_text(json.dumps(payload, indent=2))
    report_path.write_text(
        REPORT_TEMPLATE.render(
            sample_path=sample_path,
            local_port=local_port,
            remote_port=remote_port,
            miss_count=sum(case.kind == "miss" for case in cases),
            control_count=sum(case.kind == "control" for case in cases),
            modes=[
                {
                    **asdict(mode),
                    "avg_gold_logprob_text": _format_logprob(mode.avg_gold_logprob),
                }
                for mode in mode_summaries
            ],
            cases=[
                {
                    **asdict(case),
                    "baseline_gold_logprob_text": _format_logprob(case.baseline.gold.total_logprob),
                    "funnel_gold_logprob_text": _format_logprob(case.funnel.gold.total_logprob),
                    "gold_logprob_delta_text": _format_logprob(case.gold_logprob_delta),
                }
                for case in cases
            ],
        )
    )


@app.command("analyze-ruler-128k")
def analyze_ruler_128k(
    sample_path: Path = typer.Option(DEFAULT_SAMPLE_PATH, exists=True, dir_okay=False),
    report_path: Path = typer.Option(DEFAULT_REPORT_PATH, dir_okay=False),
    output_path: Path = typer.Option(
        REPO_ROOT / f"output/ruler_niah_single_1_128k_logprob_analysis_{_now_stamp()}.json",
        dir_okay=False,
    ),
    host: str = typer.Option(DEFAULT_HOST),
    container: str = typer.Option(DEFAULT_CONTAINER),
    model: str = typer.Option(DEFAULT_MODEL),
    served_model_name: str = typer.Option(DEFAULT_SERVED_MODEL),
    local_port: int = typer.Option(DEFAULT_LOCAL_PORT),
    remote_port: int = typer.Option(DEFAULT_REMOTE_PORT),
    tokenizer_python: Path = typer.Option(DEFAULT_TOKENIZER_PYTHON, exists=True, dir_okay=False),
    tokenizer_name: str = typer.Option(DEFAULT_TOKENIZER_NAME),
    cuda_visible_devices: str = typer.Option(DEFAULT_CUDA_VISIBLE_DEVICES),
    miss_ids: str = typer.Option(",".join(str(value) for value in DEFAULT_MISS_IDS)),
    control_ids: str = typer.Option(",".join(str(value) for value in DEFAULT_CONTROL_IDS)),
    keep_server_running: bool = typer.Option(False),
) -> None:
    miss_set = tuple(int(value) for value in miss_ids.split(",") if value)
    control_set = tuple(int(value) for value in control_ids.split(",") if value)
    cases = _load_cases(sample_path, miss_set, control_set)
    tokenized_cases = _tokenize_cases(tokenizer_python, tokenizer_name, cases)
    base_url = f"http://127.0.0.1:{local_port}"
    _ensure_tunnel(host, local_port, remote_port, DEFAULT_TUNNEL_PIDFILE)
    mode_results: dict[str, list[ModeCaseResult]] = {}
    mode_logs: dict[str, Path] = {}
    try:
        for mode in (BASELINE_MODE, FUNNEL_MODE):
            logger.info("Starting %s server on %s", mode.title, host)
            log_path = _start_remote_server(
                host, container, mode, model, served_model_name, remote_port, base_url, cuda_visible_devices
            )
            results = [_analyze_case(base_url, case) for case in tokenized_cases]
            if mode.expect_funnel_markers:
                _require_funnel_markers(host, container, log_path)
            mode_results[mode.name] = results
            mode_logs[mode.name] = log_path
        comparisons = [
            _summarize_case(case, baseline, funnel)
            for case, baseline, funnel in zip(cases, mode_results["baseline"], mode_results["funnel_turbo"])
        ]
        mode_summaries = [
            _mode_summary(BASELINE_MODE.title, mode_logs["baseline"], mode_results["baseline"]),
            _mode_summary(
                FUNNEL_MODE.title,
                mode_logs["funnel_turbo"],
                mode_results["funnel_turbo"],
                marker_status="confirmed",
            ),
        ]
        _write_outputs(sample_path, output_path, report_path, local_port, remote_port, comparisons, mode_summaries)
        logger.info("Wrote JSON analysis to %s", output_path)
        logger.info("Wrote report to %s", report_path)
    finally:
        if not keep_server_running:
            _stop_remote_server(host, container, model)
        _stop_tunnel(host, local_port, remote_port, DEFAULT_TUNNEL_PIDFILE)


if __name__ == "__main__":
    app()
