#!/usr/bin/env python3

"""Manage remote DeepSeek vLLM access for split-node SWE-bench runs."""

import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from urllib.request import urlopen

import typer

from minisweagent.run.benchmarks.swebench import main as swebench_main

app = typer.Typer(help=__doc__, rich_markup_mode="rich", add_completion=False, no_args_is_help=True)

DEFAULT_HOST = "b200-aws"
DEFAULT_LOCAL_PORT = 18000
DEFAULT_REMOTE_PORT = 8000
DEFAULT_WORKERS = 2
DEFAULT_COUNT = 10
DEFAULT_RUN_ID = "deepseek-v4-b200"
DEFAULT_CONFIG_NAME = "swebench_deepseek_b200.yaml"
PIDFILE = Path("/tmp/minisweagent-deepseek-b200-tunnel.pid")


def _default_slice_spec(count: int) -> str:
    return f"0:{count}"


def _default_output_dir(count: int = DEFAULT_COUNT, parent: Path = Path("output")) -> Path:
    prefix = f"deepseek_v4_b200_verified_{count}"
    return parent / f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def _config_spec(config_name: str = DEFAULT_CONFIG_NAME) -> list[str]:
    return ["swebench.yaml", config_name, "agent.mode=yolo"]


def _preds_json_to_jsonl(preds_path: Path, jsonl_path: Path) -> int:
    preds = json.loads(preds_path.read_text()) if preds_path.exists() else {}
    kept = 0
    with jsonl_path.open("w") as f:
        for pred in preds.values():
            if pred.get("model_patch"):
                f.write(json.dumps(pred) + "\n")
                kept += 1
    return kept


def _tunnel_spec(local_port: int, remote_port: int) -> str:
    return f"127.0.0.1:{local_port}:127.0.0.1:{remote_port}"


def _read_pidfile(pidfile: Path = PIDFILE) -> int | None:
    if not pidfile.exists():
        return None
    try:
        return int(pidfile.read_text().strip())
    except ValueError:
        pidfile.unlink(missing_ok=True)
        return None


def _pid_is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _pid_matches_tunnel(pid: int, host: str, local_port: int, remote_port: int) -> bool:
    cmdline_path = Path(f"/proc/{pid}/cmdline")
    if not cmdline_path.exists():
        return False
    cmdline = cmdline_path.read_text().replace("\x00", " ")
    return (
        "ssh" in cmdline
        and host in cmdline
        and "-N" in cmdline
        and _tunnel_spec(local_port, remote_port) in cmdline
    )


def _get_live_tunnel_pid(
    host: str = DEFAULT_HOST,
    local_port: int = DEFAULT_LOCAL_PORT,
    remote_port: int = DEFAULT_REMOTE_PORT,
    pidfile: Path = PIDFILE,
) -> int | None:
    pid = _read_pidfile(pidfile)
    if pid is None:
        return None
    if _pid_is_running(pid) and _pid_matches_tunnel(pid, host, local_port, remote_port):
        return pid
    pidfile.unlink(missing_ok=True)
    return None


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True)


def _require_ok(result: subprocess.CompletedProcess[str], description: str) -> None:
    if result.returncode == 0:
        return
    raise RuntimeError(f"{description} failed:\n{result.stderr or result.stdout}")


def _http_status(url: str, timeout: int = 30) -> int:
    with urlopen(url, timeout=timeout) as response:
        return response.status


def _check_ssh(host: str) -> None:
    _require_ok(_run(["ssh", "-o", "BatchMode=yes", host, "true"]), f"SSH to {host}")


def _remote_models_command(host: str, remote_port: int) -> list[str]:
    return [
        "ssh",
        "-o",
        "BatchMode=yes",
        host,
        (
            "python3 -c "
            f"\"from urllib.request import urlopen; "
            f"print(urlopen('http://127.0.0.1:{remote_port}/v1/models', timeout=30).status)\""
        ),
    ]


def _check_remote_models(host: str, remote_port: int) -> int:
    result = _run(_remote_models_command(host, remote_port))
    _require_ok(result, f"Remote health check on {host}")
    return int(result.stdout.strip())


def _check_local_models(local_port: int) -> int:
    return _http_status(f"http://127.0.0.1:{local_port}/v1/models")


def _start_tunnel_process(host: str, local_port: int, remote_port: int) -> subprocess.Popen:
    return subprocess.Popen(
        [
            "ssh",
            "-N",
            "-L",
            _tunnel_spec(local_port, remote_port),
            "-o",
            "ExitOnForwardFailure=yes",
            host,
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


def _ensure_tunnel(
    host: str = DEFAULT_HOST,
    local_port: int = DEFAULT_LOCAL_PORT,
    remote_port: int = DEFAULT_REMOTE_PORT,
    pidfile: Path = PIDFILE,
) -> int:
    if pid := _get_live_tunnel_pid(host, local_port, remote_port, pidfile):
        return pid
    process = _start_tunnel_process(host, local_port, remote_port)
    time.sleep(1)
    if process.poll() is not None:
        raise RuntimeError(f"SSH tunnel to {host} exited before becoming ready with code {process.returncode}.")
    pidfile.write_text(str(process.pid))
    return process.pid


def _stop_tunnel(
    host: str = DEFAULT_HOST,
    local_port: int = DEFAULT_LOCAL_PORT,
    remote_port: int = DEFAULT_REMOTE_PORT,
    pidfile: Path = PIDFILE,
) -> bool:
    pid = _get_live_tunnel_pid(host, local_port, remote_port, pidfile)
    if pid is None:
        pidfile.unlink(missing_ok=True)
        return False
    os.kill(pid, signal.SIGTERM)
    for _ in range(20):
        if not _pid_is_running(pid):
            pidfile.unlink(missing_ok=True)
            return True
        time.sleep(0.1)
    os.kill(pid, signal.SIGKILL)
    pidfile.unlink(missing_ok=True)
    return True


def _run_scoring(output_dir: Path, split: str, run_id: str, max_workers: int) -> None:
    preds_jsonl = output_dir / "preds.jsonl"
    kept = _preds_json_to_jsonl(output_dir / "preds.json", preds_jsonl)
    if kept == 0:
        raise RuntimeError(f"No non-empty predictions found in {output_dir / 'preds.json'}")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "swebench.harness.run_evaluation",
            "--dataset_name",
            "princeton-nlp/SWE-Bench_Verified",
            "--predictions_path",
            str(preds_jsonl),
            "--split",
            split,
            "--max_workers",
            str(max_workers),
            "--run_id",
            run_id,
        ],
        check=True,
    )


@app.command("start-tunnel")
def start_tunnel(
    host: str = typer.Option(DEFAULT_HOST, help="SSH host that can reach the DeepSeek server"),
    local_port: int = typer.Option(DEFAULT_LOCAL_PORT, help="Local forwarded port"),
    remote_port: int = typer.Option(DEFAULT_REMOTE_PORT, help="Remote vLLM port"),
):
    """Start an SSH tunnel to the remote DeepSeek server."""
    pid = _ensure_tunnel(host, local_port, remote_port)
    typer.echo(f"Tunnel running with pid {pid}: http://127.0.0.1:{local_port}/v1")


@app.command()
def check(
    host: str = typer.Option(DEFAULT_HOST, help="SSH host that can reach the DeepSeek server"),
    local_port: int = typer.Option(DEFAULT_LOCAL_PORT, help="Local forwarded port"),
    remote_port: int = typer.Option(DEFAULT_REMOTE_PORT, help="Remote vLLM port"),
):
    """Check SSH reachability and DeepSeek server health."""
    _check_ssh(host)
    typer.echo(f"SSH to {host}: OK")
    typer.echo(f"Remote /v1/models: {_check_remote_models(host, remote_port)}")
    pid = _get_live_tunnel_pid(host, local_port, remote_port)
    if pid is None:
        typer.echo("Local tunnel: not running")
        return
    typer.echo(f"Local tunnel pid: {pid}")
    typer.echo(f"Local /v1/models: {_check_local_models(local_port)}")


@app.command("stop-tunnel")
def stop_tunnel(
    host: str = typer.Option(DEFAULT_HOST, help="SSH host that can reach the DeepSeek server"),
    local_port: int = typer.Option(DEFAULT_LOCAL_PORT, help="Local forwarded port"),
    remote_port: int = typer.Option(DEFAULT_REMOTE_PORT, help="Remote vLLM port"),
):
    """Stop the SSH tunnel."""
    stopped = _stop_tunnel(host, local_port, remote_port)
    typer.echo("Tunnel stopped" if stopped else "Tunnel not running")


@app.command()
def swebench(
    count: int = typer.Argument(DEFAULT_COUNT, help="Number of instances to run from the start of the split"),
    workers: int = typer.Option(DEFAULT_WORKERS, "-w", "--workers", help="SWE-bench worker threads"),
    output: str = typer.Option("", "-o", "--output", help="Output directory"),
    subset: str = typer.Option("verified", help="SWE-bench subset"),
    split: str = typer.Option("test", help="Dataset split"),
    host: str = typer.Option(DEFAULT_HOST, help="SSH host that can reach the DeepSeek server"),
    local_port: int = typer.Option(DEFAULT_LOCAL_PORT, help="Local forwarded port"),
    remote_port: int = typer.Option(DEFAULT_REMOTE_PORT, help="Remote vLLM port"),
    run_id: str = typer.Option(DEFAULT_RUN_ID, help="Run id for local SWE-bench evaluation"),
    config_name: str = typer.Option(DEFAULT_CONFIG_NAME, help="Benchmark config merged on top of swebench.yaml"),
):
    """Run SWE-bench locally against the remote DeepSeek server and score it."""
    _check_ssh(host)
    _check_remote_models(host, remote_port)
    pid = _ensure_tunnel(host, local_port, remote_port)
    typer.echo(f"Using tunnel pid {pid}: http://127.0.0.1:{local_port}/v1")
    typer.echo(f"Local /v1/models: {_check_local_models(local_port)}")
    output_dir = Path(output) if output else _default_output_dir(count)
    swebench_main(
        subset=subset,
        split=split,
        slice_spec=_default_slice_spec(count),
        filter_spec="",
        shuffle=False,
        output=str(output_dir),
        workers=workers,
        model=None,
        model_class=None,
        redo_existing=False,
        config_spec=_config_spec(config_name),
        environment_class=None,
    )
    _run_scoring(output_dir, split, run_id, workers)
    typer.echo(f"Finished run and scoring in {output_dir}")


if __name__ == "__main__":
    app()
