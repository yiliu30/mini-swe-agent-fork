import json
import os
import signal
import subprocess
from pathlib import Path

import pytest
from typer.testing import CliRunner

from minisweagent.run.utilities.deepseek_local import (
    DEFAULT_CONFIG_NAME,
    _config_spec,
    _default_output_dir,
    _default_slice_spec,
    _get_live_tunnel_pid,
    _preds_json_to_jsonl,
    _remote_models_command,
    _stop_tunnel,
    app,
)


def test_default_slice_spec():
    assert _default_slice_spec(10) == "0:10"


def test_default_output_dir():
    output_dir = _default_output_dir()
    assert output_dir.parent == Path("output")
    assert output_dir.name.startswith("deepseek_v4_b200_verified_10_")


def test_preds_json_to_jsonl_filters_empty_patches(tmp_path):
    preds_path = tmp_path / "preds.json"
    preds_path.write_text(
        json.dumps(
            {
                "has_patch": {"instance_id": "has_patch", "model_patch": "diff --git a b", "model_name_or_path": "x"},
                "empty_patch": {"instance_id": "empty_patch", "model_patch": "", "model_name_or_path": "x"},
            }
        )
    )
    jsonl_path = tmp_path / "preds.jsonl"

    assert _preds_json_to_jsonl(preds_path, jsonl_path) == 1
    lines = jsonl_path.read_text().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["instance_id"] == "has_patch"


def test_get_live_tunnel_pid_cleans_stale_pidfile(tmp_path):
    pidfile = tmp_path / "tunnel.pid"
    pidfile.write_text("999999")

    assert _get_live_tunnel_pid(pidfile=pidfile) is None
    assert not pidfile.exists()


def test_remote_models_command():
    command = _remote_models_command("b200-aws", 8000)
    assert command[:4] == ["ssh", "-o", "BatchMode=yes", "b200-aws"]
    assert "http://127.0.0.1:8000/v1/models" in command[4]


def test_stop_tunnel_without_running_process(tmp_path):
    pidfile = tmp_path / "tunnel.pid"
    pidfile.write_text("999999")

    assert _stop_tunnel(pidfile=pidfile) is False
    assert not pidfile.exists()


def test_config_spec_uses_selected_config_name():
    assert _config_spec("swebench_deepseek_hh_b200.yaml") == [
        "swebench.yaml",
        "swebench_deepseek_hh_b200.yaml",
        "agent.mode=yolo",
    ]


def test_get_live_tunnel_pid_matches_current_process(tmp_path):
    pidfile = tmp_path / "tunnel.pid"
    proc = subprocess.Popen(
        [
            "python3",
            "-c",
            "import os, signal, time; signal.signal(signal.SIGTERM, lambda *_: exit(0)); time.sleep(60)",
            "ssh",
            "-N",
            "b200-aws",
            "127.0.0.1:18000:127.0.0.1:8000",
        ]
    )
    try:
        pidfile.write_text(str(proc.pid))
        assert _get_live_tunnel_pid(pidfile=pidfile) == proc.pid
    finally:
        os.kill(proc.pid, signal.SIGTERM)
        proc.wait(timeout=5)


def test_cli_help():
    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "split-node SWE-bench runs" in result.stdout


@pytest.mark.parametrize(
    ("command",),
    [("start-tunnel",), ("check",), ("stop-tunnel",), ("swebench",)],
)
def test_subcommand_help(command):
    runner = CliRunner()
    result = runner.invoke(app, [command, "--help"])
    assert result.exit_code == 0
    assert "--help" in result.stdout


def test_swebench_help_shows_config_name_option():
    runner = CliRunner()
    result = runner.invoke(app, ["swebench", "--help"])
    assert result.exit_code == 0
    assert "--config-name" in result.stdout
    assert DEFAULT_CONFIG_NAME in result.stdout
