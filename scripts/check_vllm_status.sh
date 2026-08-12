#!/usr/bin/env bash
set -euo pipefail

host=""
local_base_url="${LOCAL_BASE_URL:-http://127.0.0.1:18000}"
remote_port="${REMOTE_PORT:-8000}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --host)
            host="${2:?missing host value}"
            shift 2
            ;;
        --local-base-url)
            local_base_url="${2:?missing local base url}"
            shift 2
            ;;
        --remote-port)
            remote_port="${2:?missing remote port}"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 2
            ;;
    esac
done

metric_value() {
    local key="$1"
    local metrics="$2"
    awk -v key="$key" '$1 ~ key { print $2; exit }' <<<"$metrics"
}

print_status() {
    local label="$1"
    local version_json="$2"
    local models_json="$3"
    local load_json="$4"
    local metrics_text="$5"

    local version model load running waiting kv awake stop_count length_count error_count
    version="$(jq -r '.version // "unknown"' <<<"$version_json")"
    model="$(jq -r '.data[0].id // "unknown"' <<<"$models_json")"
    load="$(jq -c '.' <<<"$load_json")"
    running="$(metric_value '^vllm:num_requests_running' "$metrics_text")"
    waiting="$(metric_value '^vllm:num_requests_waiting{' "$metrics_text")"
    kv="$(metric_value '^vllm:kv_cache_usage_perc' "$metrics_text")"
    awake="$(metric_value 'sleep_state="awake"' "$metrics_text")"
    stop_count="$(metric_value 'finished_reason="stop"' "$metrics_text")"
    length_count="$(metric_value 'finished_reason="length"' "$metrics_text")"
    error_count="$(metric_value 'finished_reason="error"' "$metrics_text")"

    echo "[$label]"
    echo "health: ok"
    echo "version: $version"
    echo "model: $model"
    echo "load: $load"
    echo "requests_running: ${running:-unknown}"
    echo "requests_waiting: ${waiting:-unknown}"
    echo "kv_cache_usage: ${kv:-unknown}"
    echo "engine_awake: ${awake:-unknown}"
    echo "requests_finished_stop: ${stop_count:-unknown}"
    echo "requests_finished_length: ${length_count:-unknown}"
    echo "requests_finished_error: ${error_count:-unknown}"
    echo
}

fetch_local() {
    curl -fsS "$local_base_url/health" >/dev/null
    print_status \
        "local:$local_base_url" \
        "$(curl -fsS "$local_base_url/version")" \
        "$(curl -fsS "$local_base_url/v1/models")" \
        "$(curl -fsS "$local_base_url/load")" \
        "$(curl -fsS "$local_base_url/metrics")"
}

fetch_remote() {
    local path="$1"
    ssh -q "$host" "curl -fsS http://127.0.0.1:$remote_port$path"
}

fetch_remote_health() {
    ssh -q "$host" "curl -fsS http://127.0.0.1:$remote_port/health >/dev/null"
}

fetch_remote_status() {
    fetch_remote_health
    print_status \
        "remote:$host:127.0.0.1:$remote_port" \
        "$(fetch_remote /version)" \
        "$(fetch_remote /v1/models)" \
        "$(fetch_remote /load)" \
        "$(fetch_remote /metrics)"
}

fetch_local
if [[ -n "$host" ]]; then
    fetch_remote_status
fi
