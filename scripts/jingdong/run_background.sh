#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
log_dir="$HOME/logs/jingdong"
mkdir -p "$log_dir"
run_id="$(date +%Y%m%d_%H%M%S)_$$"
log_file="$log_dir/pipeline_$run_id.log"
pid_file="$log_dir/pipeline_$run_id.pid"

if [ "$#" -eq 0 ]; then
    echo "Usage: bash scripts/jingdong/run_background.sh --batch-id 1 --delete-local"
    exit 1
fi

cd "$repo_dir"
nohup /opt/ego/bin/python -u scripts/jingdong/pipeline.py "$@" \
    </dev/null >"$log_file" 2>&1 &
pipeline_pid=$!
echo "$pipeline_pid" >"$pid_file"
echo "PID: $pipeline_pid"
echo "Log: $log_file"
echo "PID file: $pid_file"
echo "Follow log: tail -f $log_file"
