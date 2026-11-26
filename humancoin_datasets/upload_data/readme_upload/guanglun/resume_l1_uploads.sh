#!/usr/bin/env bash
# Run in a terminal with hub credentials/proxy already configured. Never print secrets.
set -uo pipefail
platform="${1:?usage: resume_l1_uploads.sh hf|ms [guanglun|jianzhi|both]}"
scope="${2:-both}"
case "$platform" in hf|ms) ;; *) exit 2 ;; esac
case "$scope" in guanglun|jianzhi|both) ;; *) exit 2 ;; esac
exec 9>"/tmp/humancoin-l1-upload-${platform}.lock"
flock -n 9 || { printf 'An upload launcher for %s is already running.\n' "$platform"; exit 1; }
cd /share/project/liuyou/humancoin/code/roboego-hand-vis || exit 1
base=/share/project/liuyou/humancoin/data
python=/share/project/liuyou/humancoin/envs/ego/bin/python
stamp="$(date +%Y%m%d-%H%M%S)-$$"
failed=0
for dataset in guanglun jianzhi; do
  [[ "$scope" == both || "$scope" == "$dataset" ]] || continue
  if [[ "$dataset" == guanglun ]]; then
    report="$base/guanglun_50_72_publish_20261122"
  else
    report="$base/jianzhi_72_91_publish_20261123"
  fi
  mkdir -p "$report/upload_logs" || exit 1
  log="$report/upload_logs/${platform}-upload-${stamp}.log"
  printf 'Uploading %s to %s; log: %s\n' "$dataset" "$platform" "$log"
  "$python" -u -m humancoin_datasets.upload_data.readme_upload.upload2hub \
    --config "$report/upload_${platform}.json" \
    "--resume-${platform}-log-dir" "$report/upload_logs" >"$log" 2>&1
  code=$?
  printf 'Finished %s/%s, exit=%s\n' "$dataset" "$platform" "$code"
  if (( code != 0 )); then failed=1; fi
done
exit "$failed"
