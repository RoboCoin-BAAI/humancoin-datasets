#!/usr/bin/env bash
# 批量对 session-episode/* 做深度转化：源目录只读，深度写到独立输出树。
#
# 用法:
#   bash scripts/run_sessions_depth_batch.sh \
#     "/root/data/网盘L3数据/双手肌电 + MANUS/session-episode" \
#     "/root/data/网盘L3数据/双手肌电 + MANUS_depth/session-episode"
#
# 环境变量:
#   CAMERAS=head,eye          默认 head,eye
#   SKIP_RECTIFIED=1          跳过 rectified
#   START_FROM=1              从第 N 个 session 开始（1-based）
#   DRY_RUN=1                 只打印
#   DA3_PROCESS_RES=504
#   DA3_CHUNK_SIZE=420
#   DA3_OVERLAP=140
#   LOG_DIR=logs/sessions_depth

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

SRC_SESSIONS="${1:?用法: $0 <session-episode源目录> <输出session-episode目录>}"
OUT_SESSIONS="${2:?用法: $0 <session-episode源目录> <输出session-episode目录>}"

CAMERAS="${CAMERAS:-head,eye}"
SKIP_RECTIFIED="${SKIP_RECTIFIED:-0}"
START_FROM="${START_FROM:-1}"
DRY_RUN="${DRY_RUN:-0}"
DA3_PROCESS_RES="${DA3_PROCESS_RES:-504}"
DA3_CHUNK_SIZE="${DA3_CHUNK_SIZE:-420}"
DA3_OVERLAP="${DA3_OVERLAP:-140}"
LOG_DIR="${LOG_DIR:-$ROOT/logs/sessions_depth}"
mkdir -p "$LOG_DIR" "$OUT_SESSIONS"

mapfile -t SESSIONS < <(find "$SRC_SESSIONS" -mindepth 1 -maxdepth 1 -type d | sort)
TOTAL=${#SESSIONS[@]}
echo "sessions=$TOTAL  cameras=$CAMERAS  skip_rectified=$SKIP_RECTIFIED"
echo "src=$SRC_SESSIONS"
echo "out=$OUT_SESSIONS"

idx=0
for sess in "${SESSIONS[@]}"; do
  idx=$((idx + 1))
  if (( idx < START_FROM )); then
    continue
  fi
  name="$(basename "$sess")"
  if [[ ! -d "$sess/videos" ]]; then
    echo "[$idx/$TOTAL] SKIP (no videos): $name"
    continue
  fi
  out="$OUT_SESSIONS/$name"
  log="$LOG_DIR/${name}.log"
  echo "[$idx/$TOTAL] $name → $out"

  cmd=(
    python scripts/add_depth_fs_parallel.py
    --dataset-dir "$sess"
    --output-dir "$out"
    --cameras "$CAMERAS"
    --strategy da3
    --da3-model depth-anything/DA3NESTED-GIANT-LARGE-1.1
    --da3-process-res "$DA3_PROCESS_RES"
    --da3-chunk-size "$DA3_CHUNK_SIZE"
    --da3-overlap "$DA3_OVERLAP"
    --crf 6
    --preset slow
    --log-file "$log"
  )
  if [[ "$SKIP_RECTIFIED" == "1" ]]; then
    cmd+=(--skip-rectified)
  fi

  if [[ "$DRY_RUN" == "1" ]]; then
    printf '  DRY: '; printf '%q ' "${cmd[@]}"; echo
    continue
  fi
  "${cmd[@]}"
done

echo "All done."
