"""Batch convert L1 sessions with independent subprocesses and resumable logs."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from format_convert.L1.convert_l1 import discover_sessions, inspect_session, summary, session_duration, duration_allowed

CONVERTER = Path(__file__).with_name("convert_l1.py")


def completed(out: Path, source: Path) -> bool:
    try:
        provenance = json.loads((out / "meta/l1_source.json").read_text(encoding="utf-8"))
        return (provenance["source"] == str(source)
                and (out / "meta/info.json").is_file()
                and (out / "meta/episodes.jsonl").stat().st_size > 0)
    except (OSError, ValueError, KeyError):
        return False


def run_one(source: Path, root: Path, ffmpeg: str, ffprobe: str) -> str:
    out = root / source.name
    try:
        duration = session_duration(source)
    except (OSError, ValueError, KeyError):
        duration = None  # The converter records the detailed validation failure.
    if duration is not None and not duration_allowed(duration):
        record = dict(status='skipped_duration', session=str(out), source=str(source),
                      duration_s=duration, min_s=5., max_s=600.)
        status = root / '_batch_status' / f'{source.name}.json'
        status.parent.mkdir(parents=True, exist_ok=True)
        status.write_text(json.dumps(record, indent=2), encoding='utf-8')
        print(json.dumps(record), flush=True)
        return 'skipped_duration'
    if completed(out, source):
        print(f"[skip-existing] {source.name}", flush=True)
        return "skipped_existing"
    log = root / "_batch_logs" / f"{source.name}.log"
    status = root / "_batch_status" / f"{source.name}.json"
    log.parent.mkdir(parents=True, exist_ok=True)
    status.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, str(CONVERTER), str(source), "--out", str(out),
               "--ffmpeg", ffmpeg, "--ffprobe", ffprobe]
    started = time.monotonic()
    print(f"[start] {source.name}", flush=True)
    try:
        with log.open("w", encoding="utf-8") as f:
            result = subprocess.run(command, stdout=f, stderr=subprocess.STDOUT, check=False)
        code = result.returncode
    except OSError as exc:
        log.write_text(str(exc), encoding="utf-8")
        code = -1
    state = "done" if code == 0 and completed(out, source) else "failed"
    status.write_text(json.dumps(dict(status=state, source=str(source), output=str(out),
        returncode=code, duration_s=time.monotonic() - started, command=command),
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[{state}] {source.name}", flush=True)
    return state


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--limit", type=int)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--ffmpeg", default="ffmpeg")
    p.add_argument("--ffprobe", default="ffprobe")
    args = p.parse_args(argv)
    if args.workers < 1 or (args.limit is not None and args.limit < 1):
        p.error("workers and limit must be positive")
    source, out = args.source.resolve(), args.out.resolve()
    if out == source or out.is_relative_to(source) or source.is_relative_to(out):
        p.error("source and output directories must not overlap")
    sessions = discover_sessions(source)
    if args.limit:
        sessions = sessions[:args.limit]
    if not sessions:
        p.error("no L1 sessions found")
    if len({s.name for s in sessions}) != len(sessions):
        p.error("duplicate session names would share an output directory")
    if args.dry_run:
        failed = 0
        for session in sessions:
            try:
                duration = session_duration(session)
                if not duration_allowed(duration):
                    print(json.dumps(dict(status='skipped_duration', source=str(session), duration_s=duration)))
                    continue
                print(json.dumps(summary(session, inspect_session(session)), ensure_ascii=False, indent=2))
                print(f"[plan] {session} -> {out / session.name}")
            except (OSError, ValueError, KeyError) as exc:
                failed += 1
                print(f"[error] {session}: {exc}", file=sys.stderr)
        return int(bool(failed))
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        states = list(pool.map(lambda s: run_one(s, out, args.ffmpeg, args.ffprobe), sessions))
    counts = {s: states.count(s) for s in ("done", "failed", "skipped_existing", "skipped_duration")}
    print(json.dumps(counts))
    return int(bool(counts["failed"]))


if __name__ == "__main__":
    raise SystemExit(main())


'''
export PATH="/share/project/liuyou/humancoin/envs/ego/bin:$PATH"
/share/project/liuyou/humancoin/envs/ego/bin/python \
  /share/project/liuyou/humancoin/code/Multi-Frequency-LeRobot/format_convert/L1/batch_convert.py \
  --source "/share/project/liuyou/humancoin/data/L1/batch_0000001/" \
  --out "/share/project/liuyou/humancoin/data/L1_lerobot/batch_0000001/" \
  --workers 16
'''
