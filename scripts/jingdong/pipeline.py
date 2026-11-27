"""Process a Jingdong batch with per-instance stage records and upload gating."""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from humancoin_datasets.download_data.base.batch import batch_name
from humancoin_datasets.download_data.base.run import read_batch
from humancoin_datasets.upload_data.ks3 import KS3, atomic_json, prefix
from humancoin_datasets.upload_data.jingdong.run import plan as upload_plan

STAGES = ("download", "convert", "load", "quality", "annotate", "upload")


def now():
    return datetime.now(timezone.utc).isoformat()


def command(arguments):
    subprocess.run([str(value) for value in arguments], cwd=REPO, check=True)


@contextmanager
def batch_lock(path):
    import fcntl
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield


def process_instance(uri, args, client, report):
    from humancoin_datasets.download_data.downloader import KS3Downloader
    from humancoin_datasets.upload_data.readme_upload.check_load import validate_dataset_load

    name = uri.rstrip("/").rsplit("/", 1)[-1]
    import re
    if not re.fullmatch(r"instance-[A-Za-z0-9]+", name):
        raise ValueError(f"Expected a Jingdong instance directory: {uri}")
    raw = args.raw_root / args.batch_id / name
    converted = args.output_root / args.batch_id / name
    annotation = converted / "annotation/episode_000000_structured.json"
    record_path = report / (name + ".json")
    identity = dict(source=uri, raw=str(raw), output=str(converted),
                    destination=args.destination, model=args.model, sample_fps=args.sample_fps)
    identity_hash = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    record = json.loads(record_path.read_text()) if record_path.exists() else dict(
        instance=name, identity=identity, identity_hash=identity_hash,
        stages={stage: {"status": "pending"} for stage in STAGES})
    if record.get("identity_hash") != identity_hash:
        raise ValueError(f"Existing pipeline record has different settings: {record_path}")
    # Resume only a recorded, complete run whose published local dataset still exists.
    if (record.get("status") == "done" and annotation.is_file()
            and (converted / "meta/info.json").is_file()):
        print(f"REUSE {name}: all stages completed", flush=True)
        return record
    downloader = KS3Downloader(raw.parent, args.ks3_tool, args.ks3_config, args.jobs)
    actions = {
        "download": lambda: downloader.download(uri),
        "load": lambda: validate_dataset_load(
            converted, "HumanCOIN/" + name,
            json.loads((converted / "meta/info.json").read_text())),
        "upload": lambda: client.upload_directory(
            converted, upload_plan(converted, args.destination)[0]["destination"],
            recursive_copy=True, exclude=("annotation/argus",)),
    }
    artifacts = {
        "download": raw / ".ks3_download.json",
        "convert": converted / "meta/info.json",
        "annotate": annotation,
    }
    record["status"] = "running"
    for stage in STAGES:
        previous = record["stages"][stage]
        if stage == "quality":
            record["stages"][stage] = dict(status="not_configured")
            atomic_json(record_path, record)
            continue
        if stage in artifacts and artifacts[stage].is_file():
            record["stages"][stage] = dict(
                previous, status="done", reused=True,
                completion_basis="existing_artifact", artifact=str(artifacts[stage]))
            atomic_json(record_path, record)
            print(f"REUSE {name}: {stage}", flush=True)
            continue
        record["stages"][stage] = dict(status="running", started_at=now(),
                                      attempts=previous.get("attempts", 0) + 1)
        atomic_json(record_path, record)
        print(f"{name}: {stage}", flush=True)
        log = report / (name + "_" + stage + ".log")
        try:
            # Keep subprocess diagnostics per instance, including errors before model dispatch.
            if stage in ("convert", "annotate"):
                with log.open("a", encoding="utf-8") as stream:
                    run_logged_stage(stage, args, raw, converted, annotation, stream)
            else:
                result = actions[stage]()
                if stage == "upload":
                    record["stages"][stage]["result"] = result
            record["stages"][stage].update(status="done", finished_at=now())
        except Exception as error:
            record["stages"][stage].update(status="failed", finished_at=now(),
                                          error=str(error), log=str(log) if log.exists() else None)
            record["status"] = "failed"
            atomic_json(record_path, record)
            print(f"FAILED {name}: {stage}: {error}", flush=True)
            return record
        atomic_json(record_path, record)
    record.update(status="done", finished_at=now())
    atomic_json(record_path, record)
    return record


def run_logged_stage(stage, args, raw, converted, annotation, stream):
    if stage == "convert":
        values = [args.python, REPO / "third_party/Multi-Frequency-LeRobot/format_convert/jingdong/convert_jingdong.py",
                  "--source", raw, "--output-dir", converted, "--ffmpeg", args.ffmpeg]
    else:
        values = [args.python, REPO / "third_party/argus/review/base/run.py",
                  "--data", converted / "videos/chunk-000/observation.images.head_left_rectified/episode_000000.mp4",
                  "--output", annotation, "--model", args.model,
                  "--sample-fps", args.sample_fps, "--cap", "0"]
    subprocess.run([str(value) for value in values], cwd=REPO, check=True,
                   stdout=stream, stderr=subprocess.STDOUT)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-id", required=True, type=batch_name)
    parser.add_argument("--source", default="ks3://baai-eai-pwm/baai-eai-pwm-humancoin-waibu/jingdong/")
    parser.add_argument("--destination", default="ks3://baai-eai-pwm/baai-eai-pwm-humancoin-waibu/jingdong_humancoin/")
    parser.add_argument("--raw-root", type=Path, default=Path.home() / "data/jingdong")
    parser.add_argument("--output-root", type=Path, default=Path.home() / "data/jingdong_humancoin")
    parser.add_argument("--report-root", type=Path, default=Path.home() / "data/jingdong_pipeline")
    parser.add_argument("--ks3-tool", type=Path, default=Path("/opt/ks3/ks3util-2.18"))
    parser.add_argument("--ks3-config", type=Path, default=Path("/opt/ks3/ks3util-config-team"))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--ffmpeg", default="/opt/ego/bin/ffmpeg")
    parser.add_argument("--model", default="qwen3.7-plus")
    parser.add_argument("--sample-fps", type=float, default=1)
    parser.add_argument("--jobs", type=int, default=64)
    parser.add_argument("--prepare-batches", action="store_true",
                        help="Generate and upload missing batch inventories before processing")
    args = parser.parse_args(argv)
    if args.jobs < 1 or not math.isfinite(args.sample_fps) or args.sample_fps <= 0:
        parser.error("jobs and sample-fps must be positive")
    for name in ("raw_root", "output_root", "report_root"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    args.source, args.destination = prefix(args.source), prefix(args.destination)
    report = args.report_root / args.batch_id
    with batch_lock(report / ".lock"):
        client = KS3(args.ks3_tool, args.ks3_config)
        if args.prepare_batches:
            batches = args.raw_root / "batches"
            if not (batches / "inventory.json").exists():
                command([args.python, "-m", "humancoin_datasets.download_data.jingdong.inventory",
                         "--ks3-tool", args.ks3_tool, "--ks3-config", args.ks3_config,
                         "--output-dir", batches])
            command([args.python, "-m", "humancoin_datasets.upload_data.base.upload_batches",
                     "--csv-dir", batches, "--destination", args.source,
                     "--ks3-tool", args.ks3_tool, "--ks3-config", args.ks3_config])
        import tempfile
        with tempfile.TemporaryDirectory(dir=report) as temporary:
            csv = Path(temporary) / (args.batch_id + ".csv")
            uri = args.source + "batch_csv/" + csv.name
            client.copy(uri, str(csv))
            matches = [item for item in client.objects(uri) if item["uri"] == uri]
            if len(matches) != 1 or matches[0]["size"] != csv.stat().st_size:
                raise ValueError("Batch CSV size verification failed")
            paths = read_batch(csv)
            csv.replace(report / csv.name)
        results = []
        for uri in paths:
            try:
                result = process_instance(uri, args, client, report)
            except Exception as error:
                result = dict(instance=uri.rstrip("/").rsplit("/", 1)[-1],
                              status="failed", error=str(error))
                atomic_json(report / (result["instance"] + "_error.json"), result)
            results.append(result)
            atomic_json(report / "summary.json", dict(
                batch=args.batch_id, total=len(paths), processed=len(results),
                done=sum(r["status"] == "done" for r in results), results=results))
        return int(any(result["status"] != "done" for result in results))


if __name__ == "__main__":
    raise SystemExit(main())
