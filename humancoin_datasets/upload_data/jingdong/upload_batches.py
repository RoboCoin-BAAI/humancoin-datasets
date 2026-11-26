"""Upload batch CSVs directly into the Jingdong KS3 data prefix."""

import argparse
import csv
from pathlib import Path
import re

from humancoin_datasets.download_data.jingdong.inventory import SOURCE
from humancoin_datasets.upload_data.ks3 import KS3, prefix


def upload_batches(directory, destination, client, dry_run=False):
    destination = prefix(destination)
    files = sorted(Path(directory).glob("batch_*.csv"))
    if not files:
        raise ValueError(f"No batch CSVs found in {directory}")
    seen = set()
    for index, path in enumerate(files, 1):
        if path.name != f"batch_{index:07d}.csv":
            raise ValueError(f"Unexpected batch sequence: {path.name}")
        with path.open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.reader(stream))
        if not rows or rows[0] != ["目标路径"]:
            raise ValueError(f"Invalid CSV header: {path}")
        if not 1 <= len(rows) - 1 <= 100 or (index < len(files) and len(rows) != 101):
            raise ValueError(f"Invalid batch size: {path}")
        for row in rows[1:]:
            if len(row) != 1 or not re.fullmatch(
                    re.escape(destination) + r"instance-[A-Za-z0-9]+/", row[0]):
                raise ValueError(f"Unexpected instance path in {path}")
            if row[0] in seen:
                raise ValueError(f"Duplicate instance in {path}")
            seen.add(row[0])
    for path in files:
        uri = destination + path.name
        print(f"{path} -> {uri}", flush=True)
        if not dry_run:
            client.copy(str(path.resolve()), uri)
            matches = [item for item in client.objects(uri) if item["uri"] == uri]
            if len(matches) != 1 or matches[0]["size"] != path.stat().st_size:
                raise RuntimeError(f"Uploaded CSV size verification failed: {uri}")
    print(f"{'Planned' if dry_run else 'Verified'} {len(files)} CSVs, {len(seen)} instances")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches-dir", type=Path,
                        default=Path.home() / "data/jingdong/batches")
    parser.add_argument("--destination", default=SOURCE)
    parser.add_argument("--ks3-tool", type=Path, default=Path("/opt/ks3/ks3util-2.18"))
    parser.add_argument("--ks3-config", type=Path, default=Path("/opt/ks3/ks3util-config"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    upload_batches(args.batches_dir, args.destination,
                   KS3(args.ks3_tool, args.ks3_config), args.dry_run)


if __name__ == "__main__":
    main()
