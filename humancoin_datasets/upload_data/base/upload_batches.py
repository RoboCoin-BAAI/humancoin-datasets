"""Upload local CSV files into batch_csv/ under a KS3 root prefix."""

import argparse
from pathlib import Path

from humancoin_datasets.upload_data.ks3 import KS3, prefix


def upload_batches(directory, destination, client, dry_run=False):
    destination = prefix(destination) + "batch_csv/"
    directory = Path(directory).expanduser().resolve()
    if not directory.is_dir():
        raise ValueError(f"Local CSV directory does not exist: {directory}")
    files = sorted(path for path in directory.glob("*.csv") if path.is_file())
    if not files:
        raise ValueError(f"No CSVs found in {directory}")
    for path in files:
        uri = destination + path.name
        print(f"{path} -> {uri}", flush=True)
        if not dry_run:
            client.copy(str(path.resolve()), uri)
            matches = [item for item in client.objects(uri) if item["uri"] == uri]
            if len(matches) != 1 or matches[0]["size"] != path.stat().st_size:
                raise RuntimeError(f"Uploaded CSV size verification failed: {uri}")
    print(f"{'Planned' if dry_run else 'Verified'} {len(files)} CSVs")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv-dir", required=True, type=Path)
    parser.add_argument("--destination", required=True, help="KS3 root prefix")
    parser.add_argument("--ks3-tool", type=Path, default=Path("/opt/ks3/ks3util-2.18"))
    parser.add_argument("--ks3-config", type=Path, default=Path("/opt/ks3/ks3util-config-team"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    upload_batches(args.csv_dir, args.destination,
                   KS3(args.ks3_tool, args.ks3_config), args.dry_run)


if __name__ == "__main__":
    main()
