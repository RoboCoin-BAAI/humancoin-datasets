"""Fetch a remote batch CSV and download its dataset directories."""

import argparse
from pathlib import Path
import tempfile

from humancoin_datasets.download_data.base.run import read_batch
from humancoin_datasets.upload_data.ks3 import KS3, prefix


def batch_name(value):
    value = value.removeprefix("batch_").removesuffix(".csv")
    if not value.isascii() or not value.isdigit() or int(value) < 1:
        raise argparse.ArgumentTypeError("Batch ID must be positive, e.g. 1 or batch_0000001")
    return f"batch_{int(value):07d}"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="KS3 dataset root containing batch_csv/")
    parser.add_argument("--batch-id", required=True, type=batch_name)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--ks3-tool", type=Path, default=Path("/opt/ks3/ks3util-2.18"))
    parser.add_argument("--ks3-config", type=Path, default=Path("/opt/ks3/ks3util-config-team"))
    parser.add_argument("--jobs", type=int, default=64)
    args = parser.parse_args(argv)
    if not 1 <= args.jobs <= 10000:
        parser.error("--jobs must be between 1 and 10000")
    # The receipt-based downloader uses POSIX locks on the Linux execution host.
    from humancoin_datasets.download_data.downloader import KS3Downloader

    root = prefix(args.source)
    output = args.output_dir.expanduser().resolve()
    downloader = KS3Downloader(output / args.batch_id, args.ks3_tool, args.ks3_config, args.jobs)
    csv_directory = output / "batch_csv"
    csv_directory.mkdir(parents=True, exist_ok=True)
    client = KS3(args.ks3_tool, args.ks3_config)
    uri = root + "batch_csv/" + args.batch_id + ".csv"
    with tempfile.TemporaryDirectory(dir=csv_directory) as temporary:
        downloaded = Path(temporary) / (args.batch_id + ".csv")
        print(f"Fetching {uri}", flush=True)
        client.copy(uri, str(downloaded))
        matches = [item for item in client.objects(uri) if item["uri"] == uri]
        if len(matches) != 1 or matches[0]["size"] != downloaded.stat().st_size:
            raise RuntimeError("Downloaded CSV size verification failed")
        paths = read_batch(downloaded)
        downloaded.replace(csv_directory / downloaded.name)
    failed = 0
    for index, source in enumerate(paths, 1):
        print(f"[{index}/{len(paths)}] {source}", flush=True)
        try:
            result = downloader.download(source)
            print(f"{'Reused' if result.reused else 'Downloaded'} {result.path}", flush=True)
        except (OSError, RuntimeError) as error:
            failed += 1
            print(f"Failed: {error}", flush=True)
    print(f"Finished: {len(paths) - failed} succeeded, {failed} failed", flush=True)
    return int(failed > 0)


if __name__ == "__main__":
    raise SystemExit(main())
